# Copyright (c) 2023-2026 Oriol Cayon, Delft University of Technology
#
# Licensed under the EUPL, Version 1.2 or - as soon they will be approved by
# the European Commission - subsequent versions of the EUPL (the "Licence");
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at:
#
#     https://joinup.ec.europa.eu/software/page/eupl
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the Licence is distributed on an "AS IS" basis,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the Licence for the specific language governing permissions and
# limitations under the Licence.
#
# SPDX-License-Identifier: EUPL-1.2

"""Combine reel-in and reel-out phases into a single cycle optimization.

This module provides the Cycle class which composes two phase classes
(ReelinSimple and ReeloutSimple) into a single optimization problem.

The Cycle class:
- requests optimization components from both phases using the same CasADi
  Opti environment,
- merges their optimization variables and objective-component dictionaries,
- computes a combined scalar objective by summing per-phase energies and
  times and forming the same objective expression used elsewhere
  (-(energy_total / total_time_total / power_scale)),
- provides a convenience method to run the combined optimization.

Design notes:
- This code relies on the ReelinSimple._merge_phase_dicts static helper to
  merge dictionaries in a way that preserves lists or sums scalar-like values.
- The implementation tries to be conservative about combining 'power_scale':
  if both phases provide it, the first non-None is used.

"""

from typing import Any, Dict, Tuple, Optional, List

import casadi as ca
import matplotlib.pyplot as plt

from awetrim.timeseries.reelin_phase import ReelinSimple
from awetrim.timeseries.phase import Phase
import numpy as np


class CycleSimple:
    """Compose a reel-in and a reel-out phase into a single cycle optimization.

    Usage:
        cycle = Cycle(reelin_instance, reelout_instance)
        opti, vars, obj, params = cycle.get_opti_components()
        solution = cycle.run_cycle_opti()
    """

    def __init__(self, reelin: ReelinSimple, reelout: Phase) -> None:
        self.reelin = reelin
        self.reelout = reelout

    def _param_dimension(self, var: str) -> int:
        """Number of scalar components of an optimization variable, inferred
        from the phase configs. Vector parameters such as the spline coefficients
        ``C_phi`` / ``C_beta`` return their length; scalars return 1.
        """
        for phase in (self.reelout, self.reelin):
            for section in ("path_parameters", "radial_parameters", "sim_parameters"):
                val = phase.pattern_config.get(section, {}).get(var)
                if isinstance(val, ca.DM):
                    return int(val.numel())
                if isinstance(val, (list, tuple, np.ndarray)):
                    return len(val)
        return 1

    def get_opti_components(
        self,
        optimization_params: Optional[List[str]] = None,
    ) -> Tuple[ca.Opti, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Create a combined CasADi Opti instance and collect components.

        Returns:
            opti: CasADi Opti instance shared by both phases
            combined_vars: merged variables dictionaries from both phases
            combined_objective: merged objective-component dict (keys like 'energy', 'total_time')
            combined_params: merged parameter dict (mapping names to opti variables)
        """
        opti = ca.Opti()
        # Build each opti variable once, then route it to the phase(s) that own it.
        # The per-phase depower aliases ``input_depower_ro`` / ``input_depower_ri``
        # map to that phase's literal ``input_depower``, giving the two phases
        # independent depower decisions — a single shared ``input_depower`` would
        # force reel-out and reel-in depower to be equal.
        reelout_dict, reelin_dict, merged_params = {}, {}, {}
        for var in optimization_params or []:
            n = self._param_dimension(var)
            mxvar = opti.variable(n) if n > 1 else opti.variable()
            merged_params[var] = mxvar
            if var == "input_depower_ro":
                reelout_dict["input_depower"] = mxvar
            elif var == "input_depower_ri":
                reelin_dict["input_depower"] = mxvar
            else:
                # Offer non-depower params to both phases; each substitutes only
                # the keys present in its own configuration.
                reelout_dict[var] = mxvar
                reelin_dict[var] = mxvar
        opti, vars_ro, obj_ro, _ = self.reelout.get_opti_components(
            optimization_dict=reelout_dict, opti=opti
        )
        start_state_ri = self.reelin.start_state_ri.copy()
        start_state_ri["distance_radial"] = vars_ro["distance_radial"][-1]
        # Elevation continuity across the handoffs: reel-in starts where reel-out
        # ends, and the transition ends where reel-out starts. Tying these to the
        # reel-out's symbolic endpoint elevations keeps the cycle closed in
        # elevation as the reel-out spline (C_beta) is optimized.
        opti, vars_ri, obj_ri, _ = self.reelin.get_opti_components(
            optimization_dict=reelin_dict,
            opti=opti,
            start_state_opti=start_state_ri,
            elevation_start_ri_expr=obj_ro["angle_elevation_end"],
            elevation_start_ro_expr=obj_ro["angle_elevation_start"],
        )
        opti.subject_to(
            vars_ri["distance_radial"][1][-1]
            == self.reelout.pattern_config["path_parameters"]["r0"]
        )
        # Merge dictionaries using reelin's helper (robust merging rules)
        merged_vars = ReelinSimple._merge_phase_dicts(vars_ri, vars_ro)
        merged_obj = ReelinSimple._merge_phase_dicts(obj_ri, obj_ro)

        # merged_params is keyed by the original (alias) parameter names so the
        # write-back can route per-phase depower correctly.
        return opti, merged_vars, merged_obj, merged_params

    @staticmethod
    def _sum_phase_values(val):
        """Sum phase values that may be scalars or sequences.

        If val is a list/tuple, sums entries with + (works for CasADi objects).
        Otherwise returns val unchanged.
        """
        if isinstance(val, (list, tuple)):
            if len(val) == 0:
                return 0
            total = val[0]
            for v in val[1:]:
                total = total + v
            return total
        return val

    def _store_optimized_value(self, var_name: str, val) -> bool:
        """Write an optimized value back to the owning phase configuration.

        The per-phase depower aliases route to each phase's ``sim_parameters``
        ``input_depower``; every other parameter is matched by exact name across
        both phases' path / radial / sim parameter sections.
        """
        if var_name == "input_depower_ro":
            self.reelout.pattern_config.setdefault("sim_parameters", {})[
                "input_depower"
            ] = val
            return True
        if var_name == "input_depower_ri":
            self.reelin.pattern_config.setdefault("sim_parameters", {})[
                "input_depower"
            ] = val
            return True
        for phase in (self.reelout, self.reelin):
            for section in ("path_parameters", "radial_parameters", "sim_parameters"):
                if var_name in phase.pattern_config.get(section, {}):
                    phase.pattern_config[section][var_name] = val
                    return True
        return False

    def run_cycle_opti(
        self,
        optimization_params: Optional[List[str]] = None,
        reelout_warmstart_params: Optional[List[str]] = None,
    ) -> Optional[Any]:
        """Run the combined cycle optimization and return the solution (or None).

        The combined objective is formed by summing energies and times across
        phases and using the same structure: -(energy_total / total_time_total / power_scale).

        Staged warm-start (each stage seeds the next):
          1. Optimize the reel-out shape (and depower) with the winch fixed.
          2. Optimize the reel-in alone, with boundaries taken from the reel-out.
          3. Solve the combined reel-out + reel-in NLP.

        ``reelout_warmstart_params`` overrides which reel-out parameters are tuned
        in stage 1; by default it takes the reel-out path parameters present in
        ``optimization_params`` (e.g. the spline ``C_phi`` / ``C_beta``) plus
        ``input_depower`` — winch (radial) parameters are deliberately excluded so
        the warm start is a clean shape optimization.
        """

        # Stage 1: reel-out shape + depower with the winch fixed.
        if reelout_warmstart_params is None:
            reelout_path = self.reelout.pattern_config.get("path_parameters", {})
            reelout_warmstart_params = [
                p for p in (optimization_params or []) if p in reelout_path
            ]
            if "input_depower" not in reelout_warmstart_params:
                reelout_warmstart_params.append("input_depower")
        if reelout_warmstart_params:
            print(
                "Staged warm-start: optimizing reel-out "
                f"{reelout_warmstart_params} (winch fixed)..."
            )
            self.reelout.run_simulation_opti(
                optimization_params=reelout_warmstart_params, target="power"
            )
            # Drop the now-dead opti symbol left on the shared model so the later
            # phases re-symbolize depower cleanly.
            self.reelout.system_model.input_depower = ca.MX.sym("input_depower")

        # Stage 2: run the reel-out, set the reel-in boundary conditions from it,
        # and optimize the reel-in alone — a feasible, reel-out-consistent reel-in.
        print("Staged warm-start: simulating reel-out and optimizing reel-in...")
        self.run_cycle_simulation(optimize_reelin=True, plotting=False)

        opti, merged_vars, merged_obj, merged_params = self.get_opti_components(
            optimization_params=optimization_params
        )

        # Safely extract and sum phase-wise objective components
        energy_total = self._sum_phase_values(merged_obj.get("energy", 0))
        total_time_total = self._sum_phase_values(merged_obj.get("total_time", 1))

        # Choose a power_scale if provided; prefer the merged_obj value if scalar-like
        power_scale = merged_obj.get("power_scale", 1)
        if isinstance(power_scale, (list, tuple)):
            # Sum or pick first non-zero
            power_scale = self._sum_phase_values(power_scale)

        total_objective = -(energy_total / total_time_total / power_scale)

        # Minimize and solve
        opti.minimize(total_objective)
        opti.solver(
            "ipopt",
            {
                "ipopt": {
                    "bound_relax_factor": 1e-8,
                    "tol": 1e-6,
                    # "acceptable_iter": 3,
                    "acceptable_tol": 1e-5,
                    "constr_viol_tol": 1e-5,
                    "dual_inf_tol": 1e-5,
                    "hessian_approximation": "limited-memory",
                    "mu_strategy": "adaptive",
                }
            },
        )

        try:
            solution = opti.solve()
            # After successful optimization, print optimized parameter values
            print("\nOptimized cycle parameters:")
            # Update parameter values in the appropriate phase configuration
            for var_name, opt_var in merged_params.items():
                try:
                    val = solution.value(opt_var)
                except Exception:
                    # If value extraction fails, skip
                    continue
                if self._store_optimized_value(var_name, val):
                    print(f"  {var_name}: {val}")
                else:
                    print(
                        f"  Warning: {var_name} = {val} (not stored in any configuration)"
                    )

            return solution
        except Exception as exc:
            print("Cycle optimization failed:", exc)
            return None

    def _sync_reelin_to_reelout(self):
        """Simulate the reel-out and copy its endpoints into the reel-in config.

        The reel-in is set to start where the reel-out ends (radius, elevation,
        time) and to close back to the reel-out start radius. Returns the reel-out
        ``PhaseParameterized`` (with ``.energy`` / ``.total_time``).
        """
        phase_ro, _ = self.reelout.run_simulation()
        pp = self.reelin.pattern_config["path_parameters"]
        pp["distance_radial_start"] = phase_ro.return_variable("distance_radial")[-1]
        pp["distance_radial_end"] = phase_ro.return_variable("distance_radial")[0]
        pp["elevation_start_ri"] = phase_ro.return_variable("angle_elevation")[-1]
        pp["elevation_start_ro"] = phase_ro.return_variable("angle_elevation")[0]
        self.reelin.pattern_config.setdefault("sim_parameters", {})["start_time"] = (
            phase_ro.return_variable("t")[-1]
        )
        return phase_ro

    def run_cycle_alternating(
        self,
        reelout_params: Optional[List[str]] = None,
        reelin_params: Optional[List[str]] = None,
        max_iter: int = 4,
        tol: float = 1e-3,
        trust_region_weight: float = 0.01,
    ) -> Optional[Dict[str, Any]]:
        """Alternating (block-coordinate) cycle optimization.

        Each block maximizes the *true cycle power* by carrying the other phase's
        energy and time as fixed offsets, alternating reel-out then reel-in until
        the cycle power converges. Smoother and more robust than the monolithic
        ``run_cycle_opti`` for weakly-coupled cycles, at the cost of converging to
        a coordinate (rather than joint) optimum.

        Args:
            reelout_params: reel-out parameters to tune (default: shape + winch +
                depower). Uses the reel-out's own literal parameter names.
            reelin_params: reel-in parameters to tune (default: transition
                elevation + winch + depower).
            max_iter: maximum number of outer alternating iterations.
            tol: relative cycle-power change at which to stop.
            trust_region_weight: damping toward the previous iterate, applied
                from the second iteration onward (the first iteration solves
                freely to reach the productive region). Larger values keep each
                block-coordinate step closer to the last design.
        """
        if reelout_params is None:
            reelout_params = ["C_phi", "C_beta", "slope_winch_ro", "input_depower"]
        if reelin_params is None:
            reelin_params = [
                "elevation_start_riro",
                "offset_winch_ri",
                "slope_winch_ri",
                # "input_depower",
            ]

        prev_power = None
        last_feasible_result = None
        for it in range(max_iter):
            # Evaluate the current cycle at consistent boundaries.
            try:
                phase_ro = self._sync_reelin_to_reelout()
                phase_ri, phase_riro = self.reelin.run_simulation()
            except Exception as exc:
                print(
                    "Alternating optimization aborted: re-simulation failed "
                    f"({exc}); keeping the last feasible design."
                )
                break
            E_ro, T_ro = phase_ro.energy, phase_ro.total_time
            E_ri = phase_ri.energy + phase_riro.energy
            T_ri = phase_ri.total_time + phase_riro.total_time
            cycle_power = (E_ro + E_ri) / (T_ro + T_ri + 1e-12)
            print(
                f"[alternating {it}] cycle power = {cycle_power / 1000:.3f} kW "
                f"(E_ro={E_ro / 1000:.1f} kJ, E_ri={E_ri / 1000:.1f} kJ)"
            )
            if not np.isfinite(cycle_power):
                print(
                    "Alternating optimization aborted: re-simulation produced a "
                    "non-finite cycle power; keeping the last feasible design."
                )
                break
            last_feasible_result = {
                "reelout_phase": phase_ro,
                "reelin_phase": phase_ri,
                "transition_phase": phase_riro,
                "cycle_power": cycle_power,
            }
            if prev_power is not None and abs(cycle_power - prev_power) <= tol * abs(
                prev_power + 1e-12
            ):
                print(f"Alternating optimization converged after {it} iterations.")
                break
            prev_power = cycle_power

            # First iteration solves freely to reach the productive region;
            # later iterations warm-start (tight barrier) and damp toward the
            # previous iterate to stay in the hot zone.
            warm = it > 0
            tr_weight = trust_region_weight if it > 0 else 0.0

            # Block 1: reel-out for cycle power (reel-in held fixed as offsets).
            # Reel-out defines the cycle boundaries, so it is optimized first.
            reelout_result = self.reelout.run_simulation_opti(
                optimization_params=reelout_params,
                target="power",
                energy_offset=E_ri,
                time_offset=T_ri,
                warm_start=warm,
                trust_region_weight=tr_weight,
                max_iter=400,
            )
            if reelout_result is None:
                print(
                    "Alternating optimization aborted: reel-out optimization "
                    "failed; keeping the last feasible design."
                )
                break

            # Block 2: re-sync the reel-in boundaries to the new reel-out, then
            # reel-in for cycle power (reel-out held fixed as offsets).
            try:
                phase_ro = self._sync_reelin_to_reelout()
            except Exception as exc:
                print(
                    "Alternating optimization aborted: reel-out re-simulation "
                    f"failed after optimization ({exc}); keeping the last "
                    "feasible design."
                )
                break
            E_ro, T_ro = phase_ro.energy, phase_ro.total_time
            reelin_result = self.reelin.run_simulation_opti(
                optimization_params=reelin_params,
                target="power",
                energy_offset=E_ro,
                time_offset=T_ro,
                warm_start=warm,
                trust_region_weight=tr_weight,
            )
            if reelin_result is None:
                print(
                    "Alternating optimization aborted: reel-in optimization "
                    "failed; keeping the last feasible design."
                )
                break

        # Final consistent simulation + summary.
        final_result = self.run_cycle_simulation(optimize_reelin=False, plotting=False)
        return final_result if final_result is not None else last_feasible_result

    def run_cycle_simulation(
        self, optimize_reelin: bool = True, plotting: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Run reel-out simulation, then run/optimize the reel-in phase.

        Procedure:
        1. Run the reel-out simulation phase to get the final radial distance.
        2. Set the reel-in start radial distance to the reel-out final value.
        3. Optionally optimize the reel-in phase with optimization parameter
           ['elevation_start_riro'] only, and run its simulation.
        4. If plotting=True, plot reel-out and reel-in (and transition) together.

        Returns a small dict with keys 'reelout_phase', 'reelin_phase', 'reelin_phase_ro'
        containing the PhaseParameterized objects or None on failure.
        """
        # 1) Run reel-out simulation (numeric)
        axes = None  # only populated when plotting; reel-in reuses these axes
        try:
            phase_ro, _ = self.reelout.run_simulation()
            if plotting:
                # plot reel-out first and obtain axes
                fig, axes, _ = phase_ro.plot_overview_3d(
                    x_param="t", variables=self.reelout.variables_to_plot
                )
        except Exception as exc:
            print("Reel-out simulation failed:", exc)
            return None

        # extract final radial distance from reel-out
        try:
            distance_radial_end_ro = phase_ro.return_variable("distance_radial")[-1]
            distance_radial_start_ro = phase_ro.return_variable("distance_radial")[0]
            elevation_end_ro = phase_ro.return_variable("angle_elevation")[-1]
            elevation_start_ro = phase_ro.return_variable("angle_elevation")[0]
            t_end_ro = phase_ro.return_variable("t")[-1]
        except Exception as exc:
            print("Failed to read final radial distance from reel-out phase:", exc)
            return None

        # 2) Set the reel-in start distance to this value
        # self.reelin.pattern_config.setdefault("path_parameters", {})
        self.reelin.pattern_config["path_parameters"][
            "distance_radial_start"
        ] = distance_radial_end_ro
        self.reelin.pattern_config["path_parameters"][
            "distance_radial_end"
        ] = distance_radial_start_ro
        self.reelin.pattern_config["path_parameters"][
            "elevation_start_ri"
        ] = elevation_end_ro
        self.reelin.pattern_config["path_parameters"][
            "elevation_start_ro"
        ] = elevation_start_ro
        self.reelin.pattern_config["sim_parameters"]["start_time"] = t_end_ro
        # 3) Optimize (only) the reel-in if requested
        optimization_result = None
        if optimize_reelin:
            # use only elevation_start_riro as optimization parameter
            # try:
            optimization_result = self.reelin.run_simulation_opti(
                optimization_params=[
                    "elevation_start_riro",
                    "offset_winch_ri",
                    # "slope_winch_ri",
                ],
                target="energy",
            )
            # except Exception as exc:
            #     print("Reel-in optimization failed:", exc)
            #     optimization_result = None

        # After optimization (or even if not optimizing) run reel-in simulation
        try:
            # Initialize phase objects and run numeric simulation similarly to ReelinSimple.run_simulation
            phase_ri, phase_riro = self.reelin.run_simulation(
                run_plots=plotting, axes=axes
            )
        except Exception as exc:
            print("Reel-in simulation failed:", exc)
            return None

        energy_ro = phase_ro.energy
        total_time_ro = phase_ro.total_time
        energy_ri = phase_ri.energy
        total_time_ri = phase_ri.total_time
        energy_riro = phase_riro.energy
        total_time_riro = phase_riro.total_time

        # lift_ro = phase_ro.return_variable("lift_coefficient")
        # drag_ro = phase_ro.return_variable("drag_coefficient")
        # print(f"Reel-out average lift coefficient: {np.mean(lift_ro):.3f}")
        # print(f"Reel-out average drag coefficient: {np.mean(drag_ro):.3f}")
        # lift_ri = phase_ri.return_variable("lift_coefficient")
        # drag_ri = phase_ri.return_variable("drag_coefficient")
        # print(f"Reel-in average lift coefficient: {np.mean(lift_ri):.3f}")
        # print(f"Reel-in average drag coefficient: {np.mean(drag_ri):.3f}")
        # lift_riro = phase_riro.return_variable("lift_coefficient")
        # drag_riro = phase_riro.return_variable("drag_coefficient")
        # print(f"Transition average lift coefficient: {np.mean(lift_riro):.3f}")
        # print(f"Transition average drag coefficient: {np.mean(drag_riro):.3f}")

        # Print out the results
        cycle_power = (energy_ro + energy_ri + energy_riro) / (
            total_time_ro + total_time_ri + total_time_riro + 1e-12
        )
        print("Cycle simulation results:")
        print(
            f"  Reel-out energy: {energy_ro/1000:.2f} kJ, time: {total_time_ro:.2f} s"
        )
        print(f"  Reel-in energy: {energy_ri/1000:.2f} kJ, time: {total_time_ri:.2f} s")
        print(
            f"  Transition energy: {energy_riro/1000:.2f} kJ, time: {total_time_riro:.2f} s"
        )
        print(f"  Cycle power: {cycle_power/1000:.2f} kW")

        return {
            "reelout_phase": phase_ro,
            "reelin_phase": phase_ri,
            "transition_phase": phase_riro,
            "optimization_result": optimization_result,
        }
