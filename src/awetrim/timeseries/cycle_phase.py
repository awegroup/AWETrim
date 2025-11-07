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
from awetrim.timeseries.reelout_phase import Reelout


class CycleSimple:
    """Compose a reel-in and a reel-out phase into a single cycle optimization.

    Usage:
        cycle = Cycle(reelin_instance, reelout_instance)
        opti, vars, obj, params = cycle.get_opti_components()
        solution = cycle.run_cycle_opti()
    """

    def __init__(self, reelin: ReelinSimple, reelout: Reelout) -> None:
        self.reelin = reelin
        self.reelout = reelout

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
        optimization_dict = {}
        if optimization_params:
            for var in optimization_params:
                optimization_dict[var] = opti.variable()
            if "coeffs" in var:
                num_coeffs = len(
                    self.reelout.pattern_config["path_parameters"].get(var, [])
                )
                optimization_dict[var] = opti.variable(num_coeffs)
        opti, vars_ro, obj_ro, params_ro = self.reelout.get_opti_components(
            optimization_dict=optimization_dict, opti=opti
        )
        start_state_ri = self.reelin.start_state_ri.copy()
        start_state_ri["distance_radial"] = vars_ro["distance_radial"][-1]
        # TODO: ADD ELEVATION SYNC
        # pattern_config_ri = self.reelin.pattern_config_ri.copy()
        # pattern_config_ri["path_parameters"]["elevation_start_ri"] = obj_ro[
        #     "angle_elevation_end"
        # ]
        # pattern_config_ri["path_parameters"]["elevation_start_ro"] = obj_ro[
        #     "angle_elevation_start"
        # ]
        opti, vars_ri, obj_ri, params_ri = self.reelin.get_opti_components(
            optimization_dict=optimization_dict,
            opti=opti,
            start_state_opti=start_state_ri,
        )
        # print(params_ri)
        # print(params_ro)
        # raise RuntimeError("rasi")
        opti.subject_to(
            vars_ri["distance_radial"][1][-1]
            == self.reelout.pattern_config["path_parameters"]["distance_radial_start"]
        )
        # Merge dictionaries using reelin's helper (robust merging rules)

        merged_vars = ReelinSimple._merge_phase_dicts(vars_ri, vars_ro)
        merged_obj = ReelinSimple._merge_phase_dicts(obj_ri, obj_ro)

        # Merge parameter dicts (names -> opti variables). If keys collide, keep both
        merged_params = {}
        merged_params.update(params_ri or {})
        for k, v in (params_ro or {}).items():
            if k in merged_params:
                # If there is a collision, rename the second to avoid overwrite
                # (append _ro). This rarely happens if parameter naming is consistent.
                merged_params[f"{k}_ro"] = v
            else:
                merged_params[k] = v

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

    def run_cycle_opti(
        self, optimization_params: Optional[List[str]] = None
    ) -> Optional[Any]:
        """Run the combined cycle optimization and return the solution (or None).

        The combined objective is formed by summing energies and times across
        phases and using the same structure: -(energy_total / total_time_total / power_scale).
        """

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
                    "acceptable_tol": 1e-6,
                    "constr_viol_tol": 1e-6,
                    "dual_inf_tol": 1e-6,
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
                print("Processing variable:", var_name)
                try:
                    val = solution.value(opt_var)
                except Exception:
                    # If value extraction fails, skip
                    continue

                # Remove _ro suffix if present for checking configs
                base_name = var_name[:-3] if var_name.endswith("_ro") else var_name

                # Try to update the parameter in the correct phase configuration
                updated = False

                # Check reelout first if it has _ro suffix
                if var_name.endswith("_ro"):
                    if base_name in self.reelout.pattern_config.get(
                        "path_parameters", {}
                    ):
                        self.reelout.pattern_config["path_parameters"][base_name] = val
                        print(f"  {var_name} -> reelout: {val}")
                        updated = True
                    if base_name in self.reelout.pattern_config.get(
                        "radial_parameters", {}
                    ):
                        self.reelout.pattern_config["radial_parameters"][
                            base_name
                        ] = val
                        print(f"  {var_name} -> reelout: {val}")
                        updated = True

                # Check reelin if not updated or no _ro suffix
                if not updated and base_name in self.reelin.pattern_config.get(
                    "path_parameters", {}
                ):
                    self.reelin.pattern_config["path_parameters"][base_name] = val
                    print(f"  {var_name} -> reelin: {val}")
                    updated = True
                if not updated and base_name in self.reelin.pattern_config.get(
                    "radial_parameters", {}
                ):
                    self.reelin.pattern_config["radial_parameters"][base_name] = val
                    print(f"  {var_name} -> reelin: {val}")
                    updated = True

                # If still not updated, check reelout without _ro requirement
                if not updated and base_name in self.reelout.pattern_config.get(
                    "path_parameters", {}
                ):
                    self.reelout.pattern_config["path_parameters"][base_name] = val
                    print(f"  {var_name} -> reelout: {val}")
                    updated = True
                if not updated and base_name in self.reelout.pattern_config.get(
                    "radial_parameters", {}
                ):
                    self.reelout.pattern_config["radial_parameters"][base_name] = val
                    print(f"  {var_name} -> reelout: {val}")
                    updated = True
                if not updated:
                    print(
                        f"  Warning: {var_name} = {val} (not stored in any configuration)"
                    )

            return solution
        except Exception as exc:
            print("Cycle optimization failed:", exc)
            return None

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
        self.reelin.pattern_config.setdefault("path_parameters", {})
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
        reelin_phase = None
        reelin_phase_ro = None
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
                target="zero",
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

        energy_ro = phase_ro.energy
        total_time_ro = phase_ro.total_time
        energy_ri = phase_ri.energy
        total_time_ri = phase_ri.total_time
        energy_riro = phase_riro.energy
        total_time_riro = phase_riro.total_time

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
            "reelin_phase": reelin_phase,
            "reelin_phase_ro": reelin_phase_ro,
            "optimization_result": optimization_result,
        }
