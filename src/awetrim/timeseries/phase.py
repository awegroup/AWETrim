# Copyright (c) 2023-2026 Oriol Cayon, Delft University of Technology
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Single reel-out pattern optimization and simulation.

This module provides the ``Phase`` class for optimizing and simulating a
single reel-out pattern (downloop, uploop, helix, ...) for airborne wind
energy systems.
"""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import copy
import casadi as ca
import numpy as np
import yaml

from awetrim.timeseries.phase_parametrized import PhaseParameterized
from awetrim.utils.defaults import DEFAULT_RADIAL_PARAMETERS, DEFAULT_OPTI_LIMITS

START_STATE = {
    "t": 0,
    "s": 0,
    "s_dot": 2,
    "input_steering": 0,
    "tension_tether_ground": 8.4e4,  # Initial guess for tension (N)
    "speed_radial": 0,  # Positive for reel-out
}


@dataclass
class SimulationResult:
    """Container for simulation results and optimization outputs."""

    solution: Any  # CasADi solution object
    optimized_config: Dict[str, Any]
    final_distance: float
    phase_variables: Dict[str, Any]
    energy_objective: float
    total_time: float
    # Per-node numeric trajectory read straight from the NLP solution
    # (``solution.value(opti_vars[...])``). This is the optimizer's own output,
    # independent of the seed-sensitive re-simulation root-find. Empty when the
    # solve failed or the values could not be evaluated. Keys are the per-node
    # state/control names; each value is a length-``n_points`` numpy array.
    optimized_trajectory: Dict[str, np.ndarray] = field(default_factory=dict)

    def save_trajectory_csv(self, output_path: Union[str, Path]) -> None:
        """Save the optimizer's own per-node trajectory to a CSV file.

        Writes the arrays captured in :attr:`optimized_trajectory` — i.e. the
        values read directly from the NLP solution — so the saved record is the
        optimum itself rather than the re-simulated reconstruction. One column
        per variable, one row per discretization node.

        Parameters
        ----------
        output_path : str or Path
            Destination CSV path.
        """
        output_path = Path(output_path)
        if not self.optimized_trajectory:
            print(
                "No optimized trajectory to save (solve failed or values "
                "could not be evaluated)."
            )
            return
        output_path.parent.mkdir(parents=True, exist_ok=True)
        keys = list(self.optimized_trajectory.keys())
        data = np.column_stack([self.optimized_trajectory[k] for k in keys])
        np.savetxt(
            output_path,
            data,
            delimiter=",",
            header=",".join(keys),
            comments="",
        )
        print(f"Optimized trajectory saved to {output_path}")

    def save_config_to_yaml(self, output_path: Union[str, Path]) -> None:
        """Save optimized configuration to a YAML file.

        Parameters
        ----------
        output_path : str or Path
            Path where the YAML config file should be saved.

        Examples
        --------
        >>> result = reelout.run_simulation_opti()
        >>> result.save_config_to_yaml("data/LEI-V3-KITE/v3_optimized_config.yaml")
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert config to YAML-friendly format (numpy arrays -> lists)
        yaml_config = self._prepare_config_for_yaml(self.optimized_config)

        # Write to file
        with output_path.open("w", encoding="utf-8") as f:
            yaml.dump(yaml_config, f, default_flow_style=False, sort_keys=False)

        print(f"Optimized configuration saved to {output_path}")

    def _prepare_config_for_yaml(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Convert numpy arrays and other non-serializable types to YAML-friendly formats.

        Nests all reelout configuration under a "reelout" section.

        Parameters
        ----------
        config : dict
            Configuration dictionary potentially containing numpy arrays.

        Returns
        -------
        dict
            Configuration with reelout section nested and all numpy arrays converted to lists.
        """
        yaml_config = {"reelout": {}}

        for section_name, section_content in config.items():
            if isinstance(section_content, dict):
                yaml_config["reelout"][section_name] = {}
                for key, value in section_content.items():
                    if isinstance(value, np.ndarray):
                        yaml_config["reelout"][section_name][key] = value.tolist()
                    elif isinstance(value, (np.floating, np.integer)):
                        yaml_config["reelout"][section_name][key] = float(value)
                    else:
                        yaml_config["reelout"][section_name][key] = value
            else:
                yaml_config["reelout"][section_name] = section_content

        return yaml_config


class Phase:
    """Handles single reel-out pattern optimization and simulation.

    This class manages the optimization and simulation of a single reel-out
    pattern with configurable pattern type, path parameters, and radial
    parameters.

    Example:
        >>> config = {
        ...     "pattern_type": "figure8",  # or "circle", "helix", etc.
        ...     "path_parameters": {
        ...         "distance_radial_start": 100,  # Required
        ...         "distance_radial_end": 360,    # Required
        ...     },
        ...     "radial_parameters": {
        ...         "vr": 0.5  # Example parameter
        ...     }
        ... }
        >>> phase = Phase(
        ...     system_model=my_model,
        ...     pattern_config=config,
        ...     depower=1.0
        ... )
        >>> result = phase.run_simulation_opti()
    """

    def __init__(
        self,
        *,
        system_model: Any,  # Should be SystemModel but avoiding circular import
        pattern_config: Optional[Dict[str, Any]] = None,
        depower: float = None,
        start_state: Optional[Dict[str, Any]] = None,
        quasi_steady: bool = True,
    ) -> None:
        """Initialize ReeloutSimple instance.

        Args:
            system_model: The system model to use for simulation/optimization
            pattern_config: Configuration dictionary with pattern_type, path_parameters,
                          and radial_parameters
            depower: Depower setting for the kite (0 to 1)
        """
        self.quasi_steady = quasi_steady
        self.pattern_config = pattern_config or {}
        self._required_config = {
            "pattern_type": None,  # Must be provided
            "path_parameters": {},
            "radial_parameters": DEFAULT_RADIAL_PARAMETERS,  # Optional with defaults
        }
        # self._validate_config()

        self.depower = depower
        self.system_model = system_model

        # Derived configuration/state containers
        self.variables_to_plot = [
            "speed_tangential",
            "tension_tether_ground",
            "angle_elevation",
            "speed_radial",
        ]
        self._opti_params = {}
        self.start_state = start_state or START_STATE.copy()

    # def _validate_config(self) -> None:
    #     """Validate the pattern configuration and warn about missing required parameters."""
    #     missing_required = []
    #     using_defaults = []

    #     def check_section(required: Dict, actual: Dict, path: str = "") -> None:
    #         for key, default in required.items():
    #             current_path = f"{path}.{key}" if path else key
    #             if isinstance(default, dict):
    #                 # Recursively check nested dictionaries
    #                 if key not in actual:
    #                     if all(v is None for v in default.values()):
    #                         missing_required.append(current_path)
    #                     actual[key] = {}
    #                 check_section(default, actual[key], current_path)
    #             else:
    #                 if key not in actual:
    #                     if default is None:
    #                         missing_required.append(current_path)
    #                     else:
    #                         actual[key] = default
    #                         using_defaults.append(f"{current_path} = {default}")

    #     check_section(self._required_config, self.pattern_config)

    #     if missing_required:
    #         missing_str = "\n  - ".join(missing_required)
    #         raise ValueError(
    #             f"Missing required configuration parameters:\n  - {missing_str}"
    #         )

    #     if using_defaults:
    #         defaults_str = "\n  - ".join(using_defaults)
    #         warnings.warn(
    #             f"Using default values for configuration parameters:\n  - {defaults_str}",
    #             RuntimeWarning,
    #             stacklevel=2,
    #         )

    def initialize_phase(self) -> PhaseParameterized:
        """Initialize and prepare the optimization phase."""

        pattern_config_opti = copy.deepcopy(self.pattern_config)
        start_state = self.start_state
        # start_state["distance_radial"] = self.pattern_config["path_parameters"]["r0"]

        pattern_config_opti = copy.deepcopy(self.pattern_config)
        start_state_opti = copy.deepcopy(start_state)
        for var_name, mx in self._opti_params.items():
            for entry in ["path_parameters", "radial_parameters", "sim_parameters"]:
                if var_name in pattern_config_opti.get(entry, {}):
                    pattern_config_opti[entry][var_name] = mx
            if var_name == "input_depower":
                self.system_model.input_depower = mx

        self._phase = PhaseParameterized(
            self.system_model,
            quasi_steady=self.quasi_steady,
            pattern_config=self.pattern_config,
            pattern_config_opti=pattern_config_opti,
        )
        self._opti, self._opti_vars, self._objective = self._phase.opti_phase(
            start_state=start_state,
            opti=self._opti if hasattr(self, "_opti") else None,
            start_state_opti=start_state_opti,
            opti_params=getattr(self, "_opti_params", None),
        )
        return self._phase

    def get_opti_components(
        self,
        optimization_params: List[str] = None,
        optimization_dict: Dict[str, Any] = None,
        opti: Any = None,
    ) -> tuple:
        """Get optimization components (optimizer, variables, objective).

        Args:
            optimization_params: List of parameter names to optimize
            opti: Optional existing CasADi Opti instance

        Returns:
            Tuple of (optimizer, variables dict, objective dict, param dict)
        """
        if opti is None:
            opti = ca.Opti()
        self._opti = opti
        self._opti_params = {}

        profile_depower = bool(
            self.pattern_config.get("sim_parameters", {}).get(
                "optimize_depower_profile", False
            )
        )

        if optimization_params:
            for var in optimization_params:
                # In profile mode, depower is a per-node trajectory variable
                # built inside opti_phase (like input_steering), not a scalar
                # design parameter -- skip it here to avoid a double definition.
                if var == "input_depower" and profile_depower:
                    continue
                val = self.pattern_config["path_parameters"].get(var, None)

                if isinstance(val, ca.DM):
                    num_coeffs = int(val.numel())
                elif isinstance(val, (list, tuple, np.ndarray)):
                    num_coeffs = len(val)
                elif val is None:
                    num_coeffs = 1
                else:
                    num_coeffs = 1

                if num_coeffs > 1:
                    self._opti_params[var] = opti.variable(num_coeffs)
                else:
                    self._opti_params[var] = opti.variable()

        elif optimization_dict:
            self._opti_params = optimization_dict

        self.initialize_phase()

        return self._opti, self._opti_vars, self._objective, self._opti_params

    def run_simulation_opti(
        self,
        optimization_params: List[str] = None,
        target: str = "power",
        start_state: Optional[Dict[str, Any]] = None,
        warm_start_init_point: Optional[bool] = None,
        energy_offset: float = 0.0,
        time_offset: float = 0.0,
        warm_start: bool = False,
        trust_region_weight: float = 0.0,
        max_iter: Optional[int] = None,
    ) -> Optional[SimulationResult]:
        """Run optimization and return results.

        Args:
            optimization_params: List of parameters to optimize
            warm_start_init_point: Enable/disable IPOPT warm start initialization
            energy_offset, time_offset: constants added to this phase's energy and
                time in the ``power`` objective. Used by the alternating cycle
                optimizer to maximize the *full cycle* power (energy + time of the
                other phase) while only this phase's variables are free.
            warm_start: when True, configure IPOPT for a warm (tight-barrier)
                solve from a near-optimal guess (small ``mu_init`` / bound pushes)
                so it does not take the cold-start excursion away from and back to
                the optimum. Use on alternating iterations after the first.
            trust_region_weight: weight of a quadratic penalty anchoring the
                optimized parameters to their current (previous-iterate) config
                values, normalized by each parameter's bound width. Keeps a
                block-coordinate step from leaving the productive region.

        Returns:
            SimulationResult object or None if optimization failed
        """
        if warm_start_init_point is not None:
            self.pattern_config.setdefault("sim_parameters", {})[
                "warm_start_init_point"
            ] = warm_start_init_point

        opti, opti_vars, objective_dict, self._opti_params = self.get_opti_components(
            optimization_params=optimization_params
        )

        # Maximize average power (optionally the full cycle power via offsets)
        if target == "power":
            total_objective = -(
                (objective_dict["energy"] + energy_offset)
                / (objective_dict["total_time"] + time_offset)
                / objective_dict["power_scale"]
            )
        elif target == "energy":
            total_objective = -objective_dict["energy"]
        elif target == "zero":
            total_objective = 0.0

        # Control-smoothness regularization (off by default). On the flat power
        # ridge the bare optimum is non-unique, so any solver-path change moves
        # the result; this term selects the smoothest equal-power trajectory,
        # pinning a unique, reproducible optimum. Weight is tunable per problem.
        reg_weight = float(
            self.pattern_config.get("sim_parameters", {}).get("reg_weight", 0.0)
        )
        if reg_weight and objective_dict.get("reg") is not None:
            total_objective = total_objective + reg_weight * objective_dict["reg"]

        if trust_region_weight:
            total_objective = total_objective + self._trust_region_penalty(
                self._opti_params, trust_region_weight
            )

        solution = self.run_opti(
            opti, total_objective, warm_start=warm_start, max_iter=max_iter
        )
        if solution is None:
            return None

        # When the depower input is optimized as a per-node profile, persist the
        # optimized l_dp(s) into sim_parameters["input_depower_profile"] (length
        # n_points + 1, the s-grid). The follow-on run_simulation() and the saved
        # YAML then re-fly the optimized profile instead of the scalar depower.
        if "input_depower" in opti_vars:
            dep_opt = np.asarray(solution.value(opti_vars["input_depower"])).ravel()
            # opti_vars hold N node values; the simulation s-grid has N+1 points.
            # The last grid point only feeds the (unrecorded) final step, so pad
            # by repeating the last node to satisfy the length-(N+1) requirement.
            profile = np.append(dep_opt, dep_opt[-1])
            self.pattern_config.setdefault("sim_parameters", {})[
                "input_depower_profile"
            ] = profile

        # Carry the optimized node-0 state into start_state so a follow-on
        # run_simulation() re-simulates from the NLP's own initial conditions.
        # The NLP leaves the initial tension / s_dot / speed_radial / steering
        # free (only the radius is fixed); marching from the stale start_state
        # guess can converge the per-node root solver to a different root (or
        # NaN) once the winch/depower have moved a lot.
        self._update_start_state_from_solution(solution, opti_vars)

        optimized_trajectory = self._extract_optimized_trajectory(solution, opti_vars)

        return SimulationResult(
            solution=solution,
            optimized_config=self.pattern_config,
            final_distance=objective_dict.get("distance_radial_final", 0.0),
            phase_variables=opti_vars,
            optimized_trajectory=optimized_trajectory,
            energy_objective=objective_dict.get("energy", 0.0),
            total_time=objective_dict.get("total_time", 0.0),
        )

    def _param_reference(self, name: str) -> Any:
        """Current (previous-iterate) config value for an optimized parameter."""
        for entry in ("path_parameters", "radial_parameters", "sim_parameters"):
            section = self.pattern_config.get(entry, {})
            if name in section:
                return section[name]
        return None

    @staticmethod
    def _param_scale(name: str, ref: Any) -> float:
        """Normalization scale for a parameter: bound width, else |ref|."""
        bounds = DEFAULT_OPTI_LIMITS.get(name)
        if bounds and len(bounds) == 2 and (bounds[1] - bounds[0]):
            return float(bounds[1] - bounds[0])
        arr = np.ravel(np.asarray(ref, dtype=float))
        return max(float(np.max(np.abs(arr))), 1e-6)

    def _trust_region_penalty(self, opti_params: Dict[str, Any], weight: float) -> Any:
        """Quadratic penalty anchoring params to their current config values.

        Each deviation is normalized by the parameter's bound width so the
        penalty is dimensionless and comparable across parameters. Added to the
        objective to damp block-coordinate steps and keep the iterate in the
        productive region.
        """
        penalty = 0
        for name, var in opti_params.items():
            ref = self._param_reference(name)
            if ref is None:
                continue
            scale = self._param_scale(name, ref)
            penalty = penalty + ca.sumsqr((var - ca.DM(ref)) / scale)
        return weight * penalty

    def _update_start_state_from_solution(
        self, solution: Any, opti_vars: Dict[str, Any]
    ) -> None:
        """Update ``self.start_state`` with the optimized node-0 state.

        Pulls the first-node values of the free state variables from the NLP
        solution so the next ``run_simulation()`` warm-starts the per-node
        root solver at the optimum, keeping the re-simulation consistent with
        the optimized trajectory.
        """
        new_state = copy.deepcopy(self.start_state)
        for key in (
            # "s",
            # "s_dot",
            "speed_radial",
            "distance_radial",
            "input_steering",
            # "tension_tether_ground",
        ):
            var = opti_vars.get(key)
            if var is None:
                continue
            try:
                new_state[key] = float(solution.value(var[0]))
            except Exception:
                continue
        self.start_state = new_state

    def _extract_optimized_trajectory(
        self, solution: Any, opti_vars: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        """Read the optimizer's own per-node trajectory from the NLP solution.

        Returns a dict of length-``n_points`` numpy arrays evaluated directly
        from ``solution.value(opti_vars[...])`` — the optimum itself, not the
        re-simulated reconstruction. ``s`` (the s-grid) carries ``n_points + 1``
        entries, so it is trimmed to the ``n_points`` node values that align with
        the per-node decisions. Best-effort: any variable that cannot be
        evaluated is skipped, and the whole thing degrades to an empty dict
        rather than raising.
        """
        try:
            n_points = int(self.pattern_config["sim_parameters"]["n_points"])
        except (KeyError, TypeError, ValueError):
            n_points = None

        trajectory: Dict[str, np.ndarray] = {}
        for key in (
            "s",
            "s_dot",
            "input_steering",
            "speed_radial",
            "distance_radial",
            "tension_tether_ground",
            "input_depower",
        ):
            var = opti_vars.get(key)
            if var is None:
                continue
            try:
                values = np.asarray(solution.value(var), dtype=float).ravel()
            except Exception:
                continue
            if n_points is not None:
                values = values[:n_points]
            trajectory[key] = values
        return trajectory

    def run_opti(
        self,
        opti: Any,
        objective: Any,
        warm_start: bool = False,
        max_iter: Optional[int] = None,
    ) -> Optional[Any]:
        """Run the optimization problem.

        Args:
            opti: CasADi Opti instance
            objective: Objective function to minimize
            warm_start: when True, start IPOPT with a tight barrier and small
                bound pushes so a near-optimal guess does not trigger the
                cold-start excursion to the analytic center and back.

        Returns:
            Solution object or None if optimization failed
        """
        opti.minimize(objective)
        sim_parameters = self.pattern_config.get("sim_parameters", {})

        warm_start_init_point = sim_parameters.get("warm_start_init_point")
        if isinstance(warm_start_init_point, bool):
            warm_start_init_point = "yes" if warm_start_init_point else "no"

        ipopt_options = {
            # "bound_relax_factor": 1e-8,
            # tol was tightened 1e-5 -> 1e-6 in d899511 (2026-06-16), which made the
            # reel-out optimisation grind to max_iter: L-BFGS cannot drive dual
            # infeasibility to 1e-6 on the flat power objective. Restored to 1e-5
            # (acceptable_tol=2e-4 still provides the early-out).
            "tol": 1e-5,
            # The power objective is flat near the optimum and L-BFGS floors dual
            # infeasibility ~1e-2 there, so tol/acceptable_tol are never met and the
            # solve runs to max_iter. Also accept a plateaued, feasible point: stop
            # once the relative objective stops changing for acceptable_iter steps.
            "acceptable_iter": 5,
            "acceptable_tol": 2e-4,
            "acceptable_obj_change_tol": 1e-3,
            # "constr_viol_tol": 1e-6,
            "dual_inf_tol": 1e-4,
            "hessian_approximation": "limited-memory",
            "mu_strategy": "adaptive",
            "nlp_scaling_method": "gradient-based",
            "linear_solver": "mumps",
            # "limited_memory_max_history": 60,  # try 20–50
            # "limited_memory_update_type": "bfgs",  # (if supported)
            "mu_min": 1e-8,
            # "warm_start_bound_push": 1e-6,
            # "warm_start_mult_bound_push": 1e-6,
            # "warm_start_slack_bound_push": 1e-6,
            "max_iter": max_iter if max_iter is not None else 1000,
        }
        if warm_start:
            # Tight-barrier warm solve: start near the (near-optimal) guess
            # instead of the analytic center, suppressing the explore-out-and-
            # return excursion. Pairs with a primal warm start.
            ipopt_options.update(
                {
                    "mu_init": 1e-4,
                    "warm_start_init_point": "yes",
                    "warm_start_bound_push": 1e-6,
                    "warm_start_mult_bound_push": 1e-6,
                    "warm_start_slack_bound_push": 1e-6,
                    "bound_push": 1e-6,
                    "bound_frac": 1e-6,
                }
            )
        elif warm_start_init_point is not None:
            ipopt_options["warm_start_init_point"] = warm_start_init_point

        # SX expansion (expand=True) compiles the MX graph to scalar SX, a large
        # per-evaluation speedup for the multi-variable ROM aero Jacobian (the
        # bulk of the wall-time). It is unsupported when the NLP graph contains an
        # interpolant/external (tabulated wind, custom_spline winch), so fall back
        # to MX in that case. Toggle via sim_parameters["expand_nlp"].
        expand_nlp = bool(sim_parameters.get("expand_nlp", True))

        # Route simple variable box bounds (steering / depower / s_dot /
        # speed_radial / tension / radius limits, added as one-sided
        # ``subject_to`` rows) to IPOPT's lbx/ubx instead of general inequality
        # constraints. Behaviour-preserving on the *feasible set* (removes ~half
        # the inequality rows from ``g`` and the KKT, shrinking the
        # factorization), but on the flat power ridge it changes IPOPT's path
        # and therefore which equal-power point it lands on -- so it is OFF by
        # default and only safe to enable together with a regularizer
        # (``reg_weight``) that makes the optimum unique. Opt in via
        # ``sim_parameters["detect_simple_bounds"]``.
        detect_simple_bounds = bool(sim_parameters.get("detect_simple_bounds", False))

        def _set_solver(expand_flag):
            opti.solver(
                "ipopt",
                {
                    "expand": expand_flag,
                    "detect_simple_bounds": detect_simple_bounds,
                    "ipopt": ipopt_options,
                },
            )

        _set_solver(expand_nlp)

        try:
            try:
                solution = opti.solve()
            except Exception as expand_exc:
                msg = str(expand_exc).lower()
                if expand_nlp and (
                    "expand" in msg or "interpolant" in msg or "sx" in msg
                ):
                    print(
                        f"NLP expand=True unsupported for this model ({expand_exc}); "
                        "retrying without expand."
                    )
                    _set_solver(False)
                    solution = opti.solve()
                else:
                    raise

            print("\nOptimized Pattern Variables:")
            optimized_config = self.pattern_config.copy()
            for var_name, mx in self._opti_params.items():
                val = solution.value(mx)
                print(f"  {var_name}: {val}")
                if var_name in optimized_config.get("path_parameters", {}):
                    optimized_config["path_parameters"][var_name] = val
                elif var_name in optimized_config.get("radial_parameters", {}):
                    optimized_config["radial_parameters"][var_name] = val
                elif var_name in optimized_config.get("sim_parameters", {}):
                    optimized_config["sim_parameters"][var_name] = val
            self.pattern_config = optimized_config
            return solution

        except Exception as exc:
            print("Debug optimization information:")
            for var_name, mx in self._opti_params.items():
                try:
                    print(f"  {var_name}: {opti.debug.value(mx)}")
                except Exception:
                    pass
            print("Optimization failed:", exc)
            return None

    def run_simulation(
        self,
        *,
        run_plots: bool = False,
        axes: Any = None,
        phase_sim: bool = True,
        start_state: Optional[Dict[str, Any]] = None,
        return_start_state: bool = False,
    ) -> None:
        """Execute the reel-out simulation.

        Args:
            run_plots: When True, produce overview plots
            axes: Optional axes for plotting
            phase_sim: Whether to run in phase mode
            s_dot: Optional override for initial tangential speed (default: 4)
            speed_radial: Optional override for initial radial speed (default: -1)
            start_state: Optional explicit initial state. If provided, it is used
                as-is for the first simulation call; later calls can reuse the
                returned final_state to continue from where the previous
                simulation ended.
            return_final_state: When True, also return a compact final_state
                dictionary suitable for warm-starting subsequent runs.
        """

        # self.system_model.input_depower = self.depower

        # Use provided values or defaults
        phase = self._run_parametrized_phase(
            label_prefix="a",
            pattern_config=self.pattern_config,
            phase_sym=phase_sim,
            start_state=start_state,
        )

        if run_plots:
            if axes is not None:
                fig, axes, _ = phase.plot_overview_3d(
                    x_param="t",
                    variables=self.variables_to_plot,
                    axes=axes,
                )
            else:
                fig, axes, _ = phase.plot_overview_3d(
                    x_param="t",
                    variables=self.variables_to_plot,
                )
        print(
            "final radial distance:",
            phase.return_variable("distance_radial")[-1],
        )
        if return_start_state:
            # Collect a compact final-state snapshot for warm-starting follow-on runs
            start_state: Dict[str, Any] = {}
            for var in [
                "t",
                "s",
                "s_dot",
                "speed_radial",
                "distance_radial",
                "input_steering",
                "tension_tether_ground",
            ]:
                try:
                    series = phase.return_variable(var)
                    start_state[var] = float(series[0])
                except Exception:
                    # If a variable is not recorded, skip it
                    continue
            self._last_start_state = start_state
            return phase, axes or None, start_state

        return phase, axes or None

    def _run_parametrized_phase(
        self,
        label_prefix: str,
        pattern_config: Dict[str, Any],
        phase_sym: bool = False,
        start_state: Optional[Dict[str, Any]] = None,
    ) -> PhaseParameterized:
        """Run a parametrized phase simulation.

        Args:
            label_prefix: Prefix for labeling outputs
            pattern_config: Configuration for this phase
            phase_sym: Whether to run in symbolic mode
            s_dot: Optional override for initial tangential speed
            speed_radial: Optional override for initial radial speed
            start_state: Optional explicit initial state (used directly if set)

        Returns:
            PhaseParameterized object with simulation results
        """
        sim_type = "quasi-steady" if self.quasi_steady else "dynamic"
        print(f"Running simulation for {sim_type} with label: {label_prefix}")

        # Use caller-provided start_state when given; otherwise fall back to
        # defaults for the first run.
        if start_state is not None:
            start_state = copy.deepcopy(start_state)
            # start_state["s_dot"] += 0.5
            # start_state["tension_tether_ground"] = 8.3e4
            # start_state["input_steering"] += 0
            # start_state["distance_radial"] = pattern_config["path_parameters"]["r0"]
        else:
            start_state = self.start_state

        phase = PhaseParameterized(
            self.system_model,
            quasi_steady=self.quasi_steady,
            pattern_config=pattern_config,
        )
        if phase_sym:
            phase.run_simulation_phase(start_state=start_state)
        else:
            phase.run_simulation(start_state=start_state)
        return phase
