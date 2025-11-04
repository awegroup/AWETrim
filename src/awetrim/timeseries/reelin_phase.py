import copy
import casadi as ca
import matplotlib.pyplot as plt
import numpy as np

from awetrim.timeseries.phase_parametrized import PhaseParameterized
from awetrim.utils.defaults import DEFAULT_RADIAL_PARAMETERS


from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
import warnings


@dataclass
class SimulationResult:
    """Container for simulation results and optimization outputs."""

    solution: Any  # CasADi solution object
    optimized_config: Dict[str, Any]
    phase_variables: Dict[str, Any]
    energy_objective: float
    total_time: float


class ReelinSimple:
    """Encapsulates the simple reel-in workflow for parametrized simulations.

    This class handles the optimization and simulation of a two-phase kite reel-in
    maneuver: a pure reel-in phase followed by a transition phase. It manages the
    optimization of key parameters (e.g., elevation angles) to maximize efficiency
    while meeting physical constraints.

    Key Features:
        - Two-phase optimization (reel-in + transition)
        - Automatic parameter validation and default handling
        - Integrated plotting and visualization
        - Support for both symbolic (optimization) and numeric (simulation) modes

    Example:
        >>> config = {
        ...     "path_parameters": {
        ...         "distance_radial_end": 100,  # Required
        ...         "elevation_start_ri": np.radians(35),  # Optional, has default
        ...     }
        ... }
        >>> reelin = ReelinSimple(
        ...     system_model=my_model,
        ...     pattern_config=config,
        ...     depower_ri=0.8
        ... )
        >>> result = reelin.run_simulation_opti()
        >>> print(f"Optimized elevation: {result.optimized_config['elevation_start_riro']}")
    """

    def __init__(
        self,
        *,
        system_model: Any,  # Should be SystemModel but avoiding circular import
        pattern_config: Optional[Dict[str, Any]] = None,
        depower_ri: float = 1.0,
        depower_riro: float = 1.0,
    ) -> None:

        self.pattern_config = pattern_config or {}
        # Required configuration parameters and their default values (None means required with no default)
        self._required_config = {
            "path_parameters": {
                "elevation_start_ri": np.radians(30),
                "elevation_start_ro": np.radians(30),
                "elevation_start_riro": np.radians(120),
                "distance_radial_start": 360,
                "distance_radial_end": None,  # Must be provided
            },
        }
        # self._validate_config()

        self.depower_ri = depower_ri
        self.depower_riro = depower_riro

        # Derived configuration/state containers
        self.variables_to_plot = [
            "speed_tangential",
            "tension_tether_ground",
            "angle_elevation",
            "distance_radial",
        ]

        # Components and state placeholders
        self.system_model = system_model
        self.create_ri_dicts()
        self.create_riro_dicts()
        self._opti_params = {}
        self._opti = ca.Opti()

    def create_ri_dicts(self):
        self.radial_parameters_ri = self.pattern_config.get(
            "radial_parameters", DEFAULT_RADIAL_PARAMETERS
        )
        elevation_start_ri = self.pattern_config["path_parameters"].get(
            "elevation_start_ri", np.radians(30)
        )
        elevation_start_riro = self.pattern_config["path_parameters"].get(
            "elevation_start_riro", np.radians(80)
        )
        distance_radial_start = self.pattern_config["path_parameters"].get(
            "distance_radial_start", 360
        )
        start_time = self.pattern_config.get("sim_parameters", {}).get("start_time", 0)
        self.start_state_ri = {
            "t": start_time,
            "s": 0,
            "s_dot": 0.2,
            "input_steering": 0,
            "tension_tether_ground": 1e8,
            "distance_radial": distance_radial_start,
            "speed_radial": -6,
        }
        self.pattern_config_ri = {
            "pattern_type": "reel_in_simple",
            "path_parameters": {
                "elevation_start_ri": elevation_start_ri,
                "elevation_start_riro": elevation_start_riro,
            },
            "radial_parameters": self.radial_parameters_ri,
            "sim_parameters": {
                "start_angle": 0,
                "end_angle": elevation_start_riro - elevation_start_ri,
                "n_points": 100,
            },
        }

    def create_riro_dicts(self):
        self.radial_parameters_riro = self.pattern_config.get(
            "radial_parameters", DEFAULT_RADIAL_PARAMETERS
        )
        elevation_start_ro = self.pattern_config["path_parameters"].get(
            "elevation_start_ro", np.radians(30)
        )
        elevation_start_riro = self.pattern_config["path_parameters"].get(
            "elevation_start_riro", np.radians(80)
        )
        distance_radial_start = self.pattern_config["path_parameters"].get(
            "distance_radial_start", 360
        )
        self.start_state_riro = {
            "t": 0,
            "s": 0,
            "s_dot": 0.2,
            "input_steering": 0,
            "tension_tether_ground": 1e8,
            "distance_radial": distance_radial_start,
            "speed_radial": -6,
        }
        self.pattern_config_riro = {
            "pattern_type": "transition_simple",
            "path_parameters": {
                "elevation_start_ro": elevation_start_ro,
                "elevation_start_riro": elevation_start_riro,
            },
            "radial_parameters": self.radial_parameters_riro,
            "sim_parameters": {
                "start_angle": 0,
                "end_angle": elevation_start_riro - elevation_start_ro,
                "n_points": 100,
            },
        }

    def initialize_ri_phase(
        self,
        start_state_opti: Optional[Dict[str, float]] = None,
        pattern_config_opti: Optional[Dict[str, Any]] = None,
    ):
        """Prepare the initial reel-in optimization phase."""

        self.create_ri_dicts()
        self.system_model.input_depower = self.depower_ri
        if pattern_config_opti is None:
            pattern_config_opti = copy.deepcopy(self.pattern_config_ri)
        if start_state_opti is None:
            start_state_opti = copy.deepcopy(self.start_state_ri)
        for var_name, mx in self._opti_params.items():
            for entry in ["path_parameters", "radial_parameters", "sim_parameters"]:
                if var_name in pattern_config_opti.get(entry, {}):
                    pattern_config_opti[entry][var_name] = mx
            if var_name == "elevation_start_riro":
                pattern_config_opti["sim_parameters"]["end_angle"] = (
                    mx - self.pattern_config_ri["path_parameters"]["elevation_start_ri"]
                )
        self._phase_ri = PhaseParameterized(
            self.system_model,
            quasi_steady=True,
            pattern_config=self.pattern_config_ri,
            pattern_config_opti=pattern_config_opti,
        )
        (
            self._opti,
            self._opti_vars_ri,
            self._objective_ri,
        ) = self._phase_ri.opti_phase(
            start_state=self.start_state_ri,
            opti=self._opti,
            start_state_opti=start_state_opti,
            opti_params=self._opti_params,
            relax_tol=0.01,
        )
        return self._phase_ri

    def initialize_riro_phase(self, pattern_config_opti=None, start_state_opti=None):
        """Extend the optimization problem with the transition phase setup."""

        self.create_riro_dicts()
        self.system_model.input_depower = self.depower_riro
        if pattern_config_opti is None:
            pattern_config_opti = copy.deepcopy(self.pattern_config_riro)
        if start_state_opti is None:
            start_state_opti = copy.deepcopy(self.start_state_riro)
        start_state_opti["distance_radial"] = self._opti_vars_ri["distance_radial"][-1]
        for var_name, mx in self._opti_params.items():
            for entry in ["path_parameters", "radial_parameters", "sim_parameters"]:
                if var_name in pattern_config_opti.get(entry, {}):
                    pattern_config_opti[entry][var_name] = mx
            if var_name == "elevation_start_riro":
                pattern_config_opti["sim_parameters"]["end_angle"] = (
                    mx - self.pattern_config["path_parameters"]["elevation_start_ro"]
                )
                pattern_config_opti["path_parameters"]["elevation_start_riro"] = mx

        self._phase_riro = PhaseParameterized(
            self.system_model,
            quasi_steady=True,
            pattern_config=self.pattern_config_riro,
            pattern_config_opti=pattern_config_opti,
        )
        (
            self._opti,
            self._opti_vars_riro,
            self._objective_riro,
        ) = self._phase_riro.opti_phase(
            start_state=self.start_state_riro,
            opti=self._opti,
            start_state_opti=start_state_opti,
            opti_params=self._opti_params,
            relax_tol=0.01,
        )
        return self._phase_riro

    def get_opti_components(
        self,
        optimization_params=None,
        opti=None,
        optimization_dict=None,
        start_state_opti: Optional[Dict[str, float]] = None,
    ):
        """Solve the optimization problem for the transition phase."""

        if opti is None:
            opti = ca.Opti()

        self._opti = opti

        self._opti_params = {}
        if optimization_params:
            for var in optimization_params:
                self._opti_params[var] = opti.variable()
        elif optimization_dict:
            self._opti_params = optimization_dict

        self.initialize_ri_phase(start_state_opti=start_state_opti)

        self.initialize_riro_phase()

        # Combined optimization variables: expand/concatenate list/array-like entries
        combined_opti_vars = self._merge_opti_vars(
            self._opti_vars_ri, self._opti_vars_riro
        )
        # Combined objective: add scalar or per-phase objective values together
        combined_objective = self._merge_objective(
            self._objective_ri, self._objective_riro
        )
        return self._opti, combined_opti_vars, combined_objective, self._opti_params

    def run_simulation_opti(
        self,
        optimization_params: List[str] = ["elevation_start_riro"],
        target: str = "energy",
    ) -> Optional[SimulationResult]:
        """Set up and solve the optimization problem for the reel-in and transition.

        This method combines both phases (reel-in and transition) into a single
        optimization problem, minimizing the objective while satisfying constraints.
        The objective is computed as -(energy / total_time / power_scale) to
        maximize average power.

        Args:
            optimization_params: List of parameter names to optimize. These parameters
                must exist in the pattern configuration. Common choices include:
                - elevation_start_riro: Transition phase starting elevation
                - Other path or radial parameters defined in pattern_config

        Returns:
            SimulationResult object containing the optimization solution and derived
            values, or None if the optimization failed.

        Raises:
            ValueError: If any optimization parameter is missing from pattern_config
            RuntimeError: If the optimization encounters numerical difficulties

        Example:
            >>> result = reelin.run_simulation_opti(['elevation_start_riro'])
            >>> if result:
            ...     print(f"Optimal elevation: {result.optimized_config['elevation_start_riro']}")
            ...     print(f"Energy objective: {result.energy_objective}")
        """
        opti, opti_vars, objective_dict, self._opti_params = self.get_opti_components(
            optimization_params=optimization_params,
        )
        opti.subject_to(
            self._opti_vars_riro["distance_radial"][-1]
            == self.pattern_config["path_parameters"]["distance_radial_end"]
        )
        if target == "energy":
            total_objective = -(objective_dict["energy"])
        elif target == "power":
            total_objective = -(
                objective_dict["energy"]
                / objective_dict["total_time"]
                / objective_dict["power_scale"]
            )
        else:
            total_objective = 0.0

        solution = self.run_opti(opti, total_objective)
        if solution is None:
            return None

        # Package results in a cleaner format
        return SimulationResult(
            solution=solution,
            optimized_config=self.pattern_config,  # Already updated in run_opti
            phase_variables=opti_vars,
            energy_objective=objective_dict.get("energy", 0.0),
            total_time=objective_dict.get("total_time", 0.0),
        )

    def run_opti(self, opti, objective):
        """Solve the supplied opti problem and return the solution."""

        opti.minimize(objective)
        opti.solver(
            "ipopt",
            {
                "ipopt": {
                    "bound_relax_factor": 1e-8,
                    "tol": 1e-2,
                    "acceptable_iter": 3,
                    "acceptable_tol": 1e-2,
                    "constr_viol_tol": 1e-2,
                    "dual_inf_tol": 1e-2,
                    "hessian_approximation": "limited-memory",
                    "mu_strategy": "adaptive",
                }
            },
        )

        try:
            solution = opti.solve()

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
            self.create_ri_dicts()
            self.create_riro_dicts()
            return solution

        except Exception as exc:
            print("Debug optimization information:")
            optimized_config = self.pattern_config.copy()
            for var_name, mx in self._opti_params.items():
                val = opti.debug.value(mx)
                print(f"  {var_name}: {val}")
                if var_name in optimized_config.get("path_parameters", {}):
                    optimized_config["path_parameters"][var_name] = val
                elif var_name in optimized_config.get("radial_parameters", {}):
                    optimized_config["radial_parameters"][var_name] = val
                elif var_name in optimized_config.get("sim_parameters", {}):
                    optimized_config["sim_parameters"][var_name] = val
            self.pattern_config = optimized_config
            self.create_ri_dicts()
            self.create_riro_dicts()
            print("Optimization failed:", exc)
            return None

    def run_simulation(self, *, solution=None, run_plots=False):
        """Execute the reel-in and transition simulations.

        Args:
            solution: Optional CasADi solution produced via `run_opti`. If omitted,
                the method uses the latest stored solution when available, otherwise
                it relies on the current pattern configuration.
            run_plots: When True, produce overview plots using Matplotlib.
        """
        phase_ri = self._run_parametrized_phase(
            label_prefix="a",
            depower=self.depower_ri,
            start_state=self.start_state_ri,
            pattern_config=self.pattern_config_ri,
            phase_sym=True,
        )
        fig = axes = None
        if run_plots:
            fig, axes, _ = phase_ri.plot_overview_3d(
                x_param="t",
                variables=self.variables_to_plot,
            )

        elevation_riro = phase_ri.return_variable("angle_elevation")[-1]
        t_start = phase_ri.return_variable("t")[-1]
        r_start = phase_ri.return_variable("distance_radial")[-1]
        self.pattern_config_riro["path_parameters"][
            "elevation_start_riro"
        ] = elevation_riro
        self.pattern_config_riro["sim_parameters"]["end_angle"] = (
            self.pattern_config["path_parameters"]["elevation_start_riro"]
            - self.pattern_config["path_parameters"]["elevation_start_ro"]
        )

        start_state = copy.deepcopy(self.start_state_riro)
        start_state["distance_radial"] = r_start
        start_state["t"] = t_start
        phase_riro = self._run_parametrized_phase(
            label_prefix="a",
            depower=self.depower_riro,
            start_state=start_state,
            pattern_config=self.pattern_config_riro,
            phase_sym=True,
        )
        if run_plots and axes is not None:
            phase_riro.plot_overview_3d(
                x_param="t",
                variables=self.variables_to_plot,
                axes=axes,
            )
        plt.show()

        print(
            "final radial distance:",
            phase_riro.return_variable("distance_radial")[-1],
        )

    def _run_parametrized_phase(
        self,
        label_prefix,
        depower,
        start_state,
        pattern_config,
        phase_sym=False,
    ):
        """Run a parametrized phase simulation and return the PhaseParameterized object."""
        sim_type = "quasi steady"
        print(f"Running simulation for {sim_type} with label: {label_prefix}")

        self.system_model.input_depower = depower
        phase = PhaseParameterized(
            self.system_model,
            quasi_steady=True,
            pattern_config=pattern_config,
        )
        if phase_sym:
            phase.run_simulation_phase(start_state=start_state)
        else:
            phase.run_simulation(start_state=start_state)
        return phase

    @staticmethod
    def _merge_phase_dicts(primary, secondary):
        """Merge two phase dictionaries.

        Behaviour changes compared to previous implementation:
        - If a key appears in both dicts and both values are scalar-like (not
          list/tuple), the values are summed (supports numbers and CasADi
          expressions).
        - If either existing or new value is a list/tuple, both are converted to
          lists and concatenated (preserving per-phase sequences).
        - Scalars are copied as-is when first seen.

        This prevents accidental conversion of scalar objective components
        (e.g. energy, total_time) into lists, which caused TypeError on
        arithmetic operations.
        """

        def is_sequence(val):
            return isinstance(val, (list, tuple))

        def coerce_copy(value):
            if isinstance(value, list):
                return value.copy()
            if isinstance(value, tuple):
                return list(value)
            return value

        def to_list(value):
            if isinstance(value, list):
                return value.copy()
            if isinstance(value, tuple):
                return list(value)
            return [value]

        merged = {}
        for source in (primary, secondary):
            if not source:
                continue
            for key, value in source.items():
                prepared = coerce_copy(value)
                if key not in merged:
                    merged[key] = prepared
                    continue

                existing = merged[key]

                # If either side is a sequence, concatenate both as lists
                if is_sequence(existing) or is_sequence(prepared):
                    merged[key] = to_list(existing) + to_list(prepared)
                else:
                    # Both are scalars (could be numbers or CasADi expressions).
                    # Prefer summation so objective components remain scalar-like.
                    try:
                        merged[key] = existing + prepared
                    except Exception:
                        # Fallback: if addition fails for any reason, fall back to
                        # list concatenation to preserve data.
                        merged[key] = to_list(existing) + to_list(prepared)

        return merged

    @staticmethod
    def _merge_opti_vars(primary, secondary):
        """Merge optimization variable dictionaries by expanding list/array-like entries.

        For entries that are lists, tuples or numpy arrays, the result will be a
        single flat Python list containing elements from both phases. Scalar
        entries will be converted into single-item lists and concatenated.
        """

        def to_list(val):
            if isinstance(val, list):
                return val.copy()
            if isinstance(val, tuple):
                return list(val)
            if isinstance(val, np.ndarray):
                return list(val.tolist())
            return [val]

        merged = {}
        for source in (primary, secondary):
            if not source:
                continue
            for key, value in source.items():
                prepared = value
                if key not in merged:
                    # store as-is, but convert numpy arrays to list for consistency
                    if isinstance(prepared, np.ndarray):
                        merged[key] = list(prepared.tolist())
                    else:
                        merged[key] = (
                            prepared.copy() if isinstance(prepared, list) else prepared
                        )
                    continue

                existing = merged[key]
                # If either side is array-like/sequence, concatenate as lists
                if isinstance(existing, (list, tuple, np.ndarray)) or isinstance(
                    prepared, (list, tuple, np.ndarray)
                ):
                    merged[key] = to_list(existing) + to_list(prepared)
                else:
                    # Both scalars: expand into list of two elements
                    merged[key] = [existing, prepared]

        return merged

    @staticmethod
    def _merge_objective(primary, secondary):
        """Merge objective dictionaries by summing scalar or per-phase values.

        If values are lists/tuples/arrays, they are summed element-wise using +
        semantics (works with CasADi expressions). If they are scalars, they are
        added. If addition fails, falls back to concatenation to preserve data.
        """

        def sum_val(v):
            if isinstance(v, (list, tuple, np.ndarray)):
                # sum entries into a single scalar/CasADi expression
                vals = list(v) if not isinstance(v, np.ndarray) else list(v.tolist())
                if len(vals) == 0:
                    return 0
                total = vals[0]
                for e in vals[1:]:
                    total = total + e
                return total
            return v

        def to_list(v):
            if isinstance(v, list):
                return v.copy()
            if isinstance(v, tuple):
                return list(v)
            if isinstance(v, np.ndarray):
                return list(v.tolist())
            return [v]

        merged = {}
        for source in (primary, secondary):
            if not source:
                continue
            for key, value in source.items():
                if key not in merged:
                    merged[key] = value
                    continue
                existing = merged[key]
                try:
                    merged[key] = sum_val(existing) + sum_val(value)
                except Exception:
                    # Fallback: preserve data by concatenating lists
                    merged[key] = to_list(existing) + to_list(value)

        return merged

    @staticmethod
    def _copy_phase_dict(source):
        if not source:
            return {}

        def coerce(value):
            if isinstance(value, list):
                return value.copy()
            if isinstance(value, tuple):
                return list(value)
            return value

        return {key: coerce(value) for key, value in source.items()}

    # def _validate_config(self):
    #     """Validate the pattern configuration and warn about missing required parameters.

    #     Checks if all required parameters are present in the configuration and issues
    #     warnings for any that are missing or using defaults.
    #     """
    #     missing_required = []
    #     using_defaults = []

    #     def check_section(required, actual, path=""):
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
    #         import warnings

    #         warnings.warn(
    #             f"Using default values for configuration parameters:\n  - {defaults_str}",
    #             RuntimeWarning,
    #             stacklevel=2,
    #         )
