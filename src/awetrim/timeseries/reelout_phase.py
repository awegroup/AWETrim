"""Single-phase reel-out optimization and simulation.

This module provides the ReeloutSimple class for optimizing and simulating
reel-out maneuvers for airborne wind energy systems.
"""

from tracemalloc import start
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import copy
import warnings
import casadi as ca
import numpy as np
import yaml
import matplotlib.pyplot as plt

from awetrim.timeseries.phase_parametrized import PhaseParameterized
from awetrim.utils.defaults import DEFAULT_RADIAL_PARAMETERS


START_STATE_REELOUT = {
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


class Reelout:
    """Handles single-phase reel-out optimization and simulation.

    This class manages the optimization and simulation of a reel-out maneuver
    with configurable pattern type, path parameters, and radial parameters.

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
        >>> reelout = ReeloutSimple(
        ...     system_model=my_model,
        ...     pattern_config=config,
        ...     depower=1.0
        ... )
        >>> result = reelout.run_simulation_opti()
    """

    def __init__(
        self,
        *,
        system_model: Any,  # Should be SystemModel but avoiding circular import
        pattern_config: Optional[Dict[str, Any]] = None,
        depower: float = None,
        start_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize ReeloutSimple instance.

        Args:
            system_model: The system model to use for simulation/optimization
            pattern_config: Configuration dictionary with pattern_type, path_parameters,
                          and radial_parameters
            depower: Depower setting for the kite (0 to 1)
        """
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
        self.start_state = start_state or START_STATE_REELOUT.copy()

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
        opti_depower = False
        for var_name, mx in self._opti_params.items():
            for entry in ["path_parameters", "radial_parameters", "sim_parameters"]:
                if var_name in pattern_config_opti.get(entry, {}):
                    pattern_config_opti[entry][var_name] = mx
            if var_name == "input_depower":
                self.system_model.input_depower = mx
                # opti_depower = True

        # if not opti_depower:
        #     self.system_model.input_depower = self.pattern_config["sim_parameters"].get(
        #         "input_depower", 0.0
        #     )

        self._phase = PhaseParameterized(
            self.system_model,
            quasi_steady=True,
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

        if optimization_params:
            for var in optimization_params:
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
    ) -> Optional[SimulationResult]:
        """Run optimization and return results.

        Args:
            optimization_params: List of parameters to optimize

        Returns:
            SimulationResult object or None if optimization failed
        """
        opti, opti_vars, objective_dict, self._opti_params = self.get_opti_components(
            optimization_params=optimization_params
        )

        # Maximize average power
        if target == "power":
            total_objective = -(
                objective_dict["energy"]
                / objective_dict["total_time"]
                / objective_dict["power_scale"]
            )
        elif target == "energy":
            total_objective = -objective_dict["energy"]
        elif target == "zero":
            total_objective = 0.0

        solution = self.run_opti(opti, total_objective)
        if solution is None:
            return None

        return SimulationResult(
            solution=solution,
            optimized_config=self.pattern_config,
            final_distance=objective_dict.get("distance_radial_final", 0.0),
            phase_variables=opti_vars,
            energy_objective=objective_dict.get("energy", 0.0),
            total_time=objective_dict.get("total_time", 0.0),
        )

    def run_opti(self, opti: Any, objective: Any) -> Optional[Any]:
        """Run the optimization problem.

        Args:
            opti: CasADi Opti instance
            objective: Objective function to minimize

        Returns:
            Solution object or None if optimization failed
        """
        opti.minimize(objective)
        opti.solver(
            "ipopt",
            {
                "ipopt": {
                    # "bound_relax_factor": 1e-8,
                    "tol": 1e-6,
                    "acceptable_iter": 10,
                    "acceptable_tol": 2e-4,
                    # "constr_viol_tol": 1e-6,
                    "dual_inf_tol": 1e-4,
                    "hessian_approximation": "limited-memory",
                    "mu_strategy": "adaptive",
                    "nlp_scaling_method": "gradient-based",
                    "linear_solver": "mumps",
                    "limited_memory_max_history": 60,  # try 20–50
                    # "limited_memory_update_type": "bfgs",  # (if supported)
                    "mu_min": 1e-8,
                    "warm_start_init_point": "yes",
                    # "warm_start_bound_push": 1e-6,
                    # "warm_start_mult_bound_push": 1e-6,
                    # "warm_start_slack_bound_push": 1e-6,
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
        sim_type = "quasi steady"
        print(f"Running simulation for {sim_type} with label: {label_prefix}")

        # Use caller-provided start_state when given; otherwise fall back to
        # defaults for the first run.
        if start_state is not None:
            start_state = copy.deepcopy(start_state)
            start_state["s_dot"] += 0.5
            start_state["tension_tether_ground"] = 8.3e4
            start_state["input_steering"] += 0
            start_state["distance_radial"] = pattern_config["path_parameters"]["r0"]
        else:
            start_state = self.start_state

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
