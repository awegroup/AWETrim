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

"""Stateful reelout-optimization session driven by the REST endpoints.

Holds one ``SystemModel`` + ``Phase`` pair across repeated re-optimizations
(``/step`` calls). ``Phase`` is deliberately reused between steps: on a
successful solve it stores an exact node-wise warm start and writes the
optimized parameters back into ``pattern_config``, so each subsequent solve
starts from the previous optimum. On a failed solve nothing is mutated, so
retries also restart from the last good optimum.

This module has no FastAPI/pydantic dependency — the app layer passes plain
dicts (``pydantic .model_dump()`` output) and translates the exceptions
raised here into HTTP status codes.
"""

import copy
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import casadi as ca
import numpy as np

from awetrim.environment.wind_factory import create_wind_model
from awetrim.environment.wind_profiles import wind_kwargs_from_inflow
from awetrim.kinematics.parametrized_patterns import (
    create_pattern_from_dict,
    fit_bspline_pattern_to_trajectory,
    make_bspline_path_parameters_from_named_curve,
)
from awetrim.utils.defaults import DEFAULT_OPTI_LIMITS
from awetrim.system.factory import create_system_model_from_yaml
from awetrim.timeseries.phase import Phase
from awetrim.utils.config_paths import (
    LEI_V3_DOWNLOOP_SPLINE_CONFIG,
    LEI_V3_SYSTEM_CONFIG,
)
from awetrim.utils.utils import load_cycle_config_from_yaml

logger = logging.getLogger(__name__)

# Timeseries variables returned by /trajectory?resimulate=true
RESIM_VARIABLES = [
    "t",
    "x",
    "y",
    "z",
    "angle_azimuth",
    "angle_elevation",
    "distance_radial",
    "speed_radial",
    "speed_tangential",
    "tension_tether_ground",
    "mechanical_power",
    "angle_of_attack",
]


class SessionState(str, Enum):
    UNINITIALIZED = "uninitialized"
    READY = "ready"
    SOLVING = "solving"
    CONVERGED = "converged"
    FAILED = "failed"


class SessionNotInitializedError(RuntimeError):
    """Raised when /step or /trajectory is called before /init."""


class SessionBusyError(RuntimeError):
    """Raised when an operation conflicts with a running solve."""


class NoTrajectoryError(RuntimeError):
    """Raised when no successful solve exists yet."""


class SolveFailedError(RuntimeError):
    """Raised by a blocking step when the optimization does not converge
    (e.g. the requested trajectory/winch/wind combination is infeasible)."""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_optional_float_list(values: np.ndarray) -> List[Optional[float]]:
    """numpy array -> JSON-safe list (non-finite values become None)."""
    out: List[Optional[float]] = []
    for v in np.asarray(values, dtype=float).ravel():
        out.append(float(v) if np.isfinite(v) else None)
    return out


@dataclass
class SolveRecord:
    """Immutable snapshot of one successful re-optimization."""

    step_index: int
    table: Dict[str, np.ndarray]
    spline: Dict[str, Any]
    metrics: Dict[str, float]
    optimized_parameters: Dict[str, Any]
    started: datetime
    finished: datetime = field(default_factory=_utcnow)


def _compute_guidance_columns(
    pattern: Any, arrays: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    """Derive t, azimuth, elevation and their time derivatives per node.

    Total time derivatives follow ``ParametrizedKinematics.dphi_ds``:
    d(phi)/dt = (dphi/ds)*s_dot + (dphi/dr)*speed_radial (same for beta).
    """
    s = np.asarray(arrays["s"], dtype=float).ravel()
    r = np.asarray(arrays["distance_radial"], dtype=float).ravel()
    s_dot = np.asarray(arrays["s_dot"], dtype=float).ravel()
    v_r = np.asarray(arrays["speed_radial"], dtype=float).ravel()

    s_sym = ca.MX.sym("s")
    r_sym = ca.MX.sym("r")
    phi = pattern.azimuth(r_sym, s_sym)
    beta = pattern.elevation(r_sym, s_sym)
    angle_fun = ca.Function(
        "pattern_angles",
        [s_sym, r_sym],
        [
            phi,
            beta,
            ca.gradient(phi, s_sym),
            ca.gradient(beta, s_sym),
            ca.gradient(phi, r_sym),
            ca.gradient(beta, r_sym),
        ],
    )
    mapped = angle_fun.map(len(s))
    outputs = mapped(ca.DM(s).T, ca.DM(r).T)
    phi_v, beta_v, dphi_ds, dbeta_ds, dphi_dr, dbeta_dr = (
        np.asarray(o).ravel() for o in outputs
    )

    azimuth_dot = dphi_ds * s_dot + dphi_dr * v_r
    elevation_dot = dbeta_ds * s_dot + dbeta_dr * v_r

    # Time grid from ds / s_dot (trapezoidal), t[0] = 0.
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_sdot = np.where(s_dot != 0.0, 1.0 / s_dot, np.nan)
    dt = np.diff(s) * 0.5 * (inv_sdot[:-1] + inv_sdot[1:])
    t = np.concatenate([[0.0], np.cumsum(dt)])

    return {
        "t": t,
        "azimuth": phi_v,
        "elevation": beta_v,
        "azimuth_dot": azimuth_dot,
        "elevation_dot": elevation_dot,
    }


class ReeloutSession:
    """Single reelout-optimization session (one per server process)."""

    session_id = "default"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._clear()

    def _clear(self) -> None:
        self.state = SessionState.UNINITIALIZED
        self.system_model: Any = None
        self.phase: Optional[Phase] = None
        self.last_record: Optional[SolveRecord] = None
        self.last_error: Optional[str] = None
        self.step_count = 0
        self.successful_steps = 0
        self.last_step_started: Optional[datetime] = None
        self.last_step_finished: Optional[datetime] = None
        self._inflow: Optional[dict] = None
        self._wind_spec: Optional[dict] = None
        self._optimization_params: List[str] = []
        self._target: str = "power"
        self._n_points: Optional[int] = None
        self._name: str = "reelout-optimization"
        self._winch_params: Optional[dict] = None
        self._depower_mode: str = "optimize"
        # Resolution of the degree-table replies: matches the client-sent
        # trajectory when there is one, else falls back to n_points.
        self._trajectory_points: Optional[int] = None

    # ------------------------------------------------------------------
    # /init
    # ------------------------------------------------------------------
    def init(self, config: Dict[str, Any]) -> None:
        """Build SystemModel + Wind + Phase from an InitRequest-shaped dict."""
        with self._lock:
            if self.state == SessionState.SOLVING:
                raise SessionBusyError("cannot re-initialize while a solve is running")

        system_path = (
            Path(config["system_config_path"])
            if config.get("system_config_path")
            else LEI_V3_SYSTEM_CONFIG
        )
        cycle_path = (
            Path(config["cycle_config_path"])
            if config.get("cycle_config_path")
            else LEI_V3_DOWNLOOP_SPLINE_CONFIG
        )
        for path in (system_path, cycle_path):
            if not Path(path).exists():
                raise FileNotFoundError(f"config file not found: {path}")

        system_model = create_system_model_from_yaml(yaml_path=system_path)
        reelout_config, _ = load_cycle_config_from_yaml(cycle_path)
        reelout_config = copy.deepcopy(reelout_config)
        sim_parameters = reelout_config.setdefault("sim_parameters", {})

        # Pattern sphere radius: request override, else the cycle-config value.
        r0 = float(
            config.get("length") or reelout_config["path_parameters"]["r0"]
        )
        reelout_config["path_parameters"]["r0"] = r0

        guess = config.get("initial_guess")
        if guess:
            reelout_config["path_parameters"] = (
                make_bspline_path_parameters_from_named_curve(
                    spline_type="periodic",
                    r0=r0,
                    **guess,
                )
            )
            sim_parameters["start_angle"] = guess["s_init"]
            sim_parameters["end_angle"] = guess["s_final"]

        # A client-supplied flight path (degrees) takes precedence over the
        # named-curve guess: fit the pattern B-spline to it.
        trajectory = config.get("trajectory")
        if trajectory:
            self._fit_trajectory_into_path(
                reelout_config["path_parameters"], trajectory
            )

        winch = config.get("winch_params")
        if winch:
            self._apply_winch_params(
                reelout_config["radial_parameters"], sim_parameters, winch
            )

        n_points = int(config.get("n_points", 100))
        sim_parameters["n_points"] = n_points
        # The flat InitParams solver knobs; unset ones keep the cycle-config value.
        for key in ("input_depower", "reg_weight", "detect_simple_bounds"):
            value = config.get(key)
            if value is not None:
                sim_parameters[key] = value

        # Depower last: the explicit {mode, value} struct wins over the legacy
        # optimization_params / input_depower combination (which still works
        # unchanged when the struct is omitted).
        optimization_params = list(
            config.get("optimization_params")
            or ["C_phi", "C_beta", "input_depower"]
        )
        depower_mode = self._apply_depower(
            sim_parameters, optimization_params, config.get("depower")
        )
        self._apply_min_turn_radius(sim_parameters, config.get("min_turn_radius"))

        inflow = config.get("inflow_conditions")
        wind_spec = self._resolve_wind_spec(inflow)
        system_model.wind = self._build_wind(wind_spec)
        sim_parameters["expand_nlp"] = wind_spec["model_type"] != "tabulated"

        start_state = {
            "t": 0,
            "s": 0,
            "s_dot": 0.1,
            "input_steering": 0,
            "tension_tether_ground": 8.4e3,
            "speed_radial": 0,
            "distance_radial": r0,
        }
        phase = Phase(
            system_model=system_model,
            pattern_config=reelout_config,
            start_state=start_state,
        )
        # The server must never write failure plots from a worker thread.
        phase._plot_failed_iterate = lambda *args, **kwargs: None

        with self._lock:
            if self.state == SessionState.SOLVING:
                raise SessionBusyError("cannot re-initialize while a solve is running")
            self._clear()
            self.system_model = system_model
            self.phase = phase
            self._inflow = dict(inflow) if inflow else None
            self._wind_spec = wind_spec
            self._optimization_params = optimization_params
            self._depower_mode = depower_mode
            self._target = config.get("target", "power")
            self._n_points = n_points
            self._name = config.get("name") or "reelout-optimization"
            self._winch_params = dict(winch) if winch else None
            if trajectory:
                self._trajectory_points = len(trajectory["azimuth"])
            self.state = SessionState.READY

    @staticmethod
    def _resolve_wind_spec(inflow: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """InflowConditions -> ``create_wind_model`` kwargs."""
        if inflow is None:
            raise ValueError("no wind given: inflow_conditions is required")
        turbulence = inflow.get("turbulence") or 0.0
        if turbulence > 0.0:
            logger.warning(
                "inflow_conditions.turbulence=%g is ignored: the "
                "trajectory optimization is deterministic and uses the "
                "mean wind profile",
                turbulence,
            )
        return wind_kwargs_from_inflow(inflow)

    @staticmethod
    def _build_wind(spec: Dict[str, Any]) -> Any:
        spec = dict(spec)
        model_type = spec.pop("model_type")
        return create_wind_model(model_type, **spec)

    @staticmethod
    def _apply_depower(
        sim_parameters: Dict[str, Any],
        optimization_params: List[str],
        depower: Optional[Dict[str, Any]],
    ) -> str:
        """Apply a DepowerSpec-shaped dict; return the effective mode.

        Mutates ``sim_parameters`` and ``optimization_params`` in place so the
        three ways of driving the depower input stay a single switch:

        - ``fixed``   -> l_dp pinned at ``value``; dropped from the NLP.
        - ``optimize``-> one scalar l_dp optimized, seeded at ``value``.
        - ``profile`` -> l_dp optimized per node (a depower time-profile).

        With ``depower=None`` nothing is changed and the mode is *derived* from
        the legacy ``optimization_params`` / ``sim_parameters`` combination, so
        older clients keep their exact behaviour.
        """
        if depower is None:
            if sim_parameters.get("optimize_depower_profile"):
                return "profile"
            return "optimize" if "input_depower" in optimization_params else "fixed"

        mode = depower.get("mode") or "optimize"
        if mode not in ("fixed", "optimize", "profile"):
            raise ValueError(
                f"depower mode {mode!r} must be 'fixed', 'optimize' or 'profile'"
            )

        value = depower.get("value")
        if value is not None:
            sim_parameters["input_depower"] = float(value)
        else:
            # Guarantee the key exists: the optimizer writes the optimized
            # scalar back into whichever config section already holds it.
            sim_parameters.setdefault("input_depower", 1.6)

        sim_parameters["optimize_depower_profile"] = mode == "profile"
        if mode != "profile":
            # A previous profile solve would otherwise keep being re-flown.
            sim_parameters.pop("input_depower_profile", None)

        if mode == "fixed":
            while "input_depower" in optimization_params:
                optimization_params.remove("input_depower")
        elif "input_depower" not in optimization_params:
            optimization_params.append("input_depower")
        return mode

    def depower_state(
        self, n_points: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """DepowerReply-shaped dict: the setting the current path assumes.

        In profile mode the per-node values are resampled onto the same grid as
        :meth:`trajectory_degrees`, so ``profile[i]`` belongs to
        ``azimuth[i]``/``elevation[i]``.
        """
        if self.phase is None:
            return None
        pattern_config = self.phase.pattern_config
        sim_parameters = pattern_config.get("sim_parameters", {})

        profile = sim_parameters.get("input_depower_profile")
        if self._depower_mode == "profile" and profile is not None:
            values = np.asarray(profile, dtype=float).ravel()
            path_parameters = pattern_config["path_parameters"]
            s_init = float(path_parameters["s_init"])
            s_final = float(path_parameters["s_final"])
            n = int(n_points or self._trajectory_points or self._n_points or 100)
            s_src = np.linspace(s_init, s_final, values.size)
            s_dst = np.linspace(s_init, s_final, n, endpoint=True)
            resampled = np.interp(s_dst, s_src, values)
            return {
                "mode": self._depower_mode,
                "value": float(values.mean()),
                "profile": [float(v) for v in resampled],
            }

        try:
            value = float(np.asarray(sim_parameters["input_depower"]).ravel()[0])
        except (KeyError, TypeError, ValueError, IndexError):
            return None
        return {"mode": self._depower_mode, "value": value, "profile": None}

    @staticmethod
    def _apply_min_turn_radius(
        sim_parameters: Dict[str, Any], min_turn_radius: Optional[float]
    ) -> None:
        """Set/clear the optimizer's minimum-turn-radius constraint.

        ``None`` leaves the current setting (request field omitted); a positive
        value writes ``sim_parameters["min_turn_radius"]`` [m], read by
        ``PhaseParameterized.opti_phase``; ``0`` removes it. A cycle-config
        value stays in force until a request changes it.
        """
        if min_turn_radius is None:
            return
        value = float(min_turn_radius)
        if value < 0.0:
            raise ValueError(f"min_turn_radius must be >= 0, got {value}")
        if value > 0.0:
            sim_parameters["min_turn_radius"] = value
        else:
            sim_parameters.pop("min_turn_radius", None)

    def min_turn_radius(self) -> Optional[float]:
        """The minimum turn radius [m] the optimizer enforces (None = off)."""
        if self.phase is None:
            return None
        value = self.phase.pattern_config.get("sim_parameters", {}).get(
            "min_turn_radius"
        )
        return float(value) if value else None

    @staticmethod
    def _apply_winch_params(
        radial_parameters: Dict[str, Any],
        sim_parameters: Dict[str, Any],
        winch: Dict[str, Any],
    ) -> None:
        """Map a client winch law v_set = k_v*sqrt(F) onto the optimizer.

        Inverted, the law is F = (1/k_v^2) * v^2 — exactly AWETrim's
        quadratic force model T = slope*(v_r - offset)^2 with offset 0, so
        the optimizer assumes the same radial behavior the client's winch
        controller produces. f_min/f_max become the (softly clamped) force
        limits. Past f_max the controller holds the force while the reel
        speed keeps rising up to the winch's power limit, so the speed cap
        is an input: ``v_max`` (or ``p_max`` / f_max) bounds the speed_radial
        NLP variable; without either the optimizer's default bound stays.
        """
        if winch["mode"] != "reelout":
            raise ValueError(
                f"winch mode {winch['mode']!r} is not supported yet; "
                "this server optimizes the reelout phase only"
            )
        k_v = float(winch["k_v"])
        radial_parameters.update(
            {
                "reeling_strategy": "force",
                "force_model": "quadratic",
                "slope_winch_ro": 1.0 / (k_v * k_v),
                "offset_winch_ro": 0.0,
                "min_tether_force": float(winch["f_min"]),
                "max_tether_force": float(winch["f_max"]),
                "softplus": True,
                "softminus": True,
            }
        )
        radial_parameters.setdefault("softplus_beta", 1e-3)
        radial_parameters.setdefault("softminus_beta", 1e-3)
        v_max = winch.get("v_max")
        if v_max is None and winch.get("p_max") is not None:
            v_max = float(winch["p_max"]) / float(winch["f_max"])
        override = dict(sim_parameters.get("opti_limits_override") or {})
        if v_max is not None:
            override["speed_radial"] = [
                DEFAULT_OPTI_LIMITS["speed_radial"][0],
                float(v_max),
            ]
        else:
            override.pop("speed_radial", None)
        if override:
            sim_parameters["opti_limits_override"] = override
        else:
            sim_parameters.pop("opti_limits_override", None)

    @staticmethod
    def _fit_trajectory_into_path(
        path_parameters: Dict[str, Any], trajectory: Dict[str, Any]
    ) -> None:
        """Fit the pattern B-spline control points to a client flight path.

        Input is azimuth/elevation in DEGREES; a duplicated closing point is
        dropped before the periodic fit.
        """
        azimuth = np.radians(np.asarray(trajectory["azimuth"], dtype=float))
        elevation = np.radians(np.asarray(trajectory["elevation"], dtype=float))
        if np.isclose(azimuth[0], azimuth[-1]) and np.isclose(
            elevation[0], elevation[-1]
        ):
            azimuth = azimuth[:-1]
            elevation = elevation[:-1]

        s_init = float(path_parameters["s_init"])
        s_final = float(path_parameters["s_final"])
        s_samples = np.linspace(s_init, s_final, azimuth.size, endpoint=False)
        _, C_phi, C_beta = fit_bspline_pattern_to_trajectory(
            spline_type="periodic",
            M=int(path_parameters["M"]),
            s_init=s_init,
            s_final=s_final,
            az_target=azimuth,
            el_target=elevation,
            s_samples=s_samples,
            downloops=bool(path_parameters.get("downloops", True)),
        )
        path_parameters["C_phi"] = np.asarray(C_phi).ravel().tolist()
        path_parameters["C_beta"] = np.asarray(C_beta).ravel().tolist()

    def trajectory_degrees(self, n_points: Optional[int] = None) -> Dict[str, List[float]]:
        """Sample the current pattern as a closed azimuth/elevation table [deg].

        The last point equals the first (periodic pattern evaluated over one
        full period). After a successful solve this is the OPTIMIZED path,
        right after /init it is the fitted starting path.
        """
        if self.phase is None:
            raise SessionNotInitializedError("call /init first")
        pattern_config = self.phase.pattern_config
        path_parameters = pattern_config["path_parameters"]
        pattern = create_pattern_from_dict(
            pattern_config["pattern_type"], path_parameters
        )
        n = int(n_points or self._trajectory_points or self._n_points or 100)
        s = np.linspace(
            float(path_parameters["s_init"]),
            float(path_parameters["s_final"]),
            n,
            endpoint=True,  # periodic spline wraps -> last == first
        )
        r0 = float(path_parameters["r0"])
        azimuth = np.degrees(np.asarray(pattern.azimuth(r0, s)).ravel())
        elevation = np.degrees(np.asarray(pattern.elevation(r0, s)).ravel())
        return {
            "azimuth": azimuth.tolist(),
            "elevation": elevation.tolist(),
        }

    # ------------------------------------------------------------------
    # /step
    # ------------------------------------------------------------------
    def _prepare_step(
        self,
        distance_radial: Optional[float],
        winch_params: Optional[Dict[str, Any]],
        trajectory: Optional[Dict[str, Any]],
        inflow_conditions: Optional[Dict[str, Any]] = None,
        depower: Optional[Dict[str, Any]] = None,
        min_turn_radius: Optional[float] = None,
    ) -> int:
        """Validate, apply the updated inputs and transition to SOLVING."""
        with self._lock:
            if self.state == SessionState.UNINITIALIZED:
                raise SessionNotInitializedError("call /init before /step")
            if self.state == SessionState.SOLVING:
                raise SessionBusyError("a solve is already running")

            pattern_config = self.phase.pattern_config
            sim_parameters = pattern_config.setdefault("sim_parameters", {})
            if inflow_conditions is not None:
                wind_spec = self._resolve_wind_spec(inflow_conditions)
                self.system_model.wind = self._build_wind(wind_spec)
                sim_parameters["expand_nlp"] = (
                    wind_spec["model_type"] != "tabulated"
                )
                self._wind_spec = wind_spec
                self._inflow = dict(inflow_conditions)

            if winch_params is not None:
                self._apply_winch_params(
                    pattern_config["radial_parameters"], sim_parameters,
                    winch_params,
                )
                self._winch_params = dict(winch_params)

            if trajectory is not None:
                self._fit_trajectory_into_path(
                    pattern_config["path_parameters"], trajectory
                )
                self._trajectory_points = len(trajectory["azimuth"])

            if depower is not None:
                self._depower_mode = self._apply_depower(
                    sim_parameters, self._optimization_params, depower
                )

            self._apply_min_turn_radius(sim_parameters, min_turn_radius)

            if distance_radial is not None:
                path_parameters = pattern_config["path_parameters"]
                old_r0 = float(path_parameters["r0"])
                delta = float(distance_radial) - old_r0
                path_parameters["r0"] = float(distance_radial)
                self.phase.start_state["distance_radial"] = float(distance_radial)
                # Keep the node-wise warm start consistent with the new sphere.
                warm = getattr(self.phase, "_warm_start_trajectory", None)
                if warm and "distance_radial" in warm:
                    warm["distance_radial"] = (
                        np.asarray(warm["distance_radial"], dtype=float) + delta
                    )

            self.step_count += 1
            step_index = self.step_count
            self.state = SessionState.SOLVING
            self.last_step_started = _utcnow()
            self.last_step_finished = None
            return step_index

    def step(
        self,
        distance_radial: Optional[float] = None,
        max_iter: Optional[int] = None,
        winch_params: Optional[Dict[str, Any]] = None,
        trajectory: Optional[Dict[str, Any]] = None,
        inflow_conditions: Optional[Dict[str, Any]] = None,
        depower: Optional[Dict[str, Any]] = None,
        min_turn_radius: Optional[float] = None,
    ) -> int:
        """Launch one warm-started re-optimization in a background thread."""
        step_index = self._prepare_step(
            distance_radial, winch_params, trajectory, inflow_conditions,
            depower, min_turn_radius,
        )
        self._thread = threading.Thread(
            target=self._solve, args=(step_index, max_iter), daemon=True
        )
        self._thread.start()
        return step_index

    def step_blocking(
        self,
        distance_radial: Optional[float] = None,
        max_iter: Optional[int] = None,
        winch_params: Optional[Dict[str, Any]] = None,
        trajectory: Optional[Dict[str, Any]] = None,
        inflow_conditions: Optional[Dict[str, Any]] = None,
        depower: Optional[Dict[str, Any]] = None,
        min_turn_radius: Optional[float] = None,
    ) -> Dict[str, Any]:
        """One warm-started re-optimization, solved in the calling thread.

        Returns a StepParams-shaped dict: the (possibly updated) winch
        parameters, the OPTIMIZED trajectory in degrees, the depower the path
        was optimized for and the minimum turn radius it was optimized under.
        Raises SolveFailedError if the solve does not converge (the last good
        trajectory remains available via :meth:`trajectory`).
        """
        step_index = self._prepare_step(
            distance_radial, winch_params, trajectory, inflow_conditions,
            depower, min_turn_radius,
        )
        self._solve(step_index, max_iter)
        with self._lock:
            if self.state == SessionState.FAILED:
                raise SolveFailedError(
                    self.last_error or "optimization failed"
                )
            metrics = dict(self.last_record.metrics)
            state = self.state.value
        return {
            "length": self._length(),
            "winch_params": dict(self._winch_params)
            if self._winch_params
            else None,
            "trajectory": self.trajectory_degrees(),
            "depower": self.depower_state(),
            "min_turn_radius": self.min_turn_radius(),
            "state": state,
            "step_index": step_index,
            "metrics": metrics,
        }

    def _run_opti(self, warm_start: bool, max_iter: Optional[int]) -> Any:
        return self.phase.run_simulation_opti(
            optimization_params=self._optimization_params,
            target=self._target,
            warm_start=warm_start,
            max_iter=max_iter,
        )

    def _first_solve_staged(self, max_iter: Optional[int]) -> Any:
        """Cold start with a minimum-turn-radius constraint: stage it.

        A cold constrained solve lands on the best branch less often than an
        unconstrained one (measured on the LEI-V3 reference: 11/18 lengths vs
        16/18; the rest go to a "binding" branch with the loops exactly at
        R_min and a few percent less power, or into low-power basins), and
        every later warm ``/step`` follows whatever branch the first solve
        found. So: (1) solve cold WITHOUT the constraint -- its optimum usually
        already respects R_min; (2) re-solve warm WITH it, which then only
        verifies (or nudges) that optimum; (3) if either stage fails, solve
        cold with the constraint from the original starting point exactly as
        an unstaged solve would. Same contract, better odds.
        """
        pattern_config = self.phase.pattern_config
        sim_parameters = pattern_config.setdefault("sim_parameters", {})
        min_turn_radius = sim_parameters.get("min_turn_radius")
        if not min_turn_radius or self.successful_steps > 0:
            return self._run_opti(warm_start=self.successful_steps > 0, max_iter=max_iter)

        config_0 = copy.deepcopy(pattern_config)
        start_state_0 = copy.deepcopy(self.phase.start_state)

        sim_parameters.pop("min_turn_radius")
        try:
            result = self._run_opti(warm_start=False, max_iter=max_iter)
        finally:
            sim_parameters["min_turn_radius"] = min_turn_radius
        if result is not None:
            result = self._run_opti(warm_start=True, max_iter=max_iter)
            if result is not None:
                return result

        # Fallback: the plain constrained cold solve, from the untouched start.
        phase = Phase(
            system_model=self.system_model,
            pattern_config=config_0,
            start_state=start_state_0,
        )
        phase._plot_failed_iterate = lambda *args, **kwargs: None
        self.phase = phase
        return self._run_opti(warm_start=False, max_iter=max_iter)

    def _solve(self, step_index: int, max_iter: Optional[int]) -> None:
        try:
            result = self._first_solve_staged(max_iter)
            if result is None:
                self._finish_failed("optimization did not converge (IPOPT failure)")
                return
            record = self._build_record(step_index, result)
        except Exception as exc:  # worker thread must never die silently
            self._finish_failed(f"{type(exc).__name__}: {exc}")
            return

        with self._lock:
            self.last_record = record
            self.successful_steps += 1
            self.state = SessionState.CONVERGED
            self.last_error = None
            self.last_step_finished = _utcnow()

    def _finish_failed(self, message: str) -> None:
        with self._lock:
            self.state = SessionState.FAILED
            self.last_error = message
            self.last_step_finished = _utcnow()

    def _build_record(self, step_index: int, result: Any) -> SolveRecord:
        arrays = {
            key: np.asarray(value, dtype=float).ravel().copy()
            for key, value in result.optimized_trajectory.items()
        }

        pattern_config = self.phase.pattern_config
        path_parameters = pattern_config["path_parameters"]
        pattern = create_pattern_from_dict(
            pattern_config["pattern_type"], path_parameters
        )
        guidance = _compute_guidance_columns(pattern, arrays)

        table = dict(guidance)
        table["distance_radial"] = arrays["distance_radial"]
        table["speed_radial"] = arrays["speed_radial"]
        for key in ("s", "s_dot", "tension_tether_ground", "input_steering",
                    "input_depower", "turn_radius"):
            if key in arrays:
                table[key] = arrays[key]

        # Depower is a per-node variable only in profile mode; in fixed and
        # scalar-optimize mode the column is the (constant) setting the path
        # was optimized for, so clients always find it in the same place.
        if "input_depower" not in table:
            depower = self.depower_state(n_points=len(table["s"]))
            if depower is not None:
                table["input_depower"] = np.full(
                    len(table["s"]), depower["value"], dtype=float
                )

        spline = {
            "spline_type": "periodic"
            if pattern_config["pattern_type"] == "spline_periodic"
            else pattern_config["pattern_type"],
            "M": int(path_parameters["M"]),
            "C_phi": [float(c) for c in np.ravel(path_parameters["C_phi"])],
            "C_beta": [float(c) for c in np.ravel(path_parameters["C_beta"])],
            "r0": float(path_parameters["r0"]),
            "s_init": float(path_parameters["s_init"]),
            "s_final": float(path_parameters["s_final"]),
            "downloops": bool(path_parameters.get("downloops", True)),
        }

        metrics = self._extract_metrics(result)

        sim_parameters = pattern_config.get("sim_parameters", {})
        optimized_parameters = {
            "C_phi": spline["C_phi"],
            "C_beta": spline["C_beta"],
            "r0": spline["r0"],
        }
        if "input_depower" in sim_parameters:
            value = sim_parameters["input_depower"]
            try:
                optimized_parameters["input_depower"] = float(value)
            except (TypeError, ValueError):
                pass

        return SolveRecord(
            step_index=step_index,
            table=table,
            spline=spline,
            metrics=metrics,
            optimized_parameters=optimized_parameters,
            started=self.last_step_started or _utcnow(),
        )

    @staticmethod
    def _extract_metrics(result: Any) -> Dict[str, float]:
        def _numeric(expr: Any) -> float:
            if isinstance(expr, (int, float)):
                return float(expr)
            return float(result.solution.value(expr))

        energy = _numeric(result.energy_objective)
        total_time = _numeric(result.total_time)
        avg_power = energy / total_time if total_time else float("nan")
        # Tightest turn radius of the returned path, from the NLP's own
        # sub-sampled curvature expression (None if not evaluated).
        turn_radius_min = getattr(result, "turn_radius_min", None)
        if turn_radius_min is not None and not np.isfinite(turn_radius_min):
            turn_radius_min = None
        return {
            "energy_J": energy,
            "total_time_s": total_time,
            "avg_power_W": avg_power,
            "turn_radius_min_m": turn_radius_min,
        }

    # ------------------------------------------------------------------
    # /status, /trajectory, /reset
    # ------------------------------------------------------------------
    def status(self) -> Dict[str, Any]:
        with self._lock:
            record = self.last_record
            return {
                "state": self.state.value,
                "step_count": self.step_count,
                "successful_steps": self.successful_steps,
                "last_error": self.last_error,
                "metrics": dict(record.metrics) if record else None,
                "inflow_conditions": dict(self._inflow) if self._inflow else None,
                "wind": dict(self._wind_spec) if self._wind_spec else None,
                "n_points": self._n_points,
                "last_step_started": self.last_step_started,
                "last_step_finished": self.last_step_finished,
                "session_id": self.session_id,
            }

    def init_reply(self) -> Dict[str, Any]:
        """InitParams-shaped reply: session config + fitted starting path."""
        sim_parameters = self.phase.pattern_config.get("sim_parameters", {})
        return {
            "name": self._name,
            "length": self._length(),
            "winch_params": dict(self._winch_params)
            if self._winch_params
            else None,
            "inflow_conditions": dict(self._inflow) if self._inflow else None,
            "trajectory": self.trajectory_degrees(),
            "input_depower": sim_parameters.get("input_depower"),
            "reg_weight": sim_parameters.get("reg_weight"),
            "detect_simple_bounds": sim_parameters.get("detect_simple_bounds"),
            "depower": self.depower_state(),
            "min_turn_radius": self.min_turn_radius(),
            "state": self.state.value,
            "n_points": self._n_points,
            "session_id": self.session_id,
        }

    def _length(self) -> float:
        """Tether length the pattern is currently anchored to [m]."""
        return float(self.phase.pattern_config["path_parameters"]["r0"])

    def trajectory(self, resimulate: bool = False) -> Dict[str, Any]:
        with self._lock:
            record = self.last_record
            solving = self.state == SessionState.SOLVING
        if record is None:
            raise NoTrajectoryError("no successful optimization yet")

        response: Dict[str, Any] = {
            "step_index": record.step_index,
            "n_points": len(record.table["s"]),
            "table": {
                key: _as_optional_float_list(values)
                for key, values in record.table.items()
            },
            "spline": dict(record.spline),
            "metrics": dict(record.metrics),
            "optimized_parameters": dict(record.optimized_parameters),
        }

        if resimulate:
            if solving:
                raise SessionBusyError(
                    "cannot re-simulate while a solve is running"
                )
            phase_ts, _ = self.phase.run_simulation(
                run_plots=False, phase_sim=True
            )
            timeseries: Dict[str, List[Optional[float]]] = {}
            for name in RESIM_VARIABLES:
                try:
                    timeseries[name] = _as_optional_float_list(
                        phase_ts.return_variable(name)
                    )
                except Exception:
                    continue  # variable not available for this model
            response["timeseries"] = timeseries
            metrics = phase_ts.energy_metrics()
            response["energy_metrics"] = {
                key: (float(value) if np.isscalar(value) else value)
                for key, value in dict(metrics).items()
            }

        return response

    def reset(self) -> None:
        with self._lock:
            if self.state == SessionState.SOLVING:
                raise SessionBusyError("cannot reset while a solve is running")
            self._clear()
