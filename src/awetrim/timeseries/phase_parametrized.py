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

from matplotlib import pyplot as plt
from awetrim.timeseries.timeseries import TimeSeries
from awetrim import SystemModel
from awetrim.kinematics.parametrized_patterns import (
    PeriodicBSpline,
    create_pattern_from_dict,
)
from awetrim.kinematics.Kinematics import ParametrizedKinematics
import casadi as ca
import numpy as np
from awetrim.utils.defaults import (
    DEFAULT_BOUNDS,
    DEFAULT_PATTERN_CONFIG,
    DEFAULT_OPTI_LIMITS,
    DEFAULT_WINCH_CONFIG,
)
import copy
from awetrim.system.tether import RigidLinkTether
from awetrim.system.williams_tether import WilliamsTether
from awetrim.utils.reference_frames import (
    transformation_C_from_Wind,
    transformation_Wind_from_C,
)
from awetrim import State
from awetrim.system.kite import Kite
from awetrim.system.winch import Winch
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class NodeControl:
    """A per-node control input in the reel-out NLP (e.g. steering, depower).

    Centralizes the per-control boilerplate that ``opti_phase`` would otherwise
    copy-paste for every control: the node-to-node slew-rate limit and the
    control-smoothness regularization term. The decision variable itself lives
    in ``opti_vars[name]`` and its magnitude bounds come from
    ``DEFAULT_OPTI_LIMITS[name]`` via the generic vector-bound loop, so only the
    rate limit is carried here.

    The residual-function wiring stays control-specific and is NOT abstracted
    here: ``input_steering`` is an explicit argument of the residual / aoa /
    tension Functions, whereas ``input_depower`` enters as the trailing
    parameter symbol (``node_syms``). Adding a new per-node control therefore
    means: create ``opti_vars[name]``, add a ``NodeControl`` here (-> rate limit
    + smoothness reg), add its bounds to ``DEFAULT_OPTI_LIMITS`` (-> magnitude
    bounds), and wire it into the residual.
    """

    name: str
    rate_limit: tuple  # (lower, upper) bound on d(u)/dt between adjacent nodes


def _williams_pre_solve_dump(qs_solver, z0, p, lbx, ubx, *, quasi_steady):
    """Print the Williams QS-NLP inputs at node 0, plus the residual evaluated
    at the initial guess. Use when ``sim_parameters.debug_solver = True`` to
    see whether the seed is well-defined and which residual component is the
    largest before IPOPT starts iterating.

    Note: the tension decision in the NLP is ``tension_kite_scaled = T /
    WILLIAMS_TENSION_SCALE``. The dump prints the raw (unscaled) value next
    to ``tension_kite`` for human readability.
    """
    z_arr = np.asarray(ca.DM(z0)).reshape(-1)
    p_arr = np.asarray(ca.DM(p)).reshape(-1)
    lbx_arr = (
        np.asarray(ca.DM(lbx)).reshape(-1)
        if not isinstance(lbx, list)
        else np.asarray(lbx)
    )
    ubx_arr = (
        np.asarray(ca.DM(ubx)).reshape(-1)
        if not isinstance(ubx, list)
        else np.asarray(ubx)
    )
    if quasi_steady:
        z_names = [
            "input_steering",
            "s_dot",
            "speed_radial",
            "tension_kite_scaled",
            "elev_last",
            "az_last",
            "tether_length",
        ]
        p_names = ["s", "distance_radial", "input_depower", "speed_wind_ref"]
        g_names = [
            "force_x",
            "force_y",
            "force_z",
            "ground_x",
            "ground_y",
            "ground_z",
            "winch",
        ]
    else:
        z_names = [f"z[{i}]" for i in range(z_arr.size)]
        p_names = [f"p[{i}]" for i in range(p_arr.size)]
        g_names = [f"g[{i}]" for i in range(7)]
    print("[Williams pre-solve dump @ node 0]")
    print("  initial guess (z0):")
    for name, lo, val, hi in zip(z_names, lbx_arr, z_arr, ubx_arr):
        flag = "  "
        if val < lo or val > hi:
            flag = "!!"
        print(f"    {flag} {name:>16s}: {val:>+13.4e}  [{lo:>+10.3e}, {hi:>+10.3e}]")
    print("  parameters (p):")
    for name, val in zip(p_names, p_arr):
        print(f"     {name:>16s}: {val:>+13.4e}")

    # Evaluate the residual at z0 by calling the NLP with zero iterations.
    # Cheap trick: use the underlying NLP's g via ``qs_solver`` with a tiny
    # max_iter; alternatively pull the g function from solver internals.
    try:
        nlp_g = qs_solver.get_function("nlp_g")
        g_val = np.asarray(nlp_g(z_arr, p_arr)).reshape(-1)
        print("  residual at initial guess (g(z0, p)):")
        for name, val in zip(g_names, g_val):
            print(f"     {name:>16s}: {val:>+13.4e}")
        print(f"     {'||g||':>16s}: {np.linalg.norm(g_val):>+13.4e}")
    except Exception as exc:  # pragma: no cover — diagnostic only
        print(f"  (could not evaluate residual at z0: {exc})")


def _eval_path_kinematic(expr, s_value, state_obj, default=0.0):
    """Evaluate a parametrized kinematic expression at a numeric ``s``.

    Used to seed Williams' direction initial guesses (``elev_last``,
    ``az_last``) from the path geometry when the caller did not supply
    ``angle_elevation`` / ``angle_azimuth`` in the start state. Falls back
    to ``default`` if any required free symbol can't be filled from
    ``state_obj``.
    """
    free_syms = ca.symvar(expr)
    if not free_syms:
        return float(expr) if not isinstance(expr, (ca.SX, ca.MX)) else default
    state_dict = state_obj.to_dict()
    values = []
    for sym in free_syms:
        name = sym.name()
        if name == "s":
            values.append(s_value)
        elif name in state_dict and state_dict[name] is not None:
            values.append(state_dict[name])
        else:
            return float(default)
    fun = ca.Function("path_kinematic_eval", free_syms, [expr])
    try:
        return float(fun(*values))
    except (RuntimeError, TypeError):
        return float(default)



def _expand_node_function(fun):
    """SX-expand a per-node CasADi Function used N times in the NLP graph.

    The NLP calls the same trim-residual / AoA / curvature Function once per
    node. Expanding each Function ONCE (MX -> SX) gives every call SX-fast
    evaluation and derivatives while the outer graph stays MX -- the solver
    then never has to SX-expand the whole N-node graph (``expand=True`` in
    ``Phase.run_opti``), whose construction time and memory grow with N (the
    inlined copies) and whose Jacobian sparsity pass scales ~N^2. Falls back
    to the MX Function when the graph holds MX-only nodes (interpolant wind,
    custom_spline winch, externals).
    """
    try:
        return fun.expand()
    except Exception as exc:  # MX-only nodes inside: keep the MX Function
        first = str(exc).strip().splitlines()[0] if str(exc).strip() else exc
        print(f"[nlp] {fun.name()}: SX expansion unsupported ({first}); keeping MX")
        return fun


class PhaseParameterized(TimeSeries):
    def __init__(
        self,
        kite_model: SystemModel,
        quasi_steady: bool = False,
        pattern_config: dict = DEFAULT_PATTERN_CONFIG,
        pattern_config_opti: dict = None,
        sharpness_beta: float = 1e-4,
        tension_min: float = 0.0,
        tension_max: float = 1e5,
    ):
        """
        Args:

        """

        super().__init__(
            kite_model=kite_model,
        )
        self.pattern_config = pattern_config
        if not pattern_config_opti:
            self.pattern_config_opti = copy.deepcopy(pattern_config)
        else:
            self.pattern_config_opti = pattern_config_opti
        self.quasi_steady = quasi_steady

        self.kite_model = kite_model
        self.target_drag_coefficient = None
        self.target_lift_coefficient = None
        self._williams_tension_ground_function = None
        self.s = ca.MX.sym("s")
        self.t = ca.MX.sym("t")
        self.s_dot = ca.MX.sym("s_dot")
        self.s_ddot = ca.MX.sym("s_ddot")
        self.sharpness_beta = sharpness_beta
        self.tension_min = tension_min
        self.tension_max = tension_max
        self.winch_model = Winch(
            pattern_config=self.pattern_config["radial_parameters"]
        )

        pattern = create_pattern_from_dict(
            self.pattern_config["pattern_type"], self.pattern_config["path_parameters"]
        )
        km_copy = self.substitute_parametrized_kinematics(pattern)
        self.km_param = km_copy
        # self.find_optimal_angle_pitch_tether()

    def run_simulation(self, start_state, allow_failure=True, return_states=False):

        # print("Starting state:", start_state)
        pattern = create_pattern_from_dict(
            self.pattern_config["pattern_type"], self.pattern_config["path_parameters"]
        )
        km_copy = self.substitute_parametrized_kinematics(pattern=pattern)
        self.states = []
        km_copy.reset_solver()
        self.km_param = km_copy

        if self.quasi_steady:
            unknown_vars = ["length_tether", "input_steering", "s_dot", "speed_radial"]
        else:
            unknown_vars = ["length_tether", "input_steering", "s_ddot", "speed_radial"]

        if km_copy.is_tether_rigid:
            unknown_vars[0] = "tension_tether_ground"
        use_williams = isinstance(km_copy.tether, WilliamsTether)
        # Initialize state
        if isinstance(start_state, dict):
            state_obj = State(**start_state)
        else:
            state_obj = start_state

        # For dynamic runs, solve the first state with a quasi-steady residual to get a consistent starting point.
        if not self.quasi_steady:
            state_obj = self._quasi_steady_start_state(state_obj, km_copy)

        N = self.pattern_config["sim_parameters"]["n_points"]
        time_step = (
            self.pattern_config["sim_parameters"]["end_time"]
            / self.pattern_config["sim_parameters"]["n_points"]
        )
        # Capture numeric wind speed before residual_solver replaces it with a symbolic
        _wind_raw_euler = km_copy.wind.speed_wind_ref_value
        try:
            wind_ref_euler = (
                float(_wind_raw_euler) if _wind_raw_euler is not None else 0.0
            )
        except (TypeError, RuntimeError):
            wind_ref_euler = 0.0

        intg = self.integrator(time_step=time_step, kite_model=km_copy)
        qs_solver = self.residual_solver(km_copy)
        use_williams = isinstance(km_copy.tether, WilliamsTether)

        sim_params_euler = self.pattern_config.get("sim_parameters", {})
        depower_val = (
            state_obj.input_depower
            if state_obj.input_depower is not None
            else sim_params_euler.get("input_depower", 0.0)
        )

        # print("New state:", qs_solver)
        if self.quasi_steady:
            x0 = ca.vertcat(
                state_obj.tension_tether_ground,
                state_obj.input_steering,
                state_obj.s_dot,
                state_obj.speed_radial,
            )
            p = ca.vertcat(
                state_obj.s, state_obj.distance_radial, depower_val, wind_ref_euler
            )
            lbx, ubx, lbg, ubg = km_copy.get_boundaries(state_obj, unknown_vars)
            if use_williams:
                elev_guess = (
                    state_obj.angle_elevation
                    if state_obj.angle_elevation is not None
                    else 0.2
                )
                az_guess = (
                    state_obj.angle_azimuth
                    if state_obj.angle_azimuth is not None
                    else 0.0
                )
                length_guess = state_obj.distance_radial
                x0 = ca.vertcat(x0, elev_guess, az_guess, length_guess)
                r0 = state_obj.distance_radial
                lbx = list(lbx) + [-np.pi / 2 + 1e-3, -2.0 * np.pi, 0.5 * r0]
                ubx = list(ubx) + [np.pi / 2 - 1e-3, 2.0 * np.pi, 3.0 * r0]
                lbg = list(lbg) + [0.0, 0.0, 0.0]
                ubg = list(ubg) + [0.0, 0.0, 0.0]
            sol = qs_solver(x0=x0, p=p, lbg=lbg, ubg=ubg, lbx=lbx, ubx=ubx)
            x0 = p
            z0 = sol["x"]
        else:
            x0 = ca.vertcat(
                state_obj.tension_tether_ground,
                state_obj.input_steering,
                state_obj.s_dot,
                state_obj.speed_radial,
            )
            p = ca.vertcat(
                state_obj.s,
                state_obj.s_dot,
                state_obj.distance_radial,
                depower_val,
                wind_ref_euler,
            )
            lbx, ubx, lbg, ubg = km_copy.get_boundaries(state_obj, unknown_vars)
            if use_williams:
                elev_guess = (
                    state_obj.angle_elevation
                    if state_obj.angle_elevation is not None
                    else 0.2
                )
                az_guess = (
                    state_obj.angle_azimuth
                    if state_obj.angle_azimuth is not None
                    else 0.0
                )
                length_guess = state_obj.distance_radial
                x0 = ca.vertcat(x0, elev_guess, az_guess, length_guess)
                r0 = state_obj.distance_radial
                lbx = list(lbx) + [-np.pi / 2 + 1e-3, -2.0 * np.pi, 0.5 * r0]
                ubx = list(ubx) + [np.pi / 2 - 1e-3, 2.0 * np.pi, 3.0 * r0]
                lbg = list(lbg) + [0.0, 0.0, 0.0]
                ubg = list(ubg) + [0.0, 0.0, 0.0]
            sol = qs_solver(x0=x0, p=p, lbg=lbg, ubg=ubg, lbx=lbx, ubx=ubx)
            x0 = p
            z0 = sol["x"]
        # self.states.append(new_state.to_dict())
        t = self.pattern_config["sim_parameters"]["start_time"]
        for i in range(N):
            # print(f"Time: {t}, State: {x0}, Inputs: {z0}")
            try:
                sol = intg(
                    x0=x0,
                    p=t,
                    z0=z0,
                )
            except Exception as e:
                print(f"Error occurred: {e}")
                if not allow_failure:
                    raise
                break
            x0 = sol["xf"]
            z0 = sol["zf"]
            if self.quasi_steady:
                new_state = State(
                    t=t,
                    s=x0[0],
                    input_steering=float(z0[1]),
                    tension_tether_ground=float(z0[0]),
                    s_dot=float(z0[2]),
                    distance_radial=float(x0[1]),
                    speed_radial=float(z0[3]),
                )
            else:
                new_state = State(
                    t=t,
                    s=x0[0],
                    s_dot=float(x0[1]),
                    input_steering=float(z0[1]),
                    tension_tether_ground=float(z0[0]),
                    s_ddot=float(z0[2]),
                    distance_radial=float(x0[2]),
                    speed_radial=float(z0[3]),
                )
            t += time_step
            self.states.append(new_state.to_dict())

    def run_simulation_phase(
        self,
        start_state,
        allow_failure=True,
        return_states=True,
    ):
        """
        March along an s-grid. At each grid point:
        - solve residuals for unknowns (z)
        - record state at current (t, s_i)
        - if not last grid point, compute dt from ds, v, a and advance x, t.

        Conventions:
        QS   : a_s = 0  -> ds = v_s * dt
        Dyn  : ds = v_s * dt + 0.5 * a_s * dt^2  (stable quadratic root used)
        """

        # --- setup / housekeeping
        self.kite_model.reset_solver()
        pattern = create_pattern_from_dict(
            self.pattern_config["pattern_type"], self.pattern_config["path_parameters"]
        )
        # initial state object
        state_obj = (
            State(**start_state) if isinstance(start_state, dict) else start_state
        )
        # For dynamic runs, align the first state via quasi-steady residual solve
        if not self.quasi_steady:
            km_copy = self.substitute_parametrized_kinematics(
                pattern, quasi_steady=True
            )
            state_obj = self._quasi_steady_start_state(state_obj, km_copy)
        km_copy = self.substitute_parametrized_kinematics(pattern)
        self.km_param = km_copy
        self.states = []
        use_williams = isinstance(km_copy.tether, WilliamsTether)

        # unknowns to solve at each s-node
        if self.quasi_steady:
            unknown_vars = ["length_tether", "input_steering", "s_dot", "speed_radial"]
        else:
            unknown_vars = ["length_tether", "input_steering", "s_ddot", "speed_radial"]

        if km_copy.is_tether_rigid:
            unknown_vars[0] = "tension_tether_ground"

        # grid and solver
        N = int(self.pattern_config["sim_parameters"]["n_points"])
        s_grid = np.linspace(
            self.pattern_config["sim_parameters"]["start_angle"],
            self.pattern_config["sim_parameters"]["end_angle"],
            N + 1,
        )

        sim_params = self.pattern_config.get("sim_parameters", {})

        # A node that fails to trim truncates the march with a warning by
        # default (``allow_failure=True``). ``require_full_trajectory`` makes
        # truncation fatal instead: pipelines that seed an NLP from this
        # trajectory (the ``opti_phase`` warm start) are better off aborting,
        # since a constant-padded seed violates the propagation and closure
        # constraints far more than an honest failure.
        if bool(sim_params.get("require_full_trajectory", False)):
            allow_failure = False

        # Allow optional per-node depower profile; fallback to scalar
        u_dep_profile = sim_params.get("input_depower_profile")
        if u_dep_profile is not None:
            u_dep_profile = np.asarray(u_dep_profile, dtype=float).ravel()
            if u_dep_profile.size != N + 1:
                raise ValueError("input_depower_profile must have length n_points+1")
        else:
            u_dep_profile = np.full(N + 1, sim_params["input_depower"])

        # Optional per-node wind speed profile (m/s) aligned with s_grid
        wind_profile = sim_params.get("wind_speed_profile")
        if wind_profile is not None:
            wind_profile = np.asarray(wind_profile, dtype=float).ravel()
            if wind_profile.size != N + 1:
                raise ValueError("wind_speed_profile must have length n_points+1")
        else:
            base_wind = getattr(km_copy.wind, "speed_wind_ref_value", None)
            if base_wind is None:
                base_wind = 0.0
            wind_profile = np.full(N + 1, base_wind, dtype=float)

        speed_radial_profile = sim_params.get("speed_radial_profile")
        if speed_radial_profile is not None:
            speed_radial_profile = np.asarray(speed_radial_profile, dtype=float).ravel()
            if speed_radial_profile.size != N + 1:
                raise ValueError("speed_radial_profile must have length n_points+1")
        else:
            vr0 = state_obj.speed_radial if state_obj.speed_radial is not None else 0.0
            speed_radial_profile = np.full(N + 1, float(vr0), dtype=float)

        input_steering_guess_profile = sim_params.get("input_steering_guess_profile")
        if input_steering_guess_profile is not None:
            input_steering_guess_profile = np.asarray(
                input_steering_guess_profile, dtype=float
            ).ravel()
            if input_steering_guess_profile.size != N + 1:
                raise ValueError(
                    "input_steering_guess_profile must have length n_points+1"
                )

        s_dot_guess_profile = sim_params.get("s_dot_guess_profile")
        if s_dot_guess_profile is not None:
            s_dot_guess_profile = np.asarray(s_dot_guess_profile, dtype=float).ravel()
            if s_dot_guess_profile.size != N + 1:
                raise ValueError("s_dot_guess_profile must have length n_points+1")

        qs_solver = self.residual_solver(km_copy)

        # pack initial guesses / states
        s_dot_guess = state_obj.s_dot if state_obj.s_dot is not None else 2.0
        if use_williams and self.quasi_steady:
            s_dot_guess = max(s_dot_guess, 0.1)
        if self.quasi_steady:
            # z = [tension_tether_ground, input_steering, s_dot, speed_radial]
            z = ca.vertcat(
                state_obj.tension_tether_ground,
                state_obj.input_steering,
                s_dot_guess,
                state_obj.speed_radial,
            )
            # x = [s, distance_radial, input_depower, wind_speed_ref]
            x = ca.vertcat(
                s_grid[0],
                state_obj.distance_radial,
                u_dep_profile[0],
                wind_profile[0],
            )
        else:
            # z = [tension_tether_ground, input_steering, s_ddot, speed_radial]
            z = ca.vertcat(
                state_obj.tension_tether_ground,
                state_obj.input_steering,
                0.01,  # initial guess for s_ddot
                state_obj.speed_radial,
            )
            # x = [s, s_dot, distance_radial, input_depower, wind_speed_ref]
            x = ca.vertcat(
                s_grid[0],
                state_obj.s_dot,
                state_obj.distance_radial,
                u_dep_profile[0],
                wind_profile[0],
            )

        lbx, ubx, lbg, ubg = self.get_boundaries(state_obj, unknown_vars, km_copy)
        # Williams initial guesses + bounds: mirror the
        # ``tether.decision_initial_guess_for`` / ``decision_bounds_for`` seeds
        # used by ``SystemModel.solve_quasi_steady`` (see solve_single_state.py
        # for reference). Key choices:
        #   - ``tether_length`` starts slightly LONGER than the chord (1.02 x
        #     distance_radial) so the iteration begins on the feasibility
        #     side of the ground-residual constraint.
        #   - ``tension_tether_kite`` lower bound is 300 N (matches
        #     ``DEFAULT_BOUNDS``); 0 here lets IPOPT drive the kite-end
        #     tension to zero and the tether direction becomes degenerate.
        if use_williams:
            r0 = state_obj.distance_radial
            # Source Williams' four decision-symbol seeds from the same hook
            # SystemModel.solve_quasi_steady uses (see
            # WilliamsTether.decision_initial_guess_for). ``tension_tether_ground``
            # on the start state is a RIGID-tether quantity and is physically
            # different from Williams' ``tension_tether_kite``, so we don't
            # forward it here — bad seeds make IPOPT spiral on this stiff
            # geometric problem.
            tether_guess = km_copy.tether.decision_initial_guess_for(
                km_copy, state_obj.to_dict()
            )
            tension_kite_seed = float(tether_guess["tension_tether_kite"])
            length_guess = float(tether_guess["tether_length"])
            # Override elev/az with the actual path geometry at s = s_grid[0]
            # when the caller didn't supply them on the start state — the
            # hook's defaults assume a horizontal-ish kite which is wrong for
            # high-elevation flight phases.
            elev_guess = state_obj.angle_elevation
            if elev_guess is None:
                elev_guess = _eval_path_kinematic(
                    km_copy.angle_elevation,
                    s_grid[0],
                    state_obj,
                    default=float(tether_guess["elevation_last_element"]),
                )
            az_guess = state_obj.angle_azimuth
            if az_guess is None:
                az_guess = _eval_path_kinematic(
                    km_copy.angle_azimuth,
                    s_grid[0],
                    state_obj,
                    default=float(tether_guess["azimuth_last_element"]),
                )
            # Rigid-tether warm start: the Williams kite-end tension is far
            # easier to seed if we first solve the same QS problem with a
            # straight tether (no extra decisions, no ground-closure
            # constraint). Rigid converges in ~10 iterations from almost any
            # seed; we then plug its tension_tether_ground into Williams as
            # tension_tether_kite. Cheap (one extra solver build) and removes
            # the deep mismatch that was driving IPOPT into restoration.
            rigid_seeds = self._rigid_warm_start(
                pattern, state_obj, s_grid, u_dep_profile, wind_profile
            )
            if rigid_seeds is not None:
                tension_kite_seed = rigid_seeds["tension_tether_ground"]
                if state_obj.input_steering is None or state_obj.input_steering == 0:
                    state_obj.input_steering = rigid_seeds["input_steering"]
                # Honour rigid's converged s_dot exactly — the previous max(.,
                # 0.1) cap was throwing away rigid's low-regime value (0.027
                # for the user's flight phase), which forces speed_tangential
                # 4x too high in Williams's aero residual.
                s_dot_guess = max(rigid_seeds["s_dot"], 1e-3)
                if state_obj.speed_radial is None or state_obj.speed_radial == 0:
                    state_obj.speed_radial = rigid_seeds["speed_radial"]
                logger.info(
                    "Williams warm-start from rigid: T=%.0f N, input_steering=%.3f, "
                    "s_dot=%.3f, speed_radial=%.3f",
                    tension_kite_seed,
                    rigid_seeds["input_steering"],
                    rigid_seeds["s_dot"],
                    rigid_seeds["speed_radial"],
                )
        # The tension decision inside the NLP is scaled (see
        # PhaseParameterized.WILLIAMS_TENSION_SCALE) — divide the physical
        # seed and bounds before they go into z/lbx/ubx.
        tension_scale = self.WILLIAMS_TENSION_SCALE
        if use_williams and self.quasi_steady:
            z = ca.vertcat(
                state_obj.input_steering,
                s_dot_guess,
                state_obj.speed_radial,
                tension_kite_seed / tension_scale,
            )
            lbx = [
                DEFAULT_BOUNDS["input_steering"][0],
                # Allow Williams to follow rigid's low-regime s_dot (e.g. 0.027
                # in the user's flight phase). A 0.1 floor here blocks IPOPT
                # from reaching the converged kite state and is the likely
                # root cause of restoration-phase failures.
                max(DEFAULT_BOUNDS["s_dot"][0], 1e-3),
                DEFAULT_BOUNDS["speed_radial"][0],
                DEFAULT_BOUNDS["tension_tether_kite"][0] / tension_scale,
            ]
            ubx = [
                DEFAULT_BOUNDS["input_steering"][1],
                DEFAULT_BOUNDS["s_dot"][1],
                DEFAULT_BOUNDS["speed_radial"][1],
                # 1e6 N physical ceiling, scaled.
                1.0e6 / tension_scale,
            ]
            lbg = [0.0] * 7
            ubg = [0.0] * 7
        elif use_williams:
            # Dynamic Williams: same decision layout as QS but with s_ddot in
            # place of s_dot -> [input_steering, s_ddot, speed_radial,
            # tension_kite, elev_last, az_last, length].
            z = ca.vertcat(
                state_obj.input_steering,
                0.01,  # s_ddot guess
                state_obj.speed_radial,
                tension_kite_seed / tension_scale,
            )
            lbx = [
                DEFAULT_BOUNDS["input_steering"][0],
                DEFAULT_BOUNDS["s_ddot"][0],
                DEFAULT_BOUNDS["speed_radial"][0],
                DEFAULT_BOUNDS["tension_tether_kite"][0] / tension_scale,
            ]
            ubx = [
                DEFAULT_BOUNDS["input_steering"][1],
                DEFAULT_BOUNDS["s_ddot"][1],
                DEFAULT_BOUNDS["speed_radial"][1],
                1.0e6 / tension_scale,
            ]
            lbg = [0.0] * 7
            ubg = [0.0] * 7
        if use_williams:
            z = ca.vertcat(z, elev_guess, az_guess, length_guess)
            # Tether length is bounded to ``[r, r + 10]`` — physically the
            # unstrained tether is at least the chord (straight) and at most
            # ~10 m slack in the kite-to-ground configuration this script
            # explores. Squeezing this dimension removes a floppy degree of
            # freedom IPOPT was wandering through.
            # ``az_last`` is widened to ``[-2π, 2π]`` so IPOPT has buffer to
            # cross the ``±π`` wrap smoothly when the kite flies near the
            # wind-frame zenith (where the spherical azimuth wraps).
            lbx = list(lbx) + [-np.pi / 2 + 1e-3, -2.0 * np.pi, r0]
            ubx = list(ubx) + [np.pi / 2 - 1e-3, 2.0 * np.pi, r0 + 10.0]
        # lbg = ca.vertcat(lbg, DEFAULT_OPTI_LIMITS["speed_radial"][0])
        # ubg = ca.vertcat(ubg, DEFAULT_OPTI_LIMITS["speed_radial"][1])
        t = float(state_obj.t)

        # --- helper: stable Δt from ds, v, a  (ds = v*dt + 0.5*a*dt^2)
        def _dt_from_ds_v_a(ds_scalar, v_s, a_s):
            """
            Numerically stable positive root:
                dt = 2*ds / ( v + sqrt(v*v + 2*a*ds) )
            - uses CasADi ops so it works with DM/MX/SX.
            - if discriminant < 0: clip to 0 if allow_failure else raise.
            """
            disc = v_s * v_s + 2.0 * a_s * ds_scalar
            if allow_failure:
                disc = ca.fmax(
                    disc, 0.0
                )  # clip; produces the limiting solution if negative
            else:
                # optional hard check
                if isinstance(disc, (float, int)) and disc < 0:
                    raise ValueError(f"Negative discriminant: v^2+2*a*ds={disc}")
            denom = v_s + ca.sqrt(disc)
            # add tiny epsilon to avoid divide-by-zero when v≈0 and a→0
            return 2.0 * ds_scalar / (denom + 1e-12)

        # --- main loop
        debug_solver = bool(sim_params.get("debug_solver", False))

        def _is_finite_dm(value):
            try:
                return bool(np.all(np.isfinite(np.asarray(value, dtype=float))))
            except (TypeError, ValueError):
                return False

        def _clip_guess_to_bounds(z_guess):
            z_arr = np.asarray(ca.DM(z_guess), dtype=float).reshape(-1)
            lo = np.asarray(lbx, dtype=float).reshape(-1)
            hi = np.asarray(ubx, dtype=float).reshape(-1)
            if z_arr.size != lo.size:
                return z_guess
            eps = 1e-8
            midpoint = 0.5 * (lo + hi)
            z_arr = np.where(np.isfinite(z_arr), z_arr, midpoint)
            z_arr = np.minimum(np.maximum(z_arr, lo + eps), hi - eps)
            return ca.DM(z_arr)

        def _path_angle_guess(expr, s_value, default):
            if not np.isfinite(default):
                default = 0.0
            return _eval_path_kinematic(expr, s_value, state_obj, default=default)

        def _finite_or(value, default):
            return float(value) if np.isfinite(value) else float(default)

        def _perturb_list(key, base, *, multipliers=(), offsets=(), positive=False):
            """Ordered, de-duplicated seed values for one swept retry variable.

            Starts with ``base`` so the leading retries reproduce the original
            force/length-only sweep; the multipliers and additive offsets then
            spread the path speed (s_dot) and radial speed around it, since
            those move the residual just like the tether force does. A wholesale
            override is read from ``sim_params[key]`` when present.
            """
            explicit = sim_params.get(key)
            if explicit is not None:
                values = [float(v) for v in explicit]
            else:
                values = [float(base)]
                values += [float(base) * m for m in multipliers]
                values += [float(base) + o for o in offsets]
            if positive:
                values = [v for v in values if v > 0.0] or [max(float(base), 0.2)]
            seen, uniq = set(), []
            for v in values:
                k = round(v, 9)
                if k not in seen:
                    seen.add(k)
                    uniq.append(v)
            return uniq

        def _tiered_seeds(force_list, rate_list, radial_list, assemble):
            """Build seeds force-sweep first, then s_dot, then speed_radial.

            ``assemble(force, path_rate, speed_radial)`` returns the decision
            vector for the active layout. The tiers keep the cheap, most-likely
            recoveries (re-trying the force at the seeded speeds) ahead of the
            broader speed sweeps so the retry cap trims only the exotic combos.
            """
            base_rate, base_radial = rate_list[0], radial_list[0]
            nominal_force = force_list[0]
            seeds = [assemble(f, base_rate, base_radial) for f in force_list]
            for rate in rate_list[1:]:
                seeds += [assemble(f, rate, base_radial) for f in force_list]
            for radial in radial_list[1:]:
                seeds += [assemble(f, base_rate, radial) for f in force_list]
            for rate in rate_list[1:]:
                for radial in radial_list[1:]:
                    seeds.append(assemble(nominal_force, rate, radial))
            return seeds

        def _finalize_guesses(guesses):
            """Clip seeds to the box, drop duplicates, and cap the count."""
            max_retries = int(sim_params.get("solver_max_retries", 40))
            seen, clipped = set(), []
            for guess in guesses[:max_retries]:
                cg = _clip_guess_to_bounds(guess)
                key = tuple(np.round(np.asarray(ca.DM(cg), dtype=float).reshape(-1), 9))
                if key not in seen:
                    seen.add(key)
                    clipped.append(cg)
            return clipped

        def _fallback_z_guesses(z_guess, p_solver, node_index):
            """Conservative alternate seeds for a failed per-node residual solve."""
            guesses = [z_guess]
            z_arr = np.asarray(ca.DM(z_guess), dtype=float).reshape(-1)
            if z_arr.size == 0:
                return guesses

            r_idx = 1 if self.quasi_steady else 2
            r_current = float(p_solver[r_idx])
            s_current = float(p_solver[0])
            if not use_williams:
                if z_arr.size < 4:
                    return guesses
                steering_idx = 1
                speed_idx = 2
                radial_idx = 3
                steering = (
                    input_steering_guess_profile[node_index]
                    if input_steering_guess_profile is not None
                    else np.clip(
                        _finite_or(z_arr[steering_idx], 0.0),
                        DEFAULT_BOUNDS["input_steering"][0],
                        DEFAULT_BOUNDS["input_steering"][1],
                    )
                )
                if self.quasi_steady:
                    path_rate_or_accel = (
                        s_dot_guess_profile[node_index]
                        if s_dot_guess_profile is not None
                        else max(abs(_finite_or(z_arr[speed_idx], 0.2)), 0.2)
                    )
                else:
                    path_rate_or_accel = np.clip(
                        _finite_or(z_arr[speed_idx], 0.0),
                        DEFAULT_BOUNDS["s_ddot"][0],
                        DEFAULT_BOUNDS["s_ddot"][1],
                    )
                speed_radial = (
                    speed_radial_profile[node_index]
                    if speed_radial_profile is not None
                    else _finite_or(z_arr[radial_idx], 0.0)
                )
                first_var_candidates = sim_params.get(
                    "solver_retry_tension_ground",
                    [300.0, 2.0e3, 8.4e3, 2.0e4, 8.0e4],
                )
                if not km_copy.is_tether_rigid:
                    first_var_candidates = sim_params.get(
                        "solver_retry_length_tether",
                        [r_current, 1.02 * r_current, r_current + 5.0],
                    )
                if self.quasi_steady:
                    rate_candidates = _perturb_list(
                        "solver_retry_s_dot",
                        path_rate_or_accel,
                        multipliers=(0.5, 2.0),
                        positive=True,
                    )
                else:
                    rate_candidates = _perturb_list(
                        "solver_retry_s_ddot",
                        path_rate_or_accel,
                        offsets=(0.5, -0.5),
                    )
                radial_candidates = _perturb_list(
                    "solver_retry_speed_radial", speed_radial, offsets=(1.5, -1.5)
                )

                def _assemble(first_var, rate, radial):
                    return ca.vertcat(float(first_var), steering, rate, radial)

                guesses += _tiered_seeds(
                    first_var_candidates,
                    rate_candidates,
                    radial_candidates,
                    _assemble,
                )
                return _finalize_guesses(guesses)

            if z_arr.size < 7:
                return guesses

            elev = _path_angle_guess(km_copy.angle_elevation, s_current, z_arr[4])
            azim = _path_angle_guess(km_copy.angle_azimuth, s_current, z_arr[5])
            length = min(max(1.02 * r_current, lbx[-1]), ubx[-1])
            steering = (
                input_steering_guess_profile[node_index]
                if input_steering_guess_profile is not None
                else np.clip(
                    _finite_or(z_arr[0], 0.0),
                    DEFAULT_BOUNDS["input_steering"][0],
                    DEFAULT_BOUNDS["input_steering"][1],
                )
            )
            if self.quasi_steady:
                path_rate = (
                    s_dot_guess_profile[node_index]
                    if s_dot_guess_profile is not None
                    else max(abs(_finite_or(z_arr[1], 0.2)), 0.2)
                )
                rate_candidates = _perturb_list(
                    "solver_retry_s_dot",
                    path_rate,
                    multipliers=(0.5, 2.0),
                    positive=True,
                )
            else:
                path_rate = np.clip(
                    _finite_or(z_arr[1], 0.0),
                    DEFAULT_BOUNDS["s_ddot"][0],
                    DEFAULT_BOUNDS["s_ddot"][1],
                )
                rate_candidates = _perturb_list(
                    "solver_retry_s_ddot", path_rate, offsets=(0.5, -0.5)
                )
            speed_radial = (
                speed_radial_profile[node_index]
                if speed_radial_profile is not None
                else _finite_or(z_arr[2], 0.0)
            )
            radial_candidates = _perturb_list(
                "solver_retry_speed_radial", speed_radial, offsets=(1.5, -1.5)
            )
            tension_candidates = sim_params.get(
                "solver_retry_tension_kite", [300.0, 5.0e3, 2.0e4, 8.0e4]
            )

            def _assemble(tension, rate, radial):
                return ca.vertcat(
                    steering,
                    rate,
                    radial,
                    float(tension) / self.WILLIAMS_TENSION_SCALE,
                    elev,
                    azim,
                    length,
                )

            guesses += _tiered_seeds(
                tension_candidates, rate_candidates, radial_candidates, _assemble
            )
            return _finalize_guesses(guesses)

        def _solve_node(z_guess, p_solver, node_index):
            last_error = None
            accept_residual_norm = float(
                sim_params.get("solver_accept_residual_norm", 1e-5)
            )
            candidates = _fallback_z_guesses(z_guess, p_solver, node_index)
            for attempt, candidate in enumerate(candidates):
                try:
                    sol_candidate = qs_solver(
                        x0=candidate, p=p_solver, lbg=lbg, ubg=ubg, lbx=lbx, ubx=ubx
                    )
                except Exception as exc:
                    last_error = exc
                    continue

                stats = qs_solver.stats()
                finite = _is_finite_dm(sol_candidate["x"]) and _is_finite_dm(
                    sol_candidate["g"]
                )
                residual_norm = (
                    float(np.linalg.norm(np.asarray(sol_candidate["g"]).reshape(-1)))
                    if finite
                    else np.inf
                )
                accepted = stats.get("success", False) or (
                    finite and residual_norm <= accept_residual_norm
                )
                if accepted and finite:
                    if attempt:
                        logger.info(
                            "Phase solver recovered at node %d with retry %d",
                            node_index,
                            attempt,
                        )
                    return sol_candidate

                last_error = RuntimeError(
                    f"status={stats.get('return_status')}, finite={finite}, "
                    f"||g||={residual_norm:.3e}"
                )

            message = (
                f"Phase residual solve failed at node {node_index} after "
                f"{len(candidates)} attempts"
            )
            if not allow_failure:
                raise RuntimeError(message) from last_error
            logger.warning("%s: %s", message, last_error)
            return None

        for i in range(N):
            # 1) solve residuals at current s-grid node
            # Update wind speed for this grid point
            km_copy.wind.speed_wind_ref = float(wind_profile[i])
            if use_williams:
                r_idx = 1 if self.quasi_steady else 2
                r_current = float(x[r_idx])
                # Tether length is essentially the chord plus a small slack.
                # Allowing the wide ``[0.5 r, 3 r]`` box gave IPOPT too much
                # freedom; ``[r, r + 10]`` matches the physics of a taut tether.
                lbx[-1] = r_current
                ubx[-1] = r_current + 10.0
                if self.quasi_steady:
                    input_guess = (
                        input_steering_guess_profile[i]
                        if input_steering_guess_profile is not None
                        else float(z[0])
                    )
                    s_dot_guess_i = (
                        s_dot_guess_profile[i]
                        if s_dot_guess_profile is not None
                        else float(z[1])
                    )
                    speed_radial_guess = (
                        speed_radial_profile[i]
                        if speed_radial_profile is not None
                        else float(z[2])
                    )
                    # Seed tether_length slightly slack vs the chord so the
                    # ground-residual constraint starts on the feasibility
                    # side (mirrors solve_single_state.py).
                    z = ca.vertcat(
                        input_guess,
                        s_dot_guess_i,
                        speed_radial_guess,
                        z[3:-1],
                        1.02 * r_current,
                    )
                else:
                    z = ca.vertcat(z[:-1], 1.02 * r_current)
            p_solver = x
            if use_williams and debug_solver and i == 0:
                # One-shot pre-solve diagnostic at node 0: dump z0, p, bounds,
                # and residual evaluated at the initial guess (before IPOPT
                # touches it). The decision-vector layout for QS+Williams is
                # [input_steering, s_dot, speed_radial, tension_kite,
                #  elev_last, az_last, length].
                _williams_pre_solve_dump(
                    qs_solver,
                    z,
                    p_solver,
                    lbx,
                    ubx,
                    quasi_steady=self.quasi_steady,
                )
            sol = _solve_node(z, p_solver, i)
            if sol is None:
                if i == 0:
                    raise RuntimeError(
                        "Phase residual solve failed at the first node; "
                        "no valid trajectory states were generated"
                    )
                break
            if debug_solver:
                stats = qs_solver.stats()
                residual_norm = float(np.linalg.norm(np.asarray(sol["g"]).reshape(-1)))
                print(
                    f"[phase solver] i={i:03d} "
                    f"status={stats.get('return_status')} "
                    f"success={stats.get('success')} "
                    f"||g||={residual_norm:.3e}"
                )
            elif use_williams and not qs_solver.stats().get("success", False):
                stats = qs_solver.stats()
                g_vec = np.asarray(sol["g"]).reshape(-1)
                residual_norm = float(np.linalg.norm(g_vec))
                # g layout for QS+Williams: [force_residual(3) / F_scale,
                # ground_position(3) / r0, winch(1) / F_scale]
                if self.quasi_steady and g_vec.size == 7:
                    parts = (
                        f"||force||={np.linalg.norm(g_vec[0:3]):.2e}, "
                        f"||ground||={np.linalg.norm(g_vec[3:6]):.2e}, "
                        f"winch={g_vec[6]:.2e}"
                    )
                else:
                    parts = ""
                logger.warning(
                    "Williams phase solver failed at node %d: status=%s, "
                    "||g||=%.3e %s",
                    i,
                    stats.get("return_status"),
                    residual_norm,
                    parts,
                )
            z = sol["x"]  # CasADi DM

            # 2) record current state (BEFORE stepping to next s)
            if self.quasi_steady:
                if use_williams:
                    speed_radial = float(z[2])
                    if self._williams_tension_ground_function is not None:
                        tension_ground = self._williams_tension_ground_function(
                            sol["x"], p_solver
                        )
                    else:
                        tension_ground = self.winch_model.tension_curve(
                            speed_radial, input_depower=float(x[2])
                        )
                    curr_state = State(
                        t=t,
                        s=float(x[0]),
                        input_steering=float(z[0]),
                        tension_tether_ground=float(tension_ground),
                        s_dot=float(z[1]),
                        distance_radial=float(x[1]),
                        speed_radial=speed_radial,
                        input_depower=float(x[2]),
                        # Williams tether decisions (QS layout:
                        # [input_steering, s_dot, speed_radial, tension_kite,
                        #  elev_last, az_last, length]). z[3] is the *scaled*
                        # tension decision; unscale for recording.
                        tension_tether_kite=float(z[3]) * self.WILLIAMS_TENSION_SCALE,
                        elevation_last_element=float(z[4]),
                        azimuth_last_element=float(z[5]),
                        tether_length=float(z[6]),
                    )
                else:
                    curr_state = State(
                        t=t,
                        s=float(x[0]),
                        input_steering=float(z[1]),
                        tension_tether_ground=float(z[0]),
                        s_dot=float(z[2]),
                        distance_radial=float(x[1]),
                        speed_radial=float(z[3]),
                        input_depower=float(x[2]),
                    )
            else:
                if use_williams:
                    # Dynamic Williams z: [input_steering, s_ddot, speed_radial,
                    #  tension_kite, elev_last, az_last, length]. Ground tension
                    # is derived from the shape, not a decision.
                    speed_radial_dyn = float(z[2])
                    if self._williams_tension_ground_function is not None:
                        tension_ground = self._williams_tension_ground_function(
                            sol["x"], p_solver
                        )
                    else:
                        tension_ground = self.winch_model.tension_curve(
                            speed_radial_dyn, input_depower=float(x[3])
                        )
                    curr_state = State(
                        t=t,
                        s=float(x[0]),
                        s_dot=float(x[1]),
                        input_steering=float(z[0]),
                        tension_tether_ground=float(tension_ground),
                        s_ddot=float(z[1]),
                        distance_radial=float(x[2]),
                        speed_radial=speed_radial_dyn,
                        input_depower=float(x[3]),
                        tension_tether_kite=float(z[3]) * self.WILLIAMS_TENSION_SCALE,
                        elevation_last_element=float(z[4]),
                        azimuth_last_element=float(z[5]),
                        tether_length=float(z[6]),
                    )
                else:
                    curr_state = State(
                        t=t,
                        s=float(x[0]),
                        s_dot=float(x[1]),
                        input_steering=float(z[1]),
                        tension_tether_ground=float(z[0]),
                        s_ddot=float(z[2]),
                        distance_radial=float(x[2]),
                        speed_radial=float(z[3]),
                        input_depower=float(x[3]),
                    )
            self.states.append(curr_state.to_dict())

            # 4) step to next s using appropriate time increment
            ds = float(s_grid[i + 1] - s_grid[i])  # scalar number

            if self.quasi_steady:
                # a_s = 0 => dt = ds / v_s
                v_s = z[1] if use_williams else z[2]  # s_dot from QS solve
                speed_radial = z[2] if use_williams else z[3]
                dt = ds / (v_s + 1e-12)  # small epsilon to avoid division by zero
                next_r = x[1] + speed_radial * dt
                next_u_dep = u_dep_profile[i + 1]
                x = ca.vertcat(s_grid[i + 1], next_r, next_u_dep, wind_profile[i + 1])
            else:
                # dynamic: ds = v*dt + 0.5*a*dt^2
                # Williams z: [input_steering, s_ddot, speed_radial, ...];
                # rigid z:    [tension_ground, input_steering, s_ddot, speed_radial].
                v_s = x[1]  # current s_dot (state)
                a_s = z[1] if use_williams else z[2]  # current s_ddot (solve result)
                dt = _dt_from_ds_v_a(ds, v_s, a_s)

                next_s_dot = v_s + a_s * dt
                speed_radial = z[2] if use_williams else z[3]
                next_r = x[2] + speed_radial * dt
                x = ca.vertcat(
                    s_grid[i + 1],
                    next_s_dot,
                    next_r,
                    u_dep_profile[i + 1],
                    wind_profile[i + 1],
                )

            # 5) advance time (dt is a CasADi scalar DM; cast to float)
            t += float(dt)

        # print("Total time:", t)
        return self.states if return_states else None

    def _quasi_steady_start_state(self, state_obj: State, km_copy):
        """Solve a quasi-steady residual for the first state to seed dynamic sims."""

        sim_params = self.pattern_config.get("sim_parameters", {})
        qs_unknown_vars = ["length_tether", "input_steering", "s_dot", "speed_radial"]
        if km_copy.is_tether_rigid:
            qs_unknown_vars[0] = "tension_tether_ground"

        # Capture numeric wind speed before residual_solver replaces it with a symbolic.
        # speed_wind_ref_value can be a plain float, a CasADi DM (logarithmic model), or a
        # symbolic MX (already parametrized). float() works for the first two; RuntimeError
        # is raised for MX, in which case we fall back to 0.0.
        _wind_raw = km_copy.wind.speed_wind_ref_value
        try:
            wind_speed_init = float(_wind_raw) if _wind_raw is not None else 0.0
        except (TypeError, RuntimeError):
            wind_speed_init = 0.0

        # Temporarily use quasi-steady residual solver to align the initial state
        original_qs_flag = self.quasi_steady
        self.quasi_steady = True
        qs_solver_init = self.residual_solver(km_copy, quasi_steady=True)
        self.quasi_steady = original_qs_flag

        initial_tension = getattr(state_obj, "tension_tether_ground", 0.0)
        initial_length = getattr(state_obj, "length_tether", state_obj.distance_radial)
        use_williams = isinstance(km_copy.tether, WilliamsTether)
        if use_williams:
            # The Williams NLP uses ``tension_kite_scaled = T / SCALE`` as the
            # decision; divide the physical seed and bounds accordingly.
            tension_scale = self.WILLIAMS_TENSION_SCALE
            z0_init = ca.vertcat(
                state_obj.input_steering,
                state_obj.s_dot,
                state_obj.speed_radial,
                initial_tension / tension_scale,
                (
                    state_obj.angle_elevation
                    if state_obj.angle_elevation is not None
                    else 0.2
                ),
                state_obj.angle_azimuth if state_obj.angle_azimuth is not None else 0.0,
                state_obj.distance_radial,
            )
            p_init = ca.vertcat(
                state_obj.s,
                state_obj.distance_radial,
                state_obj.input_depower,
                wind_speed_init,
            )
            r0 = float(state_obj.distance_radial)
            lbx_init = [
                DEFAULT_BOUNDS["input_steering"][0],
                max(DEFAULT_BOUNDS["s_dot"][0], 0.1),
                DEFAULT_BOUNDS["speed_radial"][0],
                0.0,
                -np.pi / 2 + 1e-3,
                -2.0 * np.pi,
                0.5 * r0,
            ]
            ubx_init = [
                DEFAULT_BOUNDS["input_steering"][1],
                DEFAULT_BOUNDS["s_dot"][1],
                DEFAULT_BOUNDS["speed_radial"][1],
                1.0e5 / tension_scale,
                np.pi / 2 - 1e-3,
                2.0 * np.pi,
                3.0 * r0,
            ]
            lbg_init = [0.0] * 7
            ubg_init = [0.0] * 7
        else:
            z0_init = ca.vertcat(
                initial_tension if km_copy.is_tether_rigid else initial_length,
                state_obj.input_steering,
                state_obj.s_dot,
                state_obj.speed_radial,
            )
            p_init = ca.vertcat(
                state_obj.s,
                state_obj.distance_radial,
                state_obj.input_depower,
                wind_speed_init,
            )
            lbx_init, ubx_init, lbg_init, ubg_init = km_copy.get_boundaries(
                state_obj, qs_unknown_vars
            )
        sol_init = qs_solver_init(
            x0=z0_init,
            p=p_init,
            lbg=lbg_init,
            ubg=ubg_init,
            lbx=lbx_init,
            ubx=ubx_init,
        )
        print(sol_init["x"])
        if use_williams:
            speed_radial_init = float(sol_init["x"][2])
            return State(
                t=state_obj.t,
                s=float(p_init[0]),
                s_dot=float(sol_init["x"][1]),
                input_steering=float(sol_init["x"][0]),
                tension_tether_ground=float(
                    self.winch_model.tension_curve(
                        speed_radial_init, input_depower=float(p_init[2])
                    )
                ),
                distance_radial=float(p_init[1]),
                speed_radial=speed_radial_init,
            )
        if km_copy.is_tether_rigid:
            return State(
                t=state_obj.t,
                s=float(p_init[0]),
                s_dot=float(sol_init["x"][2]),
                input_steering=float(sol_init["x"][1]),
                tension_tether_ground=float(sol_init["x"][0]),
                distance_radial=float(p_init[1]),
                speed_radial=float(sol_init["x"][3]),
            )

        return State(
            t=state_obj.t,
            s=float(p_init[0]),
            s_dot=float(sol_init["x"][2]),
            input_steering=float(sol_init["x"][1]),
            length_tether=float(sol_init["x"][0]),
            distance_radial=float(p_init[1]),
            speed_radial=float(sol_init["x"][3]),
        )

    def _resolve_opti_limits(self) -> dict:
        """``DEFAULT_OPTI_LIMITS`` overlaid with the system's hardware limits.

        Hardware limits (KCU steering/depower actuator range and slew rate, and
        the max tether length) are sourced from ``system.yaml`` via
        ``SystemModel.hardware_limits`` (see
        ``factory._extract_hardware_limits``) and take precedence over the
        numerical fallbacks. The ``_max_tether_length`` sentinel sets the
        ``distance_radial`` upper bound (the radial chord cannot exceed the
        tether) while keeping the default lower bound. Any limit not provided by
        the system config falls back to ``DEFAULT_OPTI_LIMITS``.
        """
        limits = dict(DEFAULT_OPTI_LIMITS)
        hardware = getattr(self.kite_model, "hardware_limits", None) or {}
        for key, value in hardware.items():
            if key == "_max_tether_length":
                lb = limits.get("distance_radial", (0.0, value))[0]
                limits["distance_radial"] = (lb, float(value))
            else:
                limits[key] = tuple(value)
        return limits

    def opti_phase(
        self,
        start_state,
        opti=None,
        start_state_opti=None,
        opti_params=None,
        relax_tol=0.0,
    ):

        if not opti:
            opti = ca.Opti()

        # Stored warm-start trajectory, injected by ``Phase.initialize_phase``:
        # the previous NLP optimum between staged solves, or a forward
        # simulation the caller already ran (``Phase.run_simulation(...,
        # seed_warm_start=True)``). When a variable matches this grid it seeds
        # the NLP instead of the forward simulation: pass-to-pass warm starts
        # are then exact, and -- crucially for ``winch_mode: free_speed`` --
        # the force-law march is no longer a failure point between passes.
        # The march always closes v_r with the winch tension curve, so it can
        # fail to trim a shape that was optimized under free reel speed even
        # though the NLP is perfectly solvable from the previous optimum.
        # ``warm_start_is_optimum`` tells a previous optimum (used as-is) from
        # a simulated seed (still rate-limited below in free_speed mode).
        n_grid = int(self.pattern_config["sim_parameters"]["n_points"])
        warm_traj = {
            key: np.asarray(values, dtype=float).ravel()
            for key, values in (
                getattr(self, "warm_start_trajectory", None) or {}
            ).items()
            if np.asarray(values).size == n_grid
        }
        warm_is_optimum = bool(getattr(self, "warm_start_is_optimum", False))

        def _warm_hist(name):
            """Warm-start history: stored trajectory, else the forward sim."""
            if name in warm_traj:
                return warm_traj[name]
            return np.asarray(self.return_variable(name), dtype=float).ravel()

        # The forward march is only needed for seeds the stored trajectory
        # does not provide. When it covers every per-node decision the march
        # would re-derive values that are never used (its result was always
        # overridden by the stored trajectory) -- at ~0.3-0.4 s/node for a
        # full cycle, so skip it. Missing/mismatched keys fall back to the
        # march exactly as before.
        seed_keys = {
            "s",
            "s_dot",
            "input_steering",
            "speed_radial",
            "distance_radial",
            "tension_tether_ground",
        }
        if bool(
            self.pattern_config.get("sim_parameters", {}).get(
                "optimize_depower_profile", False
            )
        ):
            seed_keys.add("input_depower")
        if seed_keys <= set(warm_traj):
            self.states = []
            print(
                "Warm start: NLP seeded from the stored "
                f"{'optimum' if warm_is_optimum else 'forward-simulation'} "
                f"trajectory ({n_grid} nodes); warm-start forward simulation "
                "skipped."
            )
        else:
            try:
                self.run_simulation_phase(start_state)
            except RuntimeError as exc:
                if not warm_traj:
                    raise
                print(
                    f"Warm-start forward simulation failed mid-cycle ({exc}); "
                    "seeding the NLP from the stored trajectory instead."
                )
        chi_hist = _warm_hist("angle_course")
        if len(chi_hist):
            print(f"Initial course angle (chi) from warm start: {chi_hist[0]:.2f} rad")
            print(f"Final course angle (chi) from warm start: {chi_hist[-1]:.2f} rad")
        self.kite_model.reset_solver()

        if start_state_opti:
            start_state = start_state_opti
        # initial state object
        state_obj = (
            State(**start_state) if isinstance(start_state, dict) else start_state
        )
        # Replace optimized parameters with symbolic variables
        path_params = copy.deepcopy(self.pattern_config_opti.get("path_parameters", {}))
        radial_params = copy.deepcopy(
            self.pattern_config_opti.get("radial_parameters", {})
        )
        sim_params = copy.deepcopy(self.pattern_config_opti.get("sim_parameters", {}))

        # Optimizer limits: numerical defaults overlaid with the system's
        # hardware limits (KCU actuator range/rate, max tether length) from
        # system.yaml. Use ``limits`` (not DEFAULT_OPTI_LIMITS) for every bound
        # below so the hardware values take precedence.
        limits = self._resolve_opti_limits()
        # Per-config bound overrides: ``sim_parameters["opti_limits_override"]`` is
        # a {name: [lb, ub]} map applied on top of the resolved limits. Lets a
        # specific run widen a bound (e.g. C_beta for a full-cycle periodic spline
        # whose reel-in elevation exceeds the default coefficient range) without
        # editing the global DEFAULT_OPTI_LIMITS.
        for _name, _bounds in (sim_params.get("opti_limits_override") or {}).items():
            limits[_name] = tuple(_bounds)

        pattern = create_pattern_from_dict(
            self.pattern_config_opti["pattern_type"], path_params
        )

        N = int(sim_params["n_points"])

        # Optimize the depower input as a per-node profile (one decision per
        # discretization point, like ``input_steering``) instead of a single
        # scalar. Gated behind a sim_parameters flag so existing scalar-depower
        # optimization is unchanged.
        optimize_depower_profile = bool(sim_params.get("optimize_depower_profile", False))

        # Winch mode. "force_law" (default): the winch enforces its tension
        # curve T = f(v_r, l_dp) as a per-node equality -- v_r follows from the
        # force balance. "free_speed": the reel speed is a DIRECT rate-limited
        # control (like steering/depower) and the winch only bounds the tension
        # it can hold, [min_tether_force, max_tether_force]. Free speed removes
        # the tension-curve equality whose soft-clamp plateau (dT/dv_r ~ 0 when
        # riding the force limit) destroys the Jacobian rank, and makes radial
        # closure nearly linear in a direct control. The optimal T(v_r, l_dp)
        # law is then an OUTPUT to regress a controller from, not an input.
        # NLP-only: the forward warm-start simulation still marches with the
        # force law, which simply seeds v_r with a feasible trajectory.
        free_speed = sim_params.get("winch_mode", "force_law") == "free_speed"
        winch_acc = tuple(
            sim_params.get(
                "winch_acceleration",
                (
                    DEFAULT_WINCH_CONFIG["min_acceleration"],
                    DEFAULT_WINCH_CONFIG["max_acceleration"],
                ),
            )
        )
        if free_speed:
            print(
                "NLP winch mode: free_speed -- v_r is a rate-limited control "
                f"(accel {winch_acc[0]:+.1f}..{winch_acc[1]:+.1f} m/s^2), tension "
                f"bounded to [{float(radial_params.get('min_tether_force', 0.0)):.0f}, "
                f"{float(radial_params['max_tether_force']):.0f}] N"
            )
        else:
            print("NLP winch mode: force_law (tension-curve equality per node)")

        tau = ca.DM(np.linspace(0, 1, N + 1))  # numeric grid (DM column vector)

        s0 = sim_params["start_angle"]  # can be float or MX
        s1 = sim_params["end_angle"]  # MX (Opti variable) in your case

        # Symbolic affine map: s_grid is MX because s1 is MX
        s_grid = s0 + (s1 - s0) * tau
        winch_model = Winch(pattern_config=radial_params)
        km_copy = self.substitute_parametrized_kinematics(pattern)
        self.km_param = km_copy

        # --- Decision variables per node (N nodes for intervals 0..N-1)
        # Tension (~1e5-1e6 N) and radius (~1e2 m) dwarf every other decision
        # (O(1)). Leaving them unscaled wrecks the conditioning of the NLP
        # (oscillation, ||d|| ~1e3 steps, restoration, a dual-infeasibility floor).
        # Carry both as O(1) decisions ``x_scaled = x / X_SCALE`` and expose the
        # physical value through ``opti_vars`` so every downstream constraint /
        # objective is untouched. The scales match the constraint scales S["T"] /
        # S["r"] below (90th percentile of the warm start).
        def _warm_scale(name):
            vals = np.abs(_warm_hist(name))
            return float(max(np.percentile(vals, 90), 1.0)) if vals.size else 1.0

        T_SCALE = _warm_scale("tension_tether_ground")
        R_SCALE = _warm_scale("distance_radial")
        tension_scaled = opti.variable(N)  # O(1) decision: T / T_SCALE
        distance_scaled = opti.variable(N)  # O(1) decision: r / R_SCALE
        opti_vars = {
            "s": s_grid,
            "s_dot": opti.variable(N),  # tangential speed
            "input_steering": opti.variable(N),
            "speed_radial": opti.variable(N),  # reel speed v_r
            "distance_radial": distance_scaled * R_SCALE,  # physical radius r
            "tension_tether_ground": tension_scaled * T_SCALE,  # physical tension T
        }
        if optimize_depower_profile:
            # Per-node power-tape length l_dp; O(1), no scaling needed. Bounds
            # come from DEFAULT_OPTI_LIMITS["input_depower"] via the generic
            # vector-bound loop below; the rate limit is applied in the node loop.
            opti_vars["input_depower"] = opti.variable(N)

        # Per-node controls (data-driven; see NodeControl). The slew-rate limit
        # and smoothness regularizer below iterate this list instead of carrying
        # one hand-written block per control. Order (steering, then depower) is
        # preserved from the previous code so the NLP is byte-identical.
        node_controls = [
            NodeControl(
                name="input_steering",
                rate_limit=tuple(limits["steering_rate"]),
            )
        ]
        if optimize_depower_profile:
            node_controls.append(
                NodeControl(
                    name="input_depower",
                    rate_limit=tuple(
                        sim_params.get("depower_rate", limits["depower_rate"])
                    ),
                )
            )
        if free_speed:
            # Reel speed as a control: slew-rate limited by the winch drive's
            # acceleration capability, smoothness-regularized like the others.
            node_controls.append(
                NodeControl(name="speed_radial", rate_limit=winch_acc)
            )
        # # expose design params too
        # for var in self.optimization_vars:
        #     opti_vars[var] = self.optimization_vars[var]

        # --- Helper to check warm start against bounds
        def check_warm_start(var_name, values, bounds):
            if not bounds or len(bounds) != 2:
                return
            lb, ub = bounds
            values = np.asarray(values).ravel()
            violations_lb = values < lb
            violations_ub = values > ub
            if np.any(violations_lb) or np.any(violations_ub):
                n_violations = np.sum(violations_lb) + np.sum(violations_ub)
                print(
                    f"Warning: Warm start for {var_name} violates bounds in {n_violations} points"
                )
                if np.any(violations_lb):
                    min_val = np.min(values[violations_lb])
                    print(f"  - Below lower bound ({lb}): min value = {min_val}")
                if np.any(violations_ub):
                    max_val = np.max(values[violations_ub])
                    print(f"  - Above upper bound ({ub}): max value = {max_val}")

        # --- Warm starts from simulation (with bound checking)
        # The per-node decisions are length N. The warm-start simulation can be
        # SHORTER than N when a node fails to converge and the QS march breaks
        # early (e.g. the stiff reel-out/reel-in transition of a full-cycle
        # spline). Pad each seed up to N by repeating the last converged value so
        # ``set_initial`` never dimension-mismatches; IPOPT refines the tail.
        def _fit_len(values, n):
            arr = np.asarray(values, dtype=float).ravel()
            if arr.size == 0:
                return np.zeros(n)
            if arr.size >= n:
                return arr[:n]
            print(
                f"Warm start has {arr.size} nodes < {n}; padding the tail with the "
                "last converged value (forward sim truncated at a failed node)."
            )
            return np.concatenate([arr, np.full(n - arr.size, arr[-1])])

        warm_starts = {
            "s_dot": _fit_len(_warm_hist("s_dot"), N),
            "input_steering": _fit_len(_warm_hist("input_steering"), N),
            "speed_radial": _fit_len(_warm_hist("speed_radial"), N),
            "distance_radial": _fit_len(_warm_hist("distance_radial"), N),
            "tension_tether_ground": _fit_len(
                _warm_hist("tension_tether_ground"), N
            ),
        }
        if optimize_depower_profile:
            # Seed from the (constant) depower used in the warm-start simulation.
            warm_starts["input_depower"] = _fit_len(_warm_hist("input_depower"), N)
        if free_speed and ("speed_radial" not in warm_traj or not warm_is_optimum):
            # The force-law seed steps v_r sharply at the force-clamp corners,
            # violating the new winch-acceleration constraints from iteration 0
            # (a needless ~1-2 of initial primal infeasibility). Rate-limit the
            # seed with a forward-backward clip so those rows start feasible.
            # (A previous-optimum seed already satisfies the acceleration rows,
            # so it is used as-is; a stored forward-simulation seed is clipped
            # like a fresh one.) The step times come from the stored
            # trajectory when it carries ``t``, else from the march.
            vr = np.asarray(warm_starts["speed_radial"], dtype=float).copy()
            dt = _fit_len(np.diff(_warm_hist("t")), max(N - 1, 1))
            dt = np.clip(dt, 1e-3, None)
            for i in range(N - 1):
                vr[i + 1] = min(
                    max(vr[i + 1], vr[i] + winch_acc[0] * dt[i]),
                    vr[i] + winch_acc[1] * dt[i],
                )
            for i in range(N - 2, -1, -1):
                vr[i] = min(
                    max(vr[i], vr[i + 1] - winch_acc[1] * dt[i]),
                    vr[i + 1] - winch_acc[0] * dt[i],
                )
            warm_starts["speed_radial"] = vr

        # set_initial needs the raw decision variable; tension and radius are
        # exposed as scaled expressions, so seed the underlying scaled variables
        # with the warm start divided by the matching scale.
        raw_init_vars = dict(opti_vars)
        raw_init_vars["tension_tether_ground"] = tension_scaled
        raw_init_vars["distance_radial"] = distance_scaled
        init_scales = {"tension_tether_ground": T_SCALE, "distance_radial": R_SCALE}

        print("\nChecking warm start values against bounds:")
        for var_name, values in warm_starts.items():
            # Check against optimization bounds if defined
            if var_name in limits:
                check_warm_start(var_name, values, limits[var_name])
            # Set the initial value regardless of violations (scaled seeds for
            # the scaled decisions).
            init_values = np.asarray(values) / init_scales.get(var_name, 1.0)
            opti.set_initial(raw_init_vars[var_name], init_values)

        # Fix initial radius (constrain the scaled decision so the row stays
        # O(1)). With ``r0`` as a named optimization parameter the pin follows
        # the symbol instead of the numeric start state: the operating radius
        # becomes a design variable -- seeded from ``path_parameters["r0"]``
        # and bounded by ``limits["r0"]`` like every named parameter -- and the
        # radial closure ``r[-1] == r[0]`` tracks it, so the whole radius band
        # floats with r0 (each node still inside the ``distance_radial`` band).
        r0_opti = (opti_params or {}).get("r0")
        if r0_opti is not None:
            opti.subject_to(distance_scaled[0] == r0_opti / R_SCALE)
        else:
            opti.subject_to(
                distance_scaled[0] == state_obj.distance_radial / R_SCALE
            )

        # Optional radial-cycle closure: tie the final radius back to the initial
        # one so a single periodic phase represents a *closed* pumping cycle (net
        # zero tether-length change over the period). Off by default -- a
        # stand-alone reel-out phase ends at a larger radius by design -- and
        # enabled via sim_parameters["close_radial_cycle"] for the full-cycle
        # spline. Constrain the scaled decisions so the row stays O(1).
        if bool(sim_params.get("close_radial_cycle", False)):
            opti.subject_to(distance_scaled[-1] == distance_scaled[0])

        # --- Build model functions
        km_copy.establish_residual()
        flat_syms = [ca.vertcat(*opti_params.values())] if opti_params else []
        if not "input_depower" in opti_params:
            flat_syms.append(
                km_copy.input_depower
            )  # treat depower as param if not optimized

        residual = ca.Function(
            "residual",
            [
                self.s,
                self.s_dot,
                km_copy.input_steering,
                km_copy.tension_tether_ground,
                km_copy.speed_radial,
                km_copy.distance_radial,
            ]
            + flat_syms,
            [km_copy.residual],
        )
        aoa_eq = ca.Function(
            "speed_tangential_eq",
            [
                self.s,
                self.s_dot,
                km_copy.input_steering,
                km_copy.tension_tether_ground,
                km_copy.speed_radial,
                km_copy.distance_radial,
            ]
            + flat_syms,
            [km_copy.expression("angle_of_attack")],
        )
        chi_eq = ca.Function(
            "chi_eq",
            [
                self.s,
                # self.s_dot,
            ]
            + flat_syms,
            [km_copy.angle_course],
        )
        # Geodesic curvature of the pattern on the unit sphere (a pure function
        # of the spline coefficients and s; see
        # ``ParametrizedKinematics.curvature_geodesic``). Same input list as the
        # residual so it can be evaluated node-by-node with the same argument
        # bundle; the speed/tension inputs are simply unused. Feeds the optional
        # minimum-turn-radius constraint below and the per-node ``turn_radius``
        # diagnostic exported through the objective dict.
        path_kin = ParametrizedKinematics(pattern, self)
        curvature_eq = ca.Function(
            "path_curvature",
            [
                self.s,
                self.s_dot,
                km_copy.input_steering,
                km_copy.tension_tether_ground,
                km_copy.speed_radial,
                km_copy.distance_radial,
            ]
            + flat_syms,
            [
                path_kin.curvature_geodesic,
                path_kin.curvature_numerator,
                path_kin.dsigma_ds,
            ],
        )
        # Per-node Functions: expand each ONCE to SX (see
        # ``_expand_node_function``) so the N calls below evaluate and
        # differentiate as compiled scalar code without the solver having to
        # expand the whole N-node graph. ``sim_parameters
        # ["expand_node_functions"]`` (default True) turns this off; combine
        # with ``expand_nlp: false`` for linear-in-N construction time/memory.
        expand_nodes = bool(sim_params.get("expand_node_functions", True))
        if expand_nodes:
            residual = _expand_node_function(residual)
            aoa_eq = _expand_node_function(aoa_eq)
            chi_eq = _expand_node_function(chi_eq)
            curvature_eq = _expand_node_function(curvature_eq)

        # --- Local-support node functions (periodic spline, coefficients
        # optimized). A shared per-node Function takes ``s`` symbolically, so
        # its graph references ALL M spline coefficients and its Jacobian is a
        # dense M-wide block per node (~2x the NLP Jacobian nonzeros, ~10x
        # the per-iteration derivative work at M~40) -- which is what
        # whole-graph ``expand=True`` avoids by constant-folding each node's
        # numeric ``s``. Instead, build the Functions per KNOT INTERVAL from a
        # truncated-support copy of the spline (``PeriodicBSpline
        # .local_support(k)``: only the 4 basis terms non-zero on that
        # interval), lazily and cached: at most M copies of a ~0.1 s build,
        # independent of n_points, and every node call depends structurally
        # on exactly its 4+4 supporting control points. Exactness: the
        # truncation is exact on the closed interval, so picking k by
        # ``knot_interval(s_i)`` reproduces the full spline at every node and
        # sub-sample. Only with SX node functions (where the dense block
        # costs) and a periodic spline whose coefficients are decisions.
        local_support = (
            expand_nodes
            and isinstance(pattern, PeriodicBSpline)
            and pattern.support_interval is None
            and any(name in (opti_params or {}) for name in ("C_phi", "C_beta"))
        )
        _node_fn_cache = {}
        # Function INPUT symbols, captured now: the trailing flat_sym (the
        # scalar depower) is pinned to a number just before the node loop, and
        # the per-interval Functions are built lazily -- possibly after that.
        _node_fn_syms = list(flat_syms)
        _node_in_base = [
            self.s,
            self.s_dot,
            km_copy.input_steering,
            km_copy.tension_tether_ground,
            km_copy.speed_radial,
            km_copy.distance_radial,
        ]

        def _node_functions(k):
            """(residual, aoa, curvature, path) SX Functions for knot interval k."""
            fns = _node_fn_cache.get(k)
            if fns is None:
                pat_k = pattern.local_support(k)
                km_k = self.substitute_parametrized_kinematics(pattern=pat_k)
                km_k.establish_residual()
                kin_k = ParametrizedKinematics(pat_k, self)
                ins = _node_in_base + _node_fn_syms
                fns = (
                    _expand_node_function(
                        ca.Function(f"residual_k{k}", ins, [km_k.residual])
                    ),
                    _expand_node_function(
                        ca.Function(
                            f"aoa_k{k}", ins, [km_k.expression("angle_of_attack")]
                        )
                    ),
                    _expand_node_function(
                        ca.Function(
                            f"path_curvature_k{k}",
                            ins,
                            [
                                kin_k.curvature_geodesic,
                                kin_k.curvature_numerator,
                                kin_k.dsigma_ds,
                            ],
                        )
                    ),
                    _expand_node_function(
                        ca.Function(
                            f"path_k{k}",
                            [self.s] + _node_fn_syms,
                            [
                                pat_k.azimuth(km_k.distance_radial, self.s),
                                pat_k.elevation(km_k.distance_radial, self.s),
                            ],
                        )
                    ),
                )
                _node_fn_cache[k] = fns
            return fns

        def _knot_of(s_value):
            return int(pattern.knot_interval(float(s_value)))

        def _node_syms(i):
            """Call-time argument bundle of node i: the profile depower swaps
            in per node; a fixed scalar depower is its numeric value (the same
            pin the node loop applies to ``flat_syms`` below)."""
            syms = list(flat_syms)
            if optimize_depower_profile:
                syms[-1] = opti_vars["input_depower"][i]
            elif "input_depower" not in opti_params:
                syms[-1] = sim_params["input_depower"]
            return syms

        # Diagnostic flag (also lets callers pick expand_nlp per problem: with
        # local support both modes are cheap -- True ~2x faster per iteration,
        # False ~40% less memory; without it True is the only fast one).
        self.nlp_local_support = bool(local_support)
        if local_support:
            print(
                "NLP node functions: local-support periodic spline (per knot "
                f"interval, M={pattern.M}) -- sparse coefficient Jacobian"
            )
        # Optional minimum physical turn radius R_min [m] of the flown path
        # (``sim_parameters["min_turn_radius"]``; None/0 -> off, the NLP is
        # unchanged), i.e. r / |kappa| >= R_min. Enforced on
        # ``turn_radius_subsamples`` points per node interval (default 4, k = 0
        # being the node itself, node states held over the interval): the
        # spline can form a curvature peak or a cusp BETWEEN two nodes that no
        # node-wise quantity sees, so node-only enforcement misses exactly the
        # paths that matter. Written in the polynomial (cross-product) form
        #     -r sigma'^3 <= R_min * (kappa sigma'^3) <= r sigma'^3
        # rather than R_min |kappa| <= r: kappa ~ 1/sigma'^3 blows up near a
        # cusp, and a cold IPOPT path that passes through a near-cusp shape then
        # stalls on the exploding rows (300-340 m cold solves went from
        # converged to 1000-iteration failures with the |kappa| form); the
        # polynomial numerator is smooth and bounded there. Purely geometric,
        # independent of the steering ROM: tight loops and near-cusps are
        # infeasible by construction (an exactly straight, reversing path has
        # zero curvature and is not excluded -- that collapse is caught by the
        # dynamics being infeasible, not by geometry).
        min_turn_radius = float(sim_params.get("min_turn_radius") or 0.0)
        if min_turn_radius < 0.0:
            raise ValueError(
                f"sim_parameters['min_turn_radius'] must be >= 0, got {min_turn_radius}"
            )
        turn_radius_subsamples = int(sim_params.get("turn_radius_subsamples", 4))
        if turn_radius_subsamples < 1:
            raise ValueError("sim_parameters['turn_radius_subsamples'] must be >= 1")
        # Companion floor on the parametric speed sigma'(s) >= floor * sigma_ref
        # at the same samples (constraint active only). The periodic spline can
        # otherwise form a HESITATION point -- sigma' -> 0 over a sliver of s,
        # s_dot jumping up to compensate -- carrying a millimetre-long kink of
        # tiny radius. That kink is physically meaningless but shows up as a
        # sub-R_min sample in the returned path, and the sigma'^3-scaled rows
        # above cannot see it (both sides vanish below IPOPT's tolerance).
        # Legitimate optima have min sigma' >= ~0.4 sigma_ref; the artefacts
        # sit at <= 0.1, so 0.2 separates them without touching real shapes.
        turn_radius_sigma_floor = float(sim_params.get("turn_radius_sigma_floor", 0.2))
        if not 0.0 <= turn_radius_sigma_floor < 1.0:
            raise ValueError(
                "sim_parameters['turn_radius_sigma_floor'] must be in [0, 1)"
            )
        _sigma_ref = 1.0  # typical unit-sphere arc rate sigma' of the pattern
        if min_turn_radius > 0.0:
            print(
                f"NLP turn-radius constraint: R >= {min_turn_radius:g} m on "
                f"{turn_radius_subsamples} sample(s) per node interval"
            )
            # sigma' scale from the numeric starting pattern: angular path
            # length of one pass over [s0, s1] divided by the s-span.
            try:
                _pat0 = create_pattern_from_dict(
                    self.pattern_config["pattern_type"],
                    self.pattern_config["path_parameters"],
                )
                _s0 = float(self.pattern_config["sim_parameters"]["start_angle"])
                _s1 = float(self.pattern_config["sim_parameters"]["end_angle"])
                _ss = np.linspace(_s0, _s1, 400)
                _r0 = float(self.pattern_config["path_parameters"].get("r0", 1.0))
                _ph = np.asarray(_pat0.azimuth(_r0, _ss)).ravel()
                _be = np.asarray(_pat0.elevation(_r0, _ss)).ravel()
                _dsig = np.sqrt(
                    np.diff(_be) ** 2 + (np.diff(_ph) * np.cos(_be[:-1])) ** 2
                )
                _sigma_ref = float(max(np.sum(_dsig) / max(_s1 - _s0, 1e-9), 1e-3))
            except Exception:
                _sigma_ref = 1.0
        # Row scale for the polynomial constraint below: R_min * kappa * sigma'^3
        # ~ r sigma'^3 at the bound, so divide both sides by R_SCALE sigma_ref^3.
        _turn_scale = R_SCALE * _sigma_ref**3
        # Constrained NLP expressions collected for the post-solve constraint
        # report (``Phase._print_constraint_report``). Each row carries the
        # NLP's OWN expression plus the bounds actually applied, so the
        # reported margins are exactly what IPOPT saw -- inequality rows as
        # {expr, lb, ub, unit, scale}, equality rows as {expr, equality: True}
        # reporting max |scaled residual|.
        constraint_report = {}

        def _report_ineq(name, expr, bounds, unit="", scale=1.0):
            constraint_report[name] = {
                "expr": expr,
                "lb": float(bounds[0]),
                "ub": float(bounds[1]),
                "unit": unit,
                "scale": float(scale),
            }

        # --- Safety / geometry constraint
        if local_support:
            # Node elevations through the local-support path functions: the
            # height rows then depend on each node's 4 supporting C_beta
            # entries only (the dense N x M block of ``pattern.z`` on the
            # full spline is what every other node row avoids).
            elevation_nodes = ca.vertcat(
                *[
                    _node_functions(_knot_of(s_grid[i]))[3](
                        s_grid[i], *_node_syms(i)
                    )[1]
                    for i in range(N)
                ]
            )
            height = opti_vars["distance_radial"] * ca.sin(elevation_nodes)
        else:
            height = pattern.z(opti_vars["distance_radial"], s_grid[:-1])  # N entries
        opti.subject_to(height >= limits["height"][0])
        opti.subject_to(height <= limits["height"][1])
        _report_ineq("height", height, limits["height"], "m")

        # --- Optional azimuth-amplitude floor
        # ``sim_parameters["min_azimuth_amplitude"]`` [rad] guards the degenerate
        # C_phi -> 0 collapse (a zero-width, reversing "figure" on which the
        # dynamics are infeasible) with ONE smooth row: the mean square of the
        # azimuth over the nodes must be >= a^2/2. A figure-eight or helix of
        # azimuth half-width A has mean(phi^2) = A^2/2, so this reads "the
        # figure's azimuth half-width stays >= a". Off by default: the NLP is
        # unchanged when the key is absent or 0. (Box limits on where the path
        # may go are the C_phi / C_beta coefficient bounds -- convex hull of the
        # B-spline -- set via ``opti_limits_override``.)
        min_azimuth_amplitude = float(
            sim_params.get("min_azimuth_amplitude") or 0.0
        )
        if min_azimuth_amplitude < 0.0:
            raise ValueError(
                f"min_azimuth_amplitude must be >= 0, got {min_azimuth_amplitude}"
            )
        if min_azimuth_amplitude > 0.0:
            azimuth_nodes = pattern.azimuth(
                opti_vars["distance_radial"], s_grid[:-1]
            )  # N entries
            azimuth_ms = ca.sumsqr(azimuth_nodes) / azimuth_nodes.numel()
            amplitude_row = 2.0 * azimuth_ms - min_azimuth_amplitude**2
            # (``ca.depends_on(expr, opti.x)`` is False for Opti variables --
            # opti.x is a view, not the symbols -- so test for free symbols.)
            if isinstance(amplitude_row, ca.MX) and ca.symvar(amplitude_row):
                opti.subject_to(amplitude_row >= 0.0)
                _report_ineq(
                    "azimuth_amplitude (rms*sqrt2)",
                    ca.sqrt(2.0 * azimuth_ms + 1e-12),
                    (min_azimuth_amplitude, np.inf),
                    "rad",
                )
                print(
                    "Azimuth amplitude floor: "
                    f"{np.degrees(min_azimuth_amplitude):.1f} deg"
                )
            else:
                # C_phi is not an optimization parameter: the pattern's azimuth
                # is fixed, so the floor is a plain check on the given shape.
                value = (
                    float(ca.evalf(amplitude_row))
                    if isinstance(amplitude_row, ca.MX)
                    else float(amplitude_row)
                )
                if value < 0.0:
                    raise ValueError(
                        "min_azimuth_amplitude "
                        f"({np.degrees(min_azimuth_amplitude):.1f} deg) is "
                        "violated by the fixed pattern and C_phi is not optimized"
                    )

        # Constraint init and end azimuth
        # azimuth = pattern.azimuth(opti_vars["distance_radial"], s_grid[:-1])
        # opti.subject_to(azimuth[0] == 0)
        # opti.subject_to(azimuth[-1] == 0)

        # Constraint init course angle (chi)
        # chi_init = chi_eq(
        #     s_grid[0],
        #     # opti_vars["s_dot"][0],
        #     *flat_syms,
        # )
        # opti.subject_to(chi_init == np.pi)  # start flying straight downwind
        # chi_final = chi_eq(
        #     s_grid[-2],
        #     # opti_vars["s_dot"][-1],
        #     *flat_syms,
        # )
        # opti.subject_to(chi_final == np.pi)  # end flying straight upwind

        # --- Power scale based on the warm-start trajectory (LEFT RULE,
        # consistent). Prefer the previous optimum (P = T * v_r, dt = ds/s_dot,
        # the NLP's own quadrature); else the forward simulation's history.
        if {"s", "s_dot", "tension_tether_ground", "speed_radial"} <= set(warm_traj):
            dt_hist = np.diff(warm_traj["s"]) / warm_traj["s_dot"][:-1]
            P_hist = warm_traj["tension_tether_ground"] * warm_traj["speed_radial"]
        else:
            t_hist = self.return_variable("t")  # length N (QS) or N+1
            P_hist = self.return_variable("mechanical_power")  # same length
            dt_hist = np.diff(t_hist)  # length N-1
        E0 = float(np.sum(P_hist[:-1] * dt_hist))  # left Riemann sum
        T0 = float(np.sum(dt_hist))
        P0 = E0 / (T0 + 1e-12)
        P_scale = max(abs(P0), 1.0)

        # --- Auto scales from warm start (robust to outliers)
        def _scale(x, floor=1.0):
            x = np.asarray(x).ravel()
            if x.size == 0:
                return float(floor)
            s = np.percentile(np.abs(x), 90)  # “typical large” value
            return float(max(s, floor))

        r_hist = _warm_hist("distance_radial")
        vr_hist = _warm_hist("speed_radial")
        sd_hist = _warm_hist("s_dot")
        T_hist = _warm_hist("tension_tether_ground")
        u_hist = _warm_hist("input_steering")

        S = {
            "r": _scale(r_hist, floor=1.0),
            "vr": _scale(vr_hist, floor=1.0),
            "sd": _scale(sd_hist, floor=1.0),
            "T": _scale(T_hist, floor=1.0),
            "u": _scale(u_hist, floor=1.0),
        }
        # Residual equation scales (fallback: tie to tension scale)
        S_res = [max(S["T"], 1.0)] * 3

        # --- Helpful bounds to keep NLP well-posed
        sdot_min = 1e-2  # ensures dt>0
        opti.subject_to(opti_vars["s_dot"] >= sdot_min)
        if "speed_radial" in limits:
            lb, ub = limits["speed_radial"]
            print(f"Applying speed_radial limits: lb={lb}, ub={ub}")
            opti.subject_to(opti_vars["speed_radial"] >= lb)
            opti.subject_to(opti_vars["speed_radial"] <= ub)
            _report_ineq("speed_radial", opti_vars["speed_radial"], (lb, ub), "m/s")
        if "distance_radial" in limits:
            lb, ub = limits["distance_radial"]
            # Optional upper-bound relaxation: lets a phase reel a bit past the
            # global max radius. Used by the reel-in, whose first node starts at
            # the reel-out end -- which sits right at this bound when reel-out
            # maximizes production, leaving no feasible room otherwise.
            ub = ub + float(sim_params.get("distance_radial_ub_relax", 0.0))
            # Bound the scaled decision so the bound rows' Jacobian stays O(1).
            opti.subject_to(distance_scaled >= lb / R_SCALE)
            opti.subject_to(distance_scaled <= ub / R_SCALE)
            _report_ineq("distance_radial", opti_vars["distance_radial"], (lb, ub), "m")

        # --- Objective assembly with SAME quadrature as simulation (left rule)
        energy = 0
        t_eff = 0

        if "input_depower" not in opti_params and not optimize_depower_profile:
            # Scalar depower (not optimized per node): pin the trailing flat_sym
            # to the numeric value. In profile mode the trailing flat_sym stays
            # symbolic and is filled per node from opti_vars["input_depower"].
            flat_syms[-1] = sim_params["input_depower"]

        if free_speed:
            # Winch force capability band for the free-speed inequalities.
            _free_tf_hi = float(radial_params["max_tether_force"])
            _free_tf_lo = float(radial_params.get("min_tether_force", 0.0))

        # Per-node expressions accumulated in the loop for the constraint report.
        _rep_aoa = []
        _rep_rates = {ctrl.name: [] for ctrl in node_controls}
        _rep_trim = []
        _rep_continuity = []
        _rep_tension = []  # force_law: scaled residuals; free_speed: node tension
        _rep_turn_radius = []  # node-wise turn radius r/|kappa| [m] (k = 0 samples)
        _rep_turn_radius_dense = []  # all sub-samples; the constrained set when active
        _rep_sigma = []  # parametric speed sigma'/sigma_ref at the samples (floor rows)

        for i in range(N):

            # Per-node argument bundle: in profile mode swap the trailing depower
            # symbol for this node's decision; otherwise reuse the shared bundle.
            if optimize_depower_profile:
                node_syms = list(flat_syms)
                node_syms[-1] = opti_vars["input_depower"][i]
                node_depower = opti_vars["input_depower"][i]
            else:
                node_syms = flat_syms
                # Scalar-optimized depower lives on the model symbol; a fixed
                # depower is the numeric sim value. Either way this feeds the
                # depower-dependent winch offset (Winch.tension_curve).
                node_depower = (
                    km_copy.input_depower
                    if "input_depower" in opti_params
                    else sim_params["input_depower"]
                )

            # Local-support mode: this node's (and its sub-samples') functions
            # come from the knot interval the sample falls in.
            if local_support:
                residual_i_fn, aoa_i_fn, _, _ = _node_functions(_knot_of(s_grid[i]))
            else:
                residual_i_fn, aoa_i_fn = residual, aoa_eq

            # The node tension is the decision variable itself, fed straight
            # into the trim residual below -- the radial force balance is what
            # determines it (same square per-node structure as the forward
            # root-find: 3x3 in free_speed, 4x4 with the winch law). Do NOT
            # route it through a tension_tether_equation substitution: that
            # pins the variable only via the weak alpha(T) pendulum term,
            # leaving a near-null direction free to wander tens of kN off the
            # physical value.
            T_i = opti_vars["tension_tether_ground"][i]
            if free_speed:
                # v_r is a direct (rate-limited) control; the winch imposes no
                # tension curve, only the force band it can physically hold.
                # Inequalities instead of the curve equality: no soft-clamp
                # plateau, no rank loss when riding the force limit.
                opti.subject_to(T_i / S["T"] >= _free_tf_lo / S["T"])
                opti.subject_to(T_i / S["T"] <= _free_tf_hi / S["T"])
                _rep_tension.append(T_i)
            else:
                # The winch force law ties the tension to the reel speed.
                T_model = winch_model.tension_curve(
                    opti_vars["speed_radial"][i], input_depower=node_depower
                )

                # Scale the tether law residual
                opti.subject_to((T_i - T_model) / S["T"] == 0)
                _rep_tension.append((T_i - T_model) / S["T"])

            # Residual equations (scaled)
            res_i = residual_i_fn(
                s_grid[i],
                opti_vars["s_dot"][i],
                opti_vars["input_steering"][i],
                T_i,
                opti_vars["speed_radial"][i],
                opti_vars["distance_radial"][i],
                *node_syms,
            )
            opti.subject_to(res_i[0] / S_res[0] == 0)
            opti.subject_to(res_i[1] / S_res[1] == 0)
            opti.subject_to(res_i[2] / S_res[2] == 0)
            _rep_trim += [res_i[j] / S_res[j] for j in range(3)]

            # Left-rule dt_i = Δs_i / s_dot[i], guarded to avoid blow-up
            if i < N - 1:
                ds_i = s_grid[i + 1] - s_grid[i]
                # sd_safe = ca.fmax(opti_vars["s_dot"][i], max(sdot_min, S["sd"] * 1e-3))
                sd_safe = opti_vars["s_dot"][i]
                dt_i = ds_i / sd_safe

                # r_{i+1} propagation (scaled residual)
                r_cont_i = (
                    opti_vars["distance_radial"][i + 1]
                    - opti_vars["distance_radial"][i]
                    - opti_vars["speed_radial"][i] * dt_i
                ) / S["r"]
                opti.subject_to(r_cont_i == 0)
                _rep_continuity.append(r_cont_i)
                # Per-node control slew-rate limits, so the optimized profiles
                # stay physically actuatable. Data-driven over node_controls;
                # emits the same constraints in the same order (steering, then
                # depower) as the previous hand-written blocks.
                for ctrl in node_controls:
                    u_ctrl = opti_vars[ctrl.name]
                    ctrl_rate = (u_ctrl[i + 1] - u_ctrl[i]) / dt_i
                    opti.subject_to(ctrl_rate <= ctrl.rate_limit[1])
                    opti.subject_to(ctrl_rate >= ctrl.rate_limit[0])
                    _rep_rates[ctrl.name].append(ctrl_rate)
                # Accumulate energy and time: power_i = T_i * v_r_i
                energy += T_i * opti_vars["speed_radial"][i] * dt_i
                t_eff += dt_i

                # LImit angle of attack
                aoa_i = aoa_i_fn(
                    s_grid[i],
                    opti_vars["s_dot"][i],
                    opti_vars["input_steering"][i],
                    T_i,
                    opti_vars["speed_radial"][i],
                    opti_vars["distance_radial"][i],
                    *node_syms,
                )
                opti.subject_to(aoa_i <= limits["angle_of_attack"][1])
                opti.subject_to(aoa_i >= limits["angle_of_attack"][0])
                _rep_aoa.append(aoa_i)

            # Turn radius of the pattern over this node's interval: R = r_i /
            # |kappa(s)| with kappa the unit-sphere geodesic curvature (node
            # states held; s_grid has N+1 entries so s_grid[i+1] exists for
            # every node). k = 0 is the node itself (diagnostic column); the
            # sub-samples feed the ``turn_radius_min`` diagnostic always and
            # add constraint rows only when the constraint is active.
            n_sub = turn_radius_subsamples
            for k in range(n_sub):
                s_ik = s_grid[i] + (k / n_sub) * (s_grid[i + 1] - s_grid[i])
                curvature_ik_fn = (
                    _node_functions(_knot_of(s_ik))[2]
                    if local_support
                    else curvature_eq
                )
                kappa_ik, kappa_num_ik, dsigma_ik = curvature_ik_fn(
                    s_ik,
                    opti_vars["s_dot"][i],
                    opti_vars["input_steering"][i],
                    T_i,
                    opti_vars["speed_radial"][i],
                    opti_vars["distance_radial"][i],
                    *node_syms,
                )
                R_ik = opti_vars["distance_radial"][i] / ca.sqrt(kappa_ik**2 + 1e-6)
                if k == 0:
                    _rep_turn_radius.append(R_ik)
                _rep_turn_radius_dense.append(R_ik)
                if min_turn_radius > 0.0:
                    # -r sigma'^3 <= R_min kappa sigma'^3 <= r sigma'^3, i.e.
                    # r/|kappa| >= R_min without the 1/sigma'^3 blow-up; scaled
                    # by the radius scale and the typical sigma'^3 to stay O(1).
                    lhs = min_turn_radius * kappa_num_ik / _turn_scale
                    rhs = distance_scaled[i] * dsigma_ik**3 / _sigma_ref**3
                    opti.subject_to(lhs <= rhs)
                    opti.subject_to(-lhs <= rhs)
                    if turn_radius_sigma_floor > 0.0:
                        # No hesitation points: keeps the rows above meaningful
                        # and the returned point set free of micro-kinks.
                        opti.subject_to(
                            dsigma_ik / _sigma_ref >= turn_radius_sigma_floor
                        )
                        _rep_sigma.append(dsigma_ik / _sigma_ref)

        power = energy / (t_eff + 1e-12)

        # --- Control-smoothness regularizer (node-to-node change of the
        # controls), normalized by each control's bound width so it is
        # dimensionless and grid-independent. It is added to the objective in
        # ``Phase.run_simulation_opti`` weighted by ``sim_parameters["reg_weight"]``
        # (0 by default -> objective unchanged). The power objective is a flat
        # ridge -- many trajectories share essentially the same power -- so the
        # bare problem has a non-unique optimum and any solver-path change moves
        # the result. Among those equal-power solutions this term selects the
        # SMOOTHEST one, pinning a unique, reproducible optimum. It penalizes the
        # CONTROLS only: ``s_dot``/``speed_radial`` produce power, so penalizing
        # their magnitude (as the old, never-wired-in term did) would fight the
        # objective.
        def _bound_width(name, fallback=1.0):
            lims = limits.get(name)
            return float(lims[1] - lims[0]) if lims and lims[1] > lims[0] else fallback

        n_int = max(N - 1, 1)
        reg = 0
        for ctrl in node_controls:
            reg = reg + ca.sumsqr(
                ca.diff(opti_vars[ctrl.name]) / _bound_width(ctrl.name)
            )
        reg = reg / n_int

        # --- Initials for optimization parameters
        for var, mx in opti_params.items():

            if var in self.pattern_config["path_parameters"]:
                init_val = self.pattern_config["path_parameters"][var]

                opti.set_initial(mx, init_val)
                # print(f"Applying constraints for {var}")
                lb, ub = limits[var]
                opti.subject_to(mx >= lb)
                opti.subject_to(mx <= ub)

            elif var in self.pattern_config["radial_parameters"]:
                init_val = self.pattern_config["radial_parameters"][var]
                opti.set_initial(mx, init_val)
                # print(f"Setting initial for {var} to {init_val}")

                lb, ub = limits[var]
                # print(f"Applying constraints for {var}: lb={lb}, ub={ub}")
                opti.subject_to(mx >= lb)
                opti.subject_to(mx <= ub)
            elif var in self.pattern_config["sim_parameters"]:
                init_val = self.pattern_config["sim_parameters"][var]
                opti.set_initial(mx, init_val)
                print(f"Applying constraints for {var}, sim param initial {init_val}")
                lb, ub = limits[var]
                opti.subject_to(mx >= lb)
                opti.subject_to(mx <= ub)
            else:
                continue

        # Hard per-solve trust region on the optimized parameters:
        # ``sim_parameters["param_step_bound"]`` maps parameter name -> max
        # |change| from its warm-start (current config) value in THIS solve.
        # The power objective is a flat ridge and the periodic spline has
        # near-null reparametrization directions, so once the barrier drops
        # the solver takes huge steps (||d|| ~ 1e2) in the shape coefficients
        # that re-topologize the path (figure-eight loops merge or vanish) and
        # strand the iterate outside the trim-feasible region. Capping the
        # per-solve move keeps every iterate where the warm start's
        # linearization holds; staged solves re-center the box on the previous
        # optimum, so the shape can still travel far ACROSS stages.
        step_bounds = sim_params.get("param_step_bound") or {}
        for var, mx in opti_params.items():
            delta = step_bounds.get(var)
            if delta is None:
                continue
            for section in ("path_parameters", "radial_parameters", "sim_parameters"):
                if var in self.pattern_config.get(section, {}):
                    ref = ca.DM(self.pattern_config[section][var])
                    # Scalar delta: one box for the whole parameter. Vector
                    # delta: a per-element box (e.g. wider over the reel-in
                    # arc of a full-cycle spline); an element with delta 0 is
                    # frozen at its warm-start value.
                    delta_arr = np.asarray(delta, dtype=float).ravel()
                    if delta_arr.size == 1:
                        delta_box = float(delta_arr[0])
                        desc = f"+/-{delta_box:g}"
                    elif delta_arr.size == ref.numel():
                        delta_box = ca.DM(delta_arr)
                        desc = (
                            f"per-element +/-[{delta_arr.min():g}"
                            f"..{delta_arr.max():g}]"
                        )
                    else:
                        raise ValueError(
                            f"param_step_bound[{var!r}] has {delta_arr.size} "
                            f"entries; expected a scalar or {ref.numel()}"
                        )
                    opti.subject_to(mx >= ref - delta_box)
                    opti.subject_to(mx <= ref + delta_box)
                    print(
                        f"Trust region: {var} bounded to {desc} "
                        "around its warm-start value"
                    )
                    break

        # --- Default limits for vector vars (if provided)
        for var_name, mx in opti_vars.items():
            if isinstance(mx, ca.MX) and var_name in limits:
                # print(f"Applying constraints for {var_name}")
                lb, ub = limits[var_name]
                if relax_tol > 0:
                    # expand bounds outward even if bounds are negative
                    lb = lb - relax_tol * np.abs(lb)
                    ub = ub + relax_tol * np.abs(ub)
                if var_name not in constraint_report:
                    _report_ineq(var_name, mx, (lb, ub))
                # Tension and radius are exposed as scaled expressions; bound the
                # underlying O(1) scaled decision so the bound row's Jacobian stays
                # O(1) instead of carrying a ~scale coefficient.
                if var_name == "tension_tether_ground":
                    opti.subject_to(lb / T_SCALE <= tension_scaled)
                    opti.subject_to(tension_scaled <= ub / T_SCALE)
                    continue
                if var_name == "distance_radial":
                    opti.subject_to(lb / R_SCALE <= distance_scaled)
                    opti.subject_to(distance_scaled <= ub / R_SCALE)
                    continue
                if mx.shape[0] == N:
                    opti.subject_to(lb <= mx[:])
                    opti.subject_to(mx[:] <= ub)
                else:
                    opti.subject_to(lb <= mx)
                    opti.subject_to(mx <= ub)

        angle_elevation = pattern.elevation(opti_vars["distance_radial"], s_grid[:-1])
        # Soft radial-cycle closure: squared mismatch between the final and initial
        # radius, on the O(1) scaled decisions so the penalty is well conditioned.
        # Weighted into the objective by ``sim_parameters["radial_closure_weight"]``
        # in ``Phase.run_simulation_opti`` (0 -> off). A gentler alternative to the
        # hard ``close_radial_cycle`` equality for stiff full-cycle solves: ramp the
        # weight up across staged solves to tighten closure without the hard row's
        # conditioning hit. ``node N-1`` is the last decision; node 0 is pinned to r0.
        # Remaining constraint-report rows: actuation/geometry expressions
        # accumulated in the node loop, then the equality groups (feasibility
        # diagnostics -- max |scaled residual| should sit at solver tolerance).
        if _rep_aoa:
            _report_ineq(
                "angle_of_attack",
                ca.vertcat(*_rep_aoa),
                limits["angle_of_attack"],
                "deg",
                180.0 / np.pi,
            )
        _rate_units = {
            "input_steering": "1/s",
            "input_depower": "m/s",
            "speed_radial": "m/s^2",
        }
        for ctrl in node_controls:
            if _rep_rates[ctrl.name]:
                _report_ineq(
                    f"{ctrl.name}_rate",
                    ca.vertcat(*_rep_rates[ctrl.name]),
                    ctrl.rate_limit,
                    _rate_units.get(ctrl.name, "/s"),
                )
        if _rep_tension and free_speed:
            _report_ineq(
                "winch_tension_band",
                ca.vertcat(*_rep_tension),
                (_free_tf_lo, _free_tf_hi),
                "N",
            )
        elif _rep_tension:
            constraint_report["winch_tension_law (scaled)"] = {
                "expr": ca.vertcat(*_rep_tension),
                "equality": True,
            }
        if _rep_trim:
            constraint_report["trim_residual (scaled)"] = {
                "expr": ca.vertcat(*_rep_trim),
                "equality": True,
            }
        if _rep_continuity:
            constraint_report["radial_continuity (scaled)"] = {
                "expr": ca.vertcat(*_rep_continuity),
                "equality": True,
            }

        if min_turn_radius > 0.0 and _rep_turn_radius_dense:
            _report_ineq(
                "turn_radius",
                ca.vertcat(*_rep_turn_radius_dense),
                (min_turn_radius, np.inf),
                "m",
            )
        if _rep_sigma:
            _report_ineq(
                "path_param_speed (sigma'/ref)",
                ca.vertcat(*_rep_sigma),
                (turn_radius_sigma_floor, np.inf),
                "-",
            )

        radial_closure = (distance_scaled[-1] - distance_scaled[0]) ** 2
        objective_dict = {
            "energy": energy,
            "total_time": t_eff,
            "power_scale": P_scale,
            "reg": reg,
            "radial_closure": radial_closure,
            "angle_elevation_start": angle_elevation[0],
            "angle_elevation_end": angle_elevation[-1],
            "constraint_report": constraint_report,
            # Per-node physical turn radius r/|kappa| [m] (N entries) and the
            # tightest radius over the sub-sampled path (scalar); evaluated at
            # the solution by ``Phase.run_simulation_opti`` into the optimized
            # trajectory (``turn_radius`` column) and ``turn_radius_min``.
            "turn_radius": ca.vertcat(*_rep_turn_radius),
            "turn_radius_min": ca.mmin(ca.vertcat(*_rep_turn_radius_dense)),
        }
        return (
            opti,
            opti_vars,
            objective_dict,
        )

    def substitute_parametrized_kinematics(self, pattern, quasi_steady=None):

        quasi_steady = self.quasi_steady if quasi_steady is None else quasi_steady
        kinematics = ParametrizedKinematics(pattern, self)

        km_copy = copy.deepcopy(self.kite_model)

        km_copy.angle_course = kinematics.chi
        # Optimal analytical solution for speed_radial should be part of the pattern class
        # km_copy.speed_radial = km_copy.speed_radial
        # print(km_copy.speed_radial)
        # km_copy.speed_radial = kinematics.vr
        km_copy.speed_tangential = kinematics.vtau
        km_copy.timeder_angle_course = kinematics.dot_chi
        if not quasi_steady:
            km_copy.timeder_speed_radial = kinematics.dot_vr
            km_copy.timeder_speed_tangential = kinematics.dot_vtau
        else:
            km_copy.timeder_speed_radial = 0
            km_copy.timeder_speed_tangential = 0

        km_copy.angle_azimuth = kinematics.phi
        km_copy.angle_elevation = kinematics.beta

        return km_copy

    def _rigid_warm_start(
        self, pattern, state_obj, s_grid, u_dep_profile, wind_profile
    ):
        """Solve the QS problem at the first node with a ``RigidLumpedTether``
        clone of ``self.kite_model``. Returns a dict with the converged
        ``tension_tether_ground``, ``input_steering``, ``s_dot``,
        ``speed_radial`` — used to seed the Williams kite-end tension and the
        kite-state decisions before the Williams solver runs.

        Returns ``None`` if the rigid solve doesn't converge (in which case
        callers fall back to the hook-default seeds).
        """
        from awetrim.system.tether import RigidLumpedTether

        original_kite_model = self.kite_model
        rigid_kite_model = copy.deepcopy(original_kite_model)
        # Swap to rigid tether using same diameter/density as the Williams one.
        diameter = original_kite_model.tether.diameter_tether
        density = original_kite_model.tether.density_tether
        rigid_kite_model.tether = RigidLumpedTether(diameter=diameter, density=density)
        rigid_kite_model.reset_solver()

        # Temporarily swap; residual_solver and substitute_parametrized_kinematics
        # both read self.kite_model.
        self.kite_model = rigid_kite_model
        try:
            rigid_km = self.substitute_parametrized_kinematics(pattern)
            rigid_solver = self.residual_solver(rigid_km)

            if self.quasi_steady:
                s_dot_init = state_obj.s_dot if state_obj.s_dot is not None else 1.0
                tension_init = (
                    state_obj.tension_tether_ground
                    if state_obj.tension_tether_ground is not None
                    else 1.0e4
                )
                z0 = ca.vertcat(
                    tension_init,
                    state_obj.input_steering or 0.0,
                    max(s_dot_init, 0.1),
                    state_obj.speed_radial or 0.0,
                )
                p = ca.vertcat(
                    s_grid[0],
                    state_obj.distance_radial,
                    u_dep_profile[0],
                    wind_profile[0],
                )
                unknown_vars = [
                    "tension_tether_ground",
                    "input_steering",
                    "s_dot",
                    "speed_radial",
                ]
            else:
                # Dynamic case: warm-start not implemented yet; the dynamic
                # solver has its own first-state alignment via
                # _quasi_steady_start_state.
                return None

            lbx, ubx, lbg, ubg = self.get_boundaries(state_obj, unknown_vars, rigid_km)
            sol = rigid_solver(x0=z0, p=p, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
            if not rigid_solver.stats().get("success", False):
                logger.warning(
                    "Rigid warm-start for Williams did not converge "
                    "(status=%s). Falling back to hook-default seeds.",
                    rigid_solver.stats().get("return_status"),
                )
                return None
            x_opt = np.asarray(sol["x"]).reshape(-1)
            return {
                "tension_tether_ground": float(x_opt[0]),
                "input_steering": float(x_opt[1]),
                "s_dot": float(x_opt[2]),
                "speed_radial": float(x_opt[3]),
            }
        finally:
            self.kite_model = original_kite_model

    def integrator(self, time_step, kite_model=None):
        if kite_model is None:
            kite_model = self.kite_model
        kite_model.establish_residual()
        if self.quasi_steady:
            x = ca.vertcat(self.s, kite_model.distance_radial)
            if kite_model.is_tether_rigid:
                z = ca.vertcat(
                    kite_model.tension_tether_ground,
                    kite_model.input_steering,
                    self.s_dot,
                )
            else:
                z = ca.vertcat(
                    kite_model.length_tether,
                    kite_model.input_steering,
                    self.s_dot,
                )
            ode = ca.vertcat(
                self.s_dot,
            )

        else:
            x = ca.vertcat(
                self.s,
                self.s_dot,
                kite_model.distance_radial,
            )
            if kite_model.is_tether_rigid:
                z = ca.vertcat(
                    kite_model.tension_tether_ground,
                    kite_model.input_steering,
                    self.s_ddot,
                )
            else:
                z = ca.vertcat(
                    kite_model.length_tether,
                    kite_model.input_steering,
                    self.s_ddot,
                )

            ode = ca.vertcat(
                self.s_dot,
                self.s_ddot,
            )

        alg = kite_model.residual
        alg = ca.vertcat(
            alg,
            self.winch_model.radial_equation(
                tension_tether_ground=kite_model.tension_tether_ground,
                speed_radial=kite_model.speed_radial,
                input_depower=kite_model.input_depower,
            ),
        )
        z = ca.vertcat(z, kite_model.speed_radial)
        ode = ca.vertcat(ode, kite_model.speed_radial)

        dae = {"x": x, "z": z, "ode": ode, "alg": alg}
        # Create the integrator
        opts = {
            "abstol": 1e-6,
            "reltol": 1e-6,
            # "max_num_steps": 20000,
            # "max_step_size": 0.01,  # Or even 1e-3 if very stiff
        }

        # intg = ca.integrator("intg", "idas", dae, opts)
        intg = ca.integrator("intg", "idas", dae, 0, time_step, opts)
        return intg

    # Scale used for the internal Williams kite-end tension decision. The
    # decision variable in the NLP is ``tension_kite_scaled = T / scale``, so
    # IPOPT's Newton steps on it live in the same numerical range as the
    # angle/length decisions (~1-100) instead of the 1e3-1e6 range of the raw
    # tension. Conversion happens by substituting
    # ``tether.tension_tether_kite -> tension_kite_scaled * WILLIAMS_TENSION_SCALE``
    # in the algebraic before the NLP is built.
    WILLIAMS_TENSION_SCALE = 1.0e4

    def residual_solver(self, km_copy=None, quasi_steady=None):
        quasi_steady = self.quasi_steady if quasi_steady is None else quasi_steady
        if km_copy is None:
            km_copy = self.kite_model

        wind_speed_ref = ca.MX.sym("speed_wind_ref")
        km_copy.wind.speed_wind_ref = wind_speed_ref
        km_copy.establish_residual()
        use_williams = isinstance(km_copy.tether, WilliamsTether)
        # Scaled tension decision variable for Williams; substituted into the
        # algebraic below as ``tension_kite_scaled * WILLIAMS_TENSION_SCALE``.
        tension_kite_scaled = ca.MX.sym("tension_kite_scaled") if use_williams else None
        if quasi_steady:
            if use_williams:
                z = ca.vertcat(
                    km_copy.input_steering,
                    self.s_dot,
                    km_copy.speed_radial,
                    tension_kite_scaled,
                )
                p = ca.vertcat(
                    self.s,
                    km_copy.distance_radial,
                    km_copy.input_depower,
                    wind_speed_ref,
                )
            elif km_copy.is_tether_rigid:
                z = ca.vertcat(
                    km_copy.tension_tether_ground,
                    km_copy.input_steering,
                    self.s_dot,
                )
            else:
                z = ca.vertcat(
                    km_copy.length_tether,
                    km_copy.input_steering,
                    km_copy.s_dot,
                )
            if not use_williams:
                p = ca.vertcat(
                    self.s,
                    km_copy.distance_radial,
                    km_copy.input_depower,
                    wind_speed_ref,
                )

        else:
            if use_williams:
                # Mirror the QS Williams layout, with s_ddot in place of s_dot.
                # Ground tension is derived from the shape (not a free symbol),
                # so it is NOT a decision here (the old code put the derived
                # expression in z[0], which broke nlpsol).
                z = ca.vertcat(
                    km_copy.input_steering,
                    self.s_ddot,
                    km_copy.speed_radial,
                    tension_kite_scaled,
                )
            elif km_copy.is_tether_rigid:
                z = ca.vertcat(
                    km_copy.tension_tether_ground,
                    km_copy.input_steering,
                    self.s_ddot,
                )
            else:
                z = ca.vertcat(
                    km_copy.length_tether,
                    km_copy.input_steering,
                    self.s_ddot,
                )
            p = ca.vertcat(
                self.s,
                self.s_dot,
                km_copy.distance_radial,
                km_copy.input_depower,
                wind_speed_ref,
            )

        alg = km_copy.residual
        williams_force_scale = max(
            float(
                self.pattern_config.get("radial_parameters", {}).get(
                    "max_tether_force", 1e4
                )
            ),
            1.0,
        )
        williams_length_scale = ca.fmax(km_copy.distance_radial, 1.0)
        if use_williams:
            tether = km_copy.tether

            force_aero = km_copy.kite.force_aerodynamic(km_copy)
            force_gravity = km_copy.kite.force_gravity_for(km_copy)
            force_inertial = -km_copy.kite.mass_wing * km_copy.acceleration
            force_kite_course = force_aero + force_gravity + force_inertial

            direction_wind = km_copy.wind.direction_wind
            T_Wind_from_C = transformation_Wind_from_C(
                km_copy.angle_azimuth,
                km_copy.angle_elevation,
                km_copy.angle_course,
                direction_wind,
            )
            T_C_from_Wind = transformation_C_from_Wind(
                km_copy.angle_azimuth,
                km_copy.angle_elevation,
                km_copy.angle_course,
                direction_wind,
            )
            r_kite_course = ca.vertcat(0.0, 0.0, km_copy.distance_radial)
            r_kite_wind = T_Wind_from_C @ r_kite_course
            force_kite_wind = T_Wind_from_C @ force_kite_course
            omega_wind = T_Wind_from_C @ km_copy.velocity_rotation_course_frame

            # The tether reads wind/rho/g off ``env`` (= km_copy). ``omega``
            # is the wind-frame rotation we just built.
            tether_shape = tether.tether_shape_symbolic(
                env=km_copy,
                r_kite=r_kite_wind,
                force_kite_resultant=force_kite_wind,
                tension_kite=tether.tension_tether_kite,
                omega=omega_wind,
            )
            tether_force_course = T_C_from_Wind @ tether_shape["tether_force_kite"]
            force_residual = force_kite_course + tether_force_course
            tension_ground = ca.norm_2(tether_shape["tether_force_ground"])
            if quasi_steady:
                alg = ca.vertcat(
                    force_residual[0] / williams_force_scale,
                    force_residual[1] / williams_force_scale,
                    force_residual[2] / williams_force_scale,
                    tether_shape["ground_position"] / williams_length_scale,
                    self.winch_model.radial_equation(
                        tension_tether_ground=tension_ground,
                        speed_radial=km_copy.speed_radial,
                        input_depower=km_copy.input_depower,
                    )
                    / williams_force_scale,
                )
            else:
                # Full 3D force balance + ground closure, same as the QS branch.
                # The inertial term in force_residual already carries the dynamic
                # (s_ddot / radial) accelerations, so imposing all three force
                # components is correct here. The winch radial equation is
                # appended below for the dynamic case.
                alg = ca.vertcat(
                    force_residual[0] / williams_force_scale,
                    force_residual[1] / williams_force_scale,
                    force_residual[2] / williams_force_scale,
                    tether_shape["ground_position"] / williams_length_scale,
                )
        else:
            tension_ground = km_copy.tension_tether_ground
        # aoa = km_copy.angle_of_attack
        if not (use_williams and quasi_steady):
            alg = ca.vertcat(
                alg,
                (
                    self.winch_model.radial_equation(
                        tension_tether_ground=tension_ground,
                        speed_radial=km_copy.speed_radial,
                        input_depower=km_copy.input_depower,
                    )
                    / williams_force_scale
                    if use_williams
                    else self.winch_model.radial_equation(
                        tension_tether_ground=tension_ground,
                        speed_radial=km_copy.speed_radial,
                        input_depower=km_copy.input_depower,
                    )
                ),
                # km_copy.speed_radial,
            )
        if not use_williams:
            z = ca.vertcat(z, km_copy.speed_radial)
        if use_williams:
            z = ca.vertcat(
                z,
                km_copy.tether.elevation_last_element,
                km_copy.tether.azimuth_last_element,
                km_copy.tether.tether_length,
            )
            # Substitute the raw tension_tether_kite symbol with the scaled
            # decision: ``tension_kite_scaled * WILLIAMS_TENSION_SCALE``. This
            # is the heart of the conditioning fix — IPOPT's Newton steps on
            # the tension decision are now O(1) instead of O(1e4), so they no
            # longer dwarf the angle/length steps.
            scale = self.WILLIAMS_TENSION_SCALE
            t_kite_sym = km_copy.tether.tension_tether_kite
            alg = ca.substitute(alg, t_kite_sym, tension_kite_scaled * scale)
            tension_ground = ca.substitute(
                tension_ground, t_kite_sym, tension_kite_scaled * scale
            )
            z_names = {sym.name() for sym in ca.symvar(z)}
            p_names = {sym.name() for sym in ca.symvar(p)}
            missing_symbols = [
                sym
                for sym in ca.symvar(alg)
                if sym.name() not in z_names and sym.name() not in p_names
            ]
            if missing_symbols:
                z = ca.vertcat(z, *missing_symbols)
            # Diagnostic: every decision-vector entry must be a pure symbol or
            # nlpsol fails with "Argument 0(x) is not symbolic". Name any that
            # are constants/expressions so the offending state is obvious.
            non_sym = [k for k in range(z.numel()) if not z[k].is_symbolic()]
            if non_sym:
                print("[residual_solver] non-symbolic decision-vector entries:")
                for k in non_sym:
                    print(f"    z[{k}] = {z[k]}")
        nlp = {
            "x": z,
            "f": 0,
            "g": alg,
            "p": p,
        }
        if use_williams:
            self._williams_tension_ground_function = ca.Function(
                "williams_tension_tether_ground",
                [z, p],
                [tension_ground],
                ["x", "p"],
                ["tension_tether_ground"],
            )
        max_iter = 100 if use_williams else 50
        # When ``sim_parameters.debug_solver = True`` is set, bump IPOPT to
        # print_level=5 (per-iteration trail) so we can watch the iterate
        # trajectory at node 0.
        debug_solver = bool(
            self.pattern_config.get("sim_parameters", {}).get("debug_solver", False)
        )
        ipopt_print_level = 5 if debug_solver else 0
        ipopt_sb = "no" if debug_solver else "yes"
        ipopt_opts = {
            "print_level": ipopt_print_level,
            "max_iter": max_iter,
            "tol": 1e-8,
            "constr_viol_tol": 1e-8,
            "acceptable_tol": 1e-6,
            "acceptable_iter": 5,
            "sb": ipopt_sb,
        }
        if use_williams:
            # The Williams NLP mixes variables of very different orders
            # (angles ~1, lengths ~100, tension ~1e4-1e6). Keep the default
            # gradient-based scaling (equilibration-based requires HSL which
            # isn't bundled on Windows), but lower the barrier init and use
            # the adaptive mu strategy so IPOPT doesn't take huge Newton
            # steps on the tension variable.
            ipopt_opts.update(
                {
                    "mu_init": 1.0e-3,
                    "mu_strategy": "adaptive",
                }
            )
        solver_options = {
            "ipopt": ipopt_opts,
            "print_time": False,
        }

        return ca.nlpsol("solver", "ipopt", nlp, solver_options)

    def get_boundaries(self, state_obj, unknown_vars, km_copy):
        lbx, ubx, lbg, ubg = km_copy.get_boundaries(state_obj, unknown_vars)
        return lbx, ubx, lbg, ubg


import casadi as ca
import numpy as np


def register_opti_vars(obj, store=None, *, name_prefix=None):
    """
    Recursively scan `obj` (dict/list/tuple/numpy/MX) for CasADi MX symbols
    and add their leaf variables (via ca.symvar) to `store` exactly once.

    Parameters
    ----------
    obj : any
        Container (dict/list/tuple/ndarray) or MX expression/symbol.
    store : dict | None
        Mapping name -> MX to update (created if None).
    name_prefix : str | None
        If set, only add variables whose .name() starts with this prefix (e.g. "opti").

    Returns
    -------
    dict : updated store
    """
    if store is None:
        store = {}

    def _scan(x):
        # Base cases
        if isinstance(x, ca.MX):
            # collect leaf symbols from the expression/symbol
            for v in ca.symvar(x):
                nm = v.name()
                if (
                    name_prefix is None or nm.startswith(name_prefix)
                ) and nm not in store:
                    store[nm] = v
            return

        # Recurse into common containers
        if isinstance(x, dict):
            for v in x.values():
                _scan(v)
        elif isinstance(x, (list, tuple, set)):
            for v in x:
                _scan(v)
        elif isinstance(x, np.ndarray):
            for v in x.flat:
                _scan(v)
        # else: ignore scalars/others

    _scan(obj)
    return store
