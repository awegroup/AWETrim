from matplotlib import pyplot as plt
from picawe.timeseries.timeseries import TimeSeries
from picawe.kinematics.my_parametrized_patterns import create_pattern_from_dict
from picawe import SystemModel
from picawe.kinematics.my_Kinematics import ParametrizedKinematics
import casadi as ca
import numpy as np
from picawe.utils.my_defaults import DEFAULT_SPLINE_PATTERN_CONFIG, DEFAULT_OPTI_LIMITS
import copy
from picawe.system.tether import RigidLinkTether
from picawe import State
from picawe.system.kite import Kite

import logging


logger = logging.getLogger(__name__)


class ReelinPhase(TimeSeries):
    def __init__(
        self,
        kite_model: SystemModel,
        quasi_steady: bool = False,
        pattern_config: dict = DEFAULT_SPLINE_PATTERN_CONFIG,
    ):
        """
        Args:

        """

        super().__init__(
            kite_model=kite_model,
        )
        self.pattern_config = pattern_config
        self.quasi_steady = quasi_steady

        self.kite_model = kite_model
        self.target_drag_coefficient = None
        self.target_lift_coefficient = None
        self.s = ca.MX.sym("s")
        self.t = ca.MX.sym("t")
        self.s_dot = ca.MX.sym("s_dot")
        self.s_ddot = ca.MX.sym("s_ddot")
        # self.find_optimal_angle_pitch_tether()

    def run_simulation(self, start_state, allow_failure=True, return_states=False):

        # Validate input state
        if isinstance(start_state, dict):
            state_obj = State(**start_state)
        else:
            state_obj = start_state

        # Check for NaN/inf values in initial state
        state_dict = state_obj.to_dict()
        for key, value in state_dict.items():
            if isinstance(value, (float, int)) and (np.isnan(value) or np.isinf(value)):
                print(f"ERROR: Invalid initial value {value} for {key}")
                if not allow_failure:
                    raise ValueError(f"Invalid initial state: {key}={value}")
                return
        # print("Initial state dictionary created successfully! \n")

        self.substitute_parametrized_kinematics()
        self.states = []
        self.kite_model.reset_solver()

        if self.quasi_steady:
            unknown_vars = ["length_tether", "input_steering", "s_dot"]
        else:
            unknown_vars = ["length_tether", "input_steering", "s_ddot"]

        if self.kite_model.is_tether_rigid:
            unknown_vars[0] = "tension_tether_ground"

        N = self.pattern_config["n_points"]
        time_step = self.pattern_config["end_time"] / self.pattern_config["n_points"]
        intg = self.integrator(time_step=time_step, inputs=None)

        try:
            new_state = self.kite_model.solve_quasi_steady(state_obj, unknown_vars)

            # Check if solve_quasi_steady returned None or invalid state
            if new_state is None:
                print("ERROR: solve_quasi_steady returned None")
                if not allow_failure:
                    raise RuntimeError("Initial quasi-steady solve failed")
                return

            # Validate the new state
            new_state_dict = new_state.to_dict()
            for key, value in new_state_dict.items():
                if isinstance(value, (float, int)) and (
                    np.isnan(value) or np.isinf(value)
                ):
                    print(
                        f"ERROR: solve_quasi_steady produced invalid value {value} for {key}"
                    )
                    if not allow_failure:
                        raise ValueError(f"Invalid solved state: {key}={value}")
                    return

        except Exception as e:
            print(f"ERROR in solve_quasi_steady: {e}")
            if not allow_failure:
                raise
            return

        print("New state:", new_state)

        if self.quasi_steady:
            x0 = [new_state.s, new_state.distance_radial, new_state.speed_radial]
            z0 = ca.vertcat(
                new_state.tension_tether_ground,
                new_state.input_steering,
                new_state.s_dot,
            )
        else:
            x0 = [
                new_state.s,
                new_state.s_dot,
                new_state.distance_radial,
                new_state.speed_radial,
            ]
            z0 = ca.vertcat(
                new_state.tension_tether_ground,
                new_state.input_steering,
                new_state.s_ddot,
            )
        self.states.append(new_state.to_dict())
        t = self.pattern_config["start_time"]
        input_depower = state_obj.input_depower
        p = [state_obj.timeder_speed_radial, input_depower]
        max_depower = self.pattern_config["parameters"].get("max_depower", 1)
        for i in range(N):
            print(f"Time: {t}, State: {x0}, Inputs: {z0}, Parameters: {p}")
            try:
                sol = intg(
                    x0=x0,
                    p=p,
                    z0=z0,
                )
            except Exception as e:
                print(f"Error occurred: {e}")
                if not allow_failure:
                    raise
                break
            xf = sol["xf"]
            zf = sol["zf"]
            if self.quasi_steady:
                new_state = State(
                    t=t,
                    s=xf[0],
                    input_steering=float(zf[1]),
                    tension_tether_ground=float(zf[0]),
                    s_dot=float(zf[2]),
                    distance_radial=float(xf[1]),
                    speed_radial=float(xf[2]),
                    input_depower=input_depower,
                )
            else:
                new_state = State(
                    t=t,
                    s=xf[0],
                    s_dot=float(xf[1]),
                    input_steering=float(zf[1]),
                    tension_tether_ground=float(zf[0]),
                    s_ddot=float(zf[2]),
                    distance_radial=float(xf[2]),
                    speed_radial=float(xf[3]),
                    input_depower=input_depower,
                )

            if new_state.speed_radial > self.pattern_config["parameters"].get(
                "min_vr", -4
            ) and (new_state.distance_radial > self.pattern_config["parameters"]["r1"]):
                ddot_vr = -self.kite_model.acceleration_winch
            elif (
                new_state.distance_radial < self.pattern_config["parameters"]["r1"]
            ) and (new_state.speed_radial < 1):
                ddot_vr = self.kite_model.acceleration_winch
            else:
                ddot_vr = 0

            if (input_depower < max_depower) and (ddot_vr < 0):
                input_depower += self.kite_model.depower_rate * time_step
            elif (ddot_vr > 0) and (input_depower > 0):
                input_depower -= self.kite_model.depower_rate * time_step

            if new_state.tension_tether_ground < self.kite_model.mass_wing * 9.81 * 1.5:
                input_depower -= self.kite_model.depower_rate * time_step

            p = [ddot_vr, input_depower]

            t += time_step
            x0 = xf
            z0 = zf

            # Recompute angles from updated s and r BEFORE appending to states
            try:
                pattern = create_pattern_from_dict(self.pattern_config)
                pattern_result = pattern.evaluate_spline(
                    new_state.distance_radial, new_state.s
                )
                new_state.angle_azimuth = float(pattern_result["S"][0])
                new_state.angle_elevation = float(pattern_result["S"][1])
            except Exception as e:
                print(f"Warning: Could not compute angles for step {i}: {e}")

            self.states.append(new_state.to_dict())

    def _flatten_for_function_call(vals):  
        flat = []
        for v in vals:
            if isinstance(v, ca.MX) and v.numel() > 1:
                # ensure column, then split into scalars
                flat += list(ca.vertsplit(ca.reshape(v, (-1, 1))))
            else:
                flat.append(v)
        return flat

    def substitute_parametrized_kinematics(self, optimize=False):  

        pattern = create_pattern_from_dict(self.pattern_config, optimize=optimize)
        # print(pattern.r0, pattern.r1)
        kinematics = ParametrizedKinematics(pattern, self)

        self.kite_model.s = kinematics.s
        self.kite_model.s_dot = kinematics.s_dot
        self.kite_model.s_ddot = kinematics.s_ddot

        self.kite_model.angle_course = kinematics.chi
        self.kite_model.angle_elevation = kinematics.beta
        # Optimal analytical solution for speed_radial should be part of the pattern class
        # self.kite_model.speed_radial = self.kite_model.speed_radial
        # print(self.kite_model.speed_radial)
        # self.kite_model.speed_radial = kinematics.vr
        self.kite_model.speed_tangential = kinematics.vtau

        # print("chi_dot from kinematics:", kinematics.dot_chi)
        self.kite_model.timeder_angle_course = kinematics.dot_chi

        if not self.quasi_steady:
            # self.kite_model.timeder_speed_radial = kinematics.dot_vr
            self.kite_model.timeder_speed_tangential = kinematics.dot_vtau

        self.kite_model.angle_azimuth = kinematics.phi
        self.kite_model.angle_elevation = kinematics.beta

        if optimize:
            return list(kinematics.pattern.optimization_vars.values()), pattern

    @property  
    def target_lift_coefficient(self):
        return self._target_lift_coefficient

    @target_lift_coefficient.setter  
    def target_lift_coefficient(self, value):
        self._target_lift_coefficient = value

    @property  
    def target_drag_coefficient(self):
        return self._target_drag_coefficient

    @target_drag_coefficient.setter  
    def target_drag_coefficient(self, value):
        self._target_drag_coefficient = value

    def integrator(self, time_step, inputs=None):
        self.kite_model.establish_residual()
        if self.quasi_steady:
            x = ca.vertcat(
                self.kite_model.s,
                self.kite_model.distance_radial,
                self.kite_model.speed_radial,
            )
            if self.kite_model.is_tether_rigid:
                z = ca.vertcat(
                    self.kite_model.tension_tether_ground,
                    self.kite_model.input_steering,
                    self.kite_model.s_dot,
                )
            else:
                z = ca.vertcat(
                    self.kite_model.length_tether,
                    self.kite_model.input_steering,
                    self.kite_model.s_dot,
                )

            ode = ca.vertcat(
                self.kite_model.s_dot,
                self.kite_model.speed_radial,
                self.kite_model.timeder_speed_radial,
            )
            alg = ca.vertcat(
                self.kite_model.residual,
            )
        else:
            x = ca.vertcat(
                self.kite_model.s,
                self.kite_model.s_dot,
                self.kite_model.distance_radial,
                self.kite_model.speed_radial,
            )
            if self.kite_model.is_tether_rigid:
                z = ca.vertcat(
                    self.kite_model.tension_tether_ground,
                    self.kite_model.input_steering,
                    self.kite_model.s_ddot,
                )
            else:
                z = ca.vertcat(
                    self.kite_model.length_tether,
                    self.kite_model.input_steering,
                    self.kite_model.s_ddot,
                )

            ode = ca.vertcat(
                self.kite_model.s_dot,
                self.kite_model.s_ddot,
                self.kite_model.speed_radial,
                self.kite_model.timeder_speed_radial,
            )
            alg = ca.vertcat(self.kite_model.residual)

        p = ca.vertcat(
            self.kite_model.timeder_speed_radial, self.kite_model.input_depower
        )
        dae = {"x": x, "z": z, "p": p, "ode": ode, "alg": alg}
        print("x:", x)
        print("z:", z)
        print("p:", p)
        # print("ode:", ode)
        # print("alg:", alg)
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


def smooth_gate_interval(s, s0, sf, eps=1e-2):  
    """
    Smooth gate active only on [s0, sf], 0 outside.
    eps controls the ramp width at both ends.
    Works with MX/SX.
    """
    # guard: sf must be > s0
    # (you can add an assert or handle wrap if you need periodic behavior)

    def ramp(u):
        # 0 for u<=0, 1 for u>=eps, smooth in between
        u_clip = ca.fmin(ca.fmax(u / eps, 0), 1)
        return 0.5 * (1 - ca.cos(ca.pi * u_clip))

    w_lo = ramp(s - s0)  # rises near s0
    w_hi = ramp(sf - s)  # falls near sf
    return w_lo * w_hi
