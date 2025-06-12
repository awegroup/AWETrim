from picawe.timeseries.timeseries import TimeSeries
from picawe.kinematics.parametrized_patterns import create_pattern_from_dict
from picawe import SystemModel
from picawe.kinematics.Kinematics import ParametrizedKinematics
import casadi as ca
import numpy as np
from picawe.utils.defaults import DEFAULT_PATTERN_CONFIG, DEFAULT_OPTI_LIMITS
import copy
from picawe.system.tether import RigidLinkTether
from picawe import State
from picawe.system.kite import Kite


class ReelinPhase(TimeSeries):
    def __init__(
        self,
        kite_model: SystemModel,
    ):
        """
        Args:

        """

        super().__init__(
            kite_model=kite_model,
        )

        self.kite_model = kite_model

    def run_simulation(self, start_state, settings):
        """
        Simulate reeling-in cycle: RORI -> REEL-IN -> RIRO
        """

        # --- Configuration ---
        reeling_acceleration = 2  # [m/s²]
        time_step = settings["time_step"]
        max_depower = 1.0
        min_depower = 0.0
        depower_rate = 0.3  # depower change per second

        # --- Initialise integrator and state ---
        intg = self.integrator(time_step)
        start_state = self.kite_model.solve_quasi_steady(start_state)

        # --- State variables ---
        if self.kite_model.quasi_steady:
            x0 = [
                start_state.distance_radial,
                start_state.angle_elevation,
                start_state.angle_azimuth,
                start_state.angle_course,
            ]
            p = [
                start_state.timeder_angle_course,
                start_state.input_depower,
                start_state.speed_radial,
            ]
            z0 = [
                start_state.speed_tangential,
                start_state.timeder_angle_course,
                start_state.tension_tether_ground,
            ]
        else:
            x0 = [
                start_state.distance_radial,
                start_state.angle_elevation,
                start_state.angle_azimuth,
                start_state.speed_tangential,
                start_state.angle_course,
            ]
            p = [
                start_state.timeder_angle_course,
                start_state.input_depower,
                start_state.speed_radial,
            ]
            z0 = [
                0,
                start_state.input_steering,
                start_state.tension_tether_ground,
            ]

        t = start_state.t
        self.states = []
        phase = "rori"

        for i in range(10000):
            try:
                sol = intg(x0=x0, z0=z0, p=p)
            except Exception as e:
                print(f"Integration error at iteration {i}: {e}")
                break

            xf, zf = sol["xf"], sol["zf"]
            x0, z0 = xf, zf
            t += time_step

            full_state = self.assemble_full_state(xf, zf, p, t)
            self.states.append(full_state)

            if phase == "rori":
                phase = self.control_rori(full_state, p)
            elif phase == "reel-in":
                phase = self.control_reel_in(
                    full_state,
                    p,
                    settings,
                    time_step,
                    reeling_acceleration,
                    depower_rate,
                    max_depower,
                    min_depower,
                )
            elif phase == "riro":
                finished = self.control_riro(
                    full_state,
                    p,
                    settings,
                    start_state,
                    time_step,
                    reeling_acceleration,
                    depower_rate,
                    min_depower,
                )
                if finished:
                    break

    # ---------- CONTROL FUNCTIONS ----------

    def control_rori(self, full_state, p):
        if full_state["angle_course"] > 0.05:
            p[0] = -1
        elif full_state["angle_course"] < -0.05:
            p[0] = 1
        else:
            p[0] = 0
            return "reel-in"
        return "rori"

    def control_reel_in(
        self,
        full_state,
        p,
        settings,
        time_step,
        reeling_acceleration,
        depower_rate,
        max_depower,
        min_depower,
    ):
        if full_state["speed_radial"] > settings["control"]["reeling_speed"]:
            p[2] -= time_step * reeling_acceleration

        if full_state["angle_elevation"] > settings["control"]["min_elevation"]:
            p[1] = min(max_depower, p[1] + depower_rate * time_step)

        if full_state["angle_elevation"] > settings["control"]["max_elevation"]:
            print("Transition: angle_elevation max reached.")
            p[0] = -0.35
            return "riro"

        if full_state["tension_tether_ground"] < self.min_safe_tension():
            if p[1] > min_depower:
                print("Tension too low. Reducing depower.")
                p[1] = max(min_depower, p[1] - depower_rate * time_step)
            else:
                print("Tension too low. Forcing transition.")
                p[0] = -0.35
                return "riro"

        if full_state["speed_tangential"] < 0:
            print("Tangential speed negative. Forcing transition.")
            p[0] = -0.35
            return "riro"

        stop_threshold = settings["control"]["length_tether_ro"] + (
            (settings["control"]["reeling_speed"]) ** 2
        ) / (2 * reeling_acceleration)

        if full_state["distance_radial"] < stop_threshold:
            print("Transition: Distance threshold reached.")
            p[0] = -0.35
            return "riro"

        return "reel-in"

    def control_riro(
        self,
        full_state,
        p,
        settings,
        start_state,
        time_step,
        reeling_acceleration,
        depower_rate,
        min_depower,
    ):
        p[1] = max(min_depower, p[1] - depower_rate * time_step)

        if full_state["angle_elevation"] < settings["control"]["riro_elevation"] + 0.13:
            if self.wrap_to_pi(full_state["angle_course"]) > np.pi / 2:
                p[0] = -1
                print("Course angle too high. Adjusting course.")
            else:
                p[0] = 0
            p[2] = min(
                start_state.speed_radial, p[2] + time_step * reeling_acceleration
            )
        else:
            if full_state["angle_course"] < -np.pi:
                p[0] = -0.5
            if (
                full_state["speed_radial"] < 0
                and full_state["distance_radial"]
                < settings["control"]["length_tether_ro"]
                and full_state["angle_elevation"] > settings["control"]["min_elevation"]
            ):
                p[2] += time_step * reeling_acceleration
            elif (
                full_state["distance_radial"] < settings["control"]["length_tether_ro"]
            ):
                p[2] = 0

        if full_state["angle_azimuth"] > settings["control"]["riro_azimuth"]:
            print("Finished reeling-in phase.")
            return True
        return False

    # ---------- HELPER FUNCTIONS ----------

    def assemble_full_state(self, xf, zf, p, t):
        if self.kite_model.quasi_steady:
            return {
                "distance_radial": float(xf[0]),
                "angle_elevation": float(xf[1]),
                "angle_azimuth": float(xf[2]),
                "angle_course": float(xf[3]),
                "speed_tangential": float(zf[0]),
                "timeder_angle_course": float(p[0]),
                "tension_tether_ground": float(zf[2]),
                "t": t,
                "input_steering": float(zf[1]),
                "speed_radial": float(p[2]),
                "length_tether": float(xf[0]),
                "input_depower": float(p[1]),
            }
        else:
            return {
                "distance_radial": float(xf[0]),
                "angle_elevation": float(xf[1]),
                "angle_azimuth": float(xf[2]),
                "speed_tangential": float(xf[3]),
                "angle_course": float(xf[4]),
                "timeder_angle_course": float(p[0]),
                "tension_tether_ground": float(zf[2]),
                "t": t,
                "input_steering": float(zf[1]),
                "speed_radial": float(p[2]),
                "length_tether": float(zf[2]),
                "input_depower": float(p[1]),
                "timeder_speed_tangential": float(zf[0]),
            }

    def min_safe_tension(self):
        return 1.05 * (self.kite_model.mass_wing + self.kite_model.mass_kcu) * 9.81

    def wrap_to_pi(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def integrator(self, time_step, inputs=None):
        self.kite_model.timeder_speed_radial = 0
        if self.kite_model.ode is None:
            self.kite_model.establish_ode_function()
        if self.kite_model.algebraic is None:
            self.kite_model.establish_algebraic()

        if self.kite_model.quasi_steady:

            p = ca.vertcat(
                self.kite_model.timeder_angle_course,
                self.kite_model.input_depower,
                self.kite_model.speed_radial,
            )

            x = ca.vertcat(
                self.kite_model.state_vector[0:3], self.kite_model.state_vector[4]
            )
            ode = ca.vertcat(self.kite_model._ode[0:3], self.kite_model._ode[4])
            # p = ca.vertcat(self.timeder_angle_course, self.input_depower, self.speed_radial)
            if self.kite_model.is_tether_rigid:
                z = ca.vertcat(
                    self.kite_model.speed_tangential,
                    self.kite_model.input_steering,
                    self.kite_model.tension_tether_ground,
                )
            else:
                z = ca.vertcat(
                    self.kite_model.speed_tangential,
                    self.kite_model.input_steering,
                    self.kite_model.length_tether,
                )

            alg = self.kite_model.algebraic
            dae = {"x": x, "p": p, "z": z, "p": p, "ode": ode, "alg": alg}
            # Create the integrator
            opts = {
                "abstol": 1e-6,
                "reltol": 1e-6,
                "max_num_steps": 20000,
                "max_step_size": 0.01,  # Or even 1e-3 if very stiff
            }

            # intg = ca.integrator("intg", "idas", dae, opts)
            intg = ca.integrator("intg", "idas", dae, 0, time_step, opts)
            return intg

        else:
            p = ca.vertcat(
                self.kite_model.timeder_angle_course,
                self.kite_model.input_depower,
                self.kite_model.speed_radial,
            )

            x = ca.vertcat(self.kite_model.state_vector[0:5])
            ode = ca.vertcat(self.kite_model._ode[0:5])
            # p = ca.vertcat(self.timeder_angle_course, self.input_depower, self.speed_radial)
            if self.kite_model.is_tether_rigid:
                z = ca.vertcat(
                    self.kite_model.timeder_speed_tangential,
                    self.kite_model.input_steering,
                    self.kite_model.tension_tether_ground,
                )
            else:
                z = ca.vertcat(
                    self.kite_model.timeder_speed_tangential,
                    self.kite_model.input_steering,
                    self.kite_model.length_tether,
                )

            alg = self.kite_model.algebraic
            dae = {"x": x, "p": p, "z": z, "p": p, "ode": ode, "alg": alg}
            # Create the integrator
            opts = {
                "abstol": 1e-6,
                "reltol": 1e-6,
                "max_num_steps": 20000,
                "max_step_size": 0.01,  # Or even 1e-3 if very stiff
            }

            # intg = ca.integrator("intg", "idas", dae, opts)
            intg = ca.integrator("intg", "idas", dae, 0, time_step, opts)
            return intg

    def compute_course_target(self, azimuth, elevation, az_target, el_target):
        delta_chi = self.wrap_to_pi(az_target - azimuth)

        y = np.sin(delta_chi) * np.cos(el_target)
        x = np.cos(elevation) * np.sin(el_target) - np.sin(elevation) * np.cos(
            el_target
        ) * np.cos(delta_chi)

        psi_target = np.arctan2(y, x)
        return self.wrap_to_pi(psi_target)


def wrap_to_pi(angle):
    """Wrap angle to [-pi, pi]"""
    return (angle + np.pi) % (2 * np.pi) - np.pi
