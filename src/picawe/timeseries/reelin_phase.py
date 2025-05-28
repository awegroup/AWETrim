from picawe.timeseries.timeseries import TimeSeries
from picawe.kinematics.parametrized_patterns import  create_pattern_from_dict
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
        quasi_steady: bool = False,
    ):
        """
        Args:

        """

        super().__init__(
            kite_model=kite_model,
        )
        self.quasi_steady = quasi_steady
       
        self.kite_model = kite_model

    
    def run_simulation(self, start_state: State, settings: dict = None):
        """
        Simulate reeling-out followed by reeling-in transition of the kite.
        The depower is increased progressively until max during reel-out,
        and decreased progressively during transition to reel-in.
        """
        # --- Configuration ---
        reeling_acceleration = 2  # [m/s²] acceleration/deceleration of reeling
        time_step = settings["time_step"]
        max_depower = 1.0
        min_depower = 0.0
        depower_rate = 0.3  # depower change per second

        # --- Initialise integrator and state ---
        intg = self.kite_model.integrator(time_step)
        start_state = self.kite_model.solve_quasi_steady(start_state)

        # --- State variables ---
        x0 = [
            start_state.distance_radial,
            start_state.angle_elevation,
            start_state.angle_azimuth,
            start_state.angle_course,  # angle_course is the course angle
        ]
        p = [
            start_state.timeder_angle_course,  # input_steering (assumed constant)
            start_state.input_depower,
            start_state.speed_radial,
        ]
        z0 = [
            start_state.speed_tangential,
            start_state.timeder_angle_course,
            start_state.tension_tether_ground,
        ]
        t = start_state.t
        self.states = []

        # --- Control flags ---
        transition = False
        riro = False  # reel-in roll-out condition met
        rori = True
        is_tension_low = False  # flag for low tension condition
        for i in range(10000):
            try:
                sol = intg(x0=x0, z0=z0, p=p)
            except Exception as e:
                print("Integration error:", e)
                break

            xf, zf = sol["xf"], sol["zf"]
            x0, z0 = xf, zf
            t += time_step

            # --- Assemble full state ---
            full_state = {
                "distance_radial": float(xf[0]),
                "angle_elevation": float(xf[1]),
                "angle_azimuth": float(xf[2]),
                "angle_course": float(xf[3]),
                "speed_tangential": float(zf[0]),
                "timeder_angle_course": float(p[0]),
                "tension_tether_ground": float(zf[2]),
                "t": t,
                "input_steering": float(zf[0]),
                "speed_radial": float(p[2]),
                "length_tether": float(xf[0]),
                "input_depower": float(p[1]),
            }

            self.states.append(full_state)

            if not transition and not riro and full_state["angle_course"] > 0.05:
                p[0] = -1
            elif not transition and not riro and full_state["angle_course"] < -0.05:
                p[0] = 1
            else:
                p[0] = 0
                rori = False  # reset rori when transition or riro is active

            # --- Exit condition ---
            if transition and full_state["angle_elevation"] < settings["control"]["riro_elevation"]+0.15 and not riro:
                self.states.pop()
                # x0[3] = np.pi/2
                x0[1] = self.states[-1]["angle_elevation"] 
                z0[0] = 100
                z0[2] = 1e8
                riro = True
                # Erase last state to avoid transition loop
                
                print("Transition to riro")
                
            if riro and full_state["angle_azimuth"] > settings["control"]["riro_azimuth"]:
                print("Finished reeling-in phase.")
                break

            # --- Progressive depower control ---
            if not transition and p[1] < max_depower and not is_tension_low and full_state["angle_elevation"] > settings["control"]["min_elevation"]:
                p[1] = min(max_depower, p[1] + depower_rate* time_step)

            if transition and p[1] > min_depower:
                p[1] = max(min_depower, p[1] - depower_rate* time_step)

            # --- Start transition at high elevation ---
            if not transition and full_state["angle_elevation"] > settings["control"]["max_elevation"] and not riro:
                print("Transition: angle_elevation max reached.")
                x0[3] = np.pi
                z0[0] = 100
                z0[2] = 1e8
                transition = True

            # --- Maintain course and course rate ---
            if riro:
                if full_state["angle_course"] > np.pi/2:
                    # x0[3] += -1.5* time_step # keep course at pi/2
                    p[0] = -1
                    z0[0] = 40
                    z0[2] = 1e5
                else:
                    x0[3] = np.pi/2  # keep course at pi/2
                    p[0] = 0
                # x0[3] = np.pi/2  # keep course at pi/2
                p[2] = min(start_state.speed_radial, p[2] + time_step * reeling_acceleration)
            elif not rori:
                x0[3] = np.pi if transition else 0
                
            z0[1] = 0

            # --- Stop reeling out when speed exceeds control threshold ---
            if not transition and not riro and full_state["speed_radial"] > settings["control"]["reeling_speed"]:
                p[2] -= time_step * reeling_acceleration  # decelerate

            # --- Ensure tension is above safe minimum ---
            min_tension = 1.05 * (self.kite_model.mass_wing + self.kite_model.mass_kcu) * 9.81
            if not transition and full_state["tension_tether_ground"] < min_tension:
                if p[1] > min_depower:
                    print("Tension too low. Reducing depower.")
                    p[1] = max(min_depower, p[1] - depower_rate * time_step)
                    is_tension_low = True
                else:
                    print("Tension too low. Forcing transition.")
                    x0[3] = np.pi
                    z0[0] = 100
                    z0[2] = 1e8
                    transition = True

            # --- Handle tangential stall (negative v_tau) ---
            if not transition and full_state["speed_tangential"] < 0 and not riro:
                print("Tangential speed negative. Forcing transition.")
                x0[2] = 0  # reset angle_azimuth
                x0[3] = np.pi
                z0[0] = 100
                z0[1] = 0
                p[0] = 0
                z0[2] = 1e8
                transition = True

           

            # --- Check if close to switching to reel-in based on kinematic constraint ---
            stop_threshold = settings["control"]["length_tether_ro"] + ((settings["control"]["reeling_speed"]) ** 2) / (2 * reeling_acceleration)
            if not transition and full_state["distance_radial"] < stop_threshold and not riro:
                print("Transition: Distance threshold reached.")
                x0[3] = np.pi
                z0[0] = 50
                z0[2] = 1e5
                transition = True

             # --- Transition: reel-in until target length reached ---
            if transition and not riro:
                if full_state["speed_radial"] < 0 and full_state["distance_radial"] < settings["control"]["length_tether_ro"] and full_state["angle_elevation"] > settings["control"]["min_elevation"]:
                    p[2] += time_step * reeling_acceleration
                    # p[0] = -1.5
                elif full_state["distance_radial"] < settings["control"]["length_tether_ro"]:
                    p[2] = 0