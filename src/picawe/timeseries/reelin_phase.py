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

    
    def run_simulation(self,start_state: State, settings: dict = None):
        """
        Run the simulation from start_state to end_time with a given time step.
        """
        reeling_acceleration = 2

        time_step = settings["time_step"]
        intg = self.kite_model.integrator(time_step)
        start_state = self.kite_model.solve_quasi_steady(start_state)
        x0 = [
            start_state.distance_radial,
            start_state.angle_elevation,
            0,
            0,
        ]
        p = [
            0,
            start_state.input_depower,
            start_state.speed_radial,
        ]
        z0 = [
            start_state.speed_tangential,
            start_state.timeder_angle_course,
            start_state.tension_tether_ground,
        ]
        self.states = []
        t = start_state.t
        transition = False
        riro = False
        for i in range(10000):
            try:
                sol = intg(x0=x0, z0=z0, p=p)
            except Exception as e:
                print("Integration error:", e)
                break
            xf = sol["xf"]
            zf = sol["zf"]
            
            x0 = xf
            z0 = zf

            t += time_step
            full_state = {
                "distance_radial": float(xf[0]),
                "angle_elevation": float(xf[1]),
                "angle_azimuth": float(xf[2]),
                "angle_course": float(xf[3]),
                "speed_tangential": float(zf[0]),
                "timeder_angle_course": float(zf[1]),
                "tension_tether_ground": float(zf[2]),
                "t": t,
                "input_steering": float(p[0]),
                "speed_radial": float(p[2]),
                "length_tether": float(xf[0]),
                "input_depower": float(p[1]),
            }
            
                

            if transition and full_state["angle_elevation"] < settings["control"]["riro_elevation"]:
                print("Finnished reeling in")
                break
            self.states.append(full_state)

            if full_state["angle_elevation"] > np.radians(80) and not transition:
                print("Transition")
                print("Angle_elevation max reached")
                x0[3] = np.pi
                z0[0] = 100
                z0[2] = 1e8
                transition = True

            if not riro and not transition and full_state["speed_radial"] > settings["control"]["reeling_speed"]:
                # Update the radial speed
                p[2] -= time_step * reeling_acceleration
            # Update the radial speed
            if transition:
                x0[3] = np.pi
                z0[1] = 0
            else:
                x0[3] = 0
                z0[1] = 0
            if full_state["tension_tether_ground"] < 1.05*(self.kite_model.mass_wing +self.kite_model.mass_kcu)* 9.81:
                if p[1] > 0.2:
                    print("Tension min reached")
                    print("Decreasing depower")
                    p[1] -= 0.2 
                else:
                    print("Minimum tension reached")
                    print("Minimum depower reached")
                    print("Transition")
                # p[1] = 0
                    x0[3] = np.pi
                    z0[0] = 100
                    z0[2] = 1e8
                    transition = True
            if not transition and full_state["speed_tangential"] < 0:
                print("Negative speed")
                # p[1] = 0
                x0[3] = np.pi
                z0[0] = 100
                z0[2] = 1e8
                transition = True
                
            
            if transition and full_state["speed_radial"] < 0 and full_state["distance_radial"] > settings["control"]["length_tether_ro"]:
                # Update the radial speed
                p[2] += time_step * reeling_acceleration
            if transition and full_state["distance_radial"] < settings["control"]["length_tether_ro"]:
                p[2] = 0

            if not transition and full_state["distance_radial"] < (settings["control"]["length_tether_ro"]+settings["control"]["reeling_speed"]**2/(2*reeling_acceleration)):
                print("Transition")
                print("Distance radial: ", full_state["distance_radial"])

                p[2] += time_step * reeling_acceleration
                p[1] -= 0.2
                x0[3] = np.pi
                z0[0] = 50
                z0[2] = 1e5
                transition = True


            

            

