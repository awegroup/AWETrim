from picawe.timeseries.timeseries import TimeSeries
from picawe.kinematics.parametrized_patterns import ParametrizedPatterns
from picawe import SystemModel
from picawe.kinematics.Kinematics import ParametrizedKinematics
import casadi as ca
import numpy as np

class PhaseParameterized(TimeSeries):
    def __init__(
        self,
        kite_model: SystemModel,
        pattern: ParametrizedPatterns,
        quasi_steady: bool = False,
    ):
        """
        Args:

        """

        super().__init__(
            kite_model=kite_model,
        )
        self.pattern = pattern
        self.kinematics = ParametrizedKinematics(pattern, quasi_steady)
        self.substitute_parametrized_kinematics()
        self.quasi_steady = quasi_steady
        
    
    def run_simulation(self,start_state, s_array = None, time_array = None):
        if self.quasi_steady:
            unknown_vars = ["tension_tether_ground", "input_steering", "s_dot"]
        else:
            unknown_vars = ["tension_tether_ground", "input_steering", "s_ddot"]

        if self.kite_model.dof == 6:
            unknown_vars += ["angle_roll", "angle_pitch", "angle_yaw"]
        for var in unknown_vars:
            if var not in start_state:
                raise ValueError(f"Start state must contain {var}")
        qs_guess = [start_state[name] for name in unknown_vars]
        solve_func, inputs_name = self.kite_model.solve_quasi_steady_state(
            unknown_vars)
        current_state = start_state
        if time_array is not None:
            s = start_state["s"]
            s_dot = start_state["s_dot"]
            for i in range(len(time_array)-1):
                # print(i)
                p = [current_state[name] for name in inputs_name]
                lbx,ubx,lbg,ubg = self.kite_model.get_boundaries(unknown_vars)
                sol = solve_func(x0=qs_guess, p=p, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
                if sol["g"][0] < 1e-3:
                    # print('Sol found')
                    qs_guess = sol["x"]
                    qs_state = {name: float(sol["x"][i]) for i, name in enumerate(unknown_vars)}
                    time_step = time_array[i+1]-time_array[i]

                    if self.quasi_steady:
                        s += float(sol["x"][2])*time_step
                        current_state = {**qs_state, "s": s, "t": time_array[i]}
                    else:
                        dyn_state = {name: float(sol["x"][i]) for i, name in enumerate(unknown_vars)}
                        s_dot += float(sol["x"][2])*time_step
                        s += s_dot*time_step
                        current_state = {**dyn_state, "s_dot": s_dot, "s": s, "t": time_array[i]}
                    self.states.append(current_state)
                else:
                    print("Warning: Solver did not converge")
                    # qs_guess[2] = 0
                    # qs_guess[3] += -1
                    # s_dot = 0.5
                    # s+=np.radians(5)
                    # current_state = {**current_state, "s_dot": s_dot, "s": s, "t": time_array[i]}
                
                    
        


    def substitute_parametrized_kinematics(self):
        self.kite_model.angle_course = self.kinematics.chi
        self.kite_model.angle_elevation = self.kinematics.beta
        self.kite_model.speed_radial_sym = self.kite_model.speed_radial
        self.kite_model.speed_radial = self.kinematics.vr
        self.kite_model.speed_tangential = self.kinematics.vtau
        self.kite_model.timeder_angle_course = self.kinematics.dot_chi
        self.kite_model.timeder_speed_radial = self.kinematics.dot_vr
        self.kite_model.timeder_speed_tangential = self.kinematics.dot_vtau
        self.kite_model.distance_radial = self.kinematics.r
        self.kite_model.angle_azimuth = self.kinematics.phi
        self.kite_model.angle_elevation = self.kinematics.beta
        self.kite_model.s = self.kinematics.s
        self.kite_model.t = self.kinematics.t
        self.kite_model.s_dot = self.kinematics.s_dot
        self.kite_model.s_ddot = self.kinematics.s_ddot
