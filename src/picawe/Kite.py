import casadi as ca
import numpy as np
from picawe.reference_frames import transformation_C_from_W, transformation_C_from_A, transformation_C_from_K
from picawe.Kinematics import KiteKinematics
from picawe.Tether import Tether
from picawe.Wind import Wind


class Wing:

    def __init__(self, mass_wing, area_wing, aero_input):
        """
        Initialize the kite system with its parameters.
        """
        self.mass_wing = mass_wing
        self.area_wing = area_wing
        self.define_symbolic_variables_wing()
        # Aerodynamic inputs
        self.angle_pitch_depower_0 = aero_input['params'].get("angle_pitch_depower_0", ca.SX.sym('angle_pitch_depower_0'))
        self.delta_pitch_depower = aero_input['params'].get("delta_pitch_depower", ca.SX.sym('delta_pitch_depower'))
        # self.aerodynamic_coeffs_function(aero_input)
        self.aero_input = aero_input
        
    def define_symbolic_variables_wing(self):
        """
        Define symbolic variables used in the model.
        """
        base_symbolic_variables = {
            'input_steering': 'input_steering',
            'input_depower': 'input_depower',
        }
        for var_name in base_symbolic_variables.keys():
            setattr(self, var_name, ca.SX.sym(var_name))
    def aerodynamic_force_coefficients(self):
        aero_input = self.aero_input
        # Define symbolic variables
        variables = {
            "alpha": self.angle_of_attack,
            "alpha_squared": self.angle_of_attack**2,
            "u_s": self.input_steering,
            "u_p": self.input_depower,
            # Dynamically add other variables as dependencies
            "yaw_rate": self.timeder_angle_course
            / ca.norm_2(self.velocity_apparent_wind),
            "sideslip": self.angle_sideslip,
        }
        # Initialize base aerodynamic coefficients
        if aero_input["model"] == "inviscid":
            e = aero_input["params"]["oswald_efficiency"]
            AR = aero_input["params"]["aspect_ratio"]
            CD0 = aero_input["params"]["CD0"]
            C_L = 2 * ca.pi * variables["alpha"]
            C_D = C_L**2 / (ca.pi * e * AR) + CD0
        elif aero_input["model"] == "coeffs":
            C_L = aero_input["params"].get("CL0", 0)
            C_D = aero_input["params"].get("CD0", 0)
            C_S = aero_input["params"].get("CS0", 0)
        else:
            raise ValueError(
                "Invalid aerodynamic model type. Choose 'inviscid' or 'coeffs'."
            )
        
        # Apply dependencies dynamically for CL, CD, CS, C_m, C_l, and C_n
        for var, coeffs in aero_input.get("dependencies", {}).items():
            for coeff_type, coeff_value in coeffs.items():
                if coeff_type == "k_cl":
                    C_L += coeff_value * variables[var]
                elif coeff_type == "k_cd":
                    C_D += coeff_value * ca.fabs(variables[var])
                elif coeff_type == "k_cs":
                    C_S += coeff_value * variables[var]

        # alpha_min = np.radians(-5)
        # alpha_max = np.radians(20)
        # C_L = ca.if_else(
        #     ca.logic_and(variables["alpha"] >= alpha_min, variables["alpha"] <= alpha_max), C_L, 0
        # )
        # C_D = ca.if_else(
        #     ca.logic_and(variables["alpha"] >= alpha_min, variables["alpha"] <= alpha_max), C_D, 1
        # )
        

        return C_L, C_D, C_S
    
    def aerodynamic_moment_coefficients(self):
        aero_input = self.aero_input
        # Define symbolic variables
        variables = {
            "alpha": self.angle_of_attack,
            "alpha_squared": self.angle_of_attack**2,
            "u_s": self.input_steering,
            "u_p": self.input_depower,
            # Dynamically add other variables as dependencies
            "yaw_rate": self.timeder_angle_course
            / ca.norm_2(self.velocity_apparent_wind),
            "sideslip": self.angle_sideslip,
        }

        C_m = aero_input["params"].get("C_m_base", 0) # Pitch moment coefficient
        C_l = aero_input["params"].get("C_l_base", 0) # Roll moment coefficient
        C_n = aero_input["params"].get("C_n_base", 0) # Yaw moment coefficient

        # Apply dependencies dynamically for CL, CD, CS, C_m, C_l, and C_n
        for var, coeffs in aero_input.get("dependencies", {}).items():
            for coeff_type, coeff_value in coeffs.items():
                if coeff_type == "k_cm":
                    C_m += coeff_value * variables[var]
                elif coeff_type == "k_cl_roll":
                    C_l += coeff_value * variables[var]
                elif coeff_type == "k_cn":
                    C_n += coeff_value * variables[var]

        return C_m, C_l, C_n
    # def aerodynamic_coeffs_function(self, aero_input):
    #     """
    #     Create a function to compute aerodynamic coefficients with dynamic dependencies.
    #     """
    #     # Define symbolic variables
    #     variables = {
    #         "alpha": self.angle_of_attack,
    #         "alpha_squared": self.angle_of_attack**2,
    #         "u_s": self.input_steering,
    #         "u_p": self.input_depower,
    #         # Dynamically add other variables as dependencies
    #         "yaw_rate": self.timeder_angle_course/ca.norm_2(self.velocity_apparent_wind),
    #         "sideslip": self.angle_sideslip,
    #     }
    #     C_S = 0 # Side force coefficient
    #     C_m = 0 # Pitch moment coefficient
    #     C_l = 0  # Roll moment coefficient
    #     C_n = 0  # Yaw moment coefficient

    #     # Initialize base aerodynamic coefficients
    #     if aero_input["model"] == "inviscid":
    #         e = aero_input["params"]["oswald_efficiency"]
    #         AR = aero_input["params"]["aspect_ratio"]
    #         CD0 = aero_input["params"]["CD0"]

    #         C_L = 2 * ca.pi * variables["alpha"]
    #         C_D = C_L**2 / (ca.pi * e * AR) + CD0
    #     elif aero_input["model"] == "coeffs":
    #         C_L = aero_input["params"]["CL0"]
    #         C_D = aero_input["params"]["CD0"]
    #     elif aero_input["model"] == "polars":
    #         cl_data, cd_data, alpha_data = (
    #             aero_input["params"]["CL"],
    #             aero_input["params"]["CD"],
    #             aero_input["params"]["alpha"],
    #         )

    #         # Fit polynomials for CL and CD
    #         cl_coeffs = np.polyfit(alpha_data, cl_data, 2)
    #         cd_coeffs = np.polyfit(alpha_data, cd_data, 2)

    #         # Define symbolic polynomials
    #         C_L = cl_coeffs[0] * variables["alpha"]**2 + cl_coeffs[1] * variables["alpha"] + cl_coeffs[2]
    #         C_D = cd_coeffs[0] * variables["alpha"]**2 + cd_coeffs[1] * variables["alpha"] + cd_coeffs[2]
    #         C_D += aero_input["params"].get("CD0", 0)
    #         # Constrain CL and CD to valid alpha range
    #         alpha_min, alpha_max = min(alpha_data), max(alpha_data)
    #         C_L = ca.if_else(
    #             ca.logic_and(variables["alpha"] >= alpha_min, variables["alpha"] <= alpha_max), C_L, 0
    #         )
    #         C_D = ca.if_else(
    #             ca.logic_and(variables["alpha"] >= alpha_min, variables["alpha"] <= alpha_max), C_D, 1
    #         )

    #     else:
    #         raise ValueError("Invalid aerodynamic model type. Choose 'inviscid' or 'polars'.")

    #         # Apply dependencies dynamically for CL, CD, CS, C_m, C_l, and C_n
    #     for var, coeffs in aero_input.get("dependencies", {}).items():
    #         for coeff_type, coeff_value in coeffs.items():
    #             if coeff_type == "k_cl":
    #                 C_L += coeff_value * variables[var]
    #             elif coeff_type == "k_cd":
    #                 C_D += coeff_value * ca.fabs(variables[var])
    #             elif coeff_type == "k_cs":
    #                 C_S += coeff_value * variables[var]
    #             elif coeff_type == "k_cm":  # Pitch moment coefficient
    #                 C_m += coeff_value * variables[var]
    #             elif coeff_type == "k_cn": # Yaw moment coefficient
    #                 C_n += coeff_value * variables[var]

    #     # Include base moment coefficients if specified
    #     C_m += aero_input["params"].get("C_m_base", 0)
    #     C_l += aero_input["params"].get("C_l_base", 0)
    #     C_n += aero_input["params"].get("C_n_base", 0)
        
    #     self.CL = C_L
    #     self.CD = C_D
    #     self.CS = C_S
    #     self.Cm = C_m
    #     self.Cl = C_l
    #     self.Cn = C_n

    @property
    def angle_pitch_depower(self):
        """
        Compute the tether angle based on the powered angle and the tether angle at t=0.
        """
        return self.angle_pitch_depower_0 + self.input_depower * self.delta_pitch_depower

    @property
    def angle_sideslip(self):
        """
        Compute the angle_sideslip
        """
        return self.angle_yaw - self.angle_yaw_aerodynamic

    @property
    def angle_of_attack(self):
        """
        Compute the angle of attack based on the air velocity vector and tether angle.
        """

        return self.angle_pitch_aerodynamic + self.angle_pitch_depower - self.angle_pitch
    
    @property
    def velocity_apparent_wind(self):

        return self.velocity_wind - self.velocity_kite

    @property
    def angle_pitch_aerodynamic(self):
        
        return ca.atan2(self.velocity_apparent_wind[2], ca.sqrt(self.velocity_apparent_wind[0]**2 + self.velocity_apparent_wind[1]**2))

    @property
    def angle_yaw_aerodynamic(self):
        return -ca.atan(self.velocity_apparent_wind[1]/ self.velocity_apparent_wind[0])

    @property
    def force_aerodynamic(self):
        """
        Compute the aerodynamic forces based on the aerodynamic coefficients.
        """
        V_a_sq = ca.mtimes(self.velocity_apparent_wind.T, self.velocity_apparent_wind)

        CL, CD, CS = self.aerodynamic_force_coefficients()
        # Aerodynamic forces
        D = 0.5 * self.rho * V_a_sq * self.area_wing * CD
        L = 0.5 * self.rho * V_a_sq * self.area_wing * CL
        S = 0.5 * self.rho * V_a_sq * self.area_wing * CS
    
        R = transformation_C_from_A(self.angle_pitch_aerodynamic, self.angle_yaw_aerodynamic, self.angle_roll)
        aero_forces = R @ ca.vertcat(-D, S, L)
        return aero_forces
       
    @property
    def force_gravity_wing(self):

        return transformation_C_from_W(self.angle_azimuth, self.angle_elevation, self.angle_course) @ ca.vertcat(0, 0, -self.mass_wing * self.g)
    

class Kite(Wing):
    
    def __init__(self, mass_wing, area_wing, aero_input, mass_kcu = 0, g=9.81, rho=1.225, center_aerodynamic_wing = [0,0,10], center_gravity_wing = [0,0,10]):
        """
        Initialize the kite system with its parameters.
        """
        self.define_symbolic_variables_kite()
        Wing.__init__(self, mass_wing, area_wing, aero_input)
        self.mass_kcu = mass_kcu  # Mass of the kite control unit
        self.g = g  # Gravitational acceleration
        self.rho = rho  # Air density
        self.center_aerodynamic_wing = center_aerodynamic_wing  # Center of aerodynamic pressure
        self.center_gravity_wing = center_gravity_wing  # Center of gravity

    def define_symbolic_variables_kite(self):
        """
        Define symbolic variables used in the model.
        """
        base_symbolic_variables = {
            'angle_pitch': 'angle_pitch',
            'angle_roll': 'angle_roll',
            'angle_yaw': 'angle_yaw',
        }
        for var_name in base_symbolic_variables.keys():
            setattr(self, var_name, ca.SX.sym(var_name))

    @property
    def force_gravity_kcu(self):
            
        T = transformation_C_from_W(self.angle_azimuth, self.angle_elevation, self.angle_course)
        return T @ ca.vertcat(0, 0, -self.mass_kcu * self.g)
        
    @property
    def force_gravity(self):
        return self.force_gravity_wing + self.force_gravity_kcu

    # @property
    def acceleration_rotation(self):   
        return ca.vertcat(
             (self.speed_tangential * self.speed_radial) / self.distance_radial,
            (self.speed_tangential**2 / self.distance_radial) * ca.sin(self.angle_course) * ca.tan(self.angle_elevation),
             - (self.speed_tangential**2 / self.distance_radial)
        )

    # @property
    def acceleration_local(self):
        return ca.vertcat(self.timeder_speed_tangential, - self.speed_tangential * self.timeder_angle_course, self.timeder_speed_radial)
    
    # @property
    def acceleration(self):
        return self.acceleration_local() + self.acceleration_rotation()

    @property
    def force_external(self):

        return self.force_aerodynamic + self.force_gravity + self.force_tether

    # @property
    def force_residual(self):
        """
        Compute the residual for the kite system dynamics.
        """   
        # LHS and RHS
        lhs = (self.mass_wing+self.mass_kcu) * self.acceleration()
        # Residual
        return lhs - self.force_external

    @property
    def aero_moment(self):

        R_C_from_K = transformation_C_from_K(self.angle_pitch, self.angle_roll)

        center_aerodynamic_wing = R_C_from_K @ ca.vertcat(*self.center_aerodynamic_wing)

        Cm, Cl, Cn = self.aerodynamic_moment_coefficients()
        aero_moment = ca.cross(center_aerodynamic_wing, self.force_aerodynamic)
        aero_moment[1] += Cm*self.area_wing*self.rho*ca.mtimes(self.velocity_apparent_wind.T, self.velocity_apparent_wind)*4 ### Not the correct formula
        aero_moment[2] += Cn*self.area_wing*self.rho*ca.mtimes(self.velocity_apparent_wind.T, self.velocity_apparent_wind)*4 ### Not the correct formula
        return aero_moment

    @property
    def gravity_moment(self):

        R_C_from_K = transformation_C_from_K(self.angle_pitch, self.angle_roll)

        center_gravity_wing = R_C_from_K @ ca.vertcat(*self.center_gravity_wing)

        return ca.cross(center_gravity_wing, self.force_gravity_wing)

    @property
    def inertia_moment(self):

        R_C_from_K = transformation_C_from_K(self.angle_pitch, self.angle_roll)

        center_gravity_wing = R_C_from_K @ ca.vertcat(*self.center_gravity_wing)

        return ca.cross(center_gravity_wing, (self.mass_wing) * self.acceleration())

    @property
    def moment_residual(self):
        """
        Compute the residual for the kite system dynamics.
        """
        
        aero_moment = self.aero_moment

        gravity_moment = self.gravity_moment

        inertia_moment = self.inertia_moment

        # Residual
        return aero_moment + gravity_moment - inertia_moment
    
    
    @property
    def rb_residual(self):
        """
        Compute the residual for the kite system dynamics. 
        Join the force and moment residuals.
        """
        return ca.vertcat(self.force_residual(), self.moment_residual)