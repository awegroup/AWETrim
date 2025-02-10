import casadi as ca
import numpy as np
from picawe.reference_frames import transformation_C_from_W, transformation_C_from_A, transformation_C_from_K

class RigidWing:

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
            'angle_roll_control': 'angle_roll_control',
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
            # "u_s": self.input_steering,
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
    
        R = transformation_C_from_A(self.angle_pitch_aerodynamic, self.angle_yaw_aerodynamic, self.angle_roll)
        aero_forces = R @ ca.vertcat(-D, L*ca.sin(self.angle_roll_control), L*ca.cos(self.angle_roll_control))
        return aero_forces
       
    @property
    def force_gravity_wing(self):

        return transformation_C_from_W(self.angle_azimuth, self.angle_elevation, self.angle_course) @ ca.vertcat(0, 0, -self.mass_wing * self.g)
    
    
      
class RigidKite(RigidWing):
    
    def __init__(self, mass_wing, area_wing, aero_input, g=9.81, rho=1.225):
        """
        Initialize the kite system with its parameters.
        """
        self.define_symbolic_variables_kite()
        RigidWing.__init__(self, mass_wing, area_wing, aero_input)
        self.g = g  # Gravitational acceleration
        self.rho = rho  # Air density

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
    def force_gravity(self):
        return self.force_gravity_wing

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

        return self.force_aerodynamic + self.force_gravity + self.force_tether_at_kite

    # @property
    def force_residual(self):
        """
        Compute the residual for the kite system dynamics.
        """   
        # LHS and RHS
        lhs = (self.mass_wing) * self.acceleration()
        lhs = 0
        # Residual
        return lhs - self.force_external

