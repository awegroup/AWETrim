import casadi as ca
from picawe.utils import skew_symmetric
from picawe.reference_frames import transformation_C_from_W, transformation_C_from_A, transformation_C_from_K



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
                # print(C_L)
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
    def velocity_rotation_wing(self):
        return ca.vertcat(self.timeder_angle_roll, self.timeder_angle_pitch, self.timeder_angle_yaw)
    
    @property
    def velocity_apparent_wind_wing(self):
        velocity_wing_rotation = ca.cross(self.velocity_rotation_wing, self.center_gravity_wing_course)
        return self.velocity_apparent_wind - velocity_wing_rotation

    @property
    def angle_pitch_aerodynamic(self):
        
        return ca.atan2(self.velocity_apparent_wind_wing[2], ca.sqrt(self.velocity_apparent_wind_wing[0]**2 + self.velocity_apparent_wind_wing[1]**2))

    @property
    def angle_yaw_aerodynamic(self):
        return -ca.atan(self.velocity_apparent_wind_wing[1]/ self.velocity_apparent_wind_wing[0])

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
        self._override_gravity = False
        self._override_centripetal = False
        self._override_coriolis = False
        self._angle_yaw = ca.SX.sym("angle_yaw")

    def define_symbolic_variables_kite(self):
        """
        Define symbolic variables used in the model.
        """
        base_symbolic_variables = {
            'angle_pitch': 'angle_pitch',
            'angle_roll': 'angle_roll',
            'timeder_angle_pitch': 'timeder_angle_pitch',
            'timeder_angle_roll': 'timeder_angle_roll',
            'timeder_angle_yaw': 'timeder_angle_yaw',
            'acceleration_angle_pitch': 'acceleration_angle_pitch',
            'acceleration_angle_roll': 'acceleration_angle_roll',
            'acceleration_angle_yaw': 'acceleration_angle_yaw',
        }
        for var_name in base_symbolic_variables.keys():
            setattr(self, var_name, ca.SX.sym(var_name))

    @property
    def force_gravity_kcu(self):
            
        T = transformation_C_from_W(self.angle_azimuth, self.angle_elevation, self.angle_course)
        return T @ ca.vertcat(0, 0, -self.mass_kcu * self.g)
    
        
    @property
    def force_gravity(self):
        if self._override_gravity == True:
            return ca.vertcat(0, 0, 0)
        return self.force_gravity_wing + self.force_gravity_kcu
    
    @property
    def override_gravity(self):
        return self._override_gravity
    
    @override_gravity.setter
    def override_gravity(self, value):
        if not isinstance(value, bool):
            raise ValueError("override_gravity ha de ser True o False.")
        self._override_gravity = value

    @property
    def override_centripetal(self):
        return self._override_centripetal
    
    @override_centripetal.setter
    def override_centripetal(self, value):
        if not isinstance(value, bool):
            raise ValueError("override_gravity ha de ser True o False.")
        self._override_centripetal = value

    @property
    def override_coriolis(self):
        return self._override_coriolis
    
    @override_coriolis.setter
    def override_coriolis(self, value):
        if not isinstance(value, bool):
            raise ValueError("override_gravity ha de ser True o False.")
        self._override_coriolis = value
    
    # @property
    def acceleration_rotation_course(self):   
        if self._override_centripetal == True:
            return ca.vertcat(self.speed_tangential * self.speed_radial / self.distance_radial, 0, 0)
        if self._override_coriolis == True:
            return ca.cross(self.velocity_rotation_course_frame, self.velocity_kite) - ca.vertcat(2*self.speed_tangential * self.speed_radial / self.distance_radial, 0, 0)
        return ca.cross(self.velocity_rotation_course_frame, self.velocity_kite)

    # @property
    def acceleration_local(self):
        return ca.vertcat(self.timeder_speed_tangential, 0, self.timeder_speed_radial)
    
    # @property
    def acceleration(self):
        return self.acceleration_local() + self.acceleration_rotation_course()

    def velocity_rotation(self):
        return ca.vertcat(self.timeder_angle_pitch, self.timeder_angle_roll, self.timeder_angle_yaw)
    
    def acceleration_rotation_kite(self):
        return ca.vertcat(self.acceleration_angle_roll, self.acceleration_angle_pitch, self.acceleration_angle_yaw)

    @property
    def force_external(self):

        return self.force_aerodynamic + self.force_gravity + self.force_tether_at_kite

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
        aero_moment[1] += Cm*self.area_wing*self.rho*ca.mtimes(self.velocity_apparent_wind.T, self.velocity_apparent_wind)*2 ### Not the correct formula
        aero_moment[2] += Cn*self.area_wing*self.rho*ca.mtimes(self.velocity_apparent_wind.T, self.velocity_apparent_wind)*2 ### Not the correct formula
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
    def moment_external(self):

        return self.aero_moment + self.gravity_moment 
    
    
    @property
    def rb_residual(self):
        """
        Compute the residual for the kite system dynamics. 
        Join the force and moment residuals.
        """
        x_cg_c_cross = skew_symmetric(self.center_gravity_wing_course)
        omega_cross = skew_symmetric(self.velocity_rotation_course_frame)
        I = self.inertia_matrix_course
        m = (self.mass_wing+self.mass_kcu)
        m_w = self.mass_wing
        # Create the block matrix
        M = ca.vertcat(
            ca.horzcat(m * ca.SX.eye(3), -m_w * x_cg_c_cross),
            ca.horzcat(m_w * x_cg_c_cross, I)
        )

        ROT = ca.vertcat(
            ca.horzcat( omega_cross,ca.SX.zeros(3,3)),
            ca.horzcat(ca.SX.zeros(3,3), omega_cross)
        )

        acceleration = ca.vertcat(self.acceleration_local(), self.acceleration_rotation_kite())
        velocity = ca.vertcat(self.velocity_kite, self.velocity_rotation())

        lhs = M @ acceleration + ROT @ M @ velocity
        rhs = ca.vertcat(self.force_external, self.moment_external)

        return lhs - rhs
    
    @property
    def inertia_matrix_body(self):
        #TODO: Calculate based on the cg position
        return ca.diag([1, 1, 0])*self.mass_wing*self.center_gravity_wing[2]**2
    
    @property
    def inertia_matrix_course(self):
        return transformation_C_from_K(self.angle_pitch, self.angle_roll)@self.inertia_matrix_body
    
    @property
    def center_gravity_wing_course(self):
        return transformation_C_from_K(self.angle_pitch, self.angle_roll)@ca.vertcat(*self.center_gravity_wing)
    

    @property  
    def angle_yaw(self):
        
        if self.dof == 3:
            return self.angle_yaw_aerodynamic
        
        elif self.dof == 6:
            return self._angle_yaw