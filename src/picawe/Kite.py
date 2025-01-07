import casadi as ca
import numpy as np
from picawe.reference_frames import transformation_C_from_AZR, transformation_C_from_A, transformation_C_from_K
class KiteSystem:
    # Base symbolic variables
    base_symbolic_variables = {
        'dot_v_tau': 'dot_v_tau',
        'dot_chi': 'dot_chi',
        'dot_v_r': 'dot_v_r',
        'v_tau': 'v_tau',
        'v_r': 'v_r',
        'r': 'r',
        'chi': 'chi',
        'beta': 'beta',
        'u_s': 'u_s',
        'u_p': 'u_p',
        'T': 'T',
        'phi': 'phi',
        'v_w': 'v_w',
        'theta_k': 'theta_k',
        'phi_k': 'phi_k',
        'psi_k': 'psi_k',
    }

    # Property-level dependencies (mapped to other properties or symbolic variables)
    property_dependencies = {
        'wind_velocity': ['beta', 'chi', 'phi', 'v_w'],
        'aerodynamic_coeffs': ['angle_of_attack', 'u_s', 'u_p'],
        'theta_t': ['u_p'],
        'pitch_kcu': ['beta', 'chi', 'v_tau', 'v_r', 'r', 'T'],
        'angle_of_attack': ['theta_a', 'theta_t', 'pitch_kcu', 'theta_k'],
        'apparent_velocity': ['v_tau', 'v_r', 'beta', 'chi', 'phi', 'v_w'],
        'theta_a': ['apparent_velocity'],
        'chi_a': ['apparent_velocity'],
        'aerodynamic_forces': ['theta_a', 'chi_a', 'apparent_velocity', 'aerodynamic_coeffs'],
        'gravity_force': ['beta', 'chi', 'm_wing'],
        'tether_force': ['T'],
        'acceleration': ['v_tau', 'v_r', 'r', 'chi', 'beta', 'dot_v_tau', 'dot_chi', 'dot_v_r'],
        'force_residual': ['aerodynamic_forces', 'gravity_force', 'tether_force', 'acceleration', 'm_wing'],
        'sideslip': ['psi_k', 'chi_a'],
        'CL': ['angle_of_attack', 'apparent_velocity', 'u_s', 'u_p'],
        'CD': ['angle_of_attack', 'apparent_velocity', 'u_s', 'u_p'],
    }
    def __init__(self, m_wing, A, aero_input, m_kcu = 0, g=9.81, rho=1.225, x_ca_wing = [0,0,10], x_cg_wing = [0,0,10]):
        """
        Initialize the kite system with its parameters.
        """
        self.m_wing = m_wing # Mass
        self.A = A  # Reference area
        self.m_kcu = m_kcu  # Mass of the kite control unit
        self.g = g  # Gravitational acceleration
        self.rho = rho  # Air density
        self.x_ca_wing = x_ca_wing  # Center of aerodynamic pressure
        self.x_cg_wing = x_cg_wing  # Center of gravity
        
        
        # Aerodynamic inputs
        self.theta_t_0 = aero_input['params'].get("theta_t_0", ca.SX.sym('theta_t_0'))
        self.delta_theta_up = aero_input['params'].get("delta_theta_up", ca.SX.sym('delta_theta_up'))

        # Define symbolic variables for the function inputs
        self.define_symbolic_variables()
        self.aerodynamic_coeffs_function(aero_input)
        

    def define_symbolic_variables(self):
            """
            Define symbolic variables used in the model.
            """
            for var_name in self.base_symbolic_variables.keys():
                setattr(self, var_name, ca.SX.sym(var_name))

    @property
    def wind_velocity(self):
        """
        Compute the wind velocity in the body frame.
        """

        beta, chi, phi, v_w = self.beta, self.chi, self.phi, self.v_w
        # Wind velocity components in the body frame
        v_w_x = (-ca.sin(beta) * ca.cos(chi) * ca.cos(phi) - ca.sin(chi) * ca.sin(phi)) * v_w
        v_w_y = (-ca.sin(beta) * ca.sin(chi) * ca.cos(phi) + ca.sin(phi) * ca.cos(chi)) * v_w
        v_w_z = ca.cos(beta) * ca.cos(phi) * v_w

        return ca.vertcat(v_w_x, v_w_y, v_w_z)
    
    @property
    def theta_t(self):
        """
        Compute the tether angle based on the powered angle and the tether angle at t=0.
        """
        return self.theta_t_0 + self.u_p * self.delta_theta_up

    @property
    def sideslip(self):
        """
        Compute the sideslip
        """
        return self.psi_k - self.chi_a
    
    def aerodynamic_coeffs_function(self, aero_input):
        """
        Create a function to compute aerodynamic coefficients with dynamic dependencies.
        """
        # Define symbolic variables
        variables = {
            "alpha": self.angle_of_attack,
            "u_s": self.u_s,
            "u_p": self.u_p,
            # Dynamically add other variables as dependencies
            "yaw_rate": self.dot_chi/ca.norm_2(self.apparent_velocity),
            "sideslip": self.sideslip,
        }
        C_S = 0 # Side force coefficient
        C_m = 0 # Pitch moment coefficient
        C_l = 0  # Roll moment coefficient
        C_n = 0  # Yaw moment coefficient

        # Initialize base aerodynamic coefficients
        if aero_input["model"] == "inviscid":
            e = aero_input["params"]["oswald_efficiency"]
            AR = aero_input["params"]["aspect_ratio"]
            CD0 = aero_input["params"]["CD0"]

            C_L = 2 * ca.pi * variables["alpha"]
            C_D = C_L**2 / (ca.pi * e * AR) + CD0

        elif aero_input["model"] == "polars":
            cl_data, cd_data, alpha_data = (
                aero_input["params"]["CL"],
                aero_input["params"]["CD"],
                aero_input["params"]["alpha"],
            )

            # Fit polynomials for CL and CD
            cl_coeffs = np.polyfit(alpha_data, cl_data, 2)
            cd_coeffs = np.polyfit(alpha_data, cd_data, 2)

            # Define symbolic polynomials
            C_L = cl_coeffs[0] * variables["alpha"]**2 + cl_coeffs[1] * variables["alpha"] + cl_coeffs[2]
            C_D = cd_coeffs[0] * variables["alpha"]**2 + cd_coeffs[1] * variables["alpha"] + cd_coeffs[2]
            C_D += aero_input["params"].get("CD0", 0)
            # Constrain CL and CD to valid alpha range
            alpha_min, alpha_max = min(alpha_data), max(alpha_data)
            C_L = ca.if_else(
                ca.logic_and(variables["alpha"] >= alpha_min, variables["alpha"] <= alpha_max), C_L, 0
            )
            C_D = ca.if_else(
                ca.logic_and(variables["alpha"] >= alpha_min, variables["alpha"] <= alpha_max), C_D, 1
            )

        else:
            raise ValueError("Invalid aerodynamic model type. Choose 'inviscid' or 'polars'.")

            # Apply dependencies dynamically for CL, CD, CS, C_m, C_l, and C_n
        for var, coeffs in aero_input.get("dependencies", {}).items():
            for coeff_type, coeff_value in coeffs.items():
                if coeff_type == "k_cl":
                    C_L += coeff_value * variables[var]
                elif coeff_type == "k_cd":
                    C_D += coeff_value * ca.fabs(variables[var])
                elif coeff_type == "k_cs":
                    C_S += coeff_value * variables[var]
                elif coeff_type == "k_cm":  # Pitch moment coefficient
                    C_m += coeff_value * variables[var]

        # Include base moment coefficients if specified
        C_m += aero_input["params"].get("C_m_base", 0)
        C_l += aero_input["params"].get("C_l_base", 0)
        C_n += aero_input["params"].get("C_n_base", 0)
        
        self.CL = C_L
        self.CD = C_D
        self.CS = C_S
        self.Cm = C_m
        self.Cl = C_l
        self.Cn = C_n

    # @property
    # def pitch_kcu(self):
    #     return np.arctan((self.m_kcu*9.81 * np.cos(self.beta) * np.cos(self.chi) -self.v_tau*self.v_r/self.r)/
    #                        (self.T + self.m_kcu*9.81 * np.sin(self.beta) - (self.m_kcu*self.v_tau**2 / self.r)))

    @property
    def angle_of_attack(self):
        """
        Compute the angle of attack based on the air velocity vector and tether angle.
        """

        return self.theta_a + self.theta_t - self.theta_k
    
    @property
    def apparent_velocity(self):

        v_k_vec = ca.vertcat(self.v_tau, 0, self.v_r)
        v_w_vec = self.wind_velocity
        v_a_vec = v_w_vec - v_k_vec

        return v_a_vec

    @property
    def theta_a(self):
        
        theta_a = ca.atan2(self.apparent_velocity[2], ca.sqrt(self.apparent_velocity[0]**2 + self.apparent_velocity[1]**2))
        return theta_a

    @property
    def chi_a(self):
        return ca.atan(self.apparent_velocity[1]/ self.apparent_velocity[0])

    @property
    def aerodynamic_forces(self):
        """
        Compute the aerodynamic forces based on the aerodynamic coefficients.
        """
        V_a_sq = ca.mtimes(self.apparent_velocity.T, self.apparent_velocity)

        # Aerodynamic forces
        D = 0.5 * self.rho * V_a_sq * self.A * self.CD
        L = 0.5 * self.rho * V_a_sq * self.A * self.CL
        S = 0.5 * self.rho * V_a_sq * self.A * self.CS
    
        R = transformation_C_from_A(self.theta_a, self.chi_a, self.phi_k)
        aero_forces = R @ ca.vertcat(-D, S, L)
        return aero_forces
       
    @property
    def gravity_force_wing(self):

        return (self.m_wing) * self.g * ca.vertcat(
            -ca.cos(self.beta) * ca.cos(self.chi),
            -ca.sin(self.chi) * ca.cos(self.beta),
            -ca.sin(self.beta)
        )

    @property
    def gravity_force_kcu(self):
            
            return (self.m_kcu) * self.g * ca.vertcat(
                -ca.cos(self.beta) * ca.cos(self.chi),
                -ca.sin(self.chi) * ca.cos(self.beta),
                -ca.sin(self.beta)
            )
        
    @property
    def gravity_force(self):
        return self.gravity_force_wing + self.gravity_force_kcu

    @property
    def tether_force(self):

        return ca.vertcat(0, 0, -self.T)

    @property
    def acceleration(self):
        
        v_tau, v_r, r, chi, beta = self.v_tau, self.v_r, self.r, self.chi, self.beta
        dot_v_tau, dot_chi, dot_v_r = self.dot_v_tau, self.dot_chi, self.dot_v_r
        return ca.vertcat(
            dot_v_tau + (v_tau * v_r) / r,
            (v_tau**2 / r) * ca.sin(chi) * ca.tan(beta) - v_tau * dot_chi,
            dot_v_r - (v_tau**2 / r)
        )

    @property
    def force_residual(self):
        """
        Compute the residual for the kite system dynamics.
        """

        # Compute the aerodynamic forces
        aero_forces = self.aerodynamic_forces

        # Compute the gravity force
        gravity_force = self.gravity_force

        # Compute the tether force
        tether_force = self.tether_force

        # LHS and RHS
        lhs = (self.m_wing+self.m_kcu) * self.acceleration
        rhs = aero_forces + tether_force + gravity_force

        # Residual
        return lhs - rhs

    @property
    def aero_moment(self):

        R_C_from_K = transformation_C_from_K(self.theta_k, self.phi_k)

        x_ca_wing = R_C_from_K @ ca.vertcat(*self.x_ca_wing)


        aero_moment = ca.cross(x_ca_wing, self.aerodynamic_forces)
        aero_moment[1] += self.Cn*self.A*self.rho*ca.mtimes(self.apparent_velocity.T, self.apparent_velocity)*4 ### Not the correct formula
        aero_moment[2] += self.Cm*self.A*self.rho*ca.mtimes(self.apparent_velocity.T, self.apparent_velocity)*4 ### Not the correct formula
        return aero_moment

    @property
    def gravity_moment(self):

        R_C_from_K = transformation_C_from_K(self.theta_k, self.phi_k)

        x_cg_wing = R_C_from_K @ ca.vertcat(*self.x_cg_wing)

        return ca.cross(x_cg_wing, self.gravity_force_wing)

    @property
    def inertia_moment(self):

        R_C_from_K = transformation_C_from_K(self.theta_k, self.phi_k)

        x_cg_wing = R_C_from_K @ ca.vertcat(*self.x_cg_wing)

        return ca.cross(x_cg_wing, (self.m_wing) * self.acceleration)

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

        return ca.vertcat(self.force_residual, self.moment_residual)

    def get_residual_function(self):
        """
        Returns a CasADi function for the residual.
        """

        # Return a CasADi function
        return ca.Function(
            'residual_func',
            [self.dot_v_tau, self.dot_chi, self.dot_v_r, self.v_tau, self.v_r, self.r, self.chi, self.beta, self.u_s, self.u_p, self.T, self.phi, self.v_w],
            [self.force_residual],
            ['dot_v_tau', 'dot_chi', 'dot_v_r', 'v_tau', 'v_r', 'r', 'chi', 'beta', 'u_s', 'u_p', 'T', 'phi', 'v_w'],
            ['residual']
        )
    
    def get_residual_function_aero_iden(self):
        """
        Returns a CasADi function for the residual.
        """


        # Call compute_residual with the symbolic variables
        residual = self.compute_residual()

        # Return a CasADi function
        return ca.Function(
            'residual_func',
            [self.dot_v_tau, self.dot_chi, self.dot_v_r, self.v_tau, self.v_r, self.r, self.chi, self.beta, self.u_s, self.u_p, self.T, self.phi, self.v_w, 
             self.delta_theta_up, self.theta_t_0, self.CD0, self.k_cd_us, self.k_cl_us, self.k_cd_up, self.k_cl_up],
            [residual],
            ['dot_v_tau', 'dot_chi', 'dot_v_r', 'v_tau', 'v_r', 'r', 'chi', 'beta', 'u_s', 'u_p', 'T', 'phi', 'v_w', 
             'delta_theta_up', 'theta_t_0', 'CD0', 'k_cd_us', 'k_cl_us', 'k_cd_up', 'k_cl_up'],
            ['residual']
        )

    def resolve_dependencies(self, property_name):
        """
        Recursively resolve dependencies to find all symbolic variables required for a property.

        :param property_name: The name of the property (e.g., 'aerodynamic_forces').
        :return: Set of symbolic variables required for the property.
        """
        if property_name in self.base_symbolic_variables:
            # Base case: property is a direct symbolic variable
            return {property_name}

        if property_name not in self.property_dependencies:
            raise ValueError(f"Property '{property_name}' not found in dependency map.")

        resolved_deps = set()
        for dep in self.property_dependencies[property_name]:
            # Recursively resolve dependencies
            resolved_deps.update(self.resolve_dependencies(dep))
        return resolved_deps

    def get_symbolic_dependencies(self, property_name):
        """
        Get the actual symbolic variables required for a property.

        :param property_name: The name of the property (e.g., 'aerodynamic_forces').
        :return: List of CasADi symbolic variables required for the property.
        """
        dependencies = self.resolve_dependencies(property_name)
        return [getattr(self, dep) for dep in dependencies if dep in self.base_symbolic_variables]

    def extract_parameter_function(self, parameter_name):
        """
        Creates a CasADi function to compute a specific parameter from the kite state.

        :param parameter_name: The name of the parameter to extract (e.g., 'alpha').
        :return: CasADi function to compute the parameter.
        """
        # Check if the parameter is defined
        if parameter_name not in self.property_dependencies:
            raise ValueError(f"Parameter {parameter_name} is not defined.")

        # Get the symbolic expression for the parameter
        parameter_expr = getattr(self, parameter_name)

        # Get the required inputs for this parameter
        required_inputs = list(self.resolve_dependencies(parameter_name))  # Ensure it's a list

        print(f"Required inputs for {parameter_name}: {required_inputs}")

        # Map required inputs to symbolic variables
        input_symbols = [getattr(self, var) for var in required_inputs]
        print(f"Input symbols for {parameter_name}: {input_symbols}")

        # Create and return the CasADi function
        return ca.Function(f'compute_{parameter_name}', input_symbols, [parameter_expr], required_inputs, [parameter_name])
