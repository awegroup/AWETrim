import casadi as ca
import numpy as np
class KiteSystem:
    def __init__(self, m, A, aero_input, m_kcu = 0):
        """
        Initialize the kite system with its parameters.
        """
        self.m = m  # Mass
        self.A = A  # Reference area
        if "theta_t_0" in aero_input[1]:
            self.theta_t_0 = aero_input[1]["theta_t_0"]
        else:
            self.theta_t_0 = ca.SX.sym('theta_t_0')
        if "delta_theta_up" in aero_input[1]:
            self.delta_theta_up = aero_input[1]["delta_theta_up"]
        else:
            self.delta_theta_up = ca.SX.sym('delta_theta_up')
        if "CD0" in aero_input[1]:
            self.CD0 = aero_input[1]["CD0"]
        else:
            self.CD0 = 0
        if "k_cd_us" in aero_input[1]:
            self.k_cd_us = aero_input[1]["k_cd_us"]
        else:
            self.k_cd_us = 0
        if "k_cl_us" in aero_input[1]:
            self.k_cl_us = aero_input[1]["k_cl_us"]
        else:
            self.k_cl_us = 0
        if "k_cd_up" in aero_input[1]:
            self.k_cd_up = aero_input[1]["k_cd_up"]
        else:
            self.k_cd_up = 0
        if "k_cl_up" in aero_input[1]:
            self.k_cl_up = aero_input[1]["k_cl_up"]
        else:
            self.k_cl_up = 0
        self.aero_func = self.aerodynamic_coeffs_function(aero_input)
        # Define symbolic variables for the function inputs
        self.dot_v_tau = ca.SX.sym('dot_v_tau')
        self.dot_chi = ca.SX.sym('dot_chi')
        self.dot_v_r = ca.SX.sym('dot_v_r')
        self.v_tau = ca.SX.sym('v_tau')
        self.v_r = ca.SX.sym('v_r')
        self.r = ca.SX.sym('r')
        self.chi = ca.SX.sym('chi')
        self.beta = ca.SX.sym('beta')
        self.u_s = ca.SX.sym('u_s')
        self.u_p = ca.SX.sym('u_p')
        self.T = ca.SX.sym('T')
        self.phi = ca.SX.sym('phi')
        self.v_w = ca.SX.sym('v_w')
        self.g = ca.SX.sym('g')
        self.rho = ca.SX.sym('rho')
        self.mass_kcu = m_kcu

    @property
    def wind_velocity(self):
        """
        Compute the wind velocity in the body frame.
        """
        beta = self.beta
        chi = self.chi
        phi = self.phi
        v_w = self.v_w
        # Wind velocity components in the body frame
        v_w_x = (-ca.sin(beta) * ca.cos(chi) * ca.cos(phi) - ca.sin(chi) * ca.sin(phi)) * v_w
        v_w_y = (-ca.sin(beta) * ca.sin(chi) * ca.cos(phi) + ca.sin(phi) * ca.cos(chi)) * v_w
        v_w_z = ca.cos(beta) * ca.cos(phi) * v_w

        return ca.vertcat(v_w_x, v_w_y, v_w_z)
    
    @property
    def aerodynamic_coeffs(self):
        """
        Compute the aerodynamic coefficients based on the angle of attack and sideslip angle.
        """
        
        C_L, C_D, C_S = self.aero_func(self.angle_of_attack, self.u_s, self.u_p)
        C_D += self.CD0
        C_D += self.k_cd_us * ca.fabs(self.u_s)
        C_L += self.k_cl_us * ca.fabs(self.u_s)
        C_D += self.k_cd_up * self.u_p
        C_L += self.k_cl_up * self.u_p

        return C_L, C_D, C_S
    
    @property
    def theta_t(self):
        """
        Compute the tether angle based on the powered angle and the tether angle at t=0.
        """
        return self.theta_t_0 + self.u_p * self.delta_theta_up
    
    def aerodynamic_coeffs_function(self, aero_input):
        """
        Create a function to compute the aerodynamic coefficients based on the input.
        """
        alpha = ca.SX.sym('alpha')
        u_s = ca.SX.sym('u_s')
        u_p = ca.SX.sym('u_p')
        k_s = aero_input[1]["steering_coefficient"]
        if aero_input[0] == "inviscid":
            e = aero_input[1]["oswald_efficiency"]
            AR = aero_input[1]["aspect_ratio"]
            CD0 = aero_input[1]["CD0"]
            C_L = 2 * ca.pi * alpha
            C_D = C_L**2 / (ca.pi * e * AR) + CD0
            C_S = k_s * u_s
            
        elif aero_input[0] == "polars":
            cl_data = aero_input[1]["CL"]
            cd_data = aero_input[1]["CD"]
            alpha_data = aero_input[1]["alpha"]
            k_s = aero_input[1]["steering_coefficient"]

            # Fit polynomials from polar data
            cl_coeffs = np.polyfit(alpha_data, cl_data, 2)
            cd_coeffs = np.polyfit(alpha_data, cd_data, 2)

            # # Define range limits
            alpha_min = min(alpha_data)
            alpha_max = max(alpha_data)

            # Create symbolic variables
            alpha = ca.SX.sym('alpha')
            u_s = ca.SX.sym('u_s')

            # Define symbolic polynomials
            C_L = cl_coeffs[0] * alpha**2 + cl_coeffs[1] * alpha + cl_coeffs[2]
            C_D = cd_coeffs[0] * alpha**2 + cd_coeffs[1] * alpha + cd_coeffs[2]
            C_S = k_s * u_s

            # # Use ca.if_else to enforce 0 outside the range
            C_L = ca.if_else(ca.logic_and(alpha >= alpha_min, alpha <= alpha_max), C_L, 0)
            C_D = ca.if_else(ca.logic_and(alpha >= alpha_min, alpha <= alpha_max), C_D, 1)
            # C_S = ca.if_else(ca.logic_and(alpha >= alpha_min, alpha <= alpha_max), C_S, 0.0)

        # Return CasADi function
        return ca.Function(
            'aerodynamic_coeffs',
            [alpha, u_s, u_p],
            [C_L, C_D, C_S],
            ['alpha', 'u_s', 'u_p'],
            ['C_L', 'C_D', 'C_S']
        )

    @property
    def pitch_kcu(self):
        return np.arctan((self.mass_kcu*9.81 * np.cos(self.beta) * np.cos(self.chi) -self.v_tau*self.v_r/self.r)/
                           (self.T + self.mass_kcu*9.81 * np.sin(self.beta) - (self.mass_kcu*self.v_tau**2 / self.r)))

    @property
    def angle_of_attack(self):
        """
        Compute the angle of attack based on the air velocity vector and tether angle.
        """

        alpha = self.theta_a + self.theta_t - self.pitch_kcu

        return alpha
    
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

    def compute_residual(self):
        """
        Compute the residual for the kite system dynamics.
        """

        g = self.g
        rho = self.rho
        dot_v_tau = self.dot_v_tau
        dot_chi = self.dot_chi
        dot_v_r = self.dot_v_r
        v_tau = self.v_tau
        v_r = self.v_r
        r = self.r
        chi = self.chi
        beta = self.beta
        T = self.T
        theta_a = self.theta_a


        # Air velocity and aerodynamic angles
        v_a_vec = self.apparent_velocity
        V_a_sq = ca.mtimes(v_a_vec.T, v_a_vec)

        # Aerodynamic coefficients
        chi_a = ca.atan(v_a_vec[1]/ v_a_vec[0])

        C_L, C_D, C_S = self.aerodynamic_coeffs
        

        # Aerodynamic forces
        D = 0.5 * rho * V_a_sq * self.A * C_D
        L = 0.5 * rho * V_a_sq * self.A * C_L
        S = 0.5 * rho * V_a_sq * self.A * C_S

        # LHS and RHS
        lhs = self.m * ca.vertcat(
            dot_v_tau + (v_tau * v_r) / r,
            (v_tau**2 / r) * ca.sin(chi) * ca.tan(beta) - v_tau * dot_chi,
            dot_v_r - (v_tau**2 / r)
        )
        R = ca.vertcat(
            ca.horzcat(ca.cos(chi_a) * ca.cos(theta_a), -ca.sin(chi_a), ca.sin(theta_a) * ca.cos(chi_a)),
            ca.horzcat(ca.sin(chi_a) * ca.cos(theta_a), ca.cos(chi_a), ca.sin(chi_a) * ca.sin(theta_a)),
            ca.horzcat(-ca.sin(theta_a), 0, ca.cos(theta_a))
        )
        aero_forces = R @ ca.vertcat(-D, S, L)
        tether_force = ca.vertcat(0, 0, -T)
        gravity_force = self.m * g * ca.vertcat(
            -ca.cos(beta) * ca.cos(chi),
            -ca.sin(chi) * ca.cos(beta),
            -ca.sin(beta)
        )
        rhs = aero_forces + tether_force + gravity_force

        # Residual
        return lhs - rhs

    def get_residual_function(self):
        """
        Returns a CasADi function for the residual.
        """


        # Call compute_residual with the symbolic variables
        residual = self.compute_residual()

        # Return a CasADi function
        return ca.Function(
            'residual_func',
            [self.dot_v_tau, self.dot_chi, self.dot_v_r, self.v_tau, self.v_r, self.r, self.chi, self.beta, self.u_s, self.u_p, self.T, self.phi, self.v_w, self.g, self.rho],
            [residual],
            ['dot_v_tau', 'dot_chi', 'dot_v_r', 'v_tau', 'v_r', 'r', 'chi', 'beta', 'u_s', 'u_p', 'T', 'phi', 'v_w', 'g', 'rho'],
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
             self.g, self.rho, self.delta_theta_up, self.theta_t_0, self.CD0, self.k_cd_us, self.k_cl_us, self.k_cd_up, self.k_cl_up],
            [residual],
            ['dot_v_tau', 'dot_chi', 'dot_v_r', 'v_tau', 'v_r', 'r', 'chi', 'beta', 'u_s', 'u_p', 'T', 'phi', 'v_w', 
             'g', 'rho', 'delta_theta_up', 'theta_t_0', 'CD0', 'k_cd_us', 'k_cl_us', 'k_cd_up', 'k_cl_up'],
            ['residual']
        )