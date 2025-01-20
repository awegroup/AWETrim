import casadi as ca
import numpy as np
from picawe.Tether import Tether
from picawe.Kinematics import KiteKinematics
from picawe.Wind import Wind
from picawe.Kite import Kite


class State(KiteKinematics, Tether, Wind, Kite):
    
    # Property-level dependencies (mapped to other properties or symbolic variables)
    property_dependencies = {
        'wind_velocity': ['angle_elevation', 'angle_course', 'angle_azimuth', 'speed_wind'],
        'aerodynamic_coeffs': ['angle_of_attack', 'input_steering', 'input_depower'],
        'angle_pitch_depower': ['input_depower'],
        'angle_of_attack': ['angle_pitch_aerodynamic', 'angle_pitch_depower', 'angle_pitch'],
        'velocity_apparent_wind': ['speed_tangential', 'speed_radial', 'angle_elevation', 'angle_course', 'angle_azimuth', 'speed_wind'],
        'angle_pitch_aerodynamic': ['velocity_apparent_wind'],
        'angle_yaw_aerodynamic': ['velocity_apparent_wind'],
        'force_aerodynamic': ['angle_pitch_aerodynamic', 'angle_yaw_aerodynamic', 'velocity_apparent_wind', 'input_depower', 'angle_pitch', 'input_steering', 'angle_yaw', 'angle_roll'],
        'force_gravity': ['angle_elevation', 'angle_course'],
        'tension_tether': ['distance_radial', 'length_tether'],
        'acceleration': ['speed_tangential', 'speed_radial', 'distance_radial', 'angle_course', 'angle_elevation', 'timeder_speed_tangential', 'timeder_angle_course', 'timeder_speed_radial'],
        'force_residual': ['force_aerodynamic', 'force_gravity', 'force_tether', 'acceleration', 'mass_wing'],
        'angle_sideslip': ['angle_yaw', 'angle_yaw_aerodynamic'],
        'CL': ['angle_of_attack', 'velocity_apparent_wind', 'input_steering', 'input_depower'],
        'CD': ['angle_of_attack', 'velocity_apparent_wind', 'input_steering', 'input_depower'],
        'timeder_angle_elevation': ['speed_tangential', 'distance_radial', 'angle_course'],
        'timeder_angle_azimuth': ['speed_tangential', 'distance_radial', 'angle_course', 'angle_elevation'],
        'isolate_tether_tension': ['force_aerodynamic', 'force_gravity', 'acceleration'],
    }
    def __init__(self, mass_wing, area_wing, aero_input, mass_kcu = 0, g=9.81, rho=1.225, center_aerodynamic_wing = [0,0,10], center_gravity_wing = [0,0,10], E = 132e9, diameter = 0.008, density = 970):
        """
        Initialize the kite system with its parameters.
        """
        # Define symbolic variables for the function inputs
        KiteKinematics.__init__(self)
        Wind.__init__(self)
        Kite.__init__(self, mass_wing, area_wing, aero_input, mass_kcu, g, rho, center_aerodynamic_wing, center_gravity_wing)
        Tether.__init__(self, E, diameter, density)
        
        
        
    @property
    def ode(self):

        local_acceleration = (self.force_external - self.acceleration_rotation*(self.mass_wing + self.mass_kcu))/(self.mass_wing + self.mass_kcu)
        dot_r = self.speed_radial
        dot_vr = local_acceleration[2]
        dot_vt = local_acceleration[0]
        dot_chi = local_acceleration[1]/(-self.speed_tangential)
        dot_theta = self.timeder_angle_azimuth
        dot_beta = self.timeder_angle_elevation

        ode = ca.vertcat(dot_r, dot_vr, dot_vt, dot_chi, dot_theta, dot_beta)
        return ode

    @property
    def isolate_tether_tension(self):

        T = self.acceleration*(self.mass_wing + self.mass_kcu) - self.force_aerodynamic - self.force_gravity
        return -T[2]

    def solve_quasi_steady_state(self, unknown_vars, x0,solver_options = {}, dof=6, return_not_converged = True):
        """
        Solve the quasi-steady state equations for the kite system.

        :param known_state: Dictionary of known state variables and their values.
        :param unknown_vars: List of unknown state variables to solve for.
        :return: Dictionary of unknown state variables and their values.
        """
        if dof == 6:
            residual = self.rb_residual
            # Solve the system of equations
            lbx = [self.distance_radial-5, -10, -1, -np.pi / 4, -np.pi / 4, -np.pi / 4]  # Lower bounds for T, u_s, speed_tangential, phi_k, theta_k
            ubx = [self.distance_radial, 10, 500, np.pi / 4, np.pi / 4, np.pi / 4]  # Upper bounds for T, u_s, speed_tangential, phi_k, theta_k
        elif dof == 3:
            self.angle_roll = 0
            self.angle_pitch = 0
            self.angle_yaw = 0
            residual = self.force_residual
            # Solve the system of equations
            lbx = [self.distance_radial-5, -10, 0]  # Lower bounds for T, u_s, speed_tangential, phi_k, theta_k
            ubx = [self.distance_radial, 10, 500]  # Upper bounds for T, u_s, speed_tangential, phi_k, theta_k

        sym_list = [getattr(self, name) for name in unknown_vars]
            
        # NLP problem definition
        nlp = {'x': ca.vertcat(*sym_list), 'f': 0, 'g': residual}  # 'f' is set to 0 for root-finding

        # Define the NLP solver
        solver = ca.nlpsol('solver', 'ipopt', nlp, solver_options)

        # Bounds for the constraints
        lbg = [0] * residual.size1()  # Lower bounds (0 for residuals)
        ubg = [0] * residual.size1()  # Upper bounds (0 for residuals)

        # try:
        # Solve the system
        sol = solver(
            x0=x0,  # Initial guess
            lbg=lbg,
            ubg=ubg,
            lbx=lbx,
            ubx=ubx
        )
        converged = True
        if ca.norm_1(sol['g']) > 1:
            print("Quasi-steady state not found")
            converged = False
        
        return sol['x'], converged
    
    def integrate(self, current_state, time, time_step):

        x = ca.vertcat(*[getattr(self, name) for name in self.ode_states])
        ode = self.ode
        # Substitute known values into the ode function
        for name, value in current_state.items():
            if name in self.ode_states:
                continue
            else:
                variable = getattr(self, name)
                ode = ca.substitute(ode, variable, value)

        # Define the CasADi integrator
        intg = ca.integrator('intg','cvodes',{'x':x,'ode':ode},time,time+time_step)

        x0 = [current_state[name] for name in self.ode_states]
        res = intg(x0 = x0)

        # new_state.update(current_state)
        return res['xf']




    @property
    def ode_states(self):
        return ['distance_radial', 'speed_radial', 'speed_tangential', 'angle_course', 'angle_azimuth', 'angle_elevation']

    def resolve_dependencies(self, property_name):
        """
        Recursively resolve dependencies to find all symbolic variables required for a property.

        :param property_name: The name of the property (e.g., 'force_aerodynamic').
        :return: Set of symbolic variables required for the property.
        """
        # Check if the property exists in the class
        if not hasattr(self, property_name):
            raise ValueError(f"Property '{property_name}' not found in the class.")

        prop = getattr(self, property_name)

        # If it is a symbolic variable or expression, return it directly
        if isinstance(prop, ca.SX):
            if prop.is_symbolic():  # Direct symbolic variable
                return {property_name}
            else:  # Expression
                if property_name in self.property_dependencies:
                    resolved_deps = set()
                    for dep in self.property_dependencies[property_name]:
                        # Recursively resolve dependencies
                        resolved_deps.update(self.resolve_dependencies(dep))
                    return resolved_deps
                else:
                    raise ValueError(
                        f"Property '{property_name}' is a symbolic expression but not in the dependency map."
                    )

        # If the property is not symbolic or an expression, raise an error
        raise ValueError(f"Property '{property_name}' is not a symbolic variable or expression.")


    def get_symbolic_dependencies(self, property_name):
        """
        Get the actual symbolic variables required for a property.

        :param property_name: The name of the property (e.g., 'force_aerodynamic').
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
