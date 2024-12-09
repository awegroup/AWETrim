import casadi as ca

class Control:
    """
    A class to manage environmental parameters and substitute them in CasADi functions.
    """
    def __init__(self, dot_chi = 0.0, v_r = 0.0, u_p = 0.0, dot_v_r = 0.0):
        """
        Initialize the environment with default or specified parameters.

        :param dot_chi: Kite heading angle rate (float).
        :param v_r: Radial velocity (float).
        :param theta_t: Tether pitch angle (float).
        :param dot_v_r: Radial velocity rate (float).
        """
        self.dot_chi = dot_chi
        self.v_r = v_r
        self.u_p = u_p
        self.dot_v_r = dot_v_r

    def apply(self, func):
        """
        Substitutes the environmental parameters (v_w, g, rho) in a given CasADi function.

        :param func: CasADi Function to modify.
        :return: A new CasADi Function without dependency on v_w, g, or rho.
        """
        # Get symbolic inputs and names
        symbolic_inputs = func.sx_in()
        symbolic_names = func.name_in()

        # Substitution map for v_w, g, rho
        substitutions = {
            'dot_chi': self.dot_chi,
            'v_r': self.v_r,
            'u_p': self.u_p,
            'dot_v_r': self.dot_v_r
        }

        # Substitute the parameters
        output_expr = func.call(symbolic_inputs)[0]
        for name, value in substitutions.items():
            if name in symbolic_names:
                index = symbolic_names.index(name)
                output_expr = ca.substitute(output_expr, symbolic_inputs[index], value)

        # Remove substituted variables from inputs
        remaining_inputs = [sym for i, sym in enumerate(symbolic_inputs) if symbolic_names[i] not in substitutions]
        remaining_names = [name for name in symbolic_names if name not in substitutions]

        # Create and return the new function
        return ca.Function(func.name() + '_env_applied', remaining_inputs, [output_expr], remaining_names, ['residual'])