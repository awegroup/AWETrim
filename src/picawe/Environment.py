import casadi as ca

class Environment:
    """
    A class to manage environmental parameters and substitute them in CasADi functions.
    """
    def __init__(self, v_w=None, g=9.81, rho=1.225):
        """
        Initialize the environment with default or specified parameters.

        :param v_w: Wind speed magnitude (float).
        :param g: Gravitational acceleration (float).
        :param rho: Air density (float).
        """
        self.v_w = v_w
        self.g = g
        self.rho = rho

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
            'v_w': self.v_w,
            'g': self.g,
            'rho': self.rho
        }

        # Substitute the parameters
        output_expr = func.call(symbolic_inputs)[0]
        false_names = []
        for name, value in substitutions.items():
            if name in symbolic_names:
                if value is None:
                    false_names.append(name)
                else:
                    index = symbolic_names.index(name)
                    output_expr = ca.substitute(output_expr, symbolic_inputs[index], value)
            
        for false_name in false_names:
            substitutions.pop(false_name)
        # Remove substituted variables from inputs
        remaining_inputs = [sym for i, sym in enumerate(symbolic_inputs) if symbolic_names[i] not in substitutions]
        remaining_names = [name for name in symbolic_names if name not in substitutions]

        # Create and return the new function
        return ca.Function(func.name() + '_env_applied', remaining_inputs, [output_expr], remaining_names, ['residual'])
    