import casadi as ca
import numpy as np
class Tether:
    def __init__(self, E = 132e9, diameter = 0.014):
        self.E = E
        self.diameter = diameter
        self.A = np.pi * (self.diameter / 2) ** 2
        self.define_symbolic_variables_tether()

    def define_symbolic_variables_tether(self):
        """
        Define symbolic variables used in the model.
        """
        base_symbolic_variables = {
            'length_tether': 'length_tether',
        }
        for var_name in base_symbolic_variables.keys():
            setattr(self, var_name, ca.SX.sym(var_name))

    @property
    def tension_tether(self):
        return self.E * self.A / self.length_tether * (self.distance_radial - self.length_tether)
    @property
    def force_tether(self):
        return ca.vertcat(0, 0, -self.tension_tether)
