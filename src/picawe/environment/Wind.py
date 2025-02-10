import casadi as ca
from picawe.utils.reference_frames import transformation_C_from_W
class Wind:
    def __init__(self):
        self.define_symbolic_variables_wind()

    def define_symbolic_variables_wind(self):
        """
        Define symbolic variables used in the model.
        """
        base_symbolic_variables = {
            'speed_wind': 'speed_wind',
        }
        for var_name in base_symbolic_variables.keys():
            setattr(self, var_name, ca.SX.sym(var_name))

    @property
    def velocity_wind(self):
        """
        Compute the wind velocity in the body frame.
        """
        T_C_from_W = transformation_C_from_W(self.angle_azimuth, self.angle_elevation, self.angle_course)
        return T_C_from_W @ ca.vertcat(self.speed_wind,0 ,0 )