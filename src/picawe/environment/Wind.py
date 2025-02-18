import casadi as ca
from picawe.utils.reference_frames import transformation_C_from_W
class Wind:
    def __init__(self, model='logarithmic'):
        self._speed_wind_ref = ca.SX.sym("speed_wind_ref")
        self._speed_friction = ca.SX.sym("speed_friction")
        self._height_ref = 10
        self.model = model
        self.kappa = 0.4
        self.z0 = 0.01

    @property
    def speed_wind_ref(self):
        return self._speed_wind_ref
    
    @speed_wind_ref.setter
    def speed_wind_ref(self, value):
        self._speed_friction = value*self.kappa/ca.log(self.height_ref/self.z0)
        self._speed_wind_ref = value

    @property
    def height_ref(self):
        return self._height_ref
    
    @height_ref.setter
    def height_ref(self, value):
        self._height_ref = value

    @property
    def speed_friction(self):
        return self._speed_friction
    
    @speed_friction.setter
    def speed_friction(self, value):
        self._speed_friction = value

    # Should be renamed to speed_wind_kite
    @property
    def speed_wind(self):
        if self.model == "uniform":
            return self.speed_wind_ref
        elif self.model == "logarithmic":
            return self._speed_friction/self.kappa * ca.log(self.z/self.z0)


    @property
    def velocity_wind(self):
        """
        Compute the wind velocity in the body frame.
        """
        T_C_from_W = transformation_C_from_W(self.angle_azimuth, self.angle_elevation, self.angle_course)
        return T_C_from_W @ ca.vertcat(self.speed_wind,0 ,0 )