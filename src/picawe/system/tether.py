import casadi as ca
import numpy as np
from picawe.utils.reference_frames import transformation_C_from_W
class Tether:
    def __init__(self, E = 132e9, diameter = 0.01, density = 970):
        self.E = E
        self.diameter_tether = diameter
        self.area_tether = np.pi * (self.diameter_tether / 2) ** 2
        self.drag_coefficient_tether = 1.1
        self.density_tether = density
        self._tension_tether_ground = ca.SX.sym("tension_tether_ground")


    @property
    def force_tether_at_kite(self):
        force_tension = ca.vertcat(0, 0, -self.tension_tether_ground)
        force_drag = self.drag_tether_at_kite
        force_gravity = self.force_gravity_tether_at_kite
        return force_tension #+ force_drag + force_gravity

    @property
    def drag_tether_at_kite(self):
        """
            Returns the product of drag coefficient and tether surface area dependent on the position of the tether end.
            See right side of eq.14 in Van Der Vlugt et al. (2019).
         """
        return 0.125 * self.drag_coefficient_tether * self.distance_radial * self.diameter_tether *self.rho*self.velocity_apparent_wind*ca.norm_2(self.velocity_apparent_wind)

    @property
    def force_gravity_tether_at_kite(self):
        weight = transformation_C_from_W(self.angle_azimuth, self.angle_elevation, self.angle_course)@ca.vertcat(0, 0, -self.mass_tether * self.g)
        return ca.vertcat(weight[0]/2, weight[1]/2, weight[2])

    @property
    def mass_tether(self):
        return self.density_tether * self.distance_radial * self.area_tether
    
    @property
    def tension_tether_ground(self):
        return self._tension_tether_ground