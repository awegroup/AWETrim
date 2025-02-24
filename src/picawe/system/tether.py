import casadi as ca
import numpy as np
from picawe.utils.reference_frames import transformation_C_from_W
from abc import ABC, abstractmethod
from scipy import integrate


class Tether(ABC):
    def __init__(self, E=132e9, diameter=0.01, density=970):
        self.E = E
        self.diameter_tether = diameter
        self.area_tether = np.pi * (self.diameter_tether / 2) ** 2
        self.drag_coefficient_tether = 1.1
        self.density_tether = density
        self.tension_tether_ground = ca.SX.sym("tension_tether_ground")

    @property
    @abstractmethod
    def force_tether_at_kite(self):
        pass

    def mass_tether(self,state):
        return self.density_tether * state.distance_radial * self.area_tether

class RigidLinkTether(Tether):
    def __init__(self, E=132e9, diameter=0.01, density=970):
        super().__init__(E, diameter, density)

    def force_tether_at_kite(self,state):
        force_tension = ca.vertcat(0, 0, -state.tension_tether_ground)
        return force_tension

class RigidLumpedTether(Tether):

    def __init__(self, E=132e9, diameter=0.01, density=970):
        super().__init__(E, diameter, density)


    def force_tether_at_kite(self,state):
        force_tension = ca.vertcat(0, 0, -state.tension_tether_ground)
        force_drag = self.drag_tether_at_kite
        force_gravity = self.force_gravity_tether_at_kite
        return force_tension + force_drag(state) + force_gravity(state)
    
    def tension_kite(self,state):
        return ca.norm_2(self.force_tether_at_kite(state))

    def drag_tether_at_kite(self, state):
        """
        Returns the product of drag coefficient and tether surface area dependent on the position of the tether end.
        See right side of eq.14 in Van Der Vlugt et al. (2019).
        """
        drag = (
            0.125
            * self.drag_coefficient_tether
            * state.distance_radial
            * self.diameter_tether
            * state.rho
            * state.velocity_apparent_wind
            * ca.norm_2(state.velocity_apparent_wind)
        )
        # return drag
        return ca.vertcat(drag[0], drag[1], 0) # neglecting drag in the radial direction

    def force_gravity_tether_at_kite(self,state):
        weight = transformation_C_from_W(
            state.angle_azimuth, state.angle_elevation, state.angle_course
        ) @ ca.vertcat(0, 0, -self.mass_tether(state) * state.g)
        return ca.vertcat(weight[0] / 2, weight[1] / 2, weight[2])

class DistributedDragTether(Tether):

    def __init__(self, E=132e9, diameter=0.01, density=970):
        super().__init__(E, diameter, density)


    def force_tether_at_kite(self,state):
        force_tension = ca.vertcat(0, 0, -state.tension_tether_ground)
        force_drag = self.drag_tether_at_kite
        force_gravity = self.force_gravity_tether_at_kite
        return force_tension + force_drag(state) + force_gravity(state)

    def drag_tether_at_kite(self,state):
        """
        Returns the product of drag coefficient and tether surface area dependent on the position of the tether end.
        See right side of eq.14 in Van Der Vlugt et al. (2019).
        """

        def _velocity_wind_true_local(l):
            height = l *ca.sin(state.angle_elevation)
            return state.wind.velocity_wind_at_height(state,height)

        def _speed_wind_apparent_local(l):
            velocity_local = np.array([
                state.speed_tangential * l / state.distance_radial,
                0,
                state.speed_radial
            ])
            return np.linalg.norm(_velocity_wind_true_local(l) - velocity_local)

        r = state.distance_radial

        drag_integral_tangential = integrate.quad(
            lambda l: _speed_wind_apparent_local(l)* l * (_velocity_wind_true_local(l)[0] * r - state.speed_tangential * l), a=0, b=r
        )[0]
        drag_integral_normal = integrate.quad(
            lambda l: _speed_wind_apparent_local(l) * l * _velocity_wind_true_local(l)[1], a=0, b=r
        )[0]

        drag_integral_radial = integrate.quad(
            lambda l: _speed_wind_apparent_local(l) * (_velocity_wind_true_local(l)[2] - state.speed_radial), a=0, b=r
        )[0]

        return 0.5 * state.rho * self.diameter_tether * self.drag_coefficient_tether * np.array([
            drag_integral_tangential / (r**2),
            drag_integral_normal / r,
            drag_integral_radial
        ]) 

    def force_gravity_tether_at_kite(self,state):
        weight = transformation_C_from_W(
            state.angle_azimuth, state.angle_elevation, state.angle_course
        ) @ ca.vertcat(0, 0, -self.mass_tether(state) * state.g)
        return ca.vertcat(weight[0] / 2, weight[1] / 2, weight[2])