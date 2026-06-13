import casadi as ca
import numpy as np
from awetrim.utils.reference_frames import transformation_C_from_W
from abc import ABC, abstractmethod
from scipy import integrate


class Tether(ABC):
    def __init__(self, E=132e9, diameter=0.01, density=970):
        self.E = E
        self.diameter_tether = diameter
        self.area_tether = np.pi * (self.diameter_tether / 2) ** 2
        self.drag_coefficient_tether = 1.1
        self.density_tether = density

    @property
    def mass_tether(self):
        return self.mass_tether_for(self)

    def mass_tether_for(self, model):
        return self.density_tether * model.distance_radial * self.area_tether

    # ------------------------------------------------------------------
    # Tether load transferred to the kite. Link tethers model the tether as
    # a massless/dragless rigid or elastic link, so these are zero; lumped
    # and distributed tethers override them. Defining them here keeps the
    # ``drag_tether_at_kite`` / ``force_gravity_tether_at_kite`` expressions
    # (and ``tension_tether_equation``) valid for every tether type.
    # ------------------------------------------------------------------
    @property
    def drag_tether_at_kite(self):
        return self.drag_tether_at_kite_for(self)

    def drag_tether_at_kite_for(self, model):
        return ca.MX.zeros(3, 1)

    @property
    def force_gravity_tether_at_kite(self):
        return self.force_gravity_tether_at_kite_for(self)

    def force_gravity_tether_at_kite_for(self, model):
        return ca.MX.zeros(3, 1)

    # ------------------------------------------------------------------
    # Hooks for tethers that contribute their own decision variables and
    # equations to the system-level quasi-steady NLP. Defaults are empty
    # so simple tethers (RigidLinkTether, RigidLumpedTether, ...) keep the
    # existing behaviour, where their only decision (``tension_tether_ground``)
    # is discovered by name in ``SystemModel.default_unknown_vars``.
    # ------------------------------------------------------------------
    def decision_symbols_for(self, model):
        """Extra CasADi symbols the tether contributes to the joint NLP.

        Returned symbols are appended to the decision vector after the
        kite-state unknowns. Each symbol's ``.name()`` must match an entry
        in ``DEFAULT_BOUNDS`` or be supplied by ``decision_bounds_for``.
        """
        return []

    def extra_residuals_for(self, model):
        """Extra scalar equations the tether contributes to the joint NLP."""
        return ca.MX.zeros(0, 1)

    def decision_bounds_for(self, model, state_dict):
        """``{symbol_name: (lower, upper)}`` override for tether decisions
        whose bounds depend on the current numeric state (e.g. tether length
        scales with ``distance_radial``). Returning ``{}`` means fall back to
        ``DEFAULT_BOUNDS`` by name."""
        return {}

    def decision_initial_guess_for(self, model, state_dict):
        """``{symbol_name: x0}`` initial guesses for tether decisions, derived
        from the current ``state_dict`` (so a Williams shape starts close to
        the straight-line kite-to-ground configuration). Missing entries fall
        back to 1.0 in ``SystemModel.solve_quasi_steady``."""
        return {}

    def default_kite_state_unknowns(self):
        """Kite-state names ``SystemModel.default_unknown_vars`` should pick
        for this tether. Tethers that own their own tension symbol (Williams)
        omit ``tension_tether_ground``; tethers that expose it as a free
        symbol on the model (rigid/lumped) include it."""
        return ["speed_tangential", "timeder_angle_course", "tension_tether_ground"]


class RigidLinkTether(Tether):
    def __init__(self, E=132e9, diameter=0.01, density=970):
        super().__init__(E, diameter, density)
        self.tension_tether_ground = ca.MX.sym("tension_tether_ground")
        self.is_tether_rigid = True

    @property
    def force_tether_at_kite(self):
        return self.force_tether_at_kite_for(self)

    def force_tether_at_kite_for(self, model):
        force_tension = ca.vertcat(0, 0, -model.tension_tether_ground)
        return force_tension

    @property
    def tension_kite(self):
        return self.tension_tether_ground

    def tension_kite_for(self, model):
        return model.tension_tether_ground


class FlexibleLinkTether(Tether):
    def __init__(self, E=132e9, diameter=0.01, density=970):
        super().__init__(E, diameter, density)
        self.length_tether = ca.MX.sym("length_tether")
        self.timeder_length_tether = ca.MX.sym("timeder_length_tether")
        self.is_tether_rigid = False

    @property
    def force_tether_at_kite(self):
        return self.force_tether_at_kite_for(self)

    def force_tether_at_kite_for(self, model):
        force_tension = ca.vertcat(0, 0, -self.tension_kite_for(model))
        return force_tension

    @property
    def tension_kite(self):
        return self.tension_kite_for(self)

    def tension_kite_for(self, model):
        return ca.fmax(
            0,
            self.E
            * self.area_tether
            / model.length_tether
            * (model.distance_radial - model.length_tether),
        )

    @property
    def tension_tether_ground(self):
        return self.tension_kite

    def tension_tether_ground_for(self, model):
        return self.tension_kite_for(model)


class RigidLumpedTether(Tether):

    def __init__(self, E=132e9, diameter=0.01, density=970):
        super().__init__(E, diameter, density)
        self.tension_tether_ground = ca.MX.sym("tension_tether_ground")
        self.is_tether_rigid = True

    @property
    def force_tether_at_kite(self):
        return self.force_tether_at_kite_for(self)

    def force_tether_at_kite_for(self, model):
        force_tension = ca.vertcat(0, 0, -model.tension_tether_ground)
        force_drag = self.drag_tether_at_kite_for(model)
        force_gravity = self.force_gravity_tether_at_kite_for(model)
        return force_tension + force_drag + force_gravity

    @property
    def tension_kite(self):
        return ca.norm_2(self.force_tether_at_kite)

    def tension_kite_for(self, model):
        return ca.norm_2(self.force_tether_at_kite_for(model))

    @property
    def drag_tether_at_kite(self):
        return self.drag_tether_at_kite_for(self)

    def drag_tether_at_kite_for(self, model):
        """
        Returns the product of drag coefficient and tether surface area dependent on the position of the tether end.
        See right side of eq.14 in Van Der Vlugt et al. (2019).
        """
        drag = (
            0.125
            * self.drag_coefficient_tether
            * model.distance_radial
            * self.diameter_tether
            * model.rho
            * model.velocity_apparent_wind
            * ca.norm_2(model.velocity_apparent_wind)
        )
        # return drag
        return ca.vertcat(
            drag[0], drag[1], drag[2]
        )  # neglecting drag in the radial direction

    @property
    def force_gravity_tether_at_kite(self):
        return self.force_gravity_tether_at_kite_for(self)

    def force_gravity_tether_at_kite_for(self, model):
        weight = (
            -self.mass_tether_for(model)
            * model.g
            * ca.vertcat(
                ca.cos(model.angle_elevation) * ca.cos(model.angle_course),
                ca.cos(model.angle_elevation) * ca.sin(model.angle_course),
                ca.sin(model.angle_elevation),
            )
        )
        return ca.vertcat(weight[0] / 2, weight[1] / 2, weight[2])


class DistributedDragTether(Tether):

    def __init__(self, E=132e9, diameter=0.01, density=970):
        super().__init__(E, diameter, density)

    def force_tether_at_kite(self, model):
        force_tension = ca.vertcat(0, 0, -model.tension_tether_ground)
        force_drag = self.drag_tether_at_kite
        force_gravity = self.force_gravity_tether_at_kite
        return force_tension + force_drag(model) + force_gravity(model)

    def drag_tether_at_kite(self, model):
        """
        Returns the product of drag coefficient and tether surface area dependent on the position of the tether end.
        See right side of eq.14 in Van Der Vlugt et al. (2019).
        """

        def _velocity_wind_true_local(l):
            height = l * ca.sin(model.angle_elevation)
            return model.wind.velocity_wind_at_height(model, height)

        def _speed_wind_apparent_local(l):
            velocity_local = np.array(
                [
                    model.speed_tangential * l / model.distance_radial,
                    0,
                    model.speed_radial,
                ]
            )
            return np.linalg.norm(_velocity_wind_true_local(l) - velocity_local)

        r = model.distance_radial

        drag_integral_tangential = integrate.quad(
            lambda l: _speed_wind_apparent_local(l)
            * l
            * (_velocity_wind_true_local(l)[0] * r - model.speed_tangential * l),
            a=0,
            b=r,
        )[0]
        drag_integral_normal = integrate.quad(
            lambda l: _speed_wind_apparent_local(l)
            * l
            * _velocity_wind_true_local(l)[1],
            a=0,
            b=r,
        )[0]

        drag_integral_radial = integrate.quad(
            lambda l: _speed_wind_apparent_local(l)
            * (_velocity_wind_true_local(l)[2] - model.speed_radial),
            a=0,
            b=r,
        )[0]

        return (
            0.5
            * model.rho
            * self.diameter_tether
            * self.drag_coefficient_tether
            * np.array(
                [
                    drag_integral_tangential / (r**2),
                    drag_integral_normal / r,
                    drag_integral_radial,
                ]
            )
        )

    def force_gravity_tether_at_kite(self, model):
        weight = transformation_C_from_W(
            model.angle_azimuth, model.angle_elevation, model.angle_course
        ) @ ca.vertcat(0, 0, -self.mass_tether_for(model) * model.g)
        return ca.vertcat(weight[0] / 2, weight[1] / 2, weight[2])


class FlexibleLumpedTether(Tether):

    def __init__(self, E=132e9, diameter=0.01, density=970):
        super().__init__(E, diameter, density)
        self.length_tether = ca.MX.sym("length_tether")
        self.timeder_length_tether = ca.MX.sym("timeder_length_tether")
        self.is_tether_rigid = False

    @property
    def force_tether_at_kite(self):
        return self.force_tether_at_kite_for(self)

    def force_tether_at_kite_for(self, model):
        force_tension = ca.vertcat(0, 0, -self.tension_kite_for(model))
        force_drag = self.drag_tether_at_kite_for(model)
        force_gravity = self.force_gravity_tether_at_kite_for(model)
        return force_tension + force_drag + force_gravity

    @property
    def tension_kite(self):
        return self.tension_kite_for(self)

    def tension_kite_for(self, model):
        return ca.fmax(
            0,
            self.E
            * self.area_tether
            / model.length_tether
            * (model.distance_radial - model.length_tether),
        )

    @property
    def tension_tether_ground(self):
        return (
            self.tension_kite
            - self.drag_tether_at_kite[2]
            - self.force_gravity_tether_at_kite[2]
        )

    def tension_tether_ground_for(self, model):
        return (
            self.tension_kite_for(model)
            - self.drag_tether_at_kite_for(model)[2]
            - self.force_gravity_tether_at_kite_for(model)[2]
        )

    @property
    def drag_tether_at_kite(self):
        return self.drag_tether_at_kite_for(self)

    def drag_tether_at_kite_for(self, model):
        """
        Returns the product of drag coefficient and tether surface area dependent on the position of the tether end.
        See right side of eq.14 in Van Der Vlugt et al. (2019).
        """
        drag = (
            0.125
            * self.drag_coefficient_tether
            * model.distance_radial
            * self.diameter_tether
            * model.rho
            * model.velocity_apparent_wind
            * ca.norm_2(model.velocity_apparent_wind)
        )
        # return drag
        return ca.vertcat(
            drag[0], drag[1], drag[2]
        )  # neglecting drag in the radial direction

    @property
    def force_gravity_tether_at_kite(self):
        return self.force_gravity_tether_at_kite_for(self)

    def force_gravity_tether_at_kite_for(self, model):
        weight = transformation_C_from_W(
            model.angle_azimuth, model.angle_elevation, model.angle_course
        ) @ ca.vertcat(0, 0, -self.mass_tether_for(model) * model.g)
        return ca.vertcat(weight[0] / 2, weight[1] / 2, weight[2])
