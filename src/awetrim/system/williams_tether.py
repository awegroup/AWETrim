# Copyright (c) 2023-2026 Oriol Cayon, Delft University of Technology
#
# Licensed under the EUPL, Version 1.2 or - as soon they will be approved by
# the European Commission - subsequent versions of the EUPL (the "Licence");
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at:
#
#     https://joinup.ec.europa.eu/software/page/eupl
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the Licence is distributed on an "AS IS" basis,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the Licence for the specific language governing permissions and
# limitations under the Licence.
#
# SPDX-License-Identifier: EUPL-1.2

from awetrim.system.tether import Tether
from awetrim.utils.reference_frames import (
    transformation_C_from_Wind,
    transformation_Wind_from_C,
)
import casadi as ca


def _angle_between(a, b):
    """Robust angle between two CasADi vectors using atan2(|cross|, dot).

    Unlike ``arccos(dot / (|a| |b|))`` this stays differentiable when the
    vectors are parallel (the arccos form has a -inf derivative at cos=1,
    which corrupts analytic Jacobians used by the least-squares solver).
    """
    return ca.arctan2(ca.norm_2(ca.cross(a, b)), ca.dot(a, b))


class WilliamsTether(Tether):
    """Williams distributed-mass tether iterated from kite -> ground.

    Decision variables (4) owned by the tether:
      - ``elevation_last_element`` and ``azimuth_last_element``: direction of
        the kite-side segment in the wind frame.
      - ``tether_length``: unstrained total tether length.
      - ``tension_tether_kite``: tension magnitude at the kite end.

    Extra residuals (3): the iterated ground node must hit ``(0, 0, 0)``.

    Environment (wind, rho, g, omega) is never stored on the tether. The
    ``_for(model)`` methods read everything off ``model``; the standalone
    entry points (``residual_function`` etc.) take an explicit ``env``
    argument so external solvers can supply env values without mutating the
    tether instance.
    """

    def __init__(
        self,
        E=132e9,
        diameter=0.01,
        density=970,
        n_elements=30,
        elastic=False,
        cf=0.01,
    ):
        super().__init__(E, diameter, density)
        self.n_elements = n_elements
        self.elastic = elastic
        self.cf = cf
        self.is_tether_rigid = True
        # Axial stiffness E·A used by the elastic stretch model below.
        self.EA = self.E * self.area_tether

        self.elevation_last_element = ca.MX.sym("elevation_last_element")
        self.azimuth_last_element = ca.MX.sym("azimuth_last_element")
        self.tether_length = ca.MX.sym("tether_length")
        self.tension_tether_kite = ca.MX.sym("tension_tether_kite")
        # Standalone callers supply r_kite and force_kite_resultant as numerical
        # parameters; these symbols are the placeholders the symbolic graph
        # closes over.
        self.r_kite_sym = ca.MX.sym("r_kite", 3)
        self.force_kite_resultant_sym = ca.MX.sym("force_kite_resultant", 3)

    # ------------------------------------------------------------------
    # Tether base-class hooks: how this tether plugs into the joint NLP.
    # ------------------------------------------------------------------
    def decision_symbols_for(self, model):
        return [
            self.tension_tether_kite,
            self.tether_length,
            self.azimuth_last_element,
            self.elevation_last_element,
        ]

    def extra_residuals_for(self, model):
        return self.tether_shape_symbolic_for(model)["ground_position"]

    def decision_bounds_for(self, model, state_dict):
        # Length bounds scale with the kite-to-ground distance, taken from
        # the numeric ``state_dict`` so the caller gets floats (not symbolic
        # expressions). The rest fall back to DEFAULT_BOUNDS by name.
        distance_radial = float(state_dict.get("distance_radial", 200.0))
        return {
            "tether_length": (
                0.5 * distance_radial,
                3.0 * distance_radial,
            ),
        }

    def decision_initial_guess_for(self, model, state_dict):
        # ``dict.get(key, default)`` returns the stored value when the key
        # exists, even if that value is ``None`` — so we have to coalesce
        # explicitly.
        def _coalesce(name, default):
            value = state_dict.get(name)
            return float(default if value is None else value)

        distance_radial = _coalesce("distance_radial", 200.0)
        return {
            "tension_tether_kite": 1.0e5,
            "tether_length": 1.02 * distance_radial,
            "azimuth_last_element": 0.0,
            "elevation_last_element": _coalesce("angle_elevation", 0.0),
        }

    def default_kite_state_unknowns(self):
        # The tension role is filled by the tether-owned ``tension_tether_kite``
        # decision symbol contributed via ``decision_symbols_for``; the joint
        # NLP must not also list ``tension_tether_ground`` as a free unknown.
        return ["speed_tangential", "timeder_angle_course"]

    # ------------------------------------------------------------------
    # Model-coupled API: pulls env (wind, rho, g, omega) and kinematics
    # (r_kite, frame angles) off ``model``.
    # ------------------------------------------------------------------
    def tether_shape_symbolic_for(self, model):
        """Symbolic tether geometry expressed against ``model`` state.

        Builds ``r_kite`` and ``omega`` in the wind frame from the model's
        kinematics, then delegates to the standalone ``tether_shape_symbolic``
        with the tether-owned ``tension_tether_kite`` as the kite-end
        boundary.
        """
        direction_wind = getattr(model.wind, "direction_wind", 0.0)
        T_Wind_from_C = transformation_Wind_from_C(
            model.angle_azimuth,
            model.angle_elevation,
            model.angle_course,
            direction_wind,
        )
        r_kite_course = ca.vertcat(0.0, 0.0, model.distance_radial)
        r_kite_wind = T_Wind_from_C @ r_kite_course
        omega_wind = T_Wind_from_C @ model.velocity_rotation_course_frame
        return self.tether_shape_symbolic(
            env=model,
            r_kite=r_kite_wind,
            tension_kite=self.tension_tether_kite,
            omega=omega_wind,
        )

    def force_tether_at_kite_for(self, model):
        """Last-segment tether force on the kite, in the COURSE frame.

        Uses the tether-owned direction symbols and tension magnitude. The
        joint NLP's outer force balance pins these to the trim solution.
        """
        dir_last_wind = ca.vertcat(
            ca.cos(self.elevation_last_element) * ca.cos(self.azimuth_last_element),
            ca.cos(self.elevation_last_element) * ca.sin(self.azimuth_last_element),
            ca.sin(self.elevation_last_element),
        )
        direction_wind = getattr(model.wind, "direction_wind", 0.0)
        dir_last_course = transformation_C_from_Wind(
            model.angle_azimuth,
            model.angle_elevation,
            model.angle_course,
            direction_wind,
        ) @ dir_last_wind
        return -self.tension_tether_kite * dir_last_course

    def tension_tether_ground_for(self, model):
        shape = self.tether_shape_symbolic_for(model)
        return ca.norm_2(shape["tether_force_ground"])

    def tether_shape_symbolic(
        self,
        env,
        r_kite=None,
        force_kite_resultant=None,
        tension_kite=None,
        kite_tension_vector=None,
        omega=None,
        tether_length=None,
        elevation_last=None,
        azimuth_last=None,
    ):
        """Build symbolic tether geometry by iterating kite -> ground.

        Parameters
        ----------
        env : object exposing ``wind`` (with ``speed_wind_at_height(z)`` and
            ``z0``), ``rho`` and ``g``. ``model`` works directly.
        r_kite : (3,) — kite attachment in the WIND frame. Defaults to the
            placeholder symbol ``self.r_kite_sym``.
        omega : (3,) — angular velocity in the WIND frame. Defaults to zero.

        Three ways to specify the kite-side boundary, in order of preference:
          1. ``kite_tension_vector`` (3-vector): full kite-end tether vector.
             Eliminates the direction unknowns. Use in coupled solves where
             the outer kite force balance is baked in.
          2. ``tension_kite`` (scalar) + free direction symbols (default).
          3. ``force_kite_resultant`` (3-vector): convenience for standalone
             shape solves where ``T_mag = ||F_resultant||``.
        """

        if r_kite is None:
            r_kite = self.r_kite_sym
        if (
            kite_tension_vector is None
            and tension_kite is None
            and force_kite_resultant is None
        ):
            force_kite_resultant = self.force_kite_resultant_sym
        if omega is None:
            omega = ca.DM.zeros(3)

        tether_length = self.tether_length if tether_length is None else tether_length
        diameter = self.diameter_tether
        density = self.density_tether
        cdt = self.drag_coefficient_tether

        N = self.n_elements
        l_unstrained = tether_length / N
        m_s = ca.pi * diameter**2 / 4 * l_unstrained * density

        omega = ca.reshape(omega, 3, 1)
        rho = env.rho if env is not None else 1.225
        g = env.g if env is not None else 9.81
        wind = env.wind if env is not None else None

        if kite_tension_vector is not None:
            # Collapsed mode: F_resultant = T_mag * dir_last is given directly,
            # so the kite-end tether boundary is fully determined and the only
            # remaining tether unknown is tether_length.
            T_kite_mag = ca.norm_2(kite_tension_vector)
            dir_last = kite_tension_vector / T_kite_mag
        else:
            elevation_last = (
                self.elevation_last_element if elevation_last is None else elevation_last
            )
            azimuth_last = self.azimuth_last_element if azimuth_last is None else azimuth_last
            dir_last = ca.vertcat(
                ca.cos(elevation_last) * ca.cos(azimuth_last),
                ca.cos(elevation_last) * ca.sin(azimuth_last),
                ca.sin(elevation_last),
            )
            T_kite_mag = (
                tension_kite
                if tension_kite is not None
                else ca.norm_2(force_kite_resultant)
            )

        tensions = ca.MX.zeros((N, 3))
        positions = ca.MX.zeros((N + 1, 3))

        # Boundary at the kite (node N).
        positions[N, 0] = r_kite[0]
        positions[N, 1] = r_kite[1]
        positions[N, 2] = r_kite[2]

        tensions[N - 1, 0] = T_kite_mag * dir_last[0]
        tensions[N - 1, 1] = T_kite_mag * dir_last[1]
        tensions[N - 1, 2] = T_kite_mag * dir_last[2]

        if self.elastic:
            l_s = (T_kite_mag / self.EA + 1) * l_unstrained
        else:
            l_s = l_unstrained

        # Step one segment down from the kite to node N-1.
        positions[N - 1, 0] = positions[N, 0] - dir_last[0] * l_s
        positions[N - 1, 1] = positions[N, 1] - dir_last[1] * l_s
        positions[N - 1, 2] = positions[N, 2] - dir_last[2] * l_s

        velocities_apparent_wind = ca.MX.zeros((N + 1, 1))
        angle_va_tether = ca.MX.zeros((N + 1, 1))
        # Per-node force decomposition (half-segment aero attributed to each
        # node, full-segment gravity lumped onto the node). Rows 0 and N
        # remain zero (boundary nodes — no force balance is solved there).
        drag_per_node = ca.MX.zeros((N + 1, 3))
        lift_per_node = ca.MX.zeros((N + 1, 3))
        gravity_per_node = ca.MX.zeros((N + 1, 3))
        stretched_tether_length = l_s

        # Iterate from node n = N-1 down to n = 1, computing the tension in
        # the segment immediately below (segment n-1) and the position of the
        # next node below (node n-1).
        for n in range(N - 1, 0, -1):
            # Kinematics at lumped mass node n.
            v_n = ca.cross(omega, positions[n, :].T)
            a_n = ca.cross(omega, v_n)

            # Local segment direction is the segment immediately above (already
            # known), since the segment below is what we are solving for.
            t_above = tensions[n, :].T
            t_above_norm = ca.norm_2(t_above) + 1e-9
            ej = t_above / t_above_norm

            # Apparent wind at this node (true wind - segment velocity).
            # ``wind`` and ``rho`` are pulled off the env passed in by the
            # caller; the tether instance never owns them.
            if wind is not None:
                z0 = getattr(wind, "z0", 0.01)
                z_local = ca.fmax(positions[n, 2], z0 * 1.001)
                u_wind = wind.speed_wind_at_height(z_local)
            else:
                u_wind = 0.0
            vwn = u_wind * ca.vertcat(1, 0, 0)
            van = vwn - v_n

            theta = _angle_between(van, ej)
            cd_t = cdt * ca.sin(theta) ** 3 + ca.pi * self.cf * ca.cos(theta) ** 3
            cl_t = (
                cdt * ca.sin(theta) ** 2 * ca.cos(theta)
                - ca.pi * self.cf * ca.sin(theta) * ca.cos(theta) ** 2
            )
            van_norm = ca.norm_2(van) + 1e-9
            dir_D = van / van_norm
            dir_L = -(ej - ca.dot(ej, dir_D) * dir_D)
            dyn_pressure_area = 0.5 * rho * van_norm**2 * l_unstrained * diameter

            velocities_apparent_wind[n, :] = ca.norm_2(van)
            angle_va_tether[n, :] = theta

            lift_n = dyn_pressure_area * cl_t * dir_L
            drag_n = dyn_pressure_area * cd_t * dir_D

            # Half-segment aero attributed to this node (mirrors original).
            fa_n = 0.5 * (drag_n + lift_n)

            fg_n = ca.vertcat(0, 0, -m_s * g)

            # Record per-node force contributions for diagnostics / plotting.
            drag_per_node[n, 0] = 0.5 * drag_n[0]
            drag_per_node[n, 1] = 0.5 * drag_n[1]
            drag_per_node[n, 2] = 0.5 * drag_n[2]
            lift_per_node[n, 0] = 0.5 * lift_n[0]
            lift_per_node[n, 1] = 0.5 * lift_n[1]
            lift_per_node[n, 2] = 0.5 * lift_n[2]
            gravity_per_node[n, 0] = fg_n[0]
            gravity_per_node[n, 1] = fg_n[1]
            gravity_per_node[n, 2] = fg_n[2]

            # Force balance at node n: m*a = -T_{n-1} + T_n + f_g + f_a
            # Solve for the tension in the segment BELOW node n.
            t_below = t_above + fg_n + fa_n - m_s * a_n
            tensions[n - 1, 0] = t_below[0]
            tensions[n - 1, 1] = t_below[1]
            tensions[n - 1, 2] = t_below[2]

            t_below_norm = ca.norm_2(t_below) + 1e-9
            if self.elastic:
                l_s = (t_below_norm / self.EA + 1) * l_unstrained
            else:
                l_s = l_unstrained
            stretched_tether_length += l_s

            dir_below = t_below / t_below_norm
            positions[n - 1, 0] = positions[n, 0] - dir_below[0] * l_s
            positions[n - 1, 1] = positions[n, 1] - dir_below[1] * l_s
            positions[n - 1, 2] = positions[n, 2] - dir_below[2] * l_s

        # Tension in the last (kite-side) segment, with sign convention that
        # this vector is the force the tether exerts on the kite (= +T_{N-1}
        # since the segment pulls the kite toward the tether direction; the
        # external resultant on the kite must balance it).
        tension_kite = -tensions[N - 1, :].T

        res = {
            "kite_position": positions[N, :].T,
            "ground_position": positions[0, :].T,
            "tether_force_kite": tension_kite,
            "tether_force_ground": tensions[0, :].T,
            "tether_length_stretched": stretched_tether_length,
            "positions": positions,
            "tensions": tensions,
            "velocities_apparent_wind": velocities_apparent_wind,
            "angle_va_tether": angle_va_tether,
            "drag_per_node": drag_per_node,
            "lift_per_node": lift_per_node,
            "gravity_per_node": gravity_per_node,
        }
        return res

    def objective_function(
        self, env, r_kite=None, force_kite_resultant=None, omega=None
    ):
        """Residuals: the computed ground node must hit the origin."""
        shape = self.tether_shape_symbolic(
            env=env,
            r_kite=r_kite,
            force_kite_resultant=force_kite_resultant,
            omega=omega,
        )
        return shape["ground_position"]

    @property
    def decision_symbols(self):
        """The three unknowns of the kite->ground iteration."""
        return [
            self.elevation_last_element,
            self.azimuth_last_element,
            self.tether_length,
        ]

    def _split_symbols(self, expr):
        """Return (decision_syms_in_expr_order, param_syms) appearing in expr."""
        decision_set = {sym.name() for sym in self.decision_symbols}
        all_syms = ca.symvar(expr)
        params = [sym for sym in all_syms if sym.name() not in decision_set]
        return self.decision_symbols, params

    def residual_symbolic(
        self, env, r_kite=None, force_kite_resultant=None, omega=None
    ):
        """Return ``(residual_expr, param_syms)`` for the 3-vector
        ground-anchor residual. ``param_syms`` are the free CasADi symbols
        the caller must pack into ``p`` when invoking the solver."""
        residual = self.objective_function(
            env=env,
            r_kite=r_kite,
            force_kite_resultant=force_kite_resultant,
            omega=omega,
        )
        _, param_syms = self._split_symbols(residual)
        return residual, param_syms

    def residual_function(
        self, env, r_kite=None, force_kite_resultant=None, omega=None
    ):
        """``ca.Function`` mapping ``(x, p) -> residual`` where ``x``
        concatenates ``decision_symbols``. Also returns the ordered parameter
        names so the caller can pack ``p`` consistently.
        """
        residual, param_syms = self.residual_symbolic(
            env=env,
            r_kite=r_kite,
            force_kite_resultant=force_kite_resultant,
            omega=omega,
        )
        x = ca.vertcat(*self.decision_symbols)
        p = ca.vertcat(*param_syms) if param_syms else ca.SX.sym("p_empty", 0)
        fun = ca.Function(
            "williams_residual",
            [x, p],
            [residual],
            ["x", "p"],
            ["residual"],
        )
        param_names = [sym.name() for sym in param_syms]
        return fun, param_names

    def residual_jacobian_function(
        self, env, r_kite=None, force_kite_resultant=None, omega=None
    ):
        """``ca.Function`` for the 3x3 Jacobian of the ground residual w.r.t.
        the three direction+length unknowns."""
        residual, param_syms = self.residual_symbolic(
            env=env,
            r_kite=r_kite,
            force_kite_resultant=force_kite_resultant,
            omega=omega,
        )
        x = ca.vertcat(*self.decision_symbols)
        p = ca.vertcat(*param_syms) if param_syms else ca.SX.sym("p_empty", 0)
        jac = ca.jacobian(residual, x)
        fun = ca.Function(
            "williams_residual_jac",
            [x, p],
            [jac],
            ["x", "p"],
            ["jac"],
        )
        param_names = [sym.name() for sym in param_syms]
        return fun, param_names

    def shape_function(
        self, env, r_kite=None, force_kite_resultant=None, omega=None
    ):
        """``ca.Function(x, p) -> (positions, tensions)`` for post-processing
        once the standalone solver has converged."""
        shape = self.tether_shape_symbolic(
            env=env,
            r_kite=r_kite,
            force_kite_resultant=force_kite_resultant,
            omega=omega,
        )
        positions_sym = shape["positions"]
        tensions_sym = shape["tensions"]
        _, param_syms = self._split_symbols(
            ca.vertcat(
                ca.reshape(positions_sym, -1, 1), ca.reshape(tensions_sym, -1, 1)
            )
        )
        x = ca.vertcat(*self.decision_symbols)
        p = ca.vertcat(*param_syms) if param_syms else ca.SX.sym("p_empty", 0)
        fun = ca.Function(
            "williams_shape",
            [x, p],
            [positions_sym, tensions_sym],
            ["x", "p"],
            ["positions", "tensions"],
        )
        param_names = [sym.name() for sym in param_syms]
        return fun, param_names

    def solve_tether_shape(
        self, env, r_kite=None, force_kite_resultant=None, omega=None
    ):
        """IPOPT NLP solver for the standalone three-unknown problem."""
        solver_opts = {
            "ipopt": {
                "print_level": 0,
                "sb": "yes",
            },
            "print_time": False,
        }
        residual, param_syms = self.residual_symbolic(
            env=env,
            r_kite=r_kite,
            force_kite_resultant=force_kite_resultant,
            omega=omega,
        )
        nlp = {
            "x": ca.vertcat(*self.decision_symbols),
            "p": ca.vertcat(*param_syms) if param_syms else ca.SX.sym("p_empty", 0),
            "f": 0,
            "g": residual,
        }
        solver = ca.nlpsol("solver", "ipopt", nlp, solver_opts)
        param_names = [sym.name() for sym in param_syms]
        return solver, param_names
