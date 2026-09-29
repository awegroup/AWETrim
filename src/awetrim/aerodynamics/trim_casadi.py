"""Quasi-steady VSM trim as one CasADi root-finding problem.

The unknowns are the five trim states of
:func:`~awetrim.aerodynamics.vsm_quasi_steady.solve_vsm_quasi_steady_trim`
``[speed_tangential, roll_deg, pitch_deg, yaw_deg, course_rate]``, the panel
circulations ``gamma`` and, with the Williams tether, the unstrained tether
length. The residual stacks the force and moment rows that trim solves (about
the reference point, nondimensionalised the same way), the Williams ground
closure when present, and the VSM circulation residual
``(I - diag(mu) L) gamma - G_raw(gamma) = 0`` (the ``casadi_newton`` loop's
form, viscosity included). One Newton iteration with the exact Jacobian
replaces a finite-difference column of the old outer solver, each of which
cost a full inner VSM solve.

What is symbolic and what is not
--------------------------------
* The course-frame kinematics (apparent wind, transport rate ``Omega_C``,
  transport acceleration, gravity) are the CasADi ``SystemModel``'s own
  expressions with the trim unknowns substituted in -- no numeric evaluation
  per call, no restated formula. The Williams tether shape is
  ``WilliamsTether.tether_shape_symbolic`` called on the symbolic kite-end
  resultant, so the ground closure and its derivatives come from the tether
  model itself.
* The wing is kept in its BODY frame (the geometry handed in, unrotated) and
  the inflow is rotated into it, ``va_b = R^T va_w``, ``omega_b = R^T
  omega_w``, instead of rotating the panels about the reference point as the
  NumPy trim does. The two are the same rigid configuration, so the wing
  forces are identical and moments about the reference point (the rotation
  origin) transport with ``M_w = R M_b``. One consequence: the bridle lines
  rotate WITH the wing here, whereas the NumPy trim leaves them fixed in the
  world frame while it rotates the sections -- a small inconsistency there
  (4 N and 22 N m at 1 deg of pitch on the LEI-V3, 0.05 % in CL).
* The section physics -- effective angle of attack, the 2-D polars, VSM's
  lift/drag/moment construction, bridle line drag, KCU drag -- come from the
  ``xp`` kernels in :mod:`awetrim.aerodynamics.panel_kernels`,
  :mod:`awetrim.aerodynamics.kcu_drag` and
  :mod:`awetrim.aerodynamics.apparent_wind`, and the circulation residual and
  polar lookups from VSM's own CasADi panel function
  (``Solver._build_casadi_newton_function``), so nothing here restates a
  force law. Reported ``cl``/``cd``/``cs`` use VSM's bookkeeping (each panel
  in its local inflow frame), so they are comparable with every stored trim.
* The induction (AIC) matrices stay NUMERIC. They depend on the trim
  unknowns only through the trailing-wake direction, so they are frozen per
  outer pass and rebuilt at the converged inflow; the outer loop repeats until
  the rebuilt matrices leave the residual below tolerance (two to four passes
  of a few iterations each).
* The GEOMETRY is a numeric PARAMETER vector of the graph (panel control
  points, aerodynamic centres, airfoil axes, chords, widths, spanwise
  direction, projected area, bridle segment endpoints and diameters, centre
  of gravity), so one build serves every deformed shape that keeps its
  polars and its panel/bridle counts: :meth:`CasadiTrim.update_geometry`
  swaps the shape in without rebuilding (the aerostructural coupling calls
  the trim once per coupled iteration on a moving shape; the build is ~0.5 s
  at 45 panels, a warm re-solve ~0.05 s).

Solver
------
Newton with Armijo backtracking, globalised by pseudo-transient continuation
and using the corner-averaged polar Jacobian, exactly as the ``casadi_newton``
circulation loop (see its docstring for why: the piecewise-linear polars make
the residual kinked at the Cl-max corner and the viscosity gate is a genuine
discontinuity). ``polar_interpolation="bspline"`` removes both kinks at the
cost of a slightly different fixed point. IPOPT is deliberately not used: the
system is square and unconstrained, and a merit-based line search is what
stalls at the kinks. Bounds on the trim states and the tether length are
enforced by projection.

Not covered (use the NumPy solvers): ``tether_model="rigid_lumped"`` and
``reel_speed_along_tether``.
"""
from __future__ import annotations

import copy
import hashlib
import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import casadi as ca
import numpy as np

from awetrim.aerodynamics import panel_kernels as pk
from awetrim.aerodynamics.apparent_wind import apparent_wind_at
from awetrim.aerodynamics.kcu_drag import KcuDragModel
from awetrim.aerodynamics.protocols import AWETrimSystemModel, AxisDefinition
from awetrim.aerodynamics.vsm_quasi_steady import (
    DEFAULT_AXES,
    DEFAULT_BOUNDS_LOWER,
    DEFAULT_BOUNDS_UPPER,
    DEFAULT_STATION_KCU,
    DEFAULT_TRANSFORMATION_C_FROM_VSM,
    _as_3vector,
    _as_5vector,
    _baseline_geometry,
    _kcu_drag_fields,
    _numeric_value_for_symbol,
    _set_body_attitude_from_baseline,
    _set_course_rate_body,
    _system_model_mass_total,
    _ZeroGravityEnv,
    stall_fraction_fields,
)

__all__ = ["CasadiTrimOptions", "CasadiTrim", "solve_vsm_trim_casadi"]

_LOG = logging.getLogger(__name__)


@dataclass
class CasadiTrimOptions:
    """Knobs of :class:`CasadiTrim`. Defaults reproduce the NumPy trims'
    physics (linear polars, no viscosity) and the ``casadi_newton`` loop's
    solver settings. ``tether_model`` is ``None`` (tetherless, the
    :func:`solve_vsm_quasi_steady_trim` problem) or ``"williams"`` (the
    :func:`solve_vsm_qs_trim_with_williams_tether` problem)."""

    tether_model: str | None = None
    #: ``bridle_rotates_with_body=False`` reproduces the NumPy trims, which
    #: leave the bridle lines in the world frame while rotating the wing
    #: sections (see the module docstring). ``wake_tolerance`` ends the outer
    #: wake-direction passes once the residual after an AIC rebuild is below
    #: it; the Newton solve itself converges to ``tolerance``.
    polar_interpolation: str = "linear"
    is_with_artificial_viscosity: bool = False
    artificial_viscosity_factor: float = 0.035
    #: The panel kernels implement the freestream-direction force law only;
    #: ``True`` (VSM develop's quarter-chord directions) is refused at build
    #: time rather than silently solved with the wrong law.
    is_aoa_corrected: bool = False
    include_gravity: bool = False
    tolerance: float = 1e-8
    max_iterations: int = 100
    pseudo_time_step: float = 0.03
    reject_ratio: float = 0.5
    max_wake_passes: int = 6
    wake_tolerance: float = 1e-7
    bridle_rotates_with_body: bool = True
    moment_tolerance: float = 1e-2
    rho: float = 1.225
    core_radius_fraction: float = 0.05
    aerodynamic_model_type: str = "VSM"
    cd_cable: float = 1.1
    cf_cable: float = 0.01


def _skew(k: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])


def _rotation_sym(axis: np.ndarray, angle_deg):
    """Rodrigues rotation about a numeric unit axis by a symbolic angle [deg]
    (same composition as ``vsm_quasi_steady._rotation_matrix``)."""
    k = _as_3vector(axis)
    k = k / np.linalg.norm(k)
    skew = ca.DM(_skew(k))
    theta = angle_deg * np.pi / 180.0
    return ca.DM(np.eye(3)) + ca.sin(theta) * skew + (1.0 - ca.cos(theta)) * ca.mtimes(skew, skew)


def _attitude_rotation_sym(roll_deg, pitch_deg, yaw_deg, axes: AxisDefinition):
    return ca.mtimes(
        _rotation_sym(axes.radial, yaw_deg),
        ca.mtimes(_rotation_sym(axes.normal, pitch_deg), _rotation_sym(axes.course, roll_deg)),
    )


def _gravity_expression(system_model: AWETrimSystemModel):
    if hasattr(system_model, "force_gravity"):
        return system_model.force_gravity
    if hasattr(system_model, "expression"):
        return system_model.expression("force_gravity")
    return system_model.kite.force_gravity_for(system_model)


class CasadiTrim:
    """Build once per (body, system model, options); solve many times.

    Parameters mirror :func:`solve_vsm_quasi_steady_trim` /
    :func:`solve_vsm_qs_trim_with_williams_tether`. ``body_aero`` is
    deep-copied and never rotated. ``kcu_drag`` is a
    :class:`~awetrim.aerodynamics.kcu_drag.KcuDragModel` or ``None``.
    """

    #: layout of the trim-state block of the unknown vector
    STATE_NAMES = ("speed_tangential", "roll_deg", "pitch_deg", "yaw_deg", "course_rate")

    def __init__(
        self,
        body_aero: Any,
        system_model: AWETrimSystemModel,
        center_of_gravity: Any,
        reference_point: Any,
        *,
        axes: AxisDefinition = DEFAULT_AXES,
        transformation_c_from_vsm: Any = DEFAULT_TRANSFORMATION_C_FROM_VSM,
        kcu_drag: KcuDragModel | None = None,
        inertia_cg: Any = None,
        applied_moment_nm: Any = None,
        prescribed_roll_deg: float | None = None,
        bounds_lower: Any = DEFAULT_BOUNDS_LOWER,
        bounds_upper: Any = DEFAULT_BOUNDS_UPPER,
        tether_length_bounds: tuple[float, float] | None = None,
        options: CasadiTrimOptions | None = None,
    ) -> None:
        from VSM.core.Solver import Solver

        self.options = options or CasadiTrimOptions()
        opt = self.options
        if opt.tether_model not in (None, "williams"):
            raise ValueError("tether_model must be None or 'williams' (rigid_lumped: use the NumPy solver).")
        self.body = copy.deepcopy(body_aero)
        self.system_model = system_model
        self.axes = axes
        self.center_of_gravity = _as_3vector(center_of_gravity)
        self.reference_point = _as_3vector(reference_point)
        self.transformation = np.asarray(transformation_c_from_vsm, dtype=float)
        self.kcu_drag = kcu_drag
        self.inertia_cg = None if inertia_cg is None else np.asarray(inertia_cg, dtype=float)
        self.applied_moment = np.zeros(3) if applied_moment_nm is None else _as_3vector(applied_moment_nm)
        self.prescribed_roll_deg = prescribed_roll_deg
        self.bounds_lower = _as_5vector(bounds_lower, "bounds_lower")
        self.bounds_upper = _as_5vector(bounds_upper, "bounds_upper")

        # --- geometry (body frame, baseline): numeric parameters ------------
        self.n = None
        self.n_bridle = None
        self._read_geometry(body_aero, center_of_gravity)
        panels = self.body.panels

        # --- VSM's own CasADi panel function (circulation residual + polars) --
        self.vsm = Solver(
            gamma_loop_type="casadi_newton",
            allowed_error=opt.tolerance,
            is_with_artificial_viscosity=opt.is_with_artificial_viscosity,
            artificial_viscosity_factor=opt.artificial_viscosity_factor,
            polar_interpolation=opt.polar_interpolation,
            reference_point=self.reference_point,
            rho=opt.rho,
            core_radius_fraction=opt.core_radius_fraction,
            aerodynamic_model_type=opt.aerodynamic_model_type,
            # The panel kernels are the uncorrected (freestream-direction)
            # force law; VSM develop defaults to the corrected one since
            # 2026-09-17, so pin it.
            is_aoa_corrected=False,
        )
        if opt.is_aoa_corrected:
            raise ValueError(
                "CasadiTrim implements the freestream-direction force law only "
                "(is_aoa_corrected=False); the quarter-chord-corrected VSM "
                "force directions are not available in the CasADi trim. Use "
                "the least_squares trims."
            )
        self.vsm.panels = panels
        self.vsm.n_panels = self.n
        self.panel_fn = self.vsm._casadi_newton_function()
        # The polars the graph was built on (VSM's own cache key: panel count,
        # regularisation settings, table digest); update_geometry refuses a
        # body whose polars differ.
        self.polar_key = self.vsm._casadi_newton_key(*self.vsm._polar_tables())
        if opt.is_with_artificial_viscosity:
            self.stall_angles = self.vsm._panel_stall_angles()
            self.laplacian = self.vsm._build_spanwise_laplacian()
        else:
            self.stall_angles = np.full(self.n, np.inf)
            self.laplacian = np.zeros((self.n, self.n))

        self.mass_total = _system_model_mass_total(system_model)

        # --- Williams tether frames (numeric), mirroring the NumPy solver ------
        self.williams = opt.tether_model == "williams"
        if self.williams:
            from awetrim.system.williams_tether import WilliamsTether
            from awetrim.utils.reference_frames import transformation_C_from_W

            tether = getattr(system_model, "tether", None)
            if not isinstance(tether, WilliamsTether):
                raise TypeError("tether_model='williams' needs a WilliamsTether on system_model.tether.")
            self.tether = tether
            sm = system_model
            self.distance_radial = float(_numeric_value_for_symbol(sm, "distance_radial"))
            angle_az = float(_numeric_value_for_symbol(sm, "angle_azimuth"))
            angle_elev = float(_numeric_value_for_symbol(sm, "angle_elevation"))
            angle_course = float(_numeric_value_for_symbol(sm, "angle_course"))
            T_Csm_from_W = np.asarray(ca.DM(transformation_C_from_W(angle_az, angle_elev, angle_course)).full(), dtype=float)
            direction_wind = float(getattr(getattr(sm, "wind", None), "direction_wind", 0.0))
            c, s_ = np.cos(-direction_wind), np.sin(-direction_wind)
            T_Wind_from_W = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])
            self.T_Wind_from_Csm = T_Wind_from_W @ T_Csm_from_W.T
            self.T_Wind_from_VSM = self.T_Wind_from_Csm @ self.transformation
            self.r_kite_world = self.distance_radial * np.array(
                [np.cos(angle_elev) * np.cos(angle_az), np.cos(angle_elev) * np.sin(angle_az), np.sin(angle_elev)]
            )
            self.r_kite_wind = T_Wind_from_W @ self.r_kite_world
            lo, hi = tether_length_bounds or (0.99 * self.distance_radial, 1.4 * self.distance_radial)
            self.tether_length_bounds = (float(lo), float(hi))
        else:
            self.tether = None
            self.distance_radial = float("nan")
            self.tether_length_bounds = (0.0, 0.0)

        self._build_graph()

    # --------------------------------------------------------------- geometry
    #: geometry parameters of the graph, in packing order: (name, shape)
    #: with ``shape`` per panel (``"n"``), per bridle segment (``"m"``) or
    #: fixed; arrays are packed column-major so ``ca.reshape`` unpacks them.
    GEOMETRY_LAYOUT = (
        ("control_points", ("n", 3)),
        ("aero_centers", ("n", 3)),
        ("x_airf", ("n", 3)),
        ("y_airf", ("n", 3)),
        ("z_airf", ("n", 3)),
        ("chord", ("n", 1)),
        ("width", ("n", 1)),
        ("spanwise_direction", (3, 1)),
        ("projected_area", (1, 1)),
        ("max_chord", (1, 1)),
        ("center_of_gravity", (3, 1)),
        ("bridle_p1", ("m", 3)),
        ("bridle_p2", ("m", 3)),
        ("bridle_diameter", ("m", 1)),
    )

    def _geometry_shapes(self) -> list[tuple[str, tuple[int, int]]]:
        sizes = {"n": self.n, "m": self.n_bridle}
        return [
            (name, tuple(sizes.get(d, d) for d in shape)) for name, shape in self.GEOMETRY_LAYOUT
        ]

    @staticmethod
    def geometry_signature(body_aero: Any) -> tuple:
        """What a built graph is tied to besides its options: panel count,
        bridle segment count and a digest of the panel polars. Two bodies
        with the same signature can share one :class:`CasadiTrim` through
        :meth:`update_geometry`."""
        panels = list(getattr(body_aero, "panels", None) or ())
        digest = hashlib.blake2b(digest_size=16)
        for panel in panels:
            table = getattr(panel, "panel_polar_data", None)
            if table is not None:
                digest.update(np.ascontiguousarray(np.asarray(table, dtype=float)).tobytes())
        bridle = getattr(body_aero, "_bridle_line_system", None) or []
        return (len(panels), len(bridle), digest.hexdigest())

    def _read_geometry(self, body_aero: Any, center_of_gravity: Any) -> None:
        """Read the numeric geometry off ``body_aero`` (deep-copied, never
        rotated) into the parameter vector ``self._p``."""
        body = copy.deepcopy(body_aero)
        panels = body.panels
        n = len(panels)
        bridle = getattr(body, "_bridle_line_system", None) or []
        if self.n is not None and (n != self.n or len(bridle) != self.n_bridle):
            raise ValueError(
                f"geometry has {n} panels / {len(bridle)} bridle segments; the graph was "
                f"built for {self.n} / {self.n_bridle} -- build a new CasadiTrim."
            )
        self.body = body
        self.n = n
        self.n_bridle = len(bridle)
        self.control_points = np.array([p.control_point for p in panels], dtype=float)
        self.aero_centers = np.array([p.aerodynamic_center for p in panels], dtype=float)
        self.x_airf = np.array([p.x_airf for p in panels], dtype=float)
        self.y_airf = np.array([p.y_airf for p in panels], dtype=float)
        self.z_airf = np.array([p.z_airf for p in panels], dtype=float)
        self.chord = np.array([p.chord for p in panels], dtype=float)
        self.width = np.array([p.width for p in panels], dtype=float)
        self.spanwise_direction = np.asarray(body.wings[0].spanwise_direction, dtype=float)
        self.projected_area = float(body.wings[0].compute_projected_area())
        self.max_chord = float(np.max(self.chord))
        self.center_of_gravity = _as_3vector(center_of_gravity)
        # Height-ordered numerically per shape, as VSM does before charging
        # a segment (the sign of the small friction term depends on it).
        self.bridle_segments = [
            (*pk.order_line_endpoints(line[0], line[1]), float(line[2])) for line in bridle
        ]
        # The unrotated sections, for :meth:`body_at_trim` (the same baseline
        # the NumPy trims rotate from).
        self._baseline_sections, self._baseline_spanwise = _baseline_geometry(body)
        values = {
            "control_points": self.control_points,
            "aero_centers": self.aero_centers,
            "x_airf": self.x_airf,
            "y_airf": self.y_airf,
            "z_airf": self.z_airf,
            "chord": self.chord,
            "width": self.width,
            "spanwise_direction": self.spanwise_direction,
            "projected_area": np.array([self.projected_area]),
            "max_chord": np.array([self.max_chord]),
            "center_of_gravity": self.center_of_gravity,
            "bridle_p1": np.array([s[0] for s in self.bridle_segments], dtype=float).reshape(-1, 3),
            "bridle_p2": np.array([s[1] for s in self.bridle_segments], dtype=float).reshape(-1, 3),
            "bridle_diameter": np.array([s[2] for s in self.bridle_segments], dtype=float),
        }
        parts = []
        for name, shape in self._geometry_shapes():
            arr = np.asarray(values[name], dtype=float).reshape(shape)
            parts.append(arr.ravel(order="F"))
        self._p = np.concatenate(parts) if parts else np.zeros(0)

    def _unpack_geometry(self, p):
        """The parameter vector's slices as ``(rows, cols)`` CasADi matrices."""
        out, offset = {}, 0
        for name, (rows, cols) in self._geometry_shapes():
            size = rows * cols
            out[name] = ca.reshape(p[offset : offset + size], rows, cols) if size else None
            offset += size
        return out

    def update_geometry(self, body_aero: Any, center_of_gravity: Any) -> None:
        """Swap in a new shape without rebuilding the graph: same panel and
        bridle counts and the SAME polars (checked against the build's
        ``polar_key``); everything else -- the deformed panels, the bridle
        endpoints, the centre of gravity -- is numeric and re-read."""
        self._read_geometry(body_aero, center_of_gravity)
        self.vsm.panels = self.body.panels
        self.vsm.n_panels = self.n
        key = self.vsm._casadi_newton_key(*self.vsm._polar_tables())
        if key != self.polar_key:
            raise ValueError("the body's polars differ from the graph's -- build a new CasadiTrim.")

    # ------------------------------------------------------------------ graph
    def _substitute_numeric(self, exprs: list, keep: set[str]) -> list:
        """Replace every free symbol not in ``keep`` by its numeric value
        (same lookup as the NumPy trims' ``_as_numeric_3vector``)."""
        syms = [s for s in ca.symvar(ca.vertcat(*[ca.vec(e) for e in exprs])) if s.name() not in keep]
        if not syms:
            return exprs
        values = [ca.DM(np.asarray(_numeric_value_for_symbol(self.system_model, s.name()), dtype=float)) for s in syms]
        return ca.substitute(exprs, syms, values)

    def _system_expressions(self, x):
        """SystemModel kinematics with the trim unknowns (entries of the
        symbol ``x``) substituted in."""
        sm = self.system_model
        saved_speed = sm.speed_tangential
        rate_name = "timeder_angle_course_body" if hasattr(sm, "timeder_angle_course_body") else "timeder_angle_course"
        saved_rate = getattr(sm, rate_name)
        sm.speed_tangential = x[0]
        _set_course_rate_body(sm, x[4])
        try:
            va = sm.velocity_apparent_wind
            omega = sm.velocity_rotation_course_frame if hasattr(sm, "velocity_rotation_course_frame") else None
            accel = sm.acceleration_course_body if hasattr(sm, "acceleration_course_body") else sm.acceleration
            gravity = _gravity_expression(sm)
        finally:
            sm.speed_tangential = saved_speed
            setattr(sm, rate_name, saved_rate)
        exprs = [ca.MX(e) for e in (va, omega if omega is not None else ca.MX.zeros(3), accel, gravity)]
        va, omega_e, accel, gravity = self._substitute_numeric(exprs, {x.name()})
        return va, (omega_e if omega is not None else None), accel, gravity

    def _build_graph(self) -> None:
        opt, n = self.options, self.n
        x = ca.MX.sym("x_trim", 5)
        gamma = ca.MX.sym("gamma", n)
        length = ca.MX.sym("tether_length")
        aic_x = ca.MX.sym("aic_x", n, n)
        aic_y = ca.MX.sym("aic_y", n, n)
        aic_z = ca.MX.sym("aic_z", n, n)
        p = ca.MX.sym("p_geometry", int(self._p.size))
        g = self._unpack_geometry(p)
        v_tau, roll, pitch, yaw, chi_dot = (x[i] for i in range(5))
        keep = {x.name(), gamma.name(), length.name(), aic_x.name(), aic_y.name(), aic_z.name(), p.name()}

        T = ca.DM(self.transformation)
        va_c, omega_c, accel_c, gravity_c = self._system_expressions(x)
        va_w = ca.mtimes(T, va_c)
        omega_w = ca.mtimes(T, omega_c) if omega_c is not None else -chi_dot * ca.DM(self.axes.radial)
        accel_w = ca.mtimes(T, accel_c)
        gravity_w = ca.mtimes(T, gravity_c) if opt.include_gravity else ca.MX.zeros(3)

        R = _attitude_rotation_sym(roll, pitch, yaw, self.axes)
        va_b = ca.mtimes(R.T, va_w)
        omega_b = ca.mtimes(R.T, omega_w)
        r0 = self.reference_point
        r0_rows = pk.tile_vector(r0, n, ca)
        arm_cp = g["control_points"] - r0_rows
        va_panels = pk.tile_vector(va_b, n, ca) - pk.cross_rows(pk.tile_vector(omega_b, n, ca), arm_cp, ca)
        v_rel = va_panels + ca.horzcat(ca.mtimes(aic_x, gamma), ca.mtimes(aic_y, gamma), ca.mtimes(aic_z, gamma))

        out = self.panel_fn(
            v_rel, va_panels, g["x_airf"], g["y_airf"], g["z_airf"],
            g["chord"], g["width"], ca.DM(self.stall_angles), ca.mtimes(ca.DM(self.laplacian), gamma),
        )
        h, _dh, mu, alpha, umag, gate, h_avg, cl, cd, cm, cl_avg, cd_avg, cm_avg = out

        speed = ca.norm_2(va_w)
        max_chord = g["max_chord"]
        projected_area = g["projected_area"]
        gamma_scale = speed * max_chord
        q_inf = 0.5 * opt.rho * speed**2
        denom_moment = q_inf * projected_area * max_chord
        denom_force = q_inf * projected_area
        spanwise = g["spanwise_direction"]

        def aero_loads(cl_, cd_, cm_):
            force_b, moment_local = pk.panel_forces(
                alpha, umag, cl_, cd_, cm_, g["chord"], g["width"],
                g["x_airf"], g["y_airf"], g["z_airf"],
                spanwise, opt.rho, ca,
            )
            f_tot, m_tot = pk.total_force_and_moment(force_b, moment_local, g["aero_centers"] - r0_rows, ca)
            bridle_forces = []
            if opt.bridle_rotates_with_body:
                va_seg_free, omega_seg = va_b, omega_b
            else:
                # Legacy NumPy-trim convention: the bridle is charged in the
                # world frame, unrotated; its loads are rotated back into the
                # body frame so the totals below stay in one frame.
                va_seg_free, omega_seg = va_w, omega_w
            for k in range(self.n_bridle):
                p1 = g["bridle_p1"][k, :].T
                p2 = g["bridle_p2"][k, :].T
                diameter = g["bridle_diameter"][k]
                mid = 0.5 * (p1 + p2)
                va_seg = apparent_wind_at(va_seg_free, omega_seg, mid, r0, xp=ca)
                f_seg = pk.line_force(va_seg, p1, p2, diameter, opt.rho, opt.cd_cable, opt.cf_cable, ca)
                m_seg = pk.cross3(mid - ca.DM(r0), f_seg, ca)
                if not opt.bridle_rotates_with_body:
                    f_seg, m_seg = ca.mtimes(R.T, f_seg), ca.mtimes(R.T, m_seg)
                bridle_forces.append(f_seg)
                f_tot = f_tot + f_seg
                m_tot = m_tot + m_seg
            return force_b, f_tot, m_tot, bridle_forces

        def residual_rows(cl_, cd_, cm_, h_):
            force_b, f_aero_b, m_aero_b, bridle_forces = aero_loads(cl_, cd_, cm_)
            f_aero_w = ca.mtimes(R, f_aero_b)
            m_aero_w = ca.mtimes(R, m_aero_b)
            cg_arm = ca.mtimes(R, g["center_of_gravity"] - ca.DM(r0))
            inertial = -self.mass_total * (accel_w + pk.cross3(omega_w, pk.cross3(omega_w, cg_arm, ca), ca))
            if self.kcu_drag is not None:
                va_kcu = apparent_wind_at(va_w, omega_w, DEFAULT_STATION_KCU, r0, xp=ca)
                f_kcu = self.kcu_drag.force(va_kcu, self.axes.radial, opt.rho, xp=ca)
            else:
                f_kcu = ca.MX.zeros(3)
            moment = m_aero_w + pk.cross3(cg_arm, inertial, ca) + pk.cross3(ca.DM(DEFAULT_STATION_KCU - r0), f_kcu, ca)
            if opt.include_gravity:
                moment = moment + pk.cross3(cg_arm, gravity_w, ca)
            moment = moment + ca.DM(self.applied_moment)
            if self.inertia_cg is not None:
                inertia_w = ca.mtimes(R, ca.mtimes(ca.DM(self.inertia_cg), R.T))
                moment = moment - pk.cross3(omega_w, ca.mtimes(inertia_w, omega_w), ca)
            net_force = f_aero_w + inertial + gravity_w + f_kcu
            extras = dict(
                force_panels_b=force_b, f_aero_w=f_aero_w, m_aero_w=m_aero_w, f_aero_b=f_aero_b,
                inertial=inertial, gravity=gravity_w, f_kcu=f_kcu, net_force=net_force,
                moment_total=moment, cg_arm=cg_arm, bridle_forces=bridle_forces,
            )
            if self.williams:
                # Collapsed Williams closure: the kite-end tether vector IS the
                # trim resultant (3-D force balance baked in), so only the
                # moments and the ground closure remain; the residual carries
                # the ground miss normalised by the radial distance.
                f_kite_wind = ca.mtimes(ca.DM(self.T_Wind_from_VSM), net_force)
                omega_wind = ca.mtimes(ca.DM(self.T_Wind_from_Csm), omega_c) if omega_c is not None else ca.MX.zeros(3)
                shape = self.tether.tether_shape_symbolic(
                    env=(self.system_model if opt.include_gravity else _ZeroGravityEnv(self.system_model)),
                    r_kite=ca.DM(self.r_kite_wind),
                    kite_tension_vector=f_kite_wind,
                    tether_length=length,
                    omega=omega_wind,
                )
                rows_trim = ca.vertcat(moment / denom_moment, shape["ground_position"] / self.distance_radial)
                extras.update(
                    f_kite_wind=f_kite_wind, ground_position=shape["ground_position"],
                    tether_positions=shape["positions"], tether_tensions=shape["tensions"],
                )
            else:
                rows_trim = ca.vertcat(
                    moment / denom_moment,
                    ca.dot(net_force, ca.DM(self.axes.course)) / denom_force,
                    ca.dot(net_force, ca.DM(self.axes.normal)) / denom_force,
                )
            rows_gamma = (gamma - h_) / gamma_scale
            return ca.vertcat(rows_trim, rows_gamma), extras

        residual, extras = residual_rows(cl, cd, cm, h)
        residual_avg, _ = residual_rows(cl_avg, cd_avg, cm_avg, h_avg)
        n_trim_rows = 6 if self.williams else 5

        # Free unknowns: x (roll dropped when pinned), gamma, tether length.
        free_x = [0, 2, 3, 4] if self.prescribed_roll_deg is not None else [0, 1, 2, 3, 4]
        trim_rows = list(range(1, n_trim_rows)) if self.prescribed_roll_deg is not None else list(range(n_trim_rows))
        rows = trim_rows + list(range(n_trim_rows, n_trim_rows + n))
        self.free_x = np.array(free_x, dtype=int)
        self.n_trim_rows = n_trim_rows
        z = ca.vertcat(x, gamma, length)
        free_cols = free_x + list(range(5, 5 + n)) + ([5 + n] if self.williams else [])
        all_row_names = ["cmx", "cmy", "cmz"] + (["ground_x", "ground_y", "ground_z"] if self.williams else ["cfx", "cfy"])
        self.row_names = [all_row_names[i] for i in trim_rows] + [f"gamma_{i}" for i in range(n)]
        self.column_names = [self.STATE_NAMES[i] for i in free_x] + [f"gamma_{i}" for i in range(n)] + (
            ["tether_length"] if self.williams else []
        )

        params = [aic_x, aic_y, aic_z, p]
        res_free, res_avg_free = self._substitute_numeric([residual[rows], residual_avg[rows]], keep)
        args = [x, gamma, length, *params]
        self._kin_fn = ca.Function("trim_kinematics", [x], self._substitute_numeric([va_w, omega_w, R], keep))
        self._res_fn = ca.Function("trim_residual", args, [res_free, gate])
        self._jac_fn = ca.Function("trim_jacobian", args, [ca.jacobian(res_avg_free, z)[:, free_cols]])
        self._jac_exact_fn = ca.Function("trim_jacobian_exact", args, [ca.jacobian(res_free, z)[:, free_cols]])

        # Reported coefficients: VSM's own bookkeeping (each panel in its
        # local inflow frame, projected on the reference directions).
        lift, drag, side = pk.lift_drag_side_vsm(
            extras["force_panels_b"], va_panels, va_b, spanwise, ca, extras["bridle_forces"]
        )
        outputs = dict(
            residual=residual,
            va_w=va_w, va_b=va_b, omega_w=omega_w, omega_b=omega_b, R=R,
            f_aero_w=extras["f_aero_w"], m_aero_w=extras["m_aero_w"], inertial=extras["inertial"],
            gravity=extras["gravity"], f_kcu=extras["f_kcu"], net_force=extras["net_force"],
            moment_total=extras["moment_total"], cg_arm=extras["cg_arm"],
            cl=lift / denom_force, cd=drag / denom_force, cs=side / denom_force,
            cm_aero=extras["m_aero_w"] / denom_moment,
            alpha=alpha, cl_panels=cl, cd_panels=cd, mu=mu, gate=gate,
            force_panels_b=extras["force_panels_b"], v_rel=v_rel,
        )
        if self.williams:
            outputs.update(
                f_kite_wind=extras["f_kite_wind"], ground_position=extras["ground_position"],
                tether_positions=extras["tether_positions"], tether_tensions=extras["tether_tensions"],
            )
        self._output_names = list(outputs)
        self._out_fn = ca.Function("trim_outputs", args, self._substitute_numeric(list(outputs.values()), keep))

    # ---------------------------------------------------------------- numerics
    def _kinematics(self, x: np.ndarray):
        va_w, omega_w, R = self._kin_fn(x)
        return np.asarray(va_w).ravel(), np.asarray(omega_w).ravel(), np.asarray(R)

    def _set_body_inflow(self, x: np.ndarray) -> None:
        """Hand the (unrotated) VSM body the body-frame inflow of trim state
        ``x`` through its own ``va`` setter, so its wake and panel inflow match
        the graph's ``va_panels``."""
        va_w, omega_w, R = self._kinematics(x)
        va_b = R.T @ va_w
        omega_b = R.T @ omega_w
        rate = float(np.linalg.norm(omega_b))
        axis = omega_b / rate if rate > 1e-12 else np.asarray(self.axes.radial, dtype=float)
        type(self.body).va.fset(
            self.body, va_b, body_rates=rate, body_axis=axis,
            reference_point=self.reference_point, rates_in_body_frame=True,
        )

    def compute_aic(self, x: np.ndarray):
        """Numeric induction matrices for the wake direction of trim state ``x``."""
        self._set_body_inflow(x)
        va = np.array([p.va for p in self.body.panels], dtype=float)
        norms = np.linalg.norm(va, axis=1)
        return tuple(self.body.compute_AIC_matrices(
            self.options.aerodynamic_model_type, self.options.core_radius_fraction, norms, va / norms[:, None]
        ))

    def initial_circulation(self, x: np.ndarray) -> np.ndarray:
        """A converged VSM circulation at trim state ``x`` (the ``casadi_newton``
        loop with its fallback), the seed when the caller has none."""
        self._set_body_inflow(x)
        res = self.vsm.solve(self.body)
        return np.asarray(res["gamma_distribution"], dtype=float)

    def _split(self, u: np.ndarray, x_fixed: np.ndarray):
        k = len(self.free_x)
        x = x_fixed.copy()
        x[self.free_x] = u[:k]
        gamma = u[k : k + self.n]
        length = float(u[k + self.n]) if self.williams else 0.0
        return x, gamma, length

    def _project(self, u: np.ndarray) -> np.ndarray:
        k = len(self.free_x)
        u = u.copy()
        u[:k] = np.clip(u[:k], self.bounds_lower[self.free_x], self.bounds_upper[self.free_x])
        if self.williams:
            u[k + self.n] = np.clip(u[k + self.n], *self.tether_length_bounds)
        return u

    def residual(self, x, gamma, aic, length: float = 0.0):
        r, gate = self._res_fn(x, gamma, length, *aic, self._p)
        return np.asarray(r).ravel(), bool(float(gate) > 0.5)

    def jacobian(self, x, gamma, aic, length: float = 0.0, exact: bool = False):
        fn = self._jac_exact_fn if exact else self._jac_fn
        return np.asarray(fn(x, gamma, length, *aic, self._p))

    def outputs(self, x, gamma, aic, length: float = 0.0) -> dict[str, Any]:
        vals = self._out_fn(x, gamma, length, *aic, self._p)
        return {k: np.asarray(v) for k, v in zip(self._output_names, vals)}

    # ------------------------------------------------------------------ solve
    def _newton(self, u, x_fixed, aic, budget):
        """Newton + Armijo, pseudo-transient continuation when the line search
        stalls; same rules as ``Solver.gamma_loop_casadi_newton``."""
        opt = self.options
        tol = opt.tolerance
        dt_restart = opt.pseudo_time_step
        dt_min = 1e-4 * dt_restart

        def evaluate(u_):
            x_, g_, l_ = self._split(u_, x_fixed)
            r, gate = self.residual(x_, g_, aic, l_)
            return {"u": u_, "r": r, "gate": gate, "x": x_, "gamma": g_, "length": l_}

        def norm(s):
            return float(np.linalg.norm(s["r"]))

        def err(s):
            return float(np.max(np.abs(s["r"])))

        cur = evaluate(self._project(u))
        evaluations, iteration, dt, failure = 1, 0, np.inf, None
        converged = err(cur) < tol
        while not converged and iteration < budget:
            jac = self.jacobian(cur["x"], cur["gamma"], aic, cur["length"])
            system = jac if np.isinf(dt) else jac + np.eye(jac.shape[0]) / dt
            try:
                direction = np.linalg.solve(system, -cur["r"])
            except np.linalg.LinAlgError:
                direction = None
            if direction is None or not np.all(np.isfinite(direction)):
                if np.isinf(dt):
                    dt = dt_restart
                    continue
                dt *= 0.25
                if dt < dt_min:
                    failure = "singular system"
                    break
                continue
            if np.isinf(dt):
                merit = 0.5 * norm(cur) ** 2
                slope = float(direction @ (jac.T @ cur["r"]))
                step, accepted = 1.0, None
                while step >= 2.0**-8:
                    trial = evaluate(self._project(cur["u"] + step * direction))
                    evaluations += 1
                    if np.all(np.isfinite(trial["r"])) and 0.5 * norm(trial) ** 2 <= merit + 1e-4 * step * slope:
                        accepted = trial
                        break
                    step *= 0.5
                if accepted is None or slope >= 0.0:
                    dt = dt_restart
                    _LOG.debug("casadi trim: line search stalled at %s (|r| %.2e), pseudo-transient", iteration, err(cur))
                    continue
                cur = accepted
            else:
                trial = evaluate(self._project(cur["u"] + direction))
                evaluations += 1
                finite = np.all(np.isfinite(trial["r"]))
                ratio = norm(cur) / norm(trial) if finite and norm(trial) > 0 else 0.0
                crossed = finite and trial["gate"] != cur["gate"]
                if not finite or (ratio < opt.reject_ratio and not (crossed and dt <= dt_restart)):
                    dt *= 0.25
                    if dt < dt_min:
                        failure = "pseudo time step collapsed"
                        break
                    continue
                cur = trial
                dt = dt_restart if crossed else max(dt_restart, dt * min(ratio, 10.0))
                if dt > 1e6 * dt_restart:
                    dt = np.inf
            iteration += 1
            converged = err(cur) < tol
            _LOG.debug("casadi trim iteration %s: max|r| %.3e dt %.3g", iteration, err(cur), dt)
        return cur, {"converged": converged, "iterations": iteration, "evaluations": evaluations, "failure": failure}

    def solve(
        self,
        x_guess: Any,
        gamma_seed: np.ndarray | None = None,
        tether_length_guess: float | None = None,
    ) -> dict[str, Any]:
        """Solve the trim from ``x_guess`` (5 states; the pinned roll, if any,
        overrides entry 1). Returns a result dict in the vocabulary of the
        NumPy trims (``opt_x``, ``cl``, ``cd``, ``gamma_distribution``,
        ``success``, ``success_physical``, the ``williams_*`` fields, ...)
        plus ``casadi_trim`` diagnostics and the exact Jacobian at the
        solution."""
        opt = self.options
        t_start = perf_counter()
        x = np.clip(_as_5vector(x_guess, "x_guess"), self.bounds_lower, self.bounds_upper)
        if self.prescribed_roll_deg is not None:
            x[1] = float(self.prescribed_roll_deg)
        t0 = perf_counter()
        gamma = self.initial_circulation(x) if gamma_seed is None else np.asarray(gamma_seed, dtype=float)
        t_seed = perf_counter() - t0
        length = (
            (1.02 * self.distance_radial if tether_length_guess is None else float(tether_length_guess))
            if self.williams else 0.0
        )

        u = np.concatenate([x[self.free_x], gamma, [length] if self.williams else []])
        passes, info, t_aic, t_newton = [], {}, 0.0, 0.0
        aic = None
        for wake_pass in range(opt.max_wake_passes):
            t0 = perf_counter()
            aic = self.compute_aic(x)
            t_aic += perf_counter() - t0
            if wake_pass > 0:
                r, _ = self.residual(x, gamma, aic, length)
                passes[-1]["residual_after_wake_rebuild"] = float(np.max(np.abs(r)))
                if np.max(np.abs(r)) < opt.wake_tolerance:
                    break
            t0 = perf_counter()
            state, info = self._newton(u, x, aic, opt.max_iterations)
            t_newton += perf_counter() - t0
            u, x, gamma, length = state["u"], state["x"], state["gamma"], state["length"]
            passes.append({"pass": wake_pass, **info, "residual": float(np.max(np.abs(state["r"])))})
            if not info["converged"]:
                break
        else:
            _LOG.info("casadi trim: wake passes exhausted (%s)", opt.max_wake_passes)

        out = self.outputs(x, gamma, aic, length)
        r_full = out["residual"].ravel()
        cm = r_full[:3]
        va_b = out["va_b"].ravel()
        va_w = out["va_w"].ravel()
        at_bound = bool(
            np.any(np.isclose(x[self.free_x], self.bounds_lower[self.free_x]))
            or np.any(np.isclose(x[self.free_x], self.bounds_upper[self.free_x]))
            or (self.williams and np.any(np.isclose(length, self.tether_length_bounds)))
        )
        rows_checked = cm[1:] if self.prescribed_roll_deg is not None else cm
        success_physical = bool(np.max(np.abs(rows_checked)) < opt.moment_tolerance)
        result = {
            "opt_x": x,
            "success": bool(info.get("converged", False) and not at_bound),
            # Newton convergence alone; ``success`` also demands no bound.
            "converged": bool(info.get("converged", False)),
            "success_physical": success_physical,
            "at_bound": at_bound,
            "cm": cm,
            "cm_aero": out["cm_aero"].ravel(),
            "cl": float(np.asarray(out["cl"]).item()),
            "cd": float(np.asarray(out["cd"]).item()),
            "cs": float(np.asarray(out["cs"]).item()),
            "aoa_deg": float(np.rad2deg(np.arctan2(va_b[2], va_b[0]))),
            "side_slip_deg": float(np.rad2deg(np.arctan2(va_b[1], np.hypot(va_b[0], va_b[2])))),
            "aoa_course_deg": float(np.rad2deg(np.arctan2(va_w[2], va_w[0]))),
            "side_slip_course_deg": float(np.rad2deg(np.arctan2(va_w[1], np.hypot(va_w[0], va_w[2])))),
            "Umag": float(np.linalg.norm(va_w)),
            "va_vel_world": va_w,
            "omega_c_vsm": out["omega_w"].ravel(),
            "total_aero_force_vec": out["f_aero_w"].ravel(),
            "aero_moment_vec": out["m_aero_w"].ravel(),
            "inertial_force": out["inertial"].ravel(),
            "gravity_force": out["gravity"].ravel(),
            "kcu_drag_force_vsm": out["f_kcu"].ravel(),
            "net_force": out["net_force"].ravel(),
            "force_kite_resultant_vsm": out["net_force"].ravel(),
            "moment_total_nm": out["moment_total"].ravel(),
            "attitude_rotation": out["R"],
            "gamma_distribution": gamma,
            "alpha_array": out["alpha"].ravel(),
            "cl_distribution": out["cl_panels"].ravel(),
            "cd_distribution": out["cd_panels"].ravel(),
            "F_distribution_body": out["force_panels_b"],
            "n_panels": self.n,
            "tether_model": opt.tether_model,
            "reaction_roll_moment_nm": (
                float(cm[0] * 0.5 * opt.rho * np.dot(va_w, va_w) * self.projected_area * self.max_chord)
                if self.prescribed_roll_deg is not None else 0.0
            ),
            "jacobian": self.jacobian(x, gamma, aic, length, exact=True),
            "jacobian_layout": {"rows": self.row_names, "columns": self.column_names},
            "casadi_trim": {
                "passes": passes,
                "iterations": int(sum(p.get("iterations", 0) for p in passes)),
                "residual_max": float(np.max(np.abs(r_full))),
                "time_total_s": perf_counter() - t_start,
                "time_seed_s": t_seed,
                "time_aic_s": t_aic,
                "time_newton_s": t_newton,
                "polar_interpolation": opt.polar_interpolation,
            },
        }
        if self.williams:
            f_kite_wind = out["f_kite_wind"].ravel()
            direction = f_kite_wind / (np.linalg.norm(f_kite_wind) + 1e-12)
            result.update(
                cfx=0.0, cfy=0.0, cfz=0.0,
                force_kite_resultant=f_kite_wind,
                tether_force=float(np.linalg.norm(f_kite_wind)),
                r_kite=self.r_kite_wind, r_kite_world=self.r_kite_world,
                williams_x=np.array([length]),
                williams_tether_length=float(length),
                williams_elevation_last_deg=float(np.rad2deg(np.arcsin(np.clip(direction[2], -1.0, 1.0)))),
                williams_azimuth_last_deg=float(np.rad2deg(np.arctan2(direction[1], direction[0]))),
                williams_ground_residual=out["ground_position"].ravel(),
                williams_positions=out["tether_positions"],
                williams_tensions=out["tether_tensions"],
            )
        else:
            result.update(cfx=float(r_full[3]), cfy=float(r_full[4]))
            result["success_physical"] = bool(success_physical and np.max(np.abs(r_full[3:5])) < opt.moment_tolerance)
        return result


    # ------------------------------------------------------ body at the trim
    def body_at_trim(self, x: Any) -> Any:
        """A copy of the body ROTATED to the attitude of trim state ``x`` about
        the reference point, with the WORLD-frame inflow of that state set
        through the VSM ``va`` setter: the body the NumPy trims hand back, so
        anything that reads the inflow off it
        (:func:`~awetrim.aerodynamics.apparent_wind.inflow_state_of`) or its
        panels (the aerostructural load transfer) sees the same frame as the
        rotated structure."""
        x = _as_5vector(x, "x")
        body = copy.deepcopy(self.body)
        _set_body_attitude_from_baseline(
            body,
            baseline_sections=self._baseline_sections,
            baseline_spanwise=self._baseline_spanwise,
            roll_deg=float(x[1]),
            pitch_deg=float(x[2]),
            yaw_deg=float(x[3]),
            axes=self.axes,
            reference_point=self.reference_point,
        )
        # The bridle rotates WITH the wing in this trim's model (module
        # docstring); _set_body_attitude_from_baseline leaves the body's line
        # system in the world frame (the NumPy-trim convention), so rotate it
        # here unless the legacy convention was asked for. Otherwise the VSM
        # results evaluated on this body would charge an unrotated bridle
        # and disagree with the trim's own resultant by the attitude effect
        # (4 N, 22 N m at 1 deg on the LEI-V3).
        lines = getattr(body, "_bridle_line_system", None)
        if self.options.bridle_rotates_with_body and lines:
            rotation = np.asarray(body.geometry_rotation, dtype=float)
            origin = self.reference_point

            def rotate(point):
                return origin + rotation @ (np.asarray(point, dtype=float).ravel() - origin)

            body._bridle_line_system = [
                [rotate(line[0]), rotate(line[1]), *line[2:]] for line in lines
            ]
        va_w, omega_w, _R = self._kinematics(x)
        rate = float(np.linalg.norm(omega_w))
        axis = omega_w / rate if rate > 1e-12 else -np.asarray(self.axes.radial, dtype=float)
        # rates_in_body_frame=True: the axis is taken as given (world
        # components), as in the NumPy trims.
        type(body).va.fset(
            body, va_w, body_rates=rate, body_axis=axis,
            reference_point=self.reference_point, rates_in_body_frame=True,
        )
        return body

    def solve_with_body(
        self,
        x_guess: Any,
        gamma_seed: np.ndarray | None = None,
        tether_length_guess: float | None = None,
    ) -> tuple[dict[str, Any], Any]:
        """:meth:`solve`, then the per-panel outputs the aerostructural coupling
        consumes, in the contract of the NumPy trims: ``(result, body)`` with
        the body from :meth:`body_at_trim` and ``F_distribution`` /
        ``M_distribution`` / ``panel_cp_locations`` / ``alpha_at_ac`` (VSM's
        aerodynamic-centre angle, the stall-detection input) evaluated on it
        by the VSM's own ``compute_results`` -- one ``Solver.solve`` from the
        converged circulation, so no second circulation problem is solved --
        plus the stalled-fraction and KCU-drag reporting fields. The trim's
        own ``cl``/``cd``/``cm`` are kept (same bookkeeping, see the module
        docstring)."""
        result = self.solve(x_guess, gamma_seed=gamma_seed, tether_length_guess=tether_length_guess)
        body = self.body_at_trim(result["opt_x"])
        t0 = perf_counter()
        res = self.vsm.solve(body, gamma_distribution=np.asarray(result["gamma_distribution"], dtype=float))
        result["casadi_trim"]["time_vsm_results_s"] = perf_counter() - t0
        for key in (
            "F_distribution", "M_distribution", "panel_cp_locations", "alpha_at_ac",
            "alpha_uncorrected", "alpha_geometric", "bridle_line_forces",
            "bridle_line_midpoints", "gamma_converged",
        ):
            if key in res:
                result[key] = res[key]
        x_cp = np.asarray(res.get("center_of_pressure", np.nan), dtype=float)
        result["x_cp_point"] = (
            x_cp.reshape(3) if x_cp.size == 3 else np.array([float(x_cp.ravel()[0]), 0.0, 0.0])
        )
        va_w = np.asarray(result["va_vel_world"], dtype=float)
        va_unit = va_w / np.linalg.norm(va_w)
        lift_dir = self.axes.radial - np.dot(self.axes.radial, va_unit) * va_unit
        f_aero = np.asarray(result["total_aero_force_vec"], dtype=float)
        result["aero_roll_deg"] = float(np.rad2deg(np.arctan2(
            np.dot(f_aero, np.cross(lift_dir, va_unit)), np.dot(f_aero, lift_dir)
        )))
        result["tether_force"] = float(np.asarray(result["net_force"], dtype=float)[2])
        result.update(stall_fraction_fields(body, res.get("alpha_at_ac")))
        result.update(_kcu_drag_fields(
            self.kcu_drag, np.asarray(result["kcu_drag_force_vsm"], dtype=float),
            va_w, np.asarray(self.axes.radial, dtype=float), self.projected_area,
        ))
        return result, body


def solve_vsm_trim_casadi(
    body_aero: Any,
    center_of_gravity: Any,
    reference_point: Any,
    system_model: AWETrimSystemModel,
    x_guess: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    """One-shot convenience with the argument order of the NumPy trims;
    ``kwargs`` go to :class:`CasadiTrim` (``gamma_seed`` and
    ``tether_length_guess`` to :meth:`CasadiTrim.solve`)."""
    gamma_seed = kwargs.pop("gamma_seed", None)
    tether_length_guess = kwargs.pop("tether_length_guess", None)
    trim = CasadiTrim(body_aero, system_model, center_of_gravity, reference_point, **kwargs)
    return trim.solve(x_guess, gamma_seed=gamma_seed, tether_length_guess=tether_length_guess)
