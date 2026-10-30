"""Section-level aerodynamic force laws as ``xp`` kernels.

The formulas here are the ones VSM's ``BodyAerodynamics.compute_results`` and
``compute_line_aerodynamic_force`` apply in NumPy, written once over an
``xp`` math namespace so the same function serves the NumPy path and a CasADi
graph (the CasADi trim in :mod:`awetrim.aerodynamics.trim_casadi`). Pass
``xp=numpy`` (default) or ``xp=casadi``. No formula may be restated at a call
site; ``tests/aerodynamics/test_panel_kernels.py`` pins each kernel to the VSM
numbers it reproduces. Moving these kernels upstream into VSM, so its own
``compute_results`` calls them, is the intended end state.

Conventions (VSM's): every per-panel quantity is a row of an ``(n, 3)`` array
or an entry of an ``(n,)`` vector (``(n, 1)`` in CasADi). ``alpha`` is the
effective angle of attack from the relative velocity's components along the
panel's chord-normal ``x_airf`` and chord ``y_airf`` axes; ``umag`` is
``|v_rel x z_airf|``, the speed the 2-D section sees.
"""
from __future__ import annotations

from typing import Any

import numpy as np


# --------------------------------------------------------------------------- xp
def _is_casadi(xp: Any) -> bool:
    return hasattr(xp, "horzcat")


def hstack_columns(xp: Any, *cols):
    """``(n, k)`` from ``k`` per-row columns."""
    if _is_casadi(xp):
        return xp.horzcat(*cols)
    return np.column_stack(cols)


def cross_rows(a, b, xp=np):
    """Row-wise cross product of two ``(n, 3)`` arrays."""
    return hstack_columns(
        xp,
        a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1],
        a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2],
        a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0],
    )


def dot_rows(a, b, xp=np):
    """Row-wise dot product of two ``(n, 3)`` arrays -> per-row scalar."""
    if _is_casadi(xp):
        return xp.sum2(a * b)
    return np.sum(a * b, axis=1)


def norm_rows(a, xp=np):
    return xp.sqrt(dot_rows(a, a, xp))


def scale_rows(s, a, xp=np):
    """Per-row scalar ``s`` times ``(n, 3)`` rows of ``a``."""
    if _is_casadi(xp):
        return xp.repmat(s, 1, a.shape[1]) * a
    return np.asarray(s)[:, None] * a


def unit_rows(a, xp=np):
    return scale_rows(1.0 / norm_rows(a, xp), a, xp)


def as_xp_vector(v, xp=np):
    """A numeric 3-vector as the type the namespace's arithmetic expects
    (``DM`` for CasADi, so ``ndarray - MX`` never falls into NumPy's object
    arithmetic). Symbolic inputs pass through."""
    if _is_casadi(xp) and isinstance(v, np.ndarray):
        return xp.DM(v.reshape(-1))
    return v


def tile_vector(v, n: int, xp=np):
    """``(n, 3)`` with the 3-vector ``v`` (numeric or symbolic) in every row."""
    if _is_casadi(xp):
        if isinstance(v, (np.ndarray, list, tuple)):
            v = as_xp_vector(np.asarray(v, dtype=float), xp)
        return xp.repmat(xp.reshape(v, 1, 3), n, 1)
    return np.tile(np.asarray(v, dtype=float).reshape(1, 3), (n, 1))


def atan2(y, x, xp=np):
    return xp.atan2(y, x) if _is_casadi(xp) else np.arctan2(y, x)


def dot3(a, b, xp=np):
    return xp.dot(a, b) if _is_casadi(xp) else float(np.dot(a, b))


def norm3(a, xp=np):
    return xp.sqrt(xp.dot(a, a)) if _is_casadi(xp) else float(np.linalg.norm(a))


def cross3(a, b, xp=np):
    if _is_casadi(xp):
        return xp.vertcat(
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        )
    return np.cross(np.asarray(a, dtype=float), np.asarray(b, dtype=float))


# ------------------------------------------------------------------ section flow
def relative_flow(v_rel, x_airf, y_airf, z_airf, xp=np):
    """Effective angle of attack and section speed from the relative velocity.

    VSM ``Solver.compute_aerodynamic_quantities``:
    ``alpha = atan2(v_rel . x_airf, v_rel . y_airf)``, ``umag = |v_rel x z_airf|``.
    """
    alpha = atan2(dot_rows(x_airf, v_rel, xp), dot_rows(y_airf, v_rel, xp), xp)
    umag = norm_rows(cross_rows(v_rel, z_airf, xp), xp)
    return alpha, umag


def circulation_target(umag, cl, chord):
    """VSM's fixed-point target ``0.5 umag cl chord`` (Gaunaa, Li & Pirrung
    2026, Eq. 4: the circulation from the section's own inner speed)."""
    return 0.5 * umag * cl * chord


def panel_forces(
    alpha,
    umag,
    cl,
    cd,
    cm,
    chord,
    width,
    x_airf,
    y_airf,
    z_airf,
    spanwise_direction,
    rho: float,
    xp=np,
):
    """Per-panel 3-D force and local pitching moment, VSM's construction.

    ``compute_results`` with ``is_aoa_corrected=False`` and no viscous drag
    correction: the 2-D lift ``cl q c`` acts normal to the section's own
    relative-flow direction ``cos(alpha) y_airf + sin(alpha) x_airf`` (in the
    plane normal to ``z_airf``), the 2-D drag ``cd q c`` ALONG that direction
    (the local inner flow, not the global span axis), both scaled by the
    panel width; the 2-D moment ``cm q c^2`` is a vector along ``z_airf``
    times the width. ``q = 0.5 rho umag^2`` with the SECTION speed, not the
    freestream. ``spanwise_direction`` is no longer used by the force law and
    is kept for the call signature.

    The attached-trailed vortex force is NOT included here; see
    :func:`attached_trailed_forces`.

    Returns ``(force (n, 3), moment_local (n, 3))``.
    """
    q_chord = 0.5 * rho * umag**2 * chord
    lift = cl * q_chord
    drag = cd * q_chord
    moment = cm * q_chord * chord

    dir_flow = scale_rows(xp.cos(alpha), y_airf, xp) + scale_rows(xp.sin(alpha), x_airf, xp)
    dir_flow = unit_rows(dir_flow, xp)
    dir_lift = unit_rows(cross_rows(dir_flow, z_airf, xp), xp)

    force_2d = scale_rows(lift, dir_lift, xp) + scale_rows(drag, dir_flow, xp)
    force = scale_rows(width, force_2d, xp)
    moment_local = scale_rows(moment * width, z_airf, xp)
    return force, moment_local


# ---------------------------------------------------- attached trailed vortices
def attached_trailed_geometry(bound_point_1, bound_point_2, te_point_1, te_point_2,
                              aerodynamic_center_location: float, control_point_location: float):
    """Legs and force points of the attached trailed (AT) vortex segments,
    VSM ``BodyAerodynamics.compute_attached_trailed_vortex_forces``.

    Boundary ``j = 0..n`` between panels carries one chordwise leg from the
    bound vortex to the trailing edge; its force acts at the section's
    control-point station on that leg. Numeric (geometry), ``(n+1, 3)`` each.
    """
    bound = np.vstack([np.asarray(bound_point_1, dtype=float), np.asarray(bound_point_2, dtype=float)[-1:]])
    te = np.vstack([np.asarray(te_point_1, dtype=float), np.asarray(te_point_2, dtype=float)[-1:]])
    legs = te - bound
    t = (control_point_location - aerodynamic_center_location) / (1.0 - aerodynamic_center_location)
    return bound + t * legs, legs


def boundary_shares(n: int) -> np.ndarray:
    """``(n, n+1)`` map from boundary loads to panel loads, VSM's split: each
    inner boundary half to either neighbour, the tip boundaries whole."""
    shares = np.zeros((n, n + 1))
    for j in range(n + 1):
        if j == 0:
            shares[0, 0] = 1.0
        elif j == n:
            shares[n - 1, n] = 1.0
        else:
            shares[j - 1, j] = shares[j, j] = 0.5
    return shares


def attached_trailed_forces(gamma, va_panels, v_induced, legs, rho: float, xp=np):
    """Kutta-Joukowski force on the AT legs (Gaunaa, Li & Pirrung 2026, Sec. 3),
    VSM ``compute_attached_trailed_vortex_forces``:
    ``rho (gamma_j - gamma_{j-1}) (v_rel_j x l_j)``, ``gamma_{-1} = gamma_n = 0``,
    with ``v_rel_j`` the mean of the neighbouring panels' inflow plus
    ``v_induced`` (the full vortex system's induction at the AT points,
    ``(n+1, 3)``). Returns ``(n+1, 3)``.
    """
    n = va_panels.shape[0]
    to_boundary = np.zeros((n + 1, n))  # gamma_net = D gamma
    to_boundary[np.arange(n), np.arange(n)] += 1.0
    to_boundary[np.arange(1, n + 1), np.arange(n)] -= 1.0
    mean = np.abs(to_boundary) * np.where(
        (np.arange(n + 1) > 0) & (np.arange(n + 1) < n), 0.5, 1.0
    )[:, None]
    if _is_casadi(xp):
        gamma_net = xp.mtimes(xp.DM(to_boundary), gamma)
        va_boundary = xp.mtimes(xp.DM(mean), va_panels)
    else:
        gamma_net = to_boundary @ np.asarray(gamma, dtype=float).ravel()
        va_boundary = mean @ va_panels
    legs = xp.DM(legs) if _is_casadi(xp) and isinstance(legs, np.ndarray) else legs
    return scale_rows(rho * gamma_net, cross_rows(va_boundary + v_induced, legs, xp), xp)


def total_force_and_moment(force, moment_local, arm, xp=np):
    """Sum of panel forces and of their moments about a reference point,
    ``arm`` being ``application_point - reference_point`` per panel (VSM takes
    the panel aerodynamic centre)."""
    moments = moment_local + cross_rows(arm, force, xp)
    if _is_casadi(xp):
        return xp.sum1(force).T, xp.sum1(moments).T
    return force.sum(axis=0), moments.sum(axis=0)


def lift_drag_side_vsm(force, va_panels, va_ref, spanwise_direction, xp=np, extra_forces=()):
    """VSM's global lift / drag / side force bookkeeping (``compute_results``).

    Each panel's force is first decomposed in ITS OWN inflow frame -- lift
    along ``va_i x spanwise`` (normalised), drag along ``va_i``, side along
    ``lift x drag`` -- and each component is then projected on the
    corresponding REFERENCE direction built from ``va_ref``. On a rotating
    body the local inflow turns along the span, so this differs from
    projecting the total force on the reference inflow (by 8 % in drag on the
    LEI-V3 at its trim transport rate); it is the definition every reported
    ``cl``/``cd`` in this repo uses. ``extra_forces`` (e.g. bridle segments)
    are projected directly on the reference directions, as VSM does.

    Returns the three force sums [N]; divide by ``q_ref S`` for coefficients.
    """
    n = force.shape[0]
    va_ref = as_xp_vector(np.asarray(va_ref, dtype=float), xp) if isinstance(va_ref, np.ndarray) else va_ref
    span = (
        as_xp_vector(np.asarray(spanwise_direction, dtype=float), xp)
        if isinstance(spanwise_direction, (np.ndarray, list, tuple))
        else spanwise_direction  # symbolic
    )
    ref_unit = va_ref / norm3(va_ref, xp)
    lift_ref = cross3(va_ref, span, xp)
    lift_ref = lift_ref / norm3(lift_ref, xp)
    side_ref = cross3(lift_ref, ref_unit, xp)

    va_unit = unit_rows(va_panels, xp)
    dir_lift = unit_rows(cross_rows(va_panels, tile_vector(spanwise_direction, n, xp), xp), xp)
    dir_side = cross_rows(dir_lift, va_unit, xp)
    lift_local = dot_rows(force, dir_lift, xp)
    drag_local = dot_rows(force, va_unit, xp)
    side_local = dot_rows(force, dir_side, xp)
    lift = dot_rows(scale_rows(lift_local, dir_lift, xp), tile_vector(lift_ref, n, xp), xp)
    drag = dot_rows(scale_rows(drag_local, va_unit, xp), tile_vector(ref_unit, n, xp), xp)
    side = dot_rows(scale_rows(side_local, dir_side, xp), tile_vector(side_ref, n, xp), xp)
    if _is_casadi(xp):
        lift, drag, side = xp.sum1(lift), xp.sum1(drag), xp.sum1(side)
    else:
        lift, drag, side = lift.sum(), drag.sum(), side.sum()
    for f in extra_forces:
        lift = lift + dot3(f, lift_ref, xp)
        drag = drag + dot3(f, ref_unit, xp)
        side = side + dot3(f, side_ref, xp)
    return lift, drag, side


# ------------------------------------------------------------------ line drag
def order_line_endpoints(p1, p2):
    """VSM orders each bridle segment by height before charging it; the sign
    of the small friction term depends on it, so do the same (numeric)."""
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    return (p2, p1) if p1[2] > p2[2] else (p1, p2)


def line_force(va, p1, p2, diameter, rho: float, cd_cable: float = 1.1, cf_cable: float = 0.01, xp=np):
    """Aerodynamic force on one straight line segment, VSM
    ``compute_line_aerodynamic_force``: crossflow drag ``cd sin^3`` plus skin
    friction ``pi cf cos^3`` along the inflow, and the crossflow-induced lift
    ``cd sin^2 cos - pi cf sin cos^2`` along the segment's projection normal
    to the inflow. ``p1, p2`` must already be height-ordered
    (:func:`order_line_endpoints`); ``diameter`` is the DRAG diameter.
    """
    # Endpoints numeric (ndarray, the NumPy path and the fixed-geometry
    # CasADi graph) or symbolic (CasADi parameters of a geometry that changes
    # per solve); the length then is an expression too.
    if isinstance(p1, (np.ndarray, list, tuple)):
        p1 = as_xp_vector(np.asarray(p1, dtype=float), xp)
    if isinstance(p2, (np.ndarray, list, tuple)):
        p2 = as_xp_vector(np.asarray(p2, dtype=float), xp)
    chord_line = p2 - p1
    length = norm3(chord_line, xp)
    e_line = chord_line / length
    speed = norm3(va, xp)
    cos_theta = dot3(va, e_line, xp) / speed
    theta = xp.arccos(cos_theta) if xp is np else xp.acos(cos_theta)
    sin_t, cos_t = xp.sin(theta), xp.cos(theta)
    cd_t = cd_cable * sin_t**3 + np.pi * cf_cable * cos_t**3
    cl_t = cd_cable * sin_t**2 * cos_t - np.pi * cf_cable * sin_t * cos_t**2
    dir_drag = va / speed
    dir_lift = -(e_line - dot3(e_line, dir_drag, xp) * dir_drag)
    q_area = 0.5 * rho * speed**2 * length * diameter
    return q_area * (cl_t * dir_lift + cd_t * dir_drag)
