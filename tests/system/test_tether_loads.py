"""Regression tests for tether load transfer to the kite.

Link tethers model the tether as a massless/dragless rigid or elastic link, so
their drag and gravity at the kite are zero. Defining these (rather than
leaving them missing) keeps the ``drag_tether_at_kite`` /
``force_gravity_tether_at_kite`` expressions and ``tension_tether_equation``
valid for every tether type; lumped/distributed tethers override them.
"""

import casadi as ca
import numpy as np

from awetrim.system.system_model import SystemModel
from awetrim.system.tether import (
    FlexibleLinkTether,
    RigidLinkTether,
    RigidLumpedTether,
)


def _is_zero_expr(expr) -> bool:
    """True if a free-variable-free CasADi expression is identically zero."""
    return np.allclose(np.array(ca.DM(ca.evalf(expr))).ravel(), 0.0)


def test_link_tethers_have_zero_drag_and_gravity_at_kite():
    model = SystemModel()  # default FlexibleLinkTether
    for tether in (FlexibleLinkTether(), RigidLinkTether()):
        assert _is_zero_expr(tether.drag_tether_at_kite_for(model))
        assert _is_zero_expr(tether.force_gravity_tether_at_kite_for(model))


def test_tension_tether_equation_builds_with_default_tether():
    """tension_tether_equation reads the drag/gravity-at-kite expressions; with
    the default link tether these are zero rather than missing (regression for
    the AttributeError that previously broke this with FlexibleLinkTether)."""
    model = SystemModel()
    equation = model.tension_tether_equation
    assert isinstance(equation, ca.MX)
    assert _is_zero_expr(model.expression("drag_tether_at_kite"))
    assert _is_zero_expr(model.expression("force_gravity_tether_at_kite"))


def test_lumped_tether_contributes_nonzero_drag_at_kite():
    """Sanity check the override path: a lumped tether's drag is a real
    (state-dependent) expression, not the zero default."""
    model = SystemModel(tether=RigidLumpedTether())
    drag = model.expression("drag_tether_at_kite")
    assert len(ca.symvar(drag)) > 0


def _evaluate(expr, values):
    """Evaluate ``expr`` with its free symbols set by name from ``values``."""
    symbols = ca.symvar(expr)
    missing = [s.name() for s in symbols if s.name() not in values]
    assert not missing, missing
    f = ca.Function("f", symbols, [expr])
    return np.array(f(*[values[s.name()] for s in symbols])).ravel()


def test_tether_inertia_is_off_by_default():
    model = SystemModel(tether=RigidLumpedTether())
    assert _is_zero_expr(model.expression("force_inertia_tether_at_kite"))


def test_tether_inertia_matches_the_straight_tether_derivation():
    """Course-frame load of a straight tether of mass m_t = rho A r on the kite:
    -(m_t/3)(v_tau_dot + 2 v_tau v_r / r), -(m_t/3) v_tau w_z,
    -m_t (v_r_dot - v_tau^2 / (2 r)), with w_z the course frame's rotation
    about the tether, (v_tau / r) tan(beta) sin(chi) - chi_dot."""
    tether = RigidLumpedTether(diameter=0.008, density=724.0, inertia=True)
    model = SystemModel(tether=tether, quasi_steady=False)
    values = {
        "speed_tangential": 30.0, "speed_radial": 3.0, "distance_radial": 300.0,
        "timeder_speed_tangential": 1.5, "timeder_speed_radial": -0.8,
        "angle_elevation": 0.5, "angle_course": 1.2, "timeder_angle_course": 0.3,
    }
    force = _evaluate(model.expression("force_inertia_tether_at_kite"), values)
    m_t = 724.0 * np.pi * 0.004**2 * 300.0
    v_t, v_r, r = 30.0, 3.0, 300.0
    w_z = v_t / r * np.tan(0.5) * np.sin(1.2) - 0.3
    expected = -np.array([
        m_t / 3 * (1.5 + 2 * v_t * v_r / r),
        m_t / 3 * v_t * w_z,
        m_t * (-0.8 - v_t**2 / (2 * r)),
    ])
    # rtol 1e-6: velocity_rotation_course_frame uses tan(beta + 1e-6)
    np.testing.assert_allclose(force, expected, rtol=1e-6)


def test_tether_inertia_enters_the_force_at_the_kite():
    """The load is part of force_tether_at_kite (so of the force balance and of
    the bridle resultant), nowhere else."""
    on = SystemModel(tether=RigidLumpedTether(inertia=True), quasi_steady=False)
    off = SystemModel(tether=RigidLumpedTether(), quasi_steady=False)
    diff = on.expression("force_tether_at_kite") - ca.substitute(
        off.expression("force_tether_at_kite"),
        ca.vertcat(*ca.symvar(off.expression("force_tether_at_kite"))),
        ca.vertcat(*[
            next(s for s in ca.symvar(on.expression("force_tether_at_kite")) if s.name() == o.name())
            for o in ca.symvar(off.expression("force_tether_at_kite"))
        ]),
    )
    inertia = on.expression("force_inertia_tether_at_kite")
    values = {s.name(): 0.7 + 0.1 * k for k, s in enumerate(ca.symvar(ca.vertcat(diff, inertia)))}
    np.testing.assert_allclose(_evaluate(diff, values), _evaluate(inertia, values), rtol=1e-12)
