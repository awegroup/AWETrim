"""awetrim.utils.time_scheme: the march's and the NLP's step and rate kernels."""

import casadi as ca
import numpy as np
import pytest

from awetrim.utils.time_scheme import (
    H_PREV_MIN,
    backward_rate,
    bdf2_rate,
    left_step,
    trapezoid_increment,
    trapezoid_step,
)


def test_steps():
    assert left_step(0.1, 0.5) == pytest.approx(0.2)
    assert trapezoid_step(0.1, 0.4, 0.6) == pytest.approx(0.2)
    assert trapezoid_increment(0.2, 3.0, 5.0) == pytest.approx(0.8)


def test_bdf2_is_exact_for_a_quadratic_on_uneven_steps():
    """Second order: the variable-step BDF2 difference is exact for
    y = t^2 (y' = 2t) whatever the step ratio; backward Euler is not."""
    f = lambda t: t**2
    for h, h_prev in ((0.1, 0.1), (0.1, 0.3), (0.25, 0.05)):
        t = 1.0
        rate = bdf2_rate(f(t), f(t - h), f(t - h - h_prev), h, h_prev)
        assert rate == pytest.approx(2 * t, rel=1e-12)
        assert backward_rate(f(t), f(t - h), h) != pytest.approx(2 * t, rel=1e-6)


def test_bdf2_blends_to_backward_euler():
    """second_order 0 (a node with one previous point) is backward Euler, and
    the blend is linear in it; a zero previous step is floored, not divided by."""
    y, y1, y2, h, hp = 2.0, 1.5, 1.2, 0.1, 0.2
    assert bdf2_rate(y, y1, y2, h, hp, second_order=0.0) == pytest.approx(backward_rate(y, y1, h))
    half = bdf2_rate(y, y1, y2, h, hp, second_order=0.5)
    assert half == pytest.approx(0.5 * (backward_rate(y, y1, h) + bdf2_rate(y, y1, y2, h, hp)))
    assert np.isfinite(bdf2_rate(y, y1, y2, h, 0.0, second_order=0.0))
    assert H_PREV_MIN > 0


def test_kernels_take_casadi():
    """The same kernels build the node solver's and the NLP's expressions."""
    y, y1, y2, h, hp = (ca.MX.sym(n) for n in ("y", "y1", "y2", "h", "hp"))
    expr = bdf2_rate(y, y1, y2, h, hp, xp=ca)
    f = ca.Function("f", [y, y1, y2, h, hp], [expr])
    assert float(f(2.0, 1.5, 1.2, 0.1, 0.2)) == pytest.approx(bdf2_rate(2.0, 1.5, 1.2, 0.1, 0.2))
