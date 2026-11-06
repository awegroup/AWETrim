"""The forward march's diagnosis of a node where every solve attempt failed."""

import casadi as ca
import numpy as np

from awetrim.timeseries.phase_parametrized import (
    FOLD_SINGULAR_RATIO,
    _least_squares_node,
    _node_equation_names,
)


def _solver(g_expr, x, p):
    return ca.nlpsol("s", "ipopt", {"x": x, "p": p, "f": 0, "g": g_expr},
                     {"ipopt.print_level": 0, "print_time": False})


def test_a_fold_has_no_root_and_a_vanishing_singular_value():
    x, p = ca.MX.sym("x"), ca.MX.sym("p")
    solver = _solver(x**2 + p, x, p)  # no root for p > 0; the closest point x = 0 is a fold
    z, g, sv, _ = _least_squares_node(solver, [0.3], [1.0], [-10.0], [10.0])
    assert abs(float(g[0]) - 1.0) < 1e-6
    assert sv[-1] <= FOLD_SINGULAR_RATIO * max(sv[0], 1e-300) or sv[0] < 1e-6


def test_a_root_the_solver_missed_is_found():
    x, p = ca.MX.sym("x", 2), ca.MX.sym("p")
    solver = _solver(ca.vertcat(x[0] - p, x[0] + 3 * x[1]), x, p)
    z, g, sv, _ = _least_squares_node(solver, [0.5, 0.5], [2.0], [-10.0, -10.0], [10.0, 10.0])
    np.testing.assert_allclose(z, [2.0, -2.0 / 3.0], atol=1e-8)
    assert np.linalg.norm(g) < 1e-10
    assert sv[-1] > FOLD_SINGULAR_RATIO * sv[0]


def test_equation_names_follow_the_node_layout():
    assert _node_equation_names(4, False) == ["tangential", "normal (turn)", "radial", "winch law"]
    assert _node_equation_names(7, True)[-1] == "winch law"


def test_the_node_jacobian_is_built_once_per_solver():
    x, p = ca.MX.sym("x", 2), ca.MX.sym("p")
    solver = _solver(ca.vertcat(x[0] - p, x[0] + 3 * x[1]), x, p)
    cache = {}
    _least_squares_node(solver, [0.5, 0.5], [2.0], [-10.0, -10.0], [10.0, 10.0], cache=cache)
    built = cache[id(solver)]
    z, _, _, _ = _least_squares_node(solver, [1.0, 0.0], [3.0], [-10.0, -10.0], [10.0, 10.0], cache=cache)
    assert cache[id(solver)] is built and len(cache) == 1
    np.testing.assert_allclose(z, [3.0, -1.0], atol=1e-8)
