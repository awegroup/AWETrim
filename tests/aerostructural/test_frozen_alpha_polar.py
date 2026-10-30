"""Plumbing test for the frozen-shape alpha polar about a trim.

A fake VSM body (a flat two-panel wing in the VSM frame) and solver stand in
for VSM: they check that the polar keeps |v_a|, the side slip and the
rotational inflow, steps the FRD angle of attack by exactly the increment,
re-solves the trim's own inflow at 0, and marches the circulation seed
outward from 0.
"""

import numpy as np
import pytest

from awetrim.aerostructural import aerodynamic_vsm


class _Panel:
    def __init__(self, y1, y2):
        # Chord along +x (leading edge -> trailing edge), span along y.
        self.LE_point_1, self.LE_point_2 = np.array([0.0, y1, 0.0]), np.array([0.0, y2, 0.0])
        self.TE_point_1, self.TE_point_2 = np.array([1.0, y1, 0.0]), np.array([1.0, y2, 0.0])


class _Body:
    def __init__(self, va, omega):
        self.panels = [_Panel(-2.0, -0.5), _Panel(-0.5, 0.5), _Panel(0.5, 2.0)]
        self._set(va, body_rates=float(np.linalg.norm(omega)),
                  body_axis=np.asarray(omega) / np.linalg.norm(omega),
                  reference_point=np.zeros(3))

    def _get(self):
        return self._va

    def _set(self, va, body_rates=0.0, body_axis=None, reference_point=None,
             rates_in_body_frame=False):
        self._va = np.asarray(va, dtype=float)
        self.body_rates = body_rates * np.asarray(body_axis, dtype=float)
        self.reference_point = np.asarray(reference_point, dtype=float)

    va = property(_get, _set)


class _Solver:
    def __init__(self):
        self.calls = []

    def solve(self, body, gamma_distribution=None):
        self.calls.append((body.va.copy(), body.body_rates.copy(), gamma_distribution))
        return {"Fx": body.va[0], "Fy": body.va[1], "Fz": 10.0 * body.va[2],
                "gamma_converged": True,
                "gamma_distribution": np.full(3, float(len(self.calls)))}


def _alpha_beta(va):
    v = -va * np.array([-1.0, 1.0, -1.0])  # FRD components of -va
    return np.rad2deg(np.arctan2(v[2], v[0])), np.rad2deg(np.arcsin(v[1] / np.linalg.norm(v)))


def test_body_centre_panel_axes_are_frd_of_the_flat_wing():
    body = _Body([20.0, 0.0, 2.0], [0.0, -0.07, 0.0])
    np.testing.assert_allclose(
        aerodynamic_vsm.body_centre_panel_axes(body),
        [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]], atol=1e-12,
    )


def test_polar_steps_alpha_and_keeps_speed_slip_and_rates():
    va0 = np.array([19.0, 1.5, 3.0])
    omega = np.array([0.0, -0.07, 0.01])
    body = _Body(va0, omega)
    solver = _Solver()
    rows = aerodynamic_vsm.run_frozen_alpha_polar_about_trim(
        body, solver, [-2.0, 3.0, 1.0]
    )
    assert [r["d_alpha_deg"] for r in rows] == [-2.0, 0.0, 1.0, 3.0]
    alpha0, beta0 = _alpha_beta(va0)
    for row in rows:
        va = row["va_world"]
        assert np.linalg.norm(va) == pytest.approx(np.linalg.norm(va0))
        alpha, beta = _alpha_beta(va)
        assert alpha == pytest.approx(alpha0 + row["d_alpha_deg"])
        assert row["alpha_body_deg"] == pytest.approx(alpha)
        assert beta == pytest.approx(beta0)
    # The 0 step re-solves the trim's own inflow, and every solve keeps the
    # rotational inflow.
    np.testing.assert_allclose(rows[1]["va_world"], va0)
    for _va, rates, _seed in solver.calls:
        np.testing.assert_allclose(rates, omega)
    # The original body is untouched.
    np.testing.assert_array_equal(body.va, va0)


def test_polar_marches_the_seed_outward_from_zero():
    body = _Body([20.0, 0.0, 2.0], [0.0, -0.07, 0.0])
    solver = _Solver()
    aerodynamic_vsm.run_frozen_alpha_polar_about_trim(body, solver, [-2.0, -1.0, 1.0, 2.0])
    seeds = [None if s is None else float(s[0]) for _va, _r, s in solver.calls]
    # 0 (cold), +1 and +2 from it, then -1 from 0's solution, -2 from -1's.
    assert seeds == [None, 1.0, 2.0, 1.0, 4.0]
