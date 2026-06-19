"""Plumbing test for the frozen-geometry alpha sweep.

Uses a fake VSM body/solver so the routine is exercised without a real VSM
solve: it checks the imposed apparent-wind alpha convention, the row schema,
and that phi_a is computed from the summed panel force.
"""

import numpy as np
import pytest

from awetrim.aerostructural import aerodynamic_vsm


class _FakePanel:
    def __init__(self, cl, cd):
        self.cl = cl
        self.cd = cd


class _FakeBody:
    def __init__(self):
        self.va = None
        self.panels = [_FakePanel(1.0, 0.1), _FakePanel(1.0, 0.1)]

    def va_initialize(
        self,
        *,
        Umag,
        angle_of_attack,
        side_slip=0.0,
        body_rates=0.0,
        body_axis=None,
        reference_point=None,
        rates_in_body_frame=False,
    ):
        """Mimic VSM's freestream va = Umag*(cos a cos b, sin b, sin a).

        The fake solver returns a constant force, so the per-panel rotational
        inflow from ``body_rates`` is irrelevant here; we only need the imposed
        freestream vector to exercise the sweep's alpha/sideslip convention.
        """
        a, b = np.radians(angle_of_attack), np.radians(side_slip)
        self.va = Umag * np.array(
            [np.cos(a) * np.cos(b), np.sin(b), np.sin(a)]
        )


class _FakeSolver:
    """Returns a constant force purely along the lift direction (+z) so phi_a≈0."""

    def __init__(self):
        self.received_va = []

    def solve(self, body_aero):
        self.received_va.append(np.asarray(body_aero.va, dtype=float).copy())
        return {
            "cl": np.array([1.0, 1.0]),
            "cd": np.array([0.1, 0.1]),
            "F_distribution": np.array([[0.0, 0.0, 25.0], [0.0, 0.0, 25.0]]),
        }


def test_frozen_alpha_sweep_rows_and_alpha_convention():
    body = _FakeBody()
    solver = _FakeSolver()
    alphas_deg = np.array([-5.0, 0.0, 10.0])
    rows = aerodynamic_vsm.run_frozen_geometry_alpha_sweep(
        body, solver, va_magnitude=20.0, alpha_values_deg=alphas_deg
    )

    assert len(rows) == 3
    for row, alpha_deg in zip(rows, alphas_deg):
        assert set(row) == {"alpha", "cl", "cd", "phi_a", "v_a", "success"}
        assert row["alpha"] == np.radians(alpha_deg)
        assert row["v_a"] == 20.0
        assert row["cl"] == 1.0
        assert row["cd"] == 0.1
        assert row["success"] is True
        # Force purely along +z (lift dir) -> aerodynamic roll ~ 0.
        assert abs(row["phi_a"]) < 1e-9

    # alpha = atan2(va_z, va_x): imposed apparent wind matches the requested AoA.
    for va_vec, alpha_deg in zip(solver.received_va, alphas_deg):
        assert np.arctan2(va_vec[2], va_vec[0]) == pytest.approx(np.radians(alpha_deg))
        assert np.linalg.norm(va_vec) == pytest.approx(20.0)


def test_frozen_alpha_sweep_imposes_sideslip_and_axis():
    body = _FakeBody()
    solver = _FakeSolver()
    rows = aerodynamic_vsm.run_frozen_geometry_alpha_sweep(
        body,
        solver,
        va_magnitude=20.0,
        alpha_values_deg=[6.0],
        side_slip_deg=8.0,
    )
    assert rows[0]["success"] is True
    (va_vec,) = solver.received_va
    # The inflow follows VSM's wind-axes parametrisation
    # va = |va| * (cos a cos b, sin b, sin a) (same as body.va_initialize), so
    # the components — not atan2(va_z, va_x) — encode the requested a and b.
    a, b = np.radians(6.0), np.radians(8.0)
    assert va_vec[0] == pytest.approx(20.0 * np.cos(a) * np.cos(b))
    assert va_vec[1] == pytest.approx(20.0 * np.sin(b))
    assert va_vec[2] == pytest.approx(20.0 * np.sin(a))


def test_frozen_alpha_sweep_handles_solver_failure():
    class _BadSolver:
        def solve(self, body_aero):
            raise RuntimeError("VSM blew up")

    rows = aerodynamic_vsm.run_frozen_geometry_alpha_sweep(
        _FakeBody(), _BadSolver(), va_magnitude=15.0, alpha_values_deg=[0.0, 5.0]
    )
    assert len(rows) == 2
    assert all(r["success"] is False for r in rows)
    assert all(np.isnan(r["cl"]) for r in rows)
