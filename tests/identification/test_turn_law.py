"""awetrim.identification.turn_law: b(u_p), c5, c6 recovered from states that obey Eq. 33."""

import numpy as np
import pandas as pd
import pytest

from awetrim.identification import turn_law as tl

PER_US = 1.4
B0, B1, C5, C6 = -0.02, 0.022, 0.0025, 0.0004


def _row(base, u_p, steer, state, mass_scale, gravity, a=0.4, m=14.0, v_a=24.0, v_tau=23.0):
    beta, _, chi = (np.radians(v) for v in state)
    mass = m * max(mass_scale, tl.MASS_FLOOR)
    b = B0 + B1 * u_p
    weight = mass * tl.GRAVITY if gravity else 0.0
    f_chi, f_n, f_r = (-weight * np.cos(chi) * np.cos(beta), -weight * np.sin(chi) * np.cos(beta),
                       -weight * np.sin(beta))
    u_s = steer / PER_US
    w = (a * v_a**2 * u_s - b * f_n - C5 * f_chi - C6 * f_r) / (v_a + b * mass * v_tau)
    return {"anchor": tl.state_anchor(base, tl.state_label(*state), mass_scale, gravity),
            "base": base, "state": tl.state_label(*state), "mass_scale": mass_scale * 1.0,
            "gravity": gravity, "elevation_deg": state[0], "azimuth_deg": state[1],
            "course_deg": state[2], "omega_z": -w, "va_x": v_a, "va_y": 0.0, "va_z": 0.0,
            "speed_tangential": v_tau, "power_tape_length": u_p, "steering_half_difference": steer,
            "mass_total": mass, "rho": 1.225, "converged": True}


@pytest.fixture
def states():
    rows = []
    for u_p in (1.4, 1.7, 2.1, 2.4):
        base = f"w5.0_steering_{u_p:.2f}/03"
        rows += [_row(base, u_p, 0.1, tl.CENTRE, s, False) for s in (1, 0)]
    grav = "w5.0_steering_1.95/07"
    rows += [_row(grav, 1.95, 0.2, tl.CENTRE, s, False) for s in (1, 0)]
    rows += [_row(grav, 1.95, 0.2, st, 1, True)
             for st in [(10, 0, 90), (30, 0, 0), (30, 0, 180), (30, 0, 270), (60, 0, 90)]]
    return tl.turn_law_table(pd.DataFrame(rows), steering_tape_per_unit_us=PER_US)


def test_mass_floor_reads_zero(states):
    assert set(states.mass_scale) == {0.0, 1.0}


def test_b_per_anchor_and_polynomial(states):
    inertia = tl.inertia_terms(states)
    assert len(inertia) == 5
    np.testing.assert_allclose(inertia.b, B0 + B1 * inertia.u_p, rtol=1e-4)  # the 1e-6 mass floor
    np.testing.assert_allclose(inertia.a, 0.4, rtol=1e-4)
    np.testing.assert_allclose(tl.fit_inertia_polynomial(inertia, 1), [B0, B1], rtol=1e-3)


def test_gravity_terms(states):
    c5, c6, rows = tl.gravity_terms(states, tl.inertia_terms(states))
    assert len(rows) == 5
    assert c5 == pytest.approx(C5, rel=1e-3) and c6 == pytest.approx(C6, rel=1e-3)


def test_measure_without_gravity_states(states):
    out = tl.measure(states[~states.gravity])
    assert out["gravity_course_coefficient"] == 0.0 and out["gravity_radial_coefficient"] == 0.0
    assert out["density_reference"] == pytest.approx(1.225)


def test_degree_capped_by_tapes(states):
    one_tape = tl.inertia_terms(states).iloc[:1]
    assert len(tl.fit_inertia_polynomial(one_tape, 3)) == 1


def test_no_pairs_raises(states):
    with pytest.raises(ValueError):
        tl.measure(states[states.mass_scale == 1.0])


def test_anchor_label():
    assert tl.state_anchor("c/03", tl.state_label(30, 0, 90), 1.0, True) == "c/03@b30_p0_c90@m1@g"
