"""Slow integration test: the frozen-geometry alpha sweep is centred on the anchor.

Opt-in: marked ``slow`` and deselected from the default ``pytest`` run
(``addopts = -m 'not slow'`` in pyproject.toml). Run it with::

    pytest -m slow tests/aerostructural/test_frozen_alpha_sweep_slow.py

Motivation
----------
``run_frozen_geometry_alpha_sweep`` is meant to recover the aerodynamic
``C_L(alpha)`` / ``C_D(alpha)`` / ``phi_a(alpha)`` response of a *frozen* deformed
shape, so that a swept row at the anchor's angle of attack should land back on
the anchor quasi-steady state.  The sweep imposes the apparent wind as
``va = |va| * (cos a, 0, sin a)`` (zero sideslip, zero body rate).  The QS trim
imposes its inflow through ``body.va_initialize(Umag, aoa, side_slip, body_rates,
body_axis=-radial)`` — the *same* ``va`` setter and vector formula.  These two
paths therefore agree **exactly** for the symmetric component of the inflow, but
the sweep deliberately discards the anchor's sideslip and body-rate.

This test pins the behaviour with a real VSM solve on the LEI-V3 geometry:

1. ``test_frozen_sweep_reproduces_symmetric_anchor`` — for a straight anchor
   (beta = 0, no body rate) the swept row at the anchor ``aoa_course`` reproduces
   the anchor cl/cd/phi_a, and the +/-2 deg neighbours bracket it.
2. ``test_default_sweep_drops_turning_anchor_sideslip`` — the DEFAULT sweep
   (no sideslip/rate args) at a turning anchor's ``aoa_course`` does NOT reproduce
   it: phi_a collapses toward zero.  This is the "anchors and anchors+sweep look
   too different" symptom — the default isolates the symmetric response.
3. ``test_frozen_sweep_with_sideslip_reproduces_turning_anchor`` — passing the
   anchor's ``side_slip_deg`` and ``body_rates`` makes the swept row at the
   anchor ``aoa_course`` reproduce the turning anchor (its sideslip-driven
   aerodynamic roll included).  This is the correct way to sweep around an anchor.

Needs the VSM external solver plus the data/LEI-V3-KITE files; skips cleanly if
either is unavailable.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("VSM", reason="Vortex-Step-Method not installed")

from awetrim.aerodynamics.vsm_quasi_steady import DEFAULT_AXES, _default_vsm_solver
from awetrim.aerostructural.aerodynamic_vsm import run_frozen_geometry_alpha_sweep
from awetrim.identification.aero_dataset import aerodynamic_roll

pytestmark = pytest.mark.slow

REPO_ROOT = Path(__file__).resolve().parents[2]
KITE_DIR = REPO_ROOT / "data" / "LEI-V3-KITE"

if not (KITE_DIR / "aero_geometry.yaml").exists():
    pytest.skip(f"kite data dir missing: {KITE_DIR}", allow_module_level=True)

# Operating point for the imposed apparent wind. aoa0 sits comfortably below
# stall so cl rises monotonically across the +/-2 deg sweep.
UMAG = 22.0
AOA0_DEG = 6.0


def _load_aero_scripts_common():
    """Import scripts/aerodynamics/common.py under a unique name (no path clash).

    Both scripts/aerodynamics/common.py and scripts/aerostructural/common.py are
    called ``common``; load this one under a distinct module name so it does not
    collide with the one used by the PSS slow test.
    """
    path = REPO_ROOT / "scripts" / "aerodynamics" / "common.py"
    spec = importlib.util.spec_from_file_location("aerodynamics_scripts_common", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def body_and_solver():
    """Build the real LEI-V3 VSM body once and share it across the tests."""
    common = _load_aero_scripts_common()
    parser = argparse.ArgumentParser()
    common.add_common_arguments(parser)
    args = parser.parse_args([])  # LEI-V3 defaults (18 panels, uniform spanwise)
    try:
        body, _props = common.build_body(args)
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"could not build VSM body: {exc}")
    solver = _default_vsm_solver(np.zeros(3))
    return body, solver


def _anchor_reference(body, solver, *, umag, aoa_deg, beta_deg, rate):
    """Impose the inflow exactly as the QS trim does and return the anchor state.

    Mirrors the ``solve_vsm_quasi_steady_trim`` post-processing: the inflow is set
    via ``va_initialize`` (sideslip + body rate included) and ``phi_a`` is the tilt
    of the total aero force about the freestream apparent wind.

    The returned ``aoa_deg`` is the angle of attack passed to ``va_initialize`` —
    in the real trim this is exactly the reported ``aoa_course_deg`` (the trim
    feeds ``atan2(va_z, va_x)`` straight into ``va_initialize``), so it is the
    correct centre to sweep around.  Note VSM's wind-axes parametrisation means
    ``atan2(va_z, va_x) != aoa_deg`` once sideslip is nonzero, so the swept centre
    must use this ``aoa_deg``, not a re-derived freestream angle.
    """
    axes = DEFAULT_AXES
    body.va_initialize(
        Umag=umag,
        angle_of_attack=aoa_deg,
        side_slip=beta_deg,
        body_rates=rate,
        body_axis=-axes.radial,
        reference_point=np.zeros(3),
        rates_in_body_frame=False,
    )
    res = solver.solve(body)
    # Freestream apparent wind, the aerodynamic-roll reference (matches the sweep).
    aoa_rad, beta_rad = np.radians(aoa_deg), np.radians(beta_deg)
    va_free = umag * np.array(
        [np.cos(aoa_rad) * np.cos(beta_rad), np.sin(beta_rad), np.sin(aoa_rad)]
    )
    total_force = np.array(
        [float(res["Fx"]), float(res["Fy"]), float(res["Fz"])], dtype=float
    )
    return {
        "aoa_deg": float(aoa_deg),
        "cl": float(np.mean(np.asarray(res["cl"], dtype=float))),
        "cd": float(np.mean(np.asarray(res["cd"], dtype=float))),
        "phi_a_deg": float(np.rad2deg(aerodynamic_roll(total_force, va_free, axes.radial))),
    }


def test_frozen_sweep_reproduces_symmetric_anchor(body_and_solver):
    """A straight anchor is reproduced by the swept row at its aoa_course."""
    body, solver = body_and_solver
    anchor = _anchor_reference(
        body, solver, umag=UMAG, aoa_deg=AOA0_DEG, beta_deg=0.0, rate=0.0
    )
    assert anchor["aoa_deg"] == pytest.approx(AOA0_DEG, abs=1e-6)

    rows = run_frozen_geometry_alpha_sweep(
        body,
        solver,
        va_magnitude=UMAG,
        alpha_values_deg=[AOA0_DEG - 2.0, AOA0_DEG, AOA0_DEG + 2.0],
    )
    assert [r["success"] for r in rows] == [True, True, True]
    lo, mid, hi = rows

    # The middle row sits exactly at the anchor angle of attack.
    assert np.rad2deg(mid["alpha"]) == pytest.approx(AOA0_DEG, abs=1e-9)

    # ...and reproduces the anchor coefficients (same va setter + same solver).
    assert mid["cl"] == pytest.approx(anchor["cl"], abs=2e-3)
    assert mid["cd"] == pytest.approx(anchor["cd"], abs=2e-3)
    assert np.rad2deg(mid["phi_a"]) == pytest.approx(anchor["phi_a_deg"], abs=0.5)

    # The +/-2 deg neighbours bracket the anchor: cl monotonic, cd rising, and
    # the symmetric inflow keeps aerodynamic roll ~0 across the sweep.
    assert lo["cl"] < mid["cl"] < hi["cl"]
    assert lo["cd"] < mid["cd"] < hi["cd"]
    assert all(abs(np.rad2deg(r["phi_a"])) < 0.5 for r in rows)


def test_default_sweep_drops_turning_anchor_sideslip(body_and_solver):
    """The DEFAULT sweep (no sideslip/rate) does NOT reproduce a turning anchor.

    Documents why anchors with sideslip/body-rate and their *default* frozen
    sweeps look different — the default isolates the symmetric alpha response.
    """
    body, solver = body_and_solver
    anchor = _anchor_reference(
        body, solver, umag=UMAG, aoa_deg=AOA0_DEG, beta_deg=8.0, rate=0.5
    )
    (frozen,) = run_frozen_geometry_alpha_sweep(
        body,
        solver,
        va_magnitude=UMAG,
        alpha_values_deg=[anchor["aoa_deg"]],
    )
    assert frozen["success"]

    # The anchor carries a sizeable aerodynamic roll from its sideslip; the
    # default (zero-sideslip) sweep collapses it toward zero.
    assert abs(anchor["phi_a_deg"]) > 5.0
    assert abs(np.rad2deg(frozen["phi_a"])) < 1.0
    assert abs(np.rad2deg(frozen["phi_a"]) - anchor["phi_a_deg"]) > 5.0

    # Dropping the sideslip/body-rate inflow also shifts the wing coefficients.
    assert not np.isclose(frozen["cl"], anchor["cl"], atol=1e-2)


def test_frozen_sweep_with_sideslip_reproduces_turning_anchor(body_and_solver):
    """Passing the anchor's sideslip + body-rate reproduces the turning anchor.

    The correct way to sweep around an anchor: hold its ``side_slip_deg`` and
    ``body_rates`` fixed while varying alpha, so the swept row at the anchor's
    ``aoa_course`` lands back on the anchor cl/cd/phi_a (sideslip-driven
    aerodynamic roll included), and the +/-2 deg neighbours bracket it.
    """
    body, solver = body_and_solver
    beta_deg, rate = 8.0, 0.5
    anchor = _anchor_reference(
        body, solver, umag=UMAG, aoa_deg=AOA0_DEG, beta_deg=beta_deg, rate=rate
    )
    assert abs(anchor["phi_a_deg"]) > 5.0  # genuinely a turning/sideslipping state

    a0 = anchor["aoa_deg"]
    rows = run_frozen_geometry_alpha_sweep(
        body,
        solver,
        va_magnitude=UMAG,
        alpha_values_deg=[a0 - 2.0, a0, a0 + 2.0],
        side_slip_deg=beta_deg,
        body_rates=rate,
    )
    assert [r["success"] for r in rows] == [True, True, True]
    lo, mid, hi = rows

    # The swept row at the anchor angle of attack reproduces the turning anchor
    # (same va_initialize inflow + same solver), aerodynamic roll included.
    assert np.rad2deg(mid["alpha"]) == pytest.approx(a0, abs=1e-9)
    assert mid["cl"] == pytest.approx(anchor["cl"], abs=2e-3)
    assert mid["cd"] == pytest.approx(anchor["cd"], abs=2e-3)
    assert np.rad2deg(mid["phi_a"]) == pytest.approx(anchor["phi_a_deg"], abs=0.5)

    # The neighbours bracket it in lift, and the sideslip-driven roll persists
    # across the sweep (it does not collapse to zero).
    assert lo["cl"] < mid["cl"] < hi["cl"]
    assert all(abs(np.rad2deg(r["phi_a"])) > 5.0 for r in rows)
