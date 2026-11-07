"""The CasADi trim with the 3/4-chord force directions (``is_aoa_corrected=False``)
against frozen outputs.

The reference was written with AWETrim before the force law moved to VSM
(``panel_kernels`` restated in AWETrim, VSM 1201f6f) on the step-1 LEI-V3 test
kite: moving the law upstream and the call-overhead changes must not move a
``False`` trim. Regenerate (only for a deliberate model change) with
``AWETRIM_REGEN=1 pytest tests/aerodynamics/test_trim_casadi_regression.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

casadi = pytest.importorskip("casadi")
pytest.importorskip("VSM")

from awetrim.aerodynamics.trim_casadi import CasadiTrim, CasadiTrimOptions  # noqa: E402
from conftest import step1_trim_env  # noqa: E402

REFERENCE = Path(__file__).parent / "data" / "casadi_trim_aoa_false_reference.npz"
#: relative tolerance of every frozen output (round-off of reordered sums).
#: The trim state, circulation, forces and Jacobians move at <= 2e-14 when the
#: force law moves to VSM; the aerodynamic moment moves at ~1.4e-12 of its
#: largest component, in the components that are zero by symmetry (VSM's
#: operation order for the section moment, cancelling cross products).
RTOL = 2e-12


@pytest.fixture(scope="module")
def lei_v3(lei_v3_step1_kite):
    return step1_trim_env(lei_v3_step1_kite)


def _outputs(env) -> dict[str, np.ndarray]:
    """Both tether settings, solved in a fixed order (trims rotate the body)."""
    v = env["values"]
    out = {}
    for tether in (False, True):
        tag = "tether" if tether else "tetherless"
        trim = CasadiTrim(
            env["body"], env["system_model"], v["center_of_gravity"], v["reference_point"],
            kcu_drag=env["kcu"],
            options=CasadiTrimOptions(tether_model="williams" if tether else None, is_aoa_corrected=False),
        )
        res = trim.solve(v["x_guess"])
        x, gamma = np.asarray(res["opt_x"], float), np.asarray(res["gamma_distribution"], float).ravel()
        length = float(res.get("williams_tether_length", 0.0) or 0.0)
        aic = trim.compute_aic(x)
        out[f"{tag}/opt_x"] = x
        out[f"{tag}/gamma"] = gamma
        out[f"{tag}/coefficients"] = np.array([res["cl"], res["cd"], res["cs"]], float)
        out[f"{tag}/tether_length"] = np.array([length])
        out[f"{tag}/residual"] = np.asarray(trim.residual(x, gamma, aic, length)[0], float).ravel()
        out[f"{tag}/jacobian"] = np.asarray(trim.jacobian(x, gamma, aic, length, exact=True), float)
        out[f"{tag}/jacobian_avg"] = np.asarray(trim.jacobian(x, gamma, aic, length), float)
        outputs = trim.outputs(x, gamma, aic, length)
        for key in ("f_aero_w", "m_aero_w", "force_panels_b", "alpha"):
            out[f"{tag}/{key}"] = np.asarray(outputs[key], float)
    return out


def test_false_trims_match_the_frozen_reference(lei_v3):
    out = _outputs(lei_v3)
    if os.environ.get("AWETRIM_REPORT") == "1":
        ref = np.load(REFERENCE)
        for key in ref.files:
            scale = max(float(np.max(np.abs(ref[key]))), 1e-300)
            print(f"{key:28s} {float(np.max(np.abs(out[key] - ref[key]))) / scale:.2e}")
    if os.environ.get("AWETRIM_REGEN") == "1":
        REFERENCE.parent.mkdir(parents=True, exist_ok=True)
        np.savez(REFERENCE, **out)
        pytest.skip(f"reference written to {REFERENCE}")
    ref = np.load(REFERENCE)
    assert sorted(ref.files) == sorted(out)
    for key in ref.files:
        expected, got = ref[key], out[key]
        scale = max(float(np.max(np.abs(expected))), 1e-300)
        assert got.shape == expected.shape, key
        # residuals are ~0 at a converged trim: hold them to the trim's own tolerance
        tol = 1e-9 if key.endswith("/residual") else RTOL * scale
        assert np.max(np.abs(got - expected)) <= tol, (key, float(np.max(np.abs(got - expected))) / scale)
