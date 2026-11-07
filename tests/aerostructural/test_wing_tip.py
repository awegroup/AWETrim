import logging

import numpy as np
import pytest

from awetrim.aerostructural.coupled.awesio_structure import load_structure
from awetrim.aerostructural.wing_tip import (
    MAX_PANEL_SWEEP_DEG,
    correct_tip_bays,
    panel_sweep_deg,
)
from awetrim.utils.config_paths import REPO_ROOT

STRUCTURE = REPO_ROOT / "data" / "LEI-V3-KITE" / "structure.yml"


@pytest.fixture(scope="module")
def lei_v3_sections():
    structure = load_structure(STRUCTURE)
    return structure.positions[structure.leading_edge], structure.positions[structure.trailing_edge]


def flat_wing(n=7, span=8.0, chord=1.5):
    y = np.linspace(span / 2, -span / 2, n)
    le = np.column_stack([np.zeros(n), y, np.zeros(n)])
    return le, le + [chord, 0.0, 0.0]


def tube_length(le, te, outer, tip):
    return np.linalg.norm(le[tip] - le[outer]) + np.linalg.norm(te[tip] - le[tip])


def test_sweep_of_an_unswept_and_a_swept_panel():
    le = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 2.0, 0.0]])
    te = le + [1.0, 0.0, 0.0]
    np.testing.assert_allclose(panel_sweep_deg(le, te), [0.0, 45.0], atol=1e-12)


def test_the_lei_v3_tip_bays_are_beyond_the_limit_as_built(lei_v3_sections):
    le, te = lei_v3_sections
    t = np.linspace(0, 1, 6)[:, None]
    bay = panel_sweep_deg((1 - t) * le[1] + t * le[0], (1 - t) * te[1] + t * te[0])
    assert bay.max() > 60.0


def test_corrected_tip_keeps_the_tube_and_meets_the_limit(lei_v3_sections):
    le, te = lei_v3_sections
    out = correct_tip_bays(le, te)
    for outer, tip in ((1, 0), (len(le) - 2, len(le) - 1)):
        assert tube_length(out, te, outer, tip) == pytest.approx(tube_length(le, te, outer, tip), rel=1e-9)
        t = np.linspace(0, 1, 11)[:, None]
        bay = panel_sweep_deg((1 - t) * out[outer] + t * out[tip], (1 - t) * te[outer] + t * te[tip])
        assert bay.max() <= MAX_PANEL_SWEEP_DEG + 1e-9
        assert np.linalg.norm(te[tip] - out[tip]) > np.linalg.norm(te[tip] - le[tip])  # tip chord grows
    np.testing.assert_array_equal(out[1:-1], le[1:-1])
    np.testing.assert_allclose(out, out[::-1] * [1.0, -1.0, 1.0], atol=1e-6)


def test_an_unswept_wing_is_left_alone(caplog):
    le, te = flat_wing()
    with caplog.at_level(logging.INFO):
        out = correct_tip_bays(le, te)
    np.testing.assert_array_equal(out, le)
    assert not caplog.records


def test_an_inner_bay_beyond_the_limit_warns(caplog):
    le, te = flat_wing()
    le[3:] += [2.5, 0.0, 0.0]  # a step back in the leading edge: bay 2-3 swept ~70 deg
    te[3:] += [2.5, 0.0, 0.0]
    with caplog.at_level(logging.WARNING):
        correct_tip_bays(le, te)
    assert any("swept beyond" in r.message and "(2," in r.message for r in caplog.records)
