import numpy as np
import pytest

from awetrim.aerostructural.billowing import (
    arc_half_angle,
    billow_wing,
    panel_bays,
    refine_billowed_wing,
)
from awetrim.aerostructural.coupled.awesio_structure import load_structure
from awetrim.utils.config_paths import REPO_ROOT

STRUCTURE = REPO_ROOT / "data" / "LEI-V3-KITE" / "structure.yml"


@pytest.fixture(scope="module")
def lei_v3_wing():
    """Sections (LE, TE) and strut sections of the LEI-V3 full structure."""
    structure = load_structure(STRUCTURE)
    le = structure.positions[structure.leading_edge]
    te = structure.positions[structure.trailing_edge]
    struts = sorted(structure.leading_edge.index(s["nodes"][0]) for s in structure.struts)
    return le, te, struts


def test_arc_half_angle_inverts_the_chord_ratio():
    for theta in (0.1, 0.5, 1.0):
        assert arc_half_angle(np.sin(theta) / theta) == pytest.approx(theta, rel=1e-10)
    assert arc_half_angle(1.0) == 0.0


def test_inner_bays_contract_by_p_and_keep_their_fabric(lei_v3_wing):
    le, te, struts = lei_v3_wing
    wing = billow_wing(le, te, struts, 0.05)
    for bay in wing.bays[1:-1]:
        assert bay.chord == pytest.approx(0.95 * bay.chord_built, rel=1e-9)
    le_f, te_f = refine_billowed_wing(wing, 50)
    for bay in wing.bays[1:-1]:
        arc = te_f[bay.first * 50:bay.last * 50 + 1]
        length = np.linalg.norm(np.diff(arc, axis=0), axis=1).sum()
        assert length == pytest.approx(bay.fabric_length, rel=2e-4)


def test_struts_rotate_rigidly_about_their_leading_edge(lei_v3_wing):
    le, te, struts = lei_v3_wing
    wing = billow_wing(le, te, struts, 0.05)
    for k in struts:
        np.testing.assert_allclose(np.linalg.norm(wing.trailing_edge[k] - wing.leading_edge[k]),
                                   np.linalg.norm(te[k] - le[k]), rtol=1e-12)
    for k in struts[1:-1]:
        np.testing.assert_allclose(wing.leading_edge[k], le[k])


def test_zero_contraction_keeps_the_struts(lei_v3_wing):
    le, te, struts = lei_v3_wing
    wing = billow_wing(le, te, struts, 0.0)
    np.testing.assert_allclose(wing.leading_edge, le)
    np.testing.assert_allclose(wing.trailing_edge[struts], te[struts])


def test_billowed_wing_is_mirror_symmetric(lei_v3_wing):
    le, te, struts = lei_v3_wing
    wing = billow_wing(le, te, struts, 0.05)
    mirror = wing.trailing_edge[::-1] * [1.0, -1.0, 1.0]
    np.testing.assert_allclose(wing.trailing_edge, mirror, atol=1e-9)


def test_panel_bays_follow_the_sections():
    bays = panel_bays(n_panels=27 * 5, n_per_section=5, strut_sections=[0, 3, 6, 9, 12, 15, 18, 21, 24, 27])
    assert bays[0] == 0 and bays[-1] == 8
    assert np.all(bays[:15] == 0) and np.all(bays[15:30] == 1)
