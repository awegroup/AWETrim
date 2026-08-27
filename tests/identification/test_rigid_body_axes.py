"""Tests for awetrim.identification.rigid_body_axes.

The body triad is anchored to the wing's CENTRE PANEL: x along the panel
chord, z the panel normal, y completing the right-handed set.  It must
therefore be independent of the mass distribution and of everything outboard
of that panel, follow a rigid rotation of the wing exactly, and come out in
the aircraft FRD sense.
"""

from pathlib import Path

import numpy as np
import pytest
import yaml

from awetrim.identification.rigid_body_axes import (
    FRD_IN_STRUC,
    RigidBodyAxes,
    compute_centre_panel_axes,
    compute_rigid_body_axes,
    load_psm_geometry,
    load_psm_nodes_and_masses,
    wing_le_te_indices,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
KITE_DIR = PROJECT_ROOT / "data" / "LEI-V3-KITE"


def _wing_cloud(span_stations, chord=2.0, height=10.0, kcu_mass=5.0):
    """KCU node + one LE/TE node pair per strut, in the PSM node ordering.

    Structural frame: +x aft (LE at -chord/2, TE at +chord/2), +y starboard,
    +z from the bridle point up to the wing.  Node 0 is the KCU at the origin,
    then strut k occupies node ids 2k+1 (LE) and 2k+2 (TE).
    """
    nodes = [np.zeros(3)]
    masses = [kcu_mass]
    for k, y in enumerate(span_stations):
        nodes.append(np.array([-0.5 * chord, y, height]))
        nodes.append(np.array([+0.5 * chord, y, height]))
        masses.extend([1.0 + 0.1 * k, 0.5 + 0.1 * k])
    le = np.arange(1, 2 * len(span_stations) + 1, 2)
    te = np.arange(2, 2 * len(span_stations) + 1, 2)
    return np.array(nodes), np.array(masses), le, te


def _rotation(axis, angle_deg):
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    a = np.radians(angle_deg)
    k = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(a) * k + (1.0 - np.cos(a)) * (k @ k)


def test_symmetric_wing_gives_the_canonical_frd_triad():
    nodes, masses, le, te = _wing_cloud([3.0, 1.0, -1.0, -3.0])
    rb = compute_rigid_body_axes(nodes, masses, le, te)
    assert isinstance(rb, RigidBodyAxes)
    np.testing.assert_allclose(rb.body_axes, FRD_IN_STRUC, atol=1e-12)


def test_axes_are_orthonormal_and_right_handed():
    nodes, masses, le, te = _wing_cloud([4.0, 2.0, -2.0, -4.0], chord=2.5)
    # Tilt and yaw the whole wing so the triad is not axis-aligned.
    rot = _rotation([0.3, 1.0, -0.2], 17.0)
    nodes = nodes @ rot.T
    axes = compute_rigid_body_axes(nodes, masses, le, te).body_axes
    np.testing.assert_allclose(axes @ axes.T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.cross(axes[0], axes[1]), axes[2], atol=1e-12)


def test_axes_do_not_depend_on_the_mass_distribution():
    """Half the point of the anchor: mass moves, axes do not."""
    nodes, masses, le, te = _wing_cloud([3.0, 1.0, -1.0, -3.0])
    light = compute_rigid_body_axes(nodes, masses, le, te)
    heavy_masses = masses.copy()
    heavy_masses[0] += 50.0  # a much heavier KCU
    heavy = compute_rigid_body_axes(nodes, heavy_masses, le, te)

    np.testing.assert_allclose(heavy.body_axes, light.body_axes, atol=1e-14)
    # ... while the mass properties they are reported with certainly do.
    assert not np.allclose(heavy.cg, light.cg)


def test_axes_ignore_everything_outboard_of_the_centre_panel():
    """The other half: only the two innermost struts orient the frame."""
    nodes, masses, le, te = _wing_cloud([3.0, 1.0, -1.0, -3.0])
    base = compute_centre_panel_axes(nodes, le, te).axes
    # Wreck the outboard struts: sweep, twist and droop the tips.
    for k in (0, 3):
        nodes[le[k]] += np.array([0.4, 0.0, -0.5])
        nodes[te[k]] += np.array([-0.3, 0.0, 0.6])
    moved = compute_centre_panel_axes(nodes, le, te).axes
    np.testing.assert_allclose(moved, base, atol=1e-14)


def test_axes_follow_a_rigid_rotation_of_the_wing():
    nodes, masses, le, te = _wing_cloud([3.0, 1.0, -1.0, -3.0])
    base = compute_rigid_body_axes(nodes, masses, le, te).body_axes
    rot = _rotation([0.2, -0.5, 1.0], 23.0)
    rotated = compute_rigid_body_axes(nodes @ rot.T, masses, le, te).body_axes
    # Rows are vectors: each rotates with the wing.
    np.testing.assert_allclose(rotated, base @ rot.T, atol=1e-12)


def test_x_is_exactly_the_centre_panel_chord():
    """The chord is taken exactly; the span only sets the roll about it."""
    nodes, masses, le, te = _wing_cloud([3.0, 1.0, -1.0, -3.0])
    # Steering-like asymmetry on the panel: sweep its starboard strut aft.
    nodes[le[1]] += np.array([0.12, 0.0, 0.05])
    panel = compute_centre_panel_axes(nodes, le, te)
    chord = panel.le_centre - panel.te_centre
    np.testing.assert_allclose(
        panel.axes[0], chord / np.linalg.norm(chord), atol=1e-12
    )
    np.testing.assert_allclose(panel.axes @ panel.axes.T, np.eye(3), atol=1e-12)


def test_pitch_reference_is_the_centre_panel_chord():
    """Pitching the panel alone tilts body x by exactly that angle."""
    tilt_deg = 6.0
    nodes, masses, le, te = _wing_cloud([3.0, 1.0, -1.0, -3.0])
    rot = _rotation([0.0, 1.0, 0.0], tilt_deg)
    for k in (1, 2):  # the two struts bounding the centre panel
        for idx in (le[k], te[k]):
            pivot = np.array([0.0, nodes[idx][1], nodes[idx][2]])
            nodes[idx] = pivot + rot @ (nodes[idx] - pivot)
    axes = compute_centre_panel_axes(nodes, le, te).axes
    angle = np.degrees(np.arccos(np.clip(axes[0] @ FRD_IN_STRUC[0], -1.0, 1.0)))
    assert angle == pytest.approx(tilt_deg, abs=1e-9)


def test_z_is_the_panel_normal_and_y_lies_in_the_panel():
    """Dihedral on the panel rolls the frame; z stays normal to the panel."""
    nodes, masses, le, te = _wing_cloud([3.0, 1.0, -1.0, -3.0])
    nodes[le[1]] += np.array([0.0, 0.0, 0.3])  # starboard strut up
    nodes[te[1]] += np.array([0.0, 0.0, 0.3])
    panel = compute_centre_panel_axes(nodes, le, te)
    span = (0.5 * (panel.le_starboard + panel.te_starboard)
            - 0.5 * (panel.le_port + panel.te_port))
    assert panel.axes[2] @ span == pytest.approx(0.0, abs=1e-12)
    assert panel.axes[2] @ (panel.le_centre - panel.te_centre) == pytest.approx(
        0.0, abs=1e-12
    )
    # y in the panel, tilted by the dihedral the strut offset introduced.
    assert panel.axes[1][2] > 1e-3


def test_centre_panel_is_bounded_by_the_two_innermost_struts():
    nodes, masses, le, te = _wing_cloud([3.0, 1.0, -1.0, -3.0])
    panel = compute_centre_panel_axes(nodes, le, te)
    np.testing.assert_allclose(panel.le_starboard, nodes[le[1]], atol=1e-12)
    np.testing.assert_allclose(panel.le_port, nodes[le[2]], atol=1e-12)
    np.testing.assert_allclose(
        panel.le_centre, 0.5 * (nodes[le[1]] + nodes[le[2]]), atol=1e-12
    )
    assert panel.chord_centre == pytest.approx(2.0)
    assert panel.span_centre == pytest.approx(2.0)


def test_a_strut_on_the_centreline_is_skipped():
    """Odd strut count: the pair straddling the centreline is used."""
    nodes, masses, le, te = _wing_cloud([3.0, 0.0, -3.0])
    panel = compute_centre_panel_axes(nodes, le, te)
    np.testing.assert_allclose(panel.le_starboard, nodes[le[0]], atol=1e-12)
    np.testing.assert_allclose(panel.le_port, nodes[le[2]], atol=1e-12)
    # Displacing the centreline strut leaves the frame untouched.
    base = panel.axes.copy()
    nodes[le[1]] += np.array([0.3, 0.0, 0.4])
    np.testing.assert_allclose(
        compute_centre_panel_axes(nodes, le, te).axes, base, atol=1e-14
    )


def test_inertia_is_reported_in_the_body_axes_without_diagonalising():
    nodes, masses, le, te = _wing_cloud([3.0, 1.0, -1.0, -3.0])
    rot = _rotation([0.0, 1.0, 0.0], 12.0)
    rb = compute_rigid_body_axes(nodes @ rot.T, masses, le, te)
    np.testing.assert_allclose(
        rb.inertia_body, rb.body_axes @ rb.inertia_cg @ rb.body_axes.T, atol=1e-10
    )
    np.testing.assert_allclose(rb.inertia_moments, np.diag(rb.inertia_body))
    np.testing.assert_allclose(rb.cg_body, rb.body_axes @ rb.cg)
    # A structural frame does not remove the products of inertia; the tensor in
    # body axes is not required to be diagonal (this cloud's is not).
    off_diagonal = rb.inertia_body - np.diag(np.diag(rb.inertia_body))
    assert np.abs(off_diagonal).max() > 1e-6


def test_frd_canonical_directions_are_right_handed():
    np.testing.assert_allclose(
        np.cross(FRD_IN_STRUC[0], FRD_IN_STRUC[1]), FRD_IN_STRUC[2]
    )
    # Down is opposite the course-frame radial (structural +z), right is
    # opposite the course-frame left (structural -y).
    assert FRD_IN_STRUC[2][2] < 0
    assert FRD_IN_STRUC[1][1] > 0


def test_wrong_frame_is_rejected():
    """A cloud mirrored into a non-FRD frame must not silently produce axes."""
    nodes, masses, le, te = _wing_cloud([3.0, 1.0, -1.0, -3.0])
    nodes[:, 0] *= -1.0  # leading edge now aft: body x would point backwards
    with pytest.raises(ValueError, match="FRD"):
        compute_rigid_body_axes(nodes, masses, le, te)


def test_single_sided_wing_is_rejected():
    nodes, masses, le, te = _wing_cloud([3.0, 1.0])
    with pytest.raises(ValueError, match="centre panel"):
        compute_centre_panel_axes(nodes, le, te)


def test_wing_le_te_indices_follows_the_psm_id_convention():
    struc_geometry = {
        "wing_particles": {
            "headers": ["id", "x", "y", "z"],
            "data": [
                [1, -1.0, 3.0, 10.0],
                [2, 1.0, 3.0, 10.0],
                [3, -1.0, -3.0, 10.0],
                [4, 1.0, -3.0, 10.0],
            ],
        }
    }
    le, te = wing_le_te_indices(struc_geometry)
    np.testing.assert_array_equal(le, [1, 3])
    np.testing.assert_array_equal(te, [2, 4])


def test_wing_le_te_indices_rejects_unpaired_nodes():
    struc_geometry = {
        "wing_particles": {
            "headers": ["id", "x", "y", "z"],
            "data": [[1, -1.0, 3.0, 10.0], [3, -1.0, -3.0, 10.0]],
        }
    }
    with pytest.raises(ValueError, match="one leading-edge"):
        wing_le_te_indices(struc_geometry)


# --------------------------------------------------------------------------
# geometry loading: one source for the cloud, the masses and the indices
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def kite_files():
    struc = KITE_DIR / "struc_geometry.yaml"
    system = KITE_DIR / "system.yaml"
    if not struc.exists() or not system.exists():  # pragma: no cover
        pytest.skip(f"{KITE_DIR} not available")
    return struc, system


def test_load_psm_geometry_matches_the_pss_initializer(kite_files):
    """The path helper must agree with the dict API plus the sibling system.yaml."""
    struc, system = kite_files
    struc_geometry = yaml.safe_load(struc.read_text(encoding="utf-8"))
    system_config = yaml.safe_load(system.read_text(encoding="utf-8"))

    nodes, masses, le, te = load_psm_geometry(struc)
    nodes_ref, masses_ref = load_psm_nodes_and_masses(
        struc_geometry, system_config=system_config
    )
    np.testing.assert_allclose(nodes, nodes_ref)
    np.testing.assert_allclose(masses, masses_ref)

    # The indices the initializer hands back are the ones wing_le_te_indices
    # reproduces from the id convention.
    le_ref, te_ref = wing_le_te_indices(struc_geometry)
    np.testing.assert_array_equal(le, le_ref)
    np.testing.assert_array_equal(te, te_ref)


def test_loaded_masses_carry_the_kcu_and_the_bridle(kite_files):
    """The lumping is the initializer's: wing tubes and canopy, bridle, KCU."""
    struc, system = kite_files
    struc_geometry = yaml.safe_load(struc.read_text(encoding="utf-8"))
    system_config = yaml.safe_load(system.read_text(encoding="utf-8"))
    kcu_mass = float(
        system_config["components"]["kites"][0]["control_system"]["structure"]["mass"]
    )

    _, masses, _, _ = load_psm_geometry(struc)
    mass_col = struc_geometry["wing_elements"]["headers"].index("m")
    wing_mass = sum(
        float(row[mass_col]) for row in struc_geometry["wing_elements"]["data"]
    )

    assert masses[0] >= kcu_mass  # KCU node also carries a bridle share
    assert masses.sum() > wing_mass + kcu_mass  # bridle lines are in there too
