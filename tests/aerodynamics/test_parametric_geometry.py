"""Tests for awetrim.aerodynamics.parametric_geometry.

These assert geometric structure, shapes, and morphing invariants; no VSM
installation is required (the module is pure numpy + yaml).
"""

import inspect

import numpy as np
import pytest

from awetrim.aerodynamics.parametric_geometry import (
    WING_SECTION_HEADERS,
    WingSections,
    morph_wing,
    morph_wing_to,
)


# ---------------------------------------------------------------------------
# Synthetic geometries
# ---------------------------------------------------------------------------


def _flat_rect_config(span=10.0, chord=1.0, n_half=3):
    """A flat rectangular wing, tip -> centre -> tip, chord along +x."""
    ys = np.concatenate(
        [np.linspace(span / 2, 0, n_half), np.linspace(0, -span / 2, n_half)[1:]]
    )
    data = []
    for i, y in enumerate(ys):
        data.append([i + 1, 0.0, float(y), 0.0, float(chord), float(y), 0.0])
    return {
        "wing_sections": {"headers": list(WING_SECTION_HEADERS), "data": data},
        "wing_airfoils": {
            "headers": ["airfoil_id", "type", "info_dict"],
            "data": [[1, "polars", {"csv_file_path": "polars/1.csv"}]],
        },
    }


def _arched_config(span=10.0, chord=1.0, droop=1.0, n_half=4):
    """A rectangular wing with parabolic tip droop (non-zero anhedral)."""
    ys = np.concatenate(
        [np.linspace(span / 2, 0, n_half), np.linspace(0, -span / 2, n_half)[1:]]
    )
    data = []
    for i, y in enumerate(ys):
        z = -droop * (y / (span / 2)) ** 2  # 0 at centre, -droop at tips
        data.append([i + 1, 0.0, float(y), float(z), float(chord), float(y), float(z)])
    return {"wing_sections": {"headers": list(WING_SECTION_HEADERS), "data": data}}


# ---------------------------------------------------------------------------
# Construction / serialisation
# ---------------------------------------------------------------------------


def test_from_aero_geometry_shapes():
    sec = WingSections.from_aero_geometry(_flat_rect_config())
    assert sec.le.shape == (sec.n_sections, 3)
    assert sec.te.shape == (sec.n_sections, 3)
    assert sec.airfoil_ids.shape == (sec.n_sections,)
    assert sec.chord_vectors.shape == (sec.n_sections, 3)
    assert sec.quarter_chord.shape == (sec.n_sections, 3)


def test_round_trip_aero_geometry():
    cfg = _flat_rect_config()
    sec = WingSections.from_aero_geometry(cfg)
    out = sec.to_aero_geometry()
    sec2 = WingSections.from_aero_geometry(out)
    np.testing.assert_allclose(sec.le, sec2.le)
    np.testing.assert_allclose(sec.te, sec2.te)
    np.testing.assert_array_equal(sec.airfoil_ids, sec2.airfoil_ids)
    # The wing_airfoils block is carried through unchanged.
    assert out["wing_airfoils"] == cfg["wing_airfoils"]


def test_to_yaml_round_trip(tmp_path):
    sec = WingSections.from_aero_geometry(_flat_rect_config())
    path = sec.to_yaml(tmp_path / "aero_geometry.yaml")
    assert path.exists()
    sec2 = WingSections.from_yaml(path)
    np.testing.assert_allclose(sec.quarter_chord, sec2.quarter_chord)


def test_to_yaml_resolves_csv_paths(tmp_path):
    sec = WingSections.from_aero_geometry(_flat_rect_config())
    out = sec.to_aero_geometry(resolve_csv_paths_relative_to=tmp_path)
    resolved = out["wing_airfoils"]["data"][0][2]["csv_file_path"]
    from pathlib import Path

    assert Path(resolved).is_absolute()
    assert resolved.endswith("1.csv")


def test_missing_header_raises():
    cfg = _flat_rect_config()
    cfg["wing_sections"]["headers"] = ["airfoil_id", "LE_x", "LE_y"]  # incomplete
    with pytest.raises(ValueError, match="missing columns"):
        WingSections.from_aero_geometry(cfg)


# ---------------------------------------------------------------------------
# Geometric properties
# ---------------------------------------------------------------------------


def test_flat_rect_properties():
    sec = WingSections.from_aero_geometry(_flat_rect_config(span=10.0, chord=1.0))
    assert sec.projected_span == pytest.approx(10.0)
    assert sec.flat_span == pytest.approx(10.0)
    assert sec.area == pytest.approx(10.0)
    assert sec.aspect_ratio == pytest.approx(10.0)
    assert sec.mean_chord == pytest.approx(1.0)
    assert sec.anhedral_angle_deg == pytest.approx(0.0, abs=1e-9)
    assert sec.is_symmetric()


def test_arched_wing_has_positive_anhedral():
    sec = WingSections.from_aero_geometry(_arched_config(droop=1.0))
    assert sec.anhedral_angle_deg > 0.0
    assert sec.flat_span > sec.projected_span  # arc longer than the y-extent


# ---------------------------------------------------------------------------
# Morphing
# ---------------------------------------------------------------------------


def test_morph_wing_area_preserving_aspect_ratio():
    sec = WingSections.from_aero_geometry(_flat_rect_config(span=10.0, chord=1.0))
    out = morph_wing(sec, span_scale=2.0, chord_scale=0.5)
    assert out.projected_span == pytest.approx(20.0)
    assert out.area == pytest.approx(sec.area)  # area preserved
    assert out.aspect_ratio == pytest.approx(4.0 * sec.aspect_ratio)
    assert out.anhedral_angle_deg == pytest.approx(0.0, abs=1e-9)


def test_morph_wing_to_target_aspect_ratio():
    sec = WingSections.from_aero_geometry(_flat_rect_config())
    out = morph_wing_to(sec, target_aspect_ratio=20.0)
    assert out.aspect_ratio == pytest.approx(20.0, rel=1e-4)
    assert out.area == pytest.approx(sec.area, rel=1e-6)  # area preserved by default


def test_morph_wing_to_target_anhedral():
    sec = WingSections.from_aero_geometry(_arched_config(droop=1.0))
    out = morph_wing_to(sec, target_anhedral_deg=25.0)
    assert out.anhedral_angle_deg == pytest.approx(25.0, abs=1e-2)


def test_morph_wing_to_combined_targets():
    sec = WingSections.from_aero_geometry(_arched_config(droop=1.0))
    out = morph_wing_to(sec, target_aspect_ratio=12.0, target_anhedral_deg=20.0)
    assert out.aspect_ratio == pytest.approx(12.0, rel=1e-3)
    assert out.anhedral_angle_deg == pytest.approx(20.0, abs=5e-2)


def test_taper_changes_ratio_preserves_area_and_ar():
    sec = WingSections.from_aero_geometry(_flat_rect_config())
    assert sec.taper_ratio == pytest.approx(1.0)  # rectangular baseline
    out = morph_wing(sec, taper_ratio=1.5)
    assert out.taper_ratio == pytest.approx(1.5, rel=1e-6)
    assert out.area == pytest.approx(sec.area, rel=1e-9)  # area-preserving
    assert out.aspect_ratio == pytest.approx(sec.aspect_ratio, rel=1e-9)
    assert out.anhedral_angle_deg == pytest.approx(sec.anhedral_angle_deg, abs=1e-9)


def test_twist_changes_tip_twist_only():
    sec = WingSections.from_aero_geometry(_flat_rect_config())
    assert sec.tip_twist_deg == pytest.approx(0.0, abs=1e-9)
    out = morph_wing(sec, twist_deg=5.0)
    assert out.tip_twist_deg == pytest.approx(5.0, abs=1e-6)
    # Rotation preserves chord length -> area, AR, anhedral unchanged.
    np.testing.assert_allclose(out.chords, sec.chords, atol=1e-9)
    assert out.area == pytest.approx(sec.area, rel=1e-9)
    assert out.aspect_ratio == pytest.approx(sec.aspect_ratio, rel=1e-9)


def test_morph_wing_to_taper_only():
    sec = WingSections.from_aero_geometry(_arched_config(droop=1.0))
    out = morph_wing_to(sec, taper_ratio=1.3)
    assert out.taper_ratio == pytest.approx(1.3 * sec.taper_ratio, rel=1e-6)
    assert out.aspect_ratio == pytest.approx(sec.aspect_ratio, rel=1e-9)


def test_morph_wing_to_twist_only():
    sec = WingSections.from_aero_geometry(_arched_config(droop=1.0))
    out = morph_wing_to(sec, twist_deg=4.0)
    assert out.tip_twist_deg == pytest.approx(sec.tip_twist_deg + 4.0, abs=1e-5)


def test_morph_wing_to_all_targets():
    sec = WingSections.from_aero_geometry(_arched_config(droop=1.0))
    out = morph_wing_to(
        sec,
        target_aspect_ratio=12.0,
        target_anhedral_deg=20.0,
        taper_ratio=1.2,
        twist_deg=3.0,
    )
    assert out.aspect_ratio == pytest.approx(12.0, rel=1e-3)
    assert out.anhedral_angle_deg == pytest.approx(20.0, abs=5e-2)
    assert out.tip_twist_deg == pytest.approx(sec.tip_twist_deg + 3.0, abs=1e-2)


def test_taper_twist_preserve_symmetry():
    sec = WingSections.from_aero_geometry(_arched_config(droop=1.0))
    out = morph_wing(sec, taper_ratio=1.4, twist_deg=5.0)
    assert out.is_symmetric()


def test_morph_preserves_symmetry():
    sec = WingSections.from_aero_geometry(_arched_config(droop=1.0))
    out = morph_wing(sec, span_scale=1.3, chord_scale=0.9, anhedral_scale=1.4)
    assert out.is_symmetric()


def test_morph_anhedral_from_flat_raises():
    sec = WingSections.from_aero_geometry(_flat_rect_config())
    with pytest.raises(ValueError, match="flat wing"):
        morph_wing_to(sec, target_anhedral_deg=10.0)


def test_morph_wing_to_requires_a_target():
    sec = WingSections.from_aero_geometry(_flat_rect_config())
    with pytest.raises(ValueError, match="target"):
        morph_wing_to(sec)


# ---------------------------------------------------------------------------
# Public signatures
# ---------------------------------------------------------------------------


def test_public_signatures():
    morph_sig = inspect.signature(morph_wing)
    assert list(morph_sig.parameters) == [
        "sections", "span_scale", "chord_scale", "anhedral_scale",
        "taper_ratio", "twist_deg",
    ]
    to_sig = inspect.signature(morph_wing_to)
    assert list(to_sig.parameters) == [
        "sections", "target_aspect_ratio", "target_anhedral_deg",
        "taper_ratio", "twist_deg", "preserve_area", "max_iter", "tol",
    ]
