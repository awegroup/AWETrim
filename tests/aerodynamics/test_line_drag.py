# Copyright (c) 2023-2026 Oriol Cayon, Delft University of Technology
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the flat-tape bridle-line drag model.

Pure numpy + the shipped LEI-V3 geometry: no VSM, no solver. They pin the
section coefficients, the identity that makes ``averaged`` the mean of the
component model rather than a fudge factor, the round-trip through the
drag-equivalent diameter, and -- most importantly -- the two invariants a
silent regression would break: that the structural diameter is never touched,
and that the per-segment diameters stay aligned with the bridle-line system
they are written into.
"""

from __future__ import annotations

import inspect
import logging
from pathlib import Path

import numpy as np
import pytest
import yaml

from awetrim.aerodynamics.line_drag import (
    CD_CABLE_ROUND,
    CD_TAPE_BROADSIDE,
    CD_TAPE_EDGE_ON,
    MEAN_ABS_COS,
    ROLL_AVERAGED,
    ROLL_BROADSIDE,
    ROLL_EDGE_ON,
    ROLL_MODELS,
    ROLL_ROUND,
    apply_bridle_drag_diameters,
    bridle_drag_diameters,
    cd_width_round,
    cd_width_tape,
    drag_diameter_tape,
    lines_missing_width,
    settings_from_config,
    tape_summary,
    tape_thickness_from_equivalent_diameter,
)

#: LEI-V3 steering/depower tape, as data/LEI-V3-KITE/struc_geometry.yaml holds it.
WIDTH = 0.012
THICKNESS = 0.0015
DIAMETER_STRUCTURAL = 0.004787307

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STRUC_GEOMETRY = PROJECT_ROOT / "data" / "LEI-V3-KITE" / "struc_geometry.yaml"


@pytest.fixture(scope="module")
def struc_geometry() -> dict:
    if not STRUC_GEOMETRY.exists():  # pragma: no cover - shipped with the repo
        pytest.skip(f"{STRUC_GEOMETRY} not available")
    return yaml.safe_load(STRUC_GEOMETRY.read_text(encoding="utf-8"))


class _Body:
    """Stand-in for a VSM ``BodyAerodynamics``: only the line list matters."""

    def __init__(self, diameters):
        self._bridle_line_system = [
            [np.zeros(3), np.array([0.0, 0.0, 1.0]), float(d)] for d in diameters
        ]

    @property
    def diameters(self):
        return [float(line[2]) for line in self._bridle_line_system]


# --------------------------------------------------------------- geometry
def test_thickness_inverts_the_area_equivalence():
    """The structural diameter and the width imply the tape thickness."""
    assert tape_thickness_from_equivalent_diameter(
        DIAMETER_STRUCTURAL, WIDTH
    ) == pytest.approx(THICKNESS, rel=1e-6)
    # ... and the equivalence it inverts is an AREA one, not a width one.
    assert np.pi * DIAMETER_STRUCTURAL**2 / 4.0 == pytest.approx(WIDTH * THICKNESS)


def test_thickness_rejects_a_degenerate_width():
    with pytest.raises(ValueError):
        tape_thickness_from_equivalent_diameter(DIAMETER_STRUCTURAL, 0.0)


# ----------------------------------------------------------- coefficients
def test_round_cable_is_the_unchanged_law():
    assert cd_width_round(0.003) == pytest.approx(CD_CABLE_ROUND * 0.003)
    assert cd_width_round(0.003, 1.2) == pytest.approx(1.2 * 0.003)


def test_tape_limits_are_the_tabulated_coefficients():
    assert cd_width_tape(WIDTH, THICKNESS, ROLL_BROADSIDE) == pytest.approx(
        CD_TAPE_BROADSIDE * WIDTH
    )
    assert cd_width_tape(WIDTH, THICKNESS, ROLL_EDGE_ON) == pytest.approx(
        CD_TAPE_EDGE_ON * THICKNESS
    )


def test_averaged_is_the_uniform_roll_mean_of_the_component_model():
    """``averaged`` is an integral, not a fudge factor.

    The closed form ``(2/pi)(cd_b w + cd_e t)`` must equal the numerical mean of
    ``cd_b w |cos psi| + cd_e t |sin psi|`` over a uniform roll angle. If the
    two ever disagree, the "averaged" label is lying about what it computes.
    """
    psi = np.linspace(0.0, 2.0 * np.pi, 200_001)
    integrand = CD_TAPE_BROADSIDE * WIDTH * np.abs(np.cos(psi)) + (
        CD_TAPE_EDGE_ON * THICKNESS * np.abs(np.sin(psi))
    )
    assert cd_width_tape(WIDTH, THICKNESS, ROLL_AVERAGED) == pytest.approx(
        float(np.trapezoid(integrand, psi) / (2.0 * np.pi)), rel=1e-6
    )
    assert MEAN_ABS_COS == pytest.approx(2.0 / np.pi)


def test_the_limits_bracket_the_default():
    edge = cd_width_tape(WIDTH, THICKNESS, ROLL_EDGE_ON)
    averaged = cd_width_tape(WIDTH, THICKNESS, ROLL_AVERAGED)
    broadside = cd_width_tape(WIDTH, THICKNESS, ROLL_BROADSIDE)
    assert edge < averaged < broadside
    # The spread is the headline uncertainty of the whole model (~18x).
    assert broadside / edge == pytest.approx(17.9, rel=5e-2)


def test_unknown_roll_model_raises():
    with pytest.raises(ValueError):
        cd_width_tape(WIDTH, THICKNESS, "sideways")
    with pytest.raises(ValueError):
        bridle_drag_diameters({}, "sideways")


@pytest.mark.parametrize(
    "width,thickness", [(0.0, 0.0015), (0.012, 0.0), (-0.012, 0.0015)]
)
def test_tape_rejects_degenerate_dimensions(width, thickness):
    with pytest.raises(ValueError):
        cd_width_tape(width, thickness)


# ------------------------------------------------- drag-equivalent diameter
@pytest.mark.parametrize("roll_model", [ROLL_AVERAGED, ROLL_BROADSIDE, ROLL_EDGE_ON])
def test_drag_diameter_round_trips_through_the_round_law(roll_model):
    """``cd_cable * d_drag`` must reproduce the tape's own ``cd * width``.

    That identity is the whole mechanism: the solver applies a fixed cd_cable
    to whatever diameter the line system carries.
    """
    d_drag = drag_diameter_tape(DIAMETER_STRUCTURAL, WIDTH, roll_model)
    assert CD_CABLE_ROUND * d_drag == pytest.approx(
        cd_width_tape(WIDTH, THICKNESS, roll_model)
    )


def test_drag_diameter_follows_a_non_default_cd_cable():
    d_drag = drag_diameter_tape(DIAMETER_STRUCTURAL, WIDTH, ROLL_AVERAGED, 2.2)
    assert 2.2 * d_drag == pytest.approx(cd_width_tape(WIDTH, THICKNESS, ROLL_AVERAGED))


def test_round_model_is_exactly_the_structural_diameter():
    """``round`` must be a bit-for-bit no-op -- it is the reproduction switch."""
    assert drag_diameter_tape(DIAMETER_STRUCTURAL, WIDTH, ROLL_ROUND) == (
        DIAMETER_STRUCTURAL
    )


def test_lei_v3_drag_diameters_are_pinned():
    """Regression pins for the shipped tape (12 x 1.5 mm)."""
    assert drag_diameter_tape(DIAMETER_STRUCTURAL, WIDTH, ROLL_AVERAGED) == (
        pytest.approx(0.013933, rel=1e-4)
    )
    assert drag_diameter_tape(DIAMETER_STRUCTURAL, WIDTH, ROLL_BROADSIDE) == (
        pytest.approx(0.020727, rel=1e-4)
    )
    assert drag_diameter_tape(DIAMETER_STRUCTURAL, WIDTH, ROLL_EDGE_ON) == (
        pytest.approx(0.001159, rel=1e-4)
    )
    # The headline number: the default is 2.91x the drag of the circle the
    # tapes used to be flown as.
    assert drag_diameter_tape(
        DIAMETER_STRUCTURAL, WIDTH, ROLL_AVERAGED
    ) / DIAMETER_STRUCTURAL == pytest.approx(2.91, rel=5e-3)


# ------------------------------------------------------- shipped geometry
def test_lei_v3_geometry_still_declares_the_tape(struc_geometry):
    """The data file must carry the width -- without it the model is a no-op."""
    rows = tape_summary(struc_geometry)
    assert {row["name"] for row in rows} == {"steering_tape", "depower_tape"}
    for row in rows:
        assert row["width_m"] == pytest.approx(WIDTH)
        assert row["thickness_m"] == pytest.approx(THICKNESS, rel=1e-6)
        assert row["diameter_structural_m"] == pytest.approx(DIAMETER_STRUCTURAL)


def test_structural_diameters_are_never_touched(struc_geometry):
    """Only the tapes' DRAG width moves; ``d`` stays the structural one.

    ``d`` feeds line mass and EA (pss/structural_geometry_io), so a change
    there would silently alter the structure, not just the aerodynamics.
    """
    before = [row[2] for row in struc_geometry["bridle_lines"]["data"]]
    bridle_drag_diameters(struc_geometry, ROLL_BROADSIDE)
    after = [row[2] for row in struc_geometry["bridle_lines"]["data"]]
    assert after == before
    assert all(d == pytest.approx(DIAMETER_STRUCTURAL) for d in after[-2:])


def test_only_the_tape_segments_change(struc_geometry):
    round_d = bridle_drag_diameters(struc_geometry, ROLL_ROUND)
    tape_d = bridle_drag_diameters(struc_geometry, ROLL_AVERAGED)
    assert len(round_d) == len(tape_d) == 45
    changed = [i for i, (a, b) in enumerate(zip(round_d, tape_d)) if a != b]
    # Two steering tapes + one depower tape.
    assert len(changed) == 3
    for i in changed:
        assert tape_d[i] / round_d[i] == pytest.approx(2.91, rel=5e-3)


def test_segment_order_matches_the_bridle_line_system(struc_geometry):
    """Alignment contract: one entry per segment, pulley rows expanded.

    ``apply_bridle_drag_diameters`` zips this list against the body's line
    system, so a drift in either walk would put a tape diameter on a bridle
    line. Reproduced here independently of the implementation.
    """
    expected = []
    for row in struc_geometry["bridle_connections"]["data"]:
        expected.append(row[0])
        if len(row) == 4:
            expected.append(row[0])
    diameters = bridle_drag_diameters(struc_geometry, ROLL_AVERAGED)
    assert len(diameters) == len(expected)
    tape_positions = {i for i, name in enumerate(expected) if "tape" in name}
    for i, diameter in enumerate(diameters):
        if i in tape_positions:
            assert diameter > DIAMETER_STRUCTURAL
        else:
            assert diameter <= 0.006


def test_geometry_without_bridles_is_empty():
    assert bridle_drag_diameters({}) == []
    assert tape_summary({}) == []


# --------------------------------------------------------------- applying
def test_apply_replaces_in_place_and_counts(struc_geometry):
    body = _Body(bridle_drag_diameters(struc_geometry, ROLL_ROUND))
    changed = apply_bridle_drag_diameters(body, struc_geometry, ROLL_AVERAGED)
    assert changed == 3
    assert body.diameters == bridle_drag_diameters(struc_geometry, ROLL_AVERAGED)


def test_apply_is_idempotent_and_recomputes_from_the_structure(struc_geometry):
    """Re-applying must not compound -- each call reads the structural ``d``."""
    body = _Body(bridle_drag_diameters(struc_geometry, ROLL_ROUND))
    apply_bridle_drag_diameters(body, struc_geometry, ROLL_AVERAGED)
    once = list(body.diameters)
    assert apply_bridle_drag_diameters(body, struc_geometry, ROLL_AVERAGED) == 0
    assert body.diameters == once
    # ... and switching model goes to the new value, not to a product of both.
    apply_bridle_drag_diameters(body, struc_geometry, ROLL_BROADSIDE)
    assert body.diameters == bridle_drag_diameters(struc_geometry, ROLL_BROADSIDE)
    # ... and `round` restores the untouched structural diameters exactly.
    apply_bridle_drag_diameters(body, struc_geometry, ROLL_ROUND)
    assert body.diameters == bridle_drag_diameters(struc_geometry, ROLL_ROUND)


def test_apply_to_a_body_without_bridles_is_a_no_op(struc_geometry):
    class Bare:
        pass

    assert apply_bridle_drag_diameters(Bare(), struc_geometry) == 0
    assert apply_bridle_drag_diameters(_Body([]), struc_geometry) == 0


def test_apply_raises_when_the_parses_disagree(struc_geometry):
    with pytest.raises(ValueError, match="drifted apart"):
        apply_bridle_drag_diameters(_Body([0.002] * 44), struc_geometry)


# ------------------------------------------------------------- provenance
def test_missing_width_is_detected_and_warned(struc_geometry, caplog):
    """A geometry written before the ``w`` column must not fail silently."""
    legacy = yaml.safe_load(yaml.dump(struc_geometry))
    legacy["bridle_lines"]["headers"] = [
        h for h in legacy["bridle_lines"]["headers"] if h != "w"
    ]
    for row in legacy["bridle_lines"]["data"]:
        del row[5:]
    assert sorted(lines_missing_width(legacy)) == ["depower_tape", "steering_tape"]
    with caplog.at_level(logging.WARNING, logger="awetrim.aerodynamics.line_drag"):
        diameters = bridle_drag_diameters(legacy, ROLL_AVERAGED)
    assert "steering_tape" in caplog.text
    # ... and the fallback is the old behaviour, not a guessed width.
    assert diameters == bridle_drag_diameters(legacy, ROLL_ROUND)


def test_no_warning_when_the_geometry_declares_its_tapes(struc_geometry, caplog):
    assert lines_missing_width(struc_geometry) == []
    with caplog.at_level(logging.WARNING, logger="awetrim.aerodynamics.line_drag"):
        bridle_drag_diameters(struc_geometry, ROLL_AVERAGED)
    assert caplog.text == ""


# ----------------------------------------------------------------- config
def test_settings_default_when_the_block_is_absent():
    assert settings_from_config(None) == (ROLL_AVERAGED, CD_CABLE_ROUND)
    assert settings_from_config({}) == (ROLL_AVERAGED, CD_CABLE_ROUND)
    assert settings_from_config({"aerodynamic_bridle": {}}) == (
        ROLL_AVERAGED,
        CD_CABLE_ROUND,
    )


def test_settings_read_the_as_config_keys():
    assert settings_from_config(
        {"aerodynamic_bridle": {"tape_roll_model": "broadside", "cd_cable": 1.2}}
    ) == (ROLL_BROADSIDE, 1.2)


def test_shipped_kite_configs_select_a_known_model():
    for kite in ("LEI-V3-KITE", "LEI-V9-KITE"):
        path = PROJECT_ROOT / "data" / kite / "as_config.yaml"
        if not path.exists():  # pragma: no cover - shipped with the repo
            continue
        roll_model, cd_cable = settings_from_config(
            yaml.safe_load(path.read_text(encoding="utf-8"))
        )
        assert roll_model in ROLL_MODELS
        assert cd_cable > 0.0


def test_public_signatures():
    assert list(inspect.signature(cd_width_tape).parameters) == [
        "width",
        "thickness",
        "roll_model",
    ]
    assert list(inspect.signature(drag_diameter_tape).parameters) == [
        "diameter_equivalent",
        "width",
        "roll_model",
        "cd_cable",
    ]
    assert list(inspect.signature(apply_bridle_drag_diameters).parameters) == [
        "body_aero",
        "struc_geometry",
        "roll_model",
        "cd_cable",
    ]
