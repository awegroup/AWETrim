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

"""Section drag of bridle lines: round cable vs FLAT TAPE.

Why this module exists
----------------------
The steering and depower "lines" of an LEI kite are not round Dyneema; they
are flat webbing (LEI-V3: 12 x 1.5 mm). ``struc_geometry.yaml`` stores them
with an AREA-EQUIVALENT round diameter --

    d_eq = sqrt(4 w t / pi)                       (LEI-V3: 4.787307 mm)

-- which is exactly right for what it was introduced for, the structural
cross-section: line mass ``rho A l0`` and stiffness ``E A / l0`` both depend on
the area alone (``pss/structural_geometry_io.py``). But the same ``d`` is then
handed to the bridle DRAG law as the reference width, and an area-equivalent
circle is the one equivalence that carries no information about drag: a bluff
body's drag scales with its PROJECTED WIDTH, not with its cross-sectional area.
A 12 x 1.5 mm tape presents up to 12 mm to the flow, 2.5x the 4.79 mm circle
that replaces it.

The drag law itself is unchanged. Both VSM
(``BodyAerodynamics.compute_line_aerodynamic_force``) and its AWETrim mirror
(``aerostructural/aerodynamic_bridle_line_drag.py``) apply Hoerner's crossflow
("independence") principle to a line of length ``L`` at angle ``theta`` to the
apparent wind::

    F = q L d [ (cd_cable sin^3 t + pi cf_cable cos^3 t) e_D
              + (cd_cable sin^2 t cos t - pi cf_cable sin t cos^2 t) e_L ]

That form survives for a sharp-edged section -- separation stays pinned to the
edges at yaw -- so the ONLY thing that has to change for a tape is the product
``cd_cable * d``, i.e. the drag width per unit length. This module is the single
source of that quantity, for both section shapes.

How it reaches the solver
-------------------------
``cd_cable`` is baked into VSM's line-force call, and the line system carries
one number per segment (``[p1, p2, d]``). So the correction is applied as a
DRAG-EQUIVALENT DIAMETER substituted into that slot:

    d_drag = cd_width(section) / cd_cable          (LEI-V3 tape: 13.93 mm)

chosen so ``cd_cable * d_drag`` reproduces the tape's true ``cd * width``.
``d_drag`` is a drag quantity ONLY -- it must never reach the structural
tables, where the area-equivalent ``d`` remains correct. Nothing in this module
touches mass or stiffness.

Section coefficients (Hoerner, *Fluid-Dynamic Drag*)
----------------------------------------------------
Round cable, subcritical Re ~ 1e4 (a 2-6 mm line at 25 m/s): ``cd = 1.1``, the
long-standing VSM/as_config default, kept here as the reference.

Flat tape, two limiting roll orientations about the line's own axis:

============  ==============  ==================  ====
orientation   frontal dim.    streamwise/frontal  cd
============  ==============  ==================  ====
broadside     w (12 mm)       0.125               1.9
edge-on       t (1.5 mm)      8                   0.85
============  ==============  ==================  ====

``broadside`` is the 2D sharp-edged plate normal to the flow (cd -> 1.98 as the
aspect ratio grows; a tape segment is L/w ~ 130, so effectively 2D -- 1.9 is
the conservative end of the 1.9-2.0 band). ``edge-on`` reads the rectangular-
cylinder curve at a streamwise/frontal ratio of 8, past reattachment, where cd
has flattened out near 0.85. Unlike the round cable, a sharp-edged section is
Reynolds-independent, so neither number drifts with flight speed.

The dominant uncertainty is not the coefficient -- it is that those two
orientations differ by **18x** and the tape's roll angle ``psi`` is not in the
geometry (it is set by how the webbing leaves the KCU sheave, and it varies
along the line). There is therefore no single defensible value, only a default
plus a stated bracket. The default composes the two limits,

    cd_width(psi) = cd_broadside w |cos psi| + cd_edge_on t |sin psi|

which is exact at both ends, and averages it over a uniform roll angle
(``<|cos|> = <|sin|> = 2/pi``)::

    ROLL_AVERAGED   (2/pi)(cd_b w + cd_e t)  = 15.33 mm   2.91x the round-equivalent
    ROLL_BROADSIDE  cd_b w                   = 22.80 mm   4.33x
    ROLL_EDGE_ON    cd_e t                   =  1.28 mm   0.24x
    ROLL_ROUND      cd_cable d_eq            =  5.27 mm   1.00x  (pre-2026-08-21)

Use ``ROLL_BROADSIDE``/``ROLL_EDGE_ON`` as the sensitivity bracket and
``ROLL_ROUND`` to reproduce a result from before this module existed.

Two known biases, both pushing the same way (the default is a FLOOR, not a
centred estimate): taut webbing flutters, which measurably raises its drag
above any steady-section value, and the tapes run in the KCU's wake over part
of the flight envelope. Neither is modelled. A third, negligible one: scaling
``d`` scales the skin-friction term ``pi cf cos^3 t`` along with the pressure
term, though friction should follow the wetted perimeter instead -- at the
tapes' flow angles (theta ~ 70 deg) that term is under 0.5 % of ``cd_t``, so
the induced error is in the fourth decimal.
"""

from __future__ import annotations

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "CD_CABLE_ROUND",
    "CD_TAPE_BROADSIDE",
    "CD_TAPE_EDGE_ON",
    "ROLL_AVERAGED",
    "ROLL_BROADSIDE",
    "ROLL_EDGE_ON",
    "ROLL_ROUND",
    "ROLL_MODELS",
    "apply_bridle_drag_diameters",
    "bridle_drag_diameters",
    "cd_width_round",
    "lines_missing_width",
    "settings_from_config",
    "cd_width_tape",
    "drag_diameter_tape",
    "tape_summary",
    "tape_thickness_from_equivalent_diameter",
]

#: Circular cylinder in subcritical crossflow (Re ~ 1e4). Matches the VSM
#: default and ``as_config.yaml`` ``aerodynamic_bridle.cd_cable``; the drag
#: -equivalent diameter is defined against THIS value, so a config that
#: overrides cd_cable must pass it here too.
CD_CABLE_ROUND = 1.1

#: Sharp-edged flat plate normal to the flow, 2D limit (on the width).
CD_TAPE_BROADSIDE = 1.9
#: Rectangular section, streamwise/frontal ratio ~ 8, past reattachment
#: (on the thickness).
CD_TAPE_EDGE_ON = 0.85

#: Roll-angle treatments (see the module docstring).
ROLL_AVERAGED = "averaged"
ROLL_BROADSIDE = "broadside"
ROLL_EDGE_ON = "edge_on"
ROLL_ROUND = "round"
ROLL_MODELS = (ROLL_AVERAGED, ROLL_BROADSIDE, ROLL_EDGE_ON, ROLL_ROUND)

#: Mean of |cos psi| (and of |sin psi|) over a uniformly distributed roll angle.
MEAN_ABS_COS = 2.0 / math.pi


def settings_from_config(config: dict | None) -> tuple[str, float]:
    """``(roll_model, cd_cable)`` from an ``as_config`` block.

    Single-sources the two YAML key names, so every builder that
    instantiates a bridled body reads them the same way::

        aerodynamic_bridle:
          cd_cable: 1.1
          tape_roll_model: averaged   # averaged|broadside|edge_on|round

    A missing block gives the defaults, which is what keeps kites whose
    geometry declares no tape width unaffected.
    """
    block = (config or {}).get("aerodynamic_bridle") or {}
    return (
        str(block.get("tape_roll_model", ROLL_AVERAGED)),
        float(block.get("cd_cable", CD_CABLE_ROUND)),
    )


def tape_thickness_from_equivalent_diameter(
    diameter_equivalent: float, width: float
) -> float:
    """Tape thickness implied by its area-equivalent round diameter.

    Inverts ``d_eq = sqrt(4 w t / pi)``, the convention ``struc_geometry.yaml``
    already uses for the tapes, so the thickness never has to be stated twice:
    the structural ``d`` and the tape width ``w`` fix it.
    """
    width = float(width)
    if width <= 0.0:
        raise ValueError("tape width must be positive")
    return math.pi * float(diameter_equivalent) ** 2 / (4.0 * width)


def cd_width_round(diameter: float, cd_cable: float = CD_CABLE_ROUND) -> float:
    """``cd * d`` [m] of a round cable -- drag width per unit length."""
    return float(cd_cable) * float(diameter)


def cd_width_tape(
    width: float, thickness: float, roll_model: str = ROLL_AVERAGED
) -> float:
    """``cd * w_eff`` [m] of a flat tape -- drag width per unit length.

    ``roll_model`` selects how the unknown roll angle about the line's own axis
    is treated: ``averaged`` (default) composes the broadside and edge-on
    limits and averages over a uniform roll angle; ``broadside`` and ``edge_on``
    are those limits themselves, i.e. the sensitivity bracket.
    """
    width = float(width)
    thickness = float(thickness)
    if width <= 0.0 or thickness <= 0.0:
        raise ValueError("tape width and thickness must be positive")
    broadside = CD_TAPE_BROADSIDE * width
    edge_on = CD_TAPE_EDGE_ON * thickness
    if roll_model == ROLL_BROADSIDE:
        return broadside
    if roll_model == ROLL_EDGE_ON:
        return edge_on
    if roll_model == ROLL_AVERAGED:
        return MEAN_ABS_COS * (broadside + edge_on)
    raise ValueError(
        f"unknown tape roll model {roll_model!r}; expected one of {ROLL_MODELS}"
    )


def drag_diameter_tape(
    diameter_equivalent: float,
    width: float,
    roll_model: str = ROLL_AVERAGED,
    cd_cable: float = CD_CABLE_ROUND,
) -> float:
    """Diameter that makes the ROUND drag law reproduce the tape's drag.

    The bridle-line force law is applied with a fixed ``cd_cable`` and one
    diameter per segment, so the tape enters as ``d_drag = cd_width / cd_cable``
    (LEI-V3, averaged: 13.93 mm against a 4.79 mm structural diameter). Drag
    only -- see the module docstring.
    """
    if roll_model == ROLL_ROUND:
        return float(diameter_equivalent)
    thickness = tape_thickness_from_equivalent_diameter(diameter_equivalent, width)
    return cd_width_tape(width, thickness, roll_model) / float(cd_cable)


def _drag_diameter_of_row(
    row: dict, roll_model: str, cd_cable: float
) -> tuple[float, bool]:
    """``(diameter, is_tape)`` for one ``bridle_lines`` row.

    A row is a flat tape when it carries a positive optional ``w`` (tape width
    [m]); every other row keeps its structural diameter and the round law.
    """
    diameter = float(row["d"])
    width = row.get("w")
    if width in (None, "", 0) or float(width) <= 0.0:
        return diameter, False
    return (
        drag_diameter_tape(diameter, float(width), roll_model, cd_cable),
        roll_model != ROLL_ROUND,
    )


def lines_missing_width(struc_geometry: dict) -> list[str]:
    """``bridle_lines`` rows that look like a tape but declare no ``w``.

    A name-based smell test, and deliberately only that: a geometry file
    written before the ``w`` column existed still names its webbing
    ``steering_tape``/``Power Tape``, and would otherwise be flown as a
    round cable of the area-equivalent diameter WITHOUT any sign in the
    logs. Deformed snapshots are the usual offender -- they are copies of
    an older table. Fixing one means adding ``w`` to the geometry, never
    guessing a width here.
    """
    table = struc_geometry.get("bridle_lines")
    if not table:
        return []
    headers = table["headers"][1:]
    missing = []
    for row in table["data"]:
        spec = dict(zip(headers, row[1:]))
        width = spec.get("w")
        if "tape" in str(row[0]).lower() and (
            width in (None, "", 0) or float(width) <= 0.0
        ):
            missing.append(str(row[0]))
    return missing


def bridle_drag_diameters(
    struc_geometry: dict,
    roll_model: str = ROLL_AVERAGED,
    cd_cable: float = CD_CABLE_ROUND,
) -> list[float]:
    """Per-SEGMENT drag diameters, in the canonical bridle-line-system order.

    Walks ``bridle_connections`` exactly as ``BodyAerodynamics.instantiate``
    and ``aerostructural.aerodynamic_vsm.parse_bridle_line_specs`` do -- one
    segment per row, plus a second one for a 3-node pulley row -- so the
    returned list aligns element-by-element with ``_bridle_line_system``.
    Rows without a ``w`` column return their structural diameter unchanged.
    """
    if roll_model not in ROLL_MODELS:
        raise ValueError(
            f"unknown tape roll model {roll_model!r}; expected one of {ROLL_MODELS}"
        )
    if (
        "bridle_connections" not in struc_geometry
        or "bridle_lines" not in struc_geometry
    ):
        return []
    missing = lines_missing_width(struc_geometry)
    if missing and roll_model != ROLL_ROUND:
        logger.warning(
            "bridle line(s) %s look like flat tape but declare no width "
            '"w" -- they keep the round drag law of their area-equivalent '
            "diameter. Add w [m] to the bridle_lines table.",
            ", ".join(missing),
        )
    headers = struc_geometry["bridle_lines"]["headers"][1:]
    lines = {
        row[0]: dict(zip(headers, row[1:]))
        for row in struc_geometry["bridle_lines"]["data"]
    }
    diameters: list[float] = []
    for row in struc_geometry["bridle_connections"]["data"]:
        diameter, _ = _drag_diameter_of_row(lines[row[0]], roll_model, cd_cable)
        diameters.append(diameter)
        if len(row) == 4:
            diameters.append(diameter)
    return diameters


def apply_bridle_drag_diameters(
    body_aero: Any,
    struc_geometry: dict,
    roll_model: str = ROLL_AVERAGED,
    cd_cable: float = CD_CABLE_ROUND,
) -> int:
    """Swap the drag diameters into a body's bridle-line system in place.

    Call it once, right after the body is instantiated with bridles. Returns
    the number of segments whose diameter changed (0 when the geometry has no
    tapes, or under ``ROLL_ROUND``), so callers can report the correction.
    Raises if the segment count disagrees with the geometry -- that would mean
    the two parses have drifted apart and the diameters would land on the wrong
    lines.
    """
    line_system = getattr(body_aero, "_bridle_line_system", None)
    if not line_system:
        return 0
    diameters = bridle_drag_diameters(struc_geometry, roll_model, cd_cable)
    if not diameters:
        return 0
    if len(diameters) != len(line_system):
        raise ValueError(
            f"bridle-line system has {len(line_system)} segments but the "
            f"structural geometry yields {len(diameters)}; the connection "
            "parses have drifted apart"
        )
    changed = 0
    for line, diameter in zip(line_system, diameters):
        if abs(float(line[2]) - diameter) > 1e-12:
            changed += 1
        line[2] = diameter
    return changed


def tape_summary(
    struc_geometry: dict,
    roll_model: str = ROLL_AVERAGED,
    cd_cable: float = CD_CABLE_ROUND,
) -> list[dict]:
    """One row per tape in ``bridle_lines`` -- for logging and provenance."""
    if "bridle_lines" not in struc_geometry:
        return []
    headers = struc_geometry["bridle_lines"]["headers"][1:]
    rows = []
    for row in struc_geometry["bridle_lines"]["data"]:
        spec = dict(zip(headers, row[1:]))
        width = spec.get("w")
        if width in (None, "", 0) or float(width) <= 0.0:
            continue
        diameter = float(spec["d"])
        drag = drag_diameter_tape(diameter, float(width), roll_model, cd_cable)
        rows.append(
            {
                "name": row[0],
                "width_m": float(width),
                "thickness_m": tape_thickness_from_equivalent_diameter(
                    diameter, float(width)
                ),
                "diameter_structural_m": diameter,
                "diameter_drag_m": drag,
                "drag_ratio": drag / diameter,
                "roll_model": roll_model,
            }
        )
    return rows
