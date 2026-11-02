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

"""The curled tip of an LEI wing as a lifting-line section.

At the tip the leading-edge tube curls down and back and ends at the trailing
edge: the last FEM bay (outer section -> tip section) is a side-facing fin
whose leading edge runs nearly along the chord. Meshed as it is, the bound
vortex of its panels lies almost along the flow (64-66 deg of sweep on the
LEI-V3), the 2-D section is far from normal to it and the stacked panels
induce strongly on each other's control points: VSM finds several
circulations per panel (attached, at the corner, stalled) and the trim cycles
between them (2026-11-02 study: 2 of 12 realistic trims converged).

:func:`correct_tip_bays` keeps the tube: the leading edge from the outer
section to the tip, plus the tip chord, has its built length. Only the tube's
corner (the tip section's leading-edge point) moves, along the curve of
constant tube length in the tube's own plane, until every panel of the bay is
swept by at most ``max_sweep_deg``. On the LEI-V3 that is a 0.80 m tip chord
instead of 0.35 m, at 0.164 instead of 0.167 m2 of bay area, and the tip
stays attached up to about 14 deg of wing incidence. The structure is not
touched: this is the aerodynamic mesh's geometry.

:func:`panel_sweep_deg` is the check: the angle of a panel's quarter-chord
line off the normal to its mean chord (0 = unswept, 90 = the bound vortex
along the chord). A bay of any kite still above the limit after the
correction is logged as a warning.
"""

from __future__ import annotations

import logging

import numpy as np

#: Largest panel sweep the lifting line is trusted with [deg]. The LEI-V3 FEM
#: tip bay (64-66) fails, the CAD tip panel (56) and the corrected tip (50-55)
#: trim; inner bays reach 41.
MAX_PANEL_SWEEP_DEG = 55.0
#: Panels per bay the sweep is checked on (the mesh's own count does not
#: matter: a bay's sweep varies with its chord direction, not its paneling).
_CHECK_PANELS = 10


def panel_sweep_deg(leading_edge: np.ndarray, trailing_edge: np.ndarray) -> np.ndarray:
    """Sweep of each panel between consecutive sections [deg]: the angle of the
    quarter-chord line off the normal to the panel's mean chord."""
    le = np.asarray(leading_edge, dtype=float)
    te = np.asarray(trailing_edge, dtype=float)
    quarter = le + 0.25 * (te - le)
    span = np.diff(quarter, axis=0)
    chord = 0.5 * ((te - le)[1:] + (te - le)[:-1])
    cos = np.abs(np.sum(span * chord, axis=1)) / (
        np.linalg.norm(span, axis=1) * np.linalg.norm(chord, axis=1)
    )
    return np.degrees(np.arcsin(np.clip(cos, 0.0, 1.0)))


def _bay_sweep(le0, te0, le1, te1) -> float:
    """Largest panel sweep of one bay paneled linearly."""
    t = np.linspace(0.0, 1.0, _CHECK_PANELS + 1)[:, None]
    return float(panel_sweep_deg((1 - t) * le0 + t * le1, (1 - t) * te0 + t * te1).max())


def _slide_corner(outer_le, outer_te, corner, tube_end, max_sweep_deg):
    """The corner on the constant-tube-length curve nearest its built place
    whose bay is swept at most ``max_sweep_deg`` (and whose tip chord grows);
    None when there is none."""
    a, b = outer_le, tube_end
    length = np.linalg.norm(corner - a) + np.linalg.norm(b - corner)
    centre, half = 0.5 * (a + b), 0.5 * np.linalg.norm(b - a)
    u = (b - a) / (2.0 * half)
    w = (corner - centre) - ((corner - centre) @ u) * u
    if np.linalg.norm(w) < 1e-9 * length:  # the tube is straight: no plane
        return None
    w /= np.linalg.norm(w)
    major = 0.5 * length
    minor = np.sqrt(max(major**2 - half**2, 0.0))
    phi0 = np.arctan2(((corner - centre) @ w) / minor, ((corner - centre) @ u) / major)
    chord0 = np.linalg.norm(b - corner)
    for step in np.linspace(0.0, np.pi, 2001)[1:]:
        for phi in (phi0 + step, phi0 - step):
            point = centre + major * np.cos(phi) * u + minor * np.sin(phi) * w
            if np.linalg.norm(b - point) <= chord0:
                continue
            if _bay_sweep(outer_le, outer_te, point, b) <= max_sweep_deg:
                return point
    return None


def correct_tip_bays(
    leading_edge: np.ndarray,
    trailing_edge: np.ndarray,
    max_sweep_deg: float = MAX_PANEL_SWEEP_DEG,
) -> np.ndarray:
    """The sections' leading edge with each tip bay swept at most
    ``max_sweep_deg`` (module docstring); a copy, the trailing edge and every
    other section unchanged. Logs each correction and warns about any bay
    still above the limit."""
    le = np.array(leading_edge, dtype=float)
    te = np.asarray(trailing_edge, dtype=float)
    for tip, outer in ((0, 1), (len(le) - 1, len(le) - 2)):
        before = _bay_sweep(le[outer], te[outer], le[tip], te[tip])
        if before <= max_sweep_deg:
            continue
        point = _slide_corner(le[outer], te[outer], le[tip], te[tip], max_sweep_deg)
        if point is None:
            continue  # reported by the check below
        logging.info(
            "tip bay sections %d-%d: corner moved %.3f m along the constant "
            "tube length, tip chord %.3f -> %.3f m, sweep %.0f -> %.0f deg",
            outer, tip, np.linalg.norm(point - le[tip]),
            np.linalg.norm(te[tip] - le[tip]), np.linalg.norm(te[tip] - point),
            before, _bay_sweep(le[outer], te[outer], point, te[tip]),
        )
        le[tip] = point
    too_swept = [
        (k, round(s))
        for k in range(len(le) - 1)
        if (s := _bay_sweep(le[k], te[k], le[k + 1], te[k + 1])) > max_sweep_deg
    ]
    if too_swept:
        logging.warning(
            "aerodynamic bays swept beyond %.0f deg (first section, deg): %s; "
            "the lifting line is not valid there",
            max_sweep_deg, too_swept,
        )
    return le
