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

"""Canopy billowing of an LEI wing as a geometric correction of its aero mesh.

A structure without a canopy between its struts (the wireframe) flies a wing
whose trailing edge is straight from strut to strut. In flight the canopy
billows: an inextensible membrane under a locally constant pressure jump
carries a constant tension, so between two struts it is a CIRCULAR ARC whose
radius does not depend on the load (Cayon 2022, MSc thesis, Sec. 5.3). The
billowing changes only through the straight distance c between the strut
trailing edges, with the fabric length L_f fixed:

    sin(theta) / theta = c / L_f,   R = c / (2 sin(theta)),
    depth at mid-bay   = (c / 2) tan(theta / 2).

The measured contraction of c is the input: the LEI-V3 photogrammetry gives
-5.23 % (powered straight), -5.00 % (powered turn) and -3.86 % (depowered)
of the CAD strut-to-strut trailing-edge distance (Haanen 2026, MSc thesis,
Table 9.2), and Billow's full model reproduces -5 to -8 % on the same wing.

:func:`billow_wing` applies a uniform contraction ``p`` to a wing given by its
sections (leading/trailing-edge points, span order) and the section indices
of its struts:

* every strut rotates about its leading-edge node, about the local wing
  normal (strut length and leading edge kept), marching from the centre bay
  outwards, until each inner bay's straight trailing edge is (1 - p) of the
  built one;
* the trailing-edge points between two struts go on the circular arc of the
  bay's built fabric length (the trailing-edge polyline), each at its own
  fraction of that length, bulging the way the built bay already bulges;
* the two tip bays (outer strut to the leading-edge tip piece) cannot contract
  by rotating the short tip chord: they ride rigidly with the outer strut and
  keep their built shape. The photogrammetry also sees the smallest change
  there.

:func:`refine_billowed_wing` subdivides the result into aerodynamic sections
on the arc. ``p = 0`` returns the built wing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq

#: Strut rotations are searched within +-this angle [rad].
_MAX_STRUT_ROTATION = 0.6


def rotate_about_axis(vectors: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """``vectors`` (..., 3) rotated by ``angle`` about the unit ``axis`` (Rodrigues)."""
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    v = np.asarray(vectors, dtype=float)
    return (
        v * np.cos(angle)
        + np.cross(axis, v) * np.sin(angle)
        + np.multiply.outer(v @ axis, axis) * (1.0 - np.cos(angle))
    )


def arc_half_angle(chord_over_arc: float) -> float:
    """Half-angle theta of the circular arc with ``sin(theta)/theta = c / L_f``."""
    if chord_over_arc >= 1.0 - 1e-12:
        return 0.0
    return brentq(lambda t: np.sin(t) / t - chord_over_arc, 1e-9, np.pi - 1e-9)


@dataclass
class BillowedBay:
    """One bay between two struts (section indices ``first`` .. ``last``)."""

    first: int
    last: int
    fabric_length: float
    #: Each section's fraction of the fabric length, 0 at ``first``.
    fabric_fraction: np.ndarray
    chord_built: float
    chord: float
    theta: float
    #: (mid, chord direction, bulge direction, radius) of the arc; None for a
    #: bay that keeps its built shape (the tip bays, or p = 0 on a flat bay).
    arc: tuple | None = None

    @property
    def billowing(self) -> float:
        """1 - c / L_f."""
        return 1.0 - self.chord / self.fabric_length

    @property
    def depth(self) -> float:
        return 0.5 * self.chord * np.tan(0.5 * self.theta)

    def arc_point(self, s: float) -> np.ndarray:
        """Point on the arc at fabric fraction ``s``."""
        mid, u, d, radius = self.arc
        phi = self.theta * (2.0 * s - 1.0)
        return mid + radius * np.sin(phi) * u + radius * (np.cos(phi) - np.cos(self.theta)) * d


@dataclass
class BillowedWing:
    leading_edge: np.ndarray
    trailing_edge: np.ndarray
    contraction: float
    struts: list[int]
    bays: list[BillowedBay]
    #: Rigid motion of every section: section index -> (pivot, axis, angle).
    #: Rib nodes of a structure on these sections move with them.
    section_motion: dict[int, tuple] = field(default_factory=dict)

    def move_with_section(self, section: int, points: np.ndarray) -> np.ndarray:
        """``points`` carried by the rigid motion of ``section``."""
        pivot, axis, angle = self.section_motion[section]
        return pivot + rotate_about_axis(np.asarray(points, dtype=float) - pivot, axis, angle)


def billow_wing(
    leading_edge: np.ndarray,
    trailing_edge: np.ndarray,
    strut_sections,
    contraction: float,
) -> BillowedWing:
    """The wing with a uniform strut-to-strut trailing-edge contraction.

    ``leading_edge`` / ``trailing_edge``: (n, 3) section points in span order;
    ``strut_sections``: the section indices of the struts, the two tip pieces
    included (first and last). ``contraction``: p as a fraction (0.05 = 5 %).
    Raises when a strut cannot reach its bay length within
    +-``_MAX_STRUT_ROTATION``.
    """
    le = np.asarray(leading_edge, dtype=float)
    te = np.asarray(trailing_edge, dtype=float)
    p = float(contraction)
    if not 0.0 <= p < 1.0:
        raise ValueError(f"contraction must be in [0, 1), got {p}")
    struts = sorted(int(k) for k in strut_sections)
    if len(struts) < 4 or struts[0] != 0 or struts[-1] != len(le) - 1:
        raise ValueError(
            "strut_sections must include both tip pieces and at least two inner struts"
        )
    n = len(le)

    def normal(k):
        """Wing normal at section k (chord x leading-edge tangent), away from the origin."""
        tangent = le[min(k + 1, n - 1)] - le[max(k - 1, 0)]
        nrm = np.cross(te[k] - le[k], tangent)
        nrm /= np.linalg.norm(nrm)
        return nrm if nrm @ le[k] > 0 else -nrm

    def strut_te(k, angle):
        return le[k] + rotate_about_axis(te[k] - le[k], normal(k), angle)

    def gentlest_root(f):
        """Root of f with the smallest |angle|."""
        if p == 0.0:
            return 0.0
        xs = np.linspace(-_MAX_STRUT_ROTATION, _MAX_STRUT_ROTATION, 481)
        vals = np.array([f(x) for x in xs])
        roots = [
            brentq(f, xs[i], xs[i + 1])
            for i in range(len(xs) - 1)
            if np.sign(vals[i]) != np.sign(vals[i + 1])
        ]
        if not roots:
            raise ValueError(
                f"no strut rotation within +-{_MAX_STRUT_ROTATION} rad reaches a "
                f"contraction of {p:.3f}"
            )
        return min(roots, key=abs)

    bays = list(zip(struts[:-1], struts[1:]))
    inner = bays[1:-1]
    angle: dict[int, float] = {}
    te_new = te.copy()
    le_new = le.copy()

    # centre bay: the pair of struts mirror-symmetric about y = 0, rotated by
    # equal and opposite angles about mirror-image normals
    ci = min(range(len(inner)), key=lambda i: abs(le[inner[i][0], 1] + le[inner[i][1], 1]))
    a, b = inner[ci]
    target = (1.0 - p) * np.linalg.norm(te[b] - te[a])
    phi = gentlest_root(lambda x: np.linalg.norm(strut_te(b, x) - strut_te(a, -x)) - target)
    angle[a], angle[b] = -phi, phi
    te_new[a], te_new[b] = strut_te(a, -phi), strut_te(b, phi)

    def march(pairs):
        for fixed, free in pairs:
            tgt = (1.0 - p) * np.linalg.norm(te[free] - te[fixed])
            x = gentlest_root(lambda x: np.linalg.norm(strut_te(free, x) - te_new[fixed]) - tgt)
            angle[free] = x
            te_new[free] = strut_te(free, x)

    march([(inner[i][0], inner[i][1]) for i in range(ci + 1, len(inner))])
    march([(inner[i][1], inner[i][0]) for i in range(ci - 1, -1, -1)])

    section_motion = {}
    for k in struts[1:-1]:
        section_motion[k] = (le[k].copy(), normal(k), angle[k])
    # tip bays ride rigidly with their outer strut
    for (first, last), strut in ((bays[0], bays[0][1]), (bays[-1], bays[-1][0])):
        for k in range(first, last + 1):
            if k != strut:
                section_motion[k] = section_motion[strut]
                le_new[k] = le[strut] + rotate_about_axis(le[k] - le[strut], normal(strut), angle[strut])
                te_new[k] = le[strut] + rotate_about_axis(te[k] - le[strut], normal(strut), angle[strut])

    out = []
    for first, last in bays:
        idx = np.arange(first, last + 1)
        built = te[idx]
        seg = np.linalg.norm(np.diff(built, axis=0), axis=1)
        fabric = float(seg.sum())
        frac = np.concatenate([[0.0], np.cumsum(seg)]) / fabric
        c = float(np.linalg.norm(te_new[last] - te_new[first]))
        theta = arc_half_angle(c / fabric)
        bay = BillowedBay(
            first=first, last=last, fabric_length=fabric, fabric_fraction=frac,
            chord_built=float(np.linalg.norm(te[last] - te[first])), chord=c, theta=theta,
        )
        if (first, last) in inner and theta > 0.0:
            u = (te_new[last] - te_new[first]) / c
            # bulge direction: the built bay's own, normal to the new chord
            u0 = (built[-1] - built[0]) / np.linalg.norm(built[-1] - built[0])
            off = built - built[0]
            perp = off - np.outer(off @ u0, u0)
            d = perp[int(np.argmax(np.linalg.norm(perp, axis=1)))]
            d = d - (d @ u) * u
            d /= np.linalg.norm(d)
            bay.arc = (0.5 * (te_new[first] + te_new[last]), u, d, c / (2.0 * np.sin(theta)))
            for j, k in enumerate(idx[1:-1], start=1):
                te_new[k] = bay.arc_point(frac[j])
        out.append(bay)
    return BillowedWing(le_new, te_new, p, struts, out, section_motion)


def refine_billowed_wing(wing: BillowedWing, n_per_section: int) -> tuple[np.ndarray, np.ndarray]:
    """Aerodynamic edges with ``n_per_section`` panels per wing section.

    Leading edge: linear between the section points. Trailing edge: on the
    bay's arc at the interpolated fabric fraction (inner bays), linear in the
    bays that keep their built shape. Returns ``(le, te)``, (n_sections - 1)
    * n_per_section + 1 points each.
    """
    n_sub = int(n_per_section)
    if n_sub < 1:
        raise ValueError(f"n_per_section must be >= 1, got {n_per_section}")
    le_s, te_s = wing.leading_edge, wing.trailing_edge
    le_out, te_out = [], []
    for bay in wing.bays:
        for k in range(bay.first, bay.last):
            for j in range(n_sub):
                t = j / n_sub
                le_out.append((1.0 - t) * le_s[k] + t * le_s[k + 1])
                if bay.arc is not None:
                    s = (1.0 - t) * bay.fabric_fraction[k - bay.first] + t * bay.fabric_fraction[k + 1 - bay.first]
                    te_out.append(bay.arc_point(s))
                else:
                    te_out.append((1.0 - t) * te_s[k] + t * te_s[k + 1])
    le_out.append(le_s[-1])
    te_out.append(te_s[-1])
    return np.asarray(le_out), np.asarray(te_out)


def panel_bays(n_panels: int, n_per_section: int, strut_sections) -> np.ndarray:
    """Bay index of each aerodynamic panel, by construction.

    Panel j lies in wing section j // n_per_section, which lies in the bay
    between the struts around it. Not by sorting on y: at a curled tip the
    span coordinate is not monotonic.
    """
    struts = np.asarray(sorted(int(k) for k in strut_sections))
    section = np.arange(int(n_panels)) // int(n_per_section)
    return np.clip(np.searchsorted(struts, section, side="right") - 1, 0, len(struts) - 2)


__all__ = [
    "BillowedBay",
    "BillowedWing",
    "arc_half_angle",
    "billow_wing",
    "panel_bays",
    "refine_billowed_wing",
    "rotate_about_axis",
]
