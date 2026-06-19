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

"""Parametric 3D wing-planform representation for VSM ``aero_geometry.yaml``.

This module owns a small, dependency-light parametric description of a soft-kite
wing planform. It does **not** import VSM or CasADi: it reads/writes the same
``wing_sections`` table (``[airfoil_id, LE_x, LE_y, LE_z, TE_x, TE_y, TE_z]``)
that :class:`awetrim.aerodynamics.vsm_adapter.VSMAeroModelAdapter` already
consumes, so generated geometries drop straight into the VSM trim/sweep path.

The morphing operations are **quarter-chord (QC) anchored**: each section is
decomposed into a QC point and a chord vector, the QC arc is scaled, and the
chords are re-attached at the same 1/4-chord fraction. This keeps the planform
self-similar when only the aspect ratio changes and avoids distorting the
airfoil sections. The approach mirrors the QC-anchored Bezier shape-variation
generator in
``jellepoland/WES_aero_sim_for_kite_design``
(``compute_yaml_files_3D_shape_variations_QC_bezier.py``), parametrised here by
aspect ratio and anhedral.

Reference area / span conventions used throughout:

- ``flat_span``      — arc length of the QC polyline in the y-z plane, tip to tip.
- ``projected_span`` — tip-to-tip extent in y (``max(qc_y) - min(qc_y)``).
- ``area``           — flat (developed) area: trapezoidal integral of chord over
  the QC arc length.
- ``aspect_ratio``   — ``flat_span**2 / area``.
- ``anhedral_angle_deg`` — straight-line droop of the tip QC below the centre QC.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml

#: Canonical header order for the ``wing_sections`` table.
WING_SECTION_HEADERS = ["airfoil_id", "LE_x", "LE_y", "LE_z", "TE_x", "TE_y", "TE_z"]

#: Fraction of the chord at which the chord vector is anchored (quarter chord).
_QC_FRACTION = 0.25


@dataclass
class WingSections:
    """Quarter-chord-anchored description of a full-wing planform.

    Sections are stored in file order (typically tip -> centre -> tip). No
    symmetry is assumed by the morphing maths, but :attr:`is_symmetric` reports
    whether the stored sections mirror about ``y = 0``.

    Parameters
    ----------
    airfoil_ids:
        Integer airfoil identifiers, shape ``(n,)``.
    le:
        Leading-edge coordinates in the course/body frame, shape ``(n, 3)`` [m].
    te:
        Trailing-edge coordinates, shape ``(n, 3)`` [m].
    airfoils:
        The verbatim ``wing_airfoils`` block from the source ``aero_geometry``
        dict, carried through unchanged so geometries round-trip. ``None`` when
        the planform was built without airfoil metadata.
    """

    airfoil_ids: np.ndarray
    le: np.ndarray
    te: np.ndarray
    airfoils: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        self.airfoil_ids = np.asarray(self.airfoil_ids).reshape(-1)
        self.le = np.asarray(self.le, dtype=float).reshape(-1, 3)
        self.te = np.asarray(self.te, dtype=float).reshape(-1, 3)
        n = self.airfoil_ids.shape[0]
        if self.le.shape != (n, 3) or self.te.shape != (n, 3):
            raise ValueError(
                "airfoil_ids, le and te must describe the same number of "
                f"sections; got {n}, {self.le.shape}, {self.te.shape}"
            )
        if n < 2:
            raise ValueError("A wing needs at least two sections.")

    # ------------------------------------------------------------------
    # Construction / serialisation
    # ------------------------------------------------------------------

    @classmethod
    def from_aero_geometry(cls, config: Dict[str, Any]) -> "WingSections":
        """Build from a VSM ``aero_geometry`` dict (``wing_sections`` block)."""
        sections = config.get("wing_sections")
        if not sections or "headers" not in sections or "data" not in sections:
            raise ValueError(
                "config must contain a 'wing_sections' block with 'headers' "
                "and 'data'."
            )
        headers = list(sections["headers"])
        idx = {name: headers.index(name) for name in WING_SECTION_HEADERS if name in headers}
        missing = [name for name in WING_SECTION_HEADERS if name not in idx]
        if missing:
            raise ValueError(f"wing_sections.headers is missing columns: {missing}")

        rows = sections["data"]
        airfoil_ids = np.array([row[idx["airfoil_id"]] for row in rows])
        le = np.array(
            [[row[idx["LE_x"]], row[idx["LE_y"]], row[idx["LE_z"]]] for row in rows],
            dtype=float,
        )
        te = np.array(
            [[row[idx["TE_x"]], row[idx["TE_y"]], row[idx["TE_z"]]] for row in rows],
            dtype=float,
        )
        return cls(airfoil_ids=airfoil_ids, le=le, te=te, airfoils=config.get("wing_airfoils"))

    @classmethod
    def from_yaml(cls, path: Path | str) -> "WingSections":
        """Build from an ``aero_geometry.yaml`` file."""
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_aero_geometry(yaml.safe_load(f))

    def to_aero_geometry(
        self, *, resolve_csv_paths_relative_to: Path | str | None = None
    ) -> Dict[str, Any]:
        """Return a VSM ``aero_geometry`` dict for this planform.

        Parameters
        ----------
        resolve_csv_paths_relative_to:
            When given, relative ``csv_file_path`` entries in the carried
            ``wing_airfoils`` block are rewritten to absolute paths anchored at
            this directory. Use this when writing the variant YAML to a folder
            other than the one holding the polar CSVs, so VSM can still find them.
        """
        data = []
        for aid, le, te in zip(self.airfoil_ids, self.le, self.te):
            data.append(
                [
                    int(aid) if float(aid).is_integer() else aid,
                    float(le[0]), float(le[1]), float(le[2]),
                    float(te[0]), float(te[1]), float(te[2]),
                ]
            )
        config: Dict[str, Any] = {
            "wing_sections": {"headers": list(WING_SECTION_HEADERS), "data": data}
        }
        if self.airfoils is not None:
            airfoils = _deepcopy_yaml(self.airfoils)
            if resolve_csv_paths_relative_to is not None:
                _resolve_airfoil_csv_paths(airfoils, Path(resolve_csv_paths_relative_to))
            config["wing_airfoils"] = airfoils
        return config

    def to_yaml(
        self, path: Path | str, *, resolve_csv_paths_relative_to: Path | str | None = None
    ) -> Path:
        """Write this planform to ``path`` as an ``aero_geometry.yaml`` file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        config = self.to_aero_geometry(
            resolve_csv_paths_relative_to=resolve_csv_paths_relative_to
        )
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False, default_flow_style=None)
        return path

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    @property
    def n_sections(self) -> int:
        return self.airfoil_ids.shape[0]

    @property
    def chord_vectors(self) -> np.ndarray:
        """LE -> TE vectors, shape ``(n, 3)``."""
        return self.te - self.le

    @property
    def chords(self) -> np.ndarray:
        """Chord lengths, shape ``(n,)``."""
        return np.linalg.norm(self.chord_vectors, axis=1)

    @property
    def quarter_chord(self) -> np.ndarray:
        """Quarter-chord points ``le + 0.25 * (te - le)``, shape ``(n, 3)``."""
        return self.le + _QC_FRACTION * self.chord_vectors

    @property
    def _centre_index(self) -> int:
        """Index of the section closest to the symmetry plane ``y = 0``."""
        return int(np.argmin(np.abs(self.quarter_chord[:, 1])))

    @property
    def _tip_index(self) -> int:
        """Index of the section with the largest ``|y|`` (a wing tip)."""
        return int(np.argmax(np.abs(self.quarter_chord[:, 1])))

    @property
    def projected_span(self) -> float:
        """Tip-to-tip extent in y [m]."""
        y = self.quarter_chord[:, 1]
        return float(np.max(y) - np.min(y))

    @property
    def flat_span(self) -> float:
        """Arc length of the QC polyline in the y-z plane, tip to tip [m]."""
        qc_yz = self.quarter_chord[:, 1:]
        return float(np.sum(np.linalg.norm(np.diff(qc_yz, axis=0), axis=1)))

    @property
    def area(self) -> float:
        """Flat (developed) wing area [m^2].

        Trapezoidal integral of the chord over the QC arc length.
        """
        chords = self.chords
        seg = np.linalg.norm(np.diff(self.quarter_chord[:, 1:], axis=0), axis=1)
        mean_chord = 0.5 * (chords[:-1] + chords[1:])
        return float(np.sum(mean_chord * seg))

    @property
    def mean_chord(self) -> float:
        """Area-weighted mean chord ``area / flat_span`` [m]."""
        return self.area / self.flat_span

    @property
    def aspect_ratio(self) -> float:
        """Flat aspect ratio ``flat_span**2 / area``."""
        return self.flat_span**2 / self.area

    @property
    def anhedral_angle_deg(self) -> float:
        """Straight-line droop of the tip QC below the centre QC [deg].

        ``0`` for a flat wing; positive when the tips sit below the centre.
        """
        qc = self.quarter_chord
        c, t = self._centre_index, self._tip_index
        dy = abs(qc[t, 1] - qc[c, 1])
        dz = qc[c, 2] - qc[t, 2]
        return float(np.degrees(np.arctan2(dz, dy)))

    @property
    def taper_ratio(self) -> float:
        """Tip-to-root chord ratio ``c_tip / c_root``."""
        chords = self.chords
        return float(chords[self._tip_index] / chords[self._centre_index])

    @property
    def tip_twist_deg(self) -> float:
        """Geometric twist of the tip chord vs the root chord [deg].

        Measured as the chord pitch angle (in the x-z plane, about the spanwise
        y-axis) at the tip minus that at the root. Positive when the tip chord
        is pitched nose-up relative to the root.
        """
        cv = self.chord_vectors

        def pitch(v: np.ndarray) -> float:
            return float(np.degrees(np.arctan2(-v[2], v[0])))

        return pitch(cv[self._tip_index]) - pitch(cv[self._centre_index])

    def is_symmetric(self, tol: float = 1e-6) -> bool:
        """True when sections mirror about ``y = 0`` (within ``tol``)."""
        qc = self.quarter_chord
        mirrored = qc * np.array([1.0, -1.0, 1.0])
        # Every QC point must have a mirror partner among the sections.
        for point in qc:
            if np.min(np.linalg.norm(mirrored - point, axis=1)) > tol:
                return False
        return True

    def properties(self) -> Dict[str, float]:
        """Return the scalar planform metrics as a plain dict."""
        return {
            "projected_span": self.projected_span,
            "flat_span": self.flat_span,
            "area": self.area,
            "mean_chord": self.mean_chord,
            "aspect_ratio": self.aspect_ratio,
            "anhedral_angle_deg": self.anhedral_angle_deg,
            "taper_ratio": self.taper_ratio,
            "tip_twist_deg": self.tip_twist_deg,
        }


# ----------------------------------------------------------------------
# Morphing
# ----------------------------------------------------------------------


def morph_wing(
    sections: WingSections,
    *,
    span_scale: float = 1.0,
    chord_scale: float = 1.0,
    anhedral_scale: float = 1.0,
    taper_ratio: float = 1.0,
    twist_deg: float = 0.0,
) -> WingSections:
    """Return a QC-anchored morph of ``sections``.

    The quarter-chord arc is scaled about the centre section and chords are
    re-attached at the quarter-chord point:

    - ``span_scale``    scales the QC arc uniformly about the centre (x, y, z),
      stretching the planform spanwise while preserving its shape.
    - ``anhedral_scale`` additionally scales the *vertical* QC deviation, opening
      or closing the wing arc.
    - ``chord_scale``   scales every chord vector (length and orientation
      preserved direction-wise).
    - ``taper_ratio``   multiplies the tip/root chord ratio by this factor via a
      linear-in-span chord redistribution, holding the flat area constant
      (``1.0`` leaves taper unchanged).
    - ``twist_deg``     adds a linear washout: each chord is rotated about the
      spanwise y-axis through its quarter chord, from ``0`` at the centre to
      ``twist_deg`` at the tips. Lengths (and thus area/AR) are unchanged.

    With ``anhedral_scale = 1`` and ``chord_scale = 1 / span_scale`` the flat
    area is preserved and the aspect ratio scales by ``span_scale**2``. Taper and
    twist are decoupled from aspect ratio and anhedral.
    """
    qc = sections.quarter_chord
    chord_vectors = sections.chord_vectors
    c = sections._centre_index
    qc_centre = qc[c]

    scale = np.array(
        [span_scale, span_scale, span_scale * anhedral_scale], dtype=float
    )
    new_qc = qc_centre + (qc - qc_centre) * scale
    new_chord_vectors = chord_vectors * chord_scale

    eta = _spanwise_fraction(new_qc, c)  # 0 at centre, 1 at the tips
    if taper_ratio != 1.0:
        new_chord_vectors = _apply_taper(new_chord_vectors, new_qc, eta, taper_ratio)
    if twist_deg != 0.0:
        new_chord_vectors = _rotate_about_y(
            new_chord_vectors, np.deg2rad(twist_deg) * eta
        )

    new_le = new_qc - _QC_FRACTION * new_chord_vectors
    new_te = new_le + new_chord_vectors
    return WingSections(
        airfoil_ids=sections.airfoil_ids.copy(),
        le=new_le,
        te=new_te,
        airfoils=_deepcopy_yaml(sections.airfoils),
    )


def morph_wing_to(
    sections: WingSections,
    *,
    target_aspect_ratio: Optional[float] = None,
    target_anhedral_deg: Optional[float] = None,
    taper_ratio: float = 1.0,
    twist_deg: float = 0.0,
    preserve_area: bool = True,
    max_iter: int = 30,
    tol: float = 1e-5,
) -> WingSections:
    """Morph ``sections`` to hit target aspect ratio and/or anhedral angle.

    Aspect ratio and anhedral interact (changing the arc changes its developed
    length), so the scales are found by a short fixed-point iteration. Taper and
    twist are decoupled, so they are applied once after the iteration.

    Parameters
    ----------
    target_aspect_ratio:
        Desired flat aspect ratio. ``None`` leaves the aspect ratio free.
    target_anhedral_deg:
        Desired tip droop angle [deg]. ``None`` leaves the arc unchanged.
        Requires a non-zero baseline anhedral to scale from.
    taper_ratio:
        Multiplier on the tip/root chord ratio (``1.0`` leaves taper unchanged).
        Area-preserving.
    twist_deg:
        Linear tip washout to add [deg] (``0.0`` leaves twist unchanged).
    preserve_area:
        When changing aspect ratio, also scale the chords by ``1 / span_scale``
        so the flat area is held constant (the usual planform-study convention).
    """
    has_arc_target = (
        target_aspect_ratio is not None or target_anhedral_deg is not None
    )
    has_section_mod = taper_ratio != 1.0 or twist_deg != 0.0
    if not has_arc_target and not has_section_mod:
        raise ValueError(
            "Provide at least one of target_aspect_ratio, target_anhedral_deg, "
            "taper_ratio, twist_deg."
        )

    current = sections
    for _ in range(max_iter if has_arc_target else 0):
        span_scale = 1.0
        chord_scale = 1.0
        anhedral_scale = 1.0

        if target_anhedral_deg is not None:
            current_anhedral = current.anhedral_angle_deg
            if abs(current_anhedral) < 1e-9:
                raise ValueError(
                    "Cannot scale anhedral from a (near-)flat wing; the baseline "
                    "anhedral angle is ~0 deg."
                )
            anhedral_scale = target_anhedral_deg / current_anhedral

        if target_aspect_ratio is not None:
            span_scale = float(np.sqrt(target_aspect_ratio / current.aspect_ratio))
            if preserve_area:
                chord_scale = 1.0 / span_scale

        current = morph_wing(
            current,
            span_scale=span_scale,
            chord_scale=chord_scale,
            anhedral_scale=anhedral_scale,
        )

        converged = True
        if target_aspect_ratio is not None:
            converged &= abs(current.aspect_ratio - target_aspect_ratio) <= tol * max(
                1.0, target_aspect_ratio
            )
        if target_anhedral_deg is not None:
            converged &= abs(current.anhedral_angle_deg - target_anhedral_deg) <= 1e-3
        if converged:
            break

    if has_section_mod:
        current = morph_wing(current, taper_ratio=taper_ratio, twist_deg=twist_deg)

    return current


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------


def _spanwise_fraction(qc: np.ndarray, centre_index: int) -> np.ndarray:
    """Normalised spanwise station: 0 at the centre, 1 at the tips."""
    d = np.abs(qc[:, 1] - qc[centre_index, 1])
    dmax = d.max()
    return d / dmax if dmax > 0 else np.zeros_like(d)


def _apply_taper(
    chord_vectors: np.ndarray, qc: np.ndarray, eta: np.ndarray, taper_ratio: float
) -> np.ndarray:
    """Scale chord lengths by a linear-in-span factor, preserving flat area.

    The chord multiplier is ``m(eta) = a * (1 + (taper_ratio - 1) * eta)`` so the
    tip/root chord ratio is multiplied by ``taper_ratio``; ``a`` is solved so the
    trapezoidal flat area (chord integrated over the QC arc) is unchanged.
    """
    chords = np.linalg.norm(chord_vectors, axis=1)
    seg = np.linalg.norm(np.diff(qc[:, 1:], axis=0), axis=1)

    def _area(mult: np.ndarray) -> float:
        cm = chords * mult
        return float(np.sum(0.5 * (cm[:-1] + cm[1:]) * seg))

    base = 1.0 + (taper_ratio - 1.0) * eta
    area0 = _area(np.ones_like(chords))
    area_base = _area(base)
    a = area0 / area_base if area_base > 0 else 1.0
    mult = a * base
    return chord_vectors * mult[:, None]


def _rotate_about_y(vectors: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Rotate each row of ``vectors`` about the +y axis by ``angles`` [rad]."""
    cos, sin = np.cos(angles), np.sin(angles)
    x, y, z = vectors[:, 0], vectors[:, 1], vectors[:, 2]
    return np.stack([x * cos + z * sin, y, -x * sin + z * cos], axis=1)


def _deepcopy_yaml(obj: Any) -> Any:
    """Deep-copy plain YAML data (dict/list/scalars) without importing copy."""
    if isinstance(obj, dict):
        return {k: _deepcopy_yaml(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deepcopy_yaml(v) for v in obj]
    return obj


def _resolve_airfoil_csv_paths(airfoils: Dict[str, Any], anchor: Path) -> None:
    """Rewrite relative ``csv_file_path`` entries to absolute paths in place."""
    anchor = Path(anchor).resolve()
    for row in airfoils.get("data", []):
        if len(row) >= 3 and isinstance(row[2], dict) and "csv_file_path" in row[2]:
            p = row[2]["csv_file_path"]
            if not Path(p).is_absolute():
                row[2]["csv_file_path"] = str((anchor / p).resolve())
