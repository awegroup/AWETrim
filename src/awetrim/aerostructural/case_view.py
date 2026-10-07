"""What a converged coupled case looks like and how it trimmed.

Two halves, one per moment in a case's life:

* **At solve time** the drivers call :func:`save_case_view`, which writes
  ``case.json`` next to ``sim_output.h5``: the topology to draw (wing surface,
  inflated tubes, bridle lines with their names and kinds), the line tensions of
  the converged state, and the flight condition the case was solved at. The
  driver is the only place that has the live structure, so tensions and the
  Billow canopy mesh are taken there rather than reconstructed afterwards.
* **Afterwards** :func:`load_case` reads both files back and
  :func:`trim_summary` turns the solver's metadata into one table of trim
  characteristics. The wireframe and Billow drivers store their trims in
  different layouts (flat keys vs a ``trim_results`` block); the table hides
  that.

Plain numpy and dicts only, so a script can build a viewer or a report without
importing VSM or Billow.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from awetrim.aerostructural.utils import load_sim_output

CASE_VIEW_FILENAME = "case.json"

#: Line kinds the viewer distinguishes. ``wing`` is a structural element
#: between two wing nodes (the wireframe's spring net, or nothing on Billow,
#: whose wing is tubes + membrane); everything else hangs below the wing.
LINE_KINDS = ("wing", "bridle", "pulley", "depower_tape", "steering_tape")


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------


def wing_triangles(le_indices: Sequence[int], te_indices: Sequence[int]) -> np.ndarray:
    """Two triangles per rib bay, from the leading- and trailing-edge nodes.

    The wireframe has no canopy element, so this is only a surface to look
    at: LE_i, TE_i, TE_i+1, LE_i+1 split along the TE_i - LE_i+1 diagonal.
    """
    le = np.asarray(le_indices, dtype=int).ravel()
    te = np.asarray(te_indices, dtype=int).ravel()
    if le.size != te.size:
        raise ValueError(f"{le.size} leading-edge nodes but {te.size} trailing-edge nodes")
    triangles = []
    for i in range(le.size - 1):
        triangles.append((le[i], te[i], le[i + 1]))
        triangles.append((te[i], te[i + 1], le[i + 1]))
    return np.asarray(triangles, dtype=int).reshape(-1, 3)


def bridle_line_names(struc_geometry: Mapping[str, Any]) -> dict[frozenset, str]:
    """Node pair -> line name, from the geometry's ``bridle_connections``.

    A pulley row ``[name, ci, cj, ck]`` names both of its arms.
    """
    names: dict[frozenset, str] = {}
    for row in struc_geometry.get("bridle_connections", {}).get("data", []):
        name, nodes = str(row[0]), [int(n) for n in row[1:]]
        for first, second in zip(nodes[:-1], nodes[1:]):
            names.setdefault(frozenset((first, second)), name)
    return names


def classify_lines(
    connectivity: Sequence[Sequence[int]],
    *,
    wing_nodes: Sequence[int],
    struc_geometry: Mapping[str, Any],
    power_tape_index: int | None = None,
    steering_tape_indices: Sequence[int] | None = None,
    pulley_line_indices: Sequence[int] | None = None,
) -> tuple[list[str], list[str]]:
    """Kind and name of every element of the reader's connectivity.

    Kinds are those of :data:`LINE_KINDS`; names come from the geometry's
    bridle table and are empty for wing elements.
    """
    wing = {int(n) for n in wing_nodes}
    names_by_pair = bridle_line_names(struc_geometry)
    pulleys = {int(i) for i in (pulley_line_indices or [])}
    steering = {int(i) for i in (steering_tape_indices or [])[:2]}
    kinds, names = [], []
    for index, (ci, cj) in enumerate(np.asarray(connectivity, dtype=int)[:, :2]):
        pair = frozenset((int(ci), int(cj)))
        if power_tape_index is not None and index == int(power_tape_index):
            kind = "depower_tape"
        elif index in steering:
            kind = "steering_tape"
        elif index in pulleys:
            kind = "pulley"
        elif pair <= wing:
            kind = "wing"
        else:
            kind = "bridle"
        kinds.append(kind)
        names.append("" if kind == "wing" else names_by_pair.get(pair, ""))
    return kinds, names


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def save_case_view(results_dir: Path | str, view: Mapping[str, Any]) -> Path:
    """Write ``case.json`` into ``results_dir`` and return its path.

    Expected keys (all lists in the structure's node / element ordering):
    ``backend``, ``kite``, ``connectivity``, ``line_kind``, ``line_name``,
    ``line_tension`` [N], ``triangles``, ``tubes``, ``tube_diameter`` [m],
    ``fixed``, ``flight_state`` and ``masses`` [kg]. Non-finite numbers are
    stored as ``null``.
    """
    path = Path(results_dir) / CASE_VIEW_FILENAME
    path.write_text(json.dumps(_jsonable(dict(view)), indent=1), encoding="utf-8")
    return path


#: Config keys that define the flight condition of a case, as stored in case.json.
FLIGHT_STATE_KEYS = (
    "angle_elevation_deg",
    "angle_azimuth_deg",
    "angle_course_deg",
    "distance_radial",
    "speed_radial",
    "wind_speed_wind_ref",
    "is_with_gravity",
    "power_tape_final_extension",
    "steering_tape_final_extension",
)


def rom_reference_area(system_config_path: Path | str) -> float:
    """``aerodynamics.reference_area`` [m2] of the ROM the system file selects.

    NaN when the kite has no ROM or the ROM states no area.
    """
    import yaml

    from awetrim.system.factory import resolve_rom_config_path

    try:
        rom_path = resolve_rom_config_path(system_config_path)
    except (OSError, ValueError, KeyError):
        return float("nan")
    if rom_path is None or not Path(rom_path).exists():
        return float("nan")
    rom = yaml.safe_load(Path(rom_path).read_text(encoding="utf-8")) or {}
    return _float((rom.get("aerodynamics") or {}).get("reference_area"))


def build_case_view(
    *,
    backend: str,
    kite_name: str,
    config: Mapping[str, Any],
    system_config_path: Path | str,
    struc_geometry: Mapping[str, Any],
    nodes: np.ndarray,
    masses: np.ndarray,
    connectivity: Sequence[Sequence[int]],
    line_tension: Sequence[float],
    le_indices: Sequence[int],
    te_indices: Sequence[int],
    power_tape_index: int | None = None,
    steering_tape_indices: Sequence[int] | None = None,
    pulley_line_indices: Sequence[int] | None = None,
    rest_lengths: Sequence[float] | None = None,
    fixed: Sequence[int] = (0,),
    triangles: np.ndarray | None = None,
    tubes: np.ndarray | None = None,
    tube_diameter: Sequence[float] | None = None,
    drawn_lines: Sequence[bool] | None = None,
) -> dict[str, Any]:
    """The ``case.json`` payload of one converged case.

    ``connectivity``, ``line_tension`` and ``rest_lengths`` are in the
    geometry reader's element ordering. ``drawn_lines`` masks the elements the
    viewer should draw as lines (Billow replaces the canopy springs by a
    membrane, which is drawn from ``triangles`` instead). ``triangles``
    defaults to the rib-bay surface of :func:`wing_triangles`.
    """
    connectivity = np.asarray(connectivity, dtype=int)[:, :2]
    kinds, names = classify_lines(
        connectivity,
        wing_nodes=list(le_indices) + list(te_indices),
        struc_geometry=struc_geometry,
        power_tape_index=power_tape_index,
        steering_tape_indices=steering_tape_indices,
        pulley_line_indices=pulley_line_indices,
    )
    keep = (
        np.ones(len(connectivity), dtype=bool)
        if drawn_lines is None
        else np.asarray(drawn_lines, dtype=bool)
    )
    tension = np.asarray(line_tension, dtype=float).ravel()
    if tension.size != len(connectivity):
        raise ValueError(
            f"{tension.size} line tensions for {len(connectivity)} elements"
        )
    tapes = {}
    rest = None if rest_lengths is None else np.asarray(rest_lengths, dtype=float)
    if rest is not None and rest.size == len(connectivity):
        if power_tape_index is not None:
            tapes["depower"] = float(rest[int(power_tape_index)])
        if steering_tape_indices is not None and len(steering_tape_indices) >= 2:
            left, right = (int(i) for i in steering_tape_indices[:2])
            tapes["steering"] = 0.5 * float(rest[right] - rest[left])
    if triangles is None:
        triangles = wing_triangles(le_indices, te_indices)
    return {
        "backend": backend,
        "kite": kite_name,
        "n_nodes": int(np.asarray(nodes).shape[0]),
        "connectivity": connectivity[keep],
        "line_kind": [k for k, m in zip(kinds, keep) if m],
        "line_name": [n for n, m in zip(names, keep) if m],
        "line_tension": tension[keep],
        "triangles": np.asarray(triangles, dtype=int),
        "tubes": np.zeros((0, 2), dtype=int) if tubes is None else np.asarray(tubes, dtype=int)[:, :2],
        "tube_diameter": [] if tube_diameter is None else list(tube_diameter),
        "fixed": [int(i) for i in fixed],
        "masses": np.asarray(masses, dtype=float),
        "tape_lengths": tapes,
        "flight_state": {k: config.get(k) for k in FLIGHT_STATE_KEYS if k in config},
        "reference_area_rom": rom_reference_area(system_config_path),
        "system_file": Path(system_config_path).name,
    }


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def last_frame(tracking: Mapping[str, Any], meta: Mapping[str, Any], key="positions"):
    """The converged frame of a tracked array: the last finite, non-zero one.

    ``n_iter`` frames are stored; a run that stopped early leaves zero-filled
    frames after it.
    """
    frames = np.asarray(tracking[key], dtype=float)
    n_iter = int(meta.get("n_iter", len(frames)))
    candidates = range(min(n_iter, len(frames)) - 1, -1, -1)
    for index in candidates:
        frame = frames[index]
        if np.isfinite(frame).all() and not np.allclose(frame, 0.0):
            return frame
    return frames[-1]


def load_case(case_dir: Path | str) -> dict[str, Any]:
    """``sim_output.h5`` + ``case.json`` + the input ``config.yaml`` of one case.

    Returns ``meta``, ``tracking``, ``view`` (``None`` for a case solved before
    ``case.json`` existed), ``config``, ``initial`` and ``final`` node
    positions.
    """
    import yaml

    case_dir = Path(case_dir)
    meta, tracking = load_sim_output(case_dir / "sim_output.h5")
    view_path = case_dir / CASE_VIEW_FILENAME
    view = json.loads(view_path.read_text(encoding="utf-8")) if view_path.exists() else None
    config_path = case_dir / "config.yaml"
    config = (
        yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if config_path.exists()
        else {}
    )
    return {
        "case_dir": case_dir,
        "meta": meta,
        "tracking": tracking,
        "view": view,
        "config": config or {},
        "initial": np.asarray(tracking["positions"][0], dtype=float),
        "final": last_frame(tracking, meta),
    }


def rigid_motion_removed(reference: np.ndarray, deformed: np.ndarray) -> np.ndarray:
    """Per-node displacement left after the best-fit rigid motion (Kabsch).

    The coupled drivers rotate the whole kite into its trimmed attitude, so most
    of the raw node motion is rigid body. This is the part that strains the
    structure.
    """
    a = np.asarray(reference, dtype=float)
    b = np.asarray(deformed, dtype=float)
    a0, b0 = a - a.mean(axis=0), b - b.mean(axis=0)
    u, _, vt = np.linalg.svd(a0.T @ b0)
    if np.linalg.det(u @ vt) < 0:
        u[:, -1] *= -1
    return np.linalg.norm(b0 - a0 @ (u @ vt), axis=1)


def _float(value) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return value


def trim_results(meta: Mapping[str, Any]) -> dict[str, Any]:
    """The ``trim_results`` block both coupled drivers store (``{}`` if absent)."""
    trim = meta.get("trim_results")
    if isinstance(trim, str):
        try:
            trim = json.loads(trim)
        except ValueError:
            trim = None
    return dict(trim) if isinstance(trim, Mapping) else {}


def lift_and_drag(trim: Mapping[str, Any]) -> tuple[float, float]:
    """Lift and drag [N] of the trim's total aerodynamic force.

    Drag is the component along the apparent wind, lift the magnitude of the
    rest. Area-free, so it needs no reference area to mean something.
    """
    force = np.asarray(trim.get("total_aero_force_vec", [np.nan] * 3), dtype=float)
    va = np.asarray(trim.get("va_vel_world", [np.nan] * 3), dtype=float)
    norm = float(np.linalg.norm(va))
    if not (np.isfinite(force).all() and np.isfinite(va).all()) or norm == 0.0:
        return float("nan"), float("nan")
    drag = float(force @ va / norm)
    lift = float(np.linalg.norm(force - drag * va / norm))
    return lift, drag


def trim_summary(case: Mapping[str, Any]) -> list[tuple[str, str, float | str, str]]:
    """Trim characteristics of one loaded case, as ``(group, label, value, unit)``.

    ``case`` is what :func:`load_case` returns. CL and CD are given on the VSM
    deformed projected area the trim used and, when the kite's ROM states one,
    on that reference area too: a coefficient without its area is not a number.
    """
    meta, config = case["meta"], case["config"]
    view = case.get("view") or {}
    state = view.get("flight_state") or {}

    def cfg(key):
        return _float(state.get(key, config.get(key)))

    trim = trim_results(meta)
    opt_x = np.asarray(meta.get("opt_x", []), dtype=float).ravel()
    speed_apparent = _float(trim.get("Umag", meta.get("speed_apparent", meta.get("Umag"))))
    speed_tangential = _float(
        meta.get("speed_tangential", opt_x[0] if opt_x.size else np.nan)
    )
    course_rate = _float(meta.get("course_rate", opt_x[4] if opt_x.size > 4 else np.nan))

    def trim_value(key):
        return _float(trim.get(key, meta.get(key)))

    rows: list[tuple[str, str, float | str, str]] = [
        ("Flight state", "Elevation", cfg("angle_elevation_deg"), "deg"),
        ("Flight state", "Azimuth", cfg("angle_azimuth_deg"), "deg"),
        ("Flight state", "Course", cfg("angle_course_deg"), "deg"),
        ("Flight state", "Tether length", cfg("distance_radial"), "m"),
        ("Flight state", "Reel-out speed", cfg("speed_radial"), "m/s"),
        ("Flight state", "Wind speed (reference height)", cfg("wind_speed_wind_ref"), "m/s"),
        ("Flight state", "Gravity", "on" if config.get("is_with_gravity") else "off", ""),
        ("Trim", "Apparent wind speed", speed_apparent, "m/s"),
        ("Trim", "Tangential kite speed", speed_tangential, "m/s"),
        ("Trim", "Angle of attack", trim_value("aoa_deg"), "deg"),
        ("Trim", "Sideslip", trim_value("side_slip_deg"), "deg"),
        ("Trim", "Course rate", course_rate, "rad/s"),
        ("Trim", "Tether force at kite", _float(meta.get("tether_force")), "N"),
    ]
    if np.isfinite(course_rate) and abs(course_rate) > 1e-4 and np.isfinite(speed_tangential):
        rows.append(("Trim", "Turn radius", abs(speed_tangential / course_rate), "m"))
    if opt_x.size >= 4:
        rows += [
            ("Trim", "Roll (trim rotation)", opt_x[1], "deg"),
            ("Trim", "Pitch (trim rotation)", opt_x[2], "deg"),
            ("Trim", "Yaw (trim rotation)", opt_x[3], "deg"),
        ]
    elif "roll_body_deg" in meta:
        rows.append(("Trim", "Roll (trim rotation)", _float(meta["roll_body_deg"]), "deg"))

    # The VSM force: wing (and bridle lines when VSM carries them), NOT the
    # KCU drag, which the trim adds separately, nor the tether. Its L/D is
    # therefore above the kite's glide ratio v_tau / v_w.
    lift, drag = lift_and_drag(trim)
    rows += [
        ("Aerodynamics", "Lift (VSM force)", lift, "N"),
        ("Aerodynamics", "Drag (VSM force)", drag, "N"),
        ("Aerodynamics", "Lift-to-drag ratio (VSM force)", lift / drag if drag else float("nan"), ""),
    ]
    kcu = np.asarray(trim.get("kcu_drag_force_vsm", [np.nan] * 3), dtype=float)
    if np.isfinite(kcu).all():
        rows.append(("Aerodynamics", "KCU drag", float(np.linalg.norm(kcu)), "N"))
    # Coefficients only with their area. The trim's own CL/CD are on the
    # projected area of the DEFORMED wing VSM computed them on; the ROM area
    # (if the kite's ROM states one) is the one flight data are reduced on.
    if "cl" in trim:
        rows += [
            ("Aerodynamics", "CL (VSM, deformed projected area)", _float(trim["cl"]), ""),
            ("Aerodynamics", "CD (VSM, deformed projected area)", _float(trim.get("cd")), ""),
        ]
    area_rom = _float(view.get("reference_area_rom"))
    rho = _float(config.get("rho", 1.225))
    if np.isfinite(area_rom) and area_rom > 0 and np.isfinite(speed_apparent):
        q_area = 0.5 * rho * speed_apparent**2 * area_rom
        rows += [
            ("Aerodynamics", f"CL (ROM area {area_rom:.2f} m2)", lift / q_area, ""),
            ("Aerodynamics", f"CD (ROM area {area_rom:.2f} m2)", drag / q_area, ""),
        ]

    masses = np.asarray(view.get("masses", []), dtype=float)
    if masses.size:
        rows += [
            ("Structure", "Total mass", float(masses.sum()), "kg"),
            ("Structure", "KCU mass", float(masses[0]), "kg"),
        ]
    tension = np.asarray(
        [np.nan if t is None else t for t in view.get("line_tension", [])], dtype=float
    )
    kinds = view.get("line_kind", [])
    hanging = np.array([k != "wing" for k in kinds], dtype=bool)
    if tension.size and hanging.any():
        below = tension[hanging]
        rows += [
            ("Structure", "Largest bridle-line tension", float(np.nanmax(below)), "N"),
            ("Structure", "Slack bridle lines", f"{int(np.sum(below < 1.0))} of {below.size}", ""),
        ]
    # The wing (every node of the drawn surface) apart from the bridle: the
    # bridle swings and sags, the wing should barely change shape.
    wing = np.unique(np.asarray(view.get("triangles", []), dtype=int).ravel())
    if wing.size >= 3:
        wing_deformation = rigid_motion_removed(case["initial"][wing], case["final"][wing])
        rows += [
            ("Structure", "Wing deformation, max (rigid motion removed)", float(wing_deformation.max()), "m"),
            ("Structure", "Wing deformation, mean (rigid motion removed)", float(wing_deformation.mean()), "m"),
        ]
    deformation = rigid_motion_removed(case["initial"], case["final"])
    rows.append(
        ("Structure", "All nodes incl. bridle, max (rigid motion removed)", float(deformation.max()), "m")
    )
    tapes = view.get("tape_lengths") or {}
    for label, key in (("Depower tape length", "depower"), ("Steering half-difference", "steering")):
        if key in tapes:
            rows.append(("Actuation", label, _float(tapes[key]), "m"))

    residual = _float(meta.get("residual_force_n"))
    if not np.isfinite(residual) and "residual_norm" in case["tracking"]:
        norms = np.asarray(case["tracking"]["residual_norm"], dtype=float)
        nonzero = np.flatnonzero(norms)
        residual = float(norms[nonzero[-1]]) if nonzero.size else float("nan")
    rows += [
        ("Convergence", "Converged", "yes" if bool(meta.get("converged")) else "NO", ""),
        ("Convergence", "Coupled iterations", int(meta.get("n_iter", 0)), ""),
        ("Convergence", "Final force residual", residual, "N"),
        ("Convergence", "Wall time", _float(meta.get("total_time_s")), "s"),
    ]
    return rows


def format_summary(rows: Sequence[tuple[str, str, Any, str]]) -> str:
    """The summary as an aligned plain-text table, grouped."""
    lines, group = [], None
    width = max(len(label) for _, label, _, _ in rows)
    for row_group, label, value, unit in rows:
        if row_group != group:
            lines.append(f"\n{row_group}")
            group = row_group
        if isinstance(value, float):
            if not np.isfinite(value):
                text = "n/a"
            elif value != 0.0 and abs(value) < 1e-3:
                text = "0"  # solver round-off on a quantity that is zero
            else:
                text = f"{value:.4g}"
        else:
            text = str(value)
        lines.append(f"  {label:<{width}}  {text:>10} {unit}")
    return "\n".join(lines).lstrip("\n")
