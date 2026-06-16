from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.animation import FuncAnimation, writers
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from common import (
    add_common_arguments,
    build_body,
    build_system_model,
    output_dir,
    parsed_common,
    print_trim_summary,
    save_figure,
    write_json,
    DEFAULT_OUTPUT_ROOT,
)

from awetrim.aerodynamics.vsm_quasi_steady import (
    ALL_STATE_NAMES,
    DEFAULT_AXES,
    LAT_STATES,
    LONG_STATES,
    compute_vsm_trim_stability_derivatives,
    solve_vsm_qs_trim_with_williams_tether,
    solve_vsm_quasi_steady_trim,
)
from awetrim.aerodynamics.protocols import AxisDefinition
from awetrim.system.williams_tether import WilliamsTether

_SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from awetrim.aerostructural.utils import load_sim_output
from awetrim.identification.rigid_body_axes import (
    compute_rigid_body_axes,
    load_psm_nodes_and_masses,
)

# ---------------------------------------------------------------------------
# Operating condition (edit here)
# ---------------------------------------------------------------------------
# These set the trim/stability operating point. They become the argument
# defaults, so the matching CLI flags (--elevation-deg, --azimuth-deg,
# --course-deg, --wind-speed, --radial-speed, --distance-radial) still
# override them when provided.
OPERATING_CONDITION = {
    "elevation_deg": 35.0,
    "azimuth_deg": 0.0,
    "course_deg": 45.0,
    "wind_speed": 8.0,
    "radial_speed": 1.5,
    "distance_radial": 200.0,
}


def _load_rigid_body_axes_from_result(result_path: Path, struc_override: Path | None):
    """Load RigidBodyAxes from a structural result directory or struc_geometry YAML.

    Priority (same as plot_body_axes.py):
      1. struc_override (explicit --rigid-body-struc)
      2. {result_path}/struc_geometry.yaml  (deformed, saved by save_geometry_snapshot)
      3. HDF5 positions + data/{kite}/struc_geometry.yaml  (fallback)
    """
    result_path = result_path.resolve()
    case_dir = result_path if result_path.is_dir() else result_path.parent

    if struc_override is not None:
        struc_path = struc_override.resolve()
        with struc_path.open("r", encoding="utf-8") as f:
            sg = yaml.safe_load(f)
        nodes, m_arr = load_psm_nodes_and_masses(sg)
    else:
        saved = case_dir / "struc_geometry.yaml"
        if saved.exists():
            with saved.open("r", encoding="utf-8") as f:
                sg = yaml.safe_load(f)
            nodes, m_arr = load_psm_nodes_and_masses(sg)
        else:
            h5 = case_dir / "sim_output.h5"
            if not h5.exists():
                raise FileNotFoundError(
                    f"No sim_output.h5 or struc_geometry.yaml in {case_dir}"
                )
            _, tracking = load_sim_output(h5)
            nodes = np.asarray(tracking["positions"][-1], dtype=float)
            # infer struc_geometry from path layout
            parts = case_dir.parts
            try:
                ri = next(i for i, p in enumerate(parts) if p == "results")
                kite_name = (
                    parts[ri + 2]
                    if parts[ri + 1] == "aerostructural"
                    else parts[ri + 1]
                )
                project_root = Path(*parts[:ri])
                kite_data = project_root / "data" / kite_name
                fallback = kite_data / "struc_geometry.yaml"
                if not fallback.exists():
                    fallback = (
                        kite_data
                        / "deformed_results"
                        / "powered_2019"
                        / "struc_geometry.yaml"
                    )
            except (StopIteration, IndexError):
                raise FileNotFoundError(
                    "Could not infer struc_geometry path from result layout."
                )
            if not fallback.exists():
                raise FileNotFoundError(
                    f"Fallback struc_geometry not found: {fallback}"
                )
            with fallback.open("r", encoding="utf-8") as f:
                sg = yaml.safe_load(f)
            _, m_arr = load_psm_nodes_and_masses(sg)

    return compute_rigid_body_axes(nodes, m_arr)


def _axes_to_dict(axes: AxisDefinition) -> dict[str, list[float]]:
    """JSON-friendly axis definition."""
    return {
        "course": np.asarray(axes.course, dtype=float).tolist(),
        "normal": np.asarray(axes.normal, dtype=float).tolist(),
        "radial": np.asarray(axes.radial, dtype=float).tolist(),
    }


# ---------------------------------------------------------------------------
# Animation helpers
# ---------------------------------------------------------------------------


def _mode_time_response(
    eigenvalue: complex,
    eigenvector: np.ndarray,
    time_vector: np.ndarray,
    amplitude: float,
) -> np.ndarray:
    vec = np.asarray(eigenvector, dtype=complex)
    return np.real(amplitude * np.outer(vec, np.exp(eigenvalue * time_vector)))


def _rot_rad(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    ax = np.asarray(axis, dtype=float)
    ax = ax / np.linalg.norm(ax)
    kx, ky, kz = ax
    skew = np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]], dtype=float)
    return (
        np.eye(3) + np.sin(angle_rad) * skew + (1 - np.cos(angle_rad)) * (skew @ skew)
    )


def _rotate_pts(pts: np.ndarray, R: np.ndarray, origin: np.ndarray) -> np.ndarray:
    return origin + (pts - origin) @ R.T


def _save_animation(
    anim: FuncAnimation, path, *, fps: int, fmt: str, dpi: int | None = None
) -> None:
    if fmt == "mp4":
        if not writers.is_available("ffmpeg"):
            raise RuntimeError(
                "MP4 export for this animation requires ffmpeg. "
                "The clean combined animation can export MP4 through OpenCV."
            )
        anim.save(str(path), writer="ffmpeg", fps=fps, dpi=dpi or 180)
    else:
        anim.save(str(path), writer="pillow", fps=fps, dpi=dpi or 100)
    print(f"Wrote {path}")


def _physics_duration(eigenvalue: complex, *, max_s: float = 30.0) -> float:
    """Compute how many seconds of real physics time to animate.

    Stable/decaying:   5 time constants  (decays to ~0.7 %)
    Unstable/growing:  3 e-folding times (grows to ~20x)
    Purely oscillatory (Re≈0): 3 full cycles
    """
    re, im = eigenvalue.real, eigenvalue.imag
    if abs(re) > 1e-4:
        duration = (5.0 if re < 0 else 3.0) / abs(re)
    elif abs(im) > 1e-4:
        duration = 3.0 * (2.0 * np.pi / abs(im))
    else:
        duration = max_s
    return float(np.clip(duration, 0.1, max_s))


def load_bridle_geometry(bridle_yaml_path) -> dict | None:
    """Parse struc_geometry.yaml and return nodes + edge segments.

    Returns a dict with:
      nodes : {id: np.array([x,y,z])}  (id 0 = KCU/attachment point at origin)
      wing_edges   : list of (id_a, id_b) — kite structure
      bridle_edges : list of (id_a, id_b) — bridle lines
      kcu_point    : np.array([x,y,z])   — node 0 (always [0,0,0])
    Returns None if path is not given or does not exist.
    """
    if bridle_yaml_path is None:
        return None
    from pathlib import Path

    path = Path(bridle_yaml_path)
    if not path.exists():
        return None
    try:
        import yaml
    except ImportError:
        return None

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    nodes: dict[int, np.ndarray] = {0: np.zeros(3, dtype=float)}

    def _parse_particles(key: str) -> None:
        table = data.get(key, {})
        for row in table.get("data", []):
            pid, x, y, z = int(row[0]), float(row[1]), float(row[2]), float(row[3])
            nodes[pid] = np.array([x, y, z], dtype=float)

    _parse_particles("wing_particles")
    _parse_particles("bridle_particles")

    def _parse_edges(key: str) -> tuple[list[tuple[int, int]], list[str]]:
        edges: list[tuple[int, int]] = []
        names: list[str] = []
        table = data.get(key, {})
        for row in table.get("data", []):
            name = str(row[0])
            ids = [int(v) for v in row[1:] if v is not None]
            if len(ids) == 2:
                edges.append((ids[0], ids[1]))
                names.append(name)
            elif len(ids) == 3:
                # pulley: line goes ids[1] → ids[0] → ids[2]
                edges.append((ids[1], ids[0]))
                edges.append((ids[0], ids[2]))
                names.extend([name, name])
        return edges, names

    wing_edges, wing_edge_names = _parse_edges("wing_connections")
    bridle_edges, bridle_edge_names = _parse_edges("bridle_connections")

    return {
        "nodes": nodes,
        "wing_edges": wing_edges,
        "wing_edge_names": wing_edge_names,
        "bridle_edges": bridle_edges,
        "bridle_edge_names": bridle_edge_names,
        "kcu_point": nodes[0].copy(),
    }


def _update_bridle_lines(
    bridle_geom,
    wing_struct_lines,
    bridle_lines,
    kcu_marker,
    R: np.ndarray,
    origin: np.ndarray,
) -> None:
    """Rotate and redraw all bridle geometry objects for one animation frame."""
    if bridle_geom is None:
        return
    nodes_rot = {
        nid: (origin + R @ (pos - origin) if nid != 0 else pos)
        for nid, pos in bridle_geom["nodes"].items()
    }
    for ln, (ci, cj) in zip(wing_struct_lines, bridle_geom["wing_edges"]):
        p, q = nodes_rot.get(ci), nodes_rot.get(cj)
        if p is not None and q is not None:
            ln.set_data([p[0], q[0]], [p[1], q[1]])
            ln.set_3d_properties([p[2], q[2]])
    for ln, (ci, cj) in zip(bridle_lines, bridle_geom["bridle_edges"]):
        p, q = nodes_rot.get(ci), nodes_rot.get(cj)
        if p is not None and q is not None:
            ln.set_data([p[0], q[0]], [p[1], q[1]])
            ln.set_3d_properties([p[2], q[2]])
    # KCU stays at origin (node 0 is fixed attachment point, not rotated)
    if kcu_marker is not None:
        kcu_pt = bridle_geom["kcu_point"]
        kcu_marker.set_data([kcu_pt[0]], [kcu_pt[1]])
        kcu_marker.set_3d_properties([kcu_pt[2]])


# Per-state metadata used by the generic animator. `category` drives both the
# 3D rotation contribution and the side-plot units. `trim_key` indexes into
# trim_result["opt_x"] via _trim_baseline_rad below; states with trim_key=None
# are zero at trim (e.g. v, w, p, q).
_STATE_META = {
    "u": {"category": "velocity", "label": "u", "unit": "m/s", "trim_key": "u"},
    "v": {"category": "velocity", "label": "v", "unit": "m/s", "trim_key": None},
    "w": {"category": "velocity", "label": "w", "unit": "m/s", "trim_key": None},
    "z": {"category": "position", "label": "z", "unit": "m", "trim_key": None},
    "phi": {"category": "angle", "label": "phi", "unit": "deg", "trim_key": "phi"},
    "theta": {
        "category": "angle",
        "label": "theta",
        "unit": "deg",
        "trim_key": "theta",
    },
    "psi": {"category": "angle", "label": "psi", "unit": "deg", "trim_key": "psi"},
    "p": {"category": "rate", "label": "p", "unit": "rad/s", "trim_key": None},
    "q": {"category": "rate", "label": "q", "unit": "rad/s", "trim_key": None},
    "r": {"category": "rate", "label": "r", "unit": "rad/s", "trim_key": "r"},
}


def _trim_baseline_rad(trim_result: dict) -> dict[str, float]:
    """Trim values for states that have one. Angles are returned in radians."""
    opt_x = trim_result["opt_x"]
    return {
        "u": float(opt_x[0]),
        "phi": float(np.deg2rad(opt_x[1])),
        "theta": float(np.deg2rad(opt_x[2])),
        "psi": float(np.deg2rad(opt_x[3])),
        "r": float(opt_x[4]),
    }


def _nondim_weights(states: list[str], U_ref: float, L_ref: float) -> np.ndarray:
    """Per-state weight that converts a dimensional eigenvector component to a
    common dimensionless basis.

      velocity  →  v / U_ref               (m/s -> ratio of airspeed)
      angle     →  unchanged               (already in radians)
      rate      →  omega * L_ref / U_ref   (standard aerodynamic non-dim)

    With these weights every component is dimensionless and directly
    comparable: 1.0 means "a swing of one trim airspeed", "one radian",
    or "a rotation that sweeps L_ref in time L_ref/U_ref".
    """
    if U_ref <= 0.0:
        U_ref = 1.0
    if L_ref <= 0.0:
        L_ref = 1.0
    w = np.ones(len(states), dtype=float)
    for i, s in enumerate(states):
        cat = _STATE_META[s]["category"]
        if cat == "velocity":
            w[i] = 1.0 / U_ref
        elif cat == "position":
            w[i] = 1.0 / L_ref
        elif cat == "rate":
            w[i] = L_ref / U_ref
    return w


def _characteristic_length(body_aero) -> float:
    """Half-span of the body's panel geometry, used as L_ref for non-dim rates."""
    try:
        corners = np.array(
            [panel.corner_points for panel in body_aero.panels], dtype=float
        ).reshape(-1, 3)
        half_span = float(np.max(np.abs(corners[:, 1])))
        if half_span > 1e-3:
            return half_span
        max_chord = max(float(panel.chord) for panel in body_aero.panels)
        return max(max_chord, 1.0)
    except Exception:
        return 1.0


def _scale_mode_dimensionless(
    mode_vec: np.ndarray,
    weights: np.ndarray,
    amplitude_rad: float,
) -> np.ndarray:
    """Scale the dimensional eigenvector so its largest dimensionless component
    reaches ``amplitude_rad``.

    Avoids the "u blows up to 50 m/s because the attitude component happened to
    be tiny" artefact of normalising on a single physical quantity.
    """
    nondim_mag = np.abs(mode_vec) * weights
    max_nd = nondim_mag.max() if nondim_mag.size else 0.0
    if max_nd < 1e-12:
        return mode_vec
    return mode_vec * (amplitude_rad / max_nd)


def _pick_dominant_states(
    response: np.ndarray,
    states: list[str],
    *,
    weights: np.ndarray | None = None,
    max_n: int = 4,
    rel_threshold: float = 0.05,
) -> list[int]:
    """Return state indices to plot, ranked by peak |response| (weighted to a
    dimensionless basis when ``weights`` is given) above a relative cutoff."""
    if response.size == 0:
        return []
    peaks = np.max(np.abs(response), axis=1)
    if weights is not None:
        peaks = peaks * np.asarray(weights)
    if peaks.max() < 1e-12:
        return []
    cutoff = rel_threshold * peaks.max()
    ranked = sorted(
        (i for i in range(len(states)) if peaks[i] >= cutoff),
        key=lambda i: peaks[i],
        reverse=True,
    )
    return ranked[:max_n]


def animate_eigenmode(
    body_aero,
    trim_result: dict,
    eig: complex,
    mode_vec: np.ndarray,
    states: list[str],
    mode_index: int,
    *,
    block_title: str,
    out_path,
    fps: int = 8,
    n_frames: int = 60,
    amplitude_rad: float = np.deg2rad(5.0),
    max_physics_s: float = 30.0,
    fmt: str = "gif",
    dpi: int | None = None,
    reference_point: np.ndarray | None = None,
    bridle_geom: dict | None = None,
) -> None:
    """Animate one eigenmode for an arbitrary state selection.

    The 3D body rotation is composed from whichever of ``phi``, ``theta``, ``psi``
    are in ``states`` (others stay at trim). The side panel shows the time
    response of only the most relevant states for this mode (peak |component|
    >= 15 % of the mode's maximum, capped at 4 plots).
    """
    states = list(states)
    mode_vec = np.asarray(mode_vec, dtype=complex).copy()

    # Non-dimensionalisation reference values. U_ref is the trim apparent-wind
    # magnitude; L_ref is the body half-span. Used to put velocities, angles
    # and rates on a common dimensionless basis for both mode-vector scaling
    # and mode-shape comparison.
    U_ref = float(trim_result.get("Umag", trim_result["opt_x"][0]))
    if U_ref <= 0.0:
        U_ref = 1.0
    L_ref = _characteristic_length(body_aero)
    weights = _nondim_weights(states, U_ref, L_ref)
    mode_vec = _scale_mode_dimensionless(mode_vec, weights, amplitude_rad)

    t_phys_end = _physics_duration(eig, max_s=max_physics_s)
    t_phys = np.linspace(0.0, t_phys_end, n_frames)
    response = _mode_time_response(eig, mode_vec, t_phys, 1.0)  # (n_states, n_frames)

    trim_rad = _trim_baseline_rad(trim_result)
    state_to_idx = {s: i for i, s in enumerate(states)}
    dominant_indices = _pick_dominant_states(response, states, weights=weights)
    n_plots = max(len(dominant_indices), 1)

    panel_corners = np.array(
        [panel.corner_points for panel in body_aero.panels], dtype=float
    )
    origin = np.asarray(
        reference_point if reference_point is not None else [0.0, 0.0, 0.0],
        dtype=float,
    )

    fig = plt.figure(figsize=(15, max(6.0, 2.0 * n_plots + 1.5)))
    gs = fig.add_gridspec(n_plots, 3, width_ratios=[1.3, 0.55, 1.0])
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax_bars = fig.add_subplot(gs[:, 1])

    stab_char = "stable" if eig.real < 0 else "UNSTABLE"
    freq = abs(eig.imag) / (2 * np.pi)
    ax3d.set_title(
        f"{block_title} mode {mode_index}  [{stab_char}]\n"
        f"states={states}\n"
        f"λ = {eig.real:+.3f}{eig.imag:+.3f}j  f={freq:.3f} Hz",
        fontsize=8,
    )
    ax3d.set_xlabel("x [m]")
    ax3d.set_ylabel("y [m]")
    ax3d.set_zlabel("z [m]")
    ax3d.grid(True, alpha=0.25)

    all_pts = panel_corners.reshape(-1, 3)
    ctr = np.mean(all_pts, axis=0)
    hr = max(0.6 * np.max(np.ptp(all_pts, axis=0)), 1.0)
    ax3d.set_xlim(ctr[0] - hr, ctr[0] + hr)
    ax3d.set_ylim(ctr[1] - hr, ctr[1] + hr)
    ax3d.set_zlim(ctr[2] - hr, ctr[2] + hr)
    ax3d.view_init(elev=20, azim=-120)

    panel_lines = [
        ax3d.plot([], [], [], color="tab:blue", linewidth=1.1)[0]
        for _ in range(panel_corners.shape[0])
    ]

    wing_struct_lines: list = []
    bridle_lines: list = []
    kcu_marker = None
    if bridle_geom is not None:
        for _ in bridle_geom["wing_edges"]:
            (ln,) = ax3d.plot([], [], [], color="0.55", linewidth=0.7, zorder=1)
            wing_struct_lines.append(ln)
        for _ in bridle_geom["bridle_edges"]:
            (ln,) = ax3d.plot(
                [], [], [], color="tab:orange", linewidth=0.6, linestyle="--", zorder=1
            )
            bridle_lines.append(ln)
        kcu_pt = bridle_geom["kcu_point"]
        (kcu_marker,) = ax3d.plot(
            [kcu_pt[0]],
            [kcu_pt[1]],
            [kcu_pt[2]],
            "D",
            color="tab:red",
            markersize=5,
            zorder=6,
        )

    status_txt = ax3d.text2D(0.02, 0.96, "", transform=ax3d.transAxes, fontsize=8)

    # Mode-shape bar chart: per state in canonical order, drawn from the
    # *dimensionless* eigenvector so velocities (m/s) don't dwarf attitudes
    # (rad) just because of units. Each component is multiplied by its
    # non-dim weight, then normalised by the largest dimensionless component.
    # Bars below the relevance threshold are drawn faded so the user can see
    # all components at once. For complex modes the phase of each component
    # (deg) is annotated next to the bar.
    _category_color = {
        "velocity": "tab:blue",
        "position": "tab:purple",
        "angle": "tab:orange",
        "rate": "tab:green",
    }
    rel_threshold = 0.15
    nondim_mag = np.abs(mode_vec) * weights
    norm = nondim_mag.max() if nondim_mag.size else 1.0
    if norm < 1e-12:
        norm = 1.0
    rel_mag = nondim_mag / norm
    is_complex_mode = bool(np.any(np.abs(mode_vec.imag) > 1e-9))
    phases_deg = np.rad2deg(np.angle(mode_vec))

    y_positions = np.arange(len(states))
    for k, s in enumerate(states):
        meta = _STATE_META[s]
        color = _category_color[meta["category"]]
        alpha = 1.0 if rel_mag[k] >= rel_threshold else 0.35
        ax_bars.barh(y_positions[k], rel_mag[k], color=color, alpha=alpha, height=0.7)
        annotation = f"{rel_mag[k] * 100:.0f}%"
        if is_complex_mode:
            annotation += f"  ∠{phases_deg[k]:+.0f}°"
        ax_bars.text(
            min(rel_mag[k] + 0.03, 1.02),
            y_positions[k],
            annotation,
            va="center",
            fontsize=7,
            color="0.25" if alpha == 1.0 else "0.55",
        )

    ax_bars.set_yticks(y_positions)
    ax_bars.set_yticklabels([_STATE_META[s]["label"] for s in states], fontsize=8)
    ax_bars.invert_yaxis()
    ax_bars.set_xlim(0.0, 1.25)
    ax_bars.set_xticks([0.0, 0.5, 1.0])
    ax_bars.set_xlabel("|v|·w / max(|v|·w)", fontsize=8)
    ax_bars.set_title(
        f"Mode shape (non-dim)\nU_ref={U_ref:.1f} m/s, L_ref={L_ref:.2f} m",
        fontsize=8,
    )
    ax_bars.axvline(rel_threshold, color="0.5", linestyle="--", linewidth=0.7)
    ax_bars.tick_params(axis="x", labelsize=7)
    ax_bars.grid(True, axis="x", alpha=0.25)

    # Build a side plot per dominant state. Plot trim+perturbation; convert
    # angle states to degrees for display.
    side_axes: list[tuple] = (
        []
    )  # (ax, state_idx, total_series, unit, label, line, marker)
    plot_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for k, sidx in enumerate(dominant_indices):
        state = states[sidx]
        meta = _STATE_META[state]
        perturb = np.real(response[sidx, :])
        trim_val = trim_rad[meta["trim_key"]] if meta["trim_key"] is not None else 0.0
        total = trim_val + perturb
        if meta["category"] == "angle":
            display_series = np.rad2deg(total)
        else:
            display_series = total
        ax = fig.add_subplot(gs[k, 2])
        color = plot_colors[k % len(plot_colors)]
        ax.set_title(f"{meta['label']}(t)", fontsize=9)
        ax.set_xlabel("t [s]", fontsize=8)
        ax.set_ylabel(f"{meta['label']} [{meta['unit']}]", fontsize=8)
        ax.plot(t_phys, display_series, color="0.7", linewidth=1)
        (line,) = ax.plot([], [], color=color, linewidth=2)
        (marker,) = ax.plot([], [], "o", color=color)
        ax.set_xlim(0.0, t_phys_end)
        pad = max(0.05 * np.ptp(display_series), 0.1)
        ax.set_ylim(display_series.min() - pad, display_series.max() + pad)
        ax.grid(True, alpha=0.3)
        side_axes.append(
            (ax, sidx, display_series, meta["unit"], meta["label"], line, marker)
        )

    if not dominant_indices:
        ax = fig.add_subplot(gs[0, 2])
        ax.text(
            0.5,
            0.5,
            "No state in this mode exceeds the\nrelevance threshold.",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    extra = [*wing_struct_lines, *bridle_lines] + ([kcu_marker] if kcu_marker else [])
    animated_lines = [side[5] for side in side_axes]
    animated_markers = [side[6] for side in side_axes]

    def _attitude_perturb_rad(fi: int) -> tuple[float, float, float]:
        """Per-frame attitude offsets (roll, pitch, yaw) in radians."""
        phi_p = (
            float(np.real(response[state_to_idx["phi"], fi]))
            if "phi" in state_to_idx
            else 0.0
        )
        theta_p = (
            float(np.real(response[state_to_idx["theta"], fi]))
            if "theta" in state_to_idx
            else 0.0
        )
        psi_p = (
            float(np.real(response[state_to_idx["psi"], fi]))
            if "psi" in state_to_idx
            else 0.0
        )
        return phi_p, theta_p, psi_p

    def init():
        for ln in panel_lines:
            ln.set_data([], [])
            ln.set_3d_properties([])
        for line in animated_lines:
            line.set_data([], [])
        for marker in animated_markers:
            marker.set_data([], [])
        return [*panel_lines, *extra, status_txt, *animated_lines, *animated_markers]

    def update(fi: int):
        phi_p, theta_p, psi_p = _attitude_perturb_rad(fi)
        phi_t = trim_rad["phi"] + phi_p
        theta_t = trim_rad["theta"] + theta_p
        psi_t = trim_rad["psi"] + psi_p
        R = (
            _rot_rad(DEFAULT_AXES.radial, psi_t)
            @ _rot_rad(DEFAULT_AXES.normal, theta_t)
            @ _rot_rad(DEFAULT_AXES.course, phi_t)
        )
        rot = _rotate_pts(panel_corners, R, origin)
        for idx, ln in enumerate(panel_lines):
            c = np.vstack([rot[idx], rot[idx][0]])
            ln.set_data(c[:, 0], c[:, 1])
            ln.set_3d_properties(c[:, 2])
        _update_bridle_lines(
            bridle_geom, wing_struct_lines, bridle_lines, kcu_marker, R, origin
        )
        # Status line shows the dominant state's current value.
        if dominant_indices:
            parts = []
            for _, sidx, series, unit, label, _line, _marker in side_axes:
                parts.append(f"{label}={series[fi]:+.2f} {unit}")
            status_txt.set_text(f"t={t_phys[fi]:.3f} s | " + "  ".join(parts))
        else:
            status_txt.set_text(f"t={t_phys[fi]:.3f} s")
        for _, sidx, series, _unit, _label, line, marker in side_axes:
            line.set_data(t_phys[: fi + 1], series[: fi + 1])
            marker.set_data([t_phys[fi]], [series[fi]])
        return [*panel_lines, *extra, status_txt, *animated_lines, *animated_markers]

    anim = FuncAnimation(
        fig, update, init_func=init, frames=n_frames, interval=1000.0 / fps, blit=False
    )
    fig.tight_layout()
    _save_animation(anim, out_path, fps=fps, fmt=fmt, dpi=dpi)
    plt.close(fig)


def animate_all_eigenmodes_clean(
    body_aero,
    trim_result: dict,
    mode_blocks: list[tuple[str, str, np.ndarray, np.ndarray, list[str]]],
    *,
    out_path,
    fps: int = 8,
    amplitude_rad: float = np.deg2rad(5.0),
    max_physics_s: float = 30.0,
    fmt: str = "gif",
    dpi: int | None = None,
    reference_point: np.ndarray | None = None,
    bridle_geom: dict | None = None,
    mode_gap_s: float = 0.25,
) -> None:
    """Save one clean animation with all modes played sequentially.

    The playback is in physical time: a mode shown for ``t`` seconds in the
    linear response is written as ``t * fps`` frames in the output animation.
    """
    panel_corners = np.array(
        [panel.corner_points for panel in body_aero.panels], dtype=float
    )
    origin = np.asarray(
        reference_point if reference_point is not None else [0.0, 0.0, 0.0],
        dtype=float,
    )
    trim_rad = _trim_baseline_rad(trim_result)
    U_ref = float(trim_result.get("Umag", trim_result["opt_x"][0]))
    if U_ref <= 0.0:
        U_ref = 1.0
    L_ref = _characteristic_length(body_aero)

    frames: list[tuple[complex, float, float, float]] = []
    for _block_title, _slug, eigvals, eigvecs, block_states in mode_blocks:
        states = list(block_states)
        weights = _nondim_weights(states, U_ref, L_ref)
        state_to_idx = {s: i for i, s in enumerate(states)}
        for i, eig in enumerate(np.asarray(eigvals, dtype=complex)):
            mode_vec = np.asarray(eigvecs[:, i], dtype=complex).copy()
            mode_vec = _scale_mode_dimensionless(mode_vec, weights, amplitude_rad)
            t_end = _physics_duration(eig, max_s=max_physics_s)
            n_mode_frames = max(2, int(np.ceil(t_end * fps)))
            t_phys = np.arange(n_mode_frames, dtype=float) / float(fps)
            t_phys = np.minimum(t_phys, t_end)
            response = _mode_time_response(eig, mode_vec, t_phys, 1.0)

            for fi in range(n_mode_frames):
                phi_p = (
                    float(np.real(response[state_to_idx["phi"], fi]))
                    if "phi" in state_to_idx
                    else 0.0
                )
                theta_p = (
                    float(np.real(response[state_to_idx["theta"], fi]))
                    if "theta" in state_to_idx
                    else 0.0
                )
                psi_p = (
                    float(np.real(response[state_to_idx["psi"], fi]))
                    if "psi" in state_to_idx
                    else 0.0
                )
                frames.append(
                    (
                        eig,
                        trim_rad["phi"] + phi_p,
                        trim_rad["theta"] + theta_p,
                        trim_rad["psi"] + psi_p,
                    )
                )

            if mode_gap_s > 0.0 and frames:
                gap_frames = int(np.ceil(mode_gap_s * fps))
                frames.extend([frames[-1]] * gap_frames)

    if not frames:
        print("No modes found for combined animation.")
        return

    fig = plt.figure(figsize=(8, 8), facecolor="white", dpi=dpi or 180)
    ax3d = fig.add_subplot(111, projection="3d")
    ax3d.set_facecolor("white")
    ax3d.set_axis_off()
    ax3d.grid(False)
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)

    all_pts = panel_corners.reshape(-1, 3)
    if bridle_geom is not None:
        all_pts = np.vstack([all_pts, np.array(list(bridle_geom["nodes"].values()))])
    ctr = np.mean(all_pts, axis=0)
    hr = max(0.58 * np.max(np.ptp(all_pts, axis=0)), 1.0)
    ax3d.set_xlim(ctr[0] - hr, ctr[0] + hr)
    ax3d.set_ylim(ctr[1] - hr, ctr[1] + hr)
    ax3d.set_zlim(ctr[2] - hr, ctr[2] + hr)
    ax3d.set_box_aspect((1, 1, 1))
    ax3d.view_init(elev=20, azim=-120)

    panel_surfaces = Poly3DCollection(
        [panel for panel in panel_corners],
        facecolor=(0.82, 0.82, 0.82, 0.38),
        edgecolor="none",
        linewidth=0.0,
        zorder=1,
    )
    ax3d.add_collection3d(panel_surfaces)

    panel_lines = [
        ax3d.plot([], [], [], color="black", alpha=0.55, linewidth=0.45, zorder=4)[0]
        for _ in range(panel_corners.shape[0])
    ]

    wing_struct_lines: list = []
    bridle_lines: list = []
    kcu_marker = None
    if bridle_geom is not None:
        for name in bridle_geom["wing_edge_names"]:
            is_inflatable = name.startswith("le_") or name.startswith("strut_")
            (ln,) = ax3d.plot(
                [],
                [],
                [],
                color="black",
                alpha=0.98 if is_inflatable else 0.5,
                linewidth=2.4 if is_inflatable else 0.55,
                zorder=5 if is_inflatable else 3,
            )
            wing_struct_lines.append(ln)
        for _ in bridle_geom["bridle_edges"]:
            (ln,) = ax3d.plot(
                [], [], [], color="black", alpha=0.22, linewidth=0.45, zorder=2
            )
            bridle_lines.append(ln)

    extra = [panel_surfaces, *wing_struct_lines, *bridle_lines] + (
        [kcu_marker] if kcu_marker else []
    )

    def init():
        for ln in panel_lines:
            ln.set_data([], [])
            ln.set_3d_properties([])
        return [*panel_lines, *extra]

    def update(fi: int):
        _eig, phi_t, theta_t, psi_t = frames[fi]
        R = (
            _rot_rad(DEFAULT_AXES.radial, psi_t)
            @ _rot_rad(DEFAULT_AXES.normal, theta_t)
            @ _rot_rad(DEFAULT_AXES.course, phi_t)
        )
        rot = _rotate_pts(panel_corners, R, origin)
        panel_surfaces.set_verts([panel for panel in rot])
        for idx, ln in enumerate(panel_lines):
            c = np.vstack([rot[idx], rot[idx][0]])
            ln.set_data(c[:, 0], c[:, 1])
            ln.set_3d_properties(c[:, 2])
        _update_bridle_lines(
            bridle_geom, wing_struct_lines, bridle_lines, kcu_marker, R, origin
        )
        return [*panel_lines, *extra]

    if fmt == "mp4" and not writers.is_available("ffmpeg"):
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "MP4 export requires either ffmpeg or OpenCV (`cv2`)."
            ) from exc

        init()
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video = cv2.VideoWriter(str(out_path), fourcc, float(fps), (width, height))
        if not video.isOpened():
            raise RuntimeError(f"Could not open MP4 writer for {out_path}")
        try:
            for fi in range(len(frames)):
                update(fi)
                fig.canvas.draw()
                rgba = np.asarray(fig.canvas.buffer_rgba())
                bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
                video.write(bgr)
        finally:
            video.release()
            plt.close(fig)
        print(f"Wrote {out_path}")
        return

    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(frames),
        interval=1000.0 / fps,
        blit=False,
    )
    _save_animation(anim, out_path, fps=fps, fmt=fmt, dpi=dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Solve VSM aerodynamic trim and compute stability derivatives."
    )
    add_common_arguments(parser)
    # Apply the script-level operating condition as defaults (CLI still overrides).
    parser.set_defaults(**OPERATING_CONDITION)
    parser.add_argument(
        "--deformed-case",
        default=None,
        help=(
            "Name of a result case folder under --deformed-root "
            "(e.g. depower_p0000mm_steer_p0200mm). Uses that case's deformed "
            "aero_geometry.yaml/struc_geometry.yaml for the trim (shortcut for "
            "--deformed-from). Mass/inertia/CoG/tether still come from "
            "system.yaml in --config-folder."
        ),
    )
    parser.add_argument(
        "--deformed-root",
        default=None,
        help=(
            "Directory holding the deformed result cases "
            "(default: data/<kite>/deformed_results when present; otherwise "
            "results/<kite>/aerostructural)."
        ),
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="List available deformed cases under --deformed-root and exit.",
    )
    parser.add_argument("--eps-vel", type=float, default=0.1)
    parser.add_argument("--eps-angle-deg", type=float, default=0.5)
    parser.add_argument("--eps-rate", type=float, default=0.01)
    parser.add_argument(
        "--eps-position",
        type=float,
        default=0.5,
        help="Finite-difference step [m] for radial position state `z`.",
    )
    parser.add_argument(
        "--no-animate",
        action="store_true",
        help="Skip saving mode animations.",
    )
    parser.add_argument(
        "--animation-format",
        choices=["gif", "mp4"],
        default="mp4",
        help="Animation output format (default: mp4; mp4 requires ffmpeg).",
    )
    parser.add_argument(
        "--animation-fps",
        type=int,
        default=8,
        help="Frames per second for saved animations (default: 8).",
    )
    parser.add_argument(
        "--animation-frames",
        type=int,
        default=60,
        help="Fixed number of frames per animation (default: 60).",
    )
    parser.add_argument(
        "--animation-dpi",
        type=int,
        default=180,
        help="Export resolution for saved animations (default: 180 dpi).",
    )
    parser.add_argument(
        "--animation-amplitude-deg",
        type=float,
        default=5.0,
        help="Initial perturbation amplitude in degrees.",
    )
    parser.add_argument(
        "--animation-max-physics-s",
        type=float,
        default=30.0,
        help="Cap on real physics time shown per animation (default: 30 s).",
    )
    parser.add_argument(
        "--combined-animation-only",
        action="store_true",
        help=(
            "Save only the clean combined modes animation, skipping the "
            "individual per-mode animations."
        ),
    )
    parser.add_argument(
        "--combined-animation-gap-s",
        type=float,
        default=0.25,
        help="Pause between modes in the combined animation (default: 0.25 s).",
    )
    parser.add_argument(
        "--states",
        type=str,
        default=None,
        help=(
            "Comma-separated stability states from "
            f"{{{', '.join(ALL_STATE_NAMES)}}}. "
            "Defaults to the 9-state rigid-body set "
            f"{','.join(list(LONG_STATES) + list(LAT_STATES))} "
            "(the lateral velocity `v` and positions `x`, `y` are held fixed). "
            "Use 'all' to select every state. Overrides --stability-config."
        ),
    )
    parser.add_argument(
        "--include-w",
        action="store_true",
        help="Convenience: add `w` (vertical body speed) to the default state set.",
    )
    parser.add_argument(
        "--coupled",
        action="store_true",
        help=(
            "Assemble a single coupled A matrix from the selected states "
            "(default: split into longitudinal + lateral sub-blocks). "
            "Overrides --stability-config."
        ),
    )
    parser.add_argument(
        "--stability-config",
        type=Path,
        default=None,
        help=(
            "Optional YAML file with `states:` (list), `coupled:` (bool), "
            "and `frame:` ('course' or 'body') keys. CLI flags "
            "(--states / --coupled / --include-w / --stability-frame) override "
            "this file when both are provided."
        ),
    )
    parser.add_argument(
        "--stability-frame",
        choices=["course", "body"],
        default=None,
        help=(
            "Frame used for stability perturbations and force/moment outputs. "
            "`course` uses course/normal/radial axes (default). `body` uses "
            "identified rigid-body principal axes and requires --rigid-body-result."
        ),
    )
    parser.add_argument(
        "--rigid-body-result",
        type=Path,
        default=None,
        help=(
            "Path to a structural result directory (or sim_output.h5). "
            "Loads identified body axes, CG, and inertia from the PSM model, "
            "overriding --center-of-gravity and --inertia-xx/yy/zz."
        ),
    )
    parser.add_argument(
        "--rigid-body-struc",
        type=Path,
        default=None,
        help="struc_geometry.yaml to use with --rigid-body-result (auto-detected if omitted).",
    )
    args = parser.parse_args()

    # Resolve the deformed-results case selection (--deformed-case / --list-cases).
    kite_name = Path(args.config_folder).name
    data_deformed_root = Path(args.config_folder) / "deformed_results"
    deformed_root = (
        Path(args.deformed_root)
        if args.deformed_root
        else (
            data_deformed_root
            if data_deformed_root.is_dir()
            else DEFAULT_OUTPUT_ROOT.parent / kite_name / "aerostructural"
        )
    )
    if args.list_cases:
        print(f"Deformed-result cases under {deformed_root}:")
        if deformed_root.is_dir():
            cases = sorted(
                d.name
                for d in deformed_root.iterdir()
                if d.is_dir() and (d / "aero_geometry.yaml").exists()
            )
            for name in cases:
                print(f"  {name}")
            if not cases:
                print("  (none found)")
        else:
            print("  (directory does not exist)")
        return
    if args.deformed_case:
        case_dir = deformed_root / args.deformed_case
        if not case_dir.is_dir():
            parser.error(
                f"--deformed-case '{args.deformed_case}' not found under {deformed_root}. "
                "Use --list-cases to see available cases."
            )
        # build_body reads args.deformed_from to use the frozen deformed geometry.
        args.deformed_from = str(case_dir)
        print(f"Using deformed geometry from: {case_dir}")
    elif args.deformed_from is None:
        powered_case = deformed_root / "powered_2019"
        if powered_case.is_dir():
            args.deformed_from = str(powered_case)
            print(f"Using default powered deformed geometry from: {powered_case}")

    values = parsed_common(args)
    out_dir = output_dir(args, "stability_derivatives")

    if args.stability_config is None:
        default_cfg = Path(args.config_folder) / "stability_config.yaml"
        if not default_cfg.exists():
            default_cfg = Path(args.config_folder) / "stability_config.yml"
        if default_cfg.exists():
            args.stability_config = default_cfg

    # Resolve user-selected stability states. Precedence:
    #   1. --states CLI flag
    #   2. --include-w shortcut
    #   3. --stability-config YAML
    #   4. historical default (no `w`)
    # --coupled CLI flag overrides YAML when set; otherwise YAML wins.
    cfg_states: list[str] | None = None
    cfg_coupled: bool | None = None
    cfg_frame: str | None = None
    stability_config_path: Path | None = None
    if args.stability_config is not None:
        stability_config_path = args.stability_config.resolve()
        with stability_config_path.open("r", encoding="utf-8") as f:
            stab_cfg = yaml.safe_load(f) or {}
        if not isinstance(stab_cfg, dict):
            parser.error(
                f"--stability-config {args.stability_config} must be a mapping; "
                f"got {type(stab_cfg).__name__}."
            )
        raw_states = stab_cfg.get("states")
        if raw_states is not None:
            if isinstance(raw_states, str):
                if raw_states.strip().lower() == "all":
                    cfg_states = list(ALL_STATE_NAMES)
                else:
                    cfg_states = [s.strip() for s in raw_states.split(",") if s.strip()]
            elif isinstance(raw_states, (list, tuple)):
                cfg_states = [str(s).strip() for s in raw_states]
            else:
                parser.error(
                    "stability-config `states` must be a list or comma-string."
                )
        if "coupled" in stab_cfg:
            cfg_coupled = bool(stab_cfg["coupled"])
        if "frame" in stab_cfg:
            cfg_frame = str(stab_cfg["frame"]).strip().lower()
            if cfg_frame not in {"course", "body"}:
                parser.error(
                    "stability-config `frame` must be either 'course' or 'body'."
                )

    # Default 9-state rigid-body set: the full state minus the lateral velocity
    # ``v`` -> longitudinal (u, w, z, theta, q) + lateral (phi, psi, p, r).
    default_states = list(LONG_STATES) + list(LAT_STATES)

    if args.states is not None:
        if args.states.strip().lower() == "all":
            sel_states: list[str] = list(ALL_STATE_NAMES)
        else:
            sel_states = [s.strip() for s in args.states.split(",") if s.strip()]
    elif args.include_w:
        sel_states = list(default_states)
    elif cfg_states is not None:
        sel_states = list(cfg_states)
    else:
        sel_states = list(default_states)

    unknown = [s for s in sel_states if s not in ALL_STATE_NAMES]
    if unknown:
        parser.error(f"Unknown state names {unknown}. Valid: {list(ALL_STATE_NAMES)}")

    coupled = args.coupled or (cfg_coupled if cfg_coupled is not None else False)
    stability_frame = args.stability_frame or cfg_frame or "course"
    if stability_frame == "body" and args.rigid_body_result is None:
        parser.error("--stability-frame body requires --rigid-body-result.")

    if stability_config_path is not None:
        print(f"Loaded stability config: {stability_config_path}")
    print(f"Output directory: {out_dir.resolve()}")
    print(f"Resolved stability states: {sel_states}")
    print(f"Resolved coupled: {coupled}")

    # Load body and properties from config folder
    body, props = build_body(args)

    mass_wing = (
        args.mass_wing if args.mass_wing is not None else props.get("mass", 30.0)
    )

    # Inertia: use from system.yaml unless explicitly overridden
    inertia_tensor = props.get("inertia", [[100, 0, 0], [0, 20, 0], [0, 0, 100]])
    try:
        inertia_xx = (
            float(inertia_tensor[0][0]) if args.inertia_xx is None else args.inertia_xx
        )
        inertia_yy = (
            float(inertia_tensor[1][1]) if args.inertia_yy is None else args.inertia_yy
        )
        inertia_zz = (
            float(inertia_tensor[2][2]) if args.inertia_zz is None else args.inertia_zz
        )
    except (IndexError, TypeError):
        inertia_xx = args.inertia_xx if args.inertia_xx is not None else 100.0
        inertia_yy = args.inertia_yy if args.inertia_yy is not None else 20.0
        inertia_zz = args.inertia_zz if args.inertia_zz is not None else 100.0

    center_of_gravity = values["center_of_gravity"]
    trim_axes = DEFAULT_AXES
    stability_axes = DEFAULT_AXES
    rigid_body_axes = None
    deformed_aero_path: Path | None = None  # set below when deformed geometry is found
    deformed_struc_path: Path | None = None  # set below when deformed geometry is found

    # Optionally load identified rigid-body properties and deformed geometry.
    # The stability frame is selected explicitly below; loading a rigid-body
    # result no longer implies body-frame derivatives.
    if args.rigid_body_result is not None:
        from common import _resolve_csv_paths, add_vsm_path as _add_vsm_path
        import tempfile

        rb = _load_rigid_body_axes_from_result(
            args.rigid_body_result, args.rigid_body_struc
        )
        rigid_body_axes = AxisDefinition(
            course=rb.body_axes[0],  # x_body - roll
            normal=rb.body_axes[1],  # y_body - pitch
            radial=rb.body_axes[2],  # z_body - yaw
        )
        center_of_gravity = rb.cg
        inertia_xx, inertia_yy, inertia_zz = rb.principal_moments

        case_dir = args.rigid_body_result.resolve()
        if case_dir.is_file():
            case_dir = case_dir.parent

        _aero_candidate = case_dir / "aero_geometry.yaml"
        _struc_candidate = case_dir / "struc_geometry.yaml"

        if _aero_candidate.exists():
            deformed_aero_path = _aero_candidate
            _add_vsm_path(args.vsm_src)
            from VSM.core.BodyAerodynamics import BodyAerodynamics as _BA

            with deformed_aero_path.open("r", encoding="utf-8") as _f:
                aero_cfg = yaml.safe_load(_f)
            _resolve_csv_paths(aero_cfg, Path(args.config_folder))

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False, encoding="utf-8"
            ) as _tmp:
                yaml.dump(aero_cfg, _tmp)
                _tmp_path = _tmp.name
            try:
                body = _BA.instantiate(
                    n_panels=args.n_panels,
                    file_path=_tmp_path,
                    spanwise_panel_distribution=args.spanwise_panel_distribution,
                    bridle_path=props.get("struc_geometry_path"),
                )
            finally:
                Path(_tmp_path).unlink()
            print(f"VSM body rebuilt from deformed aero_geometry: {deformed_aero_path}")
        else:
            print(
                f"Warning: no deformed aero_geometry.yaml in {case_dir}. "
                "Run with is_save_geometry_snapshots: true to save it. "
                "Falling back to data/ geometry."
            )

        if _struc_candidate.exists():
            deformed_struc_path = _struc_candidate

        print(f"\nRigid-body axes loaded from: {args.rigid_body_result}")
        print(f"  CG (structural frame):  {rb.cg}")
        print(f"  Inertia [Ix, Iy, Iz]:   {rb.principal_moments}")
        print(f"  x_body (roll):   {rb.body_axes[0]}")
        print(f"  y_body (pitch):  {rb.body_axes[1]}")
        print(f"  z_body (yaw):    {rb.body_axes[2]}\n")

    if stability_frame == "body":
        stability_axes = rigid_body_axes
    else:
        stability_axes = DEFAULT_AXES

    print(f"Stability derivative frame: {stability_frame}")
    print(f"  trim axes:      course")
    print(f"  stability axes: {stability_frame}")

    system_model = build_system_model(args, mass_wing=mass_wing)
    # Robust Williams detection: ``isinstance`` can miss it when ``awetrim`` is
    # importable via two paths (the src path injected by common.py and an
    # installed copy), giving two distinct class objects.
    _tether = getattr(system_model, "tether", None)
    use_williams = (
        isinstance(_tether, WilliamsTether)
        or type(_tether).__name__ == "WilliamsTether"
    )
    if use_williams:
        print("Tether model: WilliamsTether -> running joint trim+tether solve.")
        result, solved_body = solve_vsm_qs_trim_with_williams_tether(
            body_aero=body,
            center_of_gravity=center_of_gravity,
            reference_point=values["reference_point"],
            system_model=system_model,
            x_guess=values["x_guess"],
            bounds_lower=values["bounds_lower"],
            bounds_upper=values["bounds_upper"],
            include_gravity=args.include_gravity,
            moment_tolerance=args.moment_tolerance,
            max_nfev=args.max_nfev,
            axes=trim_axes,
        )
    else:
        result, solved_body = solve_vsm_quasi_steady_trim(
            body_aero=body,
            center_of_gravity=center_of_gravity,
            reference_point=values["reference_point"],
            system_model=system_model,
            x_guess=values["x_guess"],
            bounds_lower=values["bounds_lower"],
            bounds_upper=values["bounds_upper"],
            include_gravity=args.include_gravity,
            moment_tolerance=args.moment_tolerance,
            return_timing_breakdown=True,
            max_nfev=args.max_nfev,
            axes=trim_axes,
        )
    print_trim_summary(result)

    stability = compute_vsm_trim_stability_derivatives(
        body_aero=solved_body,
        center_of_gravity=center_of_gravity,
        reference_point=values["reference_point"],
        x_trim=np.asarray(result["opt_x"], dtype=float),
        trim_result=result,
        system_model=system_model,
        mass=mass_wing,
        inertia_xx=inertia_xx,
        inertia_yy=inertia_yy,
        inertia_zz=inertia_zz,
        axes=stability_axes,
        distance_radial=args.distance_radial,
        eps_vel=args.eps_vel,
        eps_angle_deg=args.eps_angle_deg,
        eps_rate=args.eps_rate,
        eps_position=args.eps_position,
        states=sel_states,
        coupled=coupled,
    )

    # --- Diagnostic: is the Williams radial-position dependency captured? ----
    print("\n--- radial-position (z) dependency diagnostic ---")
    print(
        "tether_radial_position_model:",
        stability.get("tether_radial_position_model"),
    )
    print(f"  use_williams (this run): {use_williams}")
    print(
        "  actual tether class:    "
        f"{type(_tether).__name__}  (module={type(_tether).__module__})"
    )
    print(f"  config_folder:          {args.config_folder}")
    print(f"  eps_position requested:  {args.eps_position:g} m")
    print(f"  eps_position used:       {stability.get('eps_position_used'):g} m")
    J_full = np.asarray(stability["J_full"], dtype=float)
    z_col = J_full[:, list(ALL_STATE_NAMES).index("z")]
    out_names = stability.get("output_names", [])
    print("  J[:, z]  (force/moment sensitivity to radial distance):")
    for name, val in zip(out_names, z_col):
        print(f"    d{name}/dz = {val:+.6e}")
    print(f"  ||J[:, z]|| = {np.linalg.norm(z_col):.6e}")
    if not coupled and "A_selected_long" in stability:
        sel_long = list(stability.get("states_selected_long", []))
        if "z" in sel_long and "w" in sel_long:
            A_long = np.asarray(stability["A_selected_long"], dtype=float)
            wz = A_long[sel_long.index("w"), sel_long.index("z")]
            print(f"  A_long[w, z] (radial stiffness / mass): {wz:+.6e} 1/s^2")
    print("  (near-zero ||J[:, z]|| => radial dependency NOT captured)")
    print("--- end diagnostic ---\n")

    print(f"\nSelected states: {sel_states}  (coupled={coupled})")
    if coupled:
        print("A_selected:")
        print(np.array2string(stability["A_selected"], precision=6))
        print(
            "eig_selected:",
            np.array2string(stability["eig_selected"], precision=6),
        )
        print("stable_selected:", stability["stable_selected"])
    else:
        sel_long = stability.get("states_selected_long", [])
        sel_lat = stability.get("states_selected_lat", [])
        print(f"  longitudinal states: {sel_long}")
        print(f"  lateral states:      {sel_lat}")
        if sel_long:
            print("A_selected_long:")
            print(np.array2string(stability["A_selected_long"], precision=6))
            print(
                "eig_selected_long:",
                np.array2string(stability["eig_selected_long"], precision=6),
            )
            print("stable_selected_long:", stability["stable_selected_long"])
        if sel_lat:
            print("A_selected_lat:")
            print(np.array2string(stability["A_selected_lat"], precision=6))
            print(
                "eig_selected_lat:",
                np.array2string(stability["eig_selected_lat"], precision=6),
            )
            print("stable_selected_lat:", stability["stable_selected_lat"])

    # Historical default split — always printed for reference.
    print(
        "\nDefault decoupled blocks (states_long=[u,theta,q], states_lat=[phi,psi,p,r]):"
    )
    print("J_long:")
    print(np.array2string(stability["J_long"], precision=6))
    print("J_lat:")
    print(np.array2string(stability["J_lat"], precision=6))
    print("eig_long:", np.array2string(stability["eig_long"], precision=6))
    print("eig_lat:", np.array2string(stability["eig_lat"], precision=6))
    print("stable_long:", stability["stable_long"])
    print("stable_lat:", stability["stable_lat"])

    write_json(
        out_dir / "stability_results.json",
        {
            "trim_result": result,
            "stability": stability,
            "inertia": {
                "mass": mass_wing,
                "inertia_xx": inertia_xx,
                "inertia_yy": inertia_yy,
                "inertia_zz": inertia_zz,
            },
            "frame": {
                "trim_frame": "course",
                "stability_frame": stability_frame,
                "trim_axes": _axes_to_dict(trim_axes),
                "stability_axes": _axes_to_dict(stability_axes),
            },
            "run_settings": {
                "stability_config": (
                    str(stability_config_path)
                    if stability_config_path is not None
                    else None
                ),
                "output_dir": str(out_dir.resolve()),
                "selected_states": sel_states,
                "coupled": coupled,
            },
            "properties": props,
        },
    )

    fig_eig, ax_eig = plt.subplots(figsize=(5, 5))
    if coupled and "eig_selected" in stability:
        eig_sel = np.asarray(stability["eig_selected"])
        ax_eig.scatter(
            eig_sel.real, eig_sel.imag, label="coupled (selected)", color="#54A24B"
        )
    else:
        eig_long_sel = np.asarray(
            stability.get("eig_selected_long", stability["eig_long"])
        )
        eig_lat_sel = np.asarray(
            stability.get("eig_selected_lat", stability["eig_lat"])
        )
        if eig_long_sel.size:
            ax_eig.scatter(
                eig_long_sel.real,
                eig_long_sel.imag,
                label="longitudinal",
                color="#4C78A8",
            )
        if eig_lat_sel.size:
            ax_eig.scatter(
                eig_lat_sel.real,
                eig_lat_sel.imag,
                label="lateral",
                color="#F58518",
            )
    ax_eig.axvline(0.0, color="black", linewidth=0.8)
    ax_eig.axhline(0.0, color="black", linewidth=0.8)
    ax_eig.set_xlabel("Real part [1/s]")
    ax_eig.set_ylabel("Imaginary part [1/s]")
    ax_eig.set_title(
        f"VSM trim stability eigenvalues - {stability_frame} frame\n"
        f"states={sel_states}"
    )
    ax_eig.legend()
    ax_eig.grid(True, alpha=0.3)
    fig_eig.tight_layout()
    save_figure(fig_eig, out_dir / "stability_eigenvalues.pdf")

    fig_mat, axes = plt.subplots(1, 2, figsize=(10, 4))
    im0 = axes[0].imshow(stability["J_long"], aspect="auto", cmap="coolwarm")
    axes[0].set_title("J_long")
    axes[0].set_xlabel("state")
    axes[0].set_ylabel("output")
    fig_mat.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    im1 = axes[1].imshow(stability["J_lat"], aspect="auto", cmap="coolwarm")
    axes[1].set_title("J_lat")
    axes[1].set_xlabel("state")
    axes[1].set_ylabel("output")
    fig_mat.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    fig_mat.tight_layout()
    save_figure(fig_mat, out_dir / "stability_derivative_matrices.pdf")

    if args.no_show:
        plt.close(fig_eig)
        plt.close(fig_mat)
    else:
        plt.show()

    if not args.no_animate:
        fmt = args.animation_format
        ext = f".{fmt}"
        amplitude_rad = np.deg2rad(args.animation_amplitude_deg)
        # Use deformed geometries for animation when available, so both the
        # aerodynamic panels (blue) and structural skeleton (grey) match the
        # same deformed state used for the stability analysis.
        if deformed_aero_path is not None:
            from common import _resolve_csv_paths, add_vsm_path as _add_vsm_path
            from VSM.core.BodyAerodynamics import BodyAerodynamics as _BA
            import tempfile

            with deformed_aero_path.open("r", encoding="utf-8") as _f:
                _aero_cfg = yaml.safe_load(_f)
            _resolve_csv_paths(_aero_cfg, Path(args.config_folder))
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False, encoding="utf-8"
            ) as _tmp:
                yaml.dump(_aero_cfg, _tmp)
                _tmp_path = _tmp.name
            try:
                anim_body = _BA.instantiate(
                    n_panels=args.n_panels,
                    file_path=_tmp_path,
                    spanwise_panel_distribution=args.spanwise_panel_distribution,
                    bridle_path=props.get("struc_geometry_path"),
                )
            finally:
                Path(_tmp_path).unlink()
        else:
            anim_body, _ = build_body(args)

        bridle_geom = load_bridle_geometry(
            str(deformed_struc_path)
            if deformed_struc_path is not None
            else props.get("struc_geometry_path")
        )

        # Build the list of (block_title, slug, eigenvalues, eigenvectors,
        # states) tuples to animate. Always falls back to the default long/lat
        # split when the user did not supply --states/--coupled.
        animation_blocks: list[tuple[str, str, np.ndarray, np.ndarray, list[str]]] = []
        if coupled and "eig_selected" in stability:
            animation_blocks.append(
                (
                    "Coupled",
                    "coupled",
                    np.asarray(stability["eig_selected"], dtype=complex),
                    np.asarray(stability["vec_selected"], dtype=complex),
                    list(stability["states_selected"]),
                )
            )
        elif "states_selected_long" in stability:
            if stability["A_selected_long"].size > 0:
                animation_blocks.append(
                    (
                        "Longitudinal",
                        "long",
                        np.asarray(stability["eig_selected_long"], dtype=complex),
                        np.asarray(stability["vec_selected_long"], dtype=complex),
                        list(stability["states_selected_long"]),
                    )
                )
            if stability["A_selected_lat"].size > 0:
                animation_blocks.append(
                    (
                        "Lateral",
                        "lat",
                        np.asarray(stability["eig_selected_lat"], dtype=complex),
                        np.asarray(stability["vec_selected_lat"], dtype=complex),
                        list(stability["states_selected_lat"]),
                    )
                )
        else:
            animation_blocks.append(
                (
                    "Longitudinal",
                    "long",
                    np.asarray(stability["eig_long"], dtype=complex),
                    np.asarray(stability["vec_long"], dtype=complex),
                    ["u", "theta", "q"],
                )
            )
            animation_blocks.append(
                (
                    "Lateral",
                    "lat",
                    np.asarray(stability["eig_lat"], dtype=complex),
                    np.asarray(stability["vec_lat"], dtype=complex),
                    ["phi", "psi", "p", "r"],
                )
            )

        combined_mode_count = sum(len(block[2]) for block in animation_blocks)
        print(
            f"Rendering clean combined modes animation "
            f"({combined_mode_count} modes, real-time playback)…"
        )
        animate_all_eigenmodes_clean(
            anim_body,
            result,
            animation_blocks,
            out_path=out_dir / f"modes_combined{ext}",
            fps=args.animation_fps,
            amplitude_rad=amplitude_rad,
            max_physics_s=args.animation_max_physics_s,
            fmt=fmt,
            dpi=args.animation_dpi,
            reference_point=values["reference_point"],
            bridle_geom=bridle_geom,
            mode_gap_s=args.combined_animation_gap_s,
        )

        if not args.combined_animation_only:
            for block_title, slug, eigvals, eigvecs, block_states in animation_blocks:
                n_modes = len(eigvals)
                for i in range(n_modes):
                    eig_i = eigvals[i]
                    t_end = _physics_duration(
                        eig_i, max_s=args.animation_max_physics_s
                    )
                    print(
                        f"Rendering {block_title.lower()} mode {i}/{n_modes - 1} "
                        f"(λ={eig_i.real:+.3f}{eig_i.imag:+.3f}j  "
                        f"t_phys={t_end:.4f} s  frames={args.animation_frames})…"
                    )
                    animate_eigenmode(
                        anim_body,
                        result,
                        eig_i,
                        eigvecs[:, i],
                        block_states,
                        mode_index=i,
                        block_title=block_title,
                        out_path=out_dir / f"mode_{slug}_{i}{ext}",
                        fps=args.animation_fps,
                        n_frames=args.animation_frames,
                        amplitude_rad=amplitude_rad,
                        max_physics_s=args.animation_max_physics_s,
                        fmt=fmt,
                        dpi=args.animation_dpi,
                        reference_point=values["reference_point"],
                        bridle_geom=bridle_geom,
                    )


if __name__ == "__main__":
    main()
