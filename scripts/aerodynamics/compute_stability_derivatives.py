from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from common import (
    add_common_arguments,
    build_body,
    build_system_model,
    output_dir,
    parsed_common,
    print_trim_summary,
    save_figure,
    write_json,
)

from awetrim.aerodynamics.vsm_quasi_steady import (
    DEFAULT_AXES,
    compute_vsm_trim_stability_derivatives,
    solve_vsm_quasi_steady_trim,
)

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


def _save_animation(anim: FuncAnimation, path, *, fps: int, fmt: str) -> None:
    if fmt == "mp4":
        anim.save(str(path), writer="ffmpeg", fps=fps, dpi=100)
    else:
        anim.save(str(path), writer="pillow", fps=fps, dpi=80)
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

    def _parse_edges(key: str) -> list[tuple[int, int]]:
        edges: list[tuple[int, int]] = []
        table = data.get(key, {})
        for row in table.get("data", []):
            ids = [int(v) for v in row[1:] if v is not None]
            if len(ids) == 2:
                edges.append((ids[0], ids[1]))
            elif len(ids) == 3:
                # pulley: line goes ids[1] → ids[0] → ids[2]
                edges.append((ids[1], ids[0]))
                edges.append((ids[0], ids[2]))
        return edges

    wing_edges = _parse_edges("wing_connections")
    bridle_edges = _parse_edges("bridle_connections")

    return {
        "nodes": nodes,
        "wing_edges": wing_edges,
        "bridle_edges": bridle_edges,
        "kcu_point": nodes[0].copy(),
    }


def _build_mode_figure(
    panel_corners, origin, eig, label, block_title, mode_index, bridle_geom=None
):
    """Create the figure skeleton shared by both animation functions."""
    fig = plt.figure(figsize=(12, 8))
    ax3d = fig.add_subplot(121, projection="3d")
    stab_char = "stable" if eig.real < 0 else "UNSTABLE"
    freq = abs(eig.imag) / (2 * np.pi)
    ax3d.set_title(
        f"{block_title} mode {mode_index}  [{stab_char}]\n"
        f"λ = {eig.real:+.3f}{eig.imag:+.3f}j  f={freq:.3f} Hz",
        fontsize=9,
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

    # Pre-build animated line objects for bridle if geometry is available
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
    return (
        fig,
        ax3d,
        panel_lines,
        wing_struct_lines,
        bridle_lines,
        kcu_marker,
        status_txt,
    )


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


def animate_longitudinal_mode(
    body_aero,
    trim_result: dict,
    stability: dict,
    mode_index: int,
    *,
    out_path,
    fps: int = 8,
    n_frames: int = 60,
    amplitude_rad: float = np.deg2rad(5.0),
    max_physics_s: float = 30.0,
    fmt: str = "gif",
    reference_point: np.ndarray | None = None,
    bridle_geom: dict | None = None,
) -> None:
    """Save a GIF/MP4 of one longitudinal eigenmode (pitch + speed).

    Physics duration is auto-computed from the eigenvalue so the animation
    always covers the interesting transient. All time axes show real physics time.
    """
    eig_long = np.asarray(stability["eig_long"], dtype=complex)
    vec_long = np.asarray(stability["vec_long"], dtype=complex)

    eig = eig_long[mode_index]
    mode_vec = vec_long[:, mode_index].copy()

    norm_factor = max(abs(mode_vec[1]), 1e-12)  # normalise by theta (index 1)
    mode_vec = mode_vec / norm_factor

    t_phys_end = _physics_duration(eig, max_s=max_physics_s)
    t_phys = np.linspace(0.0, t_phys_end, n_frames)

    response = _mode_time_response(eig, mode_vec, t_phys, amplitude_rad)
    u_perturb = response[0, :]
    theta_perturb = response[1, :]

    trim_pitch_rad = np.deg2rad(float(trim_result["opt_x"][2]))
    trim_speed = float(trim_result["opt_x"][0])

    panel_corners = np.array(
        [panel.corner_points for panel in body_aero.panels], dtype=float
    )
    origin = np.asarray(
        reference_point if reference_point is not None else [0.0, 0.0, 0.0], dtype=float
    )

    fig, ax3d, panel_lines, wing_struct_lines, bridle_lines, kcu_marker, status_txt = (
        _build_mode_figure(
            panel_corners,
            origin,
            eig,
            "θ",
            "Longitudinal",
            mode_index,
            bridle_geom=bridle_geom,
        )
    )

    ax_u = fig.add_subplot(222)
    ax_u.set_title("Speed perturbation  u(t)", fontsize=9)
    ax_u.set_xlabel("t [s]", fontsize=8)
    ax_u.set_ylabel("u [m/s]", fontsize=8)
    u_total = trim_speed + u_perturb
    ax_u.plot(t_phys, u_total, color="0.7", linewidth=1)
    (u_line,) = ax_u.plot([], [], color="tab:blue", linewidth=2)
    (u_marker,) = ax_u.plot([], [], "o", color="tab:blue")
    ax_u.set_xlim(0.0, t_phys_end)
    pad = max(0.05 * np.ptp(u_total), 0.1)
    ax_u.set_ylim(u_total.min() - pad, u_total.max() + pad)
    ax_u.grid(True, alpha=0.3)

    ax_theta = fig.add_subplot(224)
    ax_theta.set_title("Pitch  θ(t)", fontsize=9)
    ax_theta.set_xlabel("t [s]", fontsize=8)
    ax_theta.set_ylabel("θ [deg]", fontsize=8)
    theta_total_deg = np.rad2deg(trim_pitch_rad + theta_perturb)
    ax_theta.plot(t_phys, theta_total_deg, color="0.7", linewidth=1)
    (theta_line,) = ax_theta.plot([], [], color="tab:orange", linewidth=2)
    (theta_marker,) = ax_theta.plot([], [], "o", color="tab:orange")
    ax_theta.set_xlim(0.0, t_phys_end)
    pad = max(0.05 * np.ptp(theta_total_deg), 0.1)
    ax_theta.set_ylim(theta_total_deg.min() - pad, theta_total_deg.max() + pad)
    ax_theta.grid(True, alpha=0.3)

    extra = [*wing_struct_lines, *bridle_lines] + ([kcu_marker] if kcu_marker else [])

    def init():
        for ln in panel_lines:
            ln.set_data([], [])
            ln.set_3d_properties([])
        u_line.set_data([], [])
        u_marker.set_data([], [])
        theta_line.set_data([], [])
        theta_marker.set_data([], [])
        return [
            *panel_lines,
            *extra,
            status_txt,
            u_line,
            u_marker,
            theta_line,
            theta_marker,
        ]

    def update(fi: int):
        theta_t = trim_pitch_rad + float(theta_perturb[fi])
        R = _rot_rad(DEFAULT_AXES.normal, theta_t)
        rot = _rotate_pts(panel_corners, R, origin)
        for idx, ln in enumerate(panel_lines):
            c = np.vstack([rot[idx], rot[idx][0]])
            ln.set_data(c[:, 0], c[:, 1])
            ln.set_3d_properties(c[:, 2])
        _update_bridle_lines(
            bridle_geom, wing_struct_lines, bridle_lines, kcu_marker, R, origin
        )
        u_t = trim_speed + float(u_perturb[fi])
        status_txt.set_text(
            f"t={t_phys[fi]:.4f} s  |  u={u_t:+.2f} m/s  θ={np.rad2deg(theta_t):+.2f}°"
        )
        u_line.set_data(t_phys[: fi + 1], u_total[: fi + 1])
        u_marker.set_data([t_phys[fi]], [u_total[fi]])
        theta_line.set_data(t_phys[: fi + 1], theta_total_deg[: fi + 1])
        theta_marker.set_data([t_phys[fi]], [theta_total_deg[fi]])
        return [
            *panel_lines,
            *extra,
            status_txt,
            u_line,
            u_marker,
            theta_line,
            theta_marker,
        ]

    anim = FuncAnimation(
        fig, update, init_func=init, frames=n_frames, interval=1000.0 / fps, blit=False
    )
    fig.tight_layout()
    _save_animation(anim, out_path, fps=fps, fmt=fmt)
    plt.close(fig)


def animate_lateral_mode(
    body_aero,
    trim_result: dict,
    stability: dict,
    mode_index: int,
    *,
    out_path,
    fps: int = 8,
    n_frames: int = 60,
    amplitude_rad: float = np.deg2rad(5.0),
    max_physics_s: float = 30.0,
    fmt: str = "gif",
    reference_point: np.ndarray | None = None,
    bridle_geom: dict | None = None,
) -> None:
    """Save a GIF/MP4 of one lateral eigenmode (roll + yaw).

    Physics duration is auto-computed from the eigenvalue. All time axes show
    real physics time in seconds.
    """
    eig_lat = np.asarray(stability["eig_lat"], dtype=complex)
    vec_lat = np.asarray(stability["vec_lat"], dtype=complex)

    eig = eig_lat[mode_index]
    mode_vec = vec_lat[:, mode_index].copy()

    # Normalise by the largest of phi/psi (indices 1, 2)
    norm_factor = max(np.max(np.abs(mode_vec[1:3])), 1e-12)
    mode_vec = mode_vec / norm_factor

    t_phys_end = _physics_duration(eig, max_s=max_physics_s)
    t_phys = np.linspace(0.0, t_phys_end, n_frames)

    response = _mode_time_response(eig, mode_vec, t_phys, amplitude_rad)
    phi_perturb = response[1, :]
    psi_perturb = response[2, :]

    trim_roll_rad = np.deg2rad(float(trim_result["opt_x"][1]))
    trim_yaw_rad = np.deg2rad(float(trim_result["opt_x"][3]))

    panel_corners = np.array(
        [panel.corner_points for panel in body_aero.panels], dtype=float
    )
    origin = np.asarray(
        reference_point if reference_point is not None else [0.0, 0.0, 0.0], dtype=float
    )

    fig, ax3d, panel_lines, wing_struct_lines, bridle_lines, kcu_marker, status_txt = (
        _build_mode_figure(
            panel_corners,
            origin,
            eig,
            "φ/ψ",
            "Lateral",
            mode_index,
            bridle_geom=bridle_geom,
        )
    )

    ax_phi = fig.add_subplot(222)
    ax_phi.set_title("Roll  φ(t)", fontsize=9)
    ax_phi.set_xlabel("t [s]", fontsize=8)
    ax_phi.set_ylabel("φ [deg]", fontsize=8)
    phi_total_deg = np.rad2deg(trim_roll_rad + phi_perturb)
    ax_phi.plot(t_phys, phi_total_deg, color="0.7", linewidth=1)
    (phi_line,) = ax_phi.plot([], [], color="tab:blue", linewidth=2)
    (phi_marker,) = ax_phi.plot([], [], "o", color="tab:blue")
    ax_phi.set_xlim(0.0, t_phys_end)
    pad = max(0.05 * np.ptp(phi_total_deg), 0.1)
    ax_phi.set_ylim(phi_total_deg.min() - pad, phi_total_deg.max() + pad)
    ax_phi.grid(True, alpha=0.3)

    ax_psi = fig.add_subplot(224)
    ax_psi.set_title("Yaw  ψ(t)", fontsize=9)
    ax_psi.set_xlabel("t [s]", fontsize=8)
    ax_psi.set_ylabel("ψ [deg]", fontsize=8)
    psi_total_deg = np.rad2deg(trim_yaw_rad + psi_perturb)
    ax_psi.plot(t_phys, psi_total_deg, color="0.7", linewidth=1)
    (psi_line,) = ax_psi.plot([], [], color="tab:orange", linewidth=2)
    (psi_marker,) = ax_psi.plot([], [], "o", color="tab:orange")
    ax_psi.set_xlim(0.0, t_phys_end)
    pad = max(0.05 * np.ptp(psi_total_deg), 0.1)
    ax_psi.set_ylim(psi_total_deg.min() - pad, psi_total_deg.max() + pad)
    ax_psi.grid(True, alpha=0.3)

    extra = [*wing_struct_lines, *bridle_lines] + ([kcu_marker] if kcu_marker else [])

    def init():
        for ln in panel_lines:
            ln.set_data([], [])
            ln.set_3d_properties([])
        phi_line.set_data([], [])
        phi_marker.set_data([], [])
        psi_line.set_data([], [])
        psi_marker.set_data([], [])
        return [
            *panel_lines,
            *extra,
            status_txt,
            phi_line,
            phi_marker,
            psi_line,
            psi_marker,
        ]

    def update(fi: int):
        phi_t = trim_roll_rad + float(phi_perturb[fi])
        psi_t = trim_yaw_rad + float(psi_perturb[fi])
        R = _rot_rad(DEFAULT_AXES.radial, psi_t) @ _rot_rad(DEFAULT_AXES.course, phi_t)
        rot = _rotate_pts(panel_corners, R, origin)
        for idx, ln in enumerate(panel_lines):
            c = np.vstack([rot[idx], rot[idx][0]])
            ln.set_data(c[:, 0], c[:, 1])
            ln.set_3d_properties(c[:, 2])
        _update_bridle_lines(
            bridle_geom, wing_struct_lines, bridle_lines, kcu_marker, R, origin
        )
        status_txt.set_text(
            f"t={t_phys[fi]:.4f} s  |  φ={np.rad2deg(phi_t):+.2f}°  ψ={np.rad2deg(psi_t):+.2f}°"
        )
        phi_line.set_data(t_phys[: fi + 1], phi_total_deg[: fi + 1])
        phi_marker.set_data([t_phys[fi]], [phi_total_deg[fi]])
        psi_line.set_data(t_phys[: fi + 1], psi_total_deg[: fi + 1])
        psi_marker.set_data([t_phys[fi]], [psi_total_deg[fi]])
        return [
            *panel_lines,
            *extra,
            status_txt,
            phi_line,
            phi_marker,
            psi_line,
            psi_marker,
        ]

    anim = FuncAnimation(
        fig, update, init_func=init, frames=n_frames, interval=1000.0 / fps, blit=False
    )
    fig.tight_layout()
    _save_animation(anim, out_path, fps=fps, fmt=fmt)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Solve VSM aerodynamic trim and compute stability derivatives."
    )
    add_common_arguments(parser)
    parser.add_argument("--eps-vel", type=float, default=0.1)
    parser.add_argument("--eps-angle-deg", type=float, default=0.5)
    parser.add_argument("--eps-rate", type=float, default=0.01)
    parser.add_argument(
        "--no-animate",
        action="store_true",
        help="Skip saving mode animations.",
    )
    parser.add_argument(
        "--animation-format",
        choices=["gif", "mp4"],
        default="gif",
        help="Animation output format (default: gif; mp4 requires ffmpeg).",
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
    args = parser.parse_args()
    values = parsed_common(args)
    out_dir = output_dir(args, "stability_derivatives")

    # Load body and properties from config folder
    body, props = build_body(args)

    # DEBUG: Check what props contains
    # Use properties from system.yaml if not explicitly overridden via args
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

    result, solved_body = solve_vsm_quasi_steady_trim(
        body_aero=body,
        center_of_gravity=values["center_of_gravity"],
        reference_point=values["reference_point"],
        system_model=build_system_model(args, mass_wing=mass_wing),
        x_guess=values["x_guess"],
        bounds_lower=values["bounds_lower"],
        bounds_upper=values["bounds_upper"],
        include_gravity=args.include_gravity,
        moment_tolerance=args.moment_tolerance,
        return_timing_breakdown=True,
        max_nfev=args.max_nfev,
    )
    print_trim_summary(result)

    stability = compute_vsm_trim_stability_derivatives(
        body_aero=solved_body,
        center_of_gravity=values["center_of_gravity"],
        reference_point=values["reference_point"],
        x_trim=np.asarray(result["opt_x"], dtype=float),
        trim_result=result,
        mass=mass_wing,
        inertia_xx=inertia_xx,
        inertia_yy=inertia_yy,
        inertia_zz=inertia_zz,
        distance_radial=args.distance_radial,
        eps_vel=args.eps_vel,
        eps_angle_deg=args.eps_angle_deg,
        eps_rate=args.eps_rate,
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
            "properties": props,
        },
    )

    fig_eig, ax_eig = plt.subplots(figsize=(5, 5))
    eig_long = np.asarray(stability["eig_long"])
    eig_lat = np.asarray(stability["eig_lat"])
    ax_eig.scatter(eig_long.real, eig_long.imag, label="longitudinal", color="#4C78A8")
    ax_eig.scatter(eig_lat.real, eig_lat.imag, label="lateral", color="#F58518")
    ax_eig.axvline(0.0, color="black", linewidth=0.8)
    ax_eig.axhline(0.0, color="black", linewidth=0.8)
    ax_eig.set_xlabel("Real part [1/s]")
    ax_eig.set_ylabel("Imaginary part [1/s]")
    ax_eig.set_title("VSM trim stability eigenvalues")
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
        anim_body, _ = build_body(args)  # fresh baseline geometry for animation
        bridle_geom = load_bridle_geometry(props.get("struc_geometry_path"))

        n_long = len(stability["eig_long"])
        for i in range(n_long):
            eig_i = stability["eig_long"][i]
            t_end = _physics_duration(eig_i, max_s=args.animation_max_physics_s)
            print(
                f"Rendering longitudinal mode {i}/{n_long - 1} "
                f"(λ={eig_i.real:+.3f}{eig_i.imag:+.3f}j  "
                f"t_phys={t_end:.4f} s  frames={args.animation_frames})…"
            )
            animate_longitudinal_mode(
                anim_body,
                result,
                stability,
                mode_index=i,
                out_path=out_dir / f"mode_long_{i}{ext}",
                fps=args.animation_fps,
                n_frames=args.animation_frames,
                amplitude_rad=amplitude_rad,
                max_physics_s=args.animation_max_physics_s,
                fmt=fmt,
                reference_point=values["reference_point"],
                bridle_geom=bridle_geom,
            )

        n_lat = len(stability["eig_lat"])
        for i in range(n_lat):
            eig_i = stability["eig_lat"][i]
            t_end = _physics_duration(eig_i, max_s=args.animation_max_physics_s)
            print(
                f"Rendering lateral mode {i}/{n_lat - 1} "
                f"(λ={eig_i.real:+.3f}{eig_i.imag:+.3f}j  "
                f"t_phys={t_end:.4f} s  frames={args.animation_frames})…"
            )
            animate_lateral_mode(
                anim_body,
                result,
                stability,
                mode_index=i,
                out_path=out_dir / f"mode_lat_{i}{ext}",
                fps=args.animation_fps,
                n_frames=args.animation_frames,
                amplitude_rad=amplitude_rad,
                max_physics_s=args.animation_max_physics_s,
                fmt=fmt,
                reference_point=values["reference_point"],
                bridle_geom=bridle_geom,
            )


if __name__ == "__main__":
    main()
