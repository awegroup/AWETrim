"""Plot rigid-body principal axes for a deformed aerostructural result.

Loads deformed node positions from a sim_output.h5 result file, combines them
with nodal masses from the structural geometry YAML, and produces a 3-D
visualisation showing:
  - deformed structural nodes (marker size proportional to nodal mass)
  - CG location
  - principal body axes (x=red, y=green, z=blue) as arrows at the CG
  - global reference frame arrows (thin, grey) at the origin

The kite name is inferred from the results path so the struc_geometry YAML is
located automatically.  Override with --struc if your layout differs.

Usage
-----
    python scripts/identification/plot_body_axes.py --result results/aerostructural/LEI-V3-KITE/depower_p0000mm_steer_p0000mm
    python scripts/identification/plot_body_axes.py --result results/aerostructural/LEI-V3-KITE/depower_p0000mm_steer_p0000mm/sim_output.h5
    python scripts/identification/plot_body_axes.py --result <case_dir> --struc data/LEI-V3-KITE/struc_geometry.yaml
    python scripts/identification/plot_body_axes.py --result <case_dir> --save output/body_axes.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – registers 3d projection

PROJECT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from awetrim.aerostructural.utils import load_sim_output
from awetrim.identification.rigid_body_axes import (
    RigidBodyAxes,
    compute_rigid_body_axes,
    load_psm_nodes_and_masses,
)


# ---------------------------------------------------------------------------
# path helpers
# ---------------------------------------------------------------------------

def _resolve_h5_path(result_arg: Path) -> Path:
    """Accept either a directory containing sim_output.h5 or the file itself."""
    result_arg = result_arg.resolve()
    if result_arg.is_file():
        return result_arg
    h5 = result_arg / "sim_output.h5"
    if h5.exists():
        return h5
    raise FileNotFoundError(f"No sim_output.h5 found at {result_arg}")


def _infer_struc_geometry_path(h5_path: Path) -> Path | None:
    """Walk up from the HDF5 path to find data/{kite_name}/struc_geometry.yaml.

    Supports both layouts:
        results/{kite_name}/aerostructural/{case_folder}/sim_output.h5  (new)
        results/aerostructural/{kite_name}/{case_folder}/sim_output.h5  (legacy)
    """
    parts = h5_path.parts
    try:
        results_idx = next(i for i, p in enumerate(parts) if p == "results")
    except StopIteration:
        return None

    project_root = Path(*parts[:results_idx])

    if results_idx + 2 < len(parts):
        if parts[results_idx + 1] == "aerostructural":
            # legacy: results/aerostructural/{kite_name}/...
            kite_name = parts[results_idx + 2]
        else:
            # new: results/{kite_name}/aerostructural/...
            kite_name = parts[results_idx + 1]
        candidate = project_root / "data" / kite_name / "struc_geometry.yaml"
        return candidate if candidate.exists() else None

    return None


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------

def _arrow(ax, origin, direction, length, color, label, linewidth=2.5, **kw):
    tip = origin + length * direction
    ax.quiver(
        *origin,
        *(tip - origin),
        color=color,
        linewidth=linewidth,
        arrow_length_ratio=0.15,
        label=label,
        **kw,
    )


def plot_body_axes(
    struc_nodes: np.ndarray,
    m_arr: np.ndarray,
    result: RigidBodyAxes,
    title: str = "Rigid-body principal axes",
) -> plt.Figure:
    """Generate the 3-D body-axis visualisation."""
    cg = result.cg
    body_axes = result.body_axes  # rows: [x_body, y_body, z_body]

    bbox_diag = np.linalg.norm(struc_nodes.max(axis=0) - struc_nodes.min(axis=0))
    arrow_len = 0.20 * bbox_diag

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # structural nodes (size proportional to mass)
    m_norm = m_arr / m_arr.max()
    sc = ax.scatter(
        struc_nodes[:, 0],
        struc_nodes[:, 1],
        struc_nodes[:, 2],
        s=40 + 180 * m_norm,
        c=m_arr,
        cmap="viridis",
        alpha=0.75,
        zorder=3,
        label="PSM nodes",
    )
    plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.1, label="nodal mass (kg)")

    ax.scatter(*cg, s=120, c="black", marker="*", zorder=5, label="CG")

    body_colors = ["tab:red", "tab:green", "tab:blue"]
    body_labels = [
        f"$x_b$  $I_x$={result.principal_moments[0]:.3f} kg·m²",
        f"$y_b$  $I_y$={result.principal_moments[1]:.3f} kg·m²",
        f"$z_b$  $I_z$={result.principal_moments[2]:.3f} kg·m²",
    ]
    for i, (color, label) in enumerate(zip(body_colors, body_labels)):
        _arrow(ax, cg, body_axes[i], arrow_len, color, label, linewidth=2.5)

    # global reference frame (thin grey)
    for i, (label) in enumerate(["$e_x$", "$e_y$", "$e_z$"]):
        _arrow(
            ax,
            np.zeros(3),
            np.eye(3)[i],
            arrow_len * 0.6,
            "grey",
            label if i == 0 else "_nolegend_",
            linewidth=1.0,
            linestyle="dashed",
        )

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)

    # equal aspect ratio
    pts = np.vstack([struc_nodes, cg.reshape(1, 3)])
    mid = (pts.max(axis=0) + pts.min(axis=0)) / 2
    half = (pts.max(axis=0) - pts.min(axis=0)).max() / 2 * 1.1
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_zlim(mid[2] - half, mid[2] + half)

    _print_summary(result)
    return fig


def _print_summary(result: RigidBodyAxes) -> None:
    print(f"\nCG (structural frame): [{result.cg[0]:+.4f}, {result.cg[1]:+.4f}, {result.cg[2]:+.4f}] m")
    print(f"CG (body frame):       [{result.cg_body[0]:+.4f}, {result.cg_body[1]:+.4f}, {result.cg_body[2]:+.4f}] m")
    print("\nInertia tensor about CG (kg·m²):")
    for row in result.inertia_cg:
        print(f"  [{row[0]:+10.4f}  {row[1]:+10.4f}  {row[2]:+10.4f}]")
    print("\nPrincipal body axes (unit vectors in structural frame):")
    for name, axis, moment in zip(["x_body", "y_body", "z_body"], result.body_axes, result.principal_moments):
        print(f"  {name}: [{axis[0]:+.4f}, {axis[1]:+.4f}, {axis[2]:+.4f}]   I = {moment:.4f} kg·m²")


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--result",
        type=Path,
        required=True,
        help="Path to sim_output.h5 or the case directory that contains it.",
    )
    parser.add_argument(
        "--struc",
        type=Path,
        default=None,
        help="Path to struc_geometry.yaml (auto-detected from results path if omitted).",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Save figure to this path instead of showing it.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    h5_path = _resolve_h5_path(args.result)
    case_dir = h5_path.parent
    case_label = case_dir.name

    # Priority for geometry source:
    #   1. --struc flag (explicit override)
    #   2. struc_geometry.yaml saved alongside the result (deformed, from save_geometry_snapshot)
    #   3. data/{kite_name}/struc_geometry.yaml inferred from the results path (undeformed masses)
    if args.struc is not None:
        struc_path = args.struc.resolve()
        use_deformed_yaml = False
    else:
        saved = case_dir / "struc_geometry.yaml"
        if saved.exists():
            struc_path = saved
            use_deformed_yaml = True
        else:
            struc_path = _infer_struc_geometry_path(h5_path)
            use_deformed_yaml = False
            if struc_path is None:
                sys.exit(
                    "Could not auto-detect struc_geometry.yaml from the results path.\n"
                    "Pass --struc path/to/struc_geometry.yaml explicitly."
                )

    if not struc_path.exists():
        sys.exit(f"struc_geometry not found: {struc_path}")

    with struc_path.open("r", encoding="utf-8") as f:
        struc_geometry = yaml.safe_load(f)

    if use_deformed_yaml:
        # Deformed YAML has updated positions — use them directly as node positions.
        deformed_nodes, m_arr = load_psm_nodes_and_masses(struc_geometry)
        print(f"Using saved deformed geometry: {struc_path}")
    else:
        # Fall back: positions from HDF5, masses from the (undeformed) data YAML.
        _, tracking = load_sim_output(h5_path)
        if "positions" not in tracking:
            sys.exit(f"'positions' dataset missing in {h5_path}")
        deformed_nodes = np.asarray(tracking["positions"][-1], dtype=float)
        _, m_arr = load_psm_nodes_and_masses(struc_geometry)
        print(f"Using HDF5 positions + geometry masses from: {struc_path}")

    if deformed_nodes.shape[0] != m_arr.shape[0]:
        sys.exit(
            f"Node count mismatch: positions have {deformed_nodes.shape[0]} nodes, "
            f"struc_geometry yields {m_arr.shape[0]} nodes."
        )

    result = compute_rigid_body_axes(deformed_nodes, m_arr)

    fig = plot_body_axes(
        deformed_nodes,
        m_arr,
        result,
        title=f"Rigid-body principal axes — {case_label} (deformed)",
    )

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"\nFigure saved to {args.save}")
    else:
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
