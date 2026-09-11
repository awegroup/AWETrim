"""Draw the converged Billow kite: canopy as fabric, tubes as tubes, bridle as line.

The canopy is shaded by its wrinkling regime -- slack, wrinkled, taut -- read
off the solved state with the same discriminant the energy kernel branches on.
That is the picture a spring-net canopy cannot draw at all: a net has no notion
of a biaxial stress state, so it cannot say which parts of the sail are carrying
load in two directions, which are carrying it in one and have gone into tension
field, and which are carrying nothing.

The model is rebuilt from the kite YAML (deterministic, ~1 s) and the node
positions are read from the run's ``sim_output.h5``, so nothing is re-solved.

Usage (from project root):
    python scripts/aerostructural/plot_billow_geometry.py
    python scripts/aerostructural/plot_billow_geometry.py --case depower_p0000mm_steer_p0000mm_135panels
"""

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import yaml as _yaml
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from awetrim import plotting
from awetrim.aerostructural.billow import structural_billow as sb
from awetrim.aerostructural.fem import read_struc_geometry_yaml
from awetrim.aerostructural.utils import load_yaml, rotate_geometry
from awetrim.structural.elements.membrane import SLACK, TAUT, WRINKLED, membrane_regimes
from common import DEFAULT_KITE_NAME, resolve_initial_geometry_rotation_kwargs, resolve_kite_paths

STRUC_GEOMETRY_FILENAME = "struc_geometry_FEM_full.yaml"
REGIME_LABELS = {SLACK: "slack", WRINKLED: "wrinkled", TAUT: "taut"}


def rebuild(project_dir, kite_name, panels_per_section=None, billow_settings=None):
    """The Billow model the run used, rebuilt from the YAML.

    ``billow_settings`` overrides ``structural_billow`` keys, which is how a
    run that used a different canopy mesh -- another triangulation, a refined
    one -- is reproduced: the mesh is a build-time choice, so rebuilding with
    the default would give a model whose element set does not match the saved
    positions.
    """
    config_path, _, _ = resolve_kite_paths(project_dir, kite_name)
    struc_geometry_path = project_dir / "data" / kite_name / STRUC_GEOMETRY_FILENAME
    with (project_dir / "data" / kite_name / "system.yaml").open(encoding="utf-8") as f:
        system_config = _yaml.safe_load(f)

    config = load_yaml(config_path)
    if panels_per_section is not None:
        config["aerodynamic"]["n_aero_panels_per_struc_section"] = panels_per_section
    if billow_settings:
        config["structural_billow"] = {
            **(config.get("structural_billow") or {}), **billow_settings
        }
    struc_geometry = load_yaml(struc_geometry_path)

    reader = read_struc_geometry_yaml.main(
        struc_geometry, config=config, system_config=system_config
    )
    struc_nodes, m_arr = reader[0], reader[1]
    canopy_sections, strut_sections = reader[7], reader[8]
    (kite_connectivity_arr, _bc, _bd, l0_arr, k_arr, c_arr, linktype_arr,
     pulley_line_indices, _pd) = reader[10:]

    struc_nodes = rotate_geometry(
        struc_nodes, **resolve_initial_geometry_rotation_kwargs(config)
    )
    return sb.instantiate(
        config=config,
        struc_geometry=struc_geometry,
        struc_nodes=struc_nodes,
        kite_connectivity_arr=kite_connectivity_arr,
        l0_arr=l0_arr,
        k_arr=k_arr,
        c_arr=c_arr,
        m_arr=m_arr,
        linktype_arr=linktype_arr,
        pulley_line_indices=pulley_line_indices,
        canopy_sections=canopy_sections,
        strut_sections=strut_sections,
    )


def frame(axes, positions, pad=0.05):
    """Equal-aspect box around the kite -- a kite drawn to unequal scale lies."""
    lower, upper = positions.min(axis=0), positions.max(axis=0)
    centre = 0.5 * (lower + upper)
    radius = 0.5 * float((upper - lower).max()) * (1.0 + pad)
    axes.set_xlim(centre[0] - radius, centre[0] + radius)
    axes.set_ylim(centre[1] - radius, centre[1] + radius)
    axes.set_zlim(centre[2] - radius, centre[2] + radius)
    axes.set_box_aspect((1.0, 1.0, 1.0))
    axes.set_xlabel("$x$ (m)")
    axes.set_ylabel("$y$ (m)")
    axes.set_zlabel("$z$ (m)")


def draw_kite(axes, structure, positions, *, colour_by_regime=True, title=""):
    """Canopy surface, inflated tubes and bridle lines at one solved state."""
    colours = plotting.PALETTE
    canopy = structure.model.element_set(sb.CANOPY)
    tubes = structure.model.element_set(sb.TUBES)
    cables = structure.model.element_set(sb.CABLES)
    pulleys = structure.model.element_set(sb.PULLEYS)

    # -- canopy ----------------------------------------------------------
    if colour_by_regime:
        regimes = membrane_regimes(positions, canopy)["regime"]
        cmap = ListedColormap(
            [colours["Sky Blue"], colours["Orange"], colours["Vermillion"]]
        )
        facecolours = cmap(np.clip(regimes, 0, 2) / 2.0)
    else:
        facecolours = colours["Sky Blue"]
        regimes = None

    surface = Poly3DCollection(
        [positions[triangle] for triangle in canopy.connectivity],
        alpha=0.9,
        edgecolor="0.4",
        linewidths=0.12,
    )
    surface.set_facecolor(facecolours)
    axes.add_collection3d(surface)

    # -- inflated tubes, drawn with a width that reads as a tube ----------
    # The fitted law carries the diameter, so the linewidth can be the real
    # one rather than a decorative constant.
    diameters = 2.0 * np.array(
        [(law.diameter / 2.0) for law in structure.tube_laws], dtype=float
    )
    scale = 26.0 / max(diameters.max(), 1e-6)
    axes.add_collection3d(
        Line3DCollection(
            [positions[edge] for edge in tubes.connectivity],
            colors=colours["Black"],
            linewidths=np.clip(diameters * scale, 1.0, 9.0),
            alpha=0.85,
        )
    )

    # -- bridle: cables straight, pulley ropes through their sheave -------
    axes.add_collection3d(
        Line3DCollection(
            [positions[edge] for edge in cables.connectivity],
            colors=colours["Bluish Green"],
            linewidths=0.9,
            alpha=0.9,
        )
    )
    rope_segments = []
    for node_i, sheave, node_k in pulleys.connectivity:
        rope_segments.append(positions[[node_i, sheave]])
        rope_segments.append(positions[[sheave, node_k]])
    axes.add_collection3d(
        Line3DCollection(
            rope_segments, colors=colours["Blue"], linewidths=0.9, alpha=0.9
        )
    )

    kcu = positions[list(structure.fixed_node_indices)]
    axes.scatter(kcu[:, 0], kcu[:, 1], kcu[:, 2], s=34,
                 color=colours["Reddish Purple"], depthshade=False, zorder=5)

    frame(axes, positions)
    axes.set_title(title)
    return regimes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="depower_p0000mm_steer_p0000mm")
    parser.add_argument("--kite", default=DEFAULT_KITE_NAME)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parents[2]
    results = (
        project_dir / "results" / args.kite / "aerostructural" / "billow" / args.case
    )
    with h5py.File(results / "sim_output.h5", "r") as handle:
        positions = np.asarray(handle["tracking/positions"])
        residual = np.asarray(handle["tracking/residual_norm"])

    structure = rebuild(project_dir, args.kite)
    reference = structure.model.nodes
    # The last stored frame is the converged one; earlier frames are the
    # coupled loop's intermediate shapes.
    final = positions[-1]
    if not np.isfinite(final).all() or np.allclose(final, 0.0):
        final = positions[np.max(np.flatnonzero(np.isfinite(positions).all(axis=(1, 2))))]

    plotting.set_plot_style()
    figure = plt.figure(figsize=(13.5, 6.4))

    axes = figure.add_subplot(1, 2, 1, projection="3d")
    draw_kite(axes, structure, reference, colour_by_regime=False,
              title="(a) As built, bridle relaxed")

    axes = figure.add_subplot(1, 2, 2, projection="3d")
    regimes = draw_kite(axes, structure, final, colour_by_regime=True,
                        title=f"(b) Trimmed, residual {residual[-1]:.2f} N")

    counts = {name: int((regimes == code).sum()) for code, name in REGIME_LABELS.items()}
    total = sum(counts.values())
    handles = [
        plt.Line2D([], [], marker="s", ls="none", markersize=8,
                   color=c, label=rf"{n} {100 * counts[n] / total:.0f}\%")
        for c, n in zip(
            [plotting.PALETTE["Sky Blue"], plotting.PALETTE["Orange"],
             plotting.PALETTE["Vermillion"]],
            ["slack", "wrinkled", "taut"],
        )
    ]
    axes.legend(handles=handles, loc="upper left", fontsize=7,
                title="canopy", framealpha=0.85, borderpad=0.4,
                handletextpad=0.5, labelspacing=0.3)

    # Total node motion is NOT deformation: the coupled driver rotates the whole
    # kite into the trimmed attitude (rotate_geometry on the solved roll/pitch/
    # yaw), so most of it is rigid body. Report the strain-carrying part
    # separately by removing the best-fit rigid motion (Kabsch).
    displacement = np.linalg.norm(final - reference, axis=1)
    centred_a = reference - reference.mean(axis=0)
    centred_b = final - final.mean(axis=0)
    u, _, vt = np.linalg.svd(centred_a.T @ centred_b)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    residual_shape = centred_b - centred_a @ rotation
    deformation = np.linalg.norm(residual_shape, axis=1)

    print(f"case              {args.case}")
    print(f"final residual    {residual[-1]:.3f} N")
    print(f"canopy regimes    " + ", ".join(f"{n} {counts[n]}" for n in
                                            ("slack", "wrinkled", "taut")))
    print(f"total node motion mean {displacement.mean():.4f} m, "
          f"max {displacement.max():.4f} m  (includes the rigid trim attitude)")
    print(f"deformation only  mean {deformation.mean():.4f} m, "
          f"max {deformation.max():.4f} m (node {int(deformation.argmax())})")

    figure.tight_layout()
    output = Path(args.output) if args.output else project_dir / f"billow_geometry_{args.case}.png"
    figure.savefig(output, dpi=180)
    print(f"\nwritten to {output}")
    plt.show()


if __name__ == "__main__":
    main()
