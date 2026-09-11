"""Build the interactive Billow kite viewer from converged result folders.

Each state is one converged coupled run under ``billow_canopy_ab/``. The model is
rebuilt from the YAML with the run's canopy mesh (deterministic, ~1 s) and the
node positions are read from its ``sim_output.h5``, so nothing is re-solved. The
canopy can be shaded by aerodynamic load per unit area (the default; one colour
scale for every state) or by wrinkling regime -- slack, wrinkled, taut -- read
off each state with the same discriminant the energy kernel branches on.

The page is ``billow_viewer_template.html`` with the payload substituted for
``__DATA__``: a single self-contained HTML file.

Usage (from project root):
    python scripts/aerostructural/export_billow_viewer.py \\
        --state cross_traction:1:"Coarse canopy" \\
        --state cross_x2_traction:2:"Canopy refined x2" \\
        --output billow_viewer.html
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from awetrim.aerostructural.billow import structural_billow as sb
from awetrim.structural.elements.membrane import membrane_regimes
from common import DEFAULT_KITE_NAME

from check_mirror_asymmetry import decompose, final_positions, mirror_partners
from plot_billow_geometry import rebuild

TEMPLATE = Path(__file__).with_name("billow_viewer_template.html")
DIGITS = 4          # 0.1 mm: well below anything the viewer can show


def rounded(array):
    return np.round(np.asarray(array, dtype=float), DIGITS).tolist()


def canopy_load(folder, positions, triangles):
    """Aerodynamic load per unit area on each canopy element [Pa].

    The nodal loads the coupled loop applied at its last iteration, divided
    by each node's tributary area (a third of every adjacent element, on the
    solved shape), averaged over the element's three nodes. With the traction
    transfer every canopy node carries its share, so this is the load field
    the structure actually saw, not a sampling of it.
    """
    with h5py.File(folder / "sim_output.h5", "r") as handle:
        residual = np.asarray(handle["tracking/residual_norm"])
        forces = np.asarray(handle["tracking/f_ext"])
    nodal = forces[int(np.flatnonzero(residual)[-1])]
    triangles = np.asarray(triangles, dtype=int)
    a, b, c = (positions[triangles[:, k]] for k in range(3))
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    tributary = np.zeros(len(positions))
    np.add.at(tributary, triangles.ravel(), np.repeat(area / 3.0, 3))
    pressure = np.linalg.norm(nodal, axis=1) / np.where(tributary > 0, tributary, np.inf)
    return pressure[triangles].mean(axis=1)


def nice_ceiling(value):
    """Smallest 1-1.2-1.5-2-2.5-3-4-5-6-8 x 10^n at or above ``value``."""
    magnitude = 10.0 ** np.floor(np.log10(value))
    for step in (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if step * magnitude >= value:
            return float(step * magnitude)
    return float(10 * magnitude)


def convergence(folder):
    """Final coupled residual [N] and the number of coupled iterations."""
    with h5py.File(folder / "sim_output.h5", "r") as handle:
        residual = np.asarray(handle["tracking/residual_norm"])
    done = np.flatnonzero(residual)
    return float(residual[done[-1]]), int(len(done))


def state_payload(project, kite, folder, refine, label, note, pattern, panels_per_section):
    structure = rebuild(project, kite, panels_per_section,
                        {"canopy_pattern": pattern, "canopy_refinement": refine})
    canopy = structure.model.element_set(sb.CANOPY)
    built = structure.model.nodes.copy()
    positions = final_positions(folder)
    per_pair, rigid, _ = decompose(positions, built, structure.grid, mirror_partners(built))
    residual, iterations = convergence(folder)
    return structure, {
        "label": label,
        "note": note,
        "triangles": np.asarray(canopy.connectivity, dtype=int).tolist(),
        "reference": rounded(built),
        "positions": rounded(positions),
        "referenceRegime": np.asarray(membrane_regimes(built, canopy)["regime"], int).tolist(),
        "regime": np.asarray(membrane_regimes(positions, canopy)["regime"], int).tolist(),
        "load": np.round(canopy_load(folder, positions, canopy.connectivity), 1).tolist(),
        "residual": residual,
        "iterations": iterations,
        "nodes": int(structure.model.n_nodes),
        "dof": int(structure.model.layout.n_dof),
        "mismatch": 1e3 * max(p[2] for p in per_pair),
        "intrinsic": 1e3 * max(p[3] for p in per_pair),
        "rigidDeg": float(rigid),
        "symmetric": pattern != "diagonal",
    }


def topology(structure):
    """Tubes, bridle and pulleys: the same for every canopy mesh of one kite."""
    tubes = structure.model.element_set(sb.TUBES)
    return {
        "tubes": np.asarray(tubes.connectivity, dtype=int).tolist(),
        "tubeDiameter": [float(law.diameter) for law in structure.tube_laws],
        "cables": np.asarray(structure.model.element_set(sb.CABLES).connectivity, int).tolist(),
        "pulleys": np.asarray(structure.model.element_set(sb.PULLEYS).connectivity, int).tolist(),
        "fixed": [int(i) for i in structure.fixed_node_indices],
    }


def load_scale(states):
    """One colour scale for every state: the 98th percentile, rounded up.

    A shared scale, so the same colour is the same load in every state; the
    percentile, so a handful of concentrated nodal loads -- at a bridle
    attachment, say -- does not wash out the rest of the canopy.
    """
    loads = np.concatenate([np.asarray(s["load"]) for s in states.values() if s.get("load")])
    return nice_ceiling(float(np.percentile(loads, 98)))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--state", action="append", required=True,
                        help="folder:refinement:label  (folder under billow_canopy_ab/)")
    parser.add_argument("--note", action="append", default=[],
                        help="HTML note per --state, in the same order (optional)")
    parser.add_argument("--pattern", default="cross")
    parser.add_argument("--panels-per-section", type=int, default=2)
    parser.add_argument("--kite", default=DEFAULT_KITE_NAME)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[2]
    root = project / "results" / args.kite / "aerostructural" / "billow_canopy_ab"
    states, shared = {}, None
    for index, spec in enumerate(args.state):
        folder, refine, label = spec.split(":", 2)
        note = args.note[index] if index < len(args.note) else ""
        structure, payload = state_payload(project, args.kite, root / folder, int(refine),
                                           label, note, args.pattern, args.panels_per_section)
        states[folder] = payload
        shared = shared or topology(structure)
        print(f"{folder}: {payload['nodes']} nodes, {len(payload['triangles'])} triangles, "
              f"residual {payload['residual']:.2f} N in {payload['iterations']} iterations, "
              f"mismatch {payload['mismatch']:.1f} mm ({payload['intrinsic']:.1f} shape)")

    page = TEMPLATE.read_text(encoding="utf-8").replace(
        "__DATA__", json.dumps({**shared, "loadScale": load_scale(states), "states": states},
                               separators=(",", ":"))
    )
    Path(args.output).write_text(page, encoding="utf-8")
    print(f"written {args.output} ({len(page) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
