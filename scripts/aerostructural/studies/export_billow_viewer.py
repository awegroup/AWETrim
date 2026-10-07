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
    python scripts/aerostructural/studies/export_billow_viewer.py \\
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
from awetrim.aerostructural.utils import load_yaml
from billow.elements import line_tensions
from billow.elements.membrane import membrane_regimes
from awetrim.aerostructural.case import DEFAULT_KITE_NAME

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


def bridle_tensions(folder, structure):
    """Tension [N] in every cable and pulley rope at the run's last iteration.

    At the configuration the solver RETURNED (``solved_positions``), not the
    relaxed one drawn on screen: lengths are rotation-invariant, but the
    relaxed shape is millimetres off equilibrium, which on a dyneema line is
    hundreds of newtons. And with the rest lengths the RUN ended on, which a
    rebuild from the YAML does not know: an actuated state moved its tapes.
    """
    with h5py.File(folder / "sim_output.h5", "r") as handle:
        residual = np.asarray(handle["tracking/residual_norm"])
        last = int(np.flatnonzero(residual)[-1])
        solved = np.asarray(handle["tracking/solved_positions"][last])
        run_rest = np.asarray(handle["tracking"].attrs["rest_lengths"], dtype=float)
    tensions = {}
    for name, rows in ((sb.CABLES, structure.cable_row), (sb.PULLEYS, structure.pulley_row)):
        element_set = structure.model.element_set(name)
        rest = element_set.params[:, 0].copy()
        for reader_index in np.flatnonzero(rows >= 0):
            if np.isfinite(run_rest[reader_index]):
                rest[rows[reader_index]] = run_rest[reader_index]
        tensions[name] = line_tensions(
            solved, element_set.with_param_column("rest_length", rest)
        )
    return tensions[sb.CABLES], tensions[sb.PULLEYS]


def line_names(project, kite, structure):
    """The geometry's name for each cable and each pulley rope, in model order.

    The reader appends the bridle after the wing, in the YAML's order, one
    element per line and two per pulley rope (one per arm), so the element
    count fixes where the bridle starts.
    """
    geometry = load_yaml(project / "data" / kite / "struc_geometry_FEM_full.yaml")
    connections = geometry["bridle_connections"]["data"]
    n_reader = len(structure.cable_row)
    reader_index = n_reader - sum(2 if len(row) == 4 else 1 for row in connections)
    name_of = {}
    for row in connections:
        for _ in range(2 if len(row) == 4 else 1):
            name_of[reader_index] = row[0]
            reader_index += 1
    names = {}
    for set_name, rows in ((sb.CABLES, structure.cable_row), (sb.PULLEYS, structure.pulley_row)):
        labels = [""] * structure.model.element_set(set_name).n_elements
        for index in np.flatnonzero(rows >= 0):
            labels[rows[index]] = name_of.get(int(index), "")
        names[set_name] = labels
    return names[sb.CABLES], names[sb.PULLEYS]


def steering_tag(folder):
    """``"steered 50 mm"`` for a run that steered, else ``None``.

    Read from the driver's ``steering_half_difference``; runs from before the
    driver could steer do not carry it and are unsteered by construction.
    """
    with h5py.File(folder / "sim_output.h5", "r") as handle:
        delta = float(handle["tracking"].attrs.get("steering_half_difference", np.nan))
    if not np.isfinite(delta) or abs(delta) < 1e-9:
        return None
    return f"steered {1e3 * delta:+.0f} mm"


def state_payload(project, kite, folder, refine, label, note, pattern, panels_per_section):
    structure = rebuild(project, kite, panels_per_section,
                        {"canopy_pattern": pattern, "canopy_refinement": refine})
    canopy = structure.model.element_set(sb.CANOPY)
    built = structure.model.nodes.copy()
    positions = final_positions(folder)
    per_pair, rigid, _ = decompose(positions, built, structure.grid, mirror_partners(built))
    residual, iterations = convergence(folder)
    cable_tension, pulley_tension = bridle_tensions(folder, structure)
    return structure, {
        "cableTension": np.round(cable_tension, 1).tolist(),
        "pulleyTension": np.round(pulley_tension, 1).tolist(),
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
        "tag": steering_tag(folder),
    }


def topology(structure, project=None, kite=None):
    """Tubes, bridle and pulleys: the same for every canopy mesh of one kite."""
    tubes = structure.model.element_set(sb.TUBES)
    names = line_names(project, kite, structure) if project is not None else ([], [])
    return {
        "cableNames": names[0],
        "pulleyNames": names[1],
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


def tension_scale(states):
    """Top of the bridle-tension scale [N]: the largest line load, rounded up.

    The maximum, not a percentile: there are only ~70 lines, and the few
    heavily loaded main lines are exactly the ones worth reading. The ramp is
    logarithmic, from 1 N, because line loads span three decades.
    """
    peak = max(max(s["cableTension"] + s["pulleyTension"]) for s in states.values())
    return nice_ceiling(float(peak))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--state", action="append", required=True,
                        help="folder:refinement:label  (folder under --root)")
    parser.add_argument("--root", default="billow_canopy_ab",
                        help="results folder the --state folders are relative "
                             "to, under results/<kite>/aerostructural/ "
                             "(billow_steering for run_steering_BILLOW.py; a "
                             "state may climb out with ../)")
    parser.add_argument("--note", action="append", default=[],
                        help="HTML note per --state, in the same order (optional)")
    parser.add_argument("--pattern", default="cross")
    parser.add_argument("--panels-per-section", type=int, default=2)
    parser.add_argument("--kite", default=DEFAULT_KITE_NAME)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[3]
    root = project / "results" / args.kite / "aerostructural" / args.root
    states, shared = {}, None
    for index, spec in enumerate(args.state):
        folder, refine, label = spec.split(":", 2)
        note = args.note[index] if index < len(args.note) else ""
        structure, payload = state_payload(project, args.kite, root / folder, int(refine),
                                           label, note, args.pattern, args.panels_per_section)
        states[folder] = payload
        shared = shared or topology(structure, project, args.kite)
        print(f"{folder}: {payload['nodes']} nodes, {len(payload['triangles'])} triangles, "
              f"residual {payload['residual']:.2f} N in {payload['iterations']} iterations, "
              f"mismatch {payload['mismatch']:.1f} mm ({payload['intrinsic']:.1f} shape)")

    page = TEMPLATE.read_text(encoding="utf-8").replace(
        "__DATA__", json.dumps({**shared, "loadScale": load_scale(states),
                                "tensionScale": tension_scale(states), "states": states},
                               separators=(",", ":"))
    )
    Path(args.output).write_text(page, encoding="utf-8")
    print(f"written {args.output} ({len(page) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
