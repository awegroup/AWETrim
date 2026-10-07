"""Pack the matched-sweep chains into one JSON payload for an interactive page.

Reads every row ``run_matched_sweep_BILLOW.py`` wrote under
``results/<kite>/aerostructural/<root>/<chain>/<row>/`` and stores, per row,
its scalars (row.json) and its solved state: node positions, canopy load per
triangle, wrinkling regime per triangle and the tension in every cable and
pulley rope -- computed exactly as ``export_billow_viewer.py`` computes them.

The topology (tubes, cables, pulleys, canopy triangles, built shape) is written
once. Per-row arrays are quantised and base64-encoded little-endian typed
arrays, so a few hundred states stay near a megabyte:

    pos     int16, millimetres          (node positions, x y z per node)
    load    uint16, pascals             (canopy load per unit area)
    regime  uint8                       (0 slack, 1 wrinkled, 2 taut)
    cable / pulley  uint16, newtons     (line tensions)

Usage (from project root):
    python scripts/aerostructural/studies/export_matched_sweep_data.py --output sweep.json
"""

import argparse
import base64
import json
from pathlib import Path

import numpy as np

from awetrim.aerostructural.billow import structural_billow as sb
from billow.elements.membrane import membrane_regimes
from awetrim.aerostructural.case import DEFAULT_KITE_NAME

from check_mirror_asymmetry import final_positions
from export_billow_viewer import bridle_tensions, canopy_load, nice_ceiling, topology
from plot_billow_geometry import rebuild


def packed(array, dtype):
    return base64.b64encode(np.ascontiguousarray(array, dtype=dtype).tobytes()).decode()


def finite_or_none(value):
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="billow_matched_sweep")
    parser.add_argument("--output", required=True)
    parser.add_argument("--pattern", default="cross")
    parser.add_argument("--panels-per-section", type=int, default=2)
    parser.add_argument("--kite", default=DEFAULT_KITE_NAME)
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[3]
    root = project / "results" / args.kite / "aerostructural" / args.root
    structure = rebuild(project, args.kite, args.panels_per_section,
                        {"canopy_pattern": args.pattern, "canopy_refinement": 1})
    canopy = structure.model.element_set(sb.CANOPY)
    triangles = np.asarray(canopy.connectivity, dtype=int)
    built = structure.model.nodes.copy()

    rows = []
    for record_path in sorted(root.glob("*/*/row.json")):
        folder = record_path.parent
        if folder.name.startswith("_") or folder.parent.name.startswith("_"):
            continue   # set aside by hand
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("status") == "failed":
            continue
        positions = final_positions(folder)
        load = canopy_load(folder, positions, triangles)
        regime = np.asarray(membrane_regimes(positions, canopy)["regime"], dtype=int)
        cable, pulley = bridle_tensions(folder, structure)
        record = {k: finite_or_none(v) for k, v in record.items() if k != "attempts"}
        record.update(
            pos=packed(np.round(1e3 * positions), "<i2"),
            load=packed(np.clip(np.round(load), 0, 65535), "<u2"),
            regime=packed(regime, "<u1"),
            cable=packed(np.clip(np.round(cable), 0, 65535), "<u2"),
            pulley=packed(np.clip(np.round(pulley), 0, 65535), "<u2"),
            wrinkled=float(np.mean(regime == 1)),
            slack_lines=int(np.sum(np.concatenate([cable, pulley]) < 1.0)),
            peak_line=float(np.max(np.concatenate([cable, pulley]))),
        )
        rows.append((record, load, cable, pulley))
        print(f"{record['chain']}/{record['name']}: {record['status']}", flush=True)

    loads = np.concatenate([load for _, load, _, _ in rows])
    peaks = [float(np.max(np.concatenate([c, p]))) for _, _, c, p in rows]
    payload = {
        **topology(structure, project, args.kite),
        "triangles": triangles.tolist(),
        "reference": packed(np.round(1e3 * built), "<i2"),
        "nNodes": int(len(built)),
        "loadScale": nice_ceiling(float(np.percentile(loads, 98))),
        "tensionScale": nice_ceiling(max(peaks)),
        "rows": [record for record, _, _, _ in rows],
    }
    text = json.dumps(payload, separators=(",", ":"))
    Path(args.output).write_text(text, encoding="utf-8")
    print(f"{len(rows)} rows -> {args.output} ({len(text) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
