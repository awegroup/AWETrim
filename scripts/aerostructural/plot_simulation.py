"""Show one converged aerostructural case: 3-D kite and trim characteristics.

Works for cases from run_simulation_PSM.py and run_simulation_BILLOW.py. Prints
the trim table (flight state, apparent wind, angle of attack, lift, drag,
tether force, deformation, convergence) and writes kite_view.html into the case
folder: a self-contained page with the kite drawn in 3-D (drag to orbit, scroll
to zoom), the bridle shaded by line tension and the same table beside it.

From the project root:

    python scripts/aerostructural/plot_simulation.py                # newest case
    python scripts/aerostructural/plot_simulation.py <case folder>
    python scripts/aerostructural/plot_simulation.py <case folder> --no-open

Nothing is re-solved; everything is read from the case folder (sim_output.h5,
case.json, config.yaml).
"""

import argparse
import json
import webbrowser
from pathlib import Path

import numpy as np

from awetrim.aerostructural.case_view import (
    CASE_VIEW_FILENAME,
    format_summary,
    load_case,
    trim_summary,
)

TEMPLATE = Path(__file__).with_name("kite_viewer_template.html")
PROJECT_DIR = Path(__file__).resolve().parents[2]
DIGITS = 4  # 0.1 mm


def newest_case(root: Path) -> Path:
    """The most recently written case folder under ``root`` (one with case.json)."""
    cases = sorted(root.glob(f"*/aerostructural/**/{CASE_VIEW_FILENAME}"),
                   key=lambda p: p.stat().st_mtime)
    if not cases:
        raise SystemExit(
            f"No case with a {CASE_VIEW_FILENAME} under {root}. Run "
            "run_simulation_PSM.py or run_simulation_BILLOW.py first, or pass "
            "a case folder."
        )
    return cases[-1].parent


def _value_text(value):
    if isinstance(value, float):
        if not np.isfinite(value):
            return "n/a"
        if value != 0.0 and abs(value) < 1e-3:
            return "0"
        return f"{value:.4g}"
    return str(value)


def viewer_payload(case) -> dict:
    """The page's data: geometry at the start and at convergence, lines, table."""
    view = case["view"]
    rounded = lambda a: np.round(np.asarray(a, dtype=float), DIGITS).tolist()  # noqa: E731
    lines = [
        {"i": int(i), "j": int(j), "kind": kind, "name": name,
         "tension": None if tension is None else round(float(tension), 1)}
        for (i, j), kind, name, tension in zip(
            view["connectivity"], view["line_kind"], view["line_name"], view["line_tension"]
        )
    ]
    return {
        "title": case["case_dir"].name,
        "backend": view["backend"],
        "kite": view["kite"],
        "initial": rounded(case["initial"]),
        "final": rounded(case["final"]),
        "triangles": view["triangles"],
        "tubes": view["tubes"],
        "tubeDiameter": view["tube_diameter"],
        "lines": lines,
        "fixed": view["fixed"],
        "summary": [
            {"group": g, "label": label, "value": _value_text(v), "unit": u}
            for g, label, v, u in trim_summary(case)
        ],
    }


def report(case_dir, *, open_browser=True, output=None) -> Path | None:
    """Print the trim table of ``case_dir`` and write its 3-D viewer page."""
    case = load_case(case_dir)
    print(f"\n{case['case_dir']}\n")
    print(format_summary(trim_summary(case)))
    if case["view"] is None:
        print(f"\nNo {CASE_VIEW_FILENAME} in this case (solved before the viewer "
              "existed): table only.")
        return None
    page = TEMPLATE.read_text(encoding="utf-8").replace(
        "__DATA__", json.dumps(viewer_payload(case), separators=(",", ":"))
    )
    output = Path(output) if output else Path(case_dir) / "kite_view.html"
    output.write_text(page, encoding="utf-8")
    print(f"\n3-D view written to {output}")
    if open_browser:
        webbrowser.open(output.resolve().as_uri())
    return output


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("case", nargs="?", default=None,
                        help="case folder (default: the newest one under results/)")
    parser.add_argument("--output", default=None, help="HTML path (default: in the case folder)")
    parser.add_argument("--no-open", action="store_true", help="do not open a browser")
    args = parser.parse_args()
    case_dir = Path(args.case) if args.case else newest_case(PROJECT_DIR / "results")
    report(case_dir, open_browser=not args.no_open, output=args.output)


if __name__ == "__main__":
    main()
