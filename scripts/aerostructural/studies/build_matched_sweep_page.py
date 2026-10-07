"""Fill the interactive matched-sweep page from its three payloads.

The page is ``matched_sweep_page_template.html`` -- a single self-contained HTML
file once filled, with sliders over depower, steering and apparent speed on a
fixed camera, the depower and steering sweeps as charts against the paper's
wireframe chains and the 2019 flight, and every solved row in a table. It needs
no server and nothing of Claude: host it anywhere or send the file.

Three placeholders are substituted:

``__DATA__``
    The payload ``export_matched_sweep_data.py`` writes (topology, built shape
    and every solved row), plus the slider grids this script derives from it:
    the paper's depower tape grid with the matched values as exact entries
    (``depower_grid`` of ``run_matched_sweep_BILLOW.py``), the steering grid and
    the target apparent speeds.
``__COMPARISON__``
    The paper's layers: the wireframe depower and steering chains, the 2019
    near-straight flight medians per phase, the turn cloud per phase and the
    attribution record. Built from the wes-quasi-steady paper's own loaders;
    the copy used for the 2026-09-13 page is ``scripts/personal/billow-paper/
    comparison.json``.
``__ANALYSIS__``
    The findings tiles and the notes under the charts
    (``matched_sweep_page_findings.json``), with the three turn-rate gains
    recomputed here: ``gain_full`` from the steering rows and ``gain_wire``
    from the wireframe steering chain, both a fit of the course rate against
    v_a u_s through the origin; ``gain_flight`` is the comparison's own.

Usage (from project root):
    python scripts/aerostructural/studies/export_matched_sweep_data.py --output sweep.json
    python scripts/aerostructural/studies/build_matched_sweep_page.py \\
        --data sweep.json \\
        --comparison scripts/personal/billow-paper/comparison.json \\
        --output matched_sweep.html
"""

import argparse
import json
from pathlib import Path

import numpy as np

from run_chain_depower_BILLOW import depower_input
from run_matched_sweep_BILLOW import UDP_REELIN, UDP_REELOUT, depower_grid

HERE = Path(__file__).parent
TEMPLATE = HERE / "matched_sweep_page_template.html"
FINDINGS = HERE / "matched_sweep_page_findings.json"


def gain_through_origin(rows, speed="va", rate="course_rate"):
    """Turn rate against v_a u_s, least squares through the origin [rad/s per m/s]."""
    pts = [(r[speed] * r["us"], r[rate]) for r in rows
           if r.get("us") and r.get(rate) is not None and r["us"] > 0]
    if not pts:
        return None
    x, y = np.array(pts).T
    return float(np.sum(x * y) / np.sum(x * x))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="export_matched_sweep_data.py output")
    parser.add_argument("--comparison", required=True, help="the paper's comparison layers (JSON)")
    parser.add_argument("--findings", default=str(FINDINGS))
    parser.add_argument("--template", default=str(TEMPLATE))
    parser.add_argument("--output", required=True)
    parser.add_argument("--udp-min", type=float, default=0.180)
    parser.add_argument("--udp-max", type=float, default=UDP_REELIN)
    parser.add_argument("--ldp-step", type=float, default=0.05)
    parser.add_argument("--udp-reelout", type=float, default=UDP_REELOUT)
    parser.add_argument("--udp-reelin", type=float, default=UDP_REELIN)
    parser.add_argument("--udp-steer", type=float, default=None,
                        help="depower of the steering chains (default: read off the rows)")
    parser.add_argument("--va-grid", default="13,15,17,19,21,23,25",
                        help="target apparent speeds on the slider [m/s]")
    parser.add_argument("--us-max", type=float, default=0.25)
    parser.add_argument("--us-step", type=float, default=0.025)
    args = parser.parse_args()

    data = json.loads(Path(args.data).read_text(encoding="utf-8"))
    comparison = json.loads(Path(args.comparison).read_text(encoding="utf-8"))
    analysis = json.loads(Path(args.findings).read_text(encoding="utf-8"))
    rows = data["rows"]

    # The results folder may hold steering chains at several depowers (earlier
    # attempts); the page shows ONE steering sweep, by default the most complete.
    steer_all = [r for r in rows if r["mode"] == "steering"]
    counts = {}
    for r in steer_all:
        counts[round(r["udp"], 4)] = counts.get(round(r["udp"], 4), 0) + 1
    udp_steer = args.udp_steer if args.udp_steer is not None else max(counts, key=counts.get)
    dropped = {u: n for u, n in counts.items() if abs(u - udp_steer) > 6e-4}
    if dropped:
        print(f"dropping steering rows at other depowers: {dropped}")
    rows = [r for r in rows if r["mode"] != "steering" or abs(round(r["udp"], 4) - udp_steer) <= 6e-4]
    data["rows"] = rows
    steer_rows = [r for r in rows if r["mode"] == "steering"]

    tapes = depower_grid(args.udp_min, args.udp_max, args.ldp_step,
                         anchors=[args.udp_reelout, udp_steer, args.udp_reelin])
    n_us = int(round(args.us_max / args.us_step))
    data.update({
        "udpGrid": [round(float(depower_input(t)), 4) for t in tapes],
        "tapeOf": tapes,
        "vaGrid": [float(v) for v in args.va_grid.split(",")],
        "usGrid": [round(args.us_step * k, 6) for k in range(n_us + 1)],
        "udpReelout": args.udp_reelout,
        "udpReelin": args.udp_reelin,
        "udpSteer": udp_steer,
    })

    analysis["gain_full"] = gain_through_origin(steer_rows)
    analysis["gain_wire"] = gain_through_origin(comparison.get("wireframe_steering", []),
                                                speed="v_a", rate="chi_dot_turn")
    analysis["gain_flight"] = comparison.get("turn", {}).get("reel-out", {}).get("gain")

    page = Path(args.template).read_text(encoding="utf-8")
    for key, payload in (("__DATA__", data), ("__COMPARISON__", comparison), ("__ANALYSIS__", analysis)):
        assert page.count(key) == 1, key
        page = page.replace(key, json.dumps(payload, separators=(",", ":")))
    Path(args.output).write_text(page, encoding="utf-8")
    print(f"{len(rows)} rows, {len(steer_rows)} steered at u_dp {udp_steer} -> {args.output} "
          f"({len(page) / 1e6:.2f} MB); gains full {analysis['gain_full']:.4f} "
          f"wire {analysis['gain_wire']:.4f} flight {analysis['gain_flight']}")


if __name__ == "__main__":
    main()
