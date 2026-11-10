# Copyright (c) 2023-2026 Oriol Cayon, Delft University of Technology
# SPDX-License-Identifier: Apache-2.0

"""Identify the power-tape map between another kite model and the ROM.

Two models of the same kite seldom pull the same at the same power-tape length:
their rigging, structure and aerodynamics differ. A client that flies the ROM's
optimized paths in its own model (e.g. V3Kite through RobustKiteControllers)
then flies a different force than the optimizer planned. This script finds the
ROM tape that pulls what the client's model pulls at each of its tapes, and
fits the map ``client tape = ROM tape + offset`` (and, to check whether a gain
is needed, ``client tape = gain x ROM tape + intercept``).

Input: the client model's forces at a few tapes, at one apparent wind each,
either

- the ``lift_drag_validation.json`` of RobustKiteControllers'
  ``examples/validate_lift_drag.jl`` (its ``v3kite`` list: ``tape``, ``v_app``,
  ``lift``, ``drag``, ``drag_kcu``), or
- a CSV with the columns ``input_depower`` [m], ``speed_apparent_wind`` [m/s],
  ``lift`` [N], ``drag`` [N], drag being wing + bridle lines + KCU (the ROM's
  ``drag_total``) and lift perpendicular to v_a.

The match is on the RESULTANT of lift and drag by default (``--target
resultant``), what the tether carries; ``--target lift`` matches lift alone.
Tether drag is left out on both sides: it belongs to the tether model. Points
whose force the ROM does not reach inside its identified depower band are
reported and skipped. The L/D the two models have at matched force is reported
too: one tape map cannot close an L/D gap, and the reel-out power scales with
roughly C_R (L/D)^2, so a client with a higher L/D flies slightly more power
than predicted.

    python examples/server/identify_depower_offset.py lift_drag_validation.json
    python examples/server/identify_depower_offset.py forces.csv --target lift \\
        --rom-config rom_config_aerostructural.yaml --output offset.yaml

The offset is in metres of tape, the ROM's unit; a client converts it to its own
depower unit (V3Kite's relative depower: offset / 5 m).
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import yaml

from awetrim.server.aero_forces import load_rom_kite
from awetrim.system.factory import resolve_rom_config_path
from awetrim.system.rom_forces import (
    MATCH_TARGETS,
    match_input_depower,
    matched_force,
    rom_aero_forces,
)
from awetrim.utils.config_paths import LEI_V3_SYSTEM_CONFIG


def read_points(path: Path) -> list:
    """``[{tape, v_a, lift, drag}]`` from a validate_lift_drag JSON or a CSV."""
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return [
            {
                "tape": float(p["tape"]),
                "v_a": float(p["v_app"]),
                "lift": float(p["lift"]),
                "drag": float(p["drag"]) + float(p.get("drag_kcu", 0.0)),
            }
            for p in data["v3kite"]
        ]
    with path.open(newline="", encoding="utf-8") as f:
        return [
            {
                "tape": float(row["input_depower"]),
                "v_a": float(row["speed_apparent_wind"]),
                "lift": float(row["lift"]),
                "drag": float(row["drag"]),
            }
            for row in csv.DictReader(f)
        ]


def client_force(point: dict, target: str) -> float:
    return math.hypot(point["lift"], point["drag"]) if target == "resultant" else point["lift"]


def identify(points, kite, band, target):
    """Per-point match, the offset and the linear fit."""
    matched, skipped = [], []
    for p in points:
        force = client_force(p, target)
        try:
            u_p = match_input_depower(kite, force, p["v_a"], band, target=target)
        except ValueError as err:
            skipped.append({"input_depower": p["tape"], "reason": str(err)})
            continue
        rom = rom_aero_forces(kite, u_p, p["v_a"])
        matched.append(
            {
                "input_depower": p["tape"],
                "rom_input_depower": u_p,
                "offset": p["tape"] - u_p,
                "speed_apparent_wind": p["v_a"],
                "force": force,
                "rom_force": matched_force(rom, target),
                "lift_to_drag": p["lift"] / p["drag"],
                "rom_lift_to_drag": rom["lift_total"] / rom["drag_total"],
            }
        )
    if not matched:
        raise SystemExit("no point is inside the ROM's depower band: nothing to fit")
    offsets = np.array([m["offset"] for m in matched])
    result = {
        "target": target,
        "offset": float(offsets.mean()),
        "offset_std": float(offsets.std()),
        "n_points": len(matched),
    }
    if len(matched) >= 3:
        rom_tapes = np.array([m["rom_input_depower"] for m in matched])
        tapes = np.array([m["input_depower"] for m in matched])
        gain, intercept = np.polyfit(rom_tapes, tapes, 1)
        result["linear"] = {"gain": float(gain), "intercept": float(intercept)}
    ratio = np.array([m["lift_to_drag"] / m["rom_lift_to_drag"] for m in matched])
    result["lift_to_drag_ratio_mean"] = float(ratio.mean())
    return result, matched, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input", type=Path, help="validate_lift_drag JSON or CSV")
    parser.add_argument("--system-config", type=Path, default=LEI_V3_SYSTEM_CONFIG)
    parser.add_argument("--rom-config", default=None,
                        help="ROM file (default: the one the system file selects)")
    parser.add_argument("--target", choices=MATCH_TARGETS, default="resultant")
    parser.add_argument("--output", type=Path, default=None,
                        help="result YAML (default: <input>_depower_offset.yaml)")
    args = parser.parse_args()

    rom_path = resolve_rom_config_path(args.system_config, args.rom_config)
    kite, validity = load_rom_kite(str(args.system_config),
                                   str(rom_path) if args.rom_config else None)
    band = validity.get("input_depower")
    if not band:
        raise SystemExit(f"{rom_path} states no validity.input_depower band")
    band = (float(min(band)), float(max(band)))

    points = read_points(args.input)
    result, matched, skipped = identify(points, kite, band, args.target)

    print(f"ROM {rom_path.name}, band {band[0]:g}-{band[1]:g} m, matching the {args.target}")
    print(f"{'tape':>6} {'v_a':>6} {'force':>7} {'ROM tape':>9} {'offset':>8} {'L/D':>6} {'ROM L/D':>8}")
    for m in matched:
        print(f"{m['input_depower']:6.3f} {m['speed_apparent_wind']:6.2f} {m['force']:7.0f} "
              f"{m['rom_input_depower']:9.3f} {m['offset']:+8.3f} "
              f"{m['lift_to_drag']:6.2f} {m['rom_lift_to_drag']:8.2f}")
    for s in skipped:
        print(f"{s['input_depower']:6.3f} skipped: {s['reason']}")
    print(f"offset {result['offset']:+.3f} m (std {result['offset_std']:.3f} m, "
          f"{result['n_points']} points)")
    if "linear" in result:
        lin = result["linear"]
        print(f"linear: tape = {lin['gain']:.3f} x ROM tape {lin['intercept']:+.3f} m")
    print(f"L/D at matched force: client / ROM = {result['lift_to_drag_ratio_mean']:.3f}")

    output = args.output or args.input.with_name(args.input.stem + "_depower_offset.yaml")
    document = {
        "rom_config": rom_path.name,
        "input_depower_band": list(band),
        "units": "input_depower and offset in m of power tape; client tape = ROM tape + offset",
        **result,
        "points": matched,
        "skipped": skipped,
    }
    output.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
