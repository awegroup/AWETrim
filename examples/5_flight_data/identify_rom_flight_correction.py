# Copyright (c) 2023-2026 Oriol Cayon, Delft University of Technology
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""What the 2019 flight adds to the simulation-only ROM: the flown tapes and the turn rate.

Every step scores the ROM's own quasi-steady solve with the logged steering as input
(validate_quasi_steady_state_v3.py), on the training cycles; cycles 60-67 stay held out.

1. The tapes. The flown depower is not logged in the model's coordinate, so for each phase
   the tape is the one that best zeroes both the median tension and the median speed error
   over the phase's samples, interpolated on a grid of tapes. Written to
   identification.yaml's ``flight_match``: it reads the flight for both ROMs and changes no
   coefficient of the simulation-only one. A tape where the solve keeps less than MIN_SOLVED
   of the samples is refused: a median over the survivors would flatter it.
2. The turn rate, in the flight-corrected ROM only. One scale on the whole steering gain of
   the law the ROM flies (the roll gain k(u_p, u_s), or with ``steering_law: turn_rate`` the
   turn law's a(u_p, u_s), as ``Kite.gain_roll_scale`` scales them) brings the turn rate
   regressed on the flight's to a slope of 1 over the reel-out samples.

The lift, the drag, theta_b and the shape of the steering gain stay the simulations'.

Usage (project root)
    python examples/5_flight_data/identify_rom_flight_correction.py
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from awetrim.identification.controls import flight_depower_match
from awetrim.utils.config_paths import LEI_V3_DATA_DIR, LEI_V3_ROM_AEROSTRUCTURAL_CONFIG

OUT_DIR = Path("results/LEI-V3-KITE/identification/rom_flight_correction")
OUT_ROM = LEI_V3_ROM_AEROSTRUCTURAL_CONFIG.with_name("rom_config_aerostructural_flight_corrected.yaml")
VALIDATOR = Path(__file__).with_name("validate_quasi_steady_state_v3.py")
TRAINING_CYCLES = ("20-27", "40-47", "75-82")
#: Least share of the phase's samples the solve must keep at an accepted tape.
MIN_SOLVED = 0.75
#: Tape grid, around the file's match [m]; the ROM ends at its validity's depowered end.
TAPE_GRID = (-0.06, -0.03, 0.0, 0.03, 0.06, 0.09)
SLOPE_TOLERANCE = 0.01
MAX_GAIN_STEPS = 3
GAIN_START = (1.0, 1.2)


def evaluate(rom: Path, match: tuple[float, float], work: Path, tag: str) -> pd.DataFrame:
    """The validator on the training cycles, in parallel."""
    work.mkdir(parents=True, exist_ok=True)

    def one(cycles: str) -> pd.DataFrame:
        out = work / f"{tag}_{cycles}.csv"
        with (work / f"{tag}_{cycles}.log").open("w", encoding="utf-8") as log:
            subprocess.run(
                [sys.executable, str(VALIDATOR), "--rom", str(rom.resolve()), "--cycles", cycles,
                 "--steering", "measured", "--no-show", "--out", str(out),
                 "--depower-match", str(match[0]), str(match[1])],
                stdout=log, stderr=subprocess.STDOUT, check=True,
            )
        return pd.read_csv(out)

    with ThreadPoolExecutor(len(TRAINING_CYCLES)) as pool:
        return pd.concat(pool.map(one, TRAINING_CYCLES), ignore_index=True)


def score(d: pd.DataFrame, match: tuple[float, float]) -> dict:
    """Median tension and speed errors [%] and solved share per phase; turn-rate slope."""
    split = 0.5 * sum(match)
    solved = np.isfinite(d.predicted_tension)
    tension = 100.0 * (d.predicted_tension / d.measured_tension - 1.0)
    speed = 100.0 * (d.predicted_speed_tangential / d.measured_speed_tangential - 1.0)
    reel_out = (d.speed_radial > 0.5) & (d.input_depower < split) & (d.kite_elevation < 0.75)
    reel_in = (d.speed_radial < -0.5) & (d.input_depower >= split)
    out = {}
    for name, mask in (("out", reel_out), ("in", reel_in)):
        out[f"tension_{name}"] = float(tension[mask & solved].median())
        out[f"speed_{name}"] = float(speed[mask & solved].median())
        out[f"solved_{name}"] = float((mask & solved).sum() / max(mask.sum(), 1))
    turning = reel_out & solved & np.isfinite(d.predicted_course_rate)
    out["turn_slope"] = float(np.polyfit(d.measured_course_rate[turning], d.predicted_course_rate[turning], 1)[0])
    return out


def identify_tapes(rom: Path, start: tuple[float, float], top: float, work: Path) -> tuple[tuple[float, float], list]:
    """Both tapes on one grid, one validator run serving both phases."""
    history = []
    for k, offset in enumerate(TAPE_GRID):
        match = (start[0] + offset, min(start[1] + offset, top))
        s = score(evaluate(rom, match, work, f"tape_{k}"), match)
        history.append({"match": list(match), **s})
        print(f"tape_{k}: {match[0]:.3f} / {match[1]:.3f} m, tension {s['tension_out']:+.1f} / {s['tension_in']:+.1f} %, "
              f"speed {s['speed_out']:+.1f} / {s['speed_in']:+.1f} %, solved {s['solved_out']:.2f} / {s['solved_in']:.2f}", flush=True)
    tapes = []
    for i, name in enumerate(("out", "in")):
        grid = sorted(history, key=lambda h: h["match"][i])
        x = np.array([h["match"][i] for h in grid])
        tension = np.array([h[f"tension_{name}"] for h in grid])
        speed = np.array([h[f"speed_{name}"] for h in grid])
        kept = np.array([h[f"solved_{name}"] for h in grid])
        fine = np.linspace(x[0], x[-1], 2001)
        cost = np.interp(fine, x, tension) ** 2 + np.interp(fine, x, speed) ** 2
        allowed = np.interp(fine, x, kept) >= MIN_SOLVED
        if not allowed.any():
            raise SystemExit(f"reel-{name}: no tape keeps {MIN_SOLVED:.0%} of the samples")
        best = fine[allowed][np.argmin(cost[allowed])]
        if best in (x[0], x[-1]):
            print(f"WARNING reel-{name}: the best tape {best:.3f} m is at the end of the grid", flush=True)
        tapes.append(round(float(best), 2))
    return tuple(tapes), history


#: The gain each steering law's scale acts on (aerodynamics.steering_law; the turn law's a
#: is what Kite.gain_roll_scale scales under turn_rate). The roll gain is unused by turn_rate.
STEERING_GAIN = {"roll_gain": "gain_roll_steering", "turn_rate": "gain_turn_steering"}


def steering_law(rom: dict) -> str:
    """The steering law a ROM document flies (the Kite's default when absent)."""
    law = str(rom["aerodynamics"].get("steering_law", "roll_gain"))
    if law not in STEERING_GAIN:
        raise SystemExit(f"steering_law {law!r}: no gain to scale")
    return law


def corrected_rom(base: dict, scale: float, path: Path) -> Path:
    """The simulation-only ROM with the whole gain of its steering law scaled."""
    doc = yaml.safe_load(yaml.safe_dump(base))
    params = doc["aerodynamics"]["params"]
    gain = STEERING_GAIN[steering_law(doc)]
    params[gain] = float(params[gain]) * float(scale)
    for key in (f"{gain}_depower", f"{gain}_squared"):
        if params.get(key):
            params[key] = [float(c) * float(scale) for c in params[key]]
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return path


def identify_gain_scale(base: dict, match: tuple[float, float], work: Path) -> tuple[float, list]:
    """The scale whose turn-rate slope is 1, by the secant method."""
    history = []

    def slope(scale, tag):
        s = score(evaluate(corrected_rom(base, scale, work / f"{tag}.yaml"), match, work, tag), match)
        history.append({"scale": float(scale), **s})
        print(f"{tag}: gain scale {scale:.3f}: turn slope {s['turn_slope']:.3f}, tension {s['tension_out']:+.1f} / "
              f"{s['tension_in']:+.1f} %, speed {s['speed_out']:+.1f} / {s['speed_in']:+.1f} %", flush=True)
        return s["turn_slope"]

    a, b = GAIN_START
    fa, fb = slope(a, "gain_0"), slope(b, "gain_1")
    for step in range(MAX_GAIN_STEPS):
        if abs(fb - 1.0) < SLOPE_TOLERANCE:
            break
        a, fa, b = b, fb, b + (1.0 - fb) * (b - a) / (fb - fa)
        fb = slope(b, f"gain_{step + 2}")
    return float(b), history


def write_match(match: tuple[float, float]) -> None:
    """``flight_match`` in place, the file's comments kept."""
    path = LEI_V3_DATA_DIR / "identification.yaml"
    text = path.read_bytes().decode("utf-8")
    for key, value in (("reel_out", match[0]), ("reel_in", match[1]), ("date", date.today().isoformat())):
        text, n = re.subn(rf"(flight_match:(?:\r?\n\s+\S.*)*?\r?\n\s+{key}: )\S+", rf"\g<1>{value}", text)
        if n != 1:
            raise SystemExit(f"flight_match.{key} not found in {path}")
    path.write_bytes(text.encode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--no-write", action="store_true", help="identify and report, change no file")
    args = parser.parse_args()

    base = yaml.safe_load(LEI_V3_ROM_AEROSTRUCTURAL_CONFIG.read_text(encoding="utf-8"))
    work = OUT_DIR / "work"
    start = flight_depower_match(LEI_V3_DATA_DIR, 2019)
    top = float(base["validity"]["input_depower"][1])
    match, tape_history = identify_tapes(LEI_V3_ROM_AEROSTRUCTURAL_CONFIG, start, top, work)
    print(f"flight_match: reel-out {match[0]} m, reel-in {match[1]} m", flush=True)
    scale, gain_history = identify_gain_scale(base, match, work)
    print(f"{steering_law(base)} gain scale {scale:.4f}", flush=True)

    report = {"training_cycles": list(TRAINING_CYCLES), "flight_match": list(match),
              "steering_law": steering_law(base), "gain_scale": scale,
              "tape_history": tape_history, "gain_history": gain_history}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "correction_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.no_write:
        return
    write_match(match)
    doc = yaml.safe_load(corrected_rom(base, scale, work / "final.yaml").read_text(encoding="utf-8"))
    doc["metadata"]["flight_correction"] = {
        "identified_by": "examples/5_flight_data/identify_rom_flight_correction.py",
        "identified_on": date.today().isoformat(), "flight": 2019,
        "training_cycles": list(TRAINING_CYCLES), "held_out_cycles": "60-67",
        "flight_match": list(match), "steering_law": steering_law(base), "gain_scale": scale,
    }
    header = (f"# AEROSTRUCTURAL ROM with the 2019 flight's scale on its {steering_law(base)} gain.\n"
              "# GENERATED by examples/5_flight_data/identify_rom_flight_correction.py; re-run it\n"
              "# rather than editing by hand. NOT YET REVIEWED: treat it as provisional.\n")
    OUT_ROM.write_text(header + yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    print(f"Wrote {OUT_ROM} and the flight_match", flush=True)


if __name__ == "__main__":
    main()
