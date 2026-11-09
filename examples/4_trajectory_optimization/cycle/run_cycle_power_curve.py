"""Mean cycle power over a range of winds, each wind one full-cycle optimization.

Goes outward from a start wind, upward and downward at the same time (or one
after the other, ``--sequential``). The start wind is solved cold
(``fit_periodic_cycle_config.py --size``, then ``run_full_cycle_opti.py``, with
its coarse pass); every further wind continues from the last converged one
(``--continue-from``: about twice as fast as a cold solve, at the same power),
falling back on a cold solve when the continuation fails. A wind that fails both
ends the direction; a start wind that does not converge ends the run: pick
another with ``--start-wind``.

Finished winds are kept in ``power_curve.json``, so running the same command again
resumes. Writes ``power_curve.csv`` and ``power_curve.png`` as it goes, the logs
and seeds under ``winds/``, into ``--output-dir`` (default
``results/<kite>/optimization/power_curve``).

Usage:
    python .../run_cycle_power_curve.py [--kite K] [--winds 4:12:0.5]
                                        [--start-wind W] [--output-dir D]
"""

import argparse
import json
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

_here = Path(__file__).resolve()
_repo_root = next(p for p in _here.parents if p.name == "examples").parent
if str(_repo_root / "src") not in sys.path:
    sys.path.insert(0, str(_repo_root / "src"))

from cycle_kites import DEFAULT_KITE, resolve_kite  # noqa: E402

FIT_SCRIPT = _here.with_name("fit_periodic_cycle_config.py")
OPTI_SCRIPT = _here.with_name("run_full_cycle_opti.py")
# "average power  <seed> kW  <optimized> kW"; a continuation has no baseline
# march, its seed column reads n/a
POWER_LINE = re.compile(r"average power\s+(?:-?[\d.]+ kW|n/a)\s+(-?[\d.]+) kW")


def parse_winds(text):
    """``"4,5,7"`` or ``"lo:hi:step"`` -> sorted wind speeds [m/s]."""
    if ":" in text:
        lo, hi, step = (float(part) for part in text.split(":"))
        return [round(float(w), 6) for w in np.arange(lo, hi + 0.5 * step, step)]
    return sorted(float(part) for part in text.split(","))


def outward_winds(winds, start):
    """``(start wind, winds above going up, winds below going down)``; the
    start is the listed wind nearest ``start``."""
    first = min(winds, key=lambda w: abs(w - start))
    return (
        first,
        sorted(w for w in winds if w > first),
        sorted((w for w in winds if w < first), reverse=True),
    )


def walk_direction(winds, last_good, results, solve_cold, solve_continued):
    """Solve ``winds`` in order, each continued from the last converged one
    (its trajectory seeds the solve: about twice as fast as a cold solve, at
    the same power); a continuation that fails falls back on a cold solve
    (seed fit + run_full_cycle_opti.py, with its coarse pass), and a wind that
    fails both ends the direction. Rows already in ``results`` (keyed by
    wind, in the same schema) are kept and steer the walk the way they would
    have: a stored failed continuation goes on to its cold fallback, a stored
    failed cold solve ends the direction. ``solve_*`` return a row."""
    for wind in winds:
        row = results.get(wind)
        if row is None:
            row = solve_continued(wind, last_good)
            results[wind] = row
        if not row["ok"] and row["method"] == "continued":
            row = solve_cold(wind)
            results[wind] = row
        if not row["ok"]:
            return
        last_good = row


class PowerCurve:
    def __init__(self, kite, output_dir):
        self.kite = kite
        self.results_dir = Path(resolve_kite(kite)["results_dir"])
        self.output_dir = Path(output_dir or self.results_dir.parent / "power_curve")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.output_dir / "power_curve.json"
        self.lock = threading.Lock()
        self.results = {}
        if self.state_path.exists():
            stored = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.results = {float(key): row for key, row in stored.items()}

    def _run(self, script, arguments, log_path):
        with open(log_path, "wb") as log:
            return subprocess.run(
                [sys.executable, str(script), "--kite", str(self.kite), *arguments],
                cwd=_repo_root,
                stdout=log,
                stderr=subprocess.STDOUT,
            ).returncode

    def _optimize(self, wind, row, folder, tag, source_flag, source):
        log_path = folder / "opti.log"
        code = self._run(
            OPTI_SCRIPT,
            [source_flag, str(source), "--tag", tag, "--wind-speed", f"{wind:g}"],
            log_path,
        )
        text = log_path.read_text(encoding="utf-8", errors="replace")
        powers = POWER_LINE.findall(text)
        configs = list(self.results_dir.glob(f"full_cycle_optimized_*_{tag}.yaml"))
        if code == 0 and powers and configs:
            row.update(ok=True, power_kw=float(powers[-1]), config=str(configs[0]))
        return row

    def _record(self, row, started):
        row["seconds"] = time.time() - started
        with self.lock:
            self.results[row["wind"]] = row
            self.write_outputs()
        outcome = f"{row['power_kw']:.2f} kW" if row["ok"] else "FAILED"
        print(f"{row['wind']:g} m/s {row['method']}: {outcome}, {row['seconds']:.0f} s", flush=True)
        return row

    def _folder(self, wind):
        folder = self.output_dir / "winds" / f"w{wind:g}"
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def solve_cold(self, wind):
        started = time.time()
        folder = self._folder(wind)
        row = {"wind": wind, "ok": False, "method": "cold", "start": None}
        seed = folder / "seed.yaml"
        code = self._run(
            FIT_SCRIPT,
            ["--size", "--wind-speed", f"{wind:g}", "--output", str(seed)],
            folder / "seed.log",
        )
        if code == 0:
            self._optimize(wind, row, folder, f"curve_w{wind:g}", "--config", seed)
        return self._record(row, started)

    def solve_continued(self, wind, previous):
        started = time.time()
        row = {"wind": wind, "ok": False, "method": "continued", "start": previous["wind"]}
        self._optimize(
            wind, row, self._folder(wind), f"curve_w{wind:g}",
            "--continue-from", previous["config"],
        )
        return self._record(row, started)

    def write_outputs(self):
        rows = [self.results[wind] for wind in sorted(self.results)]
        self.state_path.write_text(
            json.dumps({f"{row['wind']:g}": row for row in rows}, indent=1), encoding="utf-8"
        )
        lines = ["wind_m_s,converged,method,start_wind_m_s,power_kW,seconds"]
        lines += [
            ",".join(
                str(value)
                for value in (
                    f"{row['wind']:g}", row["ok"], row["method"],
                    "" if row["start"] is None else f"{row['start']:g}",
                    row.get("power_kw", ""), round(row.get("seconds", 0.0)),
                )
            )
            for row in rows
        ]
        (self.output_dir / "power_curve.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
        self._plot(rows)

    def _plot(self, rows):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
        good = [row for row in rows if row["ok"]]
        ax.plot([r["wind"] for r in good], [r["power_kw"] for r in good], color="#2a6fb0", lw=1.8)
        for method, face, label in (
            ("cold", "#2a6fb0", "Cold"),
            ("continued", "white", "From the last converged wind"),
        ):
            part = [r for r in good if r["method"] == method]
            ax.plot(
                [r["wind"] for r in part], [r["power_kw"] for r in part], "o", ms=7,
                color="#2a6fb0", mfc=face, mew=1.6, label=label,
            )
        bad = [row["wind"] for row in rows if not row["ok"]]
        if bad:
            ax.plot(
                bad, [0.0] * len(bad), "x", ms=8, mew=2, color="#2a6fb0", clip_on=False,
                label="Did not converge",
            )
        ax.set(xlabel="Wind speed at the profile's reference height (m s$^{-1}$)",
               ylabel="Mean cycle power (kW)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
        fig.savefig(self.output_dir / "power_curve.png", dpi=150)
        plt.close(fig)

    def run(self, winds, start, sequential=False):
        first, above, below = outward_winds(winds, start)
        row = self.results.get(first)
        if row is None:
            row = self.solve_cold(first)
        if not row["ok"]:
            print(f"The start wind {first:g} m/s did not converge; pick another with --start-wind")
            return 1
        if sequential:
            # one solve at a time (clean timings, half the memory)
            for direction in (above, below):
                walk_direction(direction, row, self.results, self.solve_cold, self.solve_continued)
            threads = []
        else:
            threads = [
                threading.Thread(
                    target=walk_direction,
                    args=(direction, row, self.results, self.solve_cold, self.solve_continued),
                )
                for direction in (above, below)
            ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        missing = [f"{wind:g}" for wind in winds if wind not in self.results]
        if missing:
            print("Not attempted, past a wind that failed both ways: " + ", ".join(missing))
        return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--kite",
        default=DEFAULT_KITE,
        help=f"Kite folder under data/ (or a path to one); default {DEFAULT_KITE}",
    )
    parser.add_argument(
        "--winds",
        default="4:12:0.5",
        help='Wind speeds at the profile\'s reference height (m/s): "lo:hi:step" or '
        '"4,5,7"; default 4:12:0.5',
    )
    parser.add_argument(
        "--start-wind",
        type=float,
        default=None,
        help="Wind to start from (the nearest listed one); default the kite profile's",
    )
    parser.add_argument("--output-dir", default=None, help="Default results/<kite>/optimization/power_curve")
    parser.add_argument("--sequential", action="store_true",
                        help="Walk up, then down, one solve at a time (default: both at once)")
    args = parser.parse_args()
    start = args.start_wind
    if start is None:
        start = float(resolve_kite(args.kite)["wind"]["speed_wind_ref"])
    return PowerCurve(args.kite, args.output_dir).run(parse_winds(args.winds), start,
                                                      sequential=args.sequential)


if __name__ == "__main__":
    raise SystemExit(main())
