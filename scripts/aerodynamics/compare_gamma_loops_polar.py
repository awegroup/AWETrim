"""Polar comparison of the VSM inner circulation loops on one kite.

Sweeps the angle of attack for ``base`` (production tolerance), ``anderson``
and ``casadi_newton`` and plots CL, CD, the drag polar and the inner-loop cost
per point, once cold-seeded (every point independent) and once as an
alpha-continuation (each point seeded with the previous converged circulation,
the way a trim or a sweep uses the solver). Unconverged points are hollow.

Example:
    python scripts/aerodynamics/compare_gamma_loops_polar.py --n-panels 27
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import add_common_arguments, build_body, output_dir  # noqa: E402

logging.disable(logging.WARNING)

# Fixed categorical order (blue, orange, aqua): validated colorblind-safe set.
LOOPS = {
    "base 1e-6 (production)": dict(gamma_loop_type="base", allowed_error=1e-6, color="#2a78d6"),
    "anderson 1e-8": dict(gamma_loop_type="anderson", allowed_error=1e-8, color="#eb6834"),
    "casadi_newton 1e-8": dict(gamma_loop_type="casadi_newton", allowed_error=1e-8, color="#1baf7a"),
}


def sweep(body, solver, alphas, umag, warm):
    from VSM.core.Solver import Solver  # noqa: F401

    rows = []
    prev = None
    for alpha in alphas:
        body.va_initialize(Umag=umag, angle_of_attack=float(alpha))
        t0 = time.perf_counter()
        res = solver.solve(body, gamma_distribution=prev if warm else None)
        dt = time.perf_counter() - t0
        rows.append(
            dict(
                alpha=alpha,
                cl=res["cl"],
                cd=res["cd"],
                conv=bool(res["gamma_converged"]),
                it=solver.last_iterations,
                t=dt,
                fallback=bool(getattr(solver, "last_fallback", False)),
            )
        )
        prev = np.asarray(res["gamma_distribution"], dtype=float)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_common_arguments(ap)
    ap.add_argument("--umag", type=float, default=15.0)
    ap.add_argument("--alpha-min", type=float, default=-5.0)
    ap.add_argument("--alpha-max", type=float, default=30.0)
    ap.add_argument("--alpha-step", type=float, default=1.0)
    ap.add_argument("--no-av", action="store_true", help="Disable the Li/Gaunaa artificial viscosity.")
    args = ap.parse_args()
    from VSM.core.Solver import Solver

    body, _ = build_body(args)
    av = not args.no_av
    alphas = np.arange(args.alpha_min, args.alpha_max + 0.5 * args.alpha_step, args.alpha_step)
    out = output_dir(args, "gamma_loop_polar")

    results = {}
    for name, spec in LOOPS.items():
        solver = Solver(
            gamma_loop_type=spec["gamma_loop_type"],
            allowed_error=spec["allowed_error"],
            is_with_artificial_viscosity=av,
        )
        for warm in (False, True):
            rows = sweep(body, solver, alphas, args.umag, warm)
            results[(name, warm)] = rows
            conv = sum(r["conv"] for r in rows)
            print(
                f"{name:24s} {'continuation' if warm else 'cold':12s} converged {conv}/{len(rows)}  "
                f"fallbacks {sum(r['fallback'] for r in rows)}  iterations median {np.median([r['it'] for r in rows]):.0f}  "
                f"time {sum(r['t'] for r in rows):.2f} s"
            )

    fig, axes = plt.subplots(2, 4, figsize=(17, 7.5), constrained_layout=True)
    for row, warm in enumerate((False, True)):
        ax_cl, ax_cd, ax_pol, ax_it = axes[row]
        for name, spec in LOOPS.items():
            rows = results[(name, warm)]
            a = np.array([r["alpha"] for r in rows])
            cl = np.array([r["cl"] for r in rows])
            cd = np.array([r["cd"] for r in rows])
            it = np.array([r["it"] for r in rows], dtype=float)
            conv = np.array([r["conv"] for r in rows])
            c = spec["color"]
            for ax, y in ((ax_cl, cl), (ax_cd, cd)):
                ax.plot(a, y, "-", color=c, lw=2, label=name)
                ax.plot(a[conv], y[conv], "o", color=c, ms=4)
                ax.plot(a[~conv], y[~conv], "o", mfc="white", mec=c, ms=6, mew=1.5)
            ax_pol.plot(cd, cl, "-", color=c, lw=2, label=name)
            ax_pol.plot(cd[conv], cl[conv], "o", color=c, ms=4)
            ax_pol.plot(cd[~conv], cl[~conv], "o", mfc="white", mec=c, ms=6, mew=1.5)
            ax_it.semilogy(a, it, "-", color=c, lw=2, label=name)
            ax_it.plot(a[~conv], it[~conv], "o", mfc="white", mec=c, ms=6, mew=1.5)
        seed = "alpha continuation (seeded from the previous point)" if warm else "cold seed (every point independent)"
        ax_cl.set_title(f"CL, {seed}", fontsize=10, loc="left")
        ax_cd.set_title("CD", fontsize=10, loc="left")
        ax_pol.set_title("drag polar", fontsize=10, loc="left")
        ax_it.set_title("inner-loop iterations (hollow = not converged)", fontsize=10, loc="left")
        for ax in (ax_cl, ax_cd, ax_it):
            ax.set_xlabel("angle of attack (deg)")
        ax_pol.set_xlabel("CD")
        ax_cl.set_ylabel("CL")
        ax_cd.set_ylabel("CD")
        ax_pol.set_ylabel("CL")
        ax_it.set_ylabel("iterations")
        for ax in axes[row]:
            ax.grid(True, color="#e6e6e3", lw=0.8)
            ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=9)
    kite = Path(args.config_folder).name
    fig.suptitle(
        f"{kite}: VSM circulation loops, {body.n_panels} panels, Umag {args.umag:g} m/s, "
        f"artificial viscosity {'on' if av else 'off'}",
        fontsize=12,
    )
    for ext in ("png", "pdf"):
        fig.savefig(out / f"gamma_loop_polar.{ext}", dpi=150)
    print("saved", out / "gamma_loop_polar.png")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
