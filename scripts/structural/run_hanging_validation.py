# Copyright (c) 2023-2026 Oriol Cayon, Delft University of Technology
#
# SPDX-License-Identifier: Apache-2.0

"""Validate Billow against the measured shape of a hanging V3 kite.

Ten load cases, twelve measured lengths each, scored against ``kite_fem`` on the
identical mesh -- same nodes, same bridles, same pulleys, same inflatable tube
beams. Only the canopy formulation differs (721 tension-only springs versus 392
wrinkling membrane triangles).

What this case actually tests
-----------------------------
The kite hangs from its bridle bar under self-weight and closes up: tip-to-tip
span drops from 8.305 m as built to a measured 4.867 m, while every
trailing-edge segment between struts shortens by 26-43%. The tip-load cases run
the same deformation backwards -- spreading the tips with 2-5 kg reopens the
span to 7.8-8.3 m.

A sensitivity sweep says what governs that, and it is not the canopy. Scaling
the whole moment-curvature curve by 0.3 moves the computed span from 7.19 m to
3.73 m; deleting the canopy outright moves it only to 7.36 m. So this dataset is
a *tube bending* test. Do not read a canopy conclusion off it.

Both codes' tube laws agree closely -- Billow's integrated
``M = M_max (1 - exp(-kappa/kappa_0))`` sits within 1-3% of ``kite_fem``'s
secant ``EI`` over the whole curvature range -- so a shape difference between
them is not a constitutive difference. What does differ is that ``kite_fem``'s
beams *converge* to about 0.23x its own law, with negative bending stiffness in
several cases.

Run from the project root::

    python scripts/structural/run_hanging_validation.py
    python scripts/structural/run_hanging_validation.py --cases 1 6 --no-thesis-bridle
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from awetrim.structural import MinimumEnergySolver
from awetrim.structural.elements import build_inflatable_beam_elements

from hanging_kite import (
    LENGTH_NAMES,
    SHEAR_CORRECTION,
    LOAD_CASES,
    apply_thesis_bridle,
    build_model,
    canopy_grid,
    extract_lengths,
    kite_fem_lengths,
    load_reference,
    load_vector,
    measured_lengths,
    scale_tube_laws,
)

#: Trust-region step (m). Large enough that an easy case still solves in one or
#: two rounds, small enough to stop the runaway step that otherwise overflows
#: the objective on slack-dominated configurations.
MOVE_LIMIT = 0.5
MAX_ROUNDS = 40
RESIDUAL_TARGET = 1e-4          # N, on the worst free node


def solve_case(reference, grid, pressure, tip_load, point_load, *,
               canopy_stiffness, move_limit=MOVE_LIMIT, verbose=False,
               stiffness_factor=1.0):
    """Solve one load case by advancing a trust region to force balance.

    Each ``solve`` is one trust-region step: bounded, so IPOPT cannot answer a
    flat direction with a step of order 1e4, and seeded from the previous one.
    Termination is on the force residual, never on IPOPT's own verdict -- with
    bounds active it reports success while sitting on the box boundary.
    """
    model, frames, laws = build_model(
        reference, grid, pressure=pressure, canopy_stiffness=canopy_stiffness,
    )
    if stiffness_factor != 1.0:
        area = np.pi * (reference.beam_diameter / 2.0) ** 2
        model = model.replaced(build_inflatable_beam_elements(
            model.nodes, reference.beams,
            scale_tube_laws(laws, stiffness_factor), frames,
            axial_stiffness=reference.beam_modulus * area,
            shear_stiffness=SHEAR_CORRECTION * reference.beam_shear_modulus * area,
            name="tubes",
        ))
    solver = MinimumEnergySolver(
        model, tolerance=1e-8, max_iterations=2000, move_limit=move_limit,
    )
    gravity = load_vector(reference, 0.0, 0.0)
    forces = load_vector(reference, tip_load, point_load)

    # Ramp any applied load off the gravity solution rather than cold-starting
    # it. The energy is non-convex, and a cold start under a concentrated
    # centre load walks into a folded local minimum -- case 2 closes to a span
    # of 0.7 m, where the measurement has it slightly WIDER than bare gravity.
    # Continuation from the gravity equilibrium stays on the physical branch.
    extra = forces - gravity
    fractions = [0.0] if not np.any(extra) else [0.0, 0.25, 0.5, 0.75, 1.0]

    state, iterations, residual = None, 0, np.inf
    started = time.perf_counter()
    for fraction in fractions:
        stage = gravity + fraction * extra
        for step in range(MAX_ROUNDS):
            solution = solver.solve(stage, state=state)
            state = solution.state
            iterations += solution.iterations
            residual = solution.residual_norm
            if verbose:
                print(f"      load {fraction:4.2f} round {step + 1:2d}  "
                      f"it={solution.iterations:5d} res={residual:9.2e} "
                      f"span={extract_lengths(state.positions)[9]:7.4f}", flush=True)
            if residual <= RESIDUAL_TARGET:
                break

    return {
        "positions": state.positions,
        "lengths": extract_lengths(state.positions),
        "converged": bool(residual <= RESIDUAL_TARGET),
        "residual": float(residual),
        "rounds": step + 1,
        "iterations": iterations,
        "seconds": time.perf_counter() - started,
    }


def score(model_rows: np.ndarray, measured: np.ndarray) -> dict[str, float]:
    """Per-quantity relative error, which is what the reference metric hides.

    ``kite_fem``'s own ``shapecorrelation`` is a dot-product ratio dominated by
    the two quantities of order 4-8 m, so it reads 0.98-0.9997 even where the
    span is 48% wrong. Relative error per quantity has no such blind spot.
    """
    difference = model_rows - measured
    return {
        "mad_mm": float(np.abs(difference).mean() * 1000.0),
        "mean_rel_pct": float(np.abs(difference / measured).mean() * 100.0),
        "max_rel_pct": float(np.abs(difference / measured).max() * 100.0),
        "span_err_pct": float((difference[..., 9] / measured[..., 9]) * 100.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, nargs="+", default=list(range(1, 11)))
    parser.add_argument("--canopy-stiffness", type=float, default=5000.0,
                        help="membrane E t (N/m); 5000 matches the reference net")
    parser.add_argument("--no-thesis-bridle", action="store_true",
                        help="keep the stored bridle lengths instead of Table B.1")
    parser.add_argument("--remove-ai", action="store_true",
                        help="also cut A_I, which Table B.1 marks N/A but "
                             "Figure B.1 draws as present")
    parser.add_argument("--stiffness-factor", type=float, default=1.0,
                        help="multiply the whole tube moment-curvature curve")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=Path("results/structural/hanging/validation.npz"))
    arguments = parser.parse_args()

    stored = load_reference()
    reference = (stored if arguments.no_thesis_bridle
                 else apply_thesis_bridle(stored, remove_ai=arguments.remove_ai))
    label = ("stored bridle" if arguments.no_thesis_bridle
             else ("thesis bridle, A_I cut" if arguments.remove_ai
                   else "thesis bridle, A_I kept"))
    grid = canopy_grid(stored.nodes)
    measured = measured_lengths()
    reference_model = kite_fem_lengths()

    rows = []
    for case in arguments.cases:
        pressure, tip_load, point_load = LOAD_CASES[case - 1]
        print(f"  case {case:2d}: {pressure} bar, tip {tip_load} kg, point {point_load} kg",
              flush=True)
        result = solve_case(
            reference, grid, pressure, tip_load, point_load,
            canopy_stiffness=arguments.canopy_stiffness, verbose=arguments.verbose,
            stiffness_factor=arguments.stiffness_factor,
        )
        billow = score(result["lengths"], measured[case - 1])
        kfem = score(reference_model[case - 1], measured[case - 1])
        rows.append((case, result, billow, kfem))
        print(f"      billow {billow['mean_rel_pct']:5.1f}%  kite_fem {kfem['mean_rel_pct']:5.1f}%"
              f"   span {result['lengths'][9]:.3f} / {reference_model[case-1][9]:.3f}"
              f" / {measured[case-1][9]:.3f} measured"
              f"   {result['rounds']} rounds, {result['seconds']:.0f}s,"
              f" res {result['residual']:.1e}", flush=True)

    print(f"\n[{label}]  canopy E t = {arguments.canopy_stiffness:.0f} N/m")
    print(f"{'case':>4} {'rel billow':>11} {'rel kfem':>9} {'span billow':>12} "
          f"{'span kfem':>10} {'measured':>9} {'conv':>5} {'s':>6}")
    for case, result, billow, kfem in rows:
        print(f"{case:>4} {billow['mean_rel_pct']:>10.1f}% {kfem['mean_rel_pct']:>8.1f}% "
              f"{result['lengths'][9]:>12.3f} {reference_model[case-1][9]:>10.3f} "
              f"{measured[case-1][9]:>9.3f} {str(result['converged']):>5} "
              f"{result['seconds']:>6.0f}")
    print(f"\n  mean relative error:  billow "
          f"{np.mean([b['mean_rel_pct'] for _, _, b, _ in rows]):.1f}%   kite_fem "
          f"{np.mean([k['mean_rel_pct'] for _, _, _, k in rows]):.1f}%")

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        arguments.output,
        cases=np.array([case for case, *_ in rows]),
        positions=np.stack([result["positions"] for _, result, *_ in rows]),
        lengths=np.stack([result["lengths"] for _, result, *_ in rows]),
        measured=measured[[case - 1 for case, *_ in rows]],
        kite_fem=reference_model[[case - 1 for case, *_ in rows]],
        reference_nodes=stored.nodes,
        seconds=np.array([result["seconds"] for _, result, *_ in rows]),
        residual=np.array([result["residual"] for _, result, *_ in rows]),
        iterations=np.array([result["iterations"] for _, result, *_ in rows]),
        names=np.array(LENGTH_NAMES),
        bridle=label,
        canopy_stiffness=arguments.canopy_stiffness,
        stiffness_factor=arguments.stiffness_factor,
    )
    print(f"  -> {arguments.output}")


if __name__ == "__main__":
    main()
