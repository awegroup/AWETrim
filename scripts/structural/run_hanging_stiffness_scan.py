# Copyright (c) 2023-2026 Oriol Cayon, Delft University of Technology
#
# SPDX-License-Identifier: Apache-2.0

"""How much softer than the Breukels fit does the measurement say the tubes are?

The hanging case is governed by inflatable-tube bending, and Billow's wing comes
out too open. Rather than tune bridle lengths to close the gap, scale the whole
moment-curvature curve by a single factor and ask what the measured span
implies:

    M(kappa) -> lambda * M_max (1 - exp(-kappa / (lambda kappa_0)))

Scaling ``moment_max`` and ``curvature_scale`` together multiplies the entire
curve by ``lambda`` -- initial slope ``EI_0`` included -- so it is one honest
stiffness knob rather than a reshaping of the law.

The point is not the fitted number but its *pressure dependence*. The Breukels
fit is verified in the thesis only at 0.3, 0.5 and 0.7 bar, while this test runs
at 0.15 and 0.25 bar. If the implied softening is worse at the lower pressure,
that is a signature of extrapolating the fit below its calibrated range, and it
points at a physical cause rather than a fudge factor.

Only the two bare-gravity cases are used (1 and 6), because those are the ones
where the structure alone sets the shape.

    python scripts/structural/run_hanging_stiffness_scan.py
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from pathlib import Path

import numpy as np

from awetrim.structural import MinimumEnergySolver
from awetrim.structural.elements import build_inflatable_beam_elements

from hanging_kite import (
    SHEAR_CORRECTION,
    apply_thesis_bridle,
    build_model,
    canopy_grid,
    extract_lengths,
    load_reference,
    load_vector,
    measured_lengths,
)
from run_hanging_validation import MAX_ROUNDS, MOVE_LIMIT, RESIDUAL_TARGET


def scaled(laws, factor: float):
    """Multiply the whole moment-curvature curve, initial slope included."""
    if factor == 1.0:
        return list(laws)
    return [
        dataclasses.replace(
            law,
            moment_max=law.moment_max * factor,
            bending_stiffness=law.bending_stiffness * factor,
            torsion_c1=law.torsion_c1 * factor,
        )
        for law in laws
    ]


def solve_at(reference, grid, pressure, factor, canopy_stiffness=5000.0):
    model, frames, laws = build_model(
        reference, grid, pressure=pressure, canopy_stiffness=canopy_stiffness,
    )
    if factor != 1.0:
        area = np.pi * (reference.beam_diameter / 2.0) ** 2
        model = model.replaced(build_inflatable_beam_elements(
            model.nodes, reference.beams, scaled(laws, factor), frames,
            axial_stiffness=reference.beam_modulus * area,
            shear_stiffness=SHEAR_CORRECTION * reference.beam_shear_modulus * area,
            name="tubes",
        ))
    solver = MinimumEnergySolver(model, tolerance=1e-8, max_iterations=2000,
                                 move_limit=MOVE_LIMIT)
    forces = load_vector(reference, 0.0, 0.0)
    state, residual = None, np.inf
    for _ in range(MAX_ROUNDS):
        solution = solver.solve(forces, state=state)
        state = solution.state
        residual = solution.residual_norm
        if residual <= RESIDUAL_TARGET:
            break
    return extract_lengths(state.positions), residual


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factors", type=float, nargs="+",
                        default=[1.0, 0.7, 0.5, 0.4, 0.3])
    parser.add_argument("--output", type=Path,
                        default=Path("results/structural/hanging/stiffness_scan.npz"))
    arguments = parser.parse_args()

    reference = apply_thesis_bridle(load_reference())
    grid = canopy_grid(load_reference().nodes)
    measured = measured_lengths()

    cases = {1: 0.15, 6: 0.25}
    spans = {case: [] for case in cases}
    errors = {case: [] for case in cases}

    for case, pressure in cases.items():
        target = measured[case - 1][9]
        print(f"\ncase {case}: {pressure} bar, measured span {target:.3f} m")
        for factor in arguments.factors:
            started = time.perf_counter()
            lengths, residual = solve_at(reference, grid, pressure, factor)
            relative = np.abs(
                (lengths - measured[case - 1]) / measured[case - 1]
            ).mean() * 100.0
            spans[case].append(lengths[9])
            errors[case].append(relative)
            print(f"   lambda={factor:4.2f}  span={lengths[9]:6.3f} m  "
                  f"rel={relative:5.1f}%  res={residual:.1e}  "
                  f"{time.perf_counter()-started:4.0f}s", flush=True)

    print(f"\n{'case':>5} {'p (bar)':>8} {'measured':>9} {'implied lambda':>15}")
    implied = {}
    for case, pressure in cases.items():
        target = measured[case - 1][9]
        order = np.argsort(spans[case])
        value = float(np.interp(target, np.array(spans[case])[order],
                                np.array(arguments.factors)[order]))
        implied[case] = value
        print(f"{case:>5} {pressure:>8} {target:>9.3f} {value:>15.2f}")
    print("\n  If the implied factor is smaller at 0.15 bar than at 0.25 bar, the")
    print("  Breukels fit understates low-pressure softening -- consistent with")
    print("  both test pressures lying below its verified 0.3-0.7 bar range.")

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        arguments.output,
        factors=np.array(arguments.factors),
        cases=np.array(list(cases)),
        pressures=np.array(list(cases.values())),
        spans=np.array([spans[case] for case in cases]),
        errors=np.array([errors[case] for case in cases]),
        implied=np.array([implied[case] for case in cases]),
        measured_span=np.array([measured[case - 1][9] for case in cases]),
    )
    print(f"\n  -> {arguments.output}")


if __name__ == "__main__":
    main()
