"""Is the unsteered kite's asymmetry a solver artefact or a genuine bifurcation?

Structure alone, under the exactly mirror-symmetric first-iteration aerodynamic
load (the same load as ``check_structural_symmetry``), frozen as a dead load.
Four solves of the same model with the same numerics:

``free``       unconstrained, from the built state -- what the coupled run does
``symmetric``  constrained to the mirror-symmetric subspace, from the built
               state: positions ``x_p(i) = M x_i`` AND rotation increments
               ``psi_p(i) = -M psi_i`` (pseudovectors), over ALL nodes
``released``   unconstrained, seeded with the symmetric solution
``kicked``     unconstrained, seeded with the symmetric solution plus a small
               push along the free solution's antisymmetric part

and, at the symmetric solution, the second-order test that the ``released``
solve alone cannot give (IPOPT checks first-order optimality only, so it would
happily stop on a symmetric SADDLE): the lowest eigenvalues of the tangent
stiffness on the antisymmetric and the symmetric subspaces.

How to read it:

(a) solver artefact -- the symmetric solve converges to force balance (its
    constraint reactions vanish), ``Pi_sym <= Pi_free``, ``released`` and
    ``kicked`` return to it, and the antisymmetric stiffness is positive.
(b) symmetry breaking -- the antisymmetric stiffness is negative at the
    symmetric equilibrium and the kick runs to an asymmetric state of lower
    ``Pi``: the mirror pair of asymmetric states are the physical minima.

``Pi = U - f . x`` is the total potential of the dead load (the solver's weak
anchor is excluded; it is 1e-6 N/m).

Two model checks run first, because the energy comparison means nothing unless
the model itself is mirror-symmetric -- and positions alone cannot show that:

- the reference FRAMES of every beam node against the mirror of its partner's
  (a roll mismatch about a tube axis is invisible in the node positions);
- the internal load at a random mirror-symmetric configuration, which must be
  mirror-equivariant: ``f_p = M f_i`` and ``m_p = -M m_i``.

Usage (from project root):
    python scripts/aerostructural/studies/check_symmetric_equilibrium.py
    python scripts/aerostructural/studies/check_symmetric_equilibrium.py --scale 0.1
"""

import argparse
import time
from pathlib import Path

import numpy as np

from awetrim.aerostructural.logging_config import *  # noqa: F401,F403
from awetrim.aerostructural.billow import structural_billow as sb
from billow import StructuralState
from billow.symmetry import REFLECTION_Y, frame_mirror_mismatch
from awetrim.aerostructural.case import DEFAULT_KITE_NAME

from check_mirror_asymmetry import decompose
from check_structural_symmetry import symmetric_load
from plot_billow_geometry import rebuild
from run_chain_depower_BILLOW import build_once

M = REFLECTION_Y


def slot_partners(layout, partner):
    """Rotational slot of each slot's mirror partner."""
    return np.array([layout.rotation_slot(int(partner[int(node)]))
                     for node in layout.rotational_nodes])


def mirror_operator(layout, partner):
    """``P`` on the unknown vector: the mirror image of a configuration.

    Translations map as vectors, rotation increments as pseudovectors. ``P`` is
    an involution; its +1 eigenspace is the symmetric subspace.
    """
    operator = np.zeros((layout.n_dof, layout.n_dof))
    for node in range(layout.n_nodes):
        source = layout.translation_dof(np.array([node]))[0]
        target = layout.translation_dof(np.array([int(partner[node])]))[0]
        operator[np.ix_(target, source)] = M
    for node in layout.rotational_nodes:
        source = layout.rotation_dof(np.array([int(node)]))[0]
        target = layout.rotation_dof(np.array([int(partner[int(node)])]))[0]
        operator[np.ix_(target, source)] = -M
    return operator


def potential(solution, forces):
    """``Pi = U - f . x`` at a solution, and the strain energy ``U``."""
    work = float(np.sum(forces * solution.state.positions))
    return solution.strain_energy - work, solution.strain_energy


def asymmetry(positions, built, grid, partner):
    per_pair, angle, _ = decompose(positions, built, grid, partner)
    return (1e3 * max(p[2] for p in per_pair), 1e3 * max(p[3] for p in per_pair), angle)


def check_model_symmetry(structure, partner, seed=0):
    """Frame consistency and mirror-equivariance of the internal load."""
    model = structure.model
    layout = model.layout
    state = structure.state
    slots = slot_partners(layout, partner)

    roll = np.degrees(frame_mirror_mismatch(state.frames, layout, partner))
    print(f"reference frames vs mirrored partner: max {roll.max():.3e} deg over "
          f"{len(roll)} slots ({int((roll > 1e-6).sum())} above 1e-6 deg)")

    rng = np.random.default_rng(seed)
    push = 0.01 * rng.normal(size=state.positions.shape)
    push = 0.5 * (push + push[partner] @ M.T)
    push[list(model.fixed_translation_nodes)] = 0.0
    turn = 0.01 * rng.normal(size=(len(slots), 3))
    turn = 0.5 * (turn - turn[slots] @ M.T)

    energy = structure.solver.energy
    unknowns = energy.pack_unknowns(state.positions + push, turn)
    parameters = energy.pack_parameters(
        state, model.element_sets, np.zeros_like(state.positions), None, state.positions
    )
    internal = np.asarray(energy.internal_load(unknowns, parameters)).reshape(-1)
    split = layout.n_translation_dof
    forces, moments = internal[:split].reshape(-1, 3), internal[split:].reshape(-1, 3)
    force_error = np.linalg.norm(forces[partner] - forces @ M.T, axis=1).max()
    moment_error = np.linalg.norm(moments[slots] + moments @ M.T, axis=1).max()
    print(f"internal load at a symmetric configuration: force mirror error "
          f"{force_error:.3e} N (of {np.abs(forces).max():.3e}), moment mirror error "
          f"{moment_error:.3e} N m (of {np.abs(moments).max():.3e})")


def subspace_spectra(structure, state, forces, partner, n_show=6):
    """Lowest tangent-stiffness eigenvalues on the antisymmetric / symmetric subspaces."""
    model = structure.model
    layout = model.layout
    energy = structure.solver.energy
    hessian = np.asarray(energy.tangent_stiffness()(
        energy.pack_unknowns(state.positions),
        energy.pack_parameters(state, model.element_sets, forces, None, state.positions),
    ))
    free = np.ones(layout.n_dof, dtype=bool)
    for node in model.fixed_translation_nodes:
        free[layout.translation_dof(np.array([node]))[0]] = False
    operator = mirror_operator(layout, partner)[np.ix_(free, free)]
    commutator = np.abs(operator @ hessian[np.ix_(free, free)]
                        - hessian[np.ix_(free, free)] @ operator).max()
    sign, basis = np.linalg.eigh(operator)
    stiffness = hessian[np.ix_(free, free)]
    spectra = {}
    for name, mask in (("antisymmetric", sign < 0), ("symmetric", sign > 0)):
        block = basis[:, mask].T @ stiffness @ basis[:, mask]
        values, vectors = np.linalg.eigh(0.5 * (block + block.T))
        spectra[name] = (values, basis[:, mask] @ vectors)
    return spectra, commutator, np.flatnonzero(free)


def describe_mode(vector, free_dof, layout, grid, canopy_nodes, top=4):
    """Where a mode lives: its largest translational components, by node role."""
    full = np.zeros(layout.n_dof)
    full[free_dof] = vector
    motion = np.linalg.norm(full[: layout.n_translation_dof].reshape(-1, 3), axis=1)
    rotation = np.linalg.norm(full[layout.n_translation_dof:].reshape(-1, 3), axis=1)
    where = {int(n): (int(i), int(k)) for (i, k), n in np.ndenumerate(grid)}
    labels = []
    for node in np.argsort(motion)[::-1][:top]:
        if node in where:
            i, k = where[node]
            role = f"section {i} x/c-station {k}"
        elif node in canopy_nodes:
            role = "quad centre"
        else:
            role = "bridle"
        labels.append(f"{node} ({role}, {motion[node] / motion.max():.2f})")
    share = float(np.sum(rotation**2))
    return ", ".join(labels) + f"; rotational share {share:.2f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", type=float, default=1.0,
                        help="fraction of the first-iteration aerodynamic load")
    parser.add_argument("--kick", type=float, default=0.005,
                        help="largest nodal push of the kicked solve [m]")
    parser.add_argument("--wind", type=float, default=4.2)
    parser.add_argument("--pattern", default="cross")
    parser.add_argument("--refine", type=int, default=1)
    parser.add_argument("--panels-per-section", type=int, default=2)
    parser.add_argument("--kite", default=DEFAULT_KITE_NAME)
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[3]
    settings = {"canopy_pattern": args.pattern, "canopy_refinement": args.refine}
    shared = build_once(project, args.kite, args.panels_per_section,
                        {"aerodynamic": {"gamma_loop_type": "base"}})
    shared["config"]["wind_speed_wind_ref"] = float(args.wind)
    config = dict(shared["config"].get("structural_billow") or {})
    config.update(settings)
    resolved = sb.resolve_config(config)

    def fresh():
        return rebuild(project, args.kite, args.panels_per_section, settings)

    structure = fresh()
    built = structure.model.nodes.copy()
    grid = structure.grid
    equalities, partner = sb.symmetric_equalities(structure)
    canopy_nodes = set(np.unique(structure.model.element_set(sb.CANOPY).connectivity).tolist())
    print(f"\nmodel: {structure.model.n_nodes} nodes, {structure.model.layout.n_dof} DOF, "
          f"{equalities.n_rows} symmetry equalities")
    check_model_symmetry(structure, partner)

    forces = args.scale * symmetric_load(
        shared, args.pattern, args.refine, built, grid,
        structure.model.element_set(sb.CANOPY).connectivity,
    )
    print(f"load: total {np.linalg.norm(forces.sum(axis=0)):.1f} N, mirror mismatch "
          f"{np.linalg.norm(forces - forces[partner] @ M.T, axis=1).max():.2e} N\n")

    spectra, commutator, _ = subspace_spectra(structure, structure.state, forces, partner)
    print(f"BUILT state tangent stiffness ([P, K] = {commutator:.1e}):")
    for name, (values, _) in spectra.items():
        print(f"  {name:13s} lowest: " + " ".join(f"{v:10.3e}" for v in values[:6]))

    rows = []

    def record(name, structure, seconds):
        solution = structure.last_solution
        pi, strain = potential(solution, forces)
        glob, intr, angle = asymmetry(solution.state.positions, built, grid, partner)
        rows.append((name, pi, strain, solution.residual_norm, glob, intr, angle, seconds))
        return solution.state

    def solve(structure, name):
        started = time.perf_counter()
        sb.run_billow(structure, forces.ravel(), config)
        return record(name, structure, time.perf_counter() - started)

    # free: exactly what the coupled run does on its first iteration
    free_state = solve(structure, "free")

    # symmetric: same model, same numerics, constrained
    constrained = fresh()
    unconstrained_solver = constrained.solver
    constrained.solver = sb.build_solver(constrained.model, resolved, equalities)
    symmetric_state = solve(constrained, "symmetric")
    x_sym = constrained.solver.energy.pack_unknowns(symmetric_state.positions)
    print(f"symmetric solve: |C X| = {np.abs(equalities.residual(x_sym)).max():.2e}, "
          f"frames vs mirror {np.degrees(frame_mirror_mismatch(symmetric_state.frames, constrained.model.layout, partner)).max():.2e} deg")

    # released: drop the constraints, seed with the symmetric solution
    constrained.solver = unconstrained_solver
    constrained.solver.reset_warm_start()
    constrained.state = symmetric_state.copy()
    solve(constrained, "released")

    # kicked: push along the free solution's antisymmetric part
    antisymmetric = 0.5 * (free_state.positions - free_state.positions[partner] @ M.T)
    antisymmetric *= args.kick / np.linalg.norm(antisymmetric, axis=1).max()
    constrained.solver.reset_warm_start()
    constrained.state = StructuralState(symmetric_state.positions + antisymmetric,
                                        symmetric_state.frames.copy())
    solve(constrained, "kicked")

    print(f"\n{'solve':10s} {'Pi [J]':>14s} {'U [J]':>12s} {'residual N':>11s} | "
          f"{'global mm':>9s} {'intrinsic mm':>12s} {'rigid deg':>9s} | {'s':>5s}")
    for name, pi, strain, residual, glob, intr, angle, seconds in rows:
        print(f"{name:10s} {pi:14.6f} {strain:12.6f} {residual:11.3e} | "
              f"{glob:9.2f} {intr:12.2f} {angle:9.4f} | {seconds:5.1f}")
    pi_free, pi_sym = rows[0][1], rows[1][1]
    print(f"\nPi_sym - Pi_free = {pi_sym - pi_free:+.6e} J")

    spectra, commutator, free_dof = subspace_spectra(constrained, symmetric_state, forces, partner)
    print(f"\nSYMMETRIC equilibrium tangent stiffness ([P, K] = {commutator:.1e}):")
    for name, (values, vectors) in spectra.items():
        print(f"  {name:13s} lowest: " + " ".join(f"{v:10.3e}" for v in values[:6]))
        print(f"    lowest mode: {describe_mode(vectors[:, 0], free_dof, constrained.model.layout, grid, canopy_nodes)}")


if __name__ == "__main__":
    main()
