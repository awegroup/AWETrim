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

"""Static equilibrium by minimising the total potential energy with IPOPT.

    min_X  Pi(X) = U(X) - f_ext . x - m_ext . psi + anchor

Stationarity of ``Pi`` *is* the equilibrium equation, so a converged solve
satisfies ``f_int + f_ext = 0`` to the solver tolerance rather than to whatever
residual a kinetic-damping or explicit relaxation scheme happens to leave
behind.

Only the pinned DOF carry bounds, so IPOPT effectively runs a sparse Newton
method with a filter line search on the mesh-stencil Hessian.

``converged`` is the *physical* verdict: the solve is accepted when the free
nodes are in force balance to ``max(force_tolerance, relative_force_tolerance *
max |f_ext|)``, whether or not IPOPT liked its own termination test. The
relative term matters because an absolute newton means very different things on
a 1 N bridle line and a 600 N benchmark beam. On a stiff, nearly quadratic structure IPOPT often
stops with ``Search_Direction_Becomes_Too_Small`` at an answer that is already
exact to machine precision. The raw solver verdict stays visible as
``ipopt_success`` and ``status``.

Frame updates
-------------
With rotational DOF the run is a short outer loop: solve, absorb the solved
increments into the reference frames, re-solve from ``psi = 0``. Under
force-only loading the energy does not depend on the rotation parameterisation
and the first solve is already exact, so the loop exits immediately. Under
applied nodal *moments* the ``-m . psi`` potential is work-conjugate only at
``psi = 0``, and the loop is what makes the converged answer independent of the
parameterisation. See :mod:`awetrim.structural.model`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import casadi as ca
import numpy as np

from .energy import PotentialEnergy
from .model import StructuralModel, StructuralState
from .symmetry import LinearEqualities

Array = np.ndarray

logger = logging.getLogger(__name__)

DEFAULT_IPOPT_OPTIONS: dict[str, Any] = {
    "ipopt.print_level": 0,
    "print_time": 0,
    "ipopt.sb": "yes",
}


@dataclass(frozen=True)
class StructuralSolution:
    """Outcome of one equilibrium solve."""

    state: StructuralState
    converged: bool
    ipopt_success: bool
    status: str
    iterations: int  # summed over every frame update
    strain_energy: float
    internal_forces: Array
    internal_moments: Array
    residual_forces: Array
    residual_moments: Array
    frame_updates: int
    free_node_mask: Array = field(repr=False, default_factory=lambda: np.empty(0, bool))

    @property
    def residual_norm(self) -> float:
        """Largest out-of-balance force [N] over the free nodes."""
        if not self.free_node_mask.size:
            return float(np.linalg.norm(self.residual_forces, axis=1).max(initial=0.0))
        free = self.residual_forces[self.free_node_mask]
        return float(np.linalg.norm(free, axis=1).max(initial=0.0))


class MinimumEnergySolver:
    """Compile a model once, then solve it for many load cases.

    The compiled graph is fixed by the model *topology*; every numeric input --
    reference frames, element parameters, loads, pinned positions -- is a live
    parameter, so actuation, a stiffness ramp or a load sweep reuse one build.
    """

    def __init__(
        self,
        model: StructuralModel,
        *,
        tolerance: float = 1e-8,
        force_tolerance: float = 1e-6,
        relative_force_tolerance: float = 1e-6,
        max_iterations: int = 1000,
        move_limit: float | None = None,
        anchor_stiffness: float = 1e-6,
        max_frame_updates: int = 3,
        frame_update_tolerance: float = 1e-8,
        parallelization: str = "serial",
        n_threads: int = 1,
        ipopt_options: dict[str, Any] | None = None,
        equalities: LinearEqualities | None = None,
    ) -> None:
        self.model = model
        #: Optional homogeneous linear equalities ``C X = 0`` on the unknowns
        #: (e.g. :func:`awetrim.structural.symmetry.mirror_equalities`). Part of
        #: the compiled problem, so a constrained and an unconstrained solve of
        #: the same model are two solver objects. With constraints active the
        #: reported residual includes their REACTION: it is zero only when the
        #: constrained state is an equilibrium of the unconstrained problem too,
        #: which makes it a check on the constraints rather than a nuisance.
        self.equalities = equalities
        self.move_limit = None if move_limit is None else float(move_limit)
        self.force_tolerance = float(force_tolerance)
        self.relative_force_tolerance = float(relative_force_tolerance)
        self.max_frame_updates = max(1, int(max_frame_updates))
        self.frame_update_tolerance = float(frame_update_tolerance)

        self.energy = PotentialEnergy(
            model,
            anchor_stiffness=anchor_stiffness,
            parallelization=parallelization,
            n_threads=n_threads,
        )

        options = dict(DEFAULT_IPOPT_OPTIONS)
        options["ipopt.tol"] = float(tolerance)
        options["ipopt.max_iter"] = int(max_iterations)
        options.update(ipopt_options or {})
        nlp = dict(self.energy.nlp)
        if equalities is not None:
            nlp["g"] = ca.mtimes(
                equalities.casadi_matrix(model.layout.n_dof), self.energy.unknowns
            )
        self._solver = ca.nlpsol("minimum_energy", "ipopt", nlp, options)
        self._warm: dict[str, Any] | None = None

    def reset_warm_start(self) -> None:
        """Forget the previous solution -- use when the load case jumps."""
        self._warm = None

    def _bounds(self, state: StructuralState) -> tuple[Array, Array]:
        layout = self.model.layout
        lower = np.full(layout.n_dof, -np.inf)
        upper = np.full(layout.n_dof, np.inf)

        if self.move_limit is not None:
            # A trust region, expressed as bounds IPOPT already knows how to
            # enforce. Without it, a structure with slack membrane or slack
            # tension-only cable directions can hand IPOPT a near-zero-curvature
            # search direction, which it answers with a step of order 1e4 -- the
            # objective overflows and the line search collapses before
            # restoration can help. Each solve then advances the seed by at most
            # this much, so ``solve`` becomes one trust-region step.
            seed = self.energy.pack_unknowns(state.positions)
            lower = seed - self.move_limit
            upper = seed + self.move_limit

        for node in self.model.fixed_translation_nodes:
            dof = layout.translation_dof(np.array([node]))[0]
            lower[dof] = upper[dof] = state.positions[node]
        for node in self.model.fixed_rotation_nodes:
            dof = layout.rotation_dof(np.array([node]))[0]
            lower[dof] = upper[dof] = 0.0
        return lower, upper

    def _free_node_mask(self) -> Array:
        mask = np.ones(self.model.n_nodes, dtype=bool)
        mask[list(self.model.fixed_translation_nodes)] = False
        return mask

    def solve(
        self,
        forces: Array,
        *,
        state: StructuralState | None = None,
        moments: Array | None = None,
        model: StructuralModel | None = None,
        warm_start: bool = True,
    ) -> StructuralSolution:
        """Solve for equilibrium under ``forces`` [N] (and optional ``moments`` [Nm]).

        ``state`` seeds the solve and supplies the reference frames; it defaults
        to the model reference configuration. ``model`` may supply updated
        element *parameters* (actuated rest lengths, a ramped stiffness) as long
        as the topology is unchanged.
        """
        active = model if model is not None else self.model
        if active.layout.n_dof != self.model.layout.n_dof:
            raise ValueError(
                "the supplied model has a different DOF count; parameters may "
                "change between solves but topology may not"
            )

        state = state.copy() if state is not None else self.model.initial_state()
        forces = np.asarray(forces, dtype=float).reshape(-1, 3)
        if forces.shape[0] != self.model.n_nodes:
            raise ValueError(
                f"{forces.shape[0]} force rows for {self.model.n_nodes} nodes"
            )

        rotations = None
        status, ipopt_success = "not_run", False
        iterations = 0

        for update in range(self.max_frame_updates):
            lower, upper = self._bounds(state)
            initial = self.energy.pack_unknowns(state.positions, rotations)
            parameters = self.energy.pack_parameters(
                state, active.element_sets, forces, moments, state.positions
            )

            arguments = {"x0": initial, "p": parameters, "lbx": lower, "ubx": upper}
            if self.equalities is not None:
                arguments["lbg"] = arguments["ubg"] = 0.0
            if warm_start and self._warm is not None:
                arguments["lam_x0"] = self._warm["lam_x"]
            solution = self._solver(**arguments)

            statistics = self._solver.stats()
            ipopt_success = bool(statistics.get("success", False))
            status = str(statistics.get("return_status", "unknown"))
            iterations += int(statistics.get("iter_count", 0))
            self._warm = {"lam_x": solution["lam_x"]}

            unknowns = np.asarray(solution["x"], dtype=float).reshape(-1)
            translations, increments = self.energy.unpack_unknowns(unknowns)
            state = state.absorbed(translations, increments)
            rotations = None

            if not ipopt_success:
                logger.debug(
                    "IPOPT returned %s after %d iterations; the force residual "
                    "decides whether this is accepted",
                    status,
                    iterations,
                )
                break
            if increments.size == 0 or float(np.abs(increments).max()) < self.frame_update_tolerance:
                break
        else:
            update = self.max_frame_updates - 1
            logger.info(
                "rotation increments still %.3e after %d frame updates",
                float(np.abs(increments).max()) if increments.size else 0.0,
                self.max_frame_updates,
            )

        parameters = self.energy.pack_parameters(
            state, active.element_sets, forces, moments, state.positions
        )
        settled = self.energy.pack_unknowns(state.positions)
        internal = np.asarray(
            self.energy.internal_load(settled, parameters), dtype=float
        ).reshape(-1)
        split = self.model.layout.n_translation_dof
        internal_forces = internal[:split].reshape(-1, 3)
        internal_moments = internal[split:].reshape(-1, 3)

        applied_moments = (
            np.zeros_like(internal_moments)
            if moments is None
            else np.asarray(moments, dtype=float).reshape(-1, 3)
        )
        # IPOPT can stop with Search_Direction_Becomes_Too_Small on a stiff,
        # nearly quadratic structure whose answer is already exact to machine
        # precision. The force balance is the physical question, so accept on
        # it -- and keep the raw solver verdict alongside rather than hiding it.
        residual_forces = internal_forces + forces
        free = self._free_node_mask()
        residual_norm = (
            float(np.linalg.norm(residual_forces[free], axis=1).max(initial=0.0))
            if free.any()
            else 0.0
        )
        # Scale the acceptance with the load: an absolute newton means very
        # different things on a 1 N bridle and a 600 N benchmark beam.
        load_scale = float(np.linalg.norm(forces, axis=1).max(initial=0.0))
        accept_at = max(
            self.force_tolerance, self.relative_force_tolerance * load_scale
        )
        # With a move limit the iterate can sit on the trust-region boundary,
        # where IPOPT reports success for the *boxed* problem while the
        # structure is nowhere near equilibrium. Equalities are the same trap:
        # IPOPT solves the CONSTRAINED problem, whose optimum may be held by a
        # reaction. Force balance is then the only admissible verdict.
        if self.move_limit is not None or self.equalities is not None:
            converged = residual_norm <= accept_at
        else:
            converged = ipopt_success or residual_norm <= accept_at
        if not converged:
            logger.warning(
                "minimum-energy solve did not converge (status=%s, %d iterations, "
                "residual %.3e N > %.3e N); returning the best iterate",
                status,
                iterations,
                residual_norm,
                accept_at,
            )

        return StructuralSolution(
            state=state,
            converged=converged,
            ipopt_success=ipopt_success,
            status=status,
            iterations=iterations,
            strain_energy=float(self.energy.total_energy(settled, parameters)),
            internal_forces=internal_forces,
            internal_moments=internal_moments,
            residual_forces=residual_forces,
            residual_moments=internal_moments + applied_moments,
            frame_updates=update + 1,
            free_node_mask=free,
        )
