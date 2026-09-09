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

"""Exact minimum-energy structural solve — an alternative inner solver for the
PSS/QSM coupled loop.

Same spring physics as the PSS kinetic-damping solve (linear springs,
tension-only ``noncompressive`` elements, pulley arm pairs sharing one
stretch), posed as an unconstrained minimum-potential-energy problem and
solved exactly by IPOPT via CasADi:

    min_x  sum_e (1/2) k_e s_e(x)^2  -  f_ext . x   (+ tiny anchor term)

with ``s_e`` the element stretch (``fmax(0, .)`` for tension-only elements,
combined two-arm stretch for pulley pairs). The gradient of the energy is the
internal force, so the solution satisfies ``f_int + f_ext = 0`` at the free
nodes to solver tolerance (~1e-8), which removes the ~1 N kinetic-damping
leftover that floors the coupled residual at ~4e-4 relative with PSS
(measured 2026-09-09; see the module AGENTS.md).

The PSS ``ParticleSystem`` remains the single owner of structural STATE:
positions, (actuated) rest lengths and (ramped) stiffnesses are read from it
live on every call and the solved positions are written back, so tape
actuation, the stiffness ramp and solver-state handover work unchanged.
CasADi is used internally only — nothing symbolic crosses the module
boundary.

Selected by ``structural_pss.solver: nlp`` in the aerostructural config
(default ``pss`` keeps every existing run byte-identical). Optional keys:
``nlp_tolerance`` (IPOPT tol, default 1e-8), ``nlp_max_iterations`` (default
1000) and ``nlp_anchor_stiffness`` [N/m] (default 1e-3, a tiny position
anchor that pins any force-free fully-slack node whose equilibrium position
would otherwise be indeterminate).
"""

from __future__ import annotations

import logging

import numpy as np
import casadi as ca

from . import structural_pss


def _link_type_name(link) -> str:
    """Normalised link-type string ('default' | 'noncompressive' | 'pulley')."""
    return str(getattr(link, "linktype", "default")).lower().split(".")[-1]


class NlpStructuralSolver:
    """Callable with the exact ``structural_pss.run_pss`` contract.

    ``solver(psystem, f_ext, config_structural_pss)`` returns
    ``(psystem, is_converged, struc_nodes, f_int)`` with ``f_int`` flattened
    in PSS convention, and writes the solved positions (zero velocities) back
    into the particle system.
    """

    def __init__(
        self,
        connectivity,
        link_types,
        pulley_line_indices,
        fixed_node_indices,
        n_nodes: int,
        *,
        ipopt_tolerance: float = 1e-8,
        ipopt_max_iterations: int = 1000,
        anchor_stiffness: float = 1e-3,
    ) -> None:
        self.connectivity = np.asarray(connectivity, dtype=int)
        self.link_types = [str(t) for t in link_types]
        self.fixed_node_indices = [int(i) for i in fixed_node_indices]
        self.n_nodes = int(n_nodes)
        self.n_elements = len(self.connectivity)
        self.anchor_stiffness = float(anchor_stiffness)

        # Pulley arms are appended in pairs by structural_geometry_io (ci-cj
        # then cj-ck), so consecutive entries of pulley_line_indices form one
        # rope. The pair shares a single stretch: s = l_a + l_b - (l0_a + l0_b).
        pli = [int(i) for i in (pulley_line_indices or [])]
        if len(pli) % 2 != 0:
            raise ValueError(
                f"pulley_line_indices must pair up (got {len(pli)} entries)"
            )
        self.pulley_pairs = []
        for i in range(0, len(pli), 2):
            e1, e2 = pli[i], pli[i + 1]
            if e2 != e1 + 1:
                raise ValueError(
                    "pulley arm indices are expected consecutive "
                    f"(got {e1}, {e2}); the pair grouping would be wrong"
                )
            self.pulley_pairs.append((e1, e2))

        self._warm = None
        self._build(ipopt_tolerance, int(ipopt_max_iterations))

    @classmethod
    def from_psystem(
        cls,
        psystem,
        kite_connectivity_arr,
        pulley_line_indices,
        config_structural_pss,
    ) -> "NlpStructuralSolver":
        """Build from the live particle system (link types read off its springs)."""
        link_types = [_link_type_name(link) for link in psystem.springdampers]
        return cls(
            connectivity=kite_connectivity_arr,
            link_types=link_types,
            pulley_line_indices=pulley_line_indices,
            fixed_node_indices=config_structural_pss.get("fixed_point_indices", [0]),
            n_nodes=len(psystem.particles),
            ipopt_tolerance=float(config_structural_pss.get("nlp_tolerance", 1e-8)),
            ipopt_max_iterations=int(
                config_structural_pss.get("nlp_max_iterations", 1000)
            ),
            anchor_stiffness=float(
                config_structural_pss.get("nlp_anchor_stiffness", 1e-3)
            ),
        )

    def _build(self, tolerance: float, max_iterations: int) -> None:
        n, m = self.n_nodes, self.n_elements
        X = ca.MX.sym("X", 3 * n)
        K = ca.MX.sym("K", m)
        L0 = ca.MX.sym("L0", m)
        F = ca.MX.sym("F", 3 * n)
        XA = ca.MX.sym("XA", 3 * n)
        xs = [X[3 * i: 3 * i + 3] for i in range(n)]
        lens = [
            ca.norm_2(xs[self.connectivity[e][0]] - xs[self.connectivity[e][1]])
            for e in range(m)
        ]

        energy = 0
        handled = np.zeros(m, dtype=bool)
        for e1, e2 in self.pulley_pairs:
            stretch = lens[e1] + lens[e2] - (L0[e1] + L0[e2])
            energy += 0.5 * K[e1] * stretch ** 2
            handled[e1] = handled[e2] = True
        for e in range(m):
            if handled[e]:
                continue
            stretch = lens[e] - L0[e]
            if self.link_types[e] == "noncompressive":
                energy += 0.5 * K[e] * ca.fmax(0, stretch) ** 2
            else:
                energy += 0.5 * K[e] * stretch ** 2

        objective = (
            energy
            - ca.dot(F, X)
            + 0.5 * self.anchor_stiffness * ca.sumsqr(X - XA)
        )
        parameters = ca.vertcat(K, L0, F, XA)
        self._solver = ca.nlpsol(
            "structural_nlp",
            "ipopt",
            {"x": X, "p": parameters, "f": objective},
            {
                "ipopt.print_level": 0,
                "print_time": 0,
                "ipopt.sb": "yes",
                "ipopt.tol": tolerance,
                "ipopt.max_iter": max_iterations,
            },
        )
        # Physical internal force: minus the energy gradient (anchor excluded).
        self._f_int_fun = ca.Function(
            "f_int", [X, K, L0], [-ca.gradient(energy, X)]
        )

    def __call__(self, psystem, f_ext, config_structural_pss):
        positions = np.array(
            [particle.x for particle in psystem.particles], dtype=float
        ).reshape(-1)
        stiffness = structural_pss.get_stiffnesses(psystem)
        rest_lengths = np.asarray(
            psystem.extract_rest_length, dtype=float
        ).reshape(-1)
        f_flat = np.asarray(f_ext, dtype=float).reshape(-1)
        p_vec = np.concatenate([stiffness, rest_lengths, f_flat, positions])

        lbx = np.full(3 * self.n_nodes, -np.inf)
        ubx = np.full(3 * self.n_nodes, np.inf)
        for i in self.fixed_node_indices:
            lbx[3 * i: 3 * i + 3] = ubx[3 * i: 3 * i + 3] = positions[3 * i: 3 * i + 3]

        kwargs = {"x0": positions, "p": p_vec, "lbx": lbx, "ubx": ubx}
        if self._warm is not None:
            kwargs["x0"] = self._warm["x"]
            kwargs["lam_x0"] = self._warm["lam_x"]
        solution = self._solver(**kwargs)
        stats = self._solver.stats()
        is_converged = bool(stats.get("success", False))
        if not is_converged:
            logging.warning(
                "structural NLP solve did not report success "
                "(return_status=%s); returning the best iterate",
                stats.get("return_status"),
            )
        self._warm = {"x": solution["x"], "lam_x": solution["lam_x"]}

        solved = np.asarray(solution["x"]).reshape(-1)
        f_int = np.asarray(
            self._f_int_fun(solved, stiffness, rest_lengths)
        ).reshape(-1)
        struc_nodes = solved.reshape(-1, 3)
        for idx, particle in enumerate(psystem.particles):
            particle.update_pos(struc_nodes[idx])
            particle.update_vel(np.zeros(3))

        return psystem, is_converged, struc_nodes, f_int


def resolve_structural_solver(
    config_structural_pss,
    psystem,
    kite_connectivity_arr,
    pulley_line_indices,
):
    """Return (name, callable) for the configured inner structural solver.

    ``structural_pss.solver: "pss"`` (default) returns the kinetic-damping
    ``structural_pss.run_pss``; ``"nlp"`` builds an :class:`NlpStructuralSolver`
    bound to this particle system.
    """
    name = str(config_structural_pss.get("solver", "pss")).lower()
    if name == "pss":
        return name, structural_pss.run_pss
    if name == "nlp":
        return name, NlpStructuralSolver.from_psystem(
            psystem,
            kite_connectivity_arr,
            pulley_line_indices,
            config_structural_pss,
        )
    raise ValueError(
        f"Unknown structural_pss.solver {name!r}; expected 'pss' or 'nlp'."
    )
