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

"""Inflatable tube beam: the ASKITE empirical fits, recast as a strain energy.

An inflated LEI tube is not linear-elastic. Its bending stiffness falls away as
the compressed side starts to wrinkle, until the section carries a limiting
moment and then collapses. ``kite_fem``'s ``BeamElement`` captures that with
empirical fits (coefficients ``C1``-``C19``, from the ASKITE work) and applies
them as *secant* stiffnesses: ``EI`` and ``GJ`` are recomputed each iteration
from the current deflection and twist.

That is consistent inside a Newton residual, but it is **not** a potential.
Substituting a state-dependent ``EI(kappa)`` into ``1/2 EI kappa^2`` silently
drops the ``dEI/dkappa`` terms, so the gradient would no longer be the internal
moment and a minimum-energy solve would converge to the wrong equilibrium. The
fits therefore have to be integrated, not substituted:

    W(kappa) = int_0^kappa M(x) dx

Both fits integrate in closed form, which is what this module does.

From tip-load fit to moment-curvature law
-----------------------------------------
The bending fit is written as tip load against normalised tip deflection
``v = delta / L`` of a **one-metre** cantilever (``kite_fem`` hard-codes
``L = 1`` when it inverts the fit):

    P(v) = D (1 - exp(-(N/D) v))

For a linear cantilever ``delta = P L^3 / 3EI`` and ``kappa_root = P L / EI``,
so ``kappa = 3 v / L``, and at the calibration length ``L = 1`` simply
``kappa = 3 v``. Meanwhile the root moment is ``M = P L = P``. Substituting
turns the fit into an intrinsic constitutive law:

    M(kappa) = M_max (1 - exp(-EI_0 kappa / M_max))
    EI_0 = N / 3          initial bending stiffness   [N m^2]
    M_max = D             limiting moment             [N m]

with the small-curvature limit ``M -> EI_0 kappa`` and the large-curvature
limit ``M -> M_max``, both by construction. Its energy is

    W_b(kappa) = M_max [ |kappa| - k0 (1 - exp(-|kappa|/k0)) ],  k0 = M_max/EI_0

which is quadratic (``EI_0 kappa^2 / 2``) near zero, so it is smooth there
despite the ``|kappa|``.

Recasting this way also removes a length inconsistency: ``kite_fem`` infers
``EI = P / (3 v)``, which equals the true ``EI`` only for a one-metre element,
because ``v`` is already normalised by the element length. A moment-curvature
law is length-independent by construction and applies to any element size.

The torsion fit is already intrinsic -- ``kite_fem`` measures twist per unit
length -- so it integrates directly:

    T(omega) = c1 arctan(c2 omega)
    W_t(omega) = c1 [ omega arctan(c2 omega) - ln(1 + c2^2 omega^2) / (2 c2) ]

with ``GJ_0 = c1 c2`` and a limiting torque ``c1 pi / 2``.

Collapse
--------
``kappa_collapse`` (from the ``C9``-``C12`` fit) is reported, not enforced, and
the law saturates rather than dropping.

That is a modelling choice, not a necessity. A dropping post-collapse moment
does *not* make the energy unbounded: ``M >= 0`` throughout, so ``W = int M dk``
still increases -- it merely turns concave, i.e. non-convex. A law that follows
collapse, fixed uniquely by matching both the fitted initial stiffness and the
fitted collapse curvature,

    W(k) = EI_0 kc^2 [1 - (1 + k/kc) exp(-k/kc)],   M(k) = EI_0 k exp(1 - k/kc)

was tried on the hanging V3 kite and converged without difficulty (56-77
iterations, residual 1e-9). It is not adopted because it changed the answer by
20 mm: those beams run at 9% of ``M_max`` and 95% of ``EI_0``, on the linear
part of the law, and a flying kite carries a distributed bridle-reacted load
that keeps its tubes there too. Saturation is adequate because collapse is not
reached, not because collapse is handled -- and ``kappa_collapse`` is reported
so that assumption stays checkable. See :func:`inflatable_beam_state` and
``docs/hanging_validation/``.

A further caveat on transferability: the fit comes from a *free* 1 m cantilever,
able to ovalise and wrinkle over its whole length. A kite leading edge is
restrained by canopy along its span and stiffened by struts, and local
indentation is a cross-section collapse that no one-dimensional
moment-curvature element can represent at all.

Biaxial bending uses the resultant curvature ``sqrt(omega_2^2 + omega_3^2)``,
which is the right isotropic extension for an axisymmetric tube.

Axial and shear stiffness are **not** covered by the fits and stay linear; take
them from the tube geometry.

Reference: the ``C1``-``C19`` coefficients are those in
``kite_fem/BeamElement.py`` and ``kite_fem/examples/FEM_beam_verification.py``
(ASKITE, Poland/Roeleveld, TU Delft).
"""

from __future__ import annotations

from dataclasses import dataclass

import casadi as ca
import numpy as np

from ..rotations import cayley
from .base import ElementSet, element_frame, element_rotation, element_translations
from .beam import beam_strains

Array = np.ndarray

PARAM_NAMES: tuple[str, ...] = (
    "rest_length",
    "ea",
    "ga_2",
    "ga_3",
    "bending_stiffness",   # EI_0 [N m^2]
    "moment_max",          # M_max [N m]
    "torsion_c1",          # c1 [N m]
    "torsion_c2",          # c2 [m/rad]
    "gamma_0_1",
    "gamma_0_2",
    "gamma_0_3",
    "omega_0_1",
    "omega_0_2",
    "omega_0_3",
)

#: Bending fit, tip load against normalised tip deflection of a 1 m cantilever.
BENDING_COEFFICIENTS = dict(
    C1=6582.82, C2=-272.43, C3=40852.38, C4=14.31,
    C5=271865251.42, C6=215.93, C7=14021.79, C8=-589.05,
)
#: Collapse fit, normalised tip deflection at which the tube wrinkles through.
COLLAPSE_COEFFICIENTS = dict(C9=322.55, C10=0.0239, C11=5.3833, C12=0.0461)
#: Torsion fit, torque against twist per unit length.
TORSION_COEFFICIENTS = dict(
    C13=1467.0, C14=40.908, C15=-191.8, C16=47.406,
    C17=-17703.0, C18=358.05, C19=0.0918,
)

#: Curvature of the calibration cantilever per unit normalised tip deflection.
#: ``kappa = 3 v / L`` and the fits were taken at ``L = 1 m``.
CURVATURE_PER_DEFLECTION = 3.0


@dataclass(frozen=True)
class InflatableTubeLaw:
    """Constitutive law of one inflated tube at a given diameter and pressure.

    ``bending_stiffness`` and ``moment_max`` define the saturating
    moment-curvature law, ``torsion_c1``/``torsion_c2`` the arctangent torsion
    law, and ``curvature_collapse`` the edge of validity.
    """

    diameter: float
    pressure: float          # bar, as the fits are written
    bending_stiffness: float
    moment_max: float
    torsion_c1: float
    torsion_c2: float
    curvature_collapse: float

    @classmethod
    def from_fit(cls, diameter: float, pressure: float) -> "InflatableTubeLaw":
        """Evaluate the ASKITE fits for a tube of ``diameter`` [m] at ``pressure`` [bar]."""
        if pressure <= 0.0:
            raise ValueError(
                f"inflation pressure must be positive (got {pressure} bar); the "
                "torsion fit contains ln(p)"
            )
        radius = 0.5 * float(diameter)
        b, k, t = BENDING_COEFFICIENTS, COLLAPSE_COEFFICIENTS, TORSION_COEFFICIENTS

        limit_load = (b["C1"] * radius + b["C2"]) * pressure**2 + (
            b["C3"] * radius**3 + b["C4"]
        )
        initial_slope = (b["C5"] * radius**5 + b["C6"]) * pressure + (
            b["C7"] * radius + b["C8"]
        )
        if limit_load <= 0.0 or initial_slope <= 0.0:
            raise ValueError(
                f"the bending fit is not physical for d={diameter} m, p={pressure} bar "
                f"(limit load {limit_load:.3g} N, initial slope {initial_slope:.3g} N); "
                "the fits are calibrated for roughly 0.1-0.3 m tubes at 0.1-1 bar"
            )

        deflection_collapse = (k["C9"] * radius**4 + k["C10"]) * pressure + (
            k["C11"] * radius**2 + k["C12"]
        )
        return cls(
            diameter=float(diameter),
            pressure=float(pressure),
            # M = P L and kappa = 3 v / L, both at the L = 1 m calibration length.
            bending_stiffness=initial_slope / CURVATURE_PER_DEFLECTION,
            moment_max=limit_load,
            torsion_c1=(t["C13"] * radius + t["C14"]) * pressure
            + (t["C15"] * radius + t["C16"]),
            torsion_c2=(t["C17"] * radius**4) * np.log(pressure)
            + (t["C18"] * radius**3 + t["C19"]),
            curvature_collapse=CURVATURE_PER_DEFLECTION * deflection_collapse,
        )

    @property
    def curvature_scale(self) -> float:
        """``k0 = M_max / EI_0``: the curvature at which the moment saturates."""
        return self.moment_max / self.bending_stiffness

    @property
    def torsion_stiffness(self) -> float:
        """Initial ``GJ`` [N m^2], the small-twist limit of the torsion law."""
        return self.torsion_c1 * self.torsion_c2

    @property
    def torque_max(self) -> float:
        """Limiting torque [N m], the large-twist limit of the arctangent law."""
        return 0.5 * np.pi * self.torsion_c1

    def moment(self, curvature) -> Array:
        """Bending moment [N m] at ``curvature`` [1/m]. Odd in ``curvature``."""
        curvature = np.asarray(curvature, dtype=float)
        return np.sign(curvature) * self.moment_max * (
            1.0 - np.exp(-np.abs(curvature) / self.curvature_scale)
        )

    def bending_energy_density(self, curvature) -> Array:
        """Strain energy per unit length [J/m] stored in bending."""
        magnitude = np.abs(np.asarray(curvature, dtype=float))
        scale = self.curvature_scale
        return self.moment_max * (
            magnitude - scale * (1.0 - np.exp(-magnitude / scale))
        )

    def torque(self, twist_rate) -> Array:
        """Torque [N m] at ``twist_rate`` [rad/m]."""
        return self.torsion_c1 * np.arctan(self.torsion_c2 * np.asarray(twist_rate, float))

    def torsion_energy_density(self, twist_rate) -> Array:
        """Strain energy per unit length [J/m] stored in torsion."""
        twist_rate = np.asarray(twist_rate, dtype=float)
        scaled = self.torsion_c2 * twist_rate
        return self.torsion_c1 * (
            twist_rate * np.arctan(scaled)
            - np.log1p(scaled**2) / (2.0 * self.torsion_c2)
        )

    def as_row(self) -> Array:
        """The four constitutive columns, in ``PARAM_NAMES`` order."""
        return np.array(
            [
                self.bending_stiffness,
                self.moment_max,
                self.torsion_c1,
                self.torsion_c2,
            ],
            dtype=float,
        )


def _bending_energy(magnitude, bending_stiffness, moment_max):
    """``M_max [u - k0 (1 - exp(-u/k0))]`` in CasADi, for a positive ``u``."""
    scale = moment_max / bending_stiffness
    return moment_max * (magnitude - scale * (1.0 - ca.exp(-magnitude / scale)))


@dataclass(frozen=True)
class InflatableBeamKernel:
    """Two-node inflatable tube: geometrically exact kinematics, ASKITE constitutive law.

    Shares :func:`~awetrim.structural.elements.beam.beam_strains` with the linear
    beam, so the finite-rotation behaviour is identical and only the material
    response differs.

    ``curvature_regularization`` [1/m] rounds the resultant-curvature square root
    at zero. The bending energy is quadratic there, so the value and gradient are
    unaffected to leading order; the regularisation only keeps the Hessian finite
    at exactly zero curvature, which is where every solve starts. Keep it far
    below ``M_max / EI_0``.
    """

    curvature_regularization: float = 1e-9

    nodes_per_element: int = 2
    rotational_nodes: tuple[int, ...] = (0, 1)
    param_names: tuple[str, ...] = PARAM_NAMES

    def energy(self, q, frames, p):
        x_1 = element_translations(q, 0)
        x_2 = element_translations(q, 1)
        rotation_1 = cayley(element_rotation(q, 2, 0), ca) @ element_frame(frames, 0, ca)
        rotation_2 = cayley(element_rotation(q, 2, 1), ca) @ element_frame(frames, 1, ca)

        rest_length = p[0]
        gamma, omega = beam_strains(x_1, x_2, rotation_1, rotation_2, rest_length, ca)
        d_gamma = gamma - ca.vertcat(p[8], p[9], p[10])
        d_omega = omega - ca.vertcat(p[11], p[12], p[13])

        stretch = ca.vertcat(p[1], p[2], p[3])
        axial = 0.5 * ca.dot(stretch * d_gamma, d_gamma)

        bending_stiffness, moment_max = p[4], p[5]
        curvature = ca.sqrt(
            d_omega[1] ** 2 + d_omega[2] ** 2 + self.curvature_regularization ** 2
        )
        bending = _bending_energy(curvature, bending_stiffness, moment_max)
        # Remove the offset the regularisation leaves at zero curvature, so the
        # reference configuration reports exactly zero energy.
        bending = bending - _bending_energy(
            self.curvature_regularization, bending_stiffness, moment_max
        )

        c1, c2 = p[6], p[7]
        twist = d_omega[0]
        torsion = c1 * (
            twist * ca.atan(c2 * twist) - ca.log(1.0 + (c2 * twist) ** 2) / (2.0 * c2)
        )
        return rest_length * (axial + bending + torsion)


def build_inflatable_beam_elements(
    nodes: Array,
    connectivity: Array,
    laws,
    frames: Array,
    *,
    axial_stiffness: float | Array,
    shear_stiffness: float | Array,
    name: str = "tubes",
    curvature_regularization: float = 1e-9,
) -> ElementSet:
    """Assemble inflatable tube elements, stress-free in the built configuration.

    ``laws`` is one :class:`InflatableTubeLaw` or one per element.
    ``axial_stiffness`` (``EA``) and ``shear_stiffness`` (``kappa G A``) are not
    covered by the fits and must be supplied from the tube geometry; they accept
    a scalar or one value per element.
    """
    nodes = np.asarray(nodes, dtype=float)
    connectivity = np.atleast_2d(np.asarray(connectivity, dtype=int))
    frames = np.asarray(frames, dtype=float)
    n_elements = len(connectivity)

    if isinstance(laws, InflatableTubeLaw):
        laws = [laws] * n_elements
    if len(laws) != n_elements:
        raise ValueError(f"{n_elements} tube elements but {len(laws)} laws")

    def _column(value) -> Array:
        return np.broadcast_to(np.asarray(value, dtype=float), (n_elements,)).copy()

    axial = _column(axial_stiffness)
    shear = _column(shear_stiffness)

    params = np.empty((n_elements, len(PARAM_NAMES)))
    for element, (node_a, node_b) in enumerate(connectivity):
        rest_length = float(np.linalg.norm(nodes[node_b] - nodes[node_a]))
        gamma_0, omega_0 = beam_strains(
            nodes[node_a], nodes[node_b], frames[node_a], frames[node_b],
            rest_length, np,
        )
        params[element] = np.concatenate(
            [
                [rest_length, axial[element], shear[element], shear[element]],
                laws[element].as_row(),
                gamma_0,
                omega_0,
            ]
        )
    kernel = InflatableBeamKernel(curvature_regularization=curvature_regularization)
    return ElementSet(name, kernel, connectivity, params)


def inflatable_beam_state(model, state, element_set: ElementSet, laws) -> dict[str, Array]:
    """Per-element curvature, moment and collapse flag for a solved configuration.

    Collapse is reported rather than enforced: the fitted law has no meaningful
    post-collapse branch, so what matters is knowing which elements have left
    the range it was calibrated over. ``utilisation`` is curvature over the
    collapse curvature, so anything at or above 1.0 is outside validity.

    Uses the same :func:`~awetrim.structural.elements.beam.beam_strains` the
    kernel does, so the numbers reported are the ones the energy saw.
    """
    positions = np.asarray(state.positions, dtype=float)
    frames = np.asarray(state.frames, dtype=float)
    connectivity = element_set.connectivity
    params = element_set.params
    n_elements = len(connectivity)

    if isinstance(laws, InflatableTubeLaw):
        laws = [laws] * n_elements
    if len(laws) != n_elements:
        raise ValueError(f"{n_elements} tube elements but {len(laws)} laws")

    curvature = np.empty(n_elements)
    twist = np.empty(n_elements)
    for element, (node_a, node_b) in enumerate(connectivity):
        omega = beam_strains(
            positions[node_a],
            positions[node_b],
            frames[model.layout.rotation_slot(node_a)],
            frames[model.layout.rotation_slot(node_b)],
            params[element, 0],
            np,
        )[1]
        delta = omega - params[element, 11:14]
        twist[element] = delta[0]
        curvature[element] = float(np.hypot(delta[1], delta[2]))

    collapse = np.array([law.curvature_collapse for law in laws])
    return {
        "curvature": curvature,
        "twist_rate": twist,
        "moment": np.array(
            [law.moment(k) for law, k in zip(laws, curvature)], dtype=float
        ),
        "torque": np.array(
            [law.torque(t) for law, t in zip(laws, twist)], dtype=float
        ),
        "utilisation": curvature / collapse,
        "collapsed": curvature >= collapse,
    }
