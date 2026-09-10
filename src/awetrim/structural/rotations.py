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

"""Finite-rotation kernels -- the single place these formulas are written.

Rotations are parametrised by the **Rodrigues (Cayley) vector**

    psi = 2 tan(phi/2) n          (phi = angle, n = unit axis)

rather than by the exponential map's rotation vector ``phi n``. The two agree
to O(phi^3); the Cayley form is preferred here because every map below is then
a *rational* function of its argument -- no ``sin``, ``cos``, ``atan`` and, in
particular, no ``sin(phi)/phi`` removable singularity at ``phi = 0`` that would
need an ``if_else`` branch. That matters because these kernels are
differentiated twice by CasADi inside an IPOPT objective: a branch-free
rational map gives exact, continuous Hessians everywhere in ``|phi| < pi``.

Following ``environment/profile_laws.py``, the functions take a math namespace
``xp`` so the same code serves NumPy (building reference frames, absorbing
solved increments) and CasADi (inside element energy kernels).

Conventions
-----------
* vectors are NumPy 1-D ``(3,)`` or CasADi ``3x1``; matrices are ``(3, 3)``.
* ``cayley(psi)`` returns ``R`` such that ``R x`` rotates ``x`` by ``psi``.
* ``cayley_vector(R)`` is its inverse, singular only at ``phi = pi``.
* frames are stored **row-major flattened** (``R[0,0], R[0,1], ... R[2,2]``)
  when they travel as a parameter vector.

Reference: Ibrahimbegovic (1997) *Comput. Methods Appl. Mech. Engrg.* 149,
49-71, on Cayley/Rodrigues parametrisations of geometrically exact beams.
"""

from typing import Any

import numpy as np


def _blocks(rows, xp) -> Any:
    """Assemble a 3x3 matrix from nested rows for either namespace."""
    if xp is np:
        return np.array(rows, dtype=float)
    return xp.vertcat(*[xp.horzcat(*row) for row in rows])


def _vec3(a, b, c, xp) -> Any:
    """Assemble a 3-vector from scalar components for either namespace."""
    if xp is np:
        return np.array([a, b, c], dtype=float)
    return xp.vertcat(a, b, c)


def eye3(xp) -> Any:
    """3x3 identity in the ``xp`` namespace."""
    return _blocks([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], xp)


def skew(v, xp) -> Any:
    """Skew-symmetric matrix ``S`` with ``S w = v x w``."""
    return _blocks(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ],
        xp,
    )


def axial(matrix, xp) -> Any:
    """Axial vector of the skew part of ``matrix``, times two.

    For skew ``A = skew(w)`` this returns ``2 w``; the factor is folded into
    :func:`cayley_vector` so no division is wasted here.
    """
    return _vec3(
        matrix[2, 1] - matrix[1, 2],
        matrix[0, 2] - matrix[2, 0],
        matrix[1, 0] - matrix[0, 1],
        xp,
    )


def cayley(psi, xp) -> Any:
    """Rotation matrix for the Rodrigues vector ``psi = 2 tan(phi/2) n``.

    ``R = I + 4/(4 + psi.psi) (S + S S / 2)`` with ``S = skew(psi)``. Rational
    in ``psi``, exact for every ``|phi| < pi``, and equal to ``I`` at
    ``psi = 0`` with no special case.
    """
    s = skew(psi, xp)
    scale = 4.0 / (4.0 + xp.dot(psi, psi))
    return eye3(xp) + scale * (s + 0.5 * (s @ s))


def cayley_vector(rotation, xp) -> Any:
    """Rodrigues vector of a rotation matrix -- the inverse of :func:`cayley`.

    ``psi = 2 axial(R - R^T) / (1 + tr R)``. Singular only where ``tr R = -1``
    (``phi = pi``), which incremental rotations never reach.
    """
    return 2.0 * axial(rotation, xp) / (1.0 + xp.trace(rotation))


def half_vector(psi, xp) -> Any:
    """Rodrigues vector of the *half* rotation: ``cayley(h)^2 == cayley(psi)``.

    With ``t = tan(phi/2) = |psi|/2`` the half-angle identity
    ``tan(phi/4) = (sqrt(1 + t^2) - 1)/t`` has a removable ``0/0`` at ``t = 0``.
    Rationalising it gives ``psi / (1 + sqrt(1 + psi.psi/4))``, which is exact
    and analytic at ``psi = 0`` (where it correctly returns ``psi/2``).
    """
    return psi / (1.0 + xp.sqrt(1.0 + 0.25 * xp.dot(psi, psi)))


def rotate_frame(psi, frame, xp) -> Any:
    """Apply the incremental rotation ``psi`` to a frame: ``cayley(psi) @ R``."""
    return cayley(psi, xp) @ frame


def orthonormalize(rotation: np.ndarray) -> np.ndarray:
    """Nearest orthonormal matrix (polar projection). NumPy only.

    Used when solved increments are absorbed into the stored reference frames,
    so repeated absorption cannot let them drift off SO(3).
    """
    u, _, vt = np.linalg.svd(np.asarray(rotation, dtype=float))
    projected = u @ vt
    if np.linalg.det(projected) < 0.0:
        u[:, -1] *= -1.0
        projected = u @ vt
    return projected


def frames_to_flat(frames: np.ndarray) -> np.ndarray:
    """``(n, 3, 3)`` frames -> row-major flat parameter vector of length ``9n``."""
    return np.asarray(frames, dtype=float).reshape(-1)


def flat_to_frames(flat: np.ndarray) -> np.ndarray:
    """Inverse of :func:`frames_to_flat`."""
    return np.asarray(flat, dtype=float).reshape(-1, 3, 3)


def unflatten_frame(flat, xp) -> Any:
    """Rebuild a 3x3 matrix from nine row-major entries in the ``xp`` namespace.

    ``xp.reshape`` fills column-major in CasADi, so the reshaped block is the
    transpose of what we want; taking ``.T`` restores the row-major order used
    by :func:`frames_to_flat`.
    """
    if xp is np:
        return np.asarray(flat, dtype=float).reshape(3, 3)
    return xp.reshape(flat, 3, 3).T


def rotation_vector(psi, xp, regularization: float = 1e-24) -> Any:
    """Exponential-map rotation vector ``phi n`` from the Rodrigues vector ``psi``.

    ``psi = 2 tan(phi/2) n`` and ``phi n = psi * 2 atan(|psi|/2) / |psi|``. The
    scale factor tends to 1 as ``|psi| -> 0``; ``regularization`` keeps the norm
    away from an exact zero so the expression stays differentiable at the origin
    every solve starts from.

    This is needed only for the *work* done by an applied moment. A moment is
    work-conjugate to the rotation vector, not to the Rodrigues vector, so
    ``-M . psi`` would be correct only to first order -- see
    :mod:`awetrim.structural.energy`.
    """
    norm = xp.sqrt(xp.dot(psi, psi) + regularization)
    return psi * (2.0 * xp.arctan(0.5 * norm) / norm) if xp is np else psi * (
        2.0 * xp.atan(0.5 * norm) / norm
    )
