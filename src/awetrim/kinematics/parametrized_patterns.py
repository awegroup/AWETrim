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

import casadi as ca
from abc import ABC
import numpy as np


class ParametrizedPatterns(ABC):
    def __init__(self, **kwargs):
        self.optimization_vars = {}  # Dictionary to store symbolic MX variables
        for key, value in kwargs.items():
            setattr(self, key, value)
            if isinstance(value, ca.MX):  # If value is symbolic, store it separately
                self.optimization_vars[key] = value

    def x(self, r, s):
        return r * ca.cos(self.azimuth(r, s)) * ca.cos(self.elevation(r, s))

    def y(self, r, s):
        return r * ca.sin(self.azimuth(r, s)) * ca.cos(self.elevation(r, s))

    def z(self, r, s):
        return r * ca.sin(self.elevation(r, s))


def create_pattern_from_dict(
    pattern_type,
    parameters,
) -> ParametrizedPatterns:

    # The class map is the single source of truth for what can actually be
    # instantiated. Required-parameter lists only exist for types we can build.
    pattern_classes = {
        "reel_in_simple": Reelin_Simple,
        "transition_simple": Transition_Simple,
        "spline_periodic": PeriodicBSpline,
        "spline_open": OpenBSpline,
    }

    required_params = {
        "reel_in_simple": ["elevation_start_ri", "elevation_start_riro"],
        "transition_simple": [
            "elevation_start_riro",
            "elevation_start_ro",
        ],
        "spline_periodic": ["M", "C_phi", "C_beta", "s_init", "s_final"],
        "spline_open": ["M", "C_phi", "C_beta", "s_init", "s_final", "r0"],
    }

    if pattern_type not in pattern_classes:
        raise ValueError(
            f"Unknown or unsupported pattern type: {pattern_type!r}. "
            f"Supported types: {sorted(pattern_classes)}"
        )

    missing_params = [
        param
        for param in required_params.get(pattern_type, [])
        if param not in parameters
    ]
    if missing_params:
        raise ValueError(
            f"Missing required parameters in 'parameters' for '{pattern_type}': {', '.join(missing_params)}"
        )

    return pattern_classes[pattern_type](**parameters)


class Reelin_Simple(ParametrizedPatterns):
    def __init__(
        self,
        elevation_start_ri,
        elevation_start_riro,
    ):  # <- only flags
        super().__init__(
            elevation_start_ri=elevation_start_ri,
            elevation_start_riro=elevation_start_riro,
        )

    def elevation(self, r, s):
        return self.elevation_start_ri + s * (
            self.elevation_start_riro - self.elevation_start_ri
        )

    def azimuth(self, r, s):
        return 0


class Transition_Simple(ParametrizedPatterns):
    def __init__(
        self,
        elevation_start_riro,
        elevation_start_ro,
    ):  # <- only flags
        super().__init__(
            elevation_start_riro=elevation_start_riro,
            elevation_start_ro=elevation_start_ro,
        )

    def elevation(self, r, s):
        return self.elevation_start_riro + s * (
            self.elevation_start_ro - self.elevation_start_riro
        )

    def azimuth(self, r, s):
        # Simplified: the transition is flown straight downwind (azimuth = 0),
        # matching Reelin_Simple. Kept fixed everywhere for now.
        return 0


def _tp3(x):
    """truncated power (x_+)^3 using fmax; works for MX/SX."""
    return ca.fmax(x, 0.0) ** 3


def cubic_cardinal_B3(t):
    """
    Cardinal cubic B-spline basis kernel with support [-2,2].
    Using truncated power representation:
    B3(t) = ( (t+2)_+^3 -4(t+1)_+^3 +6(t)_+^3 -4(t-1)_+^3 + (t-2)_+^3 ) / 6
    """
    return (
        _tp3(t + 2) - 4 * _tp3(t + 1) + 6 * _tp3(t) - 4 * _tp3(t - 1) + _tp3(t - 2)
    ) / 6.0


def cubic_B3_np(t):
    tp = lambda x: np.maximum(x, 0.0) ** 3
    return (tp(t + 2) - 4 * tp(t + 1) + 6 * tp(t) - 4 * tp(t - 1) + tp(t - 2)) / 6.0


def cubic_B3_np_d1(t):
    """First derivative of :func:`cubic_B3_np` (truncated-power form)."""
    tp = lambda x: np.maximum(x, 0.0) ** 2
    return (tp(t + 2) - 4 * tp(t + 1) + 6 * tp(t) - 4 * tp(t - 1) + tp(t - 2)) / 2.0


def cubic_B3_np_d2(t):
    """Second derivative of :func:`cubic_B3_np` (truncated-power form)."""
    tp = lambda x: np.maximum(x, 0.0)
    return tp(t + 2) - 4 * tp(t + 1) + 6 * tp(t) - 4 * tp(t - 1) + tp(t - 2)


def open_uniform_knots(M, p=3):
    """Open-uniform (clamped) knot vector on [0,1] for M control points, degree p."""
    if M < p + 1:
        raise ValueError(f"Need M >= p+1. Got M={M}, p={p}.")
    n_knots = M + p + 1
    n_int = n_knots - 2 * (p + 1)  # number of interior knots
    if n_int > 0:
        interior = np.linspace(0.0, 1.0, n_int + 2)[1:-1]
        T = np.r_[np.zeros(p + 1), interior, np.ones(p + 1)]
    else:
        T = np.r_[np.zeros(p + 1), np.ones(p + 1)]
    return T


def build_open_cubic_bspline_function(M, dim=1, name="open_bspline", p=3):
    """
    Open (non-periodic) clamped cubic B-spline S = spline(C,u)
    - C: (M, dim)
    - u in [0,1]
    Returns: S (1,dim)
    """
    T_np = open_uniform_knots(M, p=p)
    T = ca.DM(T_np)  # constants inside CasADi graph

    C = ca.MX.sym("C", M, dim)
    u = ca.MX.sym("u")

    # clamp u to [0,1] and handle u==1 safely for half-open intervals
    u0 = ca.fmin(ca.fmax(u, 0.0), 1.0)
    u_eval = ca.if_else(u0 == 1.0, ca.DM(1.0 - 1e-12), u0)

    # degree-0 basis N_i,0(u)
    N = [None] * M
    for i in range(M):
        left = T[i]
        right = T[i + 1]
        N[i] = ca.if_else(ca.logic_and(u_eval >= left, u_eval < right), 1.0, 0.0)

    # Cox–de Boor recursion up to degree p
    for k in range(1, p + 1):
        Nk = [0] * M
        for i in range(M):
            # left term
            den1 = T[i + k] - T[i]
            term1 = ca.if_else(den1 != 0, (u_eval - T[i]) / den1 * N[i], 0.0)

            # right term uses N[i+1]
            if i + 1 < M:
                den2 = T[i + k + 1] - T[i + 1]
                term2 = ca.if_else(
                    den2 != 0, (T[i + k + 1] - u_eval) / den2 * N[i + 1], 0.0
                )
            else:
                term2 = 0.0

            Nk[i] = term1 + term2
        N = Nk

    # Evaluate spline
    S = ca.MX.zeros(1, dim)
    for i in range(M):
        S += N[i] * C[i, :].T

    # enforce exact endpoint at u==1: S(1)=last control point (clamped convention)
    S = ca.if_else(u0 == 1.0, C[M - 1, :].T, S)

    return ca.Function(name, [C, u], [S], ["C", "u"], ["S"])


def bspline_open_basis_matrix(u_grid, M, p=3):
    u = np.asarray(u_grid).ravel()
    Np = u.size
    T = open_uniform_knots(M, p=p)

    u0 = np.clip(u, 0.0, 1.0)
    u_eval = np.where(u0 == 1.0, np.nextafter(1.0, 0.0), u0)

    # degree-0
    B = np.zeros((Np, M))
    for i in range(M):
        B[:, i] = ((T[i] <= u_eval) & (u_eval < T[i + 1])).astype(float)

    # recursion
    for k in range(1, p + 1):
        Bk = np.zeros_like(B)
        for i in range(M):
            den1 = T[i + k] - T[i]
            if den1 != 0:
                Bk[:, i] += (u_eval - T[i]) / den1 * B[:, i]
            if i + 1 < M:
                den2 = T[i + k + 1] - T[i + 1]
                if den2 != 0:
                    Bk[:, i] += (T[i + k + 1] - u_eval) / den2 * B[:, i + 1]
        B = Bk

    # exact endpoint
    at_one = u0 == 1.0
    if np.any(at_one):
        B[at_one, :] = 0.0
        B[at_one, -1] = 1.0

    return B


def periodic_bspline_basis_matrix(u_grid, M):
    """Basis matrix B[n,j] for periodic cubic B-splines evaluated on u_grid.

    ``u`` is wrapped into ``[0, 1)`` first — the periodic basis is exactly
    1-periodic, so this is loss-free and mirrors ``PeriodicBSpline._u``. Without
    it, reversed (uploop, ``u in [-1, 0]``) or multi-figure grids fall outside
    the ``i in [-2, M+1]`` support loop below and produce all-zero (singular)
    rows.
    """
    u_grid = np.asarray(u_grid).ravel()
    u_grid = u_grid - np.floor(u_grid)
    N = u_grid.size
    x = u_grid * M

    B = np.zeros((N, M))
    for i in range(-2, M + 2):
        idx = i % M
        t = x - i
        B[:, idx] += cubic_B3_np(t)

    return B


def periodic_bspline_basis_matrices(u_grid, M):
    """Value, first- and second-derivative basis matrices ``(B0, B1, B2)``.

    Same assembly as :func:`periodic_bspline_basis_matrix` with the analytic
    kernel derivatives; derivatives are with respect to ``u`` (chain factor
    ``M`` per order, since the kernel argument is ``u * M - i``). So
    ``B1 @ C`` and ``B2 @ C`` are exact ``d/du`` and ``d^2/du^2`` of the
    spline -- no finite differences.
    """
    u_grid = np.asarray(u_grid).ravel()
    u_grid = u_grid - np.floor(u_grid)
    x = u_grid * M

    mats = [np.zeros((u_grid.size, M)) for _ in range(3)]
    kernels = (cubic_B3_np, cubic_B3_np_d1, cubic_B3_np_d2)
    for i in range(-2, M + 2):
        idx = i % M
        t = x - i
        for order, (mat, kernel) in enumerate(zip(mats, kernels)):
            mat[:, idx] += float(M) ** order * kernel(t)

    return tuple(mats)


def build_periodic_cubic_bspline_function(
    M, dim=1, name="per_bspline", support_interval=None
):
    """
    Build a CasADi function S = spline(C, u) for a uniform periodic cubic B-spline.

    - M: number of control points (periodic)
    - dim: output dimension (1 for scalar, 2 for [phi,beta] etc.)
    - C: (M, dim)
    - u: scalar in [0,1] (you map s -> u outside)
    - support_interval: None for the full periodic sum (valid for any u); an
      integer k in [0, M) to keep ONLY the four basis terms that can be
      non-zero on knot interval ``u*M in [k, k+1]`` (coefficients
      ``C[(k-1..k+2) mod M]``). The truncated function is exact on that
      closed interval (the dropped terms are identically zero there, with
      zero derivatives) and WRONG elsewhere -- its point is structural
      sparsity: a symbolic-``u`` graph then depends on 4 coefficients
      instead of all M, so per-node NLP functions built from it keep the
      sparse Jacobian that whole-graph SX expansion gets from constant
      folding (see ``PhaseParameterized.opti_phase``).

    Returns:
      spline_fun(C, u) -> S (1, dim)
    """
    C = ca.MX.sym("C", M, dim)
    u = ca.MX.sym("u")  # assumed in [0,1]

    x = u * M  # in [0, M]

    S = ca.MX.zeros(1, dim)

    if support_interval is None:
        # Sum from i=-2..M+1; wrap coefficient index with python int modulo
        terms = range(-2, M + 2)
    else:
        k = int(support_interval)
        if not 0 <= k < M:
            raise ValueError(f"support_interval must be in [0, {M}), got {k}")
        # B3(x - i) != 0 only for |x - i| < 2: on x in [k, k+1] that is
        # i in {k-1, k, k+1, k+2}.
        terms = range(k - 1, k + 3)
    for i in terms:
        idx = i % M  # integer, safe for MX indexing
        t = x - i
        w = cubic_cardinal_B3(t)  # scalar
        S += w * C[idx, :].T  # (1,dim) += scalar*(1,dim)

    return ca.Function(name, [C, u], [S], ["C", "u"], ["S"])


class PeriodicBSpline(ParametrizedPatterns):

    def __init__(
        self,
        M,
        C_phi,
        C_beta,
        s_init,
        s_final,
        r0=None,
        downloops=True,
        support_interval=None,
    ):
        super().__init__(
            M=M, C_phi=C_phi, C_beta=C_beta, s_init=s_init, s_final=s_final, r0=r0
        )
        self.M = int(M)
        self.s_init = float(s_init)
        self.s_final = float(s_final)
        self.omega = 1.0 if downloops else -1.0
        # None: full periodic spline. int k: truncated to the four basis terms
        # of knot interval k -- exact only for s inside that interval, see
        # ``build_periodic_cubic_bspline_function`` / ``local_support``.
        self.support_interval = (
            None if support_interval is None else int(support_interval)
        )

        self.spline = build_periodic_cubic_bspline_function(
            self.M,
            dim=1,
            name=(
                f"periodic_bspline_{self.M}"
                if self.support_interval is None
                else f"periodic_bspline_{self.M}_k{self.support_interval}"
            ),
            support_interval=self.support_interval,
        )

        self.C_phi = C_phi
        self.C_beta = C_beta

    def local_support(self, k):
        """Copy of this spline restricted (structurally) to knot interval ``k``.

        Same coefficients and parametrisation; only the four basis terms that
        are non-zero on ``u*M in [k, k+1]`` are kept, so a symbolic-``s``
        expression built from it depends on 4 control points per coordinate
        instead of M. Exact on that interval only -- evaluate it at ``s``
        with ``knot_interval(s) == k``.
        """
        return PeriodicBSpline(
            M=self.M,
            C_phi=self.C_phi,
            C_beta=self.C_beta,
            s_init=self.s_init,
            s_final=self.s_final,
            r0=getattr(self, "r0", None),
            downloops=self.omega > 0,
            support_interval=k,
        )

    def knot_interval(self, s):
        """Knot interval index k in [0, M) of numeric ``s`` (scalar or array).

        Uses the same ``s -> u`` mapping (direction + wrap into [0, 1)) as
        the evaluation, so the truncated spline ``local_support(k)`` is exact
        at ``s`` (including ``s`` exactly on a knot, where both neighbouring
        truncations agree).
        """
        s_arr = np.asarray(s, dtype=float)
        u = self.omega * (s_arr - self.s_init) / (self.s_final - self.s_init)
        u = u - np.floor(u)
        k = np.clip(np.floor(u * self.M).astype(int), 0, self.M - 1)
        return int(k) if np.ndim(s_arr) == 0 else k

    def _u(self, s):
        u = self.omega * (s - self.s_init) / (self.s_final - self.s_init)
        # Wrap into a single period [0, 1) so s spanning multiple periods repeats
        # the (periodic) figure — this is what lets the reel-out fly more than one
        # figure. Identity for u in [0, 1), so single-figure runs are unchanged.
        if isinstance(u, np.ndarray):
            return u - np.floor(u)
        return u - ca.floor(u)

    def _eval_spline_vec(self, C, u):
        if np.isscalar(u) or (hasattr(u, "is_scalar") and u.is_scalar()):
            return self.spline(C, u)[0]

        if not hasattr(u, "numel"):
            u = ca.DM(np.asarray(u).ravel())

        u_col = ca.reshape(u, u.numel(), 1)
        N = int(u_col.numel())
        spl_map = self.spline.map(N)
        S = spl_map(C, u_col)
        return S.T

    def azimuth(self, r, s):
        u = self._u(s)
        return self._eval_spline_vec(self.C_phi, u)

    def elevation(self, r, s):
        u = self._u(s)
        return self._eval_spline_vec(self.C_beta, u)


def reelin_control_point_mask(
    path_parameters,
    elevation_fraction=0.75,
    support=2.0,
    n_samples=720,
):
    """Boolean mask over the M control points of a periodic full-cycle spline
    flagging those that shape the reel-in arc.

    The arc is identified as the samples in the top ``elevation_fraction`` of
    the pattern's elevation range (the "parked high" part of a single-phase
    pumping cycle). A control point is flagged when its cubic B-spline support
    (|x - i| < ``support`` in knot coordinates x = u*M, wrapped) overlaps any
    such sample. Used to build per-point trust-region step bounds: the
    figure-eight control points keep the tight topology-protecting box while
    the reel-in bow -- a single arc with no loops to destroy -- gets a wider
    one.
    """
    M = int(path_parameters["M"])
    pattern = PeriodicBSpline(
        M=M,
        C_phi=np.asarray(path_parameters["C_phi"], dtype=float).reshape((M, 1)),
        C_beta=np.asarray(path_parameters["C_beta"], dtype=float).reshape((M, 1)),
        s_init=float(path_parameters.get("s_init", 0.0)),
        s_final=float(path_parameters.get("s_final", 1.0)),
        downloops=bool(path_parameters.get("downloops", True)),
    )
    s_span = pattern.s_final - pattern.s_init
    s = pattern.s_init + s_span * np.linspace(0.0, 1.0, n_samples, endpoint=False)
    elevation = np.asarray(
        [float(pattern.elevation(1.0, si)) for si in s], dtype=float
    )
    threshold = elevation.min() + float(elevation_fraction) * (
        elevation.max() - elevation.min()
    )
    x = pattern._u(s) * M  # knot coordinate of each sample, wrapped to [0, M)
    x_marked = x[elevation >= threshold]
    mask = np.zeros(M, dtype=bool)
    for i in range(M):
        wrapped = np.abs((x_marked - i + M / 2.0) % M - M / 2.0)
        mask[i] = bool(np.any(wrapped < float(support)))
    return mask


class OpenBSpline(ParametrizedPatterns):
    def __init__(self, M, C_phi, C_beta, s_init, s_final, downloops=True, r0=None):
        super().__init__(
            M=M, C_phi=C_phi, C_beta=C_beta, s_init=s_init, s_final=s_final, r0=r0
        )
        self.M = int(M)
        self.s_init = float(s_init)
        self.s_final = float(s_final)
        self.omega = 1.0 if downloops else -1.0
        self.r0 = r0  # optional reference radius for compatibility

        self.spline = build_open_cubic_bspline_function(
            self.M, dim=1, name=f"open_bspline_{self.M}"
        )

        self.C_phi = C_phi
        self.C_beta = C_beta

    def _u(self, s):
        return self.omega * (s - self.s_init) / (self.s_final - self.s_init)

    def _eval_spline_vec(self, C, u):
        if np.isscalar(u) or (hasattr(u, "is_scalar") and u.is_scalar()):
            return self.spline(C, u)[0]

        if not hasattr(u, "numel"):
            u = ca.DM(np.asarray(u).ravel())

        u_col = ca.reshape(u, u.numel(), 1)
        N = int(u_col.numel())
        spl_map = self.spline.map(N)
        S = spl_map(C, u_col)
        return S.T

    def azimuth(self, r, s):
        u = self._u(s)
        return self._eval_spline_vec(self.C_phi, u)

    def elevation(self, r, s):
        u = self._u(s)
        return self._eval_spline_vec(self.C_beta, u)


def fit_bspline_pattern_to_trajectory(
    spline_type,
    M,
    s_init,
    s_final,
    az_target,
    el_target,
    s_samples,
    downloops=True,
):
    """
    Fit a periodic or open cubic B-spline to target azimuth/elevation samples.
    Returns the fitted pattern instance and control points.
    """
    s_samples = np.asarray(s_samples).ravel()
    az_target = np.asarray(az_target).ravel()
    el_target = np.asarray(el_target).ravel()

    if not (s_samples.size == az_target.size == el_target.size):
        raise ValueError("s_samples, az_target, and el_target must have same length.")

    omega = 1.0 if downloops else -1.0
    u_grid = omega * (s_samples - s_init) / (s_final - s_init)

    if spline_type == "periodic":
        B = periodic_bspline_basis_matrix(u_grid, M)
    elif spline_type == "open":
        B = bspline_open_basis_matrix(u_grid, M, p=3)
    else:
        raise ValueError("spline_type must be 'periodic' or 'open'.")

    c_phi, *_ = np.linalg.lstsq(B, az_target, rcond=None)
    c_beta, *_ = np.linalg.lstsq(B, el_target, rcond=None)

    C_phi = ca.DM(c_phi).reshape((M, 1))
    C_beta = ca.DM(c_beta).reshape((M, 1))

    if spline_type == "periodic":
        pattern = PeriodicBSpline(
            M, C_phi, C_beta, s_init, s_final, downloops=downloops
        )
    else:
        pattern = OpenBSpline(M, C_phi, C_beta, s_init, s_final, downloops=downloops)

    return pattern, C_phi, C_beta


def _spherical_path_cross_speed(phi, beta, dphi, dbeta, ddphi, ddbeta, xp):
    """``(|q' x q''|^2, |q'|^2, |q''|^2)`` of the unit-sphere path.

    ``q = [cos(phi) cos(beta), sin(phi) cos(beta), sin(beta)]`` with all
    derivatives taken in the path parameter (chain rule through the given
    angle derivatives). Unit-sphere curvature is
    ``sqrt(cross2) / speed2**1.5``; divide by the radius for 1/m; ``accel2``
    is the smoothing/bending energy density used by the reel-in design. Pure
    elementwise math over the ``xp`` namespace (numpy for sampling, casadi
    inside the fairing/design NLPs) so the formula exists exactly once.
    """
    sphi, cphi = xp.sin(phi), xp.cos(phi)
    sbeta, cbeta = xp.sin(beta), xp.cos(beta)
    qux = -sphi * cbeta * dphi - cphi * sbeta * dbeta
    quy = cphi * cbeta * dphi - sphi * sbeta * dbeta
    quz = cbeta * dbeta
    quux = (
        -sphi * cbeta * ddphi
        - cphi * sbeta * ddbeta
        - cphi * cbeta * dphi**2
        + 2.0 * sphi * sbeta * dphi * dbeta
        - cphi * cbeta * dbeta**2
    )
    quuy = (
        cphi * cbeta * ddphi
        - sphi * sbeta * ddbeta
        - sphi * cbeta * dphi**2
        - 2.0 * cphi * sbeta * dphi * dbeta
        - sphi * cbeta * dbeta**2
    )
    quuz = cbeta * ddbeta - sbeta * dbeta**2
    cx = quy * quuz - quz * quuy
    cy = quz * quux - qux * quuz
    cz = qux * quuy - quy * quux
    cross2 = cx**2 + cy**2 + cz**2
    speed2 = qux**2 + quy**2 + quz**2
    accel2 = quux**2 + quuy**2 + quuz**2
    return cross2, speed2, accel2


def fair_periodic_spline_to_curvature_limit(
    path_parameters,
    curvature_limit,
    n_samples=1000,
    margin=0.97,
    support_pad=1,
    max_rounds=5,
    max_move=0.2,
    precision=6,
):
    """Locally move periodic-spline control points until the curvature fits.

    The parametric description that generated the spline is NOT touched:
    wherever the sampled physical curvature exceeds ``curvature_limit``
    (1/m), the control points whose basis support covers the violation (plus
    ``support_pad`` neighbours per side) are freed and moved by the SMALLEST
    amount (min sum of squared moves, IPOPT) that brings the analytic
    curvature under ``margin * curvature_limit`` at every sample the freed
    points influence. All other control points -- and therefore the path
    everywhere else -- are bit-identical. If a round cannot reach the limit
    the freed neighbourhood is widened and it retries (``max_rounds``).

    Two guards keep the solver honest:

    * ``max_move`` (rad) is a TRUST REGION: every freed control point stays
      within this box around its ORIGINAL value (cumulative across rounds).
      Without it, a badly violating seed (e.g. a reel-in peak whose top
      U-turn is 10x over the limit) lets IPOPT wander from the infeasible
      start into a curvature-feasible but wildly deformed basin -- giant
      sweeping loops that pass the limit while destroying the path. The box
      makes that basin infeasible, forcing the small local rounding; it also
      bounds the path deviation (basis partition of unity: the path moves
      less than ``max_move`` anywhere). If no fix exists within the box the
      fairing reports ``converged=False`` instead of inventing a new path.
    * a parametrization-speed floor at every constrained sample (a quarter
      of its pre-round minimum): without it the solver can satisfy the
      curvature bound by stalling the parametrization between samples (a
      forming cusp -- pointwise curvature looks fine, the flown path is
      not). For the floor to bite, everything (detection, constraints and
      the verdict) is sampled at ``4 * n_samples`` -- dense relative to the
      knot spacing (tens of samples per knot interval), so ``|q'|`` cannot
      dip to zero between two constrained samples: its between-sample
      change is bounded by ``|q''| * du``, far below the floor at that
      density.

    Returns ``(new_path_parameters, report)``; the input dict is not
    modified. ``report`` keys: ``max_before`` / ``max_after`` (1/m),
    ``s_at_max_before``, ``max_path_move`` (rad, largest great-circle
    displacement of any sample), ``touched`` (sorted control-point indices
    moved; empty when the path already fits), ``changed``, ``converged``,
    ``rounds``. Curvature is unsigned, so the traversal sense
    (``downloops``) is irrelevant and samples are taken directly in the
    periodic ``u in [0, 1)``.
    """
    pp = dict(path_parameters)
    M = int(pp["M"])
    r0 = float(pp["r0"])
    limit = float(curvature_limit)
    C_phi = np.asarray(pp["C_phi"], dtype=float).ravel().copy()
    C_beta = np.asarray(pp["C_beta"], dtype=float).ravel().copy()
    if C_phi.size != M or C_beta.size != M:
        raise ValueError("C_phi/C_beta must have M entries each")

    # ONE grid for detection, constraints and the verdict, dense relative to
    # the knot spacing (~4 * n_samples / M samples per interval): between two
    # adjacent samples ``|q'|`` can change by at most ~``|q''| * du``, far
    # below the speed floor at this density, so no cusp fits in between.
    u = np.linspace(0.0, 1.0, 4 * int(n_samples), endpoint=False)
    B0, B1, B2 = periodic_bspline_basis_matrices(u, M)
    # kappa_unit bound: physical 1/m limit times the radius.
    k_lim_unit = limit * r0
    k_target_unit = float(margin) * k_lim_unit

    def cross_speed(cp, cb):
        return _spherical_path_cross_speed(
            B0 @ cp, B0 @ cb, B1 @ cp, B1 @ cb, B2 @ cp, B2 @ cb, np
        )

    def kappa_unit(cp, cb):
        cross2, speed2, _ = cross_speed(cp, cb)
        return np.sqrt(cross2) / np.maximum(speed2, 1e-30) ** 1.5

    kap0 = kappa_unit(C_phi, C_beta)
    i0 = int(np.argmax(kap0))
    report = {
        "max_before": float(kap0[i0] / r0),
        "s_at_max_before": float(u[i0]),
        "max_after": float(kap0[i0] / r0),
        "touched": [],
        "changed": False,
        "converged": bool(kap0[i0] <= k_lim_unit),
        "rounds": 0,
    }
    report["max_path_move"] = 0.0
    if not limit or report["converged"]:
        return pp, report

    C_phi_orig = C_phi.copy()
    C_beta_orig = C_beta.copy()
    touched = set()
    for round_ in range(1, int(max_rounds) + 1):
        kap = kappa_unit(C_phi, C_beta)
        if kap.max() <= k_lim_unit:
            break
        # Control points supporting any violating sample, padded (the pad
        # grows with the round so an infeasible neighbourhood widens).
        pad = int(support_pad) + (round_ - 1)
        sel = set()
        for k in np.flatnonzero(kap > k_target_unit):
            base = int(np.floor(u[k] * M))
            sel.update(
                (base + j) % M for j in range(-1 - pad, 3 + pad)
            )
        sel = sorted(sel)
        # Constrain every sample the freed points influence (elsewhere the
        # curve, and thus its curvature, cannot change).
        rows = np.flatnonzero(np.abs(B0[:, sel]).sum(axis=1) > 1e-12)
        # Anti-cusp floor: the NLP must not stall the parametrization to
        # satisfy the curvature bound between samples.
        _, speed2_now, _ = cross_speed(C_phi, C_beta)
        smin2 = 0.25 * float(speed2_now[rows].min())

        d_phi = ca.MX.sym("d_phi", len(sel))
        d_beta = ca.MX.sym("d_beta", len(sel))
        # Spline values at the constrained samples split into the constant
        # part (numpy, all M coefficients) plus the delta of the freed ones
        # -- the NLP graph only carries (n_rows x n_sel) matrices, not
        # (n_rows x M), which is what keeps this solve fast.
        angle_terms = []
        for mat in (B0, B1, B2):
            sub = ca.DM(mat[np.ix_(rows, sel)])
            angle_terms.append(ca.DM(mat[rows] @ C_phi) + sub @ d_phi)
            angle_terms.append(ca.DM(mat[rows] @ C_beta) + sub @ d_beta)
        cross2, speed2, _ = _spherical_path_cross_speed(
            angle_terms[0], angle_terms[1], angle_terms[2],
            angle_terms[3], angle_terms[4], angle_terms[5], ca,
        )
        # kappa <= target, squared to stay smooth, and NORMALIZED to O(1)
        # per row: the raw form ``cross2 - k^2 speed2^3`` spans many orders
        # of magnitude across samples (speed2 varies ~100x), which wrecks
        # IPOPT's scaling and costs hundreds of iterations. Division is safe:
        # the floor constraint keeps speed2 away from zero.
        g = ca.vertcat(
            cross2 / (speed2**3 * k_target_unit**2) - 1.0,
            1.0 - speed2 / smin2,
        )
        x = ca.vertcat(d_phi, d_beta)
        solver = ca.nlpsol(
            "fair",
            "ipopt",
            {"x": x, "f": ca.dot(x, x), "g": g},
            {
                "ipopt.print_level": 0,
                "print_time": 0,
                "ipopt.sb": "yes",
                "ipopt.max_iter": 300,
                "ipopt.tol": 1e-6,
                "ipopt.acceptable_tol": 1e-4,
            },
        )
        # Trust region: the CUMULATIVE move of every freed point stays
        # within max_move of its original value.
        lbx = np.concatenate(
            [
                (C_phi_orig[sel] - float(max_move)) - C_phi[sel],
                (C_beta_orig[sel] - float(max_move)) - C_beta[sel],
            ]
        )
        ubx = np.concatenate(
            [
                (C_phi_orig[sel] + float(max_move)) - C_phi[sel],
                (C_beta_orig[sel] + float(max_move)) - C_beta[sel],
            ]
        )
        sol = solver(
            x0=np.zeros(2 * len(sel)),
            lbx=lbx,
            ubx=ubx,
            ubg=np.zeros(2 * rows.size),
        )
        moves = np.asarray(sol["x"]).ravel()
        new_phi, new_beta = C_phi.copy(), C_beta.copy()
        new_phi[sel] += moves[: len(sel)]
        new_beta[sel] += moves[len(sel):]
        report["rounds"] = round_
        # Accept only an actual improvement (a failed solve returns its last
        # iterate); otherwise retry with a wider freed neighbourhood.
        if kappa_unit(new_phi, new_beta).max() < kap.max():
            C_phi, C_beta = new_phi, new_beta
            touched.update(sel)

    C_phi = np.round(C_phi, int(precision))
    C_beta = np.round(C_beta, int(precision))
    kap = kappa_unit(C_phi, C_beta)
    report["max_after"] = float(kap.max() / r0)
    report["touched"] = sorted(touched)
    report["changed"] = bool(touched)
    report["converged"] = bool(kap.max() <= k_lim_unit)

    def unit_q(cp, cb):
        phi, beta = B0 @ cp, B0 @ cb
        return np.column_stack(
            (np.cos(phi) * np.cos(beta), np.sin(phi) * np.cos(beta),
             np.sin(beta))
        )

    dots = np.sum(unit_q(C_phi, C_beta) * unit_q(C_phi_orig, C_beta_orig), axis=1)
    report["max_path_move"] = float(np.arccos(np.clip(dots, -1.0, 1.0)).max())
    pp["C_phi"] = C_phi.tolist()
    pp["C_beta"] = C_beta.tolist()
    return pp, report


def design_reelin_spline(
    path_parameters,
    curvature_limit,
    reelin_center,
    reelout_fraction,
    peak_elevation=None,
    n_samples=1000,
    margin=0.97,
    max_move=0.5,
    w_peak=100.0,
    w_track=0.1,
    w_tension=1.0,
    precision=6,
):
    """DESIGN the reel-in of a periodic full-cycle spline under a curvature cap.

    The parametric reel-in description is treated as a SKETCH only: when the
    fitted spline violates ``curvature_limit`` (1/m), every control point
    whose basis support lies inside the reel-in window becomes a design
    variable of one IPOPT problem that CREATES the reel-in between the two
    handover states --

    * the exit and entry states (position AND heading where the figures
      stop/resume) are exactly preserved: the window-edge control points
      stay frozen, and C2 continuity is automatic because it is the same
      periodic basis;
    * curvature <= ``margin * curvature_limit`` is a HARD constraint at
      every influenced sample (dense grid, ``4 * n_samples``), together
      with the anti-cusp parametrization-speed floor of
      :func:`fair_periodic_spline_to_curvature_limit`;
    * the objective is a spline-in-tension energy: bending ``sum |q''|^2``
      plus ``w_tension`` times the Dirichlet term ``sum |q'|^2`` (both
      normalized by the sketch's values -- the tension keeps the curve taut,
      without it the minimum-bending way around a too-sharp corner is an
      outward teardrop loop), plus a pull of the elevation at the sketch's
      apex sample toward ``peak_elevation`` (the knob is a TARGET to hit,
      never overridden), plus a weak tracking of the sketch that anchors the
      topology (which side, crossing vs tangential) without fighting the
      curvature bound.

    A spline already within the limit is returned unchanged (the sketch is
    the design). Returns ``(new_path_parameters, report)`` with keys
    ``engaged``, ``max_before`` / ``max_after`` (1/m), ``peak_target`` /
    ``peak_achieved`` (rad, NaN when no target), ``freed`` (control-point
    indices), ``changed``, ``converged``, ``max_path_move`` (rad). When the
    single solve cannot reach the limit the best improvement is kept and
    ``converged`` is False -- run the fairing (or reconsider the geometry)
    afterwards.
    """
    pp = dict(path_parameters)
    M = int(pp["M"])
    r0 = float(pp["r0"])
    limit = float(curvature_limit)
    C_phi = np.asarray(pp["C_phi"], dtype=float).ravel().copy()
    C_beta = np.asarray(pp["C_beta"], dtype=float).ravel().copy()

    u = np.linspace(0.0, 1.0, 4 * int(n_samples), endpoint=False)
    B0, B1, B2 = periodic_bspline_basis_matrices(u, M)
    k_lim_unit = limit * r0
    k_target_unit = float(margin) * k_lim_unit

    def cross_speed(cp, cb):
        return _spherical_path_cross_speed(
            B0 @ cp, B0 @ cb, B1 @ cp, B1 @ cb, B2 @ cp, B2 @ cb, np
        )

    def kappa_unit(cp, cb):
        cross2, speed2, _ = cross_speed(cp, cb)
        return np.sqrt(cross2) / np.maximum(speed2, 1e-30) ** 1.5

    kap0 = kappa_unit(C_phi, C_beta)
    i0 = int(np.argmax(kap0))
    report = {
        "engaged": False,
        "max_before": float(kap0[i0] / r0),
        "max_after": float(kap0[i0] / r0),
        "s_at_max_before": float(u[i0]),
        "peak_target": float(peak_elevation) if peak_elevation else float("nan"),
        "peak_achieved": float("nan"),
        "freed": [],
        "changed": False,
        "converged": bool(kap0[i0] <= k_lim_unit),
        "max_path_move": 0.0,
    }
    if not limit or report["converged"]:
        return pp, report

    # Control points whose FULL support [(j-2)/M, (j+2)/M] sits inside the
    # reel-in window: freeing only those leaves the window-edge (handover)
    # states and everything outside bit-identical.
    c = float(reelin_center) % 1.0
    h = 0.5 * (1.0 - float(reelout_fraction))
    sel = [
        j
        for j in range(M)
        if abs((j / M - c + 0.5) % 1.0 - 0.5) <= h - 2.0 / M
    ]
    if not sel:
        return pp, report
    report["engaged"] = True
    rows = np.flatnonzero(np.abs(B0[:, sel]).sum(axis=1) > 1e-12)

    cross2_0, speed2_0, accel2_0 = cross_speed(C_phi, C_beta)
    smin2 = 0.25 * float(speed2_0[rows].min())
    e_bend0 = float(accel2_0[rows].sum())
    # Anchor the peak at the sketch's apex sample (robust for both the
    # tangential and the crossing topology).
    i_peak = int(rows[np.argmax((B0 @ C_beta)[rows])])

    d_phi = ca.MX.sym("d_phi", len(sel))
    d_beta = ca.MX.sym("d_beta", len(sel))
    terms = []
    for mat in (B0, B1, B2):
        sub = ca.DM(mat[np.ix_(rows, sel)])
        terms.append(ca.DM(mat[rows] @ C_phi) + sub @ d_phi)
        terms.append(ca.DM(mat[rows] @ C_beta) + sub @ d_beta)
    cross2, speed2, accel2 = _spherical_path_cross_speed(
        terms[0], terms[1], terms[2], terms[3], terms[4], terms[5], ca
    )

    f = ca.sum1(accel2) / e_bend0 + float(w_tension) * ca.sum1(speed2) / float(
        speed2_0[rows].sum()
    )
    if peak_elevation:
        k_peak = int(np.searchsorted(rows, i_peak))
        f = f + float(w_peak) * (terms[1][k_peak] - float(peak_elevation)) ** 2
    f = f + float(w_track) * (
        ca.sumsqr(terms[0] - ca.DM(B0[rows] @ C_phi))
        + ca.sumsqr(terms[1] - ca.DM(B0[rows] @ C_beta))
    ) / (len(rows) * 0.1**2)
    g = ca.vertcat(
        cross2 / (speed2**3 * k_target_unit**2) - 1.0,
        1.0 - speed2 / smin2,
    )
    solver = ca.nlpsol(
        "design_reelin",
        "ipopt",
        {"x": ca.vertcat(d_phi, d_beta), "f": f, "g": g},
        {
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.sb": "yes",
            "ipopt.max_iter": 300,
            "ipopt.tol": 1e-6,
            "ipopt.acceptable_tol": 1e-4,
        },
    )
    sol = solver(
        x0=np.zeros(2 * len(sel)),
        lbx=-float(max_move),
        ubx=float(max_move),
        ubg=np.zeros(2 * rows.size),
    )
    moves = np.asarray(sol["x"]).ravel()
    new_phi, new_beta = C_phi.copy(), C_beta.copy()
    new_phi[sel] += moves[: len(sel)]
    new_beta[sel] += moves[len(sel):]
    # Keep the design only if it actually improved the worst curvature (a
    # failed solve returns its last iterate).
    if kappa_unit(new_phi, new_beta).max() < kap0.max():
        C_phi_new = np.round(new_phi, int(precision))
        C_beta_new = np.round(new_beta, int(precision))
    else:
        C_phi_new, C_beta_new = C_phi, C_beta

    kap = kappa_unit(C_phi_new, C_beta_new)
    report["max_after"] = float(kap.max() / r0)
    report["changed"] = bool(np.any(C_phi_new != C_phi) or np.any(C_beta_new != C_beta))
    report["converged"] = bool(kap.max() <= k_lim_unit)
    report["freed"] = list(sel)
    report["peak_achieved"] = float((B0 @ C_beta_new)[rows].max())

    def unit_q(cp, cb):
        phi, beta = B0 @ cp, B0 @ cb
        return np.column_stack(
            (np.cos(phi) * np.cos(beta), np.sin(phi) * np.cos(beta),
             np.sin(beta))
        )

    dots = np.sum(unit_q(C_phi_new, C_beta_new) * unit_q(C_phi, C_beta), axis=1)
    report["max_path_move"] = float(np.arccos(np.clip(dots, -1.0, 1.0)).max())
    pp["C_phi"] = C_phi_new.tolist()
    pp["C_beta"] = C_beta_new.tolist()
    return pp, report


def named_curve_angles(
    s,
    curve_type="lissajous",
    az_amp0=0.8,
    beta0=0.45,
    beta_amp0=0.35,
    downloops=True,
):
    """Return azimuth/elevation samples for a named initial curve.

    Supported curves:

    - ``lissajous`` : Gerono/Lissajous figure-eight, ``az = A sin(s)``,
      ``beta = beta0 + B sin(2s)``.
    - ``lemniscate`` : Bernoulli lemniscate figure-eight. Same orientation and
      amplitudes as ``lissajous`` (azimuth in ``[-az_amp0, az_amp0]``,
      elevation in ``[beta0 - beta_amp0, beta0 + beta_amp0]``, starting at the
      centre crossing and moving toward +azimuth) but with a smoother, more
      rounded self-crossing — a gentler initial guess for the optimiser.
    - ``helix`` : ``az = A sin(s)``, ``beta = beta0 + B cos(s)`` (not an eight).

    The returned arrays are numeric samples intended for fitting initial
    B-spline control points, not symbolic trajectory expressions.
    """
    s = np.asarray(s).ravel()
    omega = 1.0 if downloops else -1.0

    if curve_type == "lissajous":
        azimuth = az_amp0 * np.sin(omega * s)
        elevation = beta0 + beta_amp0 * np.sin(omega * 2.0 * s)
    elif curve_type == "lemniscate":
        # Bernoulli lemniscate, reparametrised so the centre crossing is at
        # s = 0. With p = omega * s:
        #   azimuth   = az_amp0 * sin(p) / (1 + cos^2 p)
        #   elevation = beta0 + beta_amp0 * 2*sqrt(2) * sin(p) cos(p) / (1 + cos^2 p)
        # max|sin p / (1 + cos^2 p)| = 1 and max|sin p cos p / (1 + cos^2 p)|
        # = sqrt(2)/4, so the 2*sqrt(2) factor normalises both excursions to
        # the requested amplitudes (matching the lissajous case exactly).
        p = omega * s
        denom = 1.0 + np.cos(p) ** 2
        azimuth = az_amp0 * np.sin(p) / denom
        elevation = (
            beta0 + beta_amp0 * 2.0 * np.sqrt(2.0) * np.sin(p) * np.cos(p) / denom
        )
    elif curve_type == "helix":
        azimuth = az_amp0 * np.sin(omega * s)
        elevation = beta0 + beta_amp0 * np.cos(omega * s)
    else:
        raise ValueError(
            "curve_type must be one of 'lissajous', 'lemniscate' or 'helix'."
        )

    return azimuth, elevation


def fit_bspline_pattern_to_named_curve(
    spline_type,
    M,
    s_init,
    s_final,
    n_fit,
    curve_type="lissajous",
    az_amp0=0.8,
    beta0=0.45,
    beta_amp0=0.35,
    downloops=True,
):
    """Fit a B-spline pattern to a named initial curve.

    ``curve_type`` is one of ``lissajous``/``lemniscate`` (figure-eights) or
    ``helix``; see :func:`named_curve_angles`.
    """
    s_samples = np.linspace(s_init, s_final, int(n_fit), endpoint=True)
    az_target, el_target = named_curve_angles(
        s_samples,
        curve_type=curve_type,
        az_amp0=az_amp0,
        beta0=beta0,
        beta_amp0=beta_amp0,
        downloops=downloops,
    )

    return fit_bspline_pattern_to_trajectory(
        spline_type=spline_type,
        M=M,
        s_init=s_init,
        s_final=s_final,
        az_target=az_target,
        el_target=el_target,
        s_samples=s_samples,
        downloops=downloops,
    )


def make_bspline_path_parameters_from_named_curve(
    spline_type,
    M,
    r0,
    s_init,
    s_final,
    n_fit,
    curve_type="lissajous",
    az_amp0=0.8,
    beta0=0.45,
    beta_amp0=0.35,
    downloops=True,
    precision=6,
):
    """Create YAML-ready path parameters for a B-spline initial curve."""
    _, C_phi, C_beta = fit_bspline_pattern_to_named_curve(
        spline_type=spline_type,
        M=M,
        s_init=s_init,
        s_final=s_final,
        n_fit=n_fit,
        curve_type=curve_type,
        az_amp0=az_amp0,
        beta0=beta0,
        beta_amp0=beta_amp0,
        downloops=downloops,
    )

    def _rounded_coefficients(coefficients):
        values = np.round(coefficients.full().flatten(), precision)
        values[np.isclose(values, 0.0)] = 0.0
        return values.tolist()

    return {
        "r0": float(r0),
        "M": int(M),
        "C_phi": _rounded_coefficients(C_phi),
        "C_beta": _rounded_coefficients(C_beta),
        "s_init": float(s_init),
        "s_final": float(s_final),
        # Carry the traversal sense through to the runtime pattern: the fit maps
        # the curve onto u with this same omega, so PeriodicBSpline/OpenBSpline
        # must rebuild with it (create_pattern_from_dict forwards it) or an
        # uploop fit (downloops=False) is evaluated in the downloop sense.
        "downloops": bool(downloops),
    }


def _smoothstep(edge0, edge1, x):
    """Hermite smoothstep in [0, 1], clamped outside [edge0, edge1]."""
    t = np.clip((np.asarray(x, dtype=float) - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def reelin_bump(s, reelout_fraction=0.7, ramp_fraction=0.25, reelin_center=0.5):
    """Smooth 0->1->0 indicator for the reel-in window, centred at ``reelin_center``.

    Returns ~1 inside the reel-in band ``[c - h, c + h]`` (``c = reelin_center``,
    ``h = (1 - f)/2``) and 0 in the reel-out, with smoothstep edges. The window
    is evaluated on the circular offset from ``c``, so it wraps across the
    periodic seam and ANY centre is valid (taken mod 1): the default
    ``c = 0.5`` keeps ``s = 0`` (and ``s = 1``) in steady, full-amplitude
    reel-out so a forward trim started there is well posed; ``c = h`` starts
    the reel-in exactly at ``s = 0``, ``c = 1 - h`` ends it there, and
    ``c = 0`` puts ``s = 0`` at the top (middle) of the reel-in. Shared by the
    path generator and the synthetic depower profile so both switch at the
    same place.
    """
    s = np.asarray(s, dtype=float).ravel()
    f = float(reelout_fraction)
    c = float(reelin_center)
    h = 0.5 * (1.0 - f)
    r = max(ramp_fraction * (1.0 - f), 1e-6)
    # Circular offset from the window centre, in [-0.5, 0.5): the window wraps
    # across the seam, and h < 0.5 guarantees the bump is 0 (with zero slope)
    # at the antipode, keeping the curve exactly periodic and C1.
    d = (s - c + 0.5) % 1.0 - 0.5
    return _smoothstep(-h, -h + r, d) * (1.0 - _smoothstep(h - r, h, d))


# ``bow_shape="lobe"`` handover: the figure still advances this much phase
# (rad) while the window fades it out / back in -- the freeze happens over the
# first ``2 * LOBE_HANDOVER_PHASE / (rate * ramp)`` of the ramp, so the
# peel-off and landing always span the same small fraction of a figure (~1/12)
# regardless of loop count or ramp width. Smaller = sharper handover bend.
LOBE_HANDOVER_PHASE = 0.5
# Fraction of each elevation ramp the "lobe" bow overlaps (1 - this is how far
# up the climb it stays on the meridian): the bow runs from
# ``ramp * (1 - overlap)`` to ``1 - ramp * (1 - overlap)`` in window units, so
# it carries path speed across the flat plateau edges; 0.5 reproduces the
# mid-ramp start of the "descent" bow, smaller concentrates the crossing (and
# rounds the plateau corners: more azimuth speed where the elevation stalls).
LOBE_BOW_OVERLAP = 0.3


def _pinned_handover_edges(
    psi_entry, psi_exit, bow_shape, lobe_handover_phase=LOBE_HANDOVER_PHASE
):
    """Reel-in window EDGE phases and their fractional figure ``gap``.

    For ``bow_shape="lobe"`` the knobs are the FROZEN phases and the window
    edges sit ``lobe_handover_phase`` outside them (the figure keeps advancing
    that much while it fades out / back in); the other shapes pin the edges
    directly. ``gap`` is the (entry - exit) phase difference as a fraction of
    a figure -- the fractional part every achievable reel-out figure count
    carries.
    """
    if bow_shape == "lobe":
        psi_in_edge = float(psi_entry) - float(lobe_handover_phase)
        psi_out_edge = float(psi_exit) + float(lobe_handover_phase)
    else:
        psi_in_edge, psi_out_edge = float(psi_entry), float(psi_exit)
    gap = ((psi_in_edge - psi_out_edge) / (2.0 * np.pi)) % 1.0
    return psi_in_edge, psi_out_edge, gap


def full_cycle_angles(
    s,
    n_loops=5,
    reelout_fraction=0.7,
    beta0=0.35,
    beta_amp0=0.12,
    az_amp0=0.3,
    beta_reelin_peak=1.1,
    az_reelin_amp=0.25,
    az_reelin_through=0.0,
    reelin_cross_pos=0.7,
    ramp_fraction=0.4,
    reelin_center=0.5,
    psi0=0.0,
    psi_entry=None,
    psi_exit=None,
    bow_shape="sym",
    downloops=True,
    lobe_handover_phase=LOBE_HANDOVER_PHASE,
):
    """Azimuth/elevation samples for a *synthetic full pumping cycle*.

    One period ``s in [0, 1)`` is a whole cycle. The figure-eight phase advances
    *continuously* over the entire period (``psi = 2*pi*n_loops*s``) so the curve
    is exactly periodic and smooth (C1) at the ``s = 0`` boundary, with full path
    speed there. The reel-in is a window centred at ``reelin_center`` (default
    ``s = 0.5``): inside it the
    figure amplitude is faded to zero and the elevation is lifted along a smooth
    arc to ``beta_reelin_peak``. To avoid a cusp at the top of the climb (the kite
    going straight up and back down the same azimuth line), the reel-in also bows
    in azimuth -- to one side on the way up and the other on the way down
    (``az_reelin_amp * sin(2*pi*xi)``) -- tracing a smooth open loop instead of
    retracing. Outside the window the kite flies figure-eights at base elevation
    ``beta0``. The default mid-period centring leaves ``s = 0`` in steady
    reel-out, a far more trim-feasible start than a fit to noisy flight data.

    Unlike :func:`named_curve_angles` (reel-out-only figure-eights), this spans
    the full cycle, so the fitted periodic spline is flown *once* per cycle.

    Parameters
    ----------
    s : array-like
        Path parameter samples in ``[0, 1)``.
    n_loops : float
        Figure-eights over the whole period (roughly ``reelout_fraction * n_loops``
        are visible; the rest fall inside the suppressed reel-in window). With
        BOTH handover phases pinned this is only the snap target and may be
        non-integer -- use :func:`full_cycle_n_loops_for_half_figures` to
        derive it (and the parity-matching ``psi_entry``) from the number of
        half-figures the reel-out should show. With a free phase it must be an
        integer for exact periodicity.
    reelout_fraction : float
        Fraction of the period spent reeling out (the rest is reel-in).
    beta0, beta_amp0, az_amp0 : float
        Base elevation, figure-eight elevation amplitude and azimuth amplitude
        (rad) during reel-out.
    beta_reelin_peak : float
        Peak elevation (rad) reached during reel-in.
    az_reelin_amp : float
        Azimuth excursion (rad) of the reel-in bow that breaks the up/down
        retrace. 0 reproduces the straight-up (cusped) reel-in.
    az_reelin_through : float
        ``bow_shape="lobe"`` only (raises otherwise when nonzero): signed
        azimuth (rad) the "cross" reel-in descends along. After the middle
        climb the kite leans out to this side near the top, comes DOWN
        parked on that azimuth, then crosses ``az = 0`` low (see
        ``reelin_cross_pos``) to reach the ``az_reelin_amp`` side -- the way
        to loop over the top on one side and still land tangentially on the
        other. 0 (default) is the "smooth" shape: the top goes directly to
        the ``az_reelin_amp`` side, nothing crosses. Sign = azimuth side
        (with ``downloops``), magnitude ~ ``|az_reelin_amp|``.
    reelin_cross_pos : float
        "Cross" shape only (``az_reelin_through`` nonzero): where on the
        descent the ``az = 0`` crossing is centred, as a fraction of the
        exit ramp -- 0 = right after the top, 1 = down at figure height
        (default 0.7). The crossing band is +-0.12 of the window around it,
        capped so the tangential landing keeps its final stretch.
    ramp_fraction : float
        Smoothstep edge width of the reel-in window (fraction of its span).
        Larger -> gentler reel-in (a single smooth hump, no flat top); ``>= 0.5``
        removes the plateau entirely.
    reelin_center : float
        Centre of the reel-in window in ``s`` (any value, taken mod 1: the
        window wraps across the periodic seam). The default 0.5 keeps
        ``s = 0`` (the periodic seam and the natural forward-trim start) in
        steady reel-out; with ``h = (1 - reelout_fraction)/2``, ``h`` starts
        the reel-in exactly at ``s = 0``, ``1 - h`` ends it there (so
        ``s = 0`` is the reel-out start), and 0 puts ``s = 0`` at the top
        (middle) of the reel-in. Off-centre windows move the seam toward or
        into the reel-in -- expect a harder trim start (see
        :func:`reelin_bump`).
    psi0 : float
        Constant figure-phase offset (rad). Shifts where in a figure-eight the
        reel-in window fades the oscillation out: the phase at the window edges
        sets how sharp the reel-out/reel-in handover is (a fade near a lobe
        extremum can leave a near-cusp), so ``psi0`` is the lever that keeps
        the handover smooth for ANY ``n_loops``. A constant offset preserves
        exact periodicity and C1 continuity at ``s = 0``. Ignored when the
        handover phases below are pinned.
    psi_entry, psi_exit : float or None
        Figure phase (rad) PINNED at the reel-in window edges: ``psi_entry``
        is the phase where the fade begins (figures -> reel-in handover) and
        ``psi_exit`` the phase where the figures resume. With ``downloops``
        the centre crossings climb, so ``psi_entry = 0`` exits the figure at
        azimuth 0 heading up (toward zenith) and ``psi_exit = pi`` re-enters
        at azimuth 0 heading the other way -- the first lobe after reel-in is
        on the OTHER side. Pinning one edge re-anchors the uniform phase
        (``psi0`` is ignored); pinning both decouples the edges: the reel-out
        gets the figure count nearest ``n_loops * reelout_fraction`` whose
        fractional part matches the requested phase gap (opposite-side
        re-entry <=> half-integer figures per reel-out), and the residual
        phase correction is hidden inside the window plateau where the figure
        amplitude is exactly zero -- which requires ``ramp_fraction < 0.5``
        (raises ``ValueError`` otherwise). Default ``None`` keeps the legacy
        ``psi0`` behaviour.
    bow_shape : str
        Reel-in azimuth-bow profile. ``"sym"`` (default) is the legacy
        antisymmetric bow ``sin(2*pi*xi)`` (out one side climbing, out the
        other descending, crossing zero at the top). ``"descent"`` is
        one-sided: the climb starts on the zero-azimuth meridian, the bow
        rises from mid-entry-ramp, the kite crosses the flat top moving
        sideways (the elevation plateau carries no path speed, so the
        azimuth must -- a bow that stays zero through the climb would stall
        the parametrization into an unflyable cusp), peaks at mid-exit-ramp
        and returns for the figure re-entry. ``"lobe"`` flies the reel-in
        as ONE giant lobe of the figure, which is what the full-cycle
        optimiser converges to: the figure PHASE FREEZES as the window
        fades in (it advances only ``LOBE_HANDOVER_PHASE`` more rad, over
        the start of the entry ramp, then holds through the window and
        resumes over the end of the exit ramp), so the figure term dies as a
        fading constant offset instead of wiggling on -- the climb peels off
        the last centre crossing and runs STRAIGHT (in azimuth/elevation) to
        the top of the zero meridian; a one-sided bow starting
        ``LOBE_BOW_OVERLAP`` of a ramp before the plateau carries the kite
        across the top; and the bow does NOT return -- the descent comes down
        on the ``az_reelin_amp`` side, where the window fade hands it over
        to the resuming figure term. For this shape ``psi_entry`` /
        ``psi_exit`` are the FROZEN phases: ``psi_entry`` is the figure
        point the climb peels off from (with ``downloops``, ``pi - 0.3`` =
        the climbing crossing a bit before azimuth 0, heading up toward the
        left lobe) and ``psi_exit`` the figure point the descent lands on
        (``3*pi/2`` = the left-lobe extreme heading down -- the descent is
        vertical there when ``|az_reelin_amp| == az_amp0`` -- then the kite
        flies the lower half of that lobe into the next crossing). The
        landing is ALWAYS on a side of the figure going down: choose
        ``az_reelin_amp`` on the ``psi_exit`` side so the descent lands
        tangentially on that lobe -- the reel-in replaces its top half, and
        the phase gap ties the re-entry side to the visible half-lobe
        PARITY: odd counts re-enter on the side OPPOSITE the peel-off lobe
        (e.g. leave after a right-lobe pass, land on the left lobe), even
        counts on the SAME side (see
        :func:`full_cycle_visible_half_figures`). Per count the ONE free
        geometric choice is ``az_reelin_through``: 0 = "smooth" (top loop
        straight to the landing side, no crossing); nonzero = "cross" (lean
        out to that side near the top, descend parked on it, cross
        ``az = 0`` low at ``reelin_cross_pos`` to reach the landing
        azimuth). The fade begins ``LOBE_HANDOVER_PHASE``
        before ``psi_entry`` and the figure is fully back that much after
        ``psi_exit``; the reel-out figure count is set as for the other
        shapes, from those edge phases. Requires both phases pinned (raises
        ``ValueError`` otherwise); the residual is hidden in the plateau. The
        sign of ``az_reelin_amp`` picks the side in all three shapes.
    downloops : bool
        Traversal sense (flips the azimuth direction).
    lobe_handover_phase : float
        ``bow_shape="lobe"`` only: the phase budget (rad) the figure still
        advances while the window fades it out / back in (default
        ``LOBE_HANDOVER_PHASE``). The freeze spans ``2 * budget / rate`` of
        the entry ramp, and the figure rate grows with the loop count -- so
        at HIGH loop counts a fixed budget completes ever earlier in the
        ramp, where the climb speed is still near zero, and the peel-off
        bend sharpens (curvature ~ rate). Enlarging the budget stretches the
        freeze back over the ramp and removes that spike. The visible lobe
        count between peel-off and landing is budget-INDEPENDENT (the fade
        flies the budget phase; the window edges sit that much outside the
        frozen phases), provided the snapped reel-out figure integer is
        unchanged -- vary the budget by less than pi/2 around the value used
        to derive ``n_loops``.
    """
    s = np.asarray(s, dtype=float).ravel()
    omega = 1.0 if downloops else -1.0
    f = float(reelout_fraction)
    if bow_shape not in ("sym", "descent", "lobe"):
        raise ValueError(
            f"bow_shape must be 'sym', 'descent' or 'lobe', got {bow_shape!r}"
        )
    if bow_shape == "lobe" and (psi_entry is None or psi_exit is None):
        raise ValueError(
            "bow_shape='lobe' freezes the figure phase through the reel-in and "
            "needs BOTH handover phases pinned (psi_entry and psi_exit)"
        )
    if az_reelin_through and bow_shape != "lobe":
        raise ValueError(
            "az_reelin_through (the through-side bulge) only exists for "
            f"bow_shape='lobe', got {bow_shape!r}"
        )

    c = float(reelin_center)
    ri = reelin_bump(
        s, reelout_fraction=f, ramp_fraction=ramp_fraction, reelin_center=c
    )
    window = 1.0 - ri  # figure amplitude: full in reel-out, 0 in reel-in

    # Position within the reel-in band (0 at entry, 1 at exit), via the same
    # circular offset as reelin_bump so the band wraps across the seam.
    h = 0.5 * (1.0 - f)
    d = (s - c + 0.5) % 1.0 - 0.5
    xi = np.clip((d + h) / max(2.0 * h, 1e-9), 0.0, 1.0)

    if psi_entry is None and psi_exit is None:
        # Continuous over the period -> periodic & C1 (psi0 shifts phase only).
        psi = 2.0 * np.pi * n_loops * s + float(psi0)
    elif psi_entry is not None and psi_exit is not None:
        # Both handover phases pinned. ``u`` is the circular path coordinate
        # anchored at the window EXIT (u = 0 where the figures resume): the
        # reel-out spans u in [0, f], the window u in [f, 1]. The reel-out
        # phase advance must end at psi_entry, so its figure count n_ro is
        # the value nearest the legacy visible count ``n_loops * f`` whose
        # fractional part equals the requested (entry - exit) phase gap.
        r_s = float(ramp_fraction) * (1.0 - f)
        if 1.0 - r_s <= f + r_s:
            raise ValueError(
                "pinning psi_entry AND psi_exit needs a reel-in plateau to "
                "absorb the phase correction: ramp_fraction must be < 0.5 "
                f"(got {float(ramp_fraction):g})"
            )
        psi_in_edge, psi_out_edge, gap = _pinned_handover_edges(
            psi_entry, psi_exit, bow_shape, lobe_handover_phase
        )
        n_ro = np.floor(n_loops * f - gap + 0.5) + gap
        if n_ro <= 0.0:
            n_ro = gap if gap > 0.0 else 1.0
        rate = 2.0 * np.pi * n_ro / f
        u = (d - h) % 1.0
        if bow_shape == "lobe":
            # Frozen phase: over the first ``rho`` of the entry ramp the
            # figure rate fades ``rate -> 0`` with a smoothstep (advancing
            # exactly LOBE_HANDOVER_PHASE, rho < 1), then psi HOLDS at
            # psi_entry through the rest of the window and resumes over the
            # last ``rho`` of the exit ramp -- the figure term
            # ``(1 - ri) * F(psi)`` dies as a fading constant offset instead
            # of wiggling on while the amplitude fades. Closed-form integrals
            # of the smoothstep keep this exact for unsorted ``s``. (If the
            # ramp is too short to hold the budget, rho caps at 1 and the
            # frozen phase drifts a little from the knob; periodicity is
            # still exact via dcorr.)
            rho = min(
                1.0, 2.0 * float(lobe_handover_phase) / max(rate * r_s, 1e-12)
            )
            l_f = rho * r_s
            t_in = np.clip((u - f) / l_f, 0.0, 1.0)
            t_out = np.clip((u - (1.0 - l_f)) / l_f, 0.0, 1.0)
            adv = (
                np.minimum(u, f)
                + l_f * (t_in - t_in**3 + 0.5 * t_in**4)
                + l_f * (t_out**3 - 0.5 * t_out**4)
            )
            # Residual (centred in (-pi, pi]) so u = 1 lands EXACTLY on the
            # exit-edge phase, hidden in the plateau where the amplitude is
            # exactly zero -- the freeze makes the jump invisible.
            dcorr = -(((rate * (f + l_f)) + np.pi) % (2.0 * np.pi) - np.pi)
            psi = (
                psi_out_edge
                + rate * adv
                + dcorr * _smoothstep(f + r_s, 1.0 - r_s, u)
            )
        else:
            # Residual to make sin(psi) exactly periodic at the u = 0/1 wrap
            # (total advance = 2*pi * integer), swallowed by a smoothstep
            # strictly inside the plateau where the figure amplitude is
            # EXACTLY zero -- it never shows in the path, and the rate
            # matches across the seam.
            dcorr = 2.0 * np.pi * (np.round(n_ro / f) - n_ro / f)
            psi = (
                float(psi_exit)
                + rate * u
                + dcorr * _smoothstep(f + r_s, 1.0 - r_s, u)
            )
    else:
        # One edge pinned: re-anchor the uniform phase so it hits the target
        # at that window edge (the legacy psi0 is replaced, not added).
        s_anchor = (c - h) % 1.0 if psi_exit is None else (c + h) % 1.0
        target = float(psi_entry if psi_exit is None else psi_exit)
        psi = 2.0 * np.pi * n_loops * (s - s_anchor) + target

    if bow_shape == "sym":
        # Bow to +az while climbing (xi < 0.5), to -az while descending
        # (xi > 0.5); gated by ``ri`` so it is zero outside the reel-in.
        bow = np.sin(omega * 2.0 * np.pi * xi)
    elif bow_shape == "descent":
        # One-sided bow: climb the zero meridian, drift out as the arc tops
        # out, cross the top moving SIDEWAYS, return during the descent. The
        # rise must begin at mid-entry-ramp (while the elevation still moves)
        # and peak at mid-exit-ramp: wherever the elevation plateau is flat
        # the azimuth must carry the path speed, or the parametrization
        # stalls (zero speed -> a cusp the trim cannot fly). Smoothstep up to
        # the peak and back keeps it C1 with zero slope at the window edges.
        edge = 0.5 * min(float(ramp_fraction), 0.5)
        bow = omega * (
            _smoothstep(edge, 1.0 - edge, xi)
            * (1.0 - _smoothstep(1.0 - edge, 1.0, xi))
        )
    else:  # "lobe"
        # One-sided bow that does NOT return: climb the zero meridian, drift
        # out over the last LOBE_BOW_OVERLAP of the entry ramp (so it carries
        # the path speed where the elevation plateau is flat), cross, and
        # stay out through the descent -- ``ri`` fades the bow and the figure
        # term (frozen at ``psi_exit``, the lobe extreme on the same side)
        # takes over, so the descent lands tangentially on the lobe instead
        # of swinging back to the meridian.
        edge = min(float(ramp_fraction), 0.5) * (1.0 - LOBE_BOW_OVERLAP)
        bow = omega * _smoothstep(edge, 1.0 - edge, xi)

    fig_az = window * az_amp0 * np.sin(omega * psi)
    if bow_shape == "lobe" and az_reelin_through:
        # "Cross" reel-in: climb the middle, lean out to the
        # ``az_reelin_through`` side near the top (S1), descend PARKED on
        # that azimuth (the returning figure term is compensated inside the
        # park so the kite comes down the side, not diagonally), then cross
        # az = 0 LOW -- centred at ``reelin_cross_pos`` of the exit ramp --
        # over S2, which crossfades the azimuth into ``fig + ri * amp``:
        # the plain lobe's tangential-landing blend, untouched from there.
        ramp = min(float(ramp_fraction), 0.5)
        pos = float(np.clip(reelin_cross_pos, 0.0, 1.0))
        x_mid = (1.0 - ramp) + ramp * pos
        x_cs = max(x_mid - 0.12, 0.5)
        x_ce = min(x_mid + 0.12, 0.97)
        # The lean-out must span the whole elevation plateau (ri flat over
        # (ramp, 1 - ramp)): wherever the elevation is flat the azimuth has
        # to carry the path speed, or the parametrization stalls (cusp).
        s_out = _smoothstep(edge, max(0.4, 1.0 - ramp), xi)
        s_cross = _smoothstep(x_cs, x_ce, xi)
        park = s_out * (1.0 - s_cross)
        az_bow = park * (omega * float(az_reelin_through) - fig_az) + (
            s_cross * ri * az_reelin_amp * omega
        )
    else:
        az_bow = ri * az_reelin_amp * bow

    azimuth = fig_az + az_bow
    elevation = (
        beta0
        + (beta_reelin_peak - beta0) * ri
        + window * beta_amp0 * np.sin(2.0 * psi)
    )
    return azimuth, elevation


def full_cycle_visible_half_figures(
    n_loops,
    reelout_fraction=0.7,
    psi_entry=None,
    psi_exit=None,
    bow_shape="sym",
    lobe_handover_phase=LOBE_HANDOVER_PHASE,
):
    """Half figure-eights (lobes) VISIBLE during reel-out, as a float.

    Replicates the pinned-phase snap of :func:`full_cycle_angles` (the
    reel-out flies ``n_ro = k + gap`` figures, ``k`` the integer that puts it
    nearest ``n_loops * reelout_fraction``) and returns ``2 * n_ro``; round
    it for the count you see. Its PARITY tells which re-entry side the
    tangential lobe descent (``az_reelin_amp`` on the ``psi_exit`` side)
    naturally gives: odd = opposite the peel-off lobe, even = same (see
    ``bow_shape="lobe"`` in :func:`full_cycle_angles`). With a free handover
    phase there is no snap and the visible count is simply
    ``2 * n_loops * reelout_fraction``.
    """
    f = float(reelout_fraction)
    if psi_entry is None or psi_exit is None:
        return 2.0 * float(n_loops) * f
    _, _, gap = _pinned_handover_edges(
        psi_entry, psi_exit, bow_shape, lobe_handover_phase
    )
    n_ro = np.floor(float(n_loops) * f - gap + 0.5) + gap
    if n_ro <= 0.0:  # mirror full_cycle_angles' degenerate fallback
        n_ro = gap if gap > 0.0 else 1.0
    return 2.0 * float(n_ro)


def full_cycle_n_loops_for_half_figures(
    n_halves,
    reelout_fraction=0.7,
    psi_entry=None,
    psi_exit=None,
    bow_shape="sym",
    lobe_handover_phase=LOBE_HANDOVER_PHASE,
):
    """``(n_loops, psi_entry)`` putting ``n_halves`` half-figures in the reel-out.

    The natural user-facing loop count is what you SEE: half figure-eights
    (lobes) flown during reel-out. ``n_loops`` in :func:`full_cycle_angles` is
    the continuous full-period figure count instead, and with pinned handover
    phases the VISIBLE reel-out figure count snaps to ``k + gap`` (integer
    ``k``; ``gap`` the fractional figure fixed by the entry/exit phases), so
    typed counts and visible lobes drift apart. This inverts the snap:

    - BOTH phases pinned: returns the (generally non-integer) ``n_loops``
      whose snapped reel-out count is closest to ``n_halves / 2`` figures.
      The pinned phases fix the PARITY of reachable half-counts (the descent
      must land on a given lobe), so when the requested parity does not match,
      ``psi_entry`` is shifted by pi -- the handover moves to the mirrored
      figure point (for the designed "lobe" shape: the climb peels off the
      OTHER, equally valid, climbing centre crossing) which flips the gap by
      half a figure while leaving the descent landing (``psi_exit``)
      untouched. The returned ``psi_entry`` REPLACES the input one. The
      count is always honoured exactly; the re-entry SIDE is chosen
      independently through the ``az_reelin_amp`` sign (tangential vs
      crossing descent, see ``bow_shape="lobe"`` in
      :func:`full_cycle_angles`).
    - Any phase free (``None``): exact periodicity needs an integer figure
      count per period, so the nearest integer to
      ``n_halves / (2 * reelout_fraction)`` is returned and ``psi_entry``
      passes through unchanged.
    """
    n_halves = int(n_halves)
    if n_halves < 1:
        raise ValueError(f"n_halves must be a positive integer, got {n_halves}")
    f = float(reelout_fraction)
    if psi_entry is None or psi_exit is None:
        return max(1, int(round(n_halves / (2.0 * f)))), psi_entry
    best = None
    for shift in (0.0, np.pi):
        candidate_entry = (float(psi_entry) + shift) % (2.0 * np.pi)
        _, _, gap = _pinned_handover_edges(
            candidate_entry, psi_exit, bow_shape, lobe_handover_phase
        )
        k = max(int(round(n_halves / 2.0 - gap)), 0)
        n_ro = k + gap
        if n_ro <= 0.0:  # mirror full_cycle_angles' degenerate fallback
            n_ro = gap if gap > 0.0 else 1.0
        err = abs(2.0 * n_ro - n_halves)
        if best is None or err < best[0]:
            best = (err, n_ro, candidate_entry)
    _, n_ro, psi_entry = best
    return n_ro / f, psi_entry


def make_full_cycle_bspline_path_parameters(
    M,
    r0,
    n_fit=600,
    n_loops=4,
    reelout_fraction=0.7,
    beta0=0.35,
    beta_amp0=0.12,
    az_amp0=0.3,
    beta_reelin_peak=1.1,
    az_reelin_amp=0.25,
    az_reelin_through=0.0,
    reelin_cross_pos=0.7,
    ramp_fraction=0.4,
    reelin_center=0.5,
    psi0=0.0,
    psi_entry=None,
    psi_exit=None,
    bow_shape="sym",
    downloops=True,
    lobe_handover_phase=LOBE_HANDOVER_PHASE,
    precision=6,
):
    """YAML-ready *periodic* path parameters for a synthetic full pumping cycle.

    Fits a periodic cubic B-spline to :func:`full_cycle_angles` over ``s in
    [0, 1]`` and returns the same dict shape as
    :func:`make_bspline_path_parameters_from_named_curve`. Use as a clean,
    trim-feasible initial guess for the single-phase full-cycle optimisation when
    a fit to flight data is too rough to trim.
    """
    s_samples = np.linspace(0.0, 1.0, int(n_fit), endpoint=False)
    az_target, el_target = full_cycle_angles(
        s_samples,
        n_loops=n_loops,
        reelout_fraction=reelout_fraction,
        beta0=beta0,
        beta_amp0=beta_amp0,
        az_amp0=az_amp0,
        beta_reelin_peak=beta_reelin_peak,
        az_reelin_amp=az_reelin_amp,
        az_reelin_through=az_reelin_through,
        reelin_cross_pos=reelin_cross_pos,
        ramp_fraction=ramp_fraction,
        reelin_center=reelin_center,
        psi0=psi0,
        psi_entry=psi_entry,
        psi_exit=psi_exit,
        bow_shape=bow_shape,
        downloops=downloops,
        lobe_handover_phase=lobe_handover_phase,
    )
    _, C_phi, C_beta = fit_bspline_pattern_to_trajectory(
        spline_type="periodic",
        M=M,
        s_init=0.0,
        s_final=1.0,
        az_target=az_target,
        el_target=el_target,
        s_samples=s_samples,
        downloops=downloops,
    )

    def _rounded_coefficients(coefficients):
        values = np.round(coefficients.full().flatten(), precision)
        values[np.isclose(values, 0.0)] = 0.0
        return values.tolist()

    return {
        "r0": float(r0),
        "M": int(M),
        "C_phi": _rounded_coefficients(C_phi),
        "C_beta": _rounded_coefficients(C_beta),
        "s_init": 0.0,
        "s_final": 1.0,
        "downloops": bool(downloops),
    }
