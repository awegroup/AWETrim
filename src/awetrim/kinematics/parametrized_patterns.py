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


def build_periodic_cubic_bspline_function(M, dim=1, name="per_bspline"):
    """
    Build a CasADi function S = spline(C, u) for a uniform periodic cubic B-spline.

    - M: number of control points (periodic)
    - dim: output dimension (1 for scalar, 2 for [phi,beta] etc.)
    - C: (M, dim)
    - u: scalar in [0,1] (you map s -> u outside)

    Returns:
      spline_fun(C, u) -> S (1, dim)
    """
    C = ca.MX.sym("C", M, dim)
    u = ca.MX.sym("u")  # assumed in [0,1]

    x = u * M  # in [0, M]

    S = ca.MX.zeros(1, dim)

    # Sum from i=-2..M+1; wrap coefficient index with python int modulo
    for i in range(-2, M + 2):
        idx = i % M  # integer, safe for MX indexing
        t = x - i
        w = cubic_cardinal_B3(t)  # scalar
        S += w * C[idx, :].T  # (1,dim) += scalar*(1,dim)

    return ca.Function(name, [C, u], [S], ["C", "u"], ["S"])


class PeriodicBSpline(ParametrizedPatterns):

    def __init__(self, M, C_phi, C_beta, s_init, s_final, r0=None, downloops=True):
        super().__init__(
            M=M, C_phi=C_phi, C_beta=C_beta, s_init=s_init, s_final=s_final, r0=r0
        )
        self.M = int(M)
        self.s_init = float(s_init)
        self.s_final = float(s_final)
        self.omega = 1.0 if downloops else -1.0

        self.spline = build_periodic_cubic_bspline_function(
            self.M, dim=1, name=f"periodic_bspline_{self.M}"
        )

        self.C_phi = C_phi
        self.C_beta = C_beta

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


def full_cycle_angles(
    s,
    n_loops=5,
    reelout_fraction=0.7,
    beta0=0.35,
    beta_amp0=0.12,
    az_amp0=0.3,
    beta_reelin_peak=1.1,
    az_reelin_amp=0.25,
    ramp_fraction=0.4,
    reelin_center=0.5,
    psi0=0.0,
    psi_entry=None,
    psi_exit=None,
    bow_shape="sym",
    downloops=True,
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
    n_loops : int
        Figure-eights over the whole period (roughly ``reelout_fraction * n_loops``
        are visible; the rest fall inside the suppressed reel-in window).
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
        and returns for the figure re-entry. The sign of ``az_reelin_amp``
        picks the side in both shapes.
    downloops : bool
        Traversal sense (flips the azimuth direction).
    """
    s = np.asarray(s, dtype=float).ravel()
    omega = 1.0 if downloops else -1.0
    f = float(reelout_fraction)

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
        gap = ((float(psi_entry) - float(psi_exit)) / (2.0 * np.pi)) % 1.0
        n_ro = np.floor(n_loops * f - gap + 0.5) + gap
        if n_ro <= 0.0:
            n_ro = gap if gap > 0.0 else 1.0
        rate = 2.0 * np.pi * n_ro / f
        u = (d - h) % 1.0
        # Residual to make sin(psi) exactly periodic at the u = 0/1 wrap
        # (total advance = 2*pi * integer), swallowed by a smoothstep strictly
        # inside the plateau where the figure amplitude is EXACTLY zero -- it
        # never shows in the path, and the rate matches across the seam.
        dcorr = 2.0 * np.pi * (np.round(n_ro / f) - n_ro / f)
        psi = (
            float(psi_exit) + rate * u + dcorr * _smoothstep(f + r_s, 1.0 - r_s, u)
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
    else:
        raise ValueError(
            f"bow_shape must be 'sym' or 'descent', got {bow_shape!r}"
        )
    az_bow = ri * az_reelin_amp * bow

    azimuth = window * az_amp0 * np.sin(omega * psi) + az_bow
    elevation = (
        beta0
        + (beta_reelin_peak - beta0) * ri
        + window * beta_amp0 * np.sin(2.0 * psi)
    )
    return azimuth, elevation


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
    ramp_fraction=0.4,
    reelin_center=0.5,
    psi0=0.0,
    psi_entry=None,
    psi_exit=None,
    bow_shape="sym",
    downloops=True,
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
        ramp_fraction=ramp_fraction,
        reelin_center=reelin_center,
        psi0=psi0,
        psi_entry=psi_entry,
        psi_exit=psi_exit,
        bow_shape=bow_shape,
        downloops=downloops,
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
