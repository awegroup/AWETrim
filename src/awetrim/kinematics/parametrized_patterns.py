# Copyright (c) 2023-2026 Oriol Cayon, Delft University of Technology
#
# Licensed under the EUPL, Version 1.2 or - as soon they will be approved by
# the European Commission - subsequent versions of the EUPL (the "Licence");
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at:
#
#     https://joinup.ec.europa.eu/software/page/eupl
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the Licence is distributed on an "AS IS" basis,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the Licence for the specific language governing permissions and
# limitations under the Licence.
#
# SPDX-License-Identifier: EUPL-1.2

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
    """Basis matrix B[n,j] for periodic cubic B-splines evaluated on u_grid."""
    u_grid = np.asarray(u_grid).ravel()
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

    Supported curves are ``lissajous`` and ``helix``. The returned arrays are
    numeric samples intended for fitting initial B-spline control points, not
    symbolic trajectory expressions.
    """
    s = np.asarray(s).ravel()
    omega = 1.0 if downloops else -1.0

    if curve_type == "lissajous":
        azimuth = az_amp0 * np.sin(omega * s)
        elevation = beta0 + beta_amp0 * np.sin(omega * 2.0 * s)
    elif curve_type == "helix":
        azimuth = az_amp0 * np.sin(omega * s)
        elevation = beta0 + beta_amp0 * np.cos(omega * s)
    else:
        raise ValueError(
            "curve_type must be one of 'lissajous' or 'helix'."
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
    """Fit a B-spline pattern to a named helix or figure-eight initial curve."""
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
    }
