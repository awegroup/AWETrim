import numpy as np
import matplotlib.pyplot as plt


def bernstein_i2_order4_periodic_segment(t, a, width=0.25, normalize=False):
    """
    Periodic B_2^4 'bump' on a circular domain t ∈ [0,1).
    Nonzero only for points whose circular distance from start 'a'
    (wrapping around) is within 'width'. Inside, evaluate B_2^4 on s∈[0,1].

        s = ((t - a) mod 1) / width     if ((t - a) mod 1) <= width
        B_2^4(s) = 6 s^2 (1 - s)^2      else 0

    Args:
        t: array-like in [0,1]
        a: start of the segment (can be any real; wraps mod 1)
        width: segment length in (0,1]
        normalize: if True, divide by 'width' to preserve area

    Returns:
        Array same shape as t.
    """
    t = np.asarray(t, dtype=float) % 1.0
    width = float(width)
    if not (0.0 < width <= 1.0):
        raise ValueError("width must be in (0, 1].")

    # circular offset from start a
    delta = (t - a) % 1.0  # in [0,1)
    mask = delta <= width  # inside the window
    B = np.zeros_like(t)
    if np.any(mask):
        s = delta[mask] / width  # normalize to [0,1]
        val = 6.0 * (s**2) * ((1.0 - s) ** 2)  # B_2^4(s)
        if normalize:
            val = val / width
        B[mask] = val
    return B


# --- phase progress (half for φ, quarter for β) ---
def progress_from_phase(t):
    th_phi = np.mod(np.arctan2(np.sin(2 * np.pi * t), np.cos(2 * np.pi * t)), 2 * np.pi)
    u_phi = (th_phi / np.pi) % 1.0
    th_beta = np.mod(
        np.arctan2(np.sin(4 * np.pi * t), np.cos(4 * np.pi * t)), 2 * np.pi
    )
    u_beta = (th_beta / (0.5 * np.pi)) % 1.0
    return u_phi, u_beta


# --- build N(u) = 1 + sum w_k p_k(u) with bumps centered at k/K ---
def build_shape(u, K, width, weights, bump_fn):
    weights = np.asarray(weights, dtype=float)
    N = np.ones_like(u)
    for k, w in enumerate(weights):
        center = k / K
        start = center
        N += w * bump_fn(u, a=start, width=width)
    return N


# --- warp class by multiplying deviation-from-center by N(u) ---
def warp_class_with_shape(
    t,
    a_phi,
    b_beta,
    c_phi,
    c_beta,
    w_phi,
    w_beta,
    width_phi=0.5,
    width_beta=0.5,
    left_first=True,
):
    sgn = -1.0 if left_first else +1.0
    phi_c = c_phi + sgn * a_phi * np.sin(2 * np.pi * t)
    beta_c = c_beta + b_beta * np.sin(4 * np.pi * t)
    u_phi, u_beta = progress_from_phase(t)
    N_phi = build_shape(
        t,
        K=len(w_phi),
        width=width_phi,
        weights=w_phi,
        bump_fn=bernstein_i2_order4_periodic_segment,
    )
    plt.plot(t, N_phi, label="N_phi")
    N_beta = build_shape(
        t,
        K=len(w_beta),
        width=width_beta,
        weights=w_beta,
        bump_fn=bernstein_i2_order4_periodic_segment,
    )
    plt.plot(t, N_beta, label="N_beta")
    phi_w = c_phi + (phi_c - c_phi) * N_phi
    beta_w = c_beta + (beta_c - c_beta) * N_beta
    return phi_w, beta_w, phi_c, beta_c


# ================== demo: plot original vs warped ==================
t = np.linspace(0, 1, 2000, endpoint=False)

# class params
a_phi = np.radians(20.0)
b_beta = np.radians(10.0)
c_phi = 0.0
c_beta = np.radians(35.0)

# shape weights (example; replace with yours)
w_phi = [
    0.1055,
    0.0178,
    -0.0795,
    -0.1463,
    0.1930,
    0.1055,
    0.0178,
    -0.0795,
    -0.1463,
    0.1930,
]
w_beta = [
    -0.0134,
    0.3121,
    -0.7440,
    0.7104,
    -0.5040,
    -0.0134,
    0.3121,
    -0.7440,
    0.7104,
    -0.5040,
]
width_phi = 0.5
width_beta = 0.5

phi_w, beta_w, phi_c, beta_c = warp_class_with_shape(
    t,
    a_phi,
    b_beta,
    c_phi,
    c_beta,
    w_phi=w_phi,
    w_beta=w_beta,
    width_phi=width_phi,
    width_beta=width_beta,
    left_first=True,
)

# --- plots: time series and (φ,β) plane ---
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))

# time series
ax[0].plot(t, np.degrees(phi_c), "--", alpha=0.5, label="φ class")
ax[0].plot(t, np.degrees(phi_w), lw=2, label="φ warped")
ax[0].plot(t, np.degrees(beta_c), "--", alpha=0.5, label="β class")
ax[0].plot(t, np.degrees(beta_w), lw=2, label="β warped")
ax[0].set_xlabel("t")
ax[0].set_ylabel("angle [deg]")
ax[0].set_title("Original vs. Warped (time series)")
ax[0].legend(ncol=2)
ax[0].grid(True, alpha=0.3)

# φ-β Lissajous
ax[1].plot(np.degrees(phi_c), np.degrees(beta_c), "--", alpha=0.4, label="class")
ax[1].plot(np.degrees(phi_w), np.degrees(beta_w), lw=2, label="warped")
ax[1].set_xlabel("φ [deg]")
ax[1].set_ylabel("β [deg]")
ax[1].set_title("(φ, β) plane")
ax[1].axis("equal")
ax[1].legend()
ax[1].grid(True, alpha=0.3)

plt.tight_layout()


from picawe.kinematics.parametrized_patterns import CST_Lissajous
import casadi as ca

# your NumPy setup: t in [0,1)
t = np.linspace(0, 100, 2000, endpoint=False)
s = np.linspace(0, 2 * np.pi, 2000, endpoint=False)  # s in [0, 2π)
# build the CasADi version

pattern = CST_Lissajous(
    omega=1,
    r0=200.0,
    az_amp0=np.radians(20.0),
    beta_amp0=np.radians(10.0),
    vr=0.0,
    beta0=np.radians(35.0),
    beta_coeffs=w_beta[:5],
    az_coeffs=w_phi[:5],
    kappa=0.0,
    kbeta=0.0,
    width_phi=width_phi,
    width_beta=width_beta,
    left_first=True,
    repeat_beta=True,  # if True, repeat the beta coefficients (for symmetry)
    repeat_phi=True,  # if True, repeat the phi coefficients (for symmetry)
)

t_sym = ca.MX.sym("t")  # not used in this example, but kept for API
s_sym = ca.MX.sym("s")

phi_expr = pattern.azimuth(t_sym, s_sym)
beta_expr = pattern.elevation(t_sym, s_sym)
f = ca.Function("phi_beta", [t_sym, s_sym], [phi_expr, beta_expr])


f_map = f.map(t.size)  # vectorized
phi_dm, beta_dm = f_map(t, s)  # 0.0 for t
phi_cas = np.array(phi_dm).reshape(-1)
beta_cas = np.array(beta_dm).reshape(-1)

# overlay on your existing axes
ax[1].plot(
    np.degrees(phi_cas), np.degrees(beta_cas), lw=1.5, alpha=0.8, label="CasADi warped"
)
ax[1].legend()
plt.show()

plt.figure()
plt.plot(np.degrees(s), pattern.radius_curvature(t, s))
plt.show()
