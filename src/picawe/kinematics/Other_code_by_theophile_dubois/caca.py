# casadi_bspline_trajectory.py
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

# ---------- 1) Reference points (REPLACE with your ~10 points) ----------
az_deg = np.array([-60, -45, -20, 0, 20, 35, 45, 50, 40, 20], dtype=float)
el_deg = np.array([10, 20, 35, 45, 55, 60, 55, 45, 30, 15], dtype=float)
assert az_deg.shape == el_deg.shape and az_deg.ndim == 1
n = az_deg.size
assert n >= 4, "Need at least 4 points for a cubic spline."

# # ---------- 2) Optional: close the loop ----------
# CLOSE_LOOP = False
# if CLOSE_LOOP:
#     az_deg = np.r_[az_deg, az_deg[:3]]  # pad k=3 points for C2 continuity
#     el_deg = np.r_[el_deg, el_deg[:3]]

# ---------- 3) Chord-length parameterization on [0,1] ----------
pts = np.vstack([az_deg, el_deg]).T
d = np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1))
s = np.hstack([[0.0], np.cumsum(d)])
if s[-1] == 0.0:
    s[-1] = 1.0
s_norm = s / s[-1]

# ---------- 4) CasADi cubic B-spline interpolants az(s), el(s) ----------
# interpolant(name, 'bspline', [grid], values, opts)
opts = {"degree": [3]}  # CasADi expects an int vector for bspline degree
spl_az = ca.interpolant("az_spline", "bspline", [s_norm], az_deg, opts)
spl_el = ca.interpolant("el_spline", "bspline", [s_norm], el_deg, opts)


# helpers to evaluate from Python (vectorized)
def eval_spline(f, t):
    # f returns DM; accept numpy array and return numpy array
    return np.array(f(t).full()).ravel()


# ---------- 5) Dense sampling ----------
m = 1200
t = np.linspace(0.0, 1.0, m)
az_fit = eval_spline(spl_az, t)
el_fit = eval_spline(spl_el, t)

# ---------- 6) 3D mapping (unit sphere) ----------
az = np.deg2rad(az_fit)
el = np.deg2rad(el_fit)
x = np.cos(el) * np.cos(az)
y = np.cos(el) * np.sin(az)
z = np.sin(el)

# ref points in 3D
az_r = np.deg2rad(az_deg)
el_r = np.deg2rad(el_deg)
xr = np.cos(el_r) * np.cos(az_r)
yr = np.cos(el_r) * np.sin(az_r)
zr = np.sin(el_r)

# ---------- 7) Plots ----------
# A) Elevation vs Azimuth
plt.figure(figsize=(7, 5))
plt.plot(az_fit, el_fit, linewidth=2)
plt.scatter(az_deg, el_deg)
plt.xlabel("Azimuth (deg)")
plt.ylabel("Elevation (deg)")
plt.title("CasADi B-spline fit: Elevation vs Azimuth")
plt.tight_layout()
plt.show()

# B) Elevation vs parameter
plt.figure(figsize=(7, 4))
plt.plot(t, el_fit)
plt.scatter(s_norm, el_deg)
plt.xlabel("Spline parameter (0..1)")
plt.ylabel("Elevation (deg)")
plt.title("Elevation along B-spline parameter")
plt.tight_layout()
plt.show()

# C) Azimuth vs parameter
plt.figure(figsize=(7, 4))
plt.plot(t, az_fit)
plt.scatter(s_norm, az_deg)
plt.xlabel("Spline parameter (0..1)")
plt.ylabel("Azimuth (deg)")
plt.title("Azimuth along B-spline parameter")
plt.tight_layout()
plt.show()

# D) 3D path on unit sphere
from mpl_toolkits.mplot3d import Axes3D  # noqa

fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection="3d")

# wireframe sphere (for context)
phi = np.linspace(0, np.pi, 24)
theta = np.linspace(-np.pi, np.pi, 48)
sx = np.outer(np.sin(phi), np.cos(theta))
sy = np.outer(np.sin(phi), np.sin(theta))
sz = np.outer(np.cos(phi), np.ones_like(theta))
ax.plot_wireframe(sx, sy, sz, linewidth=0.25)

ax.plot(x, y, z, linewidth=2)
ax.scatter(xr, yr, zr, s=40)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Trajectory on Unit Sphere (CasADi B-spline)")
ax.set_box_aspect([1, 1, 1])
plt.tight_layout()
plt.show()

# ---------- 8) Expose callable functions ----------
# Example: evaluate at t0 = 0.25
t0 = np.array([0.25])
print(
    "az(0.25), el(0.25) [deg] =", eval_spline(spl_az, t0)[0], eval_spline(spl_el, t0)[0]
)
