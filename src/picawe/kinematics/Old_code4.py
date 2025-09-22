import casadi as ca
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
matplotlib.use('TkAgg')
import numpy as np
import tkinter as tk
from mpl_toolkits.mplot3d import Axes3D

# --- B-spline setup ---
p = 3
n_ctrl = 7
U = [0.0,0.0,0.0,0.0, 1.0/4.0, 2.0/4.0, 3.0/4.0, 1.0,1.0,1.0,1.0] 
#length of vector U = p + n_ctrl + 1
# 0 and 1 are repeated to clamp the initial and final position of the spline, repeat p + 1 times
# 0 and 1 represent the start and end of the spline, u = 0 is the start and u = 1 is the end
# u is an arbitrary variable 
# s is used in the rest of the codes but they are the same

# --- This code computes the B-spline basis function and the derivatives ---

def N_ip(u, i, k, U):
    if k == 0:
        return ca.if_else(ca.logic_and(u >= U[i], u < U[i+1]), 1.0, 0.0)
    else:
        left_den = U[i+k] - U[i]
        right_den = U[i+k+1] - U[i+1]
        left = 0
        right = 0
        if left_den != 0:
            left = (u - U[i]) / left_den * N_ip(u, i, k-1, U)
        if right_den != 0:
            right = (U[i+k+1] - u) / right_den * N_ip(u, i+1, k-1, U)
        return left + right

def dN_ip(u, i, k, U):
    if k == 0:
        return ca.DM.zeros(1)
    left_den = U[i+k] - U[i]
    right_den = U[i+k+1] - U[i+1]
    left = 0
    right = 0
    if left_den != 0:
        left = k / left_den * N_ip(u, i, k-1, U)
    if right_den != 0:
        right = k / right_den * N_ip(u, i+1, k-1, U)
    return left - right

# --- B-spline factory ---
# first we normalize the variable s, such that it is compatible with the U vector [0,1]
# n_ctrl = 6 which means we have 6 control points to play with
# the first and last (c0 and c5) are used to clamp start and end position
# c1 and c4 are used to attempt to match the start and end velocity
# c2 and c3 are left symbolic
# for elevation, azimuth and radial distance we then create a spline with c2 and c3 as the final controllable points

def make_B_spline(T):

    s = ca.SX.sym('s')
    u = s / T
    c2 = ca.SX.sym('c2', 3)
    c3 = ca.SX.sym('c3', 3)
    c4 = ca.SX.sym('c4', 3)
    p0 = ca.SX.sym('p0', 3)
    v0 = ca.SX.sym('v0', 3)
    pf = ca.SX.sym('pf', 3)
    vf = ca.SX.sym('vf', 3)

    scale_start = 1/3
    scale_end = 1/3
    c1 = p0 + scale_start * v0
    c5 = pf + scale_end * vf
    C = ca.horzcat(p0, c1, c2, c3, c4, c5, pf)

    S = ca.SX.zeros(3,1)
    dS = ca.SX.zeros(3,1)
    for i in range(n_ctrl):
        Ni = N_ip(u, i, p, U)
        dNi = dN_ip(u, i, p, U)
        S += ca.reshape(Ni,1,1)*C[:,i]
        dS += ca.reshape(dNi,1,1)*C[:,i]

    return ca.Function('B_spline', [s, c2, c3, c4, p0, v0, pf, vf], [S, dS, C])

# --- Interactive example ---
# T is the same as the max value of the s variable
# we make the spline
# the rest is for the purpose of visualization, plotting with sliders to adjust p0, pf, v0, vf, c2, and c3
# as the splines describe elevation, azimuth and radial distance, we also have to convert that into x y and z to plot 
# the trajectory in 3D

if __name__ == "__main__":
    T = 10.0
    B_spline = make_B_spline(T)

    # Parameters in azimuth (deg), elevation (deg), radial (m)
    p0 = np.array([0, 0, 150])
    v0 = np.array([10, 10, 5])
    pf = np.array([90, 5, 300])
    vf = np.array([5, 5, 5])
    c2 = np.array([30, 20, 200])
    c3 = np.array([60, 40, 250])
    c4 = np.array([80, 60, 125])

    s_vals = np.linspace(0, T, 100)

    # --- Convert spherical to Cartesian ---
    def sph2cart(az, el, r):
        az = np.deg2rad(az)
        el = np.deg2rad(el)
        x = r * np.cos(el) * np.cos(az)
        y = r * np.cos(el) * np.sin(az)
        z = r * np.sin(el)
        return np.array([x, y, z])

    # --- Setup figure ---
    fig = plt.figure(figsize=(9,6))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(bottom=0.35)

    # Initial spline points
    S_vals_sph = np.array([B_spline(s, c2, c3, c4, p0, v0, pf, vf)[0].full().flatten() for s in s_vals])
    # print(np.shape(S_vals_sph))
    S_vals = np.array([sph2cart(*s) for s in S_vals_sph])
    line, = ax.plot(S_vals[:,0], S_vals[:,1], S_vals[:,2], 'b', lw=2)

    # Initial control points
    C_vals_sph = B_spline(0, c2, c3, c4, p0, v0, pf, vf)[2].full().T
    C_vals = np.array([sph2cart(*c) for c in C_vals_sph])
    ctrl_points = ax.scatter(C_vals[:,0], C_vals[:,1], C_vals[:,2], color='red', s=50)

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title('3D B-spline in Cartesian (from az/el/r)')

    # --- Slider axes positions ---
    axcolor = 'lightgoldenrodyellow'
    slider_width = 0.2
    slider_height = 0.03
    vspace = 0.01
    start_y = 0.25
    col_pos = [0.05, 0.4, 0.7]  # left, middle, right

    slider_axes = {}

    # --- Column 1: c2, c3, c4 ---
    for i, name in enumerate(['c2a','c2e','c2r','c3a','c3e','c3r','c4a','c4e','c4r']):
        y = start_y - i*(slider_height+vspace)
        slider_axes[name] = plt.axes([col_pos[0], y, slider_width, slider_height], facecolor=axcolor)

    # --- Column 2: p0, pf ---
    for i, name in enumerate(['p0a','p0e','p0r','pfa','pfe','pfr']):
        y = start_y - i*(slider_height+vspace)
        slider_axes[name] = plt.axes([col_pos[1], y, slider_width, slider_height], facecolor=axcolor)

    # --- Column 3: v0, vf ---
    for i, name in enumerate(['v0a','v0e','v0r','vfa','vfe','vfr']):
        y = start_y - i*(slider_height+vspace)
        slider_axes[name] = plt.axes([col_pos[2], y, slider_width, slider_height], facecolor=axcolor)

    # --- Create sliders ---
    sliders = {}
    # Column 1: c2/c3
    sliders['c2a'] = Slider(slider_axes['c2a'], 'c2_az', -180, 180, valinit=c2[0])
    sliders['c2e'] = Slider(slider_axes['c2e'], 'c2_el', 0, 90, valinit=c2[1])
    sliders['c2r'] = Slider(slider_axes['c2r'], 'c2_r', 120, 350, valinit=c2[2])
    sliders['c3a'] = Slider(slider_axes['c3a'], 'c3_az', -180, 180, valinit=c3[0])
    sliders['c3e'] = Slider(slider_axes['c3e'], 'c3_el', 0, 90, valinit=c3[1])
    sliders['c3r'] = Slider(slider_axes['c3r'], 'c3_r', 120, 350, valinit=c3[2])
    sliders['c4a'] = Slider(slider_axes['c4a'], 'c4_az', -180, 180, valinit=c4[0])
    sliders['c4e'] = Slider(slider_axes['c4e'], 'c4_el', 0, 90, valinit=c4[1])
    sliders['c4r'] = Slider(slider_axes['c4r'], 'c4_r', 120, 350, valinit=c4[2])

    # Column 2: p0/pf
    sliders['p0a'] = Slider(slider_axes['p0a'], 'p0_az', -180, 180, valinit=p0[0])
    sliders['p0e'] = Slider(slider_axes['p0e'], 'p0_el', 0, 90, valinit=p0[1])
    sliders['p0r'] = Slider(slider_axes['p0r'], 'p0_r', 120, 350, valinit=p0[2])
    sliders['pfa'] = Slider(slider_axes['pfa'], 'pf_az', -180, 180, valinit=pf[0])
    sliders['pfe'] = Slider(slider_axes['pfe'], 'pf_el', 0, 90, valinit=pf[1])
    sliders['pfr'] = Slider(slider_axes['pfr'], 'pf_r', 120, 350, valinit=pf[2])

    # Column 3: v0/vf
    sliders['v0a'] = Slider(slider_axes['v0a'], 'v0_az', -180, 180, valinit=v0[0])
    sliders['v0e'] = Slider(slider_axes['v0e'], 'v0_el', 0, 90, valinit=v0[1])
    sliders['v0r'] = Slider(slider_axes['v0r'], 'v0_r', 120, 350, valinit=v0[2])
    sliders['vfa'] = Slider(slider_axes['vfa'], 'vf_az', -180, 180, valinit=vf[0])
    sliders['vfe'] = Slider(slider_axes['vfe'], 'vf_el', 0, 90, valinit=vf[1])
    sliders['vfr'] = Slider(slider_axes['vfr'], 'vf_r', 120, 350, valinit=vf[2])

    # --- Update function ---
    def update(val):
        c2_new = np.array([sliders['c2a'].val, sliders['c2e'].val, sliders['c2r'].val])
        c3_new = np.array([sliders['c3a'].val, sliders['c3e'].val, sliders['c3r'].val])
        c4_new = np.array([sliders['c4a'].val, sliders['c4e'].val, sliders['c4r'].val])
        p0_new = np.array([sliders['p0a'].val, sliders['p0e'].val, sliders['p0r'].val])
        pf_new = np.array([sliders['pfa'].val, sliders['pfe'].val, sliders['pfr'].val])
        v0_new = np.array([sliders['v0a'].val, sliders['v0e'].val, sliders['v0r'].val])
        vf_new = np.array([sliders['vfa'].val, sliders['vfe'].val, sliders['vfr'].val])

        # Evaluate spline in spherical coordinates
        S_vals_sph = np.array([B_spline(s, c2_new, c3_new, c4_new, p0_new, v0_new, pf_new, vf_new)[0].full().flatten() for s in s_vals])
        C_vals_sph = B_spline(0, c2_new, c3_new, c4_new, p0_new, v0_new, pf_new, vf_new)[2].full().T

        # Convert to Cartesian for plotting
        S_vals = np.array([sph2cart(*s) for s in S_vals_sph])
        C_vals = np.array([sph2cart(*c) for c in C_vals_sph])

        # Update line and control points
        line.set_data(S_vals[:,0], S_vals[:,1])
        line.set_3d_properties(S_vals[:,2])
        ctrl_points._offsets3d = (C_vals[:,0], C_vals[:,1], C_vals[:,2])

        # Auto axes scaling
        all_points = np.vstack([S_vals, C_vals])
        ax.set_xlim(all_points[:,0].min()-0.1, all_points[:,0].max()+0.1)
        ax.set_ylim(all_points[:,1].min()-0.1, all_points[:,1].max()+0.1)
        ax.set_zlim(all_points[:,2].min()-0.1, all_points[:,2].max()+0.1)

        fig.canvas.draw_idle()

    # Connect sliders
    for slider in sliders.values():
        slider.on_changed(update)

    plt.show()
