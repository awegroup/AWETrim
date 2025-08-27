import casadi as ca
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

# --- B-spline setup ---
p = 3
n_ctrl = 6
U = [0.0,0.0,0.0,0.0, 1.0/3.0, 2.0/3.0, 1.0,1.0,1.0,1.0]

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
def make_B_spline(T):
    s = ca.SX.sym('s')
    u = s / T
    c2 = ca.SX.sym('c2', 3)
    c3 = ca.SX.sym('c3', 3)
    p0 = ca.SX.sym('p0', 3)
    v0 = ca.SX.sym('v0', 3)
    pf = ca.SX.sym('pf', 3)
    vf = ca.SX.sym('vf', 3)

    scale_start = 1/3
    scale_end = 1/3
    c1 = p0 + scale_start * v0
    c4 = pf - scale_end * vf
    C = ca.horzcat(p0, c1, c2, c3, c4, pf)

    S = ca.SX.zeros(3,1)
    dS = ca.SX.zeros(3,1)
    for i in range(n_ctrl):
        Ni = N_ip(u, i, p, U)
        dNi = dN_ip(u, i, p, U)
        S += ca.reshape(Ni,1,1)*C[:,i]
        dS += ca.reshape(dNi,1,1)*C[:,i]

    return ca.Function('B_spline', [s, c2, c3, p0, v0, pf, vf], [S, dS, C])

# --- Interactive example ---
if __name__ == "__main__":
    T = 10.0
    B_spline = make_B_spline(T)

    # Parameters
    p0 = np.array([0,0,0])
    v0 = np.array([1,0,0])
    pf = np.array([1,1,1])
    vf = np.array([0,1,0])
    c2 = np.array([0.3,0.2,0.5])
    c3 = np.array([0.6,0.5,0.8])
    s_vals = np.linspace(0, T, 100)

    # --- Setup figure ---
    fig = plt.figure(figsize=(9,6))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(bottom=0.35)

    # Initial spline points
    S_vals = np.array([B_spline(s, c2, c3, p0, v0, pf, vf)[0].full().flatten() for s in s_vals])
    line, = ax.plot(S_vals[:,0], S_vals[:,1], S_vals[:,2], 'b', lw=2)

    # Initial control points
    C_vals = B_spline(0, c2, c3, p0, v0, pf, vf)[2].full().T
    ctrl_points = ax.scatter(C_vals[:,0], C_vals[:,1], C_vals[:,2], color='red', s=50)

    ax.set_xlim(-0.5,1.5); ax.set_ylim(-0.5,1.5); ax.set_zlim(-0.5,1.5)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title('3D B-spline with adjustable c2, c3, p0, pf')

    # --- Slider axes positions ---
    axcolor = 'lightgoldenrodyellow'
    slider_width = 0.22
    slider_height = 0.03
    vspace = 0.01
    start_y = 0.25

    # Columns x-positions
    col_pos = [0.1, 0.4, 0.7]  # left, middle, right

    slider_axes = {}

    # --- Column 1: c2, c3 ---
    for i, name in enumerate(['c2x','c2y','c2z','c3x','c3y','c3z']):
        y = start_y - i*(slider_height+vspace)
        slider_axes[name] = plt.axes([col_pos[0], y, slider_width, slider_height], facecolor=axcolor)

    # --- Column 2: p0, pf ---
    for i, name in enumerate(['p0x','p0y','p0z','pfx','pfy','pfz']):
        y = start_y - i*(slider_height+vspace)
        slider_axes[name] = plt.axes([col_pos[1], y, slider_width, slider_height], facecolor=axcolor)

    # --- Column 3: v0, vf ---
    for i, name in enumerate(['v0x','v0y','v0z','vfx','vfy','vfz']):
        y = start_y - i*(slider_height+vspace)
        slider_axes[name] = plt.axes([col_pos[2], y, slider_width, slider_height], facecolor=axcolor)

    # --- Create sliders ---
    sliders = {}

    # Column 1
    sliders['c2x'] = Slider(slider_axes['c2x'], 'c2_x', -1, 2, valinit=c2[0])
    sliders['c2y'] = Slider(slider_axes['c2y'], 'c2_y', -1, 2, valinit=c2[1])
    sliders['c2z'] = Slider(slider_axes['c2z'], 'c2_z', -1, 2, valinit=c2[2])
    sliders['c3x'] = Slider(slider_axes['c3x'], 'c3_x', -1, 2, valinit=c3[0])
    sliders['c3y'] = Slider(slider_axes['c3y'], 'c3_y', -1, 2, valinit=c3[1])
    sliders['c3z'] = Slider(slider_axes['c3z'], 'c3_z', -1, 2, valinit=c3[2])

    # Column 2
    sliders['p0x'] = Slider(slider_axes['p0x'], 'p0_x', -1, 2, valinit=p0[0])
    sliders['p0y'] = Slider(slider_axes['p0y'], 'p0_y', -1, 2, valinit=p0[1])
    sliders['p0z'] = Slider(slider_axes['p0z'], 'p0_z', -1, 2, valinit=p0[2])
    sliders['pfx'] = Slider(slider_axes['pfx'], 'pf_x', -1, 2, valinit=pf[0])
    sliders['pfy'] = Slider(slider_axes['pfy'], 'pf_y', -1, 2, valinit=pf[1])
    sliders['pfz'] = Slider(slider_axes['pfz'], 'pf_z', -1, 2, valinit=pf[2])

    # Column 3
    sliders['v0x'] = Slider(slider_axes['v0x'], 'v0_x', -2, 2, valinit=v0[0])
    sliders['v0y'] = Slider(slider_axes['v0y'], 'v0_y', -2, 2, valinit=v0[1])
    sliders['v0z'] = Slider(slider_axes['v0z'], 'v0_z', -2, 2, valinit=v0[2])
    sliders['vfx'] = Slider(slider_axes['vfx'], 'vf_x', -2, 2, valinit=vf[0])
    sliders['vfy'] = Slider(slider_axes['vfy'], 'vf_y', -2, 2, valinit=vf[1])
    sliders['vfz'] = Slider(slider_axes['vfz'], 'vf_z', -2, 2, valinit=vf[2])

    # --- Update function ---
    def update(val):
        c2_new = np.array([sliders['c2x'].val, sliders['c2y'].val, sliders['c2z'].val])
        c3_new = np.array([sliders['c3x'].val, sliders['c3y'].val, sliders['c3z'].val])
        p0_new = np.array([sliders['p0x'].val, sliders['p0y'].val, sliders['p0z'].val])
        pf_new = np.array([sliders['pfx'].val, sliders['pfy'].val, sliders['pfz'].val])
        v0_new = np.array([sliders['v0x'].val, sliders['v0y'].val, sliders['v0z'].val])
        vf_new = np.array([sliders['vfx'].val, sliders['vfy'].val, sliders['vfz'].val])
        
        S_vals = np.array([B_spline(s, c2_new, c3_new, p0_new, v0_new, pf_new, vf_new)[0].full().flatten() for s in s_vals])
        C_vals = B_spline(0, c2_new, c3_new, p0_new, v0_new, pf_new, vf_new)[2].full().T

        # Update line
        line.set_data(S_vals[:,0], S_vals[:,1])
        line.set_3d_properties(S_vals[:,2])

        # Update control points
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

# --- End of file ---