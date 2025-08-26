import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
from matplotlib.widgets import Slider

# TODO: find a way to parametrize the current trajectory with as few parameters as possible, this will be used as an initial guess for optimization

# Trajectory description: 

''' 

Start with tether in the middle of operating range (150-300m). Launch the kite to the zenith, make it come down and 
reel-in a bit to reach start of reel-out. Start doing the figures of 8 (while reeling out the tether) and when you get 
close to the end of the tether, stop the figures of 8, rise up back to the zenith and repeat the cycle. 

On the way up to the zenith after a reel-out, the "reel-in" phase starts when the kite is pointing straight up in the 
last figure of 8 (easier to make sure the trajectory is "continuous", aka derivatives are the same). On the contrary, 
the "reel-out" phase starts when the kite is pointing straight down, at the start of the first figure of 8. Before and 
after the reel-out phase, the trajectory has to recenter itself, in azimuth. What I mean is, as we are starting, and ending 
the reel-out phase in the first/last figure of eight pointing up or down, the kite will be in a position of max azimuth, 
thus at the end of the reel-out it has to rise to the zenith and reduce azimuth to 0 and before the start of the reel-out 
it has to descend to operating elevation and increase azimuth to catch the start of the 8.

I thus need two trajectories, one for the reel-out phase and one for the reel-in phase. The end points need to be the same, 
and their gradients need to match as well. I think that for the reel-out I can use basic trigonometric functions to describe the motion, 
while for the reel-in phase I will use a polynomial of order 5 such that the number of parameters is constant between both 
reel-in and -out optimizations.

'''

""" Reel-out will be of the following form (can be changed in necessary) """

mean_tether = 200 #m
max_width_fig8 = mean_tether * np.sin(np.pi / 5) #m
max_height_fig8 = 100 #m
max_depth_fig8 = 150 #m

# Initial parameters
a_init = max_width_fig8
b_init = 0.5
c_init = max_height_fig8
d_init = 1.0
e_init = 2.0

accuracy = 0.01
duration = round(24 * np.pi)
n_points = int(duration / accuracy)  # number of points

# Generate initial curve (perfect figures of 8 when either x or z is twice as fast as the other)
t = np.linspace(0, duration, n_points)

start_reel_out = [0, 20, 15]
shift = np.pi/2 + 0.5
# the shift is there to make sure the figures of 8 end with an upwards motion
# and that the start of the figures of 8 end with downwards motion

x = -a_init * np.cos(b_init * (t + shift)) + start_reel_out[0]
y = max_depth_fig8 * (1 + e_init * t / duration) + start_reel_out[1]
z = c_init * np.sin(d_init * (t + shift)) + start_reel_out[2]

reel_out_curve = np.vstack([x, y, z])
gradient = np.gradient(reel_out_curve, axis=1)

gradient_end_reel_out = gradient[:, -1]
gradient_start_reel_out = gradient[:, 0]

# print(x[0], y[0], z[0])
# print(x[-1], y[-1], z[-1])
# print(reel_out_curve[:,0])
# print(reel_out_curve[:,-1])

# Set up figure and axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
line, = ax.plot(x, y, z)

# Adjust subplot to fit sliders
plt.subplots_adjust(bottom=0.22)

# Slider axes
ax_a = plt.axes([0.15, 0.14, 0.75, 0.02])
ax_b = plt.axes([0.15, 0.11, 0.75, 0.02])
ax_c = plt.axes([0.15, 0.08, 0.75, 0.02])
ax_d = plt.axes([0.15, 0.05, 0.75, 0.02])
ax_e = plt.axes([0.15, 0.02, 0.75, 0.02])

# Sliders
slider_a = Slider(ax_a, 'a', -30.0, 30.0, valinit=a_init)
slider_b = Slider(ax_b, 'b', -2.0, 2.0, valinit=b_init)
slider_c = Slider(ax_c, 'c', -30.0, 30.0, valinit=c_init)
slider_d = Slider(ax_d, 'd', -2.0, 2.0, valinit=d_init)
slider_e = Slider(ax_e, 'e', 0.0, 4.0, valinit=e_init)

# Update function
def update(val):
    a = a_init + slider_a.val
    b = b_init + slider_b.val
    c = c_init + slider_c.val
    d = d_init + slider_d.val
    e = e_init + slider_e.val

    x_new = -a * np.cos(b * (t + shift))
    y_new = max_depth_fig8 * (1 + e * t / duration)
    z_new = c * np.sin(d * (t + shift))

    margin = 0.1  # 10% margin

    ax.set_xlim(np.min(x_new) - margin, np.max(x_new) + margin)
    ax.set_ylim(np.min(y_new) - margin, np.max(y_new) + margin)
    ax.set_zlim(np.min(z_new) - margin, np.max(z_new) + margin)

    line.set_xdata(x_new)
    line.set_ydata(y_new)
    line.set_3d_properties(z_new)
    fig.canvas.draw_idle()

slider_a.on_changed(update)
slider_b.on_changed(update)
slider_c.on_changed(update)
slider_d.on_changed(update)
slider_e.on_changed(update)

plt.show()

""" Reel-in will be of the following form (can be changed if necessary) """

import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline

# Fixed endpoints
P0_num = reel_out_curve[:,-1]
Pf_num = reel_out_curve[:,0]

# Desired start/end gradients
v0_num = gradient_end_reel_out
vf_num = gradient_start_reel_out

scale_start = 1/3
scale_end = 1/3

P1_num = P0_num + scale_start * v0_num
P4_num = Pf_num - scale_end * vf_num

# Free symbolic points
P2 = ca.SX.sym('P2',3)
P3 = ca.SX.sym('P3',3)

# Precompute uniform clamped cubic B-spline basis at discrete t
def bspline_basis6(t_vals):
    # 6 control points, degree 3 → 6+3+1 = 10 knots
    knots = [0,0,0,0,0.3,0.7,1,1,1,1]  # length 10
    n_points = len(t_vals)
    B = np.zeros((6,n_points))
    for j in range(6):
        c = np.zeros(6)
        c[j] = 1
        spl = BSpline(knots,c,3)
        B[j,:] = spl(t_vals)
    return B

# Discretize t for plotting
n_points = 200
t_vals = np.linspace(0,1,n_points)
B_vals = bspline_basis6(t_vals)

# Evaluate curve symbolically in CasADi
curve_points = []
for k in range(n_points):
    t = t_vals[k]
    # linear combination: P0,P1,P2,P3,P4,P5
    pt = B_vals[0,k]*P0_num + B_vals[1,k]*P1_num + B_vals[2,k]*P2 + \
         B_vals[3,k]*P3 + B_vals[4,k]*P4_num + B_vals[5,k]*Pf_num
    curve_points.append(pt)
curve_points = ca.vertcat(*curve_points)

# CasADi function
curve_func = ca.Function('curve_func',[P2,P3],[curve_points])

# Example middle points
P2_num = np.array([-20, 300, 300])
P3_num = np.array([-60, 280, 200])

# Evaluate numerically
curve_eval = curve_func(P2_num,P3_num).full().reshape(200,3)

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(curve_eval[:,0], curve_eval[:,1], curve_eval[:,2], color='green')
ax.scatter(*zip(P0_num,P1_num,P2_num,P3_num,P4_num,Pf_num), color='red')
plt.show()

# Final trajectory plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(reel_out_curve[0,:], reel_out_curve[1,:], reel_out_curve[2,:], label='Reel-out', color='blue')
ax.plot(curve_eval[:,0], curve_eval[:,1], curve_eval[:,2], label='Curve', color='green')
plt.show()