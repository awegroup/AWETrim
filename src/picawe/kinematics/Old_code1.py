import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline

# Free symbolic points
P0 = ca.SX.sym('P0',3)
P1 = ca.SX.sym('P1',3)
P2 = ca.SX.sym('P2',3)
P3 = ca.SX.sym('P3',3)
P4 = ca.SX.sym('P4',3)
P5 = ca.SX.sym('P5',3)

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
    pt = B_vals[0,k]*P0 + B_vals[1,k]*P1 + B_vals[2,k]*P2 + \
         B_vals[3,k]*P3 + B_vals[4,k]*P4 + B_vals[5,k]*P5
    curve_points.append(pt)
curve_points = ca.vertcat(*curve_points)

# CasADi function
curve_func = ca.Function('curve_func',[P0, P1, P2, P3, P4, P5],[curve_points])

# Example middle points
P0_num = [-142.3,  34.8, 160]
P1_num = [  58.7, 102.4,  95]
P2_num = [ 129.1,  67.9, 240]
P3_num = [ -21.5, 140.2, 185]
P4_num = [ 175.6,  15.3, 225]
P5_num = [ -88.0, 120.6, 275]


# P0_cartesian = [P0_num[2] * np.cos(P0_num[1]) * np.cos(P0_num[0]), P0_num[2] * np.cos(P0_num[1]) * np.sin(P0_num[0]), P0_num[2] * np.sin(P0_num[1])]
# P1_cartesian = [P1_num[2] * np.cos(P1_num[1]) * np.cos(P1_num[0]), P1_num[2] * np.cos(P1_num[1]) * np.sin(P1_num[0]), P1_num[2] * np.sin(P1_num[1])]
# P2_cartesian = [P2_num[2] * np.cos(P2_num[1]) * np.cos(P2_num[0]), P2_num[2] * np.cos(P2_num[1]) * np.sin(P2_num[0]), P2_num[2] * np.sin(P2_num[1])]
# P3_cartesian = [P3_num[2] * np.cos(P3_num[1]) * np.cos(P3_num[0]), P3_num[2] * np.cos(P3_num[1]) * np.sin(P3_num[0]), P3_num[2] * np.sin(P3_num[1])]
# P4_cartesian = [P4_num[2] * np.cos(P4_num[1]) * np.cos(P4_num[0]), P4_num[2] * np.cos(P4_num[1]) * np.sin(P4_num[0]), P4_num[2] * np.sin(P4_num[1])]
# P5_cartesian = [P5_num[2] * np.cos(P5_num[1]) * np.cos(P5_num[0]), P5_num[2] * np.cos(P5_num[1]) * np.sin(P5_num[0]), P5_num[2] * np.sin(P5_num[1])]

# Evaluate numerically
curve_eval = curve_func(P0_num, P1_num, P2_num, P3_num, P4_num, P5_num).full().reshape(200,3)

azimuth = curve_eval[:,0]
elevation = curve_eval[:,1]
distance = curve_eval[:,2]

x = distance * np.cos(elevation) * np.cos(azimuth)
y = distance * np.cos(elevation) * np.sin(azimuth)
z = distance * np.sin(elevation)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, label='Curve', color='green')
# ax.scatter(*zip(P0_num,P1_num,P2_num,P3_num,P4_num,P5_num), color='red')
# ax.scatter(*zip(P0_cartesian,P1_cartesian,P2_cartesian,P3_cartesian,P4_cartesian,P5_cartesian), color='red')
plt.show()