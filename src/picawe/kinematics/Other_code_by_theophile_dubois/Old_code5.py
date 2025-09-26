import casadi as ca
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
matplotlib.use('TkAgg')
import numpy as np
import tkinter as tk
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import re

# --- GOAL: TODO---
''' Goal of the next step is to refactor this into a class that is called cycle, 
which has methods to load data, extract cycles, fit splines, plot, etc. '''

def convert_time_to_seconds(time_array):
    """Convert time in HH:MM:SS.sss format to total seconds."""
    seconds_array = []
    for time_str in time_array:
        parts = re.split(r"[:.]", time_str)
        h = int(parts[0])
        m = int(parts[1])
        s = float(parts[2]) + float("0." + parts[3]) if len(parts) > 3 else float(parts[2])
        seconds = h * 3600 + m * 60 + s
        seconds_array.append(seconds)
    return np.array(seconds_array)

# Import simulated csv data
file_path_full = "/home/theophile/src/Simulation_Results/trial_Uri_valid/ProtoLogger_csv/2025-09-10_11-31-10_ProtoLogger.csv"
full_data = pd.read_csv(file_path_full, header=0, sep=r"\s+")
column_titles_full = list(full_data.columns)

# Index 122 corresponds to moment of take-off so we ignore data before that
azimuth_data = full_data["kite_azimuth"].to_numpy()
elevation_data = full_data["kite_elevation"].to_numpy()
radial_distance_data = full_data["kite_distance"].to_numpy()
time_data_full = np.round(convert_time_to_seconds(full_data["time_of_day"].to_numpy()), 1)
# print("Full time data (seconds):", time_data_full)

# print("Column titles:", column_titles_full)
# print("Azimuth data:", azimuth_data)
# print("Elevation data:", elevation_data)
# print("Radial distance data:", radial_distance_data)

file_path_cycle= "/home/theophile/src/Simulation_Results/trial_Uri_valid/cycles/cycle_data_sheet_lines.csv"
full_data = pd.read_csv(file_path_cycle, header=0)
column_titles_cycle = list(full_data.columns)
# print("Column titles:", column_titles_cycle)

time_data_cycle = np.round(convert_time_to_seconds(full_data["start_time_cycle_LT"].to_numpy()), 1)
# print("Cycle time data (seconds):", time_data_cycle)

idx_start_cycle = np.array([i for i, t in enumerate(time_data_full) for j, tc in enumerate(time_data_cycle) if t == tc])
# print("Start of cycle indices:", idx_start_cycle)
useful_cycles_idx = idx_start_cycle[1:-1]  # Ignore first and last cycle for safety
# print("Useful cycle indices:", useful_cycles_idx)

def sph2cart(az, el, r):
    x = r * np.cos(el) * np.cos(az)
    y = r * np.cos(el) * np.sin(az)
    z = r * np.sin(el)
    return x, y, z

def cycle_trajectory_cartesian(cycle_idx, azimuth_data, elevation_data, radial_distance_data):

    x, y, z = sph2cart(azimuth_data, elevation_data, radial_distance_data)

    start_idx = useful_cycles_idx[cycle_idx]
    end_idx = useful_cycles_idx[cycle_idx + 1]

    x = x[start_idx:end_idx]
    y = y[start_idx:end_idx]
    z = z[start_idx:end_idx]

    return x, y, z

def plot_cycle_trajectory(cycle_idx, x, y, z, RI_start_true, RI_end_true):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, label=f'Cycle {cycle_idx+1} Trajectory')
    ax.scatter(*RI_start_true, color='red', label='Reel-In Start Point', s=25)  # Reel-In start point
    ax.scatter(*RI_end_true, color='green', label='Reel-In End Point', s=25)  # Reel-In end point
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'Kite Trajectory for Cycle {cycle_idx+1}')
    ax.legend()
    ax.set_box_aspect([1,1,1])  # keep axes equal
    plt.show() 

def find_true_RI_start(x, y, z, RI_start_est):
    # squared distances to avoid unnecessary sqrt
    distances = np.sqrt((x - RI_start_est[0])**2 + (y - RI_start_est[1])**2 + (z - RI_start_est[2])**2)
    print(distances.shape)
    idx = np.argmin(distances)  # index of closest point
    return idx, (x[idx], y[idx], z[idx])

def find_true_RI_end(x, y, z, dx, dy, dz):

    start_point = (x[0], y[0], z[0])
    end_point = (x[-1], y[-1], z[-1])

    RI_end_true = (np.mean([start_point[0], end_point[0]]),
              np.mean([start_point[1], end_point[1]]),
              np.mean([start_point[2], end_point[2]]))
    
    RI_end_gradient = (np.mean([dx[0], dx[-1]]),
                       np.mean([dy[0], dy[-1]]),
                       np.mean([dz[0], dz[-1]]))

    return RI_end_true, RI_end_gradient

cycle_idx = 0  # First useful cycle

x_cyc, y_cyc, z_cyc = cycle_trajectory_cartesian(cycle_idx, azimuth_data, elevation_data, radial_distance_data)

dx, dy, dz = np.gradient(x_cyc), np.gradient(y_cyc), np.gradient(z_cyc)
gradient = np.column_stack((dx, dy, dz))

pf, vf = find_true_RI_end(x_cyc, y_cyc, z_cyc, dx, dy, dz)
idx, p0 = find_true_RI_start(x_cyc, y_cyc, z_cyc, (260, -130, 140))

v0 = gradient[idx]

plot_cycle_trajectory(cycle_idx, x_cyc, y_cyc, z_cyc, p0, pf)


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
    # print(B_spline)

    # Parameters in azimuth (deg), elevation (deg), radial (m)
    p0 = np.array([0, 0, 150])
    v0 = np.array([10, 10, 5])
    pf = np.array([90, 5, 300])
    vf = np.array([5, 5, 5])
    c2 = np.array([30, 20, 200])
    c3 = np.array([60, 40, 250])
    c4 = np.array([80, 60, 125])

    s_vals = np.linspace(0, T, 100)

    # Initial spline points
    S_vals_sph = np.array([B_spline(s, c2, c3, c4, p0, v0, pf, vf)[0].full().flatten() for s in s_vals])
    # print(np.shape(S_vals_sph))
    S_vals = np.array([sph2cart(*s) for s in S_vals_sph])

    # Initial control points
    C_vals_sph = B_spline(0, c2, c3, c4, p0, v0, pf, vf)[2].full().T
    C_vals = np.array([sph2cart(*c) for c in C_vals_sph])
