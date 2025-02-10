import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from picawe.kinematics.parametrized_patterns import Helix, Lissajous, FigureEight
from picawe.kinematics.Kinematics import ParametrizedKinematics, KiteKinematics
from picawe import SystemModel
import casadi as ca
import time as timet
import json
from picawe.timeseries.phase import PhaseParameterized

# -----------------------------------------------
# Load data and define aerodynamic model
# -----------------------------------------------

# Define aerodynamic input
file_path = "./data/v3_aero_input.json"
# file_path = "./data/rigid_kite.json"
with open(file_path, "r") as file:
    aero_input = json.load(file)

# -----------------------------------------------
# Define the parametrized path
# -----------------------------------------------
omega = -1
x0 = 200
rh = 100
vr = 0
beta = np.radians(30)
ry = 120
rz = 40
helix = Helix(omega, x0, rh, vr, beta, kappa=0.5)
lissajous = Lissajous(omega, x0, ry, rz, vr, beta)
figure_eight = FigureEight(omega, x0, 80, 80, vr, beta)


# -----------------------------------------------
# Define the system and aerodynamic model
# -----------------------------------------------
dof = 3
quasi_steady = False
pattern = helix
kite_model = SystemModel(mass_wing=15, area_wing=20, aero_input=aero_input, mass_kcu=25, dof=dof, quasi_steady=quasi_steady)
kite_model.speed_wind = 10 
kite_model.input_depower = 0
phase = PhaseParameterized(kite_model, pattern, quasi_steady=quasi_steady)

# -----------------------------------------------
# Define simulation parameters and initial state
# -----------------------------------------------
time = np.arange(0, 100, 0.1)
start_state = {
    "t": 0,
    "s": np.pi/2,
    "s_dot": 0.4,
    "s_ddot": 0,
    "tension_tether_ground": 1e3,
    "input_steering": 0,
    "angle_roll": 0,
    "angle_pitch": 0,
    "angle_yaw": 0,
}

# -----------------------------------------------
# Run the simulation
# -----------------------------------------------
phase.run_simulation(start_state=start_state, time_array=time)


# -----------------------------------------------
# Extract the states
# -----------------------------------------------

fig, slider = phase.interactive_plot()

s = phase.return_variable("s")
mask = s > 2*np.pi
elevation = phase.return_variable("angle_elevation")[mask]
azimuth = phase.return_variable("angle_azimuth")[mask]
speed_tangential = phase.return_variable("speed_tangential")[mask]
course = phase.return_variable("angle_course")[mask]
tension_tether = phase.return_variable("tension_tether_ground")[mask]


# Find indices of max and min tangential speed
max_idx = np.argmax(speed_tangential)
min_idx = np.argmin(speed_tangential)

# Plot the trajectory with a colorbar for tangential speed
plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    azimuth, elevation, c=speed_tangential, cmap="viridis", s=10
)  # `s` adjusts marker size
cbar = plt.colorbar(scatter)
cbar.set_label("Tangential Speed [m/s]", fontsize=12)

# Plot max and min speed points
plt.scatter(
    azimuth[max_idx],
    elevation[max_idx],
    color="red",
    label="Max Speed",
    edgecolor="black",
    zorder=5,
)
plt.scatter(
    azimuth[min_idx],
    elevation[min_idx],
    color="red",
    label="Min Speed",
    edgecolor="black",
    zorder=5,
)

print(
    f"Max speed: {speed_tangential[max_idx]} m/s at phase {(s[max_idx]*omega*180/np.pi)%360} degrees"
)
# Labels, title, and legend
plt.xlabel("Azimuth [rad]", fontsize=12)
plt.ylabel("Elevation [rad]", fontsize=12)
plt.title("Flown Trajectory with Tangential Speed", fontsize=14)
plt.legend(fontsize=10)
plt.grid()
plt.show()