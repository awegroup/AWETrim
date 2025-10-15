import json
import numpy as np
from awetrim import Cycle

# -------------------- Load Aero Input --------------------
with open("./data/V11/v11_aero_input.json", "r") as file:
    aero_input = json.load(file)

with open("./data/LEI-V9-KITE/v9_aero_input.json", "r") as file:
    aero_input = json.load(file)

# -------------------- Simulation Config --------------------
SIMULATION_CONFIG = {
    "mass_ratio": 2,
    "dof": 3,
    "area_wing": 47,
    "mass_wing": 78,
    "mass_kcu": 0,
    "tether_diameter": 0.014,
    "wind_model": "logarithmic",
    "speed_friction": 0.45,
    "z0": 0.02,
    "steering_control": "roll",
}

# -------------------- Pattern Config --------------------
PATTERN_CONFIG = {
    "pattern_type": "figure_eight",
    "parameters": {
        "omega": -1.0,
        "r0": 210.0,
        "ry": 120,
        "rz": 94,
        "ky": 0.7,
        "kz": 0.7,
        "vr": 2.5,
        "beta0": 0.6775,
        "kappa": 0,
    },
    "control": {
        "input_depower": 0.0,
    },
    "start_time": 0,
    "end_time": 36,
    "n_points": 200,
    "quasi_steady": True,
}

CYCLE_SETTINGS = {
    "reelout": PATTERN_CONFIG,
    "reelin": {
        "quasi_steady": False,
        "control": {
            "max_elevation": np.radians(100),
            "min_elevation": np.radians(25),
            "reeling_speed": -7,
            "min_tether_force": SIMULATION_CONFIG["mass_wing"] * 9.81,
            "length_tether_ro": PATTERN_CONFIG["parameters"]["r0"],
            "ri_elevation": np.radians(40),  # Initial elevation for reeling in
        },
        "initial_state": {
            "angle_course": 0,
            "input_steering": 0,
            "input_depower": 0,
            "speed_tangential": 60,
            "timeder_angle_course": 0,
            "tension_tether_ground": 1e6,
        },
        "time_step": 0.1,
        "quasi_steady": True,
    },
}

# -------------------- Run Cycle --------------------
wind_speed = (
    SIMULATION_CONFIG["speed_friction"] / 0.4 * np.log(100 / SIMULATION_CONFIG["z0"])
)
print("Wind speed at 100m:", wind_speed)

cycle_sim = Cycle(aero_input, SIMULATION_CONFIG)
reelout_phase, reelin_phase = cycle_sim.run_cycle(CYCLE_SETTINGS)


# -------------------- Plotting --------------------
import matplotlib.pyplot as plt
from awetrim.utils.color_palette import get_color_list

# --- Extract variables from reel-out phase ---
t_ro = reelout_phase.return_variable("t")
lt_ro = reelout_phase.return_variable("distance_radial")
s_ro = reelout_phase.return_variable("s")
vtau_ro = reelout_phase.return_variable("speed_tangential")
azimuth_ro = reelout_phase.return_variable("angle_azimuth")
elevation_ro = reelout_phase.return_variable("angle_elevation")
roll_ro = reelout_phase.return_variable("angle_roll")
aoa_ro = reelout_phase.return_variable("angle_of_attack")
tension_ro = reelout_phase.return_variable("tension_tether_ground")
pow_reelout = reelout_phase.return_variable("mechanical_power")
x_reelout = reelout_phase.return_variable("x")
y_reelout = reelout_phase.return_variable("y")
z_reelout = reelout_phase.return_variable("z")

plt.figure()
plt.plot(azimuth_ro, elevation_ro, label="Reel-out Path", color=get_color_list()[0])
plt.xlabel("Azimuth Angle (rad)")
plt.ylabel("Elevation Angle (rad)")
# plt.show()

# --- Extract variables from reel-in phase ---
t_reelin = reelin_phase.return_variable("t")
lt_reelin = reelin_phase.return_variable("distance_radial")
vtau_reelin = reelin_phase.return_variable("speed_tangential")
tension_reelin = reelin_phase.return_variable("tension_tether_ground")
aoa_reelin = reelin_phase.return_variable("angle_of_attack")
roll_reelin = reelin_phase.return_variable("angle_roll")
depower_reelin = reelin_phase.return_variable("input_depower")
vr_reelin = reelin_phase.return_variable("speed_radial")
pow_reelin = reelin_phase.return_variable("mechanical_power")
x_reelin = reelin_phase.return_variable("x")
y_reelin = reelin_phase.return_variable("y")
z_reelin = reelin_phase.return_variable("z")
cl_reelin = reelin_phase.return_variable("lift_coefficient")
cd_reelin = reelin_phase.return_variable("drag_coefficient")
input_steering_reelin = reelin_phase.return_variable("input_steering")

depower_ro = reelout_phase.return_variable("input_depower")
vr_ro = reelout_phase.return_variable("speed_radial")
cl_ro = reelout_phase.return_variable("lift_coefficient")
cd_ro = reelout_phase.return_variable("drag_coefficient")
input_steering_ro = reelout_phase.return_variable("input_steering")

# --- Diagnostics ---
print("Mean CL: ", np.mean(reelout_phase.return_variable("lift_coefficient")))
print("Mean CD: ", np.mean(reelout_phase.return_variable("drag_coefficient")))
print("Mean AoA: ", np.mean(aoa_ro * 180 / np.pi))
print("Mean CL reelin: ", np.mean(reelin_phase.return_variable("lift_coefficient")))
print("Mean CD reelin: ", np.mean(reelin_phase.return_variable("drag_coefficient")))
print("Mean AoA reelin: ", np.mean(aoa_reelin * 180 / np.pi))
print("Mean Tether Tension Reelout: ", np.mean(tension_ro) / 1000, "kN")
print("Mean Tether Tension Reelin: ", np.mean(tension_reelin) / 1000, "kN")

# --- Energy calculations ---
dt_ro = np.diff(t_ro, prepend=t_ro[0])
energy_reelout = np.sum(pow_reelout * dt_ro)
print("Energy reelout: ", energy_reelout)
print("Mean power reelout: ", np.mean(pow_reelout))

dt_reelin = np.diff(t_reelin, prepend=t_reelin[0])
energy_reelin = np.sum(pow_reelin * dt_reelin)
print("Energy reelin: ", energy_reelin)
print("Mean power reelin: ", np.mean(pow_reelin))

# --- Combine full cycle variables ---
t_total = np.concatenate([t_ro, t_reelin])
lt_total = np.concatenate([lt_ro, lt_reelin])
vtau_total = np.concatenate([vtau_ro, vtau_reelin])
tension_total = np.concatenate([tension_ro, tension_reelin])
aoa_total = np.concatenate([aoa_ro, aoa_reelin])
roll_total = np.concatenate([roll_ro, roll_reelin])
depower_total = np.concatenate([depower_ro, depower_reelin])
vr_total = np.concatenate([vr_ro, vr_reelin])
power_total = np.concatenate([pow_reelout, pow_reelin])
cl_total = np.concatenate([cl_ro, cl_reelin])
cd_total = np.concatenate([cd_ro, cd_reelin])
input_steering_total = np.concatenate([input_steering_ro, input_steering_reelin])


# --- Total average power ---
total_energy = energy_reelout + energy_reelin
total_duration = t_total[-1] - t_total[0]
print("Total average power: ", total_energy / total_duration / 1000, "kW")


# --- 2D Plots over full cycle ---
def plot_quantity(x, y, xlabel, ylabel, title):
    plt.figure(figsize=(12, 5))
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    # plt.show()


plot_quantity(
    t_total, lt_total, "Time [s]", "Tether Length [m]", "Tether Length Over Full Cycle"
)
plot_quantity(
    t_total,
    tension_total / 1000,
    "Time [s]",
    "Tether Tension [kN]",
    "Tether Tension Over Full Cycle",
)
plot_quantity(
    t_total,
    vtau_total,
    "Time [s]",
    "Tangential Speed [m/s]",
    "Tangential Speed Over Full Cycle",
)
plot_quantity(
    t_total,
    input_steering_total,
    "Time [s]",
    "Steering Input [-]",
    "Steering Input Over Full Cycle",
)
plot_quantity(
    t_total,
    aoa_total * 180 / np.pi,
    "Time [s]",
    "Angle of Attack [deg]",
    "Angle of Attack Over Full Cycle",
)
plot_quantity(
    t_total,
    roll_total * 180 / np.pi,
    "Time [s]",
    "Roll Angle [deg]",
    "Roll Angle Over Full Cycle",
)
plot_quantity(
    t_total,
    depower_total,
    "Time [s]",
    "Depower Input [-]",
    "Depower Input Over Full Cycle",
)
plot_quantity(
    t_total, vr_total, "Time [s]", "Radial Speed [m/s]", "Radial Speed Over Full Cycle"
)
plot_quantity(
    t_total,
    power_total / 1000,
    "Time [s]",
    "Mechanical Power [kW]",
    "Mechanical Power Over Full Cycle",
)
# Plot CL and CD over full cycle
plot_quantity(
    t_total,
    cl_total,
    "Time [s]",
    "Lift Coefficient [-]",
    "Lift Coefficient Over Full Cycle",
)
plot_quantity(
    t_total,
    cd_total,
    "Time [s]",
    "Drag Coefficient [-]",
    "Drag Coefficient Over Full Cycle",
)


# --- 3D trajectory plot ---
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")
ax.plot(x_reelout, y_reelout, z_reelout, label="Reel-out", linewidth=2)
ax.plot(x_reelin, y_reelin, z_reelin, label="Reel-in", linewidth=2)
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_zlabel("Z [m]")
ax.legend()
ax.set_title("3D Trajectory: Reel-out and Reel-in")

# Equal aspect ratio
X = np.concatenate([x_reelout, x_reelin])
Y = np.concatenate([y_reelout, y_reelin])
Z = np.concatenate([z_reelout, z_reelin])
mid_x, mid_y, mid_z = (
    (X.max() + X.min()) / 2,
    (Y.max() + Y.min()) / 2,
    (Z.max() + Z.min()) / 2,
)
half_range = max(X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()) / 2
ax.set_xlim(mid_x - half_range, mid_x + half_range)
ax.set_ylim(mid_y - half_range, mid_y + half_range)
ax.set_zlim(mid_z - half_range, mid_z + half_range)

plt.tight_layout()
plt.show()
