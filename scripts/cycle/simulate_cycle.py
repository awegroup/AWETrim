import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from picawe.kinematics.parametrized_patterns import Helix
from picawe import SystemModel, State
from picawe.utils.color_palette import set_plot_style, get_color_list
from picawe.timeseries.phase_parametrized import PhaseParameterized
from picawe.timeseries.reelin_phase import ReelinPhase
from picawe.system.kite import Kite
from picawe.system.tether import FlexibleLumpedTether, RigidLumpedTether
from picawe.utils.defaults import PLOT_LABELS
import time

# -------------------- Constants --------------------
GRAVITY = 9.81

# -------------------- File Input --------------------
file_path = "./data/v9_aero_input.json"
with open(file_path, "r") as file:
    aero_input = json.load(file)

# -------------------- Configuration --------------------
SIMULATION_CONFIG = {
    "mass_ratio": 2,
    "dof": 3,
    "area_wing": 47,
    "mass_wing": 78,
    "quasi_steady": True,
    "wind_model": "logarithmic",
    "speed_friction": 0.35,
    "z0": 0.01,
}

wind_speed_at_200 = SIMULATION_CONFIG["speed_friction"] / 0.4 * np.log(200 / SIMULATION_CONFIG["z0"])
print("Wind speed at 200m: ", wind_speed_at_200)

PATTERN_CONFIG = {
    "pattern_type": "figure_eight",
    "parameters": {
        "omega": -1.0,
        "r0": 230.0,
        "ry": 120,
        "rz": 100,
        "ky": 1,
        "kz": 1,
        "vr": 1,
        "beta0": 0.55,
        "kappa": 1,
    },
    "control": {
        "input_depower": 0.0,
    },
    "start_path_angle": -np.pi/2,
    "end_path_angle": 3*np.pi/2 + np.pi,
    "n_points": 300,
}

CYCLE_SETTINGS = {
    "reelout": PATTERN_CONFIG,
    "reelin": {
        "phase_model": ReelinPhase,
        "control": {
            "max_elevation": np.degrees(85),
            "reeling_speed": -5,
            "min_tether_force": SIMULATION_CONFIG["mass_wing"] * GRAVITY,
            "length_tether_ro": PATTERN_CONFIG["parameters"]["r0"],
        },
        "time_step": 0.1
    }
}

# -------------------- Helper Functions --------------------
def create_model(config, kite, tether):
    model = SystemModel(
        dof=config["dof"],
        quasi_steady=config["quasi_steady"],
        kite=kite,
        wind_model=config["wind_model"],
        tether=tether
    )
    model.wind.speed_friction = config["speed_friction"]
    return model

# -------------------- Reelout Phase --------------------

tether = RigidLumpedTether()
kite = Kite(mass_wing=SIMULATION_CONFIG["mass_wing"], area_wing=SIMULATION_CONFIG["area_wing"],
            aero_input=aero_input, steering_control="roll")
model = create_model(SIMULATION_CONFIG, kite, tether)

phase = PhaseParameterized(model, quasi_steady=SIMULATION_CONFIG["quasi_steady"], pattern_config=PATTERN_CONFIG)
t0 = time.time()
phase.run_simulation()

print("Reelout time: ", time.time() - t0, " seconds")

# -------------------- Reelin Phase --------------------

tether = RigidLumpedTether()
kite = Kite(mass_wing=SIMULATION_CONFIG["mass_wing"], area_wing=SIMULATION_CONFIG["area_wing"],
            aero_input=aero_input, steering_control="roll")
model = create_model(SIMULATION_CONFIG, kite, tether)

phase_reelin = ReelinPhase(model, quasi_steady=SIMULATION_CONFIG["quasi_steady"])

start_state_ri = State(
    t=phase.return_variable("t")[-1],
    distance_radial=phase.return_variable("distance_radial")[-1],
    angle_elevation=phase.return_variable("angle_elevation")[-1],
    angle_azimuth=phase.return_variable("angle_azimuth")[-1],
    angle_course=0,
    input_steering=0,
    input_depower=1,
    speed_tangential=40,
    timeder_angle_course=0,
    speed_radial=phase.return_variable("speed_radial")[-1],
    tension_tether_ground=1e4,
)

CYCLE_SETTINGS["reelin"]["control"]["riro_elevation"] = phase.return_variable("angle_elevation")[0]
t0 = time.time()
phase_reelin.run_simulation(
    start_state=start_state_ri,
    settings=CYCLE_SETTINGS["reelin"]
)

print("Reelin time: ", time.time() - t0, " seconds")



# -------------------- Plottting --------------------
fig = plt.figure(figsize=(12, 6))
gs = fig.add_gridspec(8, 3, width_ratios=[1, 0.25, 2], height_ratios=[1, 1, 1, 1, 1, 1, 1, 1])

ax1 = fig.add_subplot(gs[:4, 0])
ax2 = fig.add_subplot(gs[4:, 0])
ax3 = fig.add_subplot(gs[:2, 2])
ax4 = fig.add_subplot(gs[2:4, 2])
ax5 = fig.add_subplot(gs[4:6, 2])
ax6 = fig.add_subplot(gs[6:, 2])



t_ro = phase.return_variable("t")
lt_ro = phase.return_variable("distance_radial")
s_ro = phase.return_variable("s")
vtau_ro = phase.return_variable("speed_tangential")
azimuth_ro = phase.return_variable("angle_azimuth")
elevation_ro = phase.return_variable("angle_elevation")
roll_ro = phase.return_variable("angle_roll")
aoa_ro = phase.return_variable("angle_of_attack")
tension_ro = phase.return_variable("tension_tether_ground")


print(phase_reelin.states[-1])
lt_reelin = phase_reelin.return_variable("distance_radial")
t_reelin = phase_reelin.return_variable("t")
x_reelin = phase_reelin.return_variable("x")
z_reelin = phase_reelin.return_variable("z")
x_reelout = phase.return_variable("x")
z_reelout = phase.return_variable("z")
y_reelin = phase_reelin.return_variable("y")
y_reelout = phase.return_variable("y")

pow_reelout = phase.return_variable("mechanical_power")
pow_reelin = phase_reelin.return_variable("mechanical_power")

dt = np.diff(t_ro, prepend=t_ro[0])
energy_reelout = np.sum(pow_reelout*dt)
print("Energy reelout: ", energy_reelout)

dt_reelin = np.diff(t_reelin, prepend=t_reelin[0])
energy_reelin = np.sum(pow_reelin*dt_reelin)
print("Energy reelin: ", energy_reelin)


#Join the two phases

total_t = np.concatenate((t_ro, t_reelin))
total_lt = np.concatenate((lt_ro, lt_reelin))

colors = get_color_list()
save_folder = "./results/figures/translational_paper/"

print("Total power: ", (energy_reelout + energy_reelin) / (total_t[-1] - total_t[0]))

print(np.sum(phase.return_variable("tension_tether_ground")) / 1000)

ax3.plot(s_ro*180/np.pi, vtau_ro)
ax4.plot(s_ro*180/np.pi, tension_ro)
ax5.plot(s_ro*180/np.pi, roll_ro)
ax6.plot(s_ro*180/np.pi, aoa_ro)

# ax = ax2 if quasi_steady else ax1
scatter = ax2.scatter(np.degrees(azimuth_ro), np.degrees(elevation_ro), c=vtau_ro, cmap="viridis", s=10)

cbar_ax = fig.add_axes([0.35, 0.3, 0.02, 0.4])
cbar = fig.colorbar(scatter, cax=cbar_ax)
cbar.set_label(PLOT_LABELS["speed_tangential"])

ax1.set_ylabel(PLOT_LABELS["angle_elevation"])
ax2.set_xlabel(PLOT_LABELS["angle_azimuth"])
ax2.set_ylabel(PLOT_LABELS["angle_elevation"])
ax3.set_ylabel(PLOT_LABELS["speed_tangential"])
ax4.set_ylabel("Tension [kN]")
ax5.set_ylabel(PLOT_LABELS["angle_roll"])
ax6.set_ylabel(PLOT_LABELS["angle_of_attack"])
ax6.set_xlabel(PLOT_LABELS["phase"])


ax1.set_ylim(0, 60)
ax1.set_xlim(-30,30)
ax2.set_ylim(0, 60)
ax2.set_xlim(-30,30)

ax3.legend()
set_plot_style()
plt.tight_layout()
plt.savefig(save_folder + "parametrized_circle_results_combined.pdf", bbox_inches='tight')
plt.show()

# plt.figure(figsize=(12, 6))
# aoa_reelin = phase_reelin.return_variable("angle_of_attack")
# print("Mean CL reelin: ", np.mean(phase_reelin.return_variable("lift_coefficient")))
# print("Mean CD reelin: ", np.mean(phase_reelin.return_variable("drag_coefficient")))
# print("Mean AoA reelin: ", np.mean(aoa_reelin))

# # plt.plot(t,phase.return_variable("angle_of_attack")*180/np.pi)
# plt.plot(t_reelin,aoa_reelin*180/np.pi)
# plt.xlabel("Time [s]")
# plt.ylabel("Angle of attack [deg]")
# plt.show()

plt.figure(figsize=(12, 6))
plt.plot(t_reelin, phase_reelin.return_variable("tension_tether_ground") / 1000)
plt.xlabel("Time [s]")
plt.ylabel("Tether tension [kN]")
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(total_t,total_lt)
plt.xlabel("Time [s]")
plt.ylabel("Tether length [m]")
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(t_reelin, phase_reelin.return_variable("speed_radial"))
plt.xlabel("s [m]")
plt.ylabel("Vr [deg]")
plt.show()

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_reelin, y_reelin, z_reelin, label='Reelin')
ax.plot(x_reelout, y_reelout, z_reelout, label='Reelout')
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
ax.legend()

# Set equal aspect ratio for all axes
max_range = np.array([
    x_reelin.max(), x_reelout.max(), y_reelin.max(), y_reelout.max(), z_reelin.max(), z_reelout.max()
]).max() - np.array([
    x_reelin.min(), x_reelout.min(), y_reelin.min(), y_reelout.min(), z_reelin.min(), z_reelout.min()
]).min()
X = np.concatenate([x_reelin, x_reelout])
Y = np.concatenate([y_reelin, y_reelout])
Z = np.concatenate([z_reelin, z_reelout])
mid_x = (X.max() + X.min()) * 0.5
mid_y = (Y.max() + Y.min()) * 0.5
mid_z = (Z.max() + Z.min()) * 0.5
half_range = 0.5 * max([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()])
ax.set_xlim(mid_x - half_range, mid_x + half_range)
ax.set_ylim(mid_y - half_range, mid_y + half_range)
ax.set_zlim(mid_z - half_range, mid_z + half_range)

plt.show()