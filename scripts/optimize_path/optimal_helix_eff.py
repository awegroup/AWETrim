import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from picawe.kinematics.parametrized_patterns import Helix, Lissajous, FigureEight
from picawe.system.kite import Kite
from picawe.kinematics.Kinematics import ParametrizedKinematics, KiteKinematics
from picawe import SystemModel
from picawe.utils.color_palette import set_plot_style, get_color_list
from picawe.timeseries.phase_parametrized import PhaseParameterized
import json
import copy




# Show the structure of the figure
plt.show()
# Define aerodynamic input
file_path = "./data/ap2_aero_input.json"
# file_path = "./data/rigid_kite.json"
with open(file_path, "r") as file:
    aero_input = json.load(file)


# -----------------------------------------------
# Define the parametrized path
# -----------------------------------------------
pattern_config = {
    "pattern_type": "helix",
    "initial_parameters": {
        "omega": -1.0,
        "r0": 200.0,
        "d0": 120.0,
        "vr": 0.2,
        "beta": 0.35,
        "kappa": 0
    },
    "optimization_parameters": {
        # Add any optimization-related parameters here if needed as list of names
        "d0",
        # "kappa",
        # "beta",
    }
}

start_state = {
    "t": 0,
    "s": -np.pi/2,
    "s_dot": 4,
    "s_ddot": 0,
    "tension_tether_ground": 1e5,
    "input_steering": 0,
    "angle_roll": 0,
    "angle_pitch": 0,
    "angle_yaw": 0,
}
time = np.arange(0, 50, 0.1)
s_array = np.linspace(5*np.pi/2, 9*np.pi/2, 200)
dof = 3
# -----------------------------------------------
# Define the system and aerodynamic model
# -----------------------------------------------
colors = get_color_list()
tension_tether_results = {}
phases = {}
parameters = ["speed_tangential", "tension_tether_ground", "angle_roll"]
x_param = "s"
fig, axs = plt.subplots(len(parameters),1, figsize=(10, 4), sharex=True)

N = 8
area_wing = 3
mass_wing = area_wing*2
phases_qs = []
phases_dyn = []
for i in range(N):

    pattern_config = {
        "pattern_type": "helix",
        "initial_parameters": {
            "omega": -1.0,
            "r0": 200.0,
            "d0": 60.0,
            "vr": 0.2,
            "beta": 0.35,
            "kappa": 0
        },
        "optimization_parameters": {
            # Add any optimization-related parameters here if needed as list of names
            "d0",
            # "kappa",
            # "beta",
        }
    }
    for quasi_steady in [True,False]:  # Loop over both dynamic and quasi-steady cases
       

        # aero_input["params"]["angle_pitch_depower_0"] = thetat
        # Define kite model with current parameters
        kite = Kite(mass_wing=mass_wing, area_wing=area_wing, aero_input=aero_input, mass_kcu=0, steering_control="roll")
        kite_model = SystemModel(dof=dof, quasi_steady=quasi_steady, kite=kite, wind_model="uniform")
        kite_model.speed_wind_ref = 15
        kite_model.input_depower = 0


        # Run simulation

        phase = PhaseParameterized(kite_model, quasi_steady=quasi_steady, pattern_config=pattern_config)
        phase.set_optimal_speed_radial()

        if quasi_steady:
            pattern_config = phase.optimize_pattern(start_state=start_state,  s_array=s_array)
            print(pattern_config)
            start_state = phase.states[0]

        else:
            phase.run_simulation(start_state=start_state, s_array=s_array)
        # phase.run_simulation(start_state=start_state, s_array=s_array)
        # Extract variables
        s = phase.return_variable("s")
        s_dot = phase.return_variable("s_dot")
        aoa = phase.return_variable("angle_of_attack")
        print("Mean aoa:",np.mean(aoa)*180/np.pi)
        if quasi_steady:
            phases_qs.append(copy.deepcopy(phase))
        else:
            phases_dyn.append(copy.deepcopy(phase))

        print(s_dot[0])
        start_state["s_dot"] = s_dot[0]
    aero_input["dependencies"]["alpha"]["k_cl"] -= 0.5
    aero_input["params"]["CD0"] += 0.01

# fig, slider = phase.interactive_plot()
# plt.show()
# -----------------------------------------------
# Plot results
# -----------------------------------------------
set_plot_style()
save_folder = "./results/figures/translational_paper/"
# -----------------------------------------------
# Load data and define aerodynamic model
# -----------------------------------------------
# Create figure with a custom grid layout
fig = plt.figure(figsize=(12, 6)) 

# Define grid layout (2 rows, 3 columns)
gs = fig.add_gridspec(8, 3, width_ratios=[1, 0.25,2], height_ratios=[1, 1,1,1,1,1,1,1])

# Left side subplots (square-like aspect ratio)
ax1 = fig.add_subplot(gs[:4, 0])  # Top-left
ax2 = fig.add_subplot(gs[4:8, 0])  # Bottom-left

# Right side subplots (time series)
ax3 = fig.add_subplot(gs[:2, 2])  # Top-right (spanning two columns)
ax4 = fig.add_subplot(gs[2:4, 2])  # Middle-right
ax5 = fig.add_subplot(gs[4:6, 2])  # Bottom-right
ax6 = fig.add_subplot(gs[6:8, 2])  # Bottom-right

from picawe.utils.defaults import PLOT_LABELS
# add labels
ax1.set_ylabel(PLOT_LABELS["angle_elevation"])
ax2.set_xlabel(PLOT_LABELS["angle_azimuth"])
ax2.set_ylabel(PLOT_LABELS["angle_elevation"])
ax5.set_xlabel(PLOT_LABELS["phase"])
ax3.set_ylabel(PLOT_LABELS["speed_tangential"])
ax4.set_ylabel(PLOT_LABELS["tension_tether_ground"])
ax5.set_ylabel(PLOT_LABELS["angle_roll"])
ax6.set_ylabel(PLOT_LABELS["speed_radial"])


# Adjust layout for better spacing
# mass_ratio_values = [2,4,40,10]
mean_pow_qs = []
mean_pow_dyn = []
min_pow_qs = []
min_pow_dyn = []
max_pow_qs = []
max_pow_dyn = []
mean_lift_qs = []
mean_lift_dyn = []
mean_drag_qs = []
mean_drag_dyn = []
for i in range(N):
    
    phase_qs = phases_qs[i]
    phase_dyn = phases_dyn[i]
    s_qs = phase_qs.return_variable("s")
    mask_qs = (s_qs > 5*np.pi/2) & (s_qs < 9*np.pi/2)
    s_dyn = phase_dyn.return_variable("s") 
    mask_dyn = (s_dyn > 5*np.pi/2) & (s_dyn < 9*np.pi/2)
    vtau_qs = phase_qs.return_variable("speed_tangential")[mask_qs]
    vtau_dyn = phase_dyn.return_variable("speed_tangential")[mask_dyn]
    vr_qs = phase_qs.return_variable("speed_radial")[mask_qs]
    vr_dyn = phase_dyn.return_variable("speed_radial")[mask_dyn]
    power_qs = phase_qs.return_variable("mechanical_power")[mask_qs]
    power_dyn = phase_dyn.return_variable("mechanical_power")[mask_dyn]
    tension_qs = phase_qs.return_variable("tension_tether_ground")[mask_qs]
    tension_dyn = phase_dyn.return_variable("tension_tether_ground")[mask_dyn]
    roll_qs = phase_qs.return_variable("angle_roll")[mask_qs]
    roll_dyn = phase_dyn.return_variable("angle_roll")[mask_dyn]
    azimuth_qs = phase_qs.return_variable("angle_azimuth")[mask_qs]
    azimuth_dyn = phase_dyn.return_variable("angle_azimuth")[mask_dyn]
    elevation_qs = phase_qs.return_variable("angle_elevation")[mask_qs]
    elevation_dyn = phase_dyn.return_variable("angle_elevation")[mask_dyn]
    aoa_qs = phase_qs.return_variable("angle_of_attack")[mask_qs]
    aoa_dyn = phase_dyn.return_variable("angle_of_attack")[mask_dyn]
    idx_vmax_qs = np.argmax(vtau_qs)
    print(idx_vmax_qs)
    idx_vmax_dyn = np.argmax(vtau_dyn)
    idx_vmin_qs = np.argmin(vtau_qs)
    idx_vmin_dyn = np.argmin(vtau_dyn)

    # Calculate the difference in power in percentage
    diff_power = (np.mean(power_qs) - np.mean(power_dyn)) / np.mean(power_dyn) * 100
    diff_vtau = (np.max(vtau_qs) - np.max(vtau_dyn)) / np.max(vtau_dyn) * 100
    diff_tension = (np.mean(tension_qs) - np.mean(tension_dyn)) / np.mean(tension_dyn) * 100
    diff_max_roll = np.degrees(max(roll_qs) - max(roll_dyn))
    diff_s_vmax = np.degrees(-s_dyn[idx_vmax_dyn] + s_qs[idx_vmax_qs])
    diff_s_vmin = np.degrees(-s_dyn[idx_vmin_dyn] + s_qs[idx_vmin_qs])


    print(f"Tangent speed difference: {diff_vtau.max():.2f}%")
    print(f"Power difference: {diff_power.max():.2f}%")
    print(f"Tension difference: {diff_tension.max():.2f}%")
    print(f"Max roll difference: {diff_max_roll:.2f} degrees")
    print(f"Max speed phase difference: {diff_s_vmax:.2f} degrees")
    print(f"Min speed phase difference: {diff_s_vmin:.2f} degrees")
    s_dyn = s_dyn[mask_dyn]-5*np.pi/2
    s_qs = s_qs[mask_qs]-5*np.pi/2
    if i < len(colors):
        ax3.plot(np.degrees(s_dyn), vtau_dyn, color=colors[i])
        ax3.plot(np.degrees(s_qs), vtau_qs, linestyle="--", color=colors[i])
        ax4.plot(np.degrees(s_dyn), tension_dyn/1000, color=colors[i])
        ax4.plot(np.degrees(s_qs), tension_qs/1000, linestyle="--", color=colors[i])
        ax5.plot(np.degrees(s_dyn), np.degrees(roll_dyn), color=colors[i])
        ax5.plot(np.degrees(s_qs), np.degrees(roll_qs), linestyle="--", color=colors[i])
        ax6.plot(np.degrees(s_dyn), vr_dyn, color=colors[i])
        ax6.plot(np.degrees(s_qs), vr_qs, linestyle="--", color=colors[i])
    
    mean_pow_qs.append(np.mean(tension_qs*vr_qs))
    mean_pow_dyn.append(np.mean(tension_dyn*vr_dyn))
    print("mean tension qs", np.mean(tension_qs))
    print("mean tension dyn", np.mean(tension_dyn))
    print("mean power qs", np.mean(power_qs))
    print("mean power dyn", np.mean(power_dyn))
    min_pow_qs.append(np.min(tension_qs*vr_qs))
    min_pow_dyn.append(np.min(tension_dyn*vr_dyn))
    max_pow_qs.append(np.max(tension_qs*vr_qs))
    max_pow_dyn.append(np.max(tension_dyn*vr_dyn))

    lift_qs = phase_qs.return_variable("lift_coefficient")[mask_qs]
    lift_dyn = phase_dyn.return_variable("lift_coefficient")[mask_dyn]
    drag_qs = phase_qs.return_variable("drag_coefficient")[mask_qs]
    drag_dyn = phase_dyn.return_variable("drag_coefficient")[mask_dyn]
    mean_lift_qs.append(np.mean(lift_qs))
    mean_lift_dyn.append(np.mean(lift_dyn))
    mean_drag_qs.append(np.mean(drag_qs))
    mean_drag_dyn.append(np.mean(drag_dyn))
    print("Mean aoa qs", np.mean(aoa_qs)*180/np.pi)
    print("Mean aoa dyn", np.mean(aoa_dyn)*180/np.pi)
    print("Mean lift qs", np.mean(lift_qs))
    print("Mean lift dyn", np.mean(lift_dyn))
    print("Mean drag qs", np.mean(drag_qs))
    print("Mean drag dyn", np.mean(drag_dyn))

ax1.set_xticklabels([]) # Remove x-ticks   
ax3.set_xticklabels([]) # Remove x-ticks
ax4.set_xticklabels([]) # Remove x-ticks
ax5.set_xticklabels([]) # Remove x-ticks
# Set xlim for all subplots
ax3.set_xlim([0, 360])
ax4.set_xlim([0, 360])
ax5.set_xlim([0, 360])
ax6.set_xlim([0, 360])
max_el = max(np.max(elevation_qs), np.max(elevation_dyn))*180/np.pi
ax1.set_ylim([0, max_el+10])
ax2.set_ylim([0, max_el+10])

vmin = min(np.min(vtau_qs), np.min(vtau_dyn))
vmax = max(np.max(vtau_qs), np.max(vtau_dyn))
scatter = ax1.scatter(
    azimuth_dyn*180/np.pi, elevation_dyn*180/np.pi, c=vtau_dyn, cmap="viridis", s=10,
    vmin=vmin, vmax=vmax
)  # `s` adjusts marker size
cbar_ax = fig.add_axes([0.35, 0.3, 0.02, 0.4])  # Manually positioned colorbar
cbar = fig.colorbar(scatter, cax=cbar_ax)
cbar.set_label(PLOT_LABELS["speed_tangential"])
cbar.set_ticks(np.linspace(vmin, vmax, num=5))


scatter = ax2.scatter(
    azimuth_qs*180/np.pi, elevation_qs*180/np.pi, c=vtau_qs, cmap="viridis", s=10,
    vmin=vmin, vmax=vmax
)  # `s` adjusts marker size

# plt.tight_layout()
# Save figure as pdf
plt.savefig(save_folder+"parametrized_circle_results.pdf", bbox_inches='tight')
# plt.show()

mean_LD_dyn = np.array(mean_lift_dyn)/np.array(mean_drag_dyn)
mean_LD_qs = np.array(mean_lift_qs)/np.array(mean_drag_qs)
plt.figure()
plt.plot(mean_LD_dyn,mean_pow_dyn, label="Dynamic", color = colors[0])
plt.plot(mean_LD_qs,mean_pow_qs, label="Quasi-steady",color = colors[1])
plt.fill_between(mean_LD_dyn, min_pow_dyn, max_pow_dyn, alpha=0.2, color = colors[0])
plt.fill_between(mean_LD_qs, min_pow_qs, max_pow_qs, alpha=0.2, color = colors[1])
plt.xlabel("L/D")
plt.ylabel("Mean tension [kN]")
plt.legend()
plt.show()