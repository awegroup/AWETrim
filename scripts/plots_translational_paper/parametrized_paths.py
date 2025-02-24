import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from picawe.kinematics.parametrized_patterns import Helix, Lissajous, FigureEight
from picawe.kinematics.Kinematics import ParametrizedKinematics, KiteKinematics
from picawe import SystemModel
from picawe.utils.color_palette import set_plot_style, get_color_list
from picawe.timeseries.phase_parametrized import PhaseParameterized
import json
from picawe.system.kite import Kite




# Show the structure of the figure
plt.show()
# Define aerodynamic input
file_path = "./data/v3_aero_input.json"
# file_path = "./data/rigid_kite.json"
with open(file_path, "r") as file:
    aero_input = json.load(file)



# -----------------------------------------------
# Define the parametrized path
# -----------------------------------------------
start_state = {
    "t": 0,
    "s": -np.pi/2,
    "s_dot": 4,
    "s_ddot": 0,
    "tension_tether_ground": 1e3,
    "input_steering": 0,
    "angle_roll": 0,
    "angle_pitch": 0,
    "angle_yaw": 0,
}
time = np.arange(0, 30, 0.1)
dof = 3
# -----------------------------------------------
# Define the system and aerodynamic model
# -----------------------------------------------
colors = get_color_list()
tension_tether_results = {}
phases = {}
parameters = ["speed_tangential", "tension_tether_ground", "input_steering"]
x_param = "s"
fig, axs = plt.subplots(len(parameters),1, figsize=(10, 4), sharex=True)
mass_ratio_values = [2]
area_wing = 20
for i,mr in enumerate(mass_ratio_values):
    for quasi_steady in [True,False]:  # Loop over both dynamic and quasi-steady cases
        if quasi_steady:
            linestyle = "--"
            label = None
        else:
            linestyle = "-"
            label = r"$\frac{m}{S}=$"+str(mr)
        mass_wing = mr * area_wing
        # Define kite model with current parameters
        kite = Kite(mass_wing=mass_wing, area_wing=area_wing, aero_input=aero_input, steering_control="roll")
        kite_model = SystemModel(dof=dof, quasi_steady=quasi_steady, kite=kite, wind_model="uniform") 
        kite_model.wind.speed_wind_ref = 12
        kite_model.input_depower = 0
        
        # Run simulation
        if mr == 0 and not quasi_steady:
            pass
        else:
            phase = PhaseParameterized(kite_model, quasi_steady=quasi_steady)
            phase.target_lift_coefficient = 0.6
            phase.target_drag_coefficient = 0.15
            phase.set_optimal_speed_radial()
        phase.run_simulation(start_state=start_state, time_array=time)
        # Extract variables
        s = phase.return_variable("s")
        s_dot = phase.return_variable("s_dot")


        phases[(mr, quasi_steady)] = phase
        print(s_dot[0])
        start_state["s_dot"] = s_dot[0]




for ax in axs:
    ax.set_xlim([5*np.pi/2, 9*np.pi/2])

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
ax6.set_ylabel(PLOT_LABELS["angle_of_attack"])


# Adjust layout for better spacing
# mass_ratio_values =np.arange(2,50,6) #[2,4,40,50]
mean_ft_qs = []
mean_ft_dyn = []
min_ft_qs = []
min_ft_dyn = []
max_ft_qs = []
max_ft_dyn = []
for i, mr in enumerate(mass_ratio_values):
    
    phase_qs = phases[(mr, True)]
    phase_dyn = phases[(mr, False)]
    s_qs = phase_qs.return_variable("s")
    mask_qs = (s_qs > 5*np.pi/2) & (s_qs < 9*np.pi/2)
    s_dyn = phase_dyn.return_variable("s") 
    mask_dyn = (s_dyn > 5*np.pi/2) & (s_dyn < 9*np.pi/2)
    vtau_qs = phase_qs.return_variable("speed_tangential")[mask_qs]
    vtau_dyn = phase_dyn.return_variable("speed_tangential")[mask_dyn]
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

    print(f"Mass ratio: {mr}")
    print(f"Tangent speed difference: {diff_vtau.max():.2f}%")
    print(f"Power difference: {diff_power.max():.2f}%")
    print(f"Tension difference: {diff_tension.max():.2f}%")
    print(f"Max roll difference: {diff_max_roll:.2f} degrees")
    print(f"Max speed phase difference: {diff_s_vmax:.2f} degrees")
    print(f"Min speed phase difference: {diff_s_vmin:.2f} degrees")
    s_dyn = s_dyn[mask_dyn]-5*np.pi/2
    s_qs = s_qs[mask_qs]-5*np.pi/2
    ax3.plot(np.degrees(s_dyn), vtau_dyn, label=f"$\frac{{m}}{{S}} = {mr}$", color=colors[i])
    ax3.plot(np.degrees(s_qs), vtau_qs, linestyle="--", color=colors[i])
    ax4.plot(np.degrees(s_dyn), tension_dyn/1000, label=f"$\frac{{m}}{{S}} = {mr}$", color=colors[i])
    ax4.plot(np.degrees(s_qs), tension_qs/1000, linestyle="--", color=colors[i])
    ax5.plot(np.degrees(s_dyn), np.degrees(roll_dyn), label=f"$\frac{{m}}{{S}} = {mr}$", color=colors[i])
    ax5.plot(np.degrees(s_qs), np.degrees(roll_qs), linestyle="--", color=colors[i])
    ax6.plot(np.degrees(s_dyn), np.degrees(aoa_dyn), label=f"$\frac{{m}}{{S}} = {mr}$", color=colors[i])
    ax6.plot(np.degrees(s_qs), np.degrees(aoa_qs), linestyle="--", color=colors[i])
    
    mean_ft_qs.append(np.mean(tension_qs))
    mean_ft_dyn.append(np.mean(tension_dyn))
    print("mean tension qs", np.mean(tension_qs))
    print("mean tension dyn", np.mean(tension_dyn))
    min_ft_qs.append(np.min(tension_qs))
    min_ft_dyn.append(np.min(tension_dyn))
    max_ft_qs.append(np.max(tension_qs))
    max_ft_dyn.append(np.max(tension_dyn))

    print("Mean aoa qs", np.mean(aoa_qs)*180/np.pi)
    print("Mean aoa dyn", np.mean(aoa_dyn)*180/np.pi)

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
ax1.set_ylim([0, max_el])
ax2.set_ylim([0, max_el])

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

plt.figure()
plt.plot(mass_ratio_values,mean_ft_dyn, label="Dynamic", color = colors[0])
plt.plot(mass_ratio_values,mean_ft_qs, label="Quasi-steady",color = colors[1])
plt.fill_between(mass_ratio_values, min_ft_dyn, max_ft_dyn, alpha=0.2, color = colors[0])
plt.fill_between(mass_ratio_values, min_ft_qs, max_ft_qs, alpha=0.2, color = colors[1])
plt.xlabel("Mass ratio")
plt.ylabel("Mean tension [kN]")
plt.legend()
plt.show()