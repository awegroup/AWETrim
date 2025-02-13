import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from picawe.kinematics.parametrized_patterns import Helix, Lissajous, FigureEight
from picawe.kinematics.Kinematics import ParametrizedKinematics, KiteKinematics
from picawe import SystemModel
from picawe.utils.color_palette import set_plot_style, get_color_list
from picawe.timeseries.phase import PhaseParameterized
import json




# Show the structure of the figure
plt.show()
# Define aerodynamic input
file_path = "./data/v3_aero_input.json"
# file_path = "./data/rigid_kite.json"
with open(file_path, "r") as file:
    aero_input = json.load(file)

aero_input =    {
        "model": "inviscid",
        "params": {
            "CD0": 0.05,
            "aspect_ratio": 20,
            "oswald_efficiency": 1,
            "angle_pitch_depower_0": np.radians(5),
        },
       "dependencies": {
        # "u_s": { "k_cl": 0, "k_cd": 0.0, "k_cs": 0.23, "k_cn": 0.005 },
    } 

    }
kite_model = SystemModel(mass_wing=80, area_wing=20, aero_input=aero_input, mass_kcu=0, dof=3, quasi_steady=True, steering_control="roll")
import casadi as ca
kite_model.angle_elevation = 0
kite_model.angle_azimuth = 0
kite_model.angle_course = np.pi/2
kite_model.timeder_speed_tangential = 0
kite_model.distance_radial = 200
kite_model.speed_wind = 10
kite_model.angle_roll = 0
kite_model.speed_tangential = 40
kite_model.delta_pitch_depower = 0
kite_model.input_depower = 0

cl_func = kite_model.extract_function("lift_coefficient")
cd_func = kite_model.extract_function("drag_coefficient")
aoa_func = kite_model.extract_function("angle_of_attack")
vr = np.linspace(-10,10,100)
fig, axs = plt.subplots(2, 1, figsize=(4, 8), sharex=True)
axs[0].plot(aoa_func(vr)*180/np.pi, cl_func(vr)/cd_func(vr))
axs[0].set_xlabel("Angle of attack [deg]")
axs[0].set_ylabel("Lift-to-drag ratio")
axs[1].plot(aoa_func(vr)*180/np.pi, cl_func(vr))
print(f"Max CL = {np.max(cl_func(vr)):.2f}")
maxLD = np.max(cl_func(vr)/cd_func(vr))
max_aoaLD = float(aoa_func(vr)[np.argmax(cl_func(vr)/cd_func(vr))] * 180/np.pi)
print(f"Max L/D = {maxLD:.2f} at {max_aoaLD:.2f} degrees")
plt.grid()
# plt.show()
print(cl_func)
# -----------------------------------------------
# Define the parametrized path
# -----------------------------------------------
omega = -1
x0 = 200
rh = 80
vr = 0
beta = np.radians(30)
ry = 100
rz = 40
helix = Helix(omega, x0, rh, vr, beta, kappa=0.5)
lissajous = Lissajous(omega, x0, ry, rz, vr, beta)
figure_eight = FigureEight(omega, x0, ry, rz*2, vr, beta, ky=0.5, kz=.5)


t = np.linspace(0, 2*np.pi, 1000)
s = np.linspace(0, 2*np.pi, 1000)
# plt.plot(lissajous.yd(t,s), lissajous.zd(t,s))

# plt.plot(figure_eight.yd(t,s), figure_eight.zd(t,s))
# # figure_eight = FigureEight(omega, x0, ry, rz*2, vr, beta, ky=2, kz=2)
# plt.plot(figure_eight.yd(t,s), figure_eight.zd(t,s))
# # figure_eight = FigureEight(omega, x0, ry, rz*2, vr, beta, ky=0.01, kz=0.01)
# plt.plot(figure_eight.yd(t,s), figure_eight.zd(t,s))
# plt.show()
pattern = helix

start_state = {
    "t": 0,
    "s": -np.pi/2,
    "s_dot": 0.489,
    "s_ddot": 0,
    "tension_tether_ground": 1e3,
    "input_steering": 0,
    "angle_roll": 0,
    "angle_pitch": 0,
    "angle_yaw": 0,
}
time = np.arange(0, 30, 0.01)
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
mass_ratio_values = [40]
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
        kite_model = SystemModel(mass_wing=mass_wing, area_wing=area_wing, aero_input=aero_input, 
                                 mass_kcu=0, dof=dof, quasi_steady=quasi_steady, 
                                 steering_control="roll")
        kite_model.speed_wind = 15
        kite_model.input_depower = 0
        
        # Run simulation
        phase = PhaseParameterized(kite_model, pattern, quasi_steady=quasi_steady)
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


# Adjust layout for better spacing
mass_ratio_values = [40]
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


ax1.set_xticklabels([]) # Remove x-ticks   
ax3.set_xticklabels([]) # Remove x-ticks
ax4.set_xticklabels([]) # Remove x-ticks
ax5.set_xticklabels([]) # Remove x-ticks
# Set xlim for all subplots
ax3.set_xlim([0, 360])
ax4.set_xlim([0, 360])
ax5.set_xlim([0, 360])
ax6.set_xlim([0, 360])
ax1.set_ylim([0, 40])
ax2.set_ylim([0, 40])

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

cl_func = kite_model.extract_function("lift_coefficient")
cd_func = kite_model.extract_function("drag_coefficient")
print(cl_func)