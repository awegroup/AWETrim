import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from picawe.kinematics.parametrized_patterns import Helix, Lissajous, FigureEight
from picawe.system.kite import Kite
from picawe.system.tether import RigidLinkTether, RigidLumpedTether
from picawe.kinematics.Kinematics import ParametrizedKinematics, KiteKinematics
from picawe import SystemModel, State
from picawe.utils.color_palette import set_plot_style_no_latex, get_color_list
from picawe.timeseries.phase_parametrized import PhaseParameterized
from picawe.environment import Wind
import json
import copy

T0 = 3000  # N

b = 2500  # N/(m/s)^2
eps = 1e-6  # to avoid sqrt(0)
vr = np.linspace(0, 10, 1000)
T_model = T0 + b * vr**2
beta = 1e-4  # steepness of the softplus transition
T_max = 25000
# T = min(T_model, T_max) via softplus
softplus = (1 / beta) * np.log(1 + np.exp(beta * (T_model - T_max)))
plt.plot(vr, T_model - softplus, label="Softplus approximation")
plt.show()

file_path = "./data/LEI-V9-KITE/v9_aero_input.json"
with open(file_path, "r") as file:
    aero_input = json.load(file)

speed_wind_at_100 = 10  # m/s

wind = Wind(
    wind_model="uniform",  # logarithmic
    z0=0.1,  # roughness length
)
speed_friction = 0.41 * speed_wind_at_100 / np.log(100 / wind.z0)
wind.speed_friction = speed_friction
wind.speed_wind_ref = speed_wind_at_100

print("Friction speed:", wind.speed_friction)

# -----------------------------------------------
# Define the parametrized path
# -----------------------------------------------

start_state = State(
    t=0,
    s=np.pi / 2,
    s_dot=0.8,
    s_ddot=0,
    length_tether=199.6,
    input_steering=0,
    angle_roll=0,
    angle_pitch=0,
    angle_yaw=0,
    tension_tether_ground=1e8,
    speed_radial=2,
    distance_radial=230,
)
time = np.arange(0, 50, 0.1)
s_array = np.linspace(5 * np.pi / 2, 9 * np.pi / 2, 200)
dof = 3
# -----------------------------------------------
# Define the system and aerodynamic model
# -----------------------------------------------
colors = get_color_list()
tension_tether_results = {}
phases = {}
parameters = ["speed_tangential", "tension_tether_ground", "angle_roll"]
x_param = "s"
fig, axs = plt.subplots(len(parameters), 1, figsize=(10, 4), sharex=True)

N = 1
tether = RigidLumpedTether(diameter=0.01)
mass_wing = 90
area_wing = 47
tension_min = 3000
tension_max = 25000
phases_qs = []
phases_dyn = []
for i in range(N):

    pattern_config = {
        "pattern_type": "cst_lissajous",
        "parameters": {
            "omega": 1.0,
            "r0": 200.0,
            "az_amp0": 0.6191628957199365,
            "beta_amp0": 0.17039404884183,
            "width_phi": 0.5,
            "width_beta": 0.5,
            "left_first": True,
            "normalize_bumps": False,
            "repeat_phi": True,
            "repeat_beta": True,
            "beta_coeffs": np.array([0, 0, 0, 0, 0]),
            "az_coeffs": [0, 0, 0, 0, 0],
            "kbeta": 0,
            "beta0": 0.3844981096025447,
            "kappa": 0,
            "k_vr": 2716,
        },
        "start_time": 0,
        "end_time": 60,
        "start_angle": np.pi / 2,
        "end_angle": 2 * np.pi + np.pi / 2,
        "n_points": 600,
        # "k_vr": 9000,
        "optimization_parameters": [
            "az_amp0",
            "beta_amp0",
            "beta0",
            "beta_coeffs",
            # "kappa",
            "k_vr",
        ],
    }

    # TODO: PLOT PATTERN BEFORE SIMULATING IT
    # TODO: Investigate why beta0 does not go towards where it should
    for quasi_steady in [True, False]:  # Loop over both dynamic and quasi-steady cases

        # aero_input["params"]["angle_pitch_depower_0"] = thetat
        # Define kite model with current parameters
        kite = Kite(
            mass_wing=60,
            area_wing=area_wing,
            aero_input=aero_input,
            mass_kcu=30,
            steering_control="asymmetric",
        )

        kite_model = SystemModel(
            dof=dof,
            quasi_steady=quasi_steady,
            kite=kite,
            tether=tether,
            wind_model=wind,
        )
        kite_model.input_depower = 0

        # Run simulation

        phase = PhaseParameterized(
            kite_model,
            quasi_steady=quasi_steady,
            pattern_config=pattern_config,
            tension_min=tension_min,
            tension_max=tension_max,
        )
        # phase.set_optimal_speed_radial()

        if quasi_steady:
            phase.run_simulation_opti_phase(start_state=start_state)
            pattern_config = phase.pattern_config
            print("Optimized pattern configuration:")
            print(pattern_config)
            # start_state = phase.states[0]

        # kite = Kite(
        #     mass_wing=mass_wing,
        #     area_wing=area_wing,
        #     aero_input=aero_input,
        #     mass_kcu=0,
        #     steering_control="asymmetric",
        # )
        # kite_model = SystemModel(
        #     dof=dof,
        #     quasi_steady=quasi_steady,
        #     kite=kite,
        #     tether=tether,
        #     wind_model=wind,
        # )
        # kite_model.input_depower = 0
        # phase.kite_model = kite_model

        # phase = PhaseParameterized(
        #     kite_model, quasi_steady=quasi_steady, pattern_config=pattern_config
        # )
        phase.run_simulation_phase(start_state=start_state)
        start_state = phase.states[0]

        # start_state = phase.states[0]

        # TODO: One should not run the simulation twice, but rather use the optimized pattern, but somehow there is a problem using the optimized pattern directly
        # phase.run_simulation(start_state=start_state, s_array=s_array)
        # Extract variables
        s = phase.return_variable("s")
        s_dot = phase.return_variable("s_dot")
        # aoa = phase.return_variable("angle_of_attack")
        # print("Mean aoa:", np.mean(aoa) * 180 / np.pi)
        if quasi_steady:
            phases_qs.append(copy.deepcopy(phase))
        else:
            phases_dyn.append(copy.deepcopy(phase))

        print(s_dot[0])
        start_state["s_dot"] = s_dot[0]
        start_state["s"] = s[0]
        start_state["tension_tether_ground"] = phase.return_variable(
            "tension_tether_ground"
        )[0]


# fig, slider = phase.interactive_plot()
# plt.show()
# -----------------------------------------------
# Plot results
# -----------------------------------------------
set_plot_style_no_latex()
save_folder = "./results/figures/translational_paper/"
# -----------------------------------------------
# Load data and define aerodynamic model
# -----------------------------------------------
# Create figure with a custom grid layout
fig = plt.figure(figsize=(12, 6))

# Define grid layout (2 rows, 3 columns)
gs = fig.add_gridspec(
    8, 3, width_ratios=[1, 0.25, 2], height_ratios=[1, 1, 1, 1, 1, 1, 1, 1]
)

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
ax5.set_ylabel(PLOT_LABELS["input_steering"])
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
    s_dyn = phase_dyn.return_variable("s")
    s_qs = phase_qs.return_variable("s")
    t_qs = phase_qs.return_variable("t")
    t_dyn = phase_dyn.return_variable("t")
    vtau_qs = phase_qs.return_variable("speed_tangential")
    vtau_dyn = phase_dyn.return_variable("speed_tangential")
    vr_qs = phase_qs.return_variable("speed_radial")
    vr_dyn = phase_dyn.return_variable("speed_radial")
    power_qs = phase_qs.return_variable("mechanical_power")
    power_dyn = phase_dyn.return_variable("mechanical_power")
    tension_qs = phase_qs.return_variable("tension_tether_ground")
    tension_dyn = phase_dyn.return_variable("tension_tether_ground")
    roll_qs = phase_qs.return_variable("input_steering")
    roll_dyn = phase_dyn.return_variable("input_steering")
    azimuth_qs = phase_qs.return_variable("angle_azimuth")
    azimuth_dyn = phase_dyn.return_variable("angle_azimuth")
    elevation_qs = phase_qs.return_variable("angle_elevation")
    elevation_dyn = phase_dyn.return_variable("angle_elevation")
    course_rate_qs = phase_qs.return_variable("timeder_angle_course")
    course_rate_dyn = phase_dyn.return_variable("timeder_angle_course")
    aoa_qs = phase_qs.return_variable("angle_of_attack")
    aoa_dyn = phase_dyn.return_variable("angle_of_attack")
    idx_vmax_qs = np.argmax(vtau_qs)
    print(idx_vmax_qs)
    idx_vmax_dyn = np.argmax(vtau_dyn)
    idx_vmin_qs = np.argmin(vtau_qs)
    idx_vmin_dyn = np.argmin(vtau_dyn)

    # Calculate the difference in power in percentage
    diff_power = (np.mean(power_qs) - np.mean(power_dyn)) / np.mean(power_dyn) * 100
    diff_vtau = (np.max(vtau_qs) - np.max(vtau_dyn)) / np.max(vtau_dyn) * 100
    diff_tension = (
        (np.mean(tension_qs) - np.mean(tension_dyn)) / np.mean(tension_dyn) * 100
    )
    diff_max_roll = np.degrees(max(roll_qs) - max(roll_dyn))
    diff_s_vmax = np.degrees(-s_dyn[idx_vmax_dyn] + s_qs[idx_vmax_qs])
    diff_s_vmin = np.degrees(-s_dyn[idx_vmin_dyn] + s_qs[idx_vmin_qs])

    print(f"Tangent speed difference: {diff_vtau.max():.2f}%")
    print(f"Power difference: {diff_power.max():.2f}%")
    print(f"Tension difference: {diff_tension.max():.2f}%")
    print(f"Max roll difference: {diff_max_roll:.2f} degrees")
    print(f"Max speed phase difference: {diff_s_vmax:.2f} degrees")
    print(f"Min speed phase difference: {diff_s_vmin:.2f} degrees")
    s_dyn = s_dyn
    s_qs = s_qs
    if i < len(colors):
        ax3.plot(np.degrees(s_dyn), vtau_dyn, color=colors[i])
        ax3.plot(np.degrees(s_qs), vtau_qs, linestyle="--", color=colors[i])
        ax4.plot(np.degrees(s_dyn), tension_dyn / 1000, color=colors[i])
        ax4.plot(np.degrees(s_qs), tension_qs / 1000, linestyle="--", color=colors[i])
        ax5.plot(np.degrees(s_dyn), roll_dyn, color=colors[i])
        ax5.plot(np.degrees(s_qs), roll_qs, linestyle="--", color=colors[i])
        ax6.plot(np.degrees(s_dyn), vr_dyn, color=colors[i])
        ax6.plot(np.degrees(s_qs), vr_qs, linestyle="--", color=colors[i])

    energy_qs = np.sum(power_qs * np.diff(t_qs, prepend=0.1))
    energy_dyn = np.sum(power_dyn * np.diff(t_dyn, prepend=0.1))
    print("Total energy qs", np.sum(energy_qs))
    print("Total energy dyn", np.sum(energy_dyn))
    power_qs = energy_qs / (t_qs[-1] - t_qs[0])  # average power
    power_dyn = energy_dyn / (t_dyn[-1] - t_dyn[0])  # average power
    print("mean power qs", power_qs)
    print("mean power dyn", power_dyn)

    print("mean tension qs", np.mean(tension_qs))
    print("mean tension dyn", np.mean(tension_dyn))
    min_pow_qs.append(np.min(tension_qs * vr_qs))
    min_pow_dyn.append(np.min(tension_dyn * vr_dyn))
    max_pow_qs.append(np.max(tension_qs * vr_qs))
    max_pow_dyn.append(np.max(tension_dyn * vr_dyn))

    lift_qs = phase_qs.return_variable("lift_coefficient")
    lift_dyn = phase_dyn.return_variable("lift_coefficient")
    drag_qs = phase_qs.return_variable("drag_coefficient")
    drag_dyn = phase_dyn.return_variable("drag_coefficient")
    mean_lift_qs.append(np.mean(lift_qs))
    mean_lift_dyn.append(np.mean(lift_dyn))
    mean_drag_qs.append(np.mean(drag_qs))
    mean_drag_dyn.append(np.mean(drag_dyn))
    print("Mean aoa qs", np.mean(aoa_qs) * 180 / np.pi)
    print("Mean aoa dyn", np.mean(aoa_dyn) * 180 / np.pi)
    print("Mean lift qs", np.mean(lift_qs))
    print("Mean lift dyn", np.mean(lift_dyn))
    print("Mean drag qs", np.mean(drag_qs))
    print("Mean drag dyn", np.mean(drag_dyn))

ax1.set_xticklabels([])  # Remove x-ticks
ax3.set_xticklabels([])  # Remove x-ticks
ax4.set_xticklabels([])  # Remove x-ticks
ax5.set_xticklabels([])  # Remove x-ticks
# Set xlim for all subplots
ax3.set_xlim([0, 360])
ax4.set_xlim([0, 360])
ax5.set_xlim([0, 360])
ax6.set_xlim([0, 360])
max_el = max(np.max(elevation_qs), np.max(elevation_dyn)) * 180 / np.pi
ax1.set_ylim([0, max_el + 10])
ax2.set_ylim([0, max_el + 10])

vmin = min(np.min(vtau_qs), np.min(vtau_dyn))
vmax = max(np.max(vtau_qs), np.max(vtau_dyn))
scatter = ax1.scatter(
    azimuth_dyn * 180 / np.pi,
    elevation_dyn * 180 / np.pi,
    c=vtau_dyn,
    cmap="viridis",
    s=10,
    vmin=vmin,
    vmax=vmax,
)  # `s` adjusts marker size
cbar_ax = fig.add_axes([0.35, 0.3, 0.02, 0.4])  # Manually positioned colorbar
cbar = fig.colorbar(scatter, cax=cbar_ax)
cbar.set_label(PLOT_LABELS["speed_tangential"])
cbar.set_ticks(np.linspace(vmin, vmax, num=5))


scatter = ax2.scatter(
    azimuth_qs * 180 / np.pi,
    elevation_qs * 180 / np.pi,
    c=vtau_qs,
    cmap="viridis",
    s=10,
    vmin=vmin,
    vmax=vmax,
)  # `s` adjusts marker size

# plt.tight_layout()
# Save figure as pdf
plt.savefig(save_folder + "parametrized_circle_results.pdf", bbox_inches="tight")
# plt.show()


plt.figure()
plt.plot(s_qs, course_rate_qs, label="Quasi-steady")
plt.plot(s_dyn, course_rate_dyn, label="Dynamic")
plt.xlabel("Phase [rad]")
plt.ylabel("Course rate [rad/s]")
plt.legend()
plt.grid()
plt.show()
