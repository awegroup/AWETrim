import numpy as np
import matplotlib.pyplot as plt
import json
from picawe.kinematics.parametrized_patterns import Helix
from picawe import SystemModel, State
from picawe.utils.color_palette import (
    set_plot_style,
    get_color_list,
    set_plot_style_no_latex,
)
from picawe.timeseries.phase_parametrized import PhaseParameterized
from picawe.system.kite import Kite
from picawe.system.tether import (
    FlexibleLumpedTether,
    RigidLumpedTether,
    RigidLinkTether,
)
from picawe.kinematics.parametrized_patterns import create_pattern_from_dict
from picawe import SystemModel
from picawe.kinematics.Kinematics import ParametrizedKinematics
from picawe.utils.defaults import PLOT_LABELS
from picawe.utils.talmar_equations import compute_power
import yaml
from mpl_toolkits.mplot3d import Axes3D

set_plot_style_no_latex()

# -------------------- Configuration --------------------
file_path = "./data/LEI-V9-KITE/v9_aero_input.json"
# file_path = "./data/AP2/ap2_aero_input.json"

config_file_path = "./data/LEI-V9-KITE/v9_config.yaml"
# config_file_path = "./data/AP2/ap2_config.yaml"
# Open config yaml as dict
with open(config_file_path, "r") as f:
    config_data = yaml.safe_load(f)


with open(file_path, "r") as file:
    aero_input = json.load(file)

wind_speed = 15
pattern_config = {
    "pattern_type": "helix",
    "parameters": {
        "omega": 1.0,
        "r0": 300.0,
        "d0": 100.0,
        "vr": 1.3,
        "beta0": 25 / 180 * np.pi,  # Convert degrees to radians
        "kappa": 0,
        "kbeta": 0,
    },
    "start_path_angle": -np.pi / 2,
    "end_path_angle": 2 * np.pi + np.pi / 2,  # np.pi + np.pi / 2,
    "n_points": 200,
}

save_folder = "./results/figures/translational_paper/"
colors = get_color_list()

mass_ratios = np.arange(0, 11, 1)
radii = np.linspace(5, 20, 11)
radii = np.linspace(5, 20, 41)


optimal_power_coeff = []
optimal_radius = []

for mass_ratio in mass_ratios:
    area_wing = config_data["kite"]["area"]
    span = config_data["kite"]["span"]
    mass_wing = mass_ratio * area_wing
    power_vs_radius = []
    cl_vs_radius = []
    cd_vs_radius = []

    for r in radii:
        pattern_config["parameters"]["d0"] = r * 2
        tether = RigidLinkTether()
        kite = Kite(
            mass_wing=mass_wing,
            area_wing=area_wing,
            aero_input=aero_input,
            steering_control="roll",
        )
        # kite.override_gravity = True  # Override gravity to ensure static equilibrium

        start_state = State(
            t=0,
            s=-np.pi / 2,
            s_dot=2,
            s_ddot=0,
            length_tether=199.6,
            input_steering=0,
            angle_roll=0,
            angle_pitch=0,
            angle_yaw=0,
            tension_tether_ground=1e8,
        )

        model = SystemModel(dof=3, quasi_steady=True, kite=kite, tether=tether)
        # model.override_gravity = True
        model.wind.speed_wind_ref = wind_speed
        model.input_depower = 0

        phase = PhaseParameterized(
            model, quasi_steady=True, pattern_config=pattern_config
        )
        phase.run_simulation(start_state=start_state, allow_failure=False)

        vtau = phase.return_variable("speed_tangential")
        tension = phase.return_variable("tension_tether_ground")

        vr = phase.return_variable("speed_radial")
        x = phase.return_variable("x")

        if len(x) < pattern_config["n_points"]:
            power_vs_radius.append(-np.inf)
            continue

        mean_lift_coeff = np.mean(phase.return_variable("lift_coefficient"))
        mean_drag_coeff = np.mean(phase.return_variable("drag_coefficient"))
        mean_power = np.mean(tension)
        power_vs_radius.append(mean_power)
        cl_vs_radius.append(mean_lift_coeff)
        cd_vs_radius.append(mean_drag_coeff)
        # plt.plot(vtau)
        # plt.show()

    idx_opt = np.argmax(power_vs_radius)
    optimal_power_coeff.append(
        power_vs_radius[idx_opt] / (0.5 * np.pi * span**2 * wind_speed**3 * 1.225)
    )
    optimal_radius.append(radii[idx_opt])
    talmar_power = compute_power(
        m=mass_wing,
        rho=1.225,  # Density of air at sea level in kg/m^3
        S=area_wing,
        CL=cl_vs_radius[idx_opt],
        E=cl_vs_radius[idx_opt] / cd_vs_radius[idx_opt],
        R=radii[idx_opt],
        vw=wind_speed,
    )
    talmar_power /= 0.5 * np.pi * span**2 * wind_speed**3 * 1.225

    print(
        f"Mass Ratio: {mass_ratio}, Optimal Radius: {radii[idx_opt]:.2f} m, "
        f"Optimal Power Oeff: {optimal_power_coeff[-1]:.2f} , Talmar Power Coeff: {talmar_power:.2f} "
    )

    radii = np.linspace(optimal_radius[-1], optimal_radius[-1] + 20, 41)


pattern_config["parameters"]["d0"] = 100
pattern = create_pattern_from_dict(pattern_config, optimize=False)
s = np.linspace(
    pattern_config["start_path_angle"],
    pattern_config["end_path_angle"] + 2 * np.pi,
    400,
)
t = np.linspace(0, 80, 400)
x = pattern.x(t, s)
y = pattern.y(t, s)
z = pattern.z(t, s)

# 3D Trajectory Figure
fig1 = plt.figure(figsize=(7, 6))
ax3d = fig1.add_subplot(111, projection="3d")
ax3d.plot(x, y, z, color=colors[0])
ax3d.set_xlabel("X [m]")
ax3d.set_ylabel("Y [m]")
ax3d.set_zlabel("Z [m]")
ax3d.grid(True)
plt.tight_layout()
fig1.savefig(save_folder + "helix_3d_trajectory.pdf")
plt.show()

# Optimal Radius and Power Coefficient Figure
fig2, ax1 = plt.subplots(figsize=(7, 6))
color1 = colors[1]
color2 = colors[2]
lns1 = ax1.plot(
    mass_ratios, optimal_radius, marker="s", color=color1, label="Optimal Radius [m]"
)
ax1.set_xlabel("Mass Ratio (m/area)")
ax1.set_ylabel("Optimal Radius [m]", color=color1)
ax1.tick_params(axis="y", labelcolor=color1)
ax1.grid(True)

ax2 = ax1.twinx()
lns2 = ax2.plot(
    mass_ratios,
    optimal_power_coeff,
    marker="o",
    color=color2,
    label="Optimal Power Coeff",
)
ax2.set_ylabel("Optimal Power Coefficient", color=color2)
ax2.tick_params(axis="y", labelcolor=color2)

# Legends
lns = lns1 + lns2
labels = [l.get_label() for l in lns]
ax1.legend(lns, labels, loc="upper left")

plt.tight_layout()
fig2.savefig(save_folder + "optimal_radius_power_coeff.pdf")
plt.show()
