import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from picawe.system.kite import Kite
from picawe import SystemModel, State
import casadi as ca
import time as timet
import json
import os
from picawe.utils.color_palette import (
    get_color_list,
    set_plot_style,
    set_plot_style_no_latex,
)


#  Import and define logging
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -------------------- Load Aero Input --------------------
def define_force_function(
    C_L: float,
    C_D: float,
    phi: float = 0.0,
    beta: float = 0.0,
    rho: float = 1.225,
    S: float = 46.0,
) -> ca.Function:
    """
    Constructs the symbolic force function Ft(v_r, v_w) for given lift and drag coefficients.
    """
    v_r = ca.SX.sym("v_r")
    v_w = ca.SX.sym("v_w")

    C_R = ca.sqrt(C_L**2 + C_D**2)
    Ft = (
        0.5
        * rho
        * S
        * C_R
        * (v_w * ca.cos(phi) * ca.cos(beta) - v_r) ** 2
        * (1 + (C_L / C_D) ** 2)
    )
    return ca.Function("Ft", [v_r, v_w], [Ft])


def compute_optimal_reel_speeds_and_forces(
    aero_coefficients: list,
    reelin_speed: float,
    frac_power_ri: float,
    reel_out_length: float = 200.0,
) -> tuple[dict[tuple[float, float], dict[str, np.ndarray]], np.ndarray]:
    """
    Compute the optimal reeling speeds and corresponding tether forces for given aerodynamic coefficients.

    Args:
        aero_coefficients (list of tuple): List of (C_L, C_D) aerodynamic performance pairs.
        reel_out_length (float): Length of the cable to reel out (in meters).
        reel_in_force (float): Constant force applied during reeling in (in Newtons).
        reel_in_velocity (float): Negative velocity for reeling in (in m/s).

    Returns:
        dict: Dictionary mapping each (C_L, C_D) pair to reeling speeds and tether forces.
        np.ndarray: Array of wind speeds used for simulation.
    """

    optimized_force_data = {}
    wind_speeds = np.linspace(1.5, 25.0, 100)

    phi = beta = 0.0  # The results are not sensitive to these angles in this context

    lift_coefficient, drag_coefficient = aero_coefficients
    force_function = define_force_function(
        lift_coefficient, drag_coefficient, phi, beta
    )
    v_r = ca.SX.sym("v_r")
    v_w = ca.SX.sym("v_w")

    time_ro = reel_out_length / v_r
    time_ri = reel_out_length / reelin_speed
    tether_force_expr = force_function(v_r, v_w)

    force_reelin = (
        np.sqrt(0.35**2 + 0.08**2) * 0.5 * 1.225 * 46.854 * (v_w**2 + reelin_speed**2)
    )
    power_ri = force_reelin * reelin_speed
    power_expr = tether_force_expr * v_r

    average_energy_expr = ((power_expr * time_ro) + power_ri * time_ri) / (
        time_ri + time_ro
    )
    energy_derivative = ca.gradient(average_energy_expr, v_r)

    root_solver = ca.rootfinder(
        "root_solver",
        "newton",
        ca.Function("energy_derivative", [v_r, v_w], [energy_derivative]),
    )

    optimal_reel_speeds = []
    corresponding_tether_forces = []

    for wind_speed in wind_speeds:
        initial_guess = wind_speed / 3 * np.cos(beta)
        try:
            optimal_speed = root_solver(initial_guess, wind_speed)
            optimal_reel_speeds.append(float(optimal_speed))
            corresponding_tether_forces.append(
                float(force_function(optimal_speed, wind_speed))
            )
        except Exception as e:
            logger.warning(
                f"Root solver failed for wind_speed={wind_speed:.2f} m/s: {e}"
            )
            optimal_reel_speeds.append(np.nan)
            corresponding_tether_forces.append(np.nan)

    optimized_force_data = {
        "v_r_sol": np.array(optimal_reel_speeds),
        "ft_sol": np.array(corresponding_tether_forces),
    }
    return optimized_force_data, wind_speeds


save_folder = "./results/figures/translational_paper/"
# -----------------------------------------------
# Load data and define aerodynamic model
# -----------------------------------------------
# Load initial state

# file_path = "./results/impact_inertial_forces/"
# file_name = "helix_quasi_steady.csv"
# results = pd.read_csv(os.path.join(file_path, file_name))

set_plot_style_no_latex()
# Define aerodynamic input
file_path = "./data/LEI-V9-KITE/v9_aero_input.json"
with open(file_path, "r") as file:
    aero_input = json.load(file)

# aero_input = {
#     "model": "inviscid",
#     "params": {
#         "CD0": 0.05,
#         "aspect_ratio": 10,
#         "oswald_efficiency": 1,
#         "angle_pitch_depower_0": 0,
#     },
#     "dependencies": {
#         # "u_s": { "k_cl": 0, "k_cd": 0.0, "k_cs": 0.23, "k_cn": 0.005 },
#     },
# }
# # -----------------------------------------------
# Define the state
# -----------------------------------------------
# State.__bases__ = (KiteKinematics, Tether, Wind, RigidKite)
mass_wing = 0
kite = Kite(
    mass_wing=0,
    area_wing=47,
    aero_input=aero_input,
    mass_kcu=0,
    steering_control="roll",
)
kite_model = SystemModel(dof=3, quasi_steady=True, kite=kite)

speed_wind = 10
kite_model.wind.speed_wind_ref = speed_wind
kite_model.input_depower = 0
kite_model.timeder_angle_course = 0
kite_model.distance_radial = 200

plt.figure()
CL = 0.75
CD = 0.15
CR = np.sqrt(CL**2 + CD**2)
vr_array = np.linspace(0, 5, 50)
F_opt = 2 * 1.225 * 46.854 * CR * vr_array**2 * (1 + (CL / CD) ** 2)
plt.plot(vr_array, F_opt / 9.81, label="Mean", color="black")
CL = 0.9
CD = 0.12
CR = np.sqrt(CL**2 + CD**2)
F_optmax = 2 * 1.225 * 46.854 * CR * vr_array**2 * (1 + (CL / CD) ** 2)

CL = 0.65
CD = 0.18
CR = np.sqrt(CL**2 + CD**2)
F_optmin = 2 * 1.225 * 46.854 * CR * vr_array**2 * (1 + (CL / CD) ** 2)
plt.fill_between(
    vr_array, F_optmin / 9.81, F_optmax / 9.81, color="black", alpha=0.2, label="Range"
)

plt.ylabel("Tension Tether (N)")
plt.xlabel("Optimal Reeling Speed (m/s)")

x = [-0.2, 1.4, 10]
y = [440, 2260, 2890]
plt.plot(x, y, color="blue", linestyle="--", label="Current curve KP")
plt.ylim([0, 3000])
plt.legend()
plt.tight_layout()
plt.show()


unknown_vars = ["length_tether", "input_steering", "speed_tangential"]
solve_func, inputs_name, _ = kite_model.setup_qs_solver(unknown_vars)
current_state = {
    "distance_radial": 200,
    "angle_elevation": 0,
    "angle_azimuth": 0,
    "angle_course": np.pi / 2,
    "speed_radial": 0,
}
p = [current_state[name] for name in inputs_name]
# print(p)
lbx, ubx, lbg, ubg = kite_model.get_boundaries(current_state, unknown_vars)
# print(lbx,ubx,lbg,ubg)
sol = solve_func(x0=[200, 0, 100], p=p, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
CL_fun = kite_model.extract_function("lift_coefficient")
CD_fun = kite_model.extract_function("drag_coefficient")
qs_state = {name: sol["x"][i] for i, name in enumerate(unknown_vars)}
full_state = {**current_state, **qs_state}
CL = CL_fun(*[full_state[name] for name in CL_fun.name_in()])
CD = CD_fun(*[full_state[name] for name in CD_fun.name_in()])
print(CL, CD)


kite_model.mass_wing = 90
solver_options = {
    "ipopt": {
        "print_level": 0,
        "sb": "yes",
        "tol": 1e-8,
    },
    "print_time": False,
}
tension_func = kite_model.extract_function("tension_tether_ground")
# state.override_gravity = True
colors = get_color_list()
plt.figure(figsize=(5, 4))
wind_speeds = [5, 15, 25]
for vwi, speed_wind in enumerate(wind_speeds):
    kite_model = SystemModel(dof=3, quasi_steady=True, kite=kite)
    kite_model.input_depower = 0
    kite_model.timeder_angle_course = 0
    kite_model.distance_radial = 200
    kite_model.wind.speed_wind_ref = speed_wind
    kite_model.mass_wing = 90
    kite_model.establish_residual()
    variables = ca.symvar(kite_model.residual)
    name_vars = [var.name() for var in variables]
    residual = ca.Function(
        "residual", variables, [kite_model.residual], name_vars, ["residual"]
    )

    print(residual)
    angles_elevation = np.linspace(0, 60, 10) / 180 * np.pi
    angles_course = np.linspace(0, np.pi, 5)
    angles_azimuth = np.linspace(0, 50, 10) / 180 * np.pi
    f = np.zeros((len(angles_azimuth), len(angles_elevation), len(angles_course)))
    ft = np.zeros((len(angles_azimuth), len(angles_elevation), len(angles_course)))

    for i, phi in enumerate(angles_azimuth):
        speed_tangential_i = 30
        tension_tether_i = 1000
        for j, angle_elevation in enumerate(angles_elevation):
            for k, angle_course in enumerate(angles_course):
                opti = ca.Opti()
                reeling_factor = opti.variable()
                speed_tangential = opti.variable()
                angle_roll = opti.variable()
                length_tether = opti.variable()
                opti.subject_to(
                    residual(
                        angle_course=angle_course,
                        angle_azimuth=phi,
                        angle_elevation=angle_elevation,
                        speed_tangential=speed_tangential,
                        input_steering=angle_roll,
                        # speed_wind = speed_wind,
                        speed_radial=reeling_factor * speed_wind,
                        length_tether=length_tether,
                    )["residual"]
                    == 0
                )

                opti.subject_to(0 <= (reeling_factor <= 0.35))
                opti.subject_to(mass_wing * 9.81 <= (length_tether))
                opti.subject_to(0 <= (speed_tangential))
                # opti.subject_to(0 <= (angle_roll <= 1))
                # opti.subject_to(200 >= (length_tether))
                opti.set_initial(reeling_factor, 0.3)
                opti.set_initial(speed_tangential, 50)
                opti.set_initial(angle_roll, 0)
                opti.set_initial(length_tether, 200)
                opti.minimize(-(200 - length_tether) * reeling_factor * speed_wind)
                opti.solver("ipopt", solver_options)
                try:
                    sol = opti.solve()
                    f[i, j, k] = sol.value(reeling_factor)
                    speed_tangential_i = sol.value(speed_tangential)
                    length_tether_i = sol.value(length_tether)
                    input_fun = {
                        "length_tether": length_tether_i,
                        "distance_radial": 200,
                    }
                    ft[i, j, k] = float(
                        tension_func(
                            *[input_fun[name] for name in tension_func.name_in()]
                        )
                    )
                except:
                    f[i, j, k] = np.nan
                    ft[i, j, k] = np.nan
                    print(
                        "Failed at azimuth angle: ",
                        phi,
                        " and course angle: ",
                        angle_course,
                    )

    plt.scatter(
        f * kite_model.wind.speed_wind(kite_model),
        ft,
        color=colors[vwi + 1],
        alpha=0.3,
        label="$v_w$ = " + str(speed_wind) + " m/s",
    )
# -----------------------------------------------
# Plot the results
# -----------------------------------------------
CR = np.sqrt(CL**2 + CD**2)
vr_array = np.linspace(0, 8, 50)
F_opt = 2 * 1.225 * kite_model.area_wing * CR * vr_array**2 * (1 + (CL / CD) ** 2)


plt.plot(vr_array, F_opt, label="Analytical Instantaneous", color=colors[0])

aero_coeffs = [CL, CD]
reelin_speed = 4
frac_power_ri = 0.75
reel_out_length = 200.0

force_data, wind_speeds = compute_optimal_reel_speeds_and_forces(
    aero_coeffs, reelin_speed, frac_power_ri, reel_out_length
)
plt.plot(
    force_data["v_r_sol"],
    force_data["ft_sol"],
    label="Numerical with Reel-in (4 m/s)",
    color=colors[0],
    linestyle="--",
)

reelin_speed = 8
frac_power_ri = 0.75
reel_out_length = 200.0

force_data, wind_speeds = compute_optimal_reel_speeds_and_forces(
    aero_coeffs, reelin_speed, frac_power_ri, reel_out_length
)
plt.plot(
    force_data["v_r_sol"],
    force_data["ft_sol"],
    label="Numerical with Reel-in (8 m/s)",
    color=colors[0],
    linestyle="-.",
)


plt.ylabel("Tension Tether (N)")
plt.xlabel("Reeling Speed (m/s)")
plt.legend()
plt.tight_layout()
plt.savefig(save_folder + "optimal_reeling_speed_force.pdf")
plt.show()
