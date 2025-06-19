# case1.py (refactored with eigenvalue locus using kite_analysis_tools)

import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
from picawe.stability.stability_functions import (
    load_aero_input,
    build_kite_model,
    get_initial_state,
    solve_qs_system,
    compute_jacobian_by_names,
    plot_eigenvalues,
    plot_eigenvectors,
    sweep_and_plot_locus,
    halftime_from_eigenvalue,
    plot_time_response,
    plot_single_mode_response,
)

AERO_INPUT_FILE = "./data/LEI-V9-KITE/v9_aero_input.json"
# AERO_INPUT_FILE = "./data/AP2/ap2_aero_input.json"

SELECTED_STATES = [
    "speed_tangential",
    # "angle_course",
    "speed_radial",
    "distance_radial",
    # "angle_elevation",
    # "angle_azimuth",
]

# 🧩 CONFIGURABLE PARAMETERS
SPEED_RADIAL = 0
INPUT_DEPOWER = 0
DISTANCE_RADIAL = 500.0  # Initial radial distance
ANGLE_ELEVATION = 50 / 180 * np.pi  # Convert degrees to radians
ANGLE_AZIMUTH = 0 / 180 * np.pi  # Convert degrees to radians
ANGLE_COURSE = -np.pi / 2
SPEED_WIND_REF = 12
SWEEP_VARIABLE = "mass_wing"  # Variable to sweep
SWEEP_VALUES = np.linspace(10, 500, 50)

AREA_WING = 46.0  # m^2
MASS_WING = 78.0  # kg


def run_case1():
    aero_input = load_aero_input(AERO_INPUT_FILE)
    aero_input["params"]["angle_pitch_depower_0"] += -0.05  # Adjust pitch for depower
    # aero_input = {}
    # aero_input["model"] = "coeffs"
    # aero_input["params"] = {
    #     "CL0": 0.7,
    #     "CD0": 0.1,
    #     "angle_pitch_depower_0": 0.0,
    #     "delta_pitch_depower": 0.0,
    # }
    model = build_kite_model(aero_input, area_wing=AREA_WING, mass_wing=MASS_WING)
    model.override_gravity = True  # Override gravity to ensure static equilibrium

    radius_turn = ca.SX.sym("radius_turn")
    model.timeder_angle_course = model.speed_tangential**2 / radius_turn
    initial_state = get_initial_state(
        speed_radial=SPEED_RADIAL,
        input_depower=INPUT_DEPOWER,
        angle_course=ANGLE_COURSE,
        speed_wind_ref=SPEED_WIND_REF,
        angle_elevation=ANGLE_ELEVATION,
        angle_azimuth=ANGLE_AZIMUTH,
        # timeder_angle_course=TIMEDER_ANGLE_COURSE,
    )
    # angle_turn
    initial_state["radius_turn"] = -(
        np.sin(ANGLE_ELEVATION) * DISTANCE_RADIAL
        + np.sin(ANGLE_AZIMUTH) * DISTANCE_RADIAL
    )  # Set a large radius for static equilibrium
    print("Radius turn:", initial_state["radius_turn"])
    unknowns = ["speed_tangential", "input_steering", "length_tether"]
    guess = [40, 0, initial_state["distance_radial"] * 0.99]
    solver_options = {"ipopt": {"print_level": 0, "sb": "yes"}, "print_time": False}

    qs_state, full_state, success = solve_qs_system(
        model, initial_state, unknowns, guess, solver_options
    )

    print(f"Quasi-steady state: {qs_state}")
    full_state["distance_radial"] = initial_state["distance_radial"]
    full_state["angle_elevation"] = initial_state["angle_elevation"]
    full_state["angle_azimuth"] = initial_state["angle_azimuth"]
    full_state = {**full_state, **qs_state}  # Merge states
    power_fun = model.extract_function("mechanical_power")
    aoa_fun = model.extract_function("angle_of_attack")
    cl_fun = model.extract_function("lift_coefficient")
    cd_fun = model.extract_function("drag_coefficient")
    power = power_fun(**{name: full_state[name] for name in power_fun.name_in()})
    aoa = aoa_fun(**{name: full_state[name] for name in aoa_fun.name_in()})[
        "angle_of_attack"
    ]
    cl = cl_fun(**{name: full_state[name] for name in cl_fun.name_in()})[
        "lift_coefficient"
    ]
    cd = cd_fun(**{name: full_state[name] for name in cd_fun.name_in()})[
        "drag_coefficient"
    ]
    print("\n--- Quasi-steady state results ---")
    print("CL^3/CD^2:", (cl**3) * np.cos(full_state["input_steering"]) ** 2 / (cd**2))
    print(f"Angle of attack: {np.degrees(aoa)} degrees")
    print(f"Mechanical power: {power} W")
    if not success:
        print("[ERROR] Failed to solve quasi-steady state.")
        return

    print("\n--- Case 1: Static flight equilibrium ---")
    for k, v in qs_state.items():
        print(f"{k}: {v:.4f}")

    A, eigvals, eigvecs = compute_jacobian_by_names(model, full_state, SELECTED_STATES)

    # define a small perturbation (can also be one-hot)
    x0 = np.zeros(len(SELECTED_STATES))
    x0[0] = 2  # small initial offset in first state

    # plot
    plot_time_response(A, x0, SELECTED_STATES)

    print("\nEigenvalues:")
    print(eigvals)

    print("\nEigenvectors:")
    print(eigvecs)

    plot_eigenvalues(eigvals)
    plot_eigenvectors(eigvecs, SELECTED_STATES)

    plt.show()
    # Plot only the 2nd mode (index 1)
    plot_single_mode_response(
        eigvals, eigvecs, mode_index=0, x0=x0, state_names=SELECTED_STATES
    )

    # --- Locus analysis ---
    sweep_and_plot_locus(
        sweep_variable=SWEEP_VARIABLE,
        sweep_range=SWEEP_VALUES,
        fixed_state=initial_state,
        model=model,
        unknowns=unknowns,
        guess=guess,
        solver_options=solver_options,
        selected_states=SELECTED_STATES,
    )


if __name__ == "__main__":
    run_case1()
