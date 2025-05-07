# case1.py (refactored with eigenvalue locus using kite_analysis_tools)

import numpy as np
import matplotlib.pyplot as plt
from picawe.stability.stability_functions import (
    load_aero_input,
    build_kite_model,
    get_initial_state,
    solve_qs_system,
    compute_jacobian_by_names,
    plot_eigenvalues,
    plot_eigenvectors,
    sweep_and_plot_locus
)

AERO_INPUT_FILE = "./data/v3_aero_input.json"
SELECTED_STATES = [
    "speed_tangential",
    "speed_radial",
    "distance_radial",
    "angle_course"
]

# 🧩 CONFIGURABLE PARAMETERS
SPEED_RADIAL = 2
INPUT_DEPOWER = 0
ANGLE_COURSE = np.pi/2
ANGLE_ELEVATION = np.pi/6
ANGLE_AZIMUTH = np.pi/6
SPEED_WIND_REF = 15.0
COURSE_RATE = 1

SWEEP_VARIABLE = "mass_wing"
SWEEP_VALUES = np.linspace(20, 100, 20)


def run_case2():
    aero_input = load_aero_input(AERO_INPUT_FILE)
    model = build_kite_model(aero_input)
    initial_state = get_initial_state(speed_radial=SPEED_RADIAL, 
                                      input_depower = INPUT_DEPOWER,
                                        angle_course = ANGLE_COURSE,
                                        angle_elevation = ANGLE_ELEVATION,
                                        speed_wind_ref = SPEED_WIND_REF,
                                        angle_azimuth = ANGLE_AZIMUTH,
                                        timeder_angle_course = COURSE_RATE)

    unknowns = ["speed_tangential", "input_steering", "length_tether"]
    guess = [150, 0.0, initial_state["distance_radial"]]
    solver_options = {"ipopt": {"print_level": 0, "sb": "yes"}, "print_time": False}

    qs_state, full_state, success = solve_qs_system(model, initial_state, unknowns, guess, solver_options)
    if not success:
        print("[ERROR] Failed to solve quasi-steady state.")
        return

    print("\n--- Case 1: Static flight equilibrium ---")
    for k, v in qs_state.items():
        print(f"{k}: {v:.4f}")

    A, eigvals, eigvecs = compute_jacobian_by_names(model, full_state, SELECTED_STATES)

    print("\nEigenvalues:")
    print(eigvals)

    print("\nEigenvectors:")
    print(eigvecs)

    plot_eigenvalues(eigvals)
    plot_eigenvectors(eigvecs, SELECTED_STATES)

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
    run_case2()
