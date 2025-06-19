# wind_gust.py (refactored using kite_analysis_tools)

import numpy as np
import matplotlib.pyplot as plt
from picawe.stability.stability_functions import (
    load_aero_input,
    build_kite_model,
    get_initial_state,
    solve_qs_system,
    compute_jacobians_with_input,
    plot_eigenvalues,
    plot_eigenvectors
)
from scipy.integrate import solve_ivp

# =============================================================================
# 📌 Configuration
# =============================================================================

ANALYSIS_MODE = "static"
AERO_INPUT_FILE = "./data/v3_aero_input.json"
SELECTED_STATES = [
    "speed_tangential",
    "speed_radial",
    "distance_radial",
    "angle_elevation",
]

# 🧩 CONFIGURABLE PARAMETERS
SPEED_RADIAL = 0
INPUT_DEPOWER = 0
ANGLE_COURSE = 0.0
SPEED_WIND_REF = 15.0

# =============================================================================
# 🚀 Time response computation (numerical integration)
# =============================================================================

def compute_time_response(A, B, delta_vw, t_vec):
    def linear_system(t, x):
        return A @ x + B.flatten() * delta_vw

    x0 = np.zeros(A.shape[0])
    sol = solve_ivp(linear_system, (t_vec[0], t_vec[-1]), x0, t_eval=t_vec, method="RK45")
    return sol.y.T

# =============================================================================
# 🚀 Main analysis
# =============================================================================

def run_analysis():
    aero_input = load_aero_input(AERO_INPUT_FILE)
    model = build_kite_model(aero_input)
    initial_state = get_initial_state(
        speed_radial=SPEED_RADIAL,
        input_depower=INPUT_DEPOWER,
        angle_course=ANGLE_COURSE,
        speed_wind_ref=SPEED_WIND_REF,
    )

    if ANALYSIS_MODE == "quasi_steady":
        unknowns = ["speed_tangential", "input_steering", "length_tether"]
        guess = [150, 0.0, initial_state["distance_radial"]]
    elif ANALYSIS_MODE == "static":
        unknowns = ["angle_elevation", "input_steering", "length_tether"]
        guess = [1, 0.0, initial_state["distance_radial"]]
    else:
        raise ValueError(f"Unknown mode '{ANALYSIS_MODE}'.")

    solver_options = {"ipopt": {"print_level": 0, "sb": "yes"}, "print_time": False}
    qs_state, full_state, success = solve_qs_system(model, initial_state, unknowns, guess, solver_options)
    if not success:
        print("Failed to compute quasi-steady solution.")
        return

    print(f"\n--- {ANALYSIS_MODE.replace('_', ' ').title()} Analysis ---")
    for k, v in qs_state.items():
        print(f"{k}: {v:.4f}")

    A, B, eigvals, eigvecs = compute_jacobians_with_input(model, full_state, SELECTED_STATES, input_names=["speed_wind_ref"])

    print("\nJacobian Matrix:")
    print(A)

    print("\nEigenvalues:")
    print(eigvals)

    print("\nEigenvectors:")
    print(eigvecs)

    for i in range(eigvecs.shape[1]):
        projection = np.dot(eigvecs[:, i].T, B[:, 0])
        print(f"Mode {i} projection from wind input: {projection}")

    # Simulate response to 2 m/s wind step
    delta_vw = 2.0
    t_vec = np.linspace(0, 30, 5000)
    x_t = compute_time_response(A, B, delta_vw, t_vec)

    plt.figure(figsize=(10, 6))
    for i, name in enumerate(SELECTED_STATES):
        if 'angle' in name:
            x_t[:, i] = np.rad2deg(x_t[:, i])
        if name == "angle_elevation":
            x0 = x_t[-1, i]
            half_val = 0.5 * x0
            idx_half = np.where(np.abs(x_t[:, i]) >= np.abs(half_val))[0][0]
            t_half = t_vec[idx_half]
            plt.axvline(x=t_half, color="black", linestyle='--', label='$t_{1/2}$ elevation')
        plt.plot(t_vec, x_t[:, i], label=name)

    plt.xlabel("Time [s]")
    plt.ylabel("State perturbation")
    plt.title("Time response to +2 m/s step wind gust")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plot_eigenvalues(eigvals)
    plot_eigenvectors(eigvecs, SELECTED_STATES)

if __name__ == "__main__":
    run_analysis()