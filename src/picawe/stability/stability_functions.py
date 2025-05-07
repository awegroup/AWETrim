# kite_analysis_tools.py

import numpy as np
import casadi as ca
import json
import matplotlib.pyplot as plt
from picawe import SystemModel
from picawe.system.kite import Kite
from typing import Dict, List, Tuple

def load_aero_input(filepath: str) -> Dict:
    with open(filepath, "r") as file:
        return json.load(file)

def build_kite_model(aero_input: Dict, mass_wing: float = 15.0, area_wing: float = 20.0) -> SystemModel:
    kite = Kite(
        mass_wing=mass_wing,
        area_wing=area_wing,
        aero_input=aero_input,
        mass_kcu=20.0,
        steering_control="asymmetric",
    )
    return SystemModel(dof=3, quasi_steady=False, wind_model="uniform", kite=kite)

def get_initial_state(speed_wind_ref=10.0, distance_radial=200.0, speed_radial=-2.0, angle_elevation = 0, angle_azimuth = 0, angle_course = 0, speed_tangential = 0, timeder_angle_course = 0, input_depower = 0) -> Dict[str, float]:
    return {
        "speed_wind_ref": speed_wind_ref,
        "angle_elevation": angle_elevation,
        "angle_azimuth": angle_azimuth,
        "angle_course": angle_course,
        "speed_radial": speed_radial,
        "speed_tangential": speed_tangential,
        "timeder_speed_radial": 0.0,
        "distance_radial": distance_radial,
        "input_depower": input_depower,
        "timeder_angle_course": timeder_angle_course,
        "timeder_speed_tangential": 0.0,
    }

def solve_qs_system(model: SystemModel, initial_state: Dict[str, float], unknowns: List[str], guess: List[float], solver_options: Dict) -> Tuple[Dict[str, float], Dict[str, float], bool]:
    solve_qs, inputs_name = model.setup_qs_solver(unknown_vars=unknowns, solver_options=solver_options)
    p = [initial_state[name] for name in inputs_name]
    lbx, ubx, lbg, ubg = model.get_boundaries(unknowns)

    if "length_tether" in unknowns:
        idx = unknowns.index("length_tether")
        ubx[idx] = initial_state["distance_radial"]

    try:
        sol = solve_qs(x0=guess, p=p, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        g_val = np.abs(np.array(sol["g"]))
        if np.any(np.isnan(g_val)) or np.any(np.isinf(g_val)) or np.any(g_val > 1e-3):
            print(f"[WARN] Solver failed: g_val = {g_val}")
            return {}, {}, False

        sol_vec = sol["x"]
        qs_state = {name: float(sol_vec[i]) for i, name in enumerate(unknowns)}
        all_values = {**initial_state, **qs_state}
        return qs_state, all_values, True

    except RuntimeError:
        return {}, {}, False

def compute_jacobian_by_names(model: SystemModel, state_values: Dict[str, float], selected_state_names: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.establish_ode()
    state_vector_syms = ca.vertsplit(model.state_vector)
    name_to_symbol = {var.name(): var for var in state_vector_syms}
    name_to_index = {var.name(): idx for idx, var in enumerate(state_vector_syms)}
    selected_syms = [name_to_symbol[name] for name in selected_state_names]
    selected_indices = [name_to_index[name] for name in selected_state_names]
    residuals_to_use = [model.ode[i] for i in selected_indices]
    J = ca.jacobian(ca.vertcat(*residuals_to_use), ca.vertcat(*selected_syms))
    for var in ca.symvar(J):
        name = var.name()
        if name in state_values:
            J = ca.substitute(J, var, state_values[name])
    J_np = np.array(ca.DM(J))
    eigvals, eigvecs = np.linalg.eig(J_np)
    return J_np, eigvals, eigvecs

def compute_jacobians_with_input(model: SystemModel, state_values: Dict[str, float], selected_state_names: List[str], input_names: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.establish_ode()
    state_vector_syms = ca.vertsplit(model.state_vector)
    name_to_symbol = {var.name(): var for var in state_vector_syms}
    name_to_index = {var.name(): idx for idx, var in enumerate(state_vector_syms)}
    selected_syms = [name_to_symbol[name] for name in selected_state_names]
    selected_indices = [name_to_index[name] for name in selected_state_names]
    input_syms = ca.vertsplit(model.input_vector)
    input_name_to_symbol = {var.name(): var for var in input_syms}
    input_syms_selected = []
    for input in input_names:
        if input == "speed_wind_ref":
            input_syms_selected.append(model.wind.speed_wind_ref)
        else:
            input_syms_selected.append(input_name_to_symbol[input])
    residuals_to_use = [model.ode[i] for i in selected_indices]
    A_sym = ca.jacobian(ca.vertcat(*residuals_to_use), ca.vertcat(*selected_syms))
    B_sym = ca.jacobian(ca.vertcat(*residuals_to_use), ca.vertcat(*input_syms_selected))
    for var in ca.symvar(A_sym) + ca.symvar(B_sym):
        name = var.name()
        if name in state_values:
            A_sym = ca.substitute(A_sym, var, state_values[name])
            B_sym = ca.substitute(B_sym, var, state_values[name])
    A = np.array(ca.DM(A_sym))
    B = np.array(ca.DM(B_sym))
    eigvals, eigvecs = np.linalg.eig(A)
    return A, B, eigvals, eigvecs

def plot_eigenvalues(eigvals: np.ndarray):
    plt.figure()
    plt.plot(eigvals.real, eigvals.imag, 'o', label='Eigenvalues')
    plt.axhline(0, color='k', linestyle='--')
    plt.axvline(0, color='k', linestyle='--')
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.grid(True)
    plt.legend()
    plt.title('Eigenvalues in Complex Plane')
    plt.show()

def plot_eigenvectors(eigvecs: np.ndarray, state_names: List[str]):
    eigvecs_real = np.real(eigvecs / np.linalg.norm(eigvecs, axis=0))
    plt.figure()
    for idx in range(eigvecs_real.shape[1]):
        plt.plot(state_names, eigvecs_real[:, idx], marker='o', label=f'v{idx}')
    plt.xlabel('State Variable')
    plt.ylabel('Normalized Magnitude')
    plt.title('Normalized Eigenvectors (Real Part)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


def sweep_and_plot_locus(
    sweep_variable: str,
    sweep_range: np.ndarray,
    fixed_state: Dict[str, float],
    model: SystemModel,
    unknowns: List[str],
    guess: List[float],
    solver_options: Dict,
    selected_states: List[str],
    annotate_every: int = 10
):
    all_eigvals = []

    for value in sweep_range:
        fixed_state[sweep_variable] = float(value)
        if sweep_variable == "mass_wing":
            model.mass_wing = fixed_state[sweep_variable]
        elif sweep_variable == "area_wing":
            model.area_wing = fixed_state[sweep_variable]
        try:
            qs_state, full_state, success = solve_qs_system(model, fixed_state, unknowns, guess, solver_options)
            if not success:
                all_eigvals.append([np.nan + 1j * np.nan] * len(selected_states))
                print(f"[WARN] Solver failed for {sweep_variable}={value:.3f}: No solution found.")
                continue
            guess = [qs_state[name] for name in unknowns]
            _, eigvals, _ = compute_jacobian_by_names(model, full_state, selected_states)
            all_eigvals.append(eigvals)
        except Exception as e:
            print(f"[WARN] Solver failed for {sweep_variable}={value:.3f}: {e}")
            all_eigvals.append([np.nan + 1j * np.nan] * len(selected_states))

    all_eigvals = np.array(all_eigvals)

    plt.figure(figsize=(8, 6))
    for i in range(all_eigvals.shape[1]):
        mode_vals = all_eigvals[:, i]
        plt.plot(mode_vals.real, mode_vals.imag, '-o', label=f'Mode {i}')
        for j in range(0, len(mode_vals), annotate_every):
            val = sweep_range[j]
            x, y = mode_vals[j].real, mode_vals[j].imag
            if not np.isnan(x) and not np.isnan(y):
                plt.annotate(f"{val:.1f}", (x, y), fontsize=8, textcoords="offset points", xytext=(5, 5))

    plt.axhline(0, color='gray', linestyle='--')
    plt.axvline(0, color='gray', linestyle='--')
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.title(f'Eigenvalue Locus vs {sweep_variable}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()