# kite_analysis_tools.py

import numpy as np
import casadi as ca
import json
import matplotlib.pyplot as plt
from awetrim import SystemModel
from awetrim.system.kite import Kite
from typing import Dict, List, Tuple


def load_aero_input(filepath: str) -> Dict:
    with open(filepath, "r") as file:
        return json.load(file)


def build_kite_model(
    aero_input: Dict, mass_wing: float = 10.0, area_wing: float = 20.0
) -> SystemModel:
    kite = Kite(
        mass_wing=mass_wing,
        area_wing=area_wing,
        aero_input=aero_input,
        mass_kcu=0,
        steering_control="roll",
    )
    return SystemModel(dof=3, quasi_steady=False, kite=kite)


def get_initial_state(
    speed_wind_ref=10.0,
    distance_radial=200.0,
    speed_radial=-2.0,
    angle_elevation=0,
    angle_azimuth=0,
    angle_course=0,
    speed_tangential=0,
    timeder_angle_course=0,
    input_depower=0,
) -> Dict[str, float]:
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


def solve_qs_system(
    model: SystemModel,
    initial_state: Dict[str, float],
    unknowns: List[str],
    guess: List[float],
    solver_options: Dict,
) -> Tuple[Dict[str, float], Dict[str, float], bool]:
    model.setup_qs_solver(unknown_vars=unknowns, solver_options=solver_options)
    p = [initial_state[name] for name in model._qs_inputs]
    lbx, ubx, lbg, ubg = model.get_boundaries(initial_state, unknowns)

    if "length_tether" in unknowns:
        idx = unknowns.index("length_tether")
        ubx[idx] = initial_state["distance_radial"]

    try:
        sol = model._qs_solver(x0=guess, p=p, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
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


def compute_jacobian_by_names(
    model: SystemModel, state_values: Dict[str, float], selected_state_names: List[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    model.establish_ode_function()
    state_vector_syms = ca.vertsplit(model.state_vector)
    name_to_symbol = {var.name(): var for var in state_vector_syms}
    name_to_index = {var.name(): idx for idx, var in enumerate(state_vector_syms)}
    selected_syms = [name_to_symbol[name] for name in selected_state_names]
    selected_indices = [name_to_index[name] for name in selected_state_names]
    residuals_to_use = [model._ode[i] for i in selected_indices]
    J = ca.jacobian(ca.vertcat(*residuals_to_use), ca.vertcat(*selected_syms))
    for var in ca.symvar(J):
        name = var.name()
        if name in state_values:
            J = ca.substitute(J, var, state_values[name])
    J_np = np.array(ca.DM(J))
    eigvals, eigvecs = np.linalg.eig(J_np)
    return J_np, eigvals, eigvecs


def compute_jacobians_with_input(
    model: SystemModel,
    state_values: Dict[str, float],
    selected_state_names: List[str],
    input_names: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    plt.plot(eigvals.real, eigvals.imag, "o", label="Eigenvalues")
    plt.axhline(0, color="k", linestyle="--")
    plt.axvline(0, color="k", linestyle="--")
    plt.xlabel("Real Part")
    plt.ylabel("Imaginary Part")
    plt.grid(True)
    plt.legend()
    plt.title("Eigenvalues in Complex Plane")
    # plt.show()


def plot_eigenvectors(eigvecs: np.ndarray, state_names: List[str]):
    eigvecs_real = np.real(eigvecs / np.linalg.norm(eigvecs, axis=0))
    plt.figure()
    for idx in range(eigvecs_real.shape[1]):
        plt.plot(state_names, eigvecs_real[:, idx], marker="o", label=f"v{idx}")
    plt.xlabel("State Variable")
    plt.ylabel("Normalized Magnitude")
    plt.title("Normalized Eigenvectors (Real Part)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    # plt.show()


def sweep_and_plot_locus(
    sweep_variable: str,
    sweep_range: np.ndarray,
    fixed_state: Dict[str, float],
    model: SystemModel,
    unknowns: List[str],
    guess: List[float],
    solver_options: Dict,
    selected_states: List[str],
    annotate_every: int = 10,
):
    all_eigvals = []
    all_halftimes = []
    qs_parameters = []

    for value in sweep_range:
        fixed_state[sweep_variable] = float(value)
        if sweep_variable == "mass_wing":
            model.mass_wing = fixed_state[sweep_variable]
        elif sweep_variable == "area_wing":
            model.area_wing = fixed_state[sweep_variable]
        try:
            qs_state, full_state, success = solve_qs_system(
                model, fixed_state, unknowns, guess, solver_options
            )
            lift_func = model.extract_function("lift_coefficient")
            drag_func = model.extract_function("drag_coefficient")
            CL = float(lift_func(*[full_state[name] for name in lift_func.name_in()]))
            CD = float(drag_func(*[full_state[name] for name in drag_func.name_in()]))
            print(
                f"{sweep_variable}={value}, CL={CL}, CD={CD}, "
                f"speed_wind_ref={full_state['speed_wind_ref']}"
                f", speed_tangential={full_state['speed_tangential']}"
                f", input_steering={full_state['input_steering']}"
            )
            if not success:
                all_eigvals.append([np.nan + 1j * np.nan] * len(selected_states))
                all_halftimes.append([np.nan] * len(selected_states))
                qs_parameters.append(np.nan)
                print(
                    f"[WARN] Solver failed for {sweep_variable}={value:.3f}: No solution found."
                )
                continue
            # guess = [qs_state[name] for name in unknowns]
            _, eigvals, _ = compute_jacobian_by_names(
                model, full_state, selected_states
            )
            all_eigvals.append(eigvals)
            halftimes = [halftime_from_eigenvalue(lam) for lam in eigvals]
            all_halftimes.append(halftimes)
            qs_parameters.append(
                quasi_steady_parameter(
                    rho=model.rho,
                    A=model.area_wing,
                    m=model.mass_wing,
                    g=model.g,
                    CL=float(CL),
                    CD=float(CD),
                    vw=full_state["speed_wind_ref"],
                )
            )
        except Exception as e:
            print(f"[WARN] Solver failed for {sweep_variable}={value:.3f}: {e}")
            all_eigvals.append([np.nan + 1j * np.nan] * len(selected_states))
            all_halftimes.append([np.nan] * len(selected_states))
            qs_parameters.append(np.nan)

    all_eigvals = np.array(all_eigvals)
    all_halftimes = np.array(all_halftimes)
    qs_parameters = np.array(qs_parameters)

    # Plot eigenvalue locus
    plt.figure(figsize=(8, 6))
    for i in range(all_eigvals.shape[1]):
        mode_vals = all_eigvals[:, i]
        plt.plot(mode_vals.real, mode_vals.imag, "-o", label=f"Mode {i}")
        for j in range(0, len(mode_vals), annotate_every):
            val = sweep_range[j]
            x, y = mode_vals[j].real, mode_vals[j].imag
            if not np.isnan(x) and not np.isnan(y):
                plt.annotate(
                    f"{val:.1f}",
                    (x, y),
                    fontsize=8,
                    textcoords="offset points",
                    xytext=(5, 5),
                )

    plt.axhline(0, color="gray", linestyle="--")
    plt.axvline(0, color="gray", linestyle="--")
    plt.xlabel("Real Part")
    plt.ylabel("Imaginary Part")
    plt.title(f"Eigenvalue Locus vs {sweep_variable}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot halftimes
    plt.figure(figsize=(8, 6))
    for i in range(all_halftimes.shape[1]):
        plt.plot(sweep_range, all_halftimes[:, i], "-o", label=f"Mode {i}")
    plt.xlabel(sweep_variable)
    plt.ylabel("Halftime [s]")
    plt.title(f"Halftime of Modes vs {sweep_variable}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # plt.show()

    # Plot quasi-steady parameters
    plt.figure(figsize=(8, 6))
    plt.plot(sweep_range, qs_parameters, "-o", label="Quasi-Steady Parameter")
    plt.xlabel(sweep_variable)
    plt.ylabel("Quasi-Steady Parameter (Pi_qs)")
    plt.title("Quasi-Steady Parameter vs " + sweep_variable)
    plt.grid(True)
    plt.legend()
    # plt.show()

    # Plot quasi-steady parameters
    plt.figure(figsize=(8, 6))
    plt.plot(qs_parameters, all_halftimes[:, 0], "-o", label="Quasi-Steady Parameter")
    plt.ylabel("Halftime [s]")
    plt.xlabel("Quasi-Steady Parameter (Pi_qs)")
    plt.title("Quasi-Steady Parameter vs Halftime")
    plt.grid(True)
    plt.legend()
    plt.show()


def halftime_from_eigenvalue(lam):
    """
    Compute the halftime (decay to half amplitude) for a given eigenvalue.

    Parameters
    ----------
    lam : complex or float
        Eigenvalue of the system (should have Re(lam) < 0 for decay)

    Returns
    -------
    t_half : float or np.nan
        Halftime in seconds, or NaN if eigenvalue is not decaying
    """
    real_part = np.real(lam)
    if real_part >= 0:
        return np.nan  # no decay or growing mode
    return np.log(2) / -real_part


def quasi_steady_parameter(rho, A, m, g, CL, CD, vw):
    """
    Compute the quasi-steady parameter Pi_qs.

    Parameters:
    ----------
    rho : float
        Air density [kg/m^3]
    A : float
        Reference area [m^2]
    m : float
        Mass of the system [kg]
    g : float
        Gravitational acceleration [m/s^2]
    CL : float
        Lift coefficient
    CD : float
        Drag coefficient
    vw : float
        Wind speed [m/s]

    Returns:
    -------
    Pi_qs : float
        Quasi-steady parameter (dimensionless)
    """
    aero_ratio = CL / CD
    return A / (m) * aero_ratio


def plot_time_response(
    A: np.ndarray,
    x0: np.ndarray,
    state_names: List[str],
    t_final: float = 60.0,
    num_points: int = 1000,
):
    """
    Simulate and plot the time response of a linearised system.

    Parameters
    ----------
    A : np.ndarray
        Jacobian matrix (n x n)
    x0 : np.ndarray
        Initial perturbation from steady state (n,)
    state_names : List[str]
        Names of the state variables
    t_final : float
        Final time for simulation
    num_points : int
        Number of time samples
    """
    from scipy.linalg import expm

    t = np.linspace(0, t_final, num_points)
    X = np.array([expm(A * ti) @ x0 for ti in t])  # shape: (len(t), n)

    plt.figure(figsize=(10, 6))
    for i in range(X.shape[1]):
        plt.plot(t, X[:, i], label=state_names[i])
    plt.xlabel("Time [s]")
    plt.ylabel("State Perturbation")
    plt.title("Linearised Time Response Around Steady State")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_single_mode_response(
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    mode_index: int,
    x0: np.ndarray,
    state_names: List[str],
    t_final: float = 10.0,
    num_points: int = 1000,
):
    """
    Plot the time response due to a single eigenmode.

    Parameters
    ----------
    eigvals : np.ndarray
        Eigenvalues of A (shape: [n])
    eigvecs : np.ndarray
        Right eigenvectors of A (shape: [n, n])
    mode_index : int
        Index of the mode to plot (0-based)
    x0 : np.ndarray
        Initial perturbation (same size as number of states)
    state_names : List[str]
        Names of the state variables
    t_final : float
        End time of simulation
    num_points : int
        Number of time steps
    """
    n = eigvecs.shape[0]
    λ = eigvals[mode_index]
    v = eigvecs[:, mode_index]

    # project initial condition onto mode
    V = eigvecs
    V_inv = np.linalg.inv(V)
    c = V_inv @ x0
    c_i = c[mode_index]

    t = np.linspace(0, t_final, num_points)
    x_mode = np.array([np.real(c_i * np.exp(λ * ti) * v) for ti in t])

    plt.figure(figsize=(10, 6))
    for i in range(n):
        plt.plot(t, x_mode[:, i], label=state_names[i])
    plt.xlabel("Time [s]")
    plt.ylabel("State Contribution from Mode {}".format(mode_index))
    plt.title(
        "Response of Mode {} (λ = {:.3f} {:+.3f}j)".format(mode_index, λ.real, λ.imag)
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
