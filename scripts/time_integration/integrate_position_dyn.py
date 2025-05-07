"""
Refactored Kite Simulation with centralized inputs, logging, and modular structure.
"""

import json
import time
import logging
from contextlib import contextmanager
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from picawe import SystemModel
from picawe.system.kite import Kite
import casadi as ca

# ------------------------------ Logging Setup ------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ------------------------------ Timed Block ------------------------------
@contextmanager
def timed_block(name: str):
    start = time.time()
    yield
    logger.info(f"{name} completed in {time.time() - start:.2f}s")


# ------------------------------ Simulation Config ------------------------------
def get_simulation_config() -> dict:
    """Centralized definition of all simulation input parameters."""
    return {
        "file_path": "./data/v3_aero_input.json",
        "kite_config": {
            "mass_wing": 5.0,
            "area_wing": 20.0,
            "mass_kcu": 0.0,
            "steering_control": "asymmetric",
        },
        "wind_speed_ref": 10.0,
        "initial_state": {
            "angle_elevation": np.radians(0.0),
            "angle_azimuth": 0.0,
            "angle_course": 0,
            "speed_radial": 0.0,
            "speed_tangential": 0.0,
            "timeder_speed_radial": 0.0,
            "distance_radial": 200.0,
            "input_depower": 0.0,
            "timeder_angle_course": 0.0,
            "timeder_speed_tangential": 0.0,
        },
        "solver_options": {
            "ipopt": {"print_level": 0, "sb": "yes"},
            "print_time": False,
        },
        "quasi_steady_guess": [40.0, 0.0, 200.0],
        "time_step": 0.01,
        "duration": 100.0,
        "control_inputs": {
            "input_steering": 0.0,
            "input_depower": 0.0,
            "timeder_length_tether": 0.0,
        },
    }


# ------------------------------ Main Simulation ------------------------------
def main():
    cfg = get_simulation_config()

    with open(cfg["file_path"], "r") as file:
        aero_input = json.load(file)

    # Setup kite and model
    kite = Kite(aero_input=aero_input, **cfg["kite_config"])
    model = SystemModel(dof=3, quasi_steady=False, wind_model="uniform", kite=kite)
    model.wind.speed_wind_ref = cfg["wind_speed_ref"]

    initial_state = cfg["initial_state"]

    # Solve quasi-steady state
    unknown_vars = ["speed_tangential", "input_steering", "length_tether"]
    solve_qs, inputs_name = model.setup_qs_solver(
        unknown_vars=unknown_vars,
        solver_options=cfg["solver_options"],
    )

    p = [initial_state[name] for name in inputs_name]
    lbx, ubx, lbg, ubg = model.get_boundaries(unknown_vars)
    ubx[0] = initial_state["distance_radial"]  # Constraint

    sol = solve_qs(x0=cfg["quasi_steady_guess"], p=p, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)["x"]
    qs_state = {name: float(sol[i]) for i, name in enumerate(unknown_vars)}
    logger.info("Quasi-steady state solution:")
    for k, v in qs_state.items():
        logger.info(f"{k}: {v:.4f}")

    # Merge all known values
    all_values = {**initial_state, **qs_state}
    current_keys = ["distance_radial", "angle_elevation", "angle_azimuth",
                    "speed_tangential", "angle_course", "speed_radial", "length_tether"]
    x0 = [all_values[k] for k in current_keys]

    # Prepare simulation model
    aoa_func = model.extract_function("angle_of_attack")
    model.establish_ode()
    model.establish_algebraic()
    integrator = model.integrator(cfg["time_step"])
    control_inputs = {**cfg["control_inputs"]}
    p_input = list(control_inputs.values())

    # Integrate over time
    time_range = np.arange(0, cfg["duration"], cfg["time_step"])
    states: List[Dict[str, float]] = []

    with timed_block("Time integration"):
        for t in time_range:
            try:
                xf = integrator(x0=x0, p=p_input)["xf"]
                x0 = xf
                state_now = {key: float(xf[i]) for i, key in enumerate(current_keys)}
                aoa = aoa_func(*[state_now.get(k, control_inputs.get(k)) for k in aoa_func.name_in()])
                states.append({**state_now, "aoa": float(aoa)})
            except Exception as e:
                logger.error(f"Integration failed at time {t:.2f}s: {e}")
                break

    if not states:
        logger.warning("No states collected.")
        return

    logger.info(f"Final elevation angle: {np.degrees(states[-1]['angle_elevation']):.2f}°")
    logger.info(f"Final tether force (if available): {states[-1].get('tension_tether_ground', 'N/A')}")
    visualize_results(pd.DataFrame(states))


# ------------------------------ Visualization ------------------------------
def visualize_results(df: pd.DataFrame) -> None:
    plt.figure()
    plt.plot(df["speed_tangential"], label="Tangential Speed")
    plt.plot(df["speed_radial"], label="Radial Speed")
    plt.xlabel("Time Steps")
    plt.ylabel("Speed [m/s]")
    plt.legend()

    plt.figure()
    plt.plot(np.degrees(df["aoa"]), label="Angle of Attack")
    plt.xlabel("Time Steps")
    plt.ylabel("AOA [deg]")
    plt.legend()

    plt.figure()
    plt.plot(np.degrees(df["angle_course"]), label="Course Angle [deg]")
    plt.legend()

    r, theta, phi = df["distance_radial"], df["angle_azimuth"], df["angle_elevation"]
    x = r * np.cos(phi) * np.cos(theta)
    y = r * np.cos(phi) * np.sin(theta)
    z = r * np.sin(phi)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x, y, z, label="Trajectory")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_xlim(0, 200)
    ax.set_ylim(-100, 100)
    ax.set_zlim(0, 200)
    ax.legend()

    plt.show()


if __name__ == "__main__":
    main()
