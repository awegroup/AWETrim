"""
Inverse solver: Given a prescribed tether force, solve for the wind speed.
Uses measured wind speed as initial condition.
"""

import h5py
import os
import pandas as pd
import numpy as np
from awetrim import SystemModel
from awetrim.system.kite import Kite
from awetrim.system.tether import RigidLumpedTether
from awetrim.environment.Wind import Wind
import casadi as ca
import time
import yaml
import time


def read_results(year, month, day, kite_model, addition="", path_to_main=""):
    path = "/flight_logs/"
    date = str(year) + "-" + str(month) + "-" + str(day)
    file_name = str(kite_model) + "_" + date
    hdf5_path = path_to_main + path + file_name + addition + ".h5"
    ekf_output_df, flight_data_df, config_data = read_results_from_hdf5(hdf5_path)
    return ekf_output_df, flight_data_df, config_data


def read_results_from_hdf5(hdf5_path):
    with h5py.File(hdf5_path, "r") as hf:
        ekf_group = hf["ekf_output"]
        ekf_output_df = pd.DataFrame(
            {
                col: (
                    ekf_group[col][:].astype(str)
                    if ekf_group[col].dtype.kind == "S"
                    else ekf_group[col][:]
                )
                for col in ekf_group.keys()
            }
        )

        flight_group = hf["flight_data"]
        flight_data_df = pd.DataFrame(
            {
                col: (
                    flight_group[col][:].astype(str)
                    if flight_group[col].dtype.kind == "S"
                    else flight_group[col][:]
                )
                for col in flight_group
                if isinstance(flight_group[col], h5py.Dataset)
            }
        )

        config_group = hf["config_data"]
        config_data = read_dict_from_group(config_group)

    return ekf_output_df, flight_data_df, config_data


def read_dict_from_group(group):
    config_dict = {}
    for key, value in group.attrs.items():
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        config_dict[key] = value

    for subgroup_name in group:
        subgroup = group[subgroup_name]
        config_dict[subgroup_name] = read_dict_from_group(subgroup)

    return config_dict


def load_inverse_seed(path="results/inverse_wind_speed_validation.csv"):
    """Load prior inverse simulation results to warm-start this optimizer."""
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - logging only
        print(f"Warning: could not load seed data from {path}: {exc}")
        return None


def setup_inverse_solver(kite_model):
    """
    Setup an inverse solver where we prescribe the tether force
    and solve for wind speed (speed_friction).
    """
    # Define the inverse problem:
    # Unknown variables: [tension_tether_ground, input_steering, speed_friction]
    unknown_vars = [
        "speed_tangential",
        "input_steering",
        "speed_friction",  # Changed from speed_tangential to speed_friction
    ]

    solver_options = {
        "ipopt": {
            "print_level": 0,
            "sb": "yes",
            "max_iter": 400,
        },
        "print_time": False,
    }

    kite_model.setup_qs_solver(unknown_vars, solver_options=solver_options)

    return unknown_vars, solver_options


def create_residual(
    kite_model,
    opti_vars,
    current_state,
):
    """
    Solve for wind speed given a prescribed tether force.

    Parameters
    ----------
    kite_model : SystemModel
        The system model
    current_state : dict
        Current state with known variables
    prescribed_force : float
        Prescribed tether force (in Newtons)
    measured_wind_speed : float
        Measured wind speed to use as initial condition (speed_friction)
    unknown_vars : list
        List of unknown variable names

    Returns
    -------
    dict
        Solution dictionary with solved variables
    """

    kite_model.establish_residual()

    # Decision variables solved by Opti for each sample
    decision_names = ["speed_tangential", "input_steering", "speed_friction"]
    decision_syms = []
    for name in decision_names:
        if hasattr(kite_model, name):
            decision_syms.append(getattr(kite_model, name))
        elif hasattr(kite_model.wind, name):
            decision_syms.append(getattr(kite_model.wind, name))
        else:
            raise KeyError(f"Unknown decision variable '{name}'")

    # Identify remaining symbols in the residual and substitute them with current_state values
    residual_syms = ca.symvar(kite_model.residual)
    param_syms = [
        sym
        for sym in residual_syms
        if sym.name() not in decision_names and sym.name() in current_state
    ]

    param_vals = []
    for sym in param_syms:
        if sym.name() in current_state:
            param_vals.append(current_state[sym.name()])
            # Substitute known states into the residual expression
            setattr(kite_model, sym.name(), current_state[sym.name()])
    kite_model.establish_residual()
    residual = ca.Function(
        "residual",
        decision_syms + list(opti_vars.values()),
        [kite_model.residual],
    )

    return residual


def run_inverse_validation(cycle_num=65):
    """
    Run inverse validation: given measured force, solve for wind speed.

    Parameters
    ----------
    cycle_num : int or list of int
        Cycle number(s) to process. Can be a single integer or a list of integers.
    """
    # Load data
    results, flight_data, config_data = read_results(
        "2019", "10", "08", "v3", addition="", path_to_main="./data/LEI-V3-KITE/"
    )
    print(f"Max cycle: {max(flight_data.cycle)}")

    # Filter to cycle(s)
    if isinstance(cycle_num, (list, tuple)):
        mask = flight_data.cycle.isin(cycle_num)
        print(f"Processing cycles: {cycle_num}")
    else:
        mask = flight_data.cycle == cycle_num
        print(f"Processing cycle: {cycle_num}")
    flight_data = flight_data[mask]
    results = results[mask]
    results = results.reset_index(drop=True)
    flight_data = flight_data.reset_index(drop=True)

    # Normalize depower
    flight_data["up"] = (flight_data["up"] - flight_data["up"].min()) / (
        flight_data["up"].max() - flight_data["up"].min()
    )
    mask = flight_data.up < 0.2
    flight_data = flight_data[mask]
    results = results[mask]
    results = results.reset_index(drop=True)
    flight_data = flight_data.reset_index(drop=True)
    seed_df = load_inverse_seed()
    seed_df = seed_df[mask]
    seed_df = seed_df.reset_index(drop=True)
    seed_lookup = None
    if seed_df is not None:
        seed_cols = [
            col
            for col in ["speed_friction", "speed_tangential", "input_steering"]
            if col in seed_df.columns
        ]
        if "time" in seed_df.columns and seed_cols:
            seed_lookup = (
                pd.merge_asof(
                    flight_data[["time"]].reset_index().sort_values("time"),
                    seed_df.sort_values("time")[["time"] + seed_cols],
                    on="time",
                    direction="nearest",
                    tolerance=0.2,
                )
                .set_index("index")
                .sort_index()
            )

    # Preprocess
    position = np.array(
        [results.kite_position_x, results.kite_position_y, results.kite_position_z]
    ).T
    velocity = np.array(
        [results.kite_velocity_x, results.kite_velocity_y, results.kite_velocity_z]
    ).T
    distance_radial = np.linalg.norm(position, axis=1)
    measured_speed_tangential = np.linalg.norm(
        np.cross(position, velocity), axis=1
    ) / np.maximum(distance_radial, 1e-12)
    azimuth = np.arctan2(results.kite_position_y, results.kite_position_x)

    measured_course_rate = np.gradient(flight_data.kite_course, flight_data.time)
    # Setup system from YAML config (avoids missing JSON file)
    with open("./data/LEI-V3-KITE/v3_kite_input.yaml", "r") as file:
        kite_cfg = yaml.safe_load(file)

    aero_input = kite_cfg["wing"]["aerodynamics"]
    mass_wing = kite_cfg["wing"].get("mass", 14)
    area_wing = kite_cfg["wing"].get("area", 20)
    mass_kcu = kite_cfg.get("kcu", {}).get("mass", 16)
    mass_kcu = 8.4
    tether_diameter = kite_cfg.get("tether", {}).get("diameter", 0.01)

    tether = RigidLumpedTether(diameter=tether_diameter)
    wind_model = Wind(wind_model="logarithmic", z0=0.1)

    kite = Kite(
        mass_wing=mass_wing,
        area_wing=area_wing,
        aero_input=aero_input,
        mass_kcu=mass_kcu,
        steering_control="asymmetric",
    )

    # Setup drag coefficient functions
    opti = ca.Opti()

    opti_vars = {
        "cd0": opti.variable(),
        "cd_us": opti.variable(),
        "input_depower": opti.variable(),
    }

    kite.set_drag_params(cd0=opti_vars["cd0"], cd_us=opti_vars["cd_us"])
    kite_model = SystemModel(
        dof=3, quasi_steady=True, kite=kite, tether=tether, wind_model=wind_model
    )

    # Calculate reference wind speed
    uf = (
        results.wind_speed_horizontal
        * kite_model.wind.kappa
        / np.log(results.kite_position_z / kite_model.wind.z0)
    )

    # uf = uf[mask]

    # Run inverse solver
    solutions_inverse = []
    loop_start = time.time()
    last_print_idx = -600  # force a print on first iteration
    last_print_wall = time.time()

    batch_len = 10
    error = []
    cd0_fit = []
    cd_us_fit = []
    input_depower_fit = []
    for batch in range(5):
        speed_tangential = opti.variable(batch_len)
        input_steering = opti.variable(batch_len)
        speed_friction = opti.variable(batch_len)
        for i in range(batch_len):
            row_idx = batch * batch_len + i
            # break
            # Measured values (to be used as initial conditions or prescribed values)
            measured_wind_speed = uf[row_idx]
            measured_force = float(flight_data.ground_tether_force[row_idx])

            seed_row = None
            if seed_lookup is not None and row_idx in seed_lookup.index:

                seed_row = seed_lookup.loc[row_idx]
                # print(f"Using seed row for index {row_idx}: {seed_row.to_dict()}")
            # Print progress every ~600 iterations (~1 minute at 0.1 s sampling)
            if (i - last_print_idx) >= 600:
                wall_elapsed = time.time() - last_print_wall
                print(f"\n{'='*60}")
                print(
                    f"Row {row_idx + 1}/{len(flight_data)} at t={flight_data.time[row_idx]:.1f}s | "
                    f"Measured wind (speed_friction)={measured_wind_speed:.2f} m/s, "
                    f"Measured tether force={measured_force:.2f} N | "
                    f"Chunk time: {wall_elapsed:.2f}s"
                )
                last_print_idx = i
                last_print_wall = time.time()

            # Define current state (all known quantities)
            current_state = {
                "distance_radial": distance_radial[row_idx],
                "angle_course": flight_data.kite_course[row_idx],
                "speed_radial": flight_data.tether_reelout_speed[row_idx],
                "angle_azimuth": azimuth[row_idx] - results.wind_direction[row_idx],
                "angle_elevation": flight_data.kite_elevation[row_idx],
                # "speed_friction": measured_wind_speed,  # Will be optimized in inverse
                "timeder_angle_course": measured_course_rate[row_idx],  # Approximate
                "input_depower": flight_data.up[row_idx],
                "tension_tether_ground": measured_force,  # Will be optimized in inverse
                # "measured_speed_tangential": speed_tangential[
                #     i
                # ],  # Will be optimized in inverse
            }
            # Set initial guesses

            residual = create_residual(
                kite_model,
                opti_vars,
                current_state=current_state,
            )
            # INVERSE SOLVE: Prescribe the force, solve for wind speed
            prescribed_force = measured_force
            res_i = residual(
                speed_tangential[i],
                input_steering[i],
                speed_friction[i],
                *opti_vars.values(),
            )
            opti.subject_to(res_i[0] / 1000 == 0)
            opti.subject_to(res_i[1] / 1000 == 0)
            opti.subject_to(res_i[2] / 1000 == 0)

            init_friction = measured_wind_speed
            init_tangential = (
                float(seed_row.get("speed_tangential"))
                if seed_row is not None and pd.notna(seed_row.get("speed_tangential"))
                else measured_speed_tangential[i]
            )
            init_steering = (
                float(seed_row.get("input_steering"))
                if seed_row is not None and pd.notna(seed_row.get("input_steering"))
                else 0.0
            )

            opti.set_initial(speed_friction[i], init_friction)
            opti.set_initial(speed_tangential[i], init_tangential)
            opti.set_initial(input_steering[i], init_steering)

            error.append((speed_tangential[i] - measured_speed_tangential[i]) ** 2)
        rmse = ca.sqrt(ca.sum(ca.vertcat(*error)) / len(error))
        # rmse_speed = ca.sqrt(ca.sumsqr(ca.vertcat(*error)) / len(error))

        opti.set_initial(opti_vars["cd0"], 0.087)
        opti.set_initial(opti_vars["cd_us"], 0.025)
        opti.set_initial(opti_vars["input_depower"], 0)
        # Set constraints
        opti.subject_to(opti_vars["cd0"] >= 0.05)
        opti.subject_to(opti_vars["cd0"] <= 0.15)
        opti.subject_to(opti_vars["cd_us"] >= 0.0)
        opti.subject_to(opti_vars["cd_us"] <= 0.05)
        opti.subject_to(opti_vars["input_depower"] >= -0.2)
        opti.subject_to(opti_vars["input_depower"] <= 0.5)

        opti.subject_to(speed_tangential >= 0.5)
        opti.subject_to(
            speed_tangential <= 40.0
        )  # Arbitrary upper limit for speed_tangential
        opti.subject_to(speed_friction >= 0.3)
        opti.subject_to(
            speed_friction <= 1.5
        )  # Arbitrary upper limit for speed_friction

        delta = 1.0  # tune
        err_vec = ca.vertcat(*error)
        abs_e = ca.fabs(err_vec)
        quad = 0.5 * ca.sumsqr(ca.fmin(abs_e, delta))
        lin = delta * ca.sum1(ca.fmax(abs_e - delta, 0))
        huber = (quad + lin) / err_vec.numel()
        opti.minimize(huber)
        opti.solver(
            "ipopt",
            {
                "ipopt": {
                    "bound_relax_factor": 0,
                    "tol": 1e-6,
                    "acceptable_iter": 3,
                    "acceptable_tol": 1e-5,
                    "constr_viol_tol": 1e-6,
                    "dual_inf_tol": 1e-4,
                    # "hessian_approximation": "limited-memory",
                    # "mu_strategy": "adaptive",
                    "nlp_scaling_method": "gradient-based",
                    # "limited_memory_max_history": 40,  # try 20–50
                    # "limited_memory_update_type": "bfgs",  # (if supported)
                    "max_iter": 200,
                    # "mu_target": 1e-8,
                    "print_level": 0,
                },
            },
        )
        try:
            sol = opti.solve()
            print("\nInverse solve completed.")
            print(f"  Fitted CD0   = {sol.value(opti_vars['cd0']):.5f}")
            print(f"  Fitted CD_us = {sol.value(opti_vars['cd_us']):.5f}")
            print(
                f"  Fitted depower input = {sol.value(opti_vars['input_depower']):.5f}"
            )
            cd0_fit.append(sol.value(opti_vars["cd0"]))
            cd_us_fit.append(sol.value(opti_vars["cd_us"]))
            input_depower_fit.append(sol.value(opti_vars["input_depower"]))
        except RuntimeError as e:
            print(f"Solver failed: {e}")
            print(opti.debug.value(opti_vars["cd0"]))
            print(opti.debug.value(opti_vars["cd_us"]))

    print("\nOverall fitted CD0 and CD_us values:")
    print(f"  CD0   = {np.mean(cd0_fit):.5f} ± {np.std(cd0_fit):.5f}")
    print(f"  CD_us = {np.mean(cd_us_fit):.5f} ± {np.std(cd_us_fit):.5f}")
    print(
        f"  Depower input = {np.mean(input_depower_fit):.5f} ± {np.std(input_depower_fit):.5f}"
    )
    # sol = opti.solve()


if __name__ == "__main__":
    print("Running inverse wind speed solver...")
    run_inverse_validation(cycle_num=[65])
