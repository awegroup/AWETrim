"""
Inverse solver: Given a prescribed tether force, solve for the wind speed.
Uses measured wind speed as initial condition.
"""

import h5py
import pandas as pd
import numpy as np
from awetrim import SystemModel
from awetrim.system.kite import Kite
from awetrim.system.tether import RigidLumpedTether
from awetrim.system.factory import load_aero_input_from_system_config
from awetrim.environment.Wind import Wind
from awetrim.environment.profile_laws import friction_velocity, speed_from_friction_velocity
import casadi as ca
import time
import yaml

from awetrim.utils.config_paths import LEI_V3_SYSTEM_CONFIG
from awetrim.utils.system_config import get_kite, get_tether
import time
from awetrim.utils.defaults import DEFAULT_BOUNDS


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


def setup_inverse_solver(kite_model):
    """
    Setup an inverse solver where we prescribe the tether force
    and solve for wind speed (speed_friction).
    """
    # Define the inverse problem:
    # Unknown variables: [tension_tether_ground, input_steering, speed_friction]
    unknown_vars = [
        "direction_wind",
        "input_steering",
        "speed_friction",  # Changed from speed_tangential to speed_friction
    ]

    solver_options = {
        "ipopt": {
            "print_level": 0,
            "sb": "yes",
            # "tol": 1e-5,
            "max_iter": 100,
        },
        "print_time": False,
    }

    kite_model.setup_qs_solver(unknown_vars, solver_options=solver_options)

    return unknown_vars, solver_options


def solve_inverse(
    kite_model, current_state, measured_vtau, measured_wind_speed, unknown_vars
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
    p = [current_state[name] for name in kite_model._qs_inputs]
    lbx, ubx, lbg, ubg = kite_model.get_boundaries(current_state, unknown_vars)

    # Prescribe the tension: set both lower and upper bounds to the prescribed value
    # tension_idx = unknown_vars.index("tension_tether_ground")
    # lbx[tension_idx] = prescribed_force
    # ubx[tension_idx] = prescribed_force

    # Initial guess: use measured wind speed for speed_friction
    speed_friction_idx = unknown_vars.index("speed_friction")
    qs_guess = np.array(
        [
            current_state["direction_wind"],  # direction_wind
            current_state["input_steering"],  # input_steering
            current_state[
                "speed_friction"
            ],  # speed_friction (initial guess from measurement)
        ]
    )

    sol = kite_model._qs_solver(x0=qs_guess, p=p, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

    return sol


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

    # Preprocess
    position = np.array(
        [results.kite_position_x, results.kite_position_y, results.kite_position_z]
    ).T
    velocity = np.array(
        [results.kite_velocity_x, results.kite_velocity_y, results.kite_velocity_z]
    ).T
    distance_radial = np.linalg.norm(position, axis=1)
    speed_tangential = np.linalg.norm(
        np.cross(position, velocity), axis=1
    ) / np.maximum(distance_radial, 1e-12)
    ned_azimuth = np.arctan2(results.kite_position_y, results.kite_position_x)
    wind_azimuth = flight_data.kite_azimuth.to_numpy()
    assumed_wind_direction = np.mean(ned_azimuth - wind_azimuth)
    # Normalize depower
    flight_data["up"] = (flight_data["up"] - flight_data["up"].min()) / (
        flight_data["up"].max() - flight_data["up"].min()
    )

    # Setup system from system.yml (awesIO format)
    with open(LEI_V3_SYSTEM_CONFIG, "r") as file:
        system_cfg = yaml.safe_load(file)
    kite_cfg = get_kite(system_cfg)
    wing_struct = kite_cfg["wing"]["structure"]
    cs_struct = kite_cfg.get("control_system", {}).get("structure", {})
    tether_struct = get_tether(system_cfg).get("structure", {})

    aero_input = load_aero_input_from_system_config(
        kite_cfg, config_path=LEI_V3_SYSTEM_CONFIG
    )
    mass_wing = wing_struct.get("mass", 14)
    area_wing = wing_struct.get("projected_surface_area", 20)
    mass_kcu = cs_struct.get("mass", 0.0)
    tether_diameter = tether_struct.get("diameter", 0.01)

    tether = RigidLumpedTether(diameter=tether_diameter)
    wind_model = Wind(wind_model="logarithmic", z0=0.1)
    kite = Kite(
        mass_wing=mass_wing,
        area_wing=area_wing,
        aero_input=aero_input,
        mass_kcu=mass_kcu,
        steering_control="asymmetric",
    )
    kite_model = SystemModel(
        dof=3, quasi_steady=True, kite=kite, tether=tether, wind_model=wind_model
    )

    # Setup inverse solver
    unknown_vars, solver_options = setup_inverse_solver(kite_model)

    # Extract functions
    tension_func = kite_model.extract_function("tension_tether_ground")
    cl_func = kite_model.extract_function("lift_coefficient")
    cd_func = kite_model.extract_function("drag_coefficient")
    aoa_func = kite_model.extract_function("angle_of_attack")

    # Calculate reference wind speed
    uf = friction_velocity(
        results.wind_speed_horizontal, results.kite_position_z, kite_model.wind.z0, xp=np
    )

    # Run inverse solver
    solutions_inverse = []
    loop_start = time.time()
    last_print_idx = -600  # force a print on first iteration
    last_print_wall = time.time()
    state_combined = {}
    for i, row in flight_data.iterrows():
        # if i > 600:
        #     break
        # Measured values (to be used as initial conditions or prescribed values)
        measured_wind_speed = uf[i]
        measured_force = float(flight_data.ground_tether_force[i])

        # Print progress every ~600 iterations (~1 minute at 0.1 s sampling)
        if (i - last_print_idx) >= 600:
            wall_elapsed = time.time() - last_print_wall
            print(f"\n{'='*60}")
            print(
                f"Row {i + 1}/{len(flight_data)} at t={row.time:.1f}s | "
                f"Measured wind (speed_friction)={measured_wind_speed:.2f} m/s, "
                f"Measured tether force={measured_force:.2f} N | "
                f"Chunk time: {wall_elapsed:.2f}s"
            )
            last_print_idx = i
            last_print_wall = time.time()

        # Define current state (all known quantities)
        current_state = {
            "distance_radial": distance_radial[i],
            "angle_course": row.kite_course,
            "speed_radial": row.tether_reelout_speed,
            "angle_azimuth": wind_azimuth[i],  # - results.wind_direction[i],
            "angle_elevation": row.kite_elevation,
            "speed_friction": (
                state_combined["speed_friction"]
                if "speed_friction" in state_combined
                else measured_wind_speed
            ),  # Will be optimized in inverse
            "direction_wind": (
                state_combined["direction_wind"]
                if "direction_wind" in state_combined
                else 0.0
            ),  # Will be optimized in inverse
            "input_steering": (
                state_combined["input_steering"]
                if "input_steering" in state_combined
                else 0.0
            ),  # Will be optimized in inverse
            "timeder_angle_course": 0.0,  # Approximate
            "input_depower": row.up,
            "tension_tether_ground": measured_force,  # Will be optimized in inverse
            "speed_tangential": speed_tangential[i],  # Will be optimized in inverse
        }

        # INVERSE SOLVE: Prescribe the force, solve for wind speed
        prescribed_force = measured_force
        sol = solve_inverse(
            kite_model,
            current_state,
            measured_vtau=speed_tangential[i],
            measured_wind_speed=measured_wind_speed,  # Initial guess
            unknown_vars=unknown_vars,
        )

        if np.linalg.norm(sol["g"]) < 1:
            # Extract solution
            solved_direction_wind = float(sol["x"][0])
            solved_steering = float(sol["x"][1])
            solved_wind_speed = float(sol["x"][2])

            # print(f"\nInverse solve successful!")
            # print(f"  Prescribed force: {prescribed_force:.2f} N")
            # print(f"  Solved speed tangential: {solved_vtau:.2f} m/s")
            # print(f"  Measured wind speed: {measured_wind_speed:.2f} m/s")
            # print(f"  Solved wind speed: {solved_wind_speed:.2f} m/s")
            # print(
            #     f"  Wind speed change: {(solved_wind_speed - measured_wind_speed):.2f} m/s"
            # )
            # print(f"  Constraint violation: {np.linalg.norm(sol['g']):.2e}")

            # Update current_state with solved values
            current_state["direction_wind"] = solved_direction_wind
            current_state["input_steering"] = solved_steering
            current_state["speed_friction"] = solved_wind_speed

            # Compute derived quantities
            state_combined = dict(current_state)
            state_combined["direction_wind"] = solved_direction_wind
            state_combined["input_steering"] = solved_steering
            state_combined["speed_friction"] = solved_wind_speed
            z_kite_i = row.kite_elevation * distance_radial[i]
            state_combined["speed_wind"] = speed_from_friction_velocity(
                solved_wind_speed, z_kite_i, kite_model.wind.z0, xp=np
            )
            state_combined["cl"] = float(
                cl_func(*[state_combined[name] for name in cl_func.name_in()])
            )
            state_combined["cd"] = float(
                cd_func(*[state_combined[name] for name in cd_func.name_in()])
            )
            state_combined["aoa"] = float(
                aoa_func(*[state_combined[name] for name in aoa_func.name_in()])
            )
            state_combined["tension_tether_ground"] = prescribed_force
            state_combined["time"] = row.time
            state_combined["index"] = i
            state_combined["measured_wind_speed"] = speed_from_friction_velocity(
                measured_wind_speed, z_kite_i, kite_model.wind.z0, xp=np
            )
            state_combined["measured_force"] = measured_force
            state_combined["measured_speed_tangential"] = speed_tangential[i]
            state_combined["assumed_wind_direction"] = assumed_wind_direction
            state_combined["measured_wind_direction"] = results.wind_direction[i]
            # Skip solutions that violate AoA upper bound
            aoa_upper = DEFAULT_BOUNDS.get("angle_of_attack", [None, None])[1]
            if aoa_upper is not None and state_combined["aoa"] > aoa_upper:
                continue

            solutions_inverse.append(state_combined)

        # else:
        #     print(
        #         f"Inverse solve FAILED! Constraint violation: {np.linalg.norm(sol['g']):.2e}"
        #     )

    elapsed_loop = time.time() - loop_start
    return (
        solutions_inverse,
        flight_data,
        speed_tangential,
        assumed_wind_direction,
        elapsed_loop,
    )


if __name__ == "__main__":
    print("Running inverse wind speed solver...")
    (
        solutions_inverse,
        flight_data,
        speed_tangential,
        assumed_wind_direction,
        elapsed_loop,
    ) = run_inverse_validation(cycle_num=list(range(60, 64)))

    # Convert to DataFrame for easier analysis
    df_inverse = pd.DataFrame(solutions_inverse)

    # Attach measured signals by time alignment (handles skipped inverse rows)
    df_inverse = df_inverse.drop(
        columns=["measured_speed_tangential", "measured_tether_force"],
        errors="ignore",
    )

    meas_df = pd.DataFrame(
        {
            "time": flight_data["time"].to_numpy(),
            "measured_tether_force": flight_data["ground_tether_force"].to_numpy(),
            "measured_speed_tangential": speed_tangential,
        }
    )
    df_inverse = df_inverse.merge(meas_df, on="time", how="left")
    df_inverse["aoa_deg"] = np.rad2deg(df_inverse["aoa"])

    print("\n" + "=" * 80)
    print("INVERSE SOLUTION SUMMARY")
    print("=" * 80)
    print(
        df_inverse[
            [
                "time",
                "measured_wind_speed",
                "speed_friction",
                "measured_force",
                "tension_tether_ground",
                "aoa",
            ]
        ]
    )
    print(f"\nInverse solve loop time: {elapsed_loop:.2f} s")

    # Plot measured vs solved wind speed
    import matplotlib.pyplot as plt

    def rolling_mean(arr, window):
        if len(arr) == 0 or window <= 1:
            return arr
        pad = window // 2
        padded = np.pad(arr, pad_width=pad, mode="reflect")
        kernel = np.ones(window) / float(window)
        sm = np.convolve(padded, kernel, mode="valid")
        # Trim to original length
        excess = len(sm) - len(arr)
        if excess > 0:
            start = excess // 2
            sm = sm[start : start + len(arr)]
        return sm

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    time = df_inverse["time"].values
    solved_dir_deg = np.rad2deg(
        df_inverse.get("direction_wind", pd.Series(np.nan))  # + assumed_wind_direction
    )
    measured_dir_deg = np.rad2deg(df_inverse["measured_wind_direction"].to_numpy())

    # Plot 1: Wind speed at kite height comparison
    axes[0].plot(
        time,
        df_inverse["measured_wind_speed"],
        label="Measured wind speed",
    )

    from scipy.ndimage import gaussian_filter1d

    dt = np.median(np.diff(time)) if len(time) > 1 else 1.0
    window = max(int(round(60.0 / dt)), 1)
    speed_wind_rolling = rolling_mean(
        df_inverse["measured_wind_speed"].to_numpy(), window=10
    )
    axes[0].plot(
        time,
        speed_wind_rolling,
        label="Solved wind at kite height",
    )

    axes[0].set_ylabel("Wind Speed [m/s]", fontsize=12)
    axes[0].set_title(
        "Measured vs Solved Wind Speed at Kite Height", fontsize=13, fontweight="bold"
    )
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=11, loc="upper right")

    # Plot 2: Wind speed difference at kite height
    wind_diff_kite = df_inverse["speed_wind"] - df_inverse["measured_wind_speed"]
    axes[1].plot(time, wind_diff_kite, "g-", linewidth=2, marker="^")
    axes[1].axhline(y=0, color="k", linestyle="--", alpha=0.5)
    axes[1].set_xlabel("Time [s]", fontsize=12)
    axes[1].set_ylabel("Wind Speed Difference [m/s]", fontsize=12)
    axes[1].set_title(
        "Solved - Measured Wind Speed at Kite Height", fontsize=13, fontweight="bold"
    )
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Wind direction (measured vs solved + assumed offset) with 1-minute means

    mean_meas_dir = rolling_mean(measured_dir_deg, window)
    mean_solved_dir = rolling_mean(solved_dir_deg, window)

    axes[2].plot(
        time,
        mean_meas_dir,
        "--",
        color="tab:red",
        label="Measured wind dir (1-min mean)",
    )
    axes[2].plot(
        time,
        mean_solved_dir,
        "-",
        color="tab:orange",
        label="Solved wind dir (offset, 1-min mean)",
    )
    axes[2].set_ylabel("Wind Direction [deg]", fontsize=12)
    axes[2].set_xlabel("Time [s]", fontsize=12)
    axes[2].set_title(
        "Wind Direction: Measured vs Solved", fontsize=13, fontweight="bold"
    )
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=11, loc="upper right")

    plt.tight_layout()
    plt.savefig(
        "results/inverse_wind_speed_comparison.png", dpi=150, bbox_inches="tight"
    )
    print("\nPlot saved to results/inverse_wind_speed_comparison.png")
    plt.show()

    # Additional comparisons: tether force, tangential speed, AoA/CL/CD
    fig2, axes2 = plt.subplots(4, 1, figsize=(12, 12))

    # Tether force
    axes2[0].plot(time, df_inverse["measured_tether_force"], label="Measured")
    axes2[0].plot(time, df_inverse["tension_tether_ground"], label="Solved")
    axes2[0].set_ylabel("Tether Force [N]", fontsize=12)
    axes2[0].set_title(
        "Tether Force: Measured vs Solved", fontsize=13, fontweight="bold"
    )
    axes2[0].grid(True, alpha=0.3)
    axes2[0].legend(fontsize=11)

    # Tangential speed
    axes2[1].plot(time, df_inverse["measured_speed_tangential"], label="Measured")
    axes2[1].plot(time, df_inverse["speed_tangential"], label="Solved")
    axes2[1].set_ylabel("Tangential Speed [m/s]", fontsize=12)
    axes2[1].set_title(
        "Tangential Speed: Measured vs Solved", fontsize=13, fontweight="bold"
    )
    axes2[1].grid(True, alpha=0.3)
    axes2[1].legend(fontsize=11)

    # AoA (deg)
    axes2[2].plot(time, df_inverse["aoa_deg"], label="AoA [deg]")
    axes2[2].set_ylabel("AoA [deg]", fontsize=12)
    axes2[2].set_title("Angle of Attack (model)", fontsize=13, fontweight="bold")
    axes2[2].grid(True, alpha=0.3)
    axes2[2].legend(fontsize=11)

    # CL/CD (model outputs)
    axes2[3].plot(time, df_inverse["cl"], label="CL")
    axes2[3].plot(time, df_inverse["cd"], label="CD")
    axes2[3].set_xlabel("Time [s]", fontsize=12)
    axes2[3].set_ylabel("Coefficient", fontsize=12)
    axes2[3].set_title(
        "Aerodynamic Coefficients (model)", fontsize=13, fontweight="bold"
    )
    axes2[3].grid(True, alpha=0.3)
    axes2[3].legend(fontsize=11)

    plt.tight_layout()
    plt.savefig("results/inverse_extra_comparisons.png", dpi=150, bbox_inches="tight")
    print("Plot saved to results/inverse_extra_comparisons.png")
    plt.show()

    # Save results
    df_inverse.to_csv("results/inverse_wind_speed_validation.csv", index=False)
    print("\nResults saved to results/inverse_wind_speed_validation.csv")
