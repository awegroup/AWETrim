import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from picawe import SystemModel
from picawe.system.kite import Kite
from picawe.system.tether import (
    RigidLumpedTether,
    FlexibleLumpedTether,
    RigidLinkTether,
)
from picawe.environment.Wind import Wind
import casadi as ca
import time
import matplotlib.pyplot as plt
from picawe.utils.color_palette import set_plot_style
from picawe.utils.defaults import PLOT_LABELS
from validation_utils import read_results, read_results_from_hdf5, read_dict_from_group


results, flight_data, config_data = read_results("2023", "11", "27", "v9", addition="")
print(max(flight_data.cycle))
# mask = (flight_data.cycle>10)&(flight_data.cycle<70)
mask = flight_data.cycle.isin(range(10, 163))
mask = flight_data.cycle.isin(range(44, 46))
# mask = flight_data.cycle == 45
# mask = mask & (flight_data.kite_elevation < 0.75)
flight_data = flight_data[mask]
results = results[mask]
results = results.reset_index(drop=True)
flight_data = flight_data.reset_index(drop=True)

csv_file = "./processed_data/VSM_results_alpha_sweep.csv"
v3_polar_data = pd.read_csv(csv_file)

# -----------------------------------------
# Define the system
# -----------------------------------------

import json

# -----------------------------------------------
# Load data and define aerodynamic model
# -----------------------------------------------

# Preprocessing - calculate once for all simulations
position = np.array(
    [results.kite_position_x, results.kite_position_y, results.kite_position_z]
).T
velocity = np.array(
    [results.kite_velocity_x, results.kite_velocity_y, results.kite_velocity_z]
).T
distance_radial = np.linalg.norm(position, axis=1)
speed_tangential = np.linalg.norm(velocity, axis=1)
speed_tangential = np.linalg.norm(np.cross(position, velocity), axis=1) / np.maximum(
    distance_radial, 1e-12
)
flight_data["speed_tangential"] = speed_tangential

flight_data.kite_azimuth = (
    flight_data.kite_azimuth
)  # -0.1            # Calculate misalignment!!! at each cycle


window_size = 5
flight_data["course_rate"] = np.gradient(
    np.unwrap(flight_data["kite_course"]), flight_data["time"]
)

# Make flight_data.up between 0 and 1 by normalizing it
flight_data["up"] = flight_data["up"] - flight_data["up"].min()
flight_data["up"] = flight_data["up"] / flight_data["up"].max()

# Smooth the course rate
flight_data["course_rate"] = (
    flight_data["course_rate"].rolling(window=window_size, min_periods=1).mean()
)

# Get wind heights available in the data
wind_heights = []
for column in flight_data.columns:
    if "m_Wind_Speed_m_s" in column:
        wind_heights.append(int(column.split("m_")[0]))

# Run simulation for both aerodynamic models
aero_files = [
    "./data/LEI-V9-KITE/v9_aero_input.json",
]
aero_labels = ["Variable"]
all_solutions = {}
all_failed_indices = set()  # Global tracking of failed indices

for aero_file, label in zip(aero_files, aero_labels):
    print(f"\nRunning simulation with {aero_file}...")

    with open(aero_file, "r") as file:
        aero_input = json.load(file)

    tether = RigidLumpedTether(diameter=0.001)
    wind_model = Wind(
        wind_model="logarithmic",
        z0=0.01,  # Roughness length
    )
    kite = Kite(
        mass_wing=90,
        area_wing=47,
        aero_input=aero_input,
        mass_kcu=0,
        steering_control="asymmetric",
    )
    kite_model = SystemModel(
        dof=3, quasi_steady=True, kite=kite, tether=tether, wind_model=wind_model
    )

    solutions = []
    start = time.time()

    unknown_vars = [
        "tension_tether_ground",
        "input_steering",
        "speed_tangential",
    ]

    solver_options = {
        "ipopt": {
            "print_level": 0,
            "sb": "yes",
            "max_iter": 200,
        },
        "print_time": False,
    }

    qs_guess = [1e6, 0, 100]
    kite_model.setup_qs_solver(unknown_vars, solver_options=solver_options)

    cl_func = kite_model.extract_function("lift_coefficient")
    cd_func = kite_model.extract_function("drag_coefficient")
    aoa_func = kite_model.extract_function("angle_of_attack")
    tension_func = kite_model.extract_function("tension_tether_ground")
    speed_apparent_wind_func = kite_model.extract_function("speed_apparent_wind")

    count = 0

    # Main loop for this aerodynamic model
    for i, row in flight_data.iterrows():
        kite_height = row.kite_position_z if "kite_position_z" in row else 100

        # Sort the heights just in case
        wind_heights_sorted = sorted(wind_heights)

        # Find the two closest heights below and above the kite
        for j in range(len(wind_heights_sorted) - 1):
            h_low = wind_heights_sorted[j]
            h_high = wind_heights_sorted[j + 1]
            if h_low <= kite_height <= h_high:
                break
        else:
            # Handle extrapolation if kite is outside the available height range
            if kite_height < wind_heights_sorted[0]:
                h_low = h_high = wind_heights_sorted[0]
            else:
                h_low = h_high = wind_heights_sorted[-1]

        # Retrieve the wind speeds at those heights
        ws_low = row[f"{h_low}m_Wind_Speed_m_s"]
        ws_high = row[f"{h_high}m_Wind_Speed_m_s"]

        # Linear interpolation
        if h_low == h_high:
            wind_speed = ws_low  # No interpolation needed
        else:
            wind_speed = ws_low + (ws_high - ws_low) * (kite_height - h_low) / (
                h_high - h_low
            )
        # Apply the log-law formula
        uf = (
            wind_speed
            * kite_model.wind.kappa
            / np.log(kite_height / kite_model.wind.z0)
        )

        current_state = {
            "distance_radial": distance_radial[i],
            "angle_course": row.kite_course,
            "speed_radial": row.tether_reelout_speed,
            "angle_azimuth": row.kite_azimuth,
            "angle_elevation": row.kite_elevation,
            "speed_friction": uf,
            "timeder_angle_course": row.course_rate,
            "input_depower": row.up,
        }

        p = [current_state[name] for name in kite_model._qs_inputs]
        lbx, ubx, lbg, ubg = kite_model.get_boundaries(current_state, unknown_vars)
        sol = kite_model._qs_solver(
            x0=qs_guess, p=p, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg
        )

        if np.linalg.norm(sol["g"]) < 1:
            qs_guess = sol["x"]
            qs_state = {name: float(qs_guess[i]) for i, name in enumerate(unknown_vars)}
            state_combined = {**qs_state, **current_state}
            state_combined["lift_coefficient"] = float(
                cl_func(*[state_combined[name] for name in cl_func.name_in()])
            )
            state_combined["drag_coefficient"] = float(
                cd_func(*[state_combined[name] for name in cd_func.name_in()])
            )
            state_combined["angle_of_attack"] = float(
                aoa_func(*[state_combined[name] for name in aoa_func.name_in()])
            )
            state_combined["tension_tether_ground"] = float(
                tension_func(*[state_combined[name] for name in tension_func.name_in()])
            )
            state_combined["speed_apparent_wind"] = float(
                speed_apparent_wind_func(
                    *[
                        state_combined[name]
                        for name in speed_apparent_wind_func.name_in()
                    ]
                )
            )
            state_combined["time"] = row.time
            state_combined["original_index"] = i  # Include the original index

            solutions.append(state_combined)
        else:
            # print("Angle of attack not found, index:", i)
            # print(state_combined["angle_of_attack"] * 180 / np.pi)
            qs_guess[0] = 1e10
            qs_guess[2] = 50
            print("Quasi steady solution not found, index:", i)
            all_failed_indices.add(i)  # Track the failed index globally
            count += 1

    print(f"Number of failed solutions: {count}/ {len(flight_data)}")
    end = time.time()
    print(f"Time taken: {end - start} seconds for {len(flight_data)} iterations")

    # Store solutions for this aerodynamic model
    solutions_df = pd.DataFrame(solutions)
    solutions_df = solutions_df[solutions_df["tension_tether_ground"].notna()]

    all_solutions[label] = solutions_df

# Print total failed indices for debugging
print(f"\nTotal failed indices across all models: {len(all_failed_indices)}")
print(f"Failed indices: {sorted(list(all_failed_indices))[:10]}...")  # Show first 10

# Remove failed indices from flight_data and results after all models are processed
if all_failed_indices:
    print(
        f"\nRemoving {len(all_failed_indices)} failed rows from flight_data and results"
    )

    # Convert to list and sort
    indices_to_remove = sorted(list(all_failed_indices))

    # Create a mask for valid indices (those that are NOT in failed indices)
    valid_mask = ~flight_data.index.isin(indices_to_remove)

    # Filter flight_data and results using the mask
    flight_data = flight_data[valid_mask].reset_index(drop=True)
    results = results[valid_mask].reset_index(drop=True)  # Also filter results

    # Update related arrays using the same mask
    position = position[valid_mask]
    velocity = velocity[valid_mask]
    distance_radial = distance_radial[valid_mask]
    speed_tangential = speed_tangential[valid_mask]

    print(f"Updated flight_data length: {len(flight_data)}")
    print(f"Updated results length: {len(results)}")

# Ensure all solution DataFrames have matching length and index as filtered data
for label, solutions_df in all_solutions.items():
    if not solutions_df.empty:
        # Remove the original_index column if it exists
        if "original_index" in solutions_df.columns:
            solutions_df = solutions_df.drop(columns=["original_index"])

        # Reset index to match flight_data indexing (0, 1, 2, ...)
        solutions_df_reset = solutions_df.reset_index(drop=True)
        all_solutions[label] = solutions_df_reset

        print(f"Solutions DataFrame for {label} length: {len(solutions_df_reset)}")
        print(f"Flight data length: {len(flight_data)}")

        # Verify they match
        if len(solutions_df_reset) != len(flight_data):
            print(
                f"WARNING: Length mismatch for {label}! Solutions: {len(solutions_df_reset)}, Flight data: {len(flight_data)}"
            )
            print(
                f"Difference: {len(flight_data) - len(solutions_df_reset)} missing solutions"
            )
            print(
                f"Expected same length as flight_data after filtering: {len(flight_data)}"
            )

            # Let's check if this is due to the filtering process
            expected_solutions = len(flight_data)
            actual_solutions = len(solutions_df_reset)
            missing_solutions = expected_solutions - actual_solutions
            print(f"Missing {missing_solutions} solutions somewhere in the process")

# Print comparison results for both models
for label, solutions_df in all_solutions.items():
    print(f"\n=== Results for {label} aerodynamic model ===")

    CD = (
        results["wing_drag_coefficient"]
        + results["bridles_drag_coefficient"]
        + results["kcu_drag_coefficient"]
    )

    # Initialize empty masks that will accumulate indices from all cycles
    mask_pow = pd.Series([False] * len(flight_data), index=flight_data.index)
    mask_dep = pd.Series([False] * len(flight_data), index=flight_data.index)
    mask_trans = pd.Series([False] * len(flight_data), index=flight_data.index)

    # Process each cycle individually
    for i_cycle in flight_data.cycle.unique():
        print(f"\nProcessing cycle {i_cycle}")
        mask_cycle = flight_data.cycle == i_cycle
        cycle_indices = flight_data[mask_cycle].index

        # Get data for this cycle only
        cycle_speed = speed_tangential[mask_cycle]
        cycle_flight_data = flight_data[mask_cycle]

        # Apply the "after first peak" condition using global indices
        cycle_mask_pow = cycle_flight_data.flight_phase_index == 1

        # Depowered phase for this cycle
        cycle_mask_dep = cycle_flight_data.flight_phase_index == 3

        # Transition phase for this cycle: not powered and not depowered
        cycle_mask_trans = cycle_flight_data.flight_phase_index.isin([2, 4])

        # Add this cycle's masks to the global masks
        mask_pow[cycle_indices[cycle_mask_pow]] = True
        mask_dep[cycle_indices[cycle_mask_dep]] = True
        mask_trans[cycle_indices[cycle_mask_trans]] = True

    print(
        f"\nTotal across all cycles: Powered {mask_pow.sum()}, Depowered {mask_dep.sum()}, Transition {mask_trans.sum()} samples"
    )

    # Print stats for transition region
    print("--- Transition region stats ---")
    print(
        "Mean CL transition, exp. data: ",
        np.mean(results["wing_lift_coefficient"][mask_trans]),
    )
    print("Mean CD transition, exp. data: ", np.mean(CD[mask_trans]))
    print(
        "Mean CL transition,  kcu: ",
        np.mean(solutions_df["lift_coefficient"][mask_trans]),
    )
    print(
        "Mean CD transition,  kcu: ",
        np.mean(solutions_df["drag_coefficient"][mask_trans]),
    )
    print(
        "Mean aoa transition, exp. data: ",
        np.mean(results["wing_angle_of_attack_bridle"][mask_trans]),
    )
    print(
        "Mean aoa transition,  kcu: ",
        np.mean(solutions_df["angle_of_attack"][mask_trans]) * 180 / np.pi,
    )
    print(
        "Mean CL powered, exp. data: ",
        np.mean(results["wing_lift_coefficient"][mask_pow]),
    )
    print("Mean CD powered, exp. data: ", np.mean(CD[mask_pow]))
    print(
        "Mean CL powered,  kcu: ", np.mean(solutions_df["lift_coefficient"][mask_pow])
    )
    print(
        "Mean CD powered,  kcu: ", np.mean(solutions_df["drag_coefficient"][mask_pow])
    )
    print(
        "Mean CL depowered, exp. data: ",
        np.mean(results["wing_lift_coefficient"][~mask_pow]),
    )
    print("Mean CD depowered, exp. data: ", np.mean(CD[~mask_pow]))
    print(
        "Mean CL depowered,  kcu: ",
        np.mean(solutions_df["lift_coefficient"][~mask_pow]),
    )
    print(
        "Mean CD depowered,  kcu: ",
        np.mean(solutions_df["drag_coefficient"][~mask_pow]),
    )

    print("Mean aoa, exp. data: ", np.mean(results["wing_angle_of_attack_bridle"]))
    print(
        "Mean aoa powered,  kcu: ",
        np.mean(solutions_df["angle_of_attack"][mask_pow]) * 180 / np.pi,
    )
    print(
        "Mean aoa depowered,  kcu: ",
        np.mean(solutions_df["angle_of_attack"][~mask_pow]) * 180 / np.pi,
    )

    total_power = (
        sum(
            solutions_df["tension_tether_ground"][mask_pow]
            * solutions_df["speed_radial"][mask_pow]
            * flight_data["time"][mask_pow].diff().fillna(0).values
        )
        / flight_data["time"][mask_pow].diff().fillna(0).sum()
    )
    measured_power = (
        sum(
            flight_data["ground_tether_force"][mask_pow]
            * flight_data["tether_reelout_speed"][mask_pow]
            * flight_data["time"][mask_pow].diff().fillna(0).values
        )
        / flight_data["time"][mask_pow].diff().fillna(0).sum()
    )
    print("Estimated power KCU reelout: ", total_power, "W")
    print("Measured power reelout: ", measured_power, "W")

    total_power_dep = (
        sum(
            solutions_df["tension_tether_ground"][mask_dep]
            * solutions_df["speed_radial"][mask_dep]
            * flight_data["time"][mask_dep].diff().fillna(0).values
        )
        / flight_data["time"][mask_dep].diff().fillna(0).sum()
    )
    measured_power_dep = (
        sum(
            flight_data["ground_tether_force"][mask_dep]
            * flight_data["tether_reelout_speed"][mask_dep]
            * flight_data["time"][mask_dep].diff().fillna(0).values
        )
        / flight_data["time"][mask_dep].diff().fillna(0).sum()
    )
    print("Estimated power KCU depower: ", total_power_dep, "W")
    print("Measured power depower: ", measured_power_dep, "W")

    # Transition power calculation
    total_power_trans = (
        sum(
            solutions_df["tension_tether_ground"][mask_trans]
            * solutions_df["speed_radial"][mask_trans]
            * flight_data["time"][mask_trans].diff().fillna(0).values
        )
        / flight_data["time"][mask_trans].diff().fillna(0).sum()
    )
    measured_power_trans = (
        sum(
            flight_data["ground_tether_force"][mask_trans]
            * flight_data["tether_reelout_speed"][mask_trans]
            * flight_data["time"][mask_trans].diff().fillna(0).values
        )
        / flight_data["time"][mask_trans].diff().fillna(0).sum()
    )
    print("Estimated power KCU transition: ", total_power_trans, "W")
    print("Measured power transition: ", measured_power_trans, "W")

    # =============================================================================
    # SIMPLIFIED CYCLE-BASED ANALYSIS
    # =============================================================================

    def compute_cycle_phase_averages():
        """Compute mean tether force and tangential speed per cycle and phase."""
        cycle_data = []

        for i_cycle in flight_data.cycle.unique():
            mask_cycle = flight_data.cycle == i_cycle

            if mask_cycle.sum() < 5:  # Skip cycles with too few data points
                continue

            # Get phase masks for this specific cycle
            cycle_mask_pow = mask_pow[mask_cycle]
            cycle_mask_dep = mask_dep[mask_cycle]
            cycle_mask_trans = mask_trans[mask_cycle]

            # Process each phase separately within this cycle
            phases_to_process = [
                ("Powered", cycle_mask_pow),
                ("Depowered", cycle_mask_dep),
                ("Transition", cycle_mask_trans),
            ]

            for phase_name, phase_mask in phases_to_process:
                if (
                    phase_mask.sum() >= 2
                ):  # Need at least 2 points for meaningful average
                    # Get indices for this phase within this cycle
                    valid_phase_indices = flight_data[mask_cycle].index[phase_mask]

                    measured_force = flight_data.loc[
                        valid_phase_indices, "ground_tether_force"
                    ]
                    predicted_force = solutions_df.loc[
                        valid_phase_indices, "tension_tether_ground"
                    ]

                    mean_measured_force = np.mean(
                        flight_data.loc[valid_phase_indices, "ground_tether_force"]
                    )
                    mean_predicted_force = np.mean(
                        solutions_df.loc[valid_phase_indices, "tension_tether_ground"]
                    )
                    rmse_force = np.sqrt(
                        np.mean((predicted_force - measured_force) ** 2)
                    )
                    measured_speed = np.mean(speed_tangential[valid_phase_indices])
                    predicted_speed = np.mean(
                        solutions_df.loc[valid_phase_indices, "speed_tangential"]
                    )
                    cycle_info = {
                        "cycle": i_cycle,
                        "phase": phase_name,
                        "n_points": len(valid_phase_indices),
                        "measured_tether_force": mean_measured_force,
                        "predicted_tether_force": mean_predicted_force,
                        "median_tether_force": np.median(mean_measured_force),
                        "measured_speed_tangential": measured_speed,
                        "predicted_speed_tangential": predicted_speed,
                        "tether_force_error": mean_predicted_force
                        - mean_measured_force,
                        "tether_force_relative_error": (
                            (mean_predicted_force - mean_measured_force)
                            / mean_measured_force
                            * 100
                            if mean_measured_force != 0
                            else 0
                        ),
                        "speed_error": predicted_speed - measured_speed,
                        "speed_relative_error": (
                            (predicted_speed - measured_speed) / measured_speed * 100
                            if measured_speed != 0
                            else 0
                        ),
                        "median_speed_tangential": np.median(measured_speed),
                    }
                    cycle_data.append(cycle_info)

        return pd.DataFrame(cycle_data)

    def compute_cycle_averages():
        """Compute mean tether force and tangential speed per cycle."""
        cycle_data = []

        for i_cycle in flight_data.cycle.unique():
            mask_cycle = flight_data.cycle == i_cycle

            measured_energy = np.sum(
                flight_data.loc[mask_cycle, "ground_tether_force"]
                * flight_data.loc[mask_cycle, "tether_reelout_speed"]
                * flight_data.loc[mask_cycle, "time"].diff().fillna(0).values
            )
            predicted_energy = np.sum(
                solutions_df.loc[mask_cycle, "tension_tether_ground"]
                * solutions_df.loc[mask_cycle, "speed_radial"]
                * flight_data.loc[mask_cycle, "time"].diff().fillna(0).values
            )
            measured_power = (
                measured_energy
                / flight_data.loc[mask_cycle, "time"].diff().fillna(0).sum()
            )
            predicted_power = (
                predicted_energy
                / flight_data.loc[mask_cycle, "time"].diff().fillna(0).sum()
            )
            power_relative_error = (
                (predicted_power - measured_power) / measured_power * 100
                if measured_power != 0
                else 0
            )
            cycle_info = {
                "cycle": i_cycle,
                "n_points": mask_cycle.sum(),
                "measured_energy": measured_energy,
                "predicted_energy": predicted_energy,
                "energy_error": predicted_energy - measured_energy,
                "measured_power": measured_power,
                "predicted_power": predicted_power,
                "energy_relative_error": (
                    (predicted_energy - measured_energy) / measured_energy * 100
                    if measured_energy != 0
                    else 0
                ),
                "power_relative_error": power_relative_error,
            }
            cycle_data.append(cycle_info)

        return pd.DataFrame(cycle_data)

    print(f"\n" + "=" * 60)
    print(f"CYCLE-BASED ANALYSIS FOR {label.upper()} MODEL")
    print("=" * 60)

    cycle_phase_df = compute_cycle_phase_averages()
    cycle_df = compute_cycle_averages()
    print(cycle_df)
    print(
        f"  Median Power Relative Error: {np.median(cycle_df['power_relative_error']):.1f} %"
    )
    print(
        f" Power Relative Error IQR: {np.percentile(cycle_df['power_relative_error'], 75) - np.percentile(cycle_df['power_relative_error'], 25):.1f} %"
    )
    print(
        f"  Median Power Absolute Error: {np.median(abs(cycle_df['power_relative_error'])):.1f} %"
    )
    print(
        f" Power Absolute Error IQR: {np.percentile(abs(cycle_df['power_relative_error']), 75) - np.percentile(abs(cycle_df['power_relative_error']), 25):.1f} %"
    )

    print(f"\nComputed averages for {len(cycle_phase_df)} cycle-phase combinations")

    # Print summary by phase
    for phase in ["Powered", "Depowered", "Transition"]:
        phase_data = cycle_phase_df[cycle_phase_df["phase"] == phase]
        if len(phase_data) > 0:
            print(f"\n{phase} Phase:")
            print(f"  Cycles: {len(phase_data)}")
            print(
                f"  Median Tether Force Error: {np.median(phase_data['tether_force_error']):.1f} N"
            )
            print(
                f"  Tether Force IQR: {np.percentile(phase_data['tether_force_error'], 75) - np.percentile(phase_data['tether_force_error'], 25):.1f} N"
            )
            print(
                f"  Median Tether Force Relative Error: {np.median(phase_data['tether_force_relative_error']):.1f} %"
            )
            print(
                f"  Tether Force Relative Error IQR: {np.percentile(phase_data['tether_force_relative_error'], 75) - np.percentile(phase_data['tether_force_relative_error'], 25):.1f} %"
            )
            print(
                f"  Median Tether Force Absolute Error: {np.median(abs(phase_data['tether_force_relative_error'])):.1f} %"
            )
            print(
                f"  Tether Force Absolute Error IQR: {np.percentile(abs(phase_data['tether_force_relative_error']), 75) - np.percentile(abs(phase_data['tether_force_relative_error']), 25):.1f} %"
            )
            print(
                f"  Median Speed Relative Error: {np.median(phase_data['speed_relative_error']):.1f} %"
            )
            print(
                f"  Speed Relative Error IQR: {np.percentile(phase_data['speed_relative_error'], 75) - np.percentile(phase_data['speed_relative_error'], 25):.1f} %"
            )
            print(
                f"  Median Speed Absolute Error: {np.median(abs(phase_data['speed_relative_error'])):.1f} %"
            )
            print(
                f"  Speed Absolute Error IQR: {np.percentile(abs(phase_data['speed_relative_error']), 75) - np.percentile(abs(phase_data['speed_relative_error']), 25):.1f} %"
            )

    # =============================================================================
    # SIMPLIFIED ERROR BOXPLOT
    # =============================================================================

    def plot_error_boxplots(cycle_phase_df, label):
        """Create simple boxplots showing errors by phase."""

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        # Colors for phases
        colors_phase = {
            "Powered": "#4CAF50",
            "Depowered": "#FF9800",
            "Transition": "#2196F3",
        }

        # Tether force error boxplot
        tether_data = []
        speed_data = []
        phase_labels = []

        for phase in ["Powered", "Depowered", "Transition"]:
            phase_data = cycle_phase_df[cycle_phase_df["phase"] == phase]
            if len(phase_data) > 0:
                tether_data.append(phase_data["tether_force_relative_error"].values)
                speed_data.append(phase_data["speed_relative_error"].values)
                phase_labels.append(phase)

        if tether_data:
            # Tether force errors
            bp1 = ax1.boxplot(tether_data, labels=phase_labels, patch_artist=True)
            for patch, phase in zip(bp1["boxes"], phase_labels):
                patch.set_facecolor(colors_phase[phase])
                patch.set_alpha(0.7)

            ax1.axhline(y=0, color="r", linestyle="--", alpha=0.5)
            ax1.set_ylabel("Tether Force Relative Error (%)")
            ax1.set_title("Tether Force Error by Phase")
            ax1.grid(True, alpha=0.3)

            # Speed errors
            bp2 = ax2.boxplot(speed_data, labels=phase_labels, patch_artist=True)
            for patch, phase in zip(bp2["boxes"], phase_labels):
                patch.set_facecolor(colors_phase[phase])
                patch.set_alpha(0.7)

            ax2.axhline(y=0, color="r", linestyle="--", alpha=0.5)
            ax2.set_ylabel("Tangential Speed Relative Error (m/s)")
            ax2.set_title("Tangential Speed Error by Phase")
            ax2.grid(True, alpha=0.3)

        plt.suptitle(
            f"Validation Errors by Phase - {label} Model",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()

        # Save plot
        save_path = f"./results/figures/validation_errors_{label.lower()}.pdf"
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"\nError boxplot saved to: {save_path}")
        plt.show()

        return fig

    # Generate the simplified plot
    plot_error_boxplots(cycle_phase_df, label)

    # Save cycle data to CSV for future analysis
    csv_path = f"./results/cycle_data_{label.lower()}.csv"
    cycle_phase_df.to_csv(csv_path, index=False)
    print(f"Cycle data saved to: {csv_path}")

set_plot_style()


def plot_main_results_comparison(
    all_solutions,
    flight_data,
    speed_tangential,
    PLOT_LABELS,
    save_folder="./results/figures/translational_paper/",
    show=True,
):
    import matplotlib.pyplot as plt
    from picawe.utils.color_palette import get_color_list

    plt.figure()
    label = "Variable"
    plt.plot(
        all_solutions[label]["speed_apparent_wind"]
        * all_solutions[label]["input_steering"],
        all_solutions[label]["timeder_angle_course"],
        ".",
    )
    for cols in flight_data.columns:
        if "apparent" in cols:
            print(cols)
    plt.plot(
        flight_data["kite_apparent_windspeed"]
        * -flight_data["kcu_actual_steering"]
        / max(flight_data["kcu_actual_steering"]),
        flight_data["course_rate"],
        ".",
    )
    plt.show()
    colors = get_color_list()
    # Define phase masks and colors
    phase_masks = [mask_pow, mask_dep, mask_trans]
    phase_names = ["Reel-out", "Reel-in", "Transition"]
    phase_colors = ["#4CAF50", "#FF9800", "#2196F3"]

    # Figure 1: Dynamics (left panels)
    fig1 = plt.figure(figsize=(5, 6))
    gs1 = fig1.add_gridspec(3, 1, height_ratios=[1, 1, 1])

    ax3 = fig1.add_subplot(gs1[0, 0])
    ax4 = fig1.add_subplot(gs1[1, 0])
    ax5 = fig1.add_subplot(gs1[2, 0])

    ax3.set_ylabel(PLOT_LABELS["speed_tangential"])
    ax4.set_ylabel("$F_{t,g}$ (kN)")
    ax5.set_ylabel(PLOT_LABELS["input_steering"])
    ax5.set_xlabel("Time (s)")

    # Plot measured data first
    ax3.plot(
        flight_data["time"],
        speed_tangential,
        label="Measured",
        color=colors[0],
    )
    ax4.plot(
        flight_data["time"],
        flight_data["ground_tether_force"] / 1000,
        color=colors[0],
    )
    ax5.plot(
        flight_data["time"],
        flight_data["kcu_actual_steering"] / max(flight_data["kcu_actual_steering"]),
        color=colors[0],
    )

    # Plot results for aerodynamic models
    for i, (label, solutions_df) in enumerate(all_solutions.items()):
        color = colors[i + 1]

        ax3.plot(
            solutions_df["time"],
            solutions_df["speed_tangential"],
            label=f"QS {label}",
            color=color,
        )
        ax4.plot(
            solutions_df["time"],
            solutions_df["tension_tether_ground"] / 1000,
            color=color,
        )
        ax5.plot(
            solutions_df["time"],
            -solutions_df["input_steering"],
            color=color,
        )
    axs = [ax3, ax4, ax5]
    # Add phase shading to all subplots
    for ax in axs:
        for mask, phase, color in zip(phase_masks, phase_names, phase_colors):
            # Find contiguous regions for shading
            mask_arr = np.array(mask)
            if mask_arr.any():
                # Find start and end indices of contiguous True regions
                idx = np.where(mask_arr)[0]
                if len(idx) > 0:
                    # Group contiguous indices
                    from itertools import groupby
                    from operator import itemgetter

                    for k, g in groupby(enumerate(idx), lambda x: x[0] - x[1]):
                        group = list(map(itemgetter(1), g))
                        start = group[0]
                        end = group[-1]
                        ax.axvspan(
                            flight_data["time"].iloc[start],
                            flight_data["time"].iloc[end],
                            color=color,
                            alpha=0.15,
                        )
    # Add legend for phase colors to ax4
    import matplotlib.patches as mpatches

    phase_patches = [
        mpatches.Patch(color=color, alpha=0.15, label=phase)
        for phase, color in zip(phase_names, phase_colors)
    ]
    ax4.legend(handles=phase_patches, loc="best", frameon=True)
    ax3.legend(loc="best", frameon=True)
    ax3.set_ylim(0, 40)
    ax4.set_ylim(0, 40)
    for ax in axs:
        ax.set_xlim(flight_data["time"].min(), flight_data["time"].max())

    plt.tight_layout()
    plt.savefig(save_folder + "validation_v9_dynamics.pdf", bbox_inches="tight")
    if show:
        plt.show()

    # Figure 2: Aerodynamics (right panels)
    fig2 = plt.figure(figsize=(8, 6))
    gs2 = fig2.add_gridspec(3, 1, height_ratios=[1, 1, 1])

    ax6 = fig2.add_subplot(gs2[0, 0])
    ax7 = fig2.add_subplot(gs2[1, 0])
    ax8 = fig2.add_subplot(gs2[2, 0])

    ax6.set_ylabel("$C_L$")
    ax7.set_ylabel("$C_D$")
    ax8.set_ylabel("AoA (deg)")
    ax8.set_xlabel("Time (s)")

    # Plot measured aerodynamic data
    ax6.plot(
        flight_data["time"],
        results["wing_lift_coefficient"],
        label="Measured",
        color=colors[0],
    )
    ax7.plot(
        flight_data["time"],
        results["wing_drag_coefficient"]
        + results["kcu_drag_coefficient"]
        + results["bridles_drag_coefficient"],
        label="Measured",
        color=colors[0],
    )
    ax8.plot(
        flight_data["time"],
        results["wing_angle_of_attack_bridle"],
        label="Measured",
        color=colors[0],
    )

    # Plot aerodynamic coefficients for each model
    for i, (label, solutions_df) in enumerate(all_solutions.items()):
        color = colors[i + 1]

        ax6.plot(
            solutions_df["time"],
            solutions_df["lift_coefficient"],
            label=f"QS {label}",
            color=color,
        )
        ax7.plot(
            solutions_df["time"],
            solutions_df["drag_coefficient"],
            label=f"QS {label}",
            color=color,
        )
        ax8.plot(
            solutions_df["time"],
            np.degrees(solutions_df["angle_of_attack"]),
            label=f"QS {label}",
            color=color,
        )

    # Add legends and grid for aerodynamics figure
    for ax in [ax6, ax7, ax8]:
        ax.legend(loc="best", fontsize="small")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_folder + "validation_v9_aerodynamics.pdf", bbox_inches="tight")
    if show:
        plt.show()


# --- Plotting section ---
plot_main_results_comparison(
    all_solutions,
    flight_data,
    speed_tangential,
    PLOT_LABELS,
    save_folder="./results/figures/translational_paper/",
    show=True,
)


plt.figure()
plt.plot(
    flight_data["time"],
    flight_data["up"],
    label="Measured Up",
)
plt.show()
