import h5py
import pandas as pd
import numpy as np
from awetrim import SystemModel
from awetrim.system.kite import Kite
from awetrim.system.tether import (
    RigidLumpedTether,
    FlexibleLumpedTether,
    RigidLinkTether,
)
from awetrim.system.factory import load_aero_input_from_system_config
from awetrim.system.williams_tether import WilliamsTether
from awetrim.environment.Wind import Wind
import casadi as ca
import time
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import yaml

# Validation reproduces a specific flight, so use the as-flown KCU mass.
from awetrim.utils.config_paths import LEI_V3_SYSTEM_FLOWN_CONFIG

RUN_WILLIAMS = False
RUN_RIGID = True
SHOW_PLOTS = True

if not SHOW_PLOTS:
    plt.switch_backend("Agg")


def build_williams_validation_solver(kite_model):
    tether = kite_model.tether
    if not isinstance(tether, WilliamsTether):
        raise TypeError(f"Expected WilliamsTether, got {type(tether).__name__}")

    # Williams formulation: solve force balance (including tether force) and
    # ground closure simultaneously with free tether-end angles and tension.
    # The tether reads wind/rho/g/omega off ``kite_model`` via the
    # ``_for(model)`` API; nothing is stored on the tether instance.
    x = ca.vertcat(
        kite_model.speed_tangential,
        kite_model.input_steering,
        tether.tension_tether_kite,
        tether.tether_length,
        tether.azimuth_last_element,
        tether.elevation_last_element,
    )
    p = ca.vertcat(
        kite_model.distance_radial,
        kite_model.angle_course,
        kite_model.speed_radial,
        kite_model.angle_azimuth,
        kite_model.angle_elevation,
        kite_model.wind.speed_friction,
        kite_model.timeder_angle_course,
        kite_model.input_depower,
    )

    tether_shape = tether.tether_shape_symbolic_for(kite_model)
    tether_force_ground = tether_shape["tether_force_ground"]
    tension_ground = ca.norm_2(tether_force_ground)
    ground_residual = tether_shape["ground_position"]
    force_balance = kite_model.force_residual
    residual = ca.vertcat(force_balance, ground_residual)

    outputs = ca.vertcat(
        force_balance,
        tension_ground,
        ground_residual,
        tether.tension_tether_kite,
        tether.elevation_last_element,
        tether.azimuth_last_element,
        kite_model.kite.lift_coefficient_for(kite_model),
        kite_model.kite.drag_coefficient_for(kite_model),
        kite_model.kite.angle_of_attack_for(kite_model),
        kite_model.speed_apparent_wind,
        kite_model.kite.pitch_bridle_for(kite_model),
        kite_model.angle_pitch_aerodynamic,
        kite_model.kite.angle_roll_aerodynamic_for(kite_model),
        kite_model.input_steering,
        kite_model.kite.roll_bridle_for(kite_model),
    )
    output_fun = ca.Function(
        "williams_validation_outputs",
        [x, p],
        [outputs],
        ["x", "p"],
        ["outputs"],
    )
    nlp = {"x": x, "p": p, "f": 0, "g": residual}
    solver = ca.nlpsol(
        "williams_validation_solver",
        "ipopt",
        nlp,
        {
            "ipopt": {
                "print_level": 0,
                "sb": "yes",
                "max_iter": 400,
                "tol": 1e-8,
                "constr_viol_tol": 1e-8,
            },
            "print_time": False,
        },
    )
    return solver, output_fun


def read_results(year, month, day, kite_model, addition="", path_to_main=""):
    path = ""
    date = str(year) + "-" + str(month) + "-" + str(day)
    file_name = str(kite_model) + "_" + date
    hdf5_path = path_to_main + path + file_name + addition + ".h5"
    ekf_output_df, flight_data_df, config_data = read_results_from_hdf5(hdf5_path)
    return ekf_output_df, flight_data_df, config_data


def read_results_from_hdf5(hdf5_path):
    with h5py.File(hdf5_path, "r") as hf:
        # Read the ekf_output_df DataFrame
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

        # Read the flight_data DataFrame
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

        # Read config_data
        config_group = hf["config_data"]
        config_data = read_dict_from_group(config_group)

    return ekf_output_df, flight_data_df, config_data


def read_dict_from_group(group):
    config_dict = {}
    for key, value in group.attrs.items():
        if isinstance(value, bytes):
            value = value.decode("utf-8")  # Decode byte strings back to regular strings
        config_dict[key] = value

    for subgroup_name in group:
        subgroup = group[subgroup_name]
        config_dict[subgroup_name] = read_dict_from_group(subgroup)

    return config_dict


results, flight_data, config_data = read_results(
    "2019",
    "10",
    "08",
    "LEI-V3-Kite",
    addition="",
    path_to_main="./results/LEI-V3-KITE/ekf/",
)
print(max(flight_data.cycle))
# mask = (flight_data.cycle>10)&(flight_data.cycle<70)
mask = flight_data.cycle.isin(range(10, 120))
# mask = flight_data.cycle.isin(range(64, 68))
mask = flight_data.cycle == 60
# mask = mask & (flight_data.kite_elevation < 0.75)
flight_data = flight_data[mask]
results = results[mask]
results = results.reset_index(drop=True)
flight_data = flight_data.reset_index(drop=True)

# results = results.iloc[0:800].reset_index(drop=True)
# flight_data = flight_data.iloc[0:800].reset_index(drop=True)

# csv_file = "./processed_data/VSM_results_alpha_sweep.csv"
# v3_polar_data = pd.read_csv(csv_file)

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

azimuth = np.arctan2(results.kite_position_y, results.kite_position_x)


# Normalize up between 0 and 1 which now is between 0.08 and 0.8
flight_data["up"] = (flight_data["up"] - flight_data["up"].min()) / (
    flight_data["up"].max() - flight_data["up"].min()
)

course_rate = np.gradient(np.unwrap(flight_data.kite_course), flight_data.time)
course_rate = gaussian_filter1d(course_rate, sigma=1)
plt.plot(flight_data.time, course_rate)
if SHOW_PLOTS:
    plt.show()
else:
    plt.close()
flight_data["course_rate"] = course_rate
# Run simulation for aerodynamic model defined in system.yml (awesIO format)
with open(LEI_V3_SYSTEM_FLOWN_CONFIG, "r") as file:
    kite_cfg = yaml.safe_load(file)

aero_labels = []
aero_cfgs = []
if RUN_RIGID:
    aero_labels.append("Variable")
    aero_cfgs.append(kite_cfg)
if RUN_WILLIAMS:
    aero_labels.append("Williams")
    aero_cfgs.append(kite_cfg)
all_solutions = {}

for cfg, label in zip(aero_cfgs, aero_labels):
    print(f"\nRunning simulation with {label} aerodynamic model...")

    wing_struct = cfg["components"]["kite"]["wing"]["structure"]
    cs_struct = cfg["components"]["kite"].get("control_system", {}).get("structure", {})
    tether_struct = cfg["components"].get("tether", {}).get("structure", {})

    aero_input = load_aero_input_from_system_config(
        cfg, config_path=LEI_V3_SYSTEM_FLOWN_CONFIG
    )
    mass_wing = wing_struct.get("mass", 14)
    area_wing = wing_struct.get("projected_surface_area", 20)
    mass_kcu = cs_struct.get("mass", 0.0)
    tether_diameter = tether_struct.get("diameter", 0.01)

    if label == "Williams":
        tether = WilliamsTether(
            diameter=tether_diameter,
            n_elements=10,
            elastic=False,
            cf=0.01,
        )
    else:
        tether = RigidLumpedTether(diameter=tether_diameter)
    wind_model = Wind(wind_model="logarithmic", z0=0.1, direction_wind=0)
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

    # ...existing code...

    solutions = []
    start = time.time()
    uf_window = []
    vw_averaged = []
    wdir_window = []
    failed_indices = set()

    if label == "Williams":
        # Williams formulation: solve force balance + ground closure with
        # free tether-end tension and angles.
        unknown_vars = [
            "speed_tangential",
            "input_steering",
            "tension_tether_kite",
            "tether_length",
            "azimuth_last_element",
            "elevation_last_element",
        ]
    else:
        unknown_vars = [
            "tension_tether_ground",
            "input_steering",
            "speed_tangential",
        ]

    solver_options = {
        "ipopt": {
            "print_level": 0,
            "sb": "yes",
            "max_iter": 400,
        },
        "print_time": False,
    }

    window_size = 50
    qs_guess = [1e5, 0, 60]

    if label == "Williams":
        qs_solver, williams_output_fun = build_williams_validation_solver(kite_model)
        kite_model._qs_inputs = [
            "distance_radial",
            "angle_course",
            "speed_radial",
            "angle_azimuth",
            "angle_elevation",
            "speed_friction",
            "timeder_angle_course",
            "input_depower",
        ]
        qs_guess = [
            30.0,
            0.0,
            1.0e5,
            float(np.mean(distance_radial)) if len(distance_radial) else 230.0,
            float(np.mean(azimuth)) if len(azimuth) else 0.0,
            float(np.mean(flight_data.kite_elevation)) if len(flight_data) else 0.3,
        ]
    else:
        cl_func = kite_model.extract_function("lift_coefficient")
        cd_func = kite_model.extract_function("drag_coefficient")
        aoa_func = kite_model.extract_function("angle_of_attack")
        tension_func = kite_model.extract_function("tension_tether_ground")
        speed_apparent_wind_func = kite_model.extract_function("speed_apparent_wind")
        pitch_bridle_func = kite_model.extract_function("pitch_bridle")
        pitch_aero_func = kite_model.extract_function("angle_pitch_aerodynamic")
        roll_aero_func = kite_model.extract_function("angle_roll_aerodynamic")
        input_steering_func = kite_model.extract_function("input_steering")
        roll_bridle_func = kite_model.extract_function("roll_bridle")
        kite_model.setup_qs_solver(unknown_vars, solver_options=solver_options)
        qs_solver = kite_model._qs_solver

    uf = (
        results.wind_speed_horizontal
        * kite_model.wind.kappa
        / np.log(results.kite_position_z / kite_model.wind.z0)
    )
    uf_mean = np.mean(uf)

    # Run the simulation loop for this aerodynamic model
    for i, row in flight_data.iterrows():
        print(f"Processing row {i + 1}/{len(flight_data)}")
        # ...existing simulation loop code...
        uf_window.append(
            results.wind_speed_horizontal[i]
            * kite_model.wind.kappa
            / np.log(results.kite_position_z[i] / kite_model.wind.z0)
        )
        wdir_window.append(results.wind_direction[i])
        if len(uf_window) > window_size:
            uf_window.pop(0)
            wdir_window.pop(0)

        uf = np.mean(uf_window)
        wdir = np.mean(wdir_window)

        current_state = {
            "distance_radial": distance_radial[i],
            "angle_course": row.kite_course,
            "speed_radial": row.tether_reelout_speed,
            "angle_azimuth": azimuth[i] - wdir,
            "angle_elevation": row.kite_elevation,
            "speed_friction": uf,
            "timeder_angle_course": course_rate[i],
            "input_depower": row.up,
        }

        p = np.asarray(
            [current_state[name] for name in kite_model._qs_inputs],
            dtype=float,
        )
        if label == "Williams":
            r_i = float(current_state["distance_radial"])
            # [speed_tangential, input_steering, tension_kite, tether_length, az, elev]
            lbx = [0.0, -3.0, 300.0, 0.5 * r_i, -np.pi, -0.5 * np.pi]
            ubx = [200.0, 3.0, 1.0e9, 3.0 * r_i, np.pi, 0.5 * np.pi]
            # 3 force-balance residuals + 3 ground-position residuals.
            lbg = np.zeros(6, dtype=float)
            ubg = np.zeros(6, dtype=float)
            qs_guess = np.clip(
                np.asarray(qs_guess, dtype=float),
                np.asarray(lbx, dtype=float),
                np.asarray(ubx, dtype=float),
            )
        else:
            lbx, ubx, lbg, ubg = kite_model.get_boundaries(current_state, unknown_vars)
            qs_guess = np.asarray(qs_guess, dtype=float)

        sol = qs_solver(
            x0=np.asarray(qs_guess, dtype=float),
            p=p,
            lbx=np.asarray(lbx, dtype=float),
            ubx=np.asarray(ubx, dtype=float),
            lbg=np.asarray(lbg, dtype=float),
            ubg=np.asarray(ubg, dtype=float),
        )
        qs_guess = np.asarray(sol["x"], dtype=float).reshape(-1)
        qs_state = {name: float(sol["x"][i]) for i, name in enumerate(unknown_vars)}
        state_combined = {**qs_state, **current_state}
        if np.linalg.norm(sol["g"]) < 1:
            if label == "Williams":
                out = np.asarray(
                    williams_output_fun(x=sol["x"], p=p)["outputs"]
                ).reshape(-1)
                # Output layout (see build_williams_validation_solver):
                # [force_balance(3), tension_ground, ground_residual(3),
                #  tension_kite, elev_last, az_last, lift_coef, drag_coef,
                #  aoa, v_apparent, pitch_bridle, pitch_aero, roll_aero,
                #  input_steering, roll_bridle]
                state_combined["force_residual_x"] = float(out[0])
                state_combined["force_residual_y"] = float(out[1])
                state_combined["force_residual_z"] = float(out[2])
                state_combined["tension_tether_ground"] = float(out[3])
                state_combined["ground_residual_x"] = float(out[4])
                state_combined["ground_residual_y"] = float(out[5])
                state_combined["ground_residual_z"] = float(out[6])
                state_combined["tension_tether_kite"] = float(out[7])
                state_combined["elevation_last_element"] = float(out[8])
                state_combined["azimuth_last_element"] = float(out[9])
                state_combined["lift_coefficient"] = float(out[10])
                state_combined["drag_coefficient"] = float(out[11])
                state_combined["angle_of_attack"] = float(out[12])
                state_combined["speed_apparent_wind"] = float(out[13])
                state_combined["pitch_bridle"] = float(out[14])
                state_combined["angle_pitch_aerodynamic"] = float(out[15])
                state_combined["angle_roll_aerodynamic"] = float(out[16])
                state_combined["input_steering"] = float(out[17])
                state_combined["roll_bridle"] = float(out[18])
            else:
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
                    tension_func(
                        *[state_combined[name] for name in tension_func.name_in()]
                    )
                )
                state_combined["pitch_bridle"] = float(
                    pitch_bridle_func(
                        *[state_combined[name] for name in pitch_bridle_func.name_in()]
                    )
                )
                state_combined["angle_pitch_aerodynamic"] = float(
                    pitch_aero_func(
                        *[state_combined[name] for name in pitch_aero_func.name_in()]
                    )
                )
                state_combined["angle_roll_aerodynamic"] = float(
                    roll_aero_func(
                        *[state_combined[name] for name in roll_aero_func.name_in()]
                    )
                )
                state_combined["speed_apparent_wind"] = float(
                    speed_apparent_wind_func(
                        *[
                            state_combined[name]
                            for name in speed_apparent_wind_func.name_in()
                        ]
                    )
                )
                state_combined["input_steering"] = float(
                    input_steering_func(
                        *[
                            state_combined[name]
                            for name in input_steering_func.name_in()
                        ]
                    )
                )
                state_combined["roll_bridle"] = float(
                    roll_bridle_func(
                        *[state_combined[name] for name in roll_bridle_func.name_in()]
                    )
                )
            state_combined["time"] = row.time
            state_combined["original_index"] = i  # Track original index
            solutions.append(state_combined)
            # print(
            #     "angle_of_attack (deg):",
            #     state_combined["angle_of_attack"] * 180 / np.pi,
            # )

        else:
            for dict_entry in state_combined:
                state_combined[dict_entry] = np.nan
            solutions.append(state_combined)
            if label == "Williams":
                qs_guess = [
                    60.0,
                    0.0,
                    1.0e5,
                    float(current_state["distance_radial"]),
                    float(current_state["angle_azimuth"]),
                    float(current_state["angle_elevation"]),
                ]
            else:
                qs_guess[0] = 1e10
                qs_guess[2] = 100
            print("Quasi steady solution not found, index:", i)
            failed_indices.add(i)

    end = time.time()
    print(f"Time taken: {end - start} seconds for {len(flight_data)} iterations")

    print(f"Number of failed solutions: {len(failed_indices)}/ {len(flight_data)}")
    # Remove failed indices from flight_data and results after simulation
    # if failed_indices:
    #     print(
    #         f"\nRemoving {len(failed_indices)} failed rows from flight_data and results"
    #     )
    #     indices_to_remove = sorted(list(failed_indices))
    #     valid_mask = ~flight_data.index.isin(indices_to_remove)
    #     flight_data = flight_data[valid_mask].reset_index(drop=True)
    #     results = results[valid_mask].reset_index(drop=True)
    #     position = position[valid_mask]
    #     velocity = velocity[valid_mask]
    #     distance_radial = distance_radial[valid_mask]
    #     speed_tangential = speed_tangential[valid_mask]
    #     print(f"Updated flight_data length: {len(flight_data)}")
    #     print(f"Updated results length: {len(results)}")

    # Store solutions for this aerodynamic model
    solutions_df = pd.DataFrame(solutions)
    # solutions_df = solutions_df[solutions_df["tension_tether_ground"].notna()]
    # Remove original_index column if present
    # if "original_index" in solutions_df.columns:
    #     solutions_df = solutions_df.drop(columns=["original_index"])
    solutions_df = solutions_df.reset_index(drop=True)
    all_solutions[label] = solutions_df

# Print comparison results for both models
for label, solutions_df in all_solutions.items():
    # --- Create phase masks ---
    mask_pow = (
        (flight_data.up < 0.1)
        & (flight_data.kite_elevation < 0.75)
        & (flight_data.tether_reelout_speed > 0.5)
    )
    mask_dep = (flight_data.tether_reelout_speed < -0.5) & (flight_data.up > 0.9)
    mask_trans = ~(mask_pow | mask_dep)

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
    # --- Cycle-based analysis ---

    # --- Error boxplots by phase ---
    def plot_error_boxplots(cycle_phase_df, label):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        colors_phase = {
            "Powered": "#4CAF50",
            "Depowered": "#FF9800",
            "Transition": "#2196F3",
        }
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
            bp1 = ax1.boxplot(tether_data, tick_labels=phase_labels, patch_artist=True)
            for patch, phase in zip(bp1["boxes"], phase_labels):
                patch.set_facecolor(colors_phase[phase])
                patch.set_alpha(0.7)
            ax1.axhline(y=0, color="r", linestyle="--", alpha=0.5)
            ax1.set_ylabel("Tether Force Relative Error (%)")
            ax1.set_title("Tether Force Error by Phase")
            ax1.grid(True, alpha=0.3)
            bp2 = ax2.boxplot(speed_data, tick_labels=phase_labels, patch_artist=True)
            for patch, phase in zip(bp2["boxes"], phase_labels):
                patch.set_facecolor(colors_phase[phase])
                patch.set_alpha(0.7)
            ax2.axhline(y=0, color="r", linestyle="--", alpha=0.5)
            ax2.set_ylabel("Tangential Speed Relative Error (%)")
            ax2.set_title("Tangential Speed Error by Phase")
            ax2.grid(True, alpha=0.3)
        plt.suptitle(
            f"Validation Errors by Phase - {label} Model",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        save_path = f"./results/figures/validation_errors_{label.lower()}.pdf"
        # plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"\nError boxplot saved to: {save_path}")
        if SHOW_PLOTS:
            plt.show()
        else:
            plt.close(fig)
        return fig

    plot_error_boxplots(cycle_phase_df, label)

    print(f"\n=== Results for {label} aerodynamic model ===")
    dt = 0.1
    total_time = len(flight_data) * dt

    CD = (
        results["wing_drag_coefficient"]
        + results["kcu_drag_coefficient"]
        + results["bridles_drag_coefficient"]
    )

    mask_pow = (
        (flight_data.up < 0.1)
        & (flight_data.kite_elevation < 0.75)
        & (flight_data.tether_reelout_speed > 0.5)
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

    mask = mask_pow
    total_power = (
        sum(
            solutions_df["tension_tether_ground"][mask]
            * solutions_df["speed_radial"][mask]
            * dt
        )
        / total_time
    )
    measured_power = (
        sum(
            flight_data["ground_tether_force"][mask]
            * flight_data["tether_reelout_speed"][mask]
            * dt
        )
        / total_time
    )
    print("Estimated power KCU reelout: ", total_power, "W")
    print("Measured power reelout: ", measured_power, "W")

    mask_dep = (flight_data.tether_reelout_speed < -0.5) & (flight_data.up > 0.9)
    total_power_dep = (
        sum(
            solutions_df["tension_tether_ground"][mask_dep]
            * solutions_df["speed_radial"][mask_dep]
            * dt
        )
        / total_time
    )
    measured_power_dep = (
        sum(
            flight_data["ground_tether_force"][mask_dep]
            * flight_data["tether_reelout_speed"][mask_dep]
            * dt
        )
        / total_time
    )
    print("Estimated power KCU depower: ", total_power_dep, "W")
    print("Measured power depower: ", measured_power_dep, "W")


from awetrim.utils.color_palette import set_plot_style_no_latex
from awetrim.utils.defaults import PLOT_LABELS

set_plot_style_no_latex()


def plot_main_results_comparison(
    all_solutions,
    flight_data,
    speed_tangential,
    PLOT_LABELS,
    save_folder="./results/figures/translational_paper/",
    show=True,
):
    import matplotlib.pyplot as plt
    from awetrim.utils.color_palette import get_color_list

    colors = get_color_list()
    # Define phase masks and colors
    phase_masks = [mask_pow, mask_dep, mask_trans]
    phase_names = ["Reel-out", "Reel-in", "Transition"]
    phase_colors = ["#4CAF50", "#FF9800", "#2196F3"]

    plt.figure()
    label = next(iter(all_solutions))
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
    if show:
        plt.show()
    else:
        plt.close()
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
    # Smooth the EKF pitch and roll signals for clearer plots

    pitch_ekf = -np.degrees(results["kite_pitch"] - results["radial_pitch"])
    roll_ekf = -np.degrees(results["kite_roll"] - results["radial_roll"])

    pitch_ekf_smooth = gaussian_filter1d(pitch_ekf, sigma=3)
    roll_ekf_smooth = gaussian_filter1d(roll_ekf, sigma=3)

    ax5.plot(
        flight_data["time"],
        pitch_ekf_smooth,
        color=colors[0],
        label="Kite pitch EKF",
    )
    ax5.plot(
        flight_data["time"],
        roll_ekf_smooth,
        color=colors[0],
        label="Kite roll EKF",
        linestyle="--",
    )
    ax5.plot(
        flight_data["time"],
        -flight_data["kcu_actual_steering"] / (max(flight_data["kcu_actual_steering"])),
        color="black",
        linestyle="-.",
        label="Steering input EKF",
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
            -np.degrees(solutions_df["pitch_bridle"])
            + np.degrees(solutions_df["angle_pitch_aerodynamic"]),
            color=color,
            label="Kite pitch QS",
        )
        ax5.plot(
            solutions_df["time"],
            np.degrees(solutions_df["angle_roll_aerodynamic"]),
            color=color,
            linestyle="--",
            label="Aerodynamic roll QS",
        )
        ax5.plot(
            solutions_df["time"],
            np.degrees(solutions_df["roll_bridle"]),
            color=color,
            linestyle=":",
            label="Kite roll QS",
        )
        ax5.plot(
            solutions_df["time"],
            solutions_df["input_steering"],
            color="blue",
            linestyle="-.",
            label="Steering input QS",
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
    ax5.legend(loc="best", frameon=True)
    ax3.set_ylim(0, 40)
    ax4.set_ylim(0, 5)
    ax5.set_ylim(-20, 20)
    for ax in axs:
        ax.set_xlim(flight_data["time"].min(), flight_data["time"].max())

    plt.tight_layout()
    # plt.savefig(save_folder + "validation_v3_dynamics.pdf", bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig1)

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
    # plt.savefig(save_folder + "validation_v3_aerodynamics.pdf", bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig2)


# --- Plotting section ---
plot_main_results_comparison(
    all_solutions,
    flight_data,
    speed_tangential,
    PLOT_LABELS,
    save_folder="./results/figures/translational_paper/",
    show=SHOW_PLOTS,
)
