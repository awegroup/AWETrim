import csv
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from awetrim import SystemModel, State
from awetrim.environment.Wind import Wind
from awetrim.kinematics.find_RO_start_end_angles import (
    find_RO_start_end_angles,
)
from awetrim.system.kite import Kite
from awetrim.system.tether import RigidLumpedTether
from awetrim.timeseries.phase_parametrized import PhaseParameterized
from awetrim.utils.color_palette import set_plot_style, get_color_list
import my_reel_in

PLOT_VARIABLES = my_reel_in.PLOT_VARIABLES
BASE_VARIABLES = my_reel_in.BASE_VARIABLES
DERIVED_VARIABLES = my_reel_in.DERIVED_VARIABLES
CSV_HEADER = my_reel_in.CSV_HEADER
REEL_OUT_OUTPUT_PATH = Path("results/timeseries/reel_out_timeseries.csv")

AGGREGATED_RESULTS = None


def define_system(
    tether_diameter,
    mass_wing,
    mass_kcu,
    area_wing,
    aero_input,
    wind_model,
):
    """Instantiate a SystemModel with the supplied components."""

    tether = RigidLumpedTether(diameter=tether_diameter)
    kite = Kite(
        mass_wing=mass_wing,
        mass_kcu=mass_kcu,
        area_wing=area_wing,
        aero_input=aero_input,
        steering_control="asymmetric",
    )

    model = SystemModel(
        dof=3,
        kite=kite,
        tether=tether,
        wind_model=wind_model,
    )
    return model


def run_sim(
    pattern_config,
    label_prefix,
    depower,
    start_state,
    model,
    quasi_steady,
):
    """Run a parametrized phase simulation and return the populated PhaseParameterized object."""

    sim_type = "quasi steady" if quasi_steady else "dynamic"
    print(f"Running simulation for {sim_type} with label: {label_prefix}")

    model.input_depower = depower

    phase = PhaseParameterized(
        model,
        quasi_steady=quasi_steady,
        pattern_config=pattern_config,
    )
    phase.run_simulation_phase(start_state=start_state, return_states=True)
    return phase


def main(run_plots=True, save_csv=True, depower_denom=0.28):
    global AGGREGATED_RESULTS
    # ---------- Config ----------
    mass_wing = 61
    mass_kcu = 30
    area_wing = 46.85
    tether_diameter = 0.01

    speed_wind_at_100 = 7.6374  # m/s (6 m/s at reference height of 6 m)
    wind_model = Wind(
        wind_model="logarithmic",
        z0=0.0002,
    )
    speed_friction = 0.41 * speed_wind_at_100 / np.log(100 / wind_model.z0)
    wind_model.speed_friction = speed_friction

    with open("./data/LEI-V9-KITE/v9_aero_input.json", "r") as file:
        aero_input_v9 = json.load(file)

    # ---------- Load precomputed L_shape fit data ----------
    segment_name = "L_shape"
    filename = f"fit_results_{segment_name}.pkl"
    with open(filename, "rb") as f:
        fit_data = pickle.load(f)

    r0 = fit_data["r0"]
    duration = fit_data["duration"]
    az_amp0 = fit_data["best_params"]["az_amp0"]
    beta_amp0 = fit_data["best_params"]["beta_amp0"]
    beta_coeffs = fit_data["best_params"]["beta_coeffs"]
    az_coeffs = fit_data["best_params"]["az_coeffs"]
    beta0 = fit_data["best_params"]["beta0"]

    pattern_type = "cst_lissajous"
    parameters = {
        "omega": 1.0,
        "r0": r0,
        "az_amp0": az_amp0,
        "beta_amp0": beta_amp0,
        "width_phi": 0.5,
        "width_beta": 0.5,
        "left_first": True,
        "normalize_bumps": False,
        "repeat_phi": True,
        "repeat_beta": True,
        "beta_coeffs": np.array(beta_coeffs),
        "az_coeffs": az_coeffs,
        "kbeta": 0,
        "beta0": beta0,
        "kappa": 0,
    }

    s_start_opt, range_opt, cycles = find_RO_start_end_angles(
        pattern_type, parameters
    )

    # --------- Load winch and depower data ----------
    with open("fit_winch_results_RO_phase_settings.pkl", "rb") as f:
        winch_depower_data = pickle.load(f)

    v_knots = winch_depower_data[0]["v_knots"]
    C_fitted = winch_depower_data[0]["C_fitted"]
    s = winch_depower_data[0]["s"]
    depower = winch_depower_data[0]["depower"]

    def depower_norm(denom):
        return ((depower / 100) - 0.4) / denom
    
    depower_normalized = depower_norm(depower_denom)

    Realistic_RO_eg = {
        "reeling_strategy": "force",  # "force" or "constant"
        "force_model": "custom_spline",  # "linear" or "quadratic"
        "reeling_speed": 0,  # m/s, only for constant reeling
        "v_knots": v_knots,
        "C_fitted": C_fitted,
    }

    pattern_config = {
        "pattern_type": pattern_type,
        "path_parameters": parameters,
        "radial_parameters": Realistic_RO_eg,
        "start_time": 0,
        "end_time": duration + 1,
        "start_angle": s_start_opt,
        "end_angle": s_start_opt + range_opt + cycles * (2 * np.pi),
        "n_points": 500,
        "optimization_parameters": [],
    }

    # ---------- Starting state ----------

    base_start_state = {
        "t": 0,
        "s": 0,
        "s_dot": 2,
        "s_ddot": 0,
        "input_steering": 0,
        "tension_tether_ground": 1e8,
        "distance_radial": r0,
        "speed_radial": 0,
        "input_depower": 0,
    }
    base_start_state_QS = State(**base_start_state)
    base_start_state_Dyn = State(**base_start_state)

    system_model_qs = define_system(
        tether_diameter,
        mass_wing,
        mass_kcu,
        area_wing,
        aero_input_v9,
        wind_model,
    )
    phaseQS = run_sim(
        pattern_config,
        "V9",
        depower_normalized,
        base_start_state_QS,
        system_model_qs,
        quasi_steady=True,
    )

    # Use the quasi-steady initial state as the baseline for the dynamic run
    if phaseQS.states:
        base_start_state_Dyn = phaseQS.states[0]

    system_model_dyn = define_system(
        tether_diameter,
        mass_wing,
        mass_kcu,
        area_wing,
        aero_input_v9,
        wind_model,
    )
    phaseDyn = run_sim(
        pattern_config,
        "V9",
        depower_normalized,
        base_start_state_Dyn,
        system_model_dyn,
        quasi_steady=False,
    )

    aggregated_data = {
        "quasi_steady": {"t": [], "segment": []},
        "dynamic": {"t": [], "segment": []},
    }
    for var_name in BASE_VARIABLES + DERIVED_VARIABLES:
        aggregated_data["quasi_steady"][var_name] = []
        aggregated_data["dynamic"][var_name] = []

    def extend_phase(phase, sim_key, segment_label):
        try:
            times = np.asarray(phase.return_variable("t"), dtype=float)
        except Exception:
            times = np.array([], dtype=float)
        if times.size == 0:
            return
        times = np.nan_to_num(times, nan=0.0)
        aggregated_data[sim_key]["t"].extend(times.tolist())
        aggregated_data[sim_key]["segment"].extend([segment_label] * times.size)

        r_vals = None
        beta_vals = None
        phi_vals = None
        for var_name in BASE_VARIABLES:
            try:
                values = np.asarray(phase.return_variable(var_name), dtype=float)
            except Exception:
                values = np.array([], dtype=float)
            if values.size != times.size:
                temp = np.full(times.shape, np.nan, dtype=float)
                count = min(values.size, times.size)
                if count > 0:
                    temp[:count] = values[:count]
                values = temp
            aggregated_data[sim_key][var_name].extend(values.tolist())
            if var_name == "distance_radial":
                r_vals = values
            elif var_name == "angle_elevation":
                beta_vals = values
            elif var_name == "angle_azimuth":
                phi_vals = values

        if r_vals is None:
            r_vals = np.full(times.shape, np.nan, dtype=float)
        if beta_vals is None:
            beta_vals = np.full(times.shape, np.nan, dtype=float)
        if phi_vals is None:
            phi_vals = np.full(times.shape, np.nan, dtype=float)

        x_vals = r_vals * np.cos(beta_vals) * np.cos(phi_vals)
        y_vals = r_vals * np.cos(beta_vals) * np.sin(phi_vals)
        z_vals = r_vals * np.sin(beta_vals)
        aggregated_data[sim_key]["x_position"].extend(x_vals.tolist())
        aggregated_data[sim_key]["y_position"].extend(y_vals.tolist())
        aggregated_data[sim_key]["z_position"].extend(z_vals.tolist())

    extend_phase(phaseQS, "quasi_steady", "reel_out")
    extend_phase(phaseDyn, "dynamic", "reel_out")

    AGGREGATED_RESULTS = aggregated_data
    print("Aggregated reel-out timeseries data.")
    print(aggregated_data.keys())

    if save_csv:
        REEL_OUT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with REEL_OUT_OUTPUT_PATH.open("w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(CSV_HEADER)
            for sim_key, sim_label in [
                ("quasi_steady", "quasi_steady"),
                ("dynamic", "dynamic"),
            ]:
                times = aggregated_data[sim_key]["t"]
                segments = aggregated_data[sim_key]["segment"]
                for idx in range(len(times)):
                    row = [segments[idx], sim_label, times[idx]]
                    row.extend(
                        aggregated_data[sim_key][var][idx] for var in BASE_VARIABLES
                    )
                    row.extend(
                        aggregated_data[sim_key][var][idx] for var in DERIVED_VARIABLES
                    )
                    writer.writerow(row)
        print(f"Saved reel-out timeseries to {REEL_OUT_OUTPUT_PATH}")

    dynamic_phase = phaseDyn
    qs_phase = phaseQS

    if run_plots:
        qs_tension = [state["tension_tether_ground"] for state in phaseQS.states]
        dyn_tension = [state["tension_tether_ground"] for state in phaseDyn.states]

        plt.figure()
        plt.plot(qs_tension, label="Quasi-Steady")
        plt.plot(dyn_tension, label="Dynamic")
        plt.legend()
        plt.show()

        set_plot_style()
        fig, axes_map, _ = dynamic_phase.plot_overview_3d(
            label="V9 Dynamic",
            color=get_color_list()[2],
            linestyle="-",
            variables=[
                "speed_tangential",
                "tension_tether_ground",
                "input_steering",
                "speed_radial",
            ],
            x_param="t",
        )

        qs_phase.plot_overview_3d(
            label="V9 Quasi-Steady",
            color=get_color_list()[1],
            linestyle="--",
            variables=[
                "speed_tangential",
                "tension_tether_ground",
                "input_steering",
                "speed_radial",
            ],
            x_param="t",
            axes=axes_map,
        )

        fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=2)
        plt.tight_layout()
        plt.show()

    metrics = dynamic_phase.energy_metrics(qs_phase)
    print("\n--- V9 ---")
    print(
        f"Power QS: {metrics['avg_power_other']:.2f}, Power Dyn: {metrics['avg_power_self']:.2f}."
    )
    print(
        f"Mean power QS: {metrics['mean_power_other']:.2f}, Mean power Dyn: {metrics['mean_power_self']:.2f}"
    )
    print(f"Delta Power: {metrics['power_diff_percent']:.2f}%")
    print(f"Estimated time lag: {metrics['best_time_lag']:.3f} s")
    print(f"Delta F_t,mean: {metrics['delta_ft_mean_percent']:.2f}%")
    print(f"Delta F_t,max: {metrics['delta_ft_max_percent']:.2f}%")
    print(f"Delta F_t,min: {metrics['delta_ft_min_percent']:.2f}%")
    print(f"Delta v_tau,max: {metrics['delta_vtau_max_percent']:.2f}%")
    print(f"Delta v_tau,min: {metrics['delta_vtau_min_percent']:.2f}%")
    print(f"Delta s_v_tau,max: {metrics['s_lag_vtau_max_deg']:.2f} deg")
    print(f"Delta s_v_tau,min: {metrics['s_lag_vtau_min_deg']:.2f} deg")

    return phaseQS, phaseDyn


def get_results(run_if_needed=True):
    global AGGREGATED_RESULTS
    if run_if_needed and AGGREGATED_RESULTS is None:
        main(run_plots=False, save_csv=False)
    return AGGREGATED_RESULTS

if __name__ == "__main__":
    main()
