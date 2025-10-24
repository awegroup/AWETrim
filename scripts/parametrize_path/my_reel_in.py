import csv
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from awetrim import SystemModel, State
from awetrim.environment.Wind import Wind
from awetrim.system.kite import Kite
from awetrim.system.tether import RigidLumpedTether
from awetrim.timeseries.phase_parametrized import PhaseParameterized
from awetrim.utils.color_palette import set_plot_style, get_color_list
from awetrim.utils.defaults import PLOT_LABELS

PLOT_VARIABLES = [
    "distance_radial",
    "speed_radial",
    "speed_tangential",
    "tension_tether_ground",
    "lift_coefficient",
    "drag_coefficient",
    "mechanical_power",
]
BASE_VARIABLES = PLOT_VARIABLES + [
    "angle_elevation",
    "angle_azimuth",
]
DERIVED_VARIABLES = ["x_position", "y_position", "z_position"]
CSV_HEADER = ["segment", "simulation", "time"] + BASE_VARIABLES + DERIVED_VARIABLES
REEL_IN_OUTPUT_PATH = Path("results/timeseries/reel_in_timeseries.csv")

AGGREGATED_RESULTS = None
init_conditions_QS = None
init_conditions_Dyn = None


def define_system(
    tether_diameter,
    mass_wing,
    mass_kcu,
    area_wing,
    aero_input,
    wind_model,
):

    tether = RigidLumpedTether(
        diameter=tether_diameter,
    )
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

    if quasi_steady:
        sim_type = "quasi steady"
    else:
        sim_type = "dynamic"
    print(f"Running simulation for {sim_type} with label: {label_prefix}")

    model.input_depower = depower

    phase = PhaseParameterized(
        model, quasi_steady=quasi_steady, pattern_config=pattern_config
    )
    phase.run_simulation_phase(start_state=start_state, return_states=True)

    return phase


def main(run_plots=True, save_csv=True, depower_denom=0.28):
    global AGGREGATED_RESULTS, init_conditions_QS, init_conditions_Dyn

    aggregated_data = {
        "quasi_steady": {"t": [], "segment": []},
        "dynamic": {"t": [], "segment": []},
    }
    for var_name in BASE_VARIABLES + DERIVED_VARIABLES:
        aggregated_data["quasi_steady"][var_name] = []
        aggregated_data["dynamic"][var_name] = []
    cumulative_time = {"quasi_steady": 0.0, "dynamic": 0.0}

    def extend_phase(phase, sim_key, segment_label, time_offset):
        try:
            times = np.asarray(phase.return_variable("t"), dtype=float)
        except Exception:
            times = np.array([], dtype=float)
        if times.size == 0:
            return time_offset
        times = np.nan_to_num(times, nan=0.0)
        shifted = time_offset + (times - times[0])
        aggregated_data[sim_key]["t"].extend(shifted.tolist())
        aggregated_data[sim_key]["segment"].extend([segment_label] * shifted.size)

        r_vals = None
        beta_vals = None
        phi_vals = None
        for var_name in BASE_VARIABLES:
            try:
                values = np.asarray(phase.return_variable(var_name), dtype=float)
            except Exception:
                values = np.array([], dtype=float)
            if values.size != shifted.size:
                temp = np.full(shifted.shape, np.nan, dtype=float)
                count = min(values.size, shifted.size)
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
            r_vals = np.full(shifted.shape, np.nan, dtype=float)
        if beta_vals is None:
            beta_vals = np.full(shifted.shape, np.nan, dtype=float)
        if phi_vals is None:
            phi_vals = np.full(shifted.shape, np.nan, dtype=float)

        x_vals = r_vals * np.cos(beta_vals) * np.cos(phi_vals)
        y_vals = r_vals * np.cos(beta_vals) * np.sin(phi_vals)
        z_vals = r_vals * np.sin(beta_vals)
        aggregated_data[sim_key]["x_position"].extend(x_vals.tolist())
        aggregated_data[sim_key]["y_position"].extend(y_vals.tolist())
        aggregated_data[sim_key]["z_position"].extend(z_vals.tolist())
        return shifted[-1]

    # ---------- Config ----------
    mass_wing = 61
    mass_kcu = 30
    area_wing = 46.85
    tether_diameter = 0.01

    speed_wind_at_100 = (
        7.6374  # m/s (6 m/s at reference height of 6 m) got from KP software for LOG
    )
    wind_model = Wind(
        wind_model="logarithmic",
        z0=0.0002,
    )
    speed_friction = 0.41 * speed_wind_at_100 / np.log(100 / wind_model.z0)
    wind_model.speed_friction = speed_friction

    # color palette available via get_color_list() as needed

    with open("./data/LEI-V9-KITE/v9_aero_input.json", "r") as file:
        aero_input_v9 = json.load(file)

    system_model = define_system(
        tether_diameter,
        mass_wing,
        mass_kcu,
        area_wing,
        aero_input_v9,
        wind_model,
    )
    # ---------- Load precomputed spline fit data ----------
    segment_name = "RI_Spline"  # input("Enter segment name (e.g., 'RI' or 'RI_RO' or 'RO_RI or 'RI_Spline'): ").strip()

    filename = f"fit_results_{segment_name}.pkl"
    with open(filename, "rb") as f:
        fit_data = pickle.load(f)

    r0 = fit_data["r0"]
    r1 = fit_data["r1"]
    C_az = fit_data["C_az"]
    C_el = fit_data["C_el"]
    s_norm_az = fit_data["s_norm_az"]
    s_norm_el = fit_data["s_norm_el"]

    pattern_config = {
        "pattern_type": "spline",
        "path_parameters": {
            "r0": r0,
            "r1": r1,
            "C_az": C_az,
            "C_el": C_el,
            "s_norm_az": s_norm_az,
            "s_norm_el": s_norm_el,
        },
        "radial_parameters": [],
        "start_angle": 0,
        "end_angle": 1,
        "n_points": 100,
        "optimization_parameters": [],
    }

    # ---------Load winch and depower data ----------

    if segment_name == "RI_Spline":
        with open("fit_winch_results_RI_Spline_phase_settings.pkl", "rb") as f:
            winch_depower_data = pickle.load(f)

    init_condit_QS_dict = [
        {
            "t": 0,
            "s": 0,
            "s_dot": 0.05,
            "s_ddot": 0,
            "input_steering": 0,
            "tension_tether_ground": 1e8,
            "distance_radial": r0,
            "speed_radial": 0,
            "input_depower": 0,
        }
    ]

    init_condit_Dyn_dict = [
        {
            "t": 0,
            "s": 0,
            "s_dot": 0.05,
            "s_ddot": 0,
            "input_steering": 0,
            "tension_tether_ground": 1e8,
            "distance_radial": r0,
            "speed_radial": 0,
            "input_depower": 0,
        }
    ]

    quasi_steady_s_ends = []
    dynamic_s_ends = []
    time = 0
    for phase_idx in range(len(winch_depower_data)):

        v_knots = winch_depower_data[phase_idx]["v_knots"]
        C_fitted = winch_depower_data[phase_idx]["C_fitted"]
        s_start = winch_depower_data[phase_idx]["s"]
        depower = winch_depower_data[phase_idx]["depower"]
        if phase_idx == len(winch_depower_data) - 1:
            s_end = 1
        else:
            s_end = winch_depower_data[phase_idx + 1]["s"]
        print(s_start)
        print(s_end)

        def depower_norm(denom):
            return ((depower / 100) - 0.4) / denom
        
        depower_normalized = depower_norm(depower_denom)

        Realistic_RI_eg = {
            "reeling_strategy": "force",  # "force" or "constant"
            "force_model": "custom_spline",  # "linear" or "quadratic"
            "reeling_speed": 0,  # m/s, only for constant reeling
            "v_knots": v_knots,
            "C_fitted": C_fitted,
        }

        pattern_config["start_angle"] = s_start
        pattern_config["end_angle"] = s_end
        pattern_config["radial_parameters"] = Realistic_RI_eg
        base_start_state_QS = State(**init_condit_QS_dict[phase_idx])
        base_start_state_Dyn = State(**init_condit_Dyn_dict[phase_idx])
        system_model = define_system(
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
            system_model,
            quasi_steady=True,
        )

        # if phase_idx == 0:
        base_start_state_Dyn = phaseQS.states[0]

        phaseDyn = run_sim(
            pattern_config,
            "V9",
            depower_normalized,
            base_start_state_Dyn,
            system_model,
            quasi_steady=False,
        )

        quasi_steady_s_ends.append(phaseQS.states[-1]["s"])
        dynamic_s_ends.append(phaseDyn.states[-1]["s"])
        init_condit_QS_dict.append(
            {
                "t": phaseQS.states[-1]["t"],
                "s": phaseQS.states[-1]["s"],
                "s_dot": base_start_state_QS.s_dot,
                "s_ddot": (
                    phaseQS.states[-1]["s_ddot"]
                    if phaseQS.states[-1]["s_ddot"] is not None
                    else 0
                ),
                "input_steering": phaseQS.states[-1]["input_steering"],
                "tension_tether_ground": base_start_state_QS.tension_tether_ground,
                "input_depower": (
                    phaseQS.states[-1]["input_depower"]
                    if phaseQS.states[-1]["input_depower"] is not None
                    else 0
                ),
                "speed_radial": base_start_state_QS.speed_radial,
                "distance_radial": phaseQS.states[-1]["distance_radial"],
            }
        )

        init_condit_Dyn_dict.append(
            {
                "t": phaseDyn.states[-1]["t"],
                "s": phaseDyn.states[-1]["s"],
                "s_dot": phaseDyn.states[-1]["s_dot"],
                "s_ddot": (
                    phaseDyn.states[-1]["s_ddot"]
                    if phaseDyn.states[-1]["s_ddot"] is not None
                    else 0
                ),
                "input_steering": phaseDyn.states[-1]["input_steering"],
                "tension_tether_ground": phaseDyn.states[-1]["tension_tether_ground"],
                "input_depower": (
                    phaseDyn.states[-1]["input_depower"]
                    if phaseDyn.states[-1]["input_depower"] is not None
                    else 0
                ),
                "speed_radial": phaseDyn.states[-1]["speed_radial"],
                "distance_radial": phaseDyn.states[-1]["distance_radial"],
            }
        )

        dynamic_phase = phaseDyn
        qs_phase = phaseQS
        cumulative_time["quasi_steady"] = extend_phase(
            qs_phase, "quasi_steady", "reel_in", cumulative_time["quasi_steady"]
        )
        cumulative_time["dynamic"] = extend_phase(
            dynamic_phase, "dynamic", "reel_in", cumulative_time["dynamic"]
        )

        # First series creates the overview figure
        # fig, axes_map, scatter = phaseDyn.plot_overview_3d(
        #     label="V9 Dynamic",
        #     color=get_color_list()[2],
        #     linestyle="-",
        #     variables=[
        #         "speed_tangential",
        #         "tension_tether_ground",
        #         "lift_coefficient",
        #         "speed_radial",
        #     ],
        #     x_param="t",
        # )

        # # Second series overlays on the same axes
        # qs_phase.plot_overview_3d(
        #     label="V9 Quasi-Steady",
        #     color=get_color_list()[1],
        #     linestyle="--",
        #     variables=[
        #         "speed_tangential",
        #         "tension_tether_ground",
        #         "lift_coefficient",
        #         "speed_radial",
        #     ],
        #     x_param="t",
        #     axes=axes_map,
        # )

        # fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=2)
        # set_plot_style()
        # plt.tight_layout()
        # # # Save the figure as pdf
        # # plt.savefig("./results/figures/reelout_cst.pdf", bbox_inches="tight")
        # plt.show()

        # metrics = dynamic_phase.energy_metrics(qs_phase)
        # print("\n--- V9 ---")
        # print(
        #     f"Power QS: {metrics['avg_power_other']:.2f}, Power Dyn: {metrics['avg_power_self']:.2f}."
        # )
        # print(
        #     f"Mean power QS: {metrics['mean_power_other']:.2f}, Mean power Dyn: {metrics['mean_power_self']:.2f}"
        # )
        # print(f"Δ Power: {metrics['power_diff_percent']:.2f}%")
        # print(f"Estimated time lag: {metrics['best_time_lag']:.3f} s")
        # print(f"ΔF_t,mean: {metrics['delta_ft_mean_percent']:.2f}%")
        # print(f"ΔF_t,max: {metrics['delta_ft_max_percent']:.2f}%")
        # print(f"ΔF_t,min: {metrics['delta_ft_min_percent']:.2f}%")
        # print(f"Δv_tau,max: {metrics['delta_vtau_max_percent']:.2f}%")
        # print(f"Δv_tau,min: {metrics['delta_vtau_min_percent']:.2f}%")
        # print(f"Δs_v_tau,max: {metrics['s_lag_vtau_max_deg']:.2f} deg")
        # print(f"Δs_v_tau,min: {metrics['s_lag_vtau_min_deg']:.2f} deg")
        # plt.show()

    AGGREGATED_RESULTS = aggregated_data

    has_timeseries_data = bool(aggregated_data["quasi_steady"]["t"]) or bool(
        aggregated_data["dynamic"]["t"]
    )
    if has_timeseries_data and run_plots:
        set_plot_style()
        fig, axes = plt.subplots(
            len(PLOT_VARIABLES),
            1,
            sharex=True,
            figsize=(10, 3 * len(PLOT_VARIABLES)),
        )
        axes = np.atleast_1d(axes)
        for idx, var_name in enumerate(PLOT_VARIABLES):
            ax = axes[idx]
            ylabel = PLOT_LABELS.get(var_name, var_name)
            for sim_key, sim_label in [
                ("quasi_steady", "Quasi-Steady"),
                ("dynamic", "Dynamic"),
            ]:
                times = aggregated_data[sim_key]["t"]
                values = aggregated_data[sim_key][var_name]
                if times and values:
                    ax.plot(times, values, label=sim_label)
            ax.set_ylabel(ylabel)
            ax.grid(True, linestyle="--", alpha=0.3)
        axes[-1].set_xlabel("Time [s]")
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            axes[0].legend(loc="best")
        plt.tight_layout()
        plt.show()

        fig3d = plt.figure(figsize=(8, 6))
        ax3d = fig3d.add_subplot(111, projection="3d")
        plotted_any = False
        for sim_key, sim_label in [
            ("quasi_steady", "Quasi-Steady"),
            ("dynamic", "Dynamic"),
        ]:
            x_vals = np.asarray(aggregated_data[sim_key]["x_position"], dtype=float)
            y_vals = np.asarray(aggregated_data[sim_key]["y_position"], dtype=float)
            z_vals = np.asarray(aggregated_data[sim_key]["z_position"], dtype=float)
            finite_mask = (
                np.isfinite(x_vals) & np.isfinite(y_vals) & np.isfinite(z_vals)
            )
            if finite_mask.any():
                ax3d.plot(
                    x_vals[finite_mask],
                    y_vals[finite_mask],
                    z_vals[finite_mask],
                    label=sim_label,
                )
                plotted_any = True
        if plotted_any:
            ax3d.set_xlabel(PLOT_LABELS.get("x", "x"))
            ax3d.set_ylabel(PLOT_LABELS.get("y", "y"))
            ax3d.set_zlabel(PLOT_LABELS.get("z", "z"))
            x_combined = []
            y_combined = []
            z_combined = []
            for sim_key in ["quasi_steady", "dynamic"]:
                x_arr = np.asarray(aggregated_data[sim_key]["x_position"], dtype=float)
                y_arr = np.asarray(aggregated_data[sim_key]["y_position"], dtype=float)
                z_arr = np.asarray(aggregated_data[sim_key]["z_position"], dtype=float)
                finite = np.isfinite(x_arr) & np.isfinite(y_arr) & np.isfinite(z_arr)
                if finite.any():
                    x_combined.append(x_arr[finite])
                    y_combined.append(y_arr[finite])
                    z_combined.append(z_arr[finite])
            if x_combined:
                x_all = np.concatenate(x_combined)
                y_all = np.concatenate(y_combined)
                z_all = np.concatenate(z_combined)
                ranges = np.array([np.ptp(x_all), np.ptp(y_all), np.ptp(z_all)])
                overall = np.nanmax(ranges) if ranges.size else 0.0
                if overall > 0:
                    mid_x = 0.5 * (np.nanmax(x_all) + np.nanmin(x_all))
                    mid_y = 0.5 * (np.nanmax(y_all) + np.nanmin(y_all))
                    mid_z = 0.5 * (np.nanmax(z_all) + np.nanmin(z_all))
                    half = overall / 2.0
                    ax3d.set_xlim(mid_x - half, mid_x + half)
                    ax3d.set_ylim(mid_y - half, mid_y + half)
                    ax3d.set_zlim(mid_z - half, mid_z + half)
                    ax3d.set_box_aspect([1, 1, 1])
            ax3d.legend(loc="best")
            plt.tight_layout()
            plt.show()
        else:
            plt.close(fig3d)

    if has_timeseries_data and save_csv:
        REEL_IN_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with REEL_IN_OUTPUT_PATH.open("w", newline="") as csvfile:
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
        print(f"Saved aggregated timeseries to {REEL_IN_OUTPUT_PATH}")

    total_qs_time = (
        aggregated_data["quasi_steady"]["t"][-1]
        if aggregated_data["quasi_steady"]["t"]
        else 0.0
    )
    total_dyn_time = (
        aggregated_data["dynamic"]["t"][-1] if aggregated_data["dynamic"]["t"] else 0.0
    )
    time = max(total_qs_time, total_dyn_time, time)
    if total_qs_time:
        print(f"Total quasi-steady time: {total_qs_time:.3f} s")
    if total_dyn_time:
        print(f"Total dynamic time: {total_dyn_time:.3f} s")
    print("Total time:", time)

    init_conditions_QS = init_condit_QS_dict[-1]
    init_conditions_Dyn = init_condit_Dyn_dict[-1]
    return init_conditions_QS, init_conditions_Dyn, AGGREGATED_RESULTS


def get_initial_conditions(run_if_needed=True):
    global init_conditions_QS, init_conditions_Dyn
    if run_if_needed and (init_conditions_QS is None or init_conditions_Dyn is None):
        main(run_plots=False, save_csv=False)
    return init_conditions_QS, init_conditions_Dyn

if __name__ == "__main__":
    main()
