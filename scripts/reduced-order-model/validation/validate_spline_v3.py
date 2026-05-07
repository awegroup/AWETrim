import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

from awetrim.kinematics.parametrized_patterns import (
    fit_bspline_pattern_to_trajectory,
)
from awetrim.utils.color_palette import set_plot_style, custom_cmap
from awetrim.utils.config_paths import (
    LEI_V3_DOWNLOOP_SPLINE_CONFIG,
    LEI_V3_SYSTEM_CONFIG,
)
from awetrim.environment.Wind import Wind
from awetrim.system.factory import create_system_model_from_yaml
from awetrim.timeseries.phase import Phase

set_plot_style()


# -----------------------------------------------------------------------------
# Helpers to derive cycle IDs from flight phases
# -----------------------------------------------------------------------------


def cycles_from_phases(phase_series: pd.Series) -> np.ndarray:
    """Return a cycle id per row using phase sequence 1->2->3->4 as a cycle.

    The function collapses consecutive identical phases, looks for repeated
    sequences [1, 2, 3, 4], and assigns cycle ids starting at 0 to the rows
    that belong to each detected sequence. Rows outside detected cycles get -1.
    """

    phases = phase_series.to_numpy()
    if phases.size == 0:
        return np.array([], dtype=int)

    # Collapse consecutive repeats so we can search for ordered runs
    change_idx = np.flatnonzero(np.diff(phases) != 0) + 1
    run_starts = np.concatenate(([0], change_idx))
    run_values = phases[run_starts]
    run_ends = np.concatenate((change_idx, [phases.size]))

    cycle_ids = np.full(phases.shape, -1, dtype=int)
    pattern = [1, 2, 3, 4]
    cycle = -1

    for i in range(len(run_values) - len(pattern) + 1):
        if list(run_values[i : i + 4]) != pattern:
            continue

        cycle += 1
        start_idx = run_starts[i]
        end_idx = run_starts[i + 4] if i + 4 < len(run_starts) else phases.size
        cycle_ids[start_idx:end_idx] = cycle

    return cycle_ids


# -----------------------------------------------------------------------------
# Winch identification over full flight
# -----------------------------------------------------------------------------


def fit_winch_models_full_flight(flight_df: pd.DataFrame) -> dict:
    """Fit winch slope/offset per phase using the entire flight dataset."""

    winch_models = {}
    figs = []
    for ph in [1, 2, 3, 4]:
        ph_mask = flight_df.get("flight_phase_index", pd.Series(dtype=int)) == ph
        phase_rows = flight_df[ph_mask]
        if phase_rows.empty:
            print(f"Winch fit: no data for phase {ph}")
            continue

        tether_force = phase_rows["ground_tether_force"].to_numpy()
        reelout_speed = phase_rows["tether_reelout_speed"].to_numpy()

        valid_mask = np.isfinite(tether_force) & np.isfinite(reelout_speed)
        # Remove obvious bad forces
        valid_mask &= tether_force > 2000

        tf = tether_force[valid_mask]
        spd = reelout_speed[valid_mask]
        if tf.size < 2:
            print(f"Winch fit: insufficient points for phase {ph}")
            continue

        # Quantile clipping on both axes to knock outliers down
        tf_lo, tf_hi = np.quantile(tf, [0.02, 0.98])
        spd_lo, spd_hi = np.quantile(spd, [0.02, 0.98])
        clip_mask = (tf >= tf_lo) & (tf <= tf_hi) & (spd >= spd_lo) & (spd <= spd_hi)
        tf_clip = tf[clip_mask]
        spd_clip = spd[clip_mask]
        if tf_clip.size < 2:
            print(f"Winch fit: insufficient clipped points for phase {ph}")
            continue

        # First fit
        coeffs = np.polyfit(spd_clip, tf_clip, 1)
        slope = float(coeffs[0])
        offset = float(-coeffs[1] / coeffs[0]) if coeffs[0] != 0 else 0.0

        # Residual-based outlier removal (MAD)
        pred = slope * spd_clip + slope * offset
        resid = tf_clip - pred
        mad = np.median(np.abs(resid - np.median(resid))) + 1e-9
        inlier_mask = np.abs(resid) <= 3.0 * mad
        spd_in = spd_clip[inlier_mask]
        tf_in = tf_clip[inlier_mask]
        if tf_in.size >= 2:
            coeffs = np.polyfit(spd_in, tf_in, 1)
            slope = float(coeffs[0])
            offset = float(-coeffs[1] / coeffs[0]) if coeffs[0] != 0 else 0.0
        else:
            # Fall back to clipped set if inliers vanished
            spd_in = spd_clip
            tf_in = tf_clip

        winch_models[ph] = {"slope_winch_ro": slope, "offset_winch_ro": offset}

        # Plot fit for this phase
        fig = plt.figure(figsize=(5, 3.2))
        ax = fig.add_subplot(111)
        ax.scatter(spd, tf, s=6, alpha=0.25, label="raw")
        ax.scatter(spd_in, tf_in, s=10, alpha=0.7, label="inliers")
        if spd_in.size >= 2:
            spd_lin = np.linspace(np.min(spd_in), np.max(spd_in), 50)
            ax.plot(
                spd_lin,
                slope * spd_lin - slope * offset,
                "r--",
                label=f"fit slope={slope:.2e}, offset={offset:.2f}",
            )
        ax.set_xlabel("Reel-out speed (m/s)")
        ax.set_ylabel("Tether force (N)")
        ax.set_title(f"Winch fit phase {ph} (full flight)")
        ax.grid(True)
        ax.legend()
        figs.append(fig)
        # plt.show()
    return winch_models


# -----------------------------------------------------------------------------
# Data loading helpers (adapted from validatev3)
# -----------------------------------------------------------------------------


def read_results(year, month, day, kite_model, addition="", path_to_main=""):
    path = "/flight_logs/"
    date = f"{year}-{month}-{day}"
    file_name = f"{kite_model}_{date}"
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


# -----------------------------------------------------------------------------
# Cycle simulation helper
# -----------------------------------------------------------------------------


def simulate_cycle(
    cycle_id,
    ekf_df,
    flight_df,
    path_to_main,
    downloop_cfg_path,
    m_per_second=1.0,
    npoints_per_second=2.0,
    wind_profile=None,
    winch_models=None,
    use_dynamic: bool = False,
):
    # Use phase-specific duration to size discretizations
    cycle_mask = flight_df["cycle_by_phase"] == cycle_id
    cycle_rows = flight_df[cycle_mask]
    if cycle_rows.empty:
        print(f"Skipping cycle {cycle_id}: no rows")
        return

    combined_exp = []
    combined_exp_t = []
    combined_sim = []
    combined_sim_t = []
    combined_sim_force = []
    phase_labels = []
    phase_stats = []
    combined_series = {
        "tension": {"sim": [], "exp": []},
        "speed_radial": {"sim": [], "exp": []},
        "s_dot": {"sim": [], "exp": []},
        "elevation": {"sim": [], "exp": []},
        "azimuth": {"sim": [], "exp": []},
        "power": {"sim": [], "exp": []},
    }
    t_offset_exp = 0.0
    t_offset_sim = 0.0
    r0 = None
    phase_start_state = None

    def _extract_final_state(phase_obj):
        final_state = {}
        try:
            if hasattr(phase_obj, "states") and phase_obj.states:
                fs = phase_obj.states[-1]
                for key in [
                    "t",
                    "s",
                    "s_dot",
                    "speed_radial",
                    "distance_radial",
                    "input_steering",
                    "tension_tether_ground",
                    "input_depower",
                ]:
                    if key in fs:
                        try:
                            final_state[key] = float(fs[key])
                        except Exception:
                            final_state[key] = fs[key]
            else:
                for key in [
                    "t",
                    "s",
                    "s_dot",
                    "speed_radial",
                    "distance_radial",
                    "input_steering",
                    "tension_tether_ground",
                    "input_depower",
                ]:
                    try:
                        series = np.array(phase_obj.return_variable(key))
                        if series.size > 0:
                            final_state[key] = float(series[-1])
                    except Exception:
                        continue
        except Exception:
            pass
        return final_state

    def _extract_initial_state(phase_obj):
        """Extract the first recorded state from a Phase object as a dict."""
        init_state = {}
        try:
            if hasattr(phase_obj, "states") and phase_obj.states:
                fs = phase_obj.states[0]
                for key in [
                    "t",
                    "s",
                    "s_dot",
                    "speed_radial",
                    "distance_radial",
                    "input_steering",
                    "tension_tether_ground",
                    "input_depower",
                ]:
                    if key in fs:
                        try:
                            init_state[key] = float(fs[key])
                        except Exception:
                            init_state[key] = fs[key]
            else:
                for key in [
                    "t",
                    "s",
                    "s_dot",
                    "speed_radial",
                    "distance_radial",
                    "input_steering",
                    "tension_tether_ground",
                    "input_depower",
                ]:
                    try:
                        series = np.array(phase_obj.return_variable(key))
                        if series.size > 0:
                            init_state[key] = float(series[0])
                    except Exception:
                        continue
        except Exception:
            pass
        return init_state

    # Compute durations per phase within the cycle
    phase_durations = {}
    for ph in [1, 2, 3, 4]:
        pmask = cycle_rows["flight_phase_index"] == ph
        print(f"Cycle {cycle_id}, phase {ph}: {pmask.sum()} rows")
        if pmask.any():
            rows = cycle_rows[pmask]
            phase_durations[ph] = float(rows["time"].iloc[-1] - rows["time"].iloc[0])

    target_phases = [1, 2, 3, 4]

    for target_phase in target_phases:
        phase_duration_s = phase_durations.get(target_phase)
        if phase_duration_s is None:
            print(f"Skipping cycle {cycle_id}: no phase {target_phase} rows")
            continue

        M = max(4, int(np.ceil(phase_duration_s * m_per_second)))
        n_points = max(2, int(np.ceil(phase_duration_s * npoints_per_second)))
        print(
            f"Cycle {cycle_id}: phase {target_phase} duration ~{phase_duration_s:.1f}s -> M={M}, n_points={n_points}"
        )

        # Focus on the target phase portion within the cycle
        mask = cycle_mask & flight_df["flight_phase_index"].isin([target_phase])
        flight_cycle = flight_df[mask].reset_index(drop=True)
        ekf_cycle = ekf_df[mask].reset_index(drop=True)
        if flight_cycle.empty:
            print(f"Skipping cycle {cycle_id}: no rows for phase {target_phase}")
            continue

        pos_x = ekf_cycle["kite_position_x"].to_numpy()
        pos_y = ekf_cycle["kite_position_y"].to_numpy()
        pos_z = ekf_cycle["kite_position_z"].to_numpy()

        azimuth = flight_cycle["kite_azimuth"].to_numpy()
        elevation = flight_cycle["kite_elevation"].to_numpy()
        distance_radial = np.linalg.norm(np.column_stack((pos_x, pos_y, pos_z)), axis=1)
        tether_force = flight_cycle["ground_tether_force"].to_numpy()
        reelout_speed = flight_cycle["tether_reelout_speed"].to_numpy()
        time_exp = flight_cycle["time"].to_numpy(dtype=float)

        if r0 is None:
            r0 = float(distance_radial[0])
        s_samples = np.linspace(0.0, 1.0, len(azimuth), endpoint=True)

        pattern, C_phi, C_beta = fit_bspline_pattern_to_trajectory(
            spline_type="open",
            M=M,
            s_init=0.0,
            s_final=1.0,
            az_target=azimuth,
            el_target=elevation,
            s_samples=s_samples,
            downloops=True,
        )

        # Winch fit: prefer full-flight phase-specific model if available
        slope_winch = None
        offset_winch = None
        if winch_models and target_phase in winch_models:
            slope_winch = winch_models[target_phase]["slope_winch_ro"]
            offset_winch = winch_models[target_phase]["offset_winch_ro"]
            print(
                f"Winch model (full-flight phase {target_phase}): Tether force = {slope_winch:.2e} * reel-out speed + {offset_winch:.2f} N"
            )
        else:
            valid_mask = np.isfinite(tether_force) & np.isfinite(reelout_speed)
            valid_mask &= tether_force > 1000
            coeffs = np.polyfit(reelout_speed[valid_mask], tether_force[valid_mask], 1)
            slope_winch = coeffs[0]
            offset_winch = -coeffs[1] / coeffs[0] if coeffs[0] != 0 else 0.0
            print(
                f"Winch model (per-phase fallback): Tether force = {slope_winch:.2e} * reel-out speed + {offset_winch:.2f} N"
            )
        with open(downloop_cfg_path, "r") as f:
            downloop_cfg = yaml.safe_load(f)

        phase_yaml = downloop_cfg.get("reelout", {})
        radial_params = phase_yaml.get("radial_parameters", {})
        sim_params = phase_yaml.get("sim_parameters", {}).copy()
        sim_params["n_points"] = n_points
        # Spline is fitted on [0, 1]; force the sim grid to match regardless of yaml
        sim_params["start_angle"] = 0.0
        sim_params["end_angle"] = 1.0

        # Wind: default to EKF-derived profile averaged over ±2.5 s, resampled to sim s-grid
        wind_col = "wind_speed_horizontal"

        wind_samples = ekf_cycle[wind_col].to_numpy(dtype=float)
        wind_ref = float(np.nanmean(wind_samples))

        if wind_samples.size > 0:
            window_half = 2.5  # seconds on each side -> 5 s window
            wind_smooth = np.full_like(wind_samples, np.nan, dtype=float)
            for i, t_val in enumerate(time_exp):
                win_mask = np.abs(time_exp - t_val) <= window_half
                vals = wind_samples[win_mask]
                if vals.size > 0 and np.isfinite(vals).any():
                    wind_smooth[i] = float(np.nanmean(vals))
            if not np.isfinite(wind_ref):
                wind_ref = 0.0
            wind_smooth = np.where(np.isfinite(wind_smooth), wind_smooth, wind_ref)
            s_sim = np.linspace(0.0, 1.0, sim_params["n_points"] + 1, endpoint=True)
            wind_profile = np.interp(s_sim, s_samples, wind_smooth)
            sim_params["wind_speed_profile"] = wind_profile.tolist()
            wind_ref = float(np.nanmean(wind_profile))
        else:
            print("No wind samples available; defaulting wind_ref to 0")
            wind_ref = 0.0

        print(f"Using wind reference speed {wind_ref:.2f} m/s from EKF-derived profile")

        wind_model = Wind(
            wind_model="uniform",
            speed_wind_ref=wind_ref,
            direction_wind=0,
        )

        radial_params["slope_winch_ro"] = slope_winch
        radial_params["offset_winch_ro"] = offset_winch
        print(radial_params)
        path_params = {
            "r0": r0,
            "M": int(M),
            "C_phi": C_phi.full().flatten().tolist(),
            "C_beta": C_beta.full().flatten().tolist(),
            "s_init": 0.0,
            "s_final": 1.0,
        }

        # Use final state of previous phase as the next initializer

        current_start_state = {
            "t": 0,
            "s": 0,
            "s_dot": 0.02,
            "input_steering": 0,
            "tension_tether_ground": 4e3,
            "speed_radial": 0,
            "distance_radial": path_params["r0"],
        }
        if target_phase == 1:
            current_start_state["s_dot"] = 0.05
        else:
            current_start_state["s_dot"] = 0.1

        if "up" in flight_cycle.columns:
            u_p_raw = flight_cycle["up"].to_numpy(dtype=float)
            s_meas = np.linspace(0.0, 1.0, u_p_raw.size, endpoint=True)
            s_sim = np.linspace(0.0, 1.0, sim_params["n_points"] + 1, endpoint=True)
            u_dep_profile = np.interp(s_sim, s_meas, u_p_raw)
            sim_params["input_depower_profile"] = u_dep_profile.tolist()
            sim_params["input_depower"] = float(u_dep_profile[0])
            print(
                f"Using depower profile from measurements (resampled to {sim_params['n_points']+1} points)"
            )
        else:
            sim_params["input_depower"] = 0.0

        phase_config = {
            "pattern_type": "spline_open",
            "path_parameters": path_params,
            "radial_parameters": radial_params,
            "sim_parameters": sim_params,
        }

        print("\nPhase config preview (path parameters shortened):")
        print(
            {
                "pattern_type": phase_config["pattern_type"],
                "r0": phase_config["path_parameters"]["r0"],
                "M": phase_config["path_parameters"]["M"],
                "sim_n_points": phase_config["sim_parameters"]["n_points"],
                "u_dep_profile_len": len(sim_params.get("input_depower_profile", [])),
            }
        )

        system_model = create_system_model_from_yaml(LEI_V3_SYSTEM_CONFIG)
        system_model.wind = wind_model

        if use_dynamic:
            # First phase: get a warm-start from a quasi-steady run if we don't
            # already have a dynamic start state from a previous dynamic run.
            if phase_start_state is None:
                qs_phase = Phase(
                    system_model=system_model,
                    pattern_config=phase_config,
                    start_state=current_start_state,
                    quasi_steady=True,
                )
                qs_phase_obj, _ = qs_phase.run_simulation(
                    run_plots=False, start_state=current_start_state
                )
                # Use the first state from the quasi-steady run to initialize
                # the dynamic simulation for the first phase.
                init_state = _extract_initial_state(qs_phase_obj) or current_start_state
            else:
                init_state = phase_start_state

            dyn_phase = Phase(
                system_model=system_model,
                pattern_config=phase_config,
                start_state=init_state,
                quasi_steady=False,
            )
            phase, _ = dyn_phase.run_simulation(run_plots=False, start_state=init_state)
        else:
            phase_sim = Phase(
                system_model=system_model,
                pattern_config=phase_config,
                start_state=current_start_state,
            )

            phase, _ = phase_sim.run_simulation(
                run_plots=False, start_state=current_start_state
            )
        print(
            f"Phase {target_phase} simulation finished. Final radial distance:",
            phase.return_variable("distance_radial")[-1],
        )
        r0 = phase.return_variable("distance_radial")[-1]
        phase_energy = float(getattr(phase, "energy", np.nan))
        phase_time = float(getattr(phase, "total_time", np.nan))
        phase_end_r = float(r0)

        # Experimental energy: sum(F * v * dt) using left rule to match sim convention
        dt_exp = np.diff(time_exp)
        f_exp = tether_force[:-1]
        v_exp = reelout_speed[:-1]
        valid = np.isfinite(dt_exp) & np.isfinite(f_exp) & np.isfinite(v_exp)
        exp_energy = float(np.sum(f_exp[valid] * v_exp[valid] * dt_exp[valid]))
        exp_time = float(np.sum(dt_exp[valid]))
        exp_power = exp_energy / exp_time if exp_time > 0 else np.nan
        sim_power = phase_energy / phase_time if phase_time > 0 else np.nan

        phase_stats.append(
            {
                "phase": target_phase,
                "energy": phase_energy,
                "time": phase_time,
                "end_r": phase_end_r,
                "exp_energy": exp_energy,
                "exp_time": exp_time,
                "exp_power": exp_power,
                "sim_power": sim_power,
            }
        )
        # force_tether = phase.return_variable("tension_tether_ground")
        # speed_radial = phase.return_variable("speed_radial")

        # Collect trajectories for combined plots
        exp_pts = np.column_stack((pos_x, pos_y, pos_z))
        exp_t = (
            flight_cycle["time"].to_numpy() - float(flight_cycle["time"].iloc[0])
        ) + t_offset_exp
        combined_exp.append(exp_pts)
        combined_exp_t.append(exp_t)
        phase_labels.append(target_phase)

        try:
            sim_x = np.array(phase.return_variable("x"))
            sim_y = np.array(phase.return_variable("y"))
            sim_z = np.array(phase.return_variable("z"))
            sim_t = np.array(phase.return_variable("t")) + t_offset_sim
            sim_pts = np.column_stack((sim_x, sim_y, sim_z))
            combined_sim.append(sim_pts)
            combined_sim_t.append(sim_t)
            sim_force = np.array(phase.return_variable("tension_tether_ground"))
            combined_sim_force.append(sim_force)
        except Exception:
            print(f"Phase {target_phase}: simulation trajectory (x,y,z) not available")
            combined_sim.append(None)
            combined_sim_t.append(None)
            combined_sim_force.append(None)

        # Collect time-series for combined plots across phases
        try:
            sim_t_local = np.array(phase.return_variable("t"))
            exp_t_local = time_exp - float(time_exp[0]) + t_offset_exp
            combined_series["tension"]["exp"].append((exp_t_local, tether_force))
            combined_series["tension"]["sim"].append(
                (
                    sim_t_local + t_offset_sim,
                    phase.return_variable("tension_tether_ground"),
                )
            )
            combined_series["speed_radial"]["exp"].append(
                (exp_t_local[:-1], reelout_speed[:-1])
            )
            combined_series["speed_radial"]["sim"].append(
                (sim_t_local + t_offset_sim, phase.return_variable("speed_radial"))
            )
            combined_series["s_dot"]["sim"].append(
                (sim_t_local + t_offset_sim, phase.return_variable("s_dot"))
            )
            combined_series["elevation"]["exp"].append((exp_t_local, elevation))
            combined_series["elevation"]["sim"].append(
                (sim_t_local + t_offset_sim, phase.return_variable("angle_elevation"))
            )
            combined_series["azimuth"]["exp"].append((exp_t_local, azimuth))
            combined_series["azimuth"]["sim"].append(
                (sim_t_local + t_offset_sim, phase.return_variable("angle_azimuth"))
            )
            sim_power_ts = np.array(
                phase.return_variable("tension_tether_ground")
            ) * np.array(phase.return_variable("speed_radial"))
            exp_len = min(len(exp_t_local[:-1]), len(f_exp), len(v_exp))
            combined_series["power"]["exp"].append(
                (exp_t_local[:exp_len], (f_exp[:exp_len] * v_exp[:exp_len]))
            )
            combined_series["power"]["sim"].append(
                (sim_t_local + t_offset_sim, sim_power_ts)
            )
        except Exception as exc:
            print(
                f"Phase {target_phase}: skipping combined time-series capture ({exc})"
            )

        # Advance offset so next phase starts where this one ended
        t_offset_exp += float(phase_duration_s)
        t_offset_sim += float(phase.return_variable("t")[-1])

        # Prepare initializer for next phase from final simulated state
        try:
            final_state = _extract_final_state(phase)
            next_state = {**current_start_state}
            for key in [
                "s",
                # "s_dot",
                "speed_radial",
                "distance_radial",
                "input_steering",
                "tension_tether_ground",
                "input_depower",
            ]:
                if key in final_state:
                    next_state[key] = final_state[key]
            next_state["s_dot"] = float(final_state.get("s_dot", 0.1))
            next_state["t"] = 0.0
            phase_start_state = next_state
        except Exception as exc:
            print(
                f"Phase {target_phase}: unable to capture final state for warm start ({exc})"
            )

    if phase_stats:
        total_energy = float(np.nansum([p["energy"] for p in phase_stats]))
        total_time = float(np.nansum([p["time"] for p in phase_stats]))
        cycle_power = total_energy / total_time if total_time > 0 else np.nan
        final_r = phase_stats[-1]["end_r"]
        total_energy_exp = float(np.nansum([p["exp_energy"] for p in phase_stats]))
        total_time_exp = float(np.nansum([p["exp_time"] for p in phase_stats]))
        cycle_power_exp = (
            total_energy_exp / total_time_exp if total_time_exp > 0 else np.nan
        )
        print("\nCycle summary:")
        for ps in phase_stats:
            print(
                f"  Phase {ps['phase']}: sim_energy={ps['energy']:.2f} J, sim_time={ps['time']:.2f} s, sim_power={ps['sim_power']:.2f} W | exp_energy={ps['exp_energy']:.2f} J, exp_time={ps['exp_time']:.2f} s, exp_power={ps['exp_power']:.2f} W, end_r={ps['end_r']:.2f} m"
            )
        print(
            f"  Cycle totals: sim_energy={total_energy:.2f} J, sim_time={total_time:.2f} s, sim_power={cycle_power:.2f} W | exp_energy={total_energy_exp:.2f} J, exp_time={total_time_exp:.2f} s, exp_power={cycle_power_exp:.2f} W, final tether length={final_r:.2f} m"
        )

    # Combined 3D plot and time evolution if we have data
    if combined_exp:
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111, projection="3d")
        for pts, lbl in zip(combined_exp, phase_labels):
            ax.plot(
                pts[:, 0], pts[:, 1], pts[:, 2], label=f"Exp phase {lbl}", linewidth=1.5
            )
        for pts, lbl in zip(combined_sim, phase_labels):
            if pts is None:
                continue
            ax.plot(
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],
                "--",
                label=f"Sim phase {lbl}",
                linewidth=1.2,
            )
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        ax.legend()
        ax.set_title(f"Cycle {cycle_id} 3D trajectory (exp vs sim)")

        plt.figure(figsize=(7, 4))
        for t_arr, pts, lbl in zip(combined_exp_t, combined_exp, phase_labels):
            r = np.linalg.norm(pts, axis=1)
            plt.plot(t_arr, r, label=f"Exp phase {lbl}")
        for t_arr, pts, lbl in zip(combined_sim_t, combined_sim, phase_labels):
            if pts is None or t_arr is None:
                continue
            r = np.linalg.norm(pts, axis=1)
            plt.plot(t_arr, r, "--", label=f"Sim phase {lbl}")
        plt.xlabel("Time in phase (s)")
        plt.ylabel("Radial distance (m)")
        plt.grid(True)
        plt.legend()
        plt.title(f"Cycle {cycle_id} radial distance vs time")

    # Simulation-only 3D trajectory colored by tether force
    if any(pts is not None for pts in combined_sim) and any(
        f is not None for f in combined_sim_force
    ):
        sim_segments = []
        force_segments = []
        phase_first_points = []
        for pts, f in zip(combined_sim, combined_sim_force):
            if pts is None or f is None:
                continue
            n = min(len(pts), len(f))
            if n == 0:
                continue
            sim_segments.append(pts[1:n])
            force_segments.append(f[1:n])
            phase_first_points.append((pts[0], f[0]))

        if sim_segments:
            pts_all = np.concatenate(sim_segments, axis=0)
            force_all = np.concatenate(force_segments, axis=0)
            fig = plt.figure(figsize=(7, 5))
            ax = fig.add_subplot(111, projection="3d")
            fig.patch.set_facecolor("white")
            ax.set_facecolor("white")
            xaxis = getattr(ax, "w_xaxis", ax.xaxis)
            yaxis = getattr(ax, "w_yaxis", ax.yaxis)
            zaxis = getattr(ax, "w_zaxis", ax.zaxis)
            for pane in [xaxis.pane, yaxis.pane, zaxis.pane]:
                pane.set_facecolor("white")
                # pane.set_edgecolor("black")
                # pane.set_alpha(1.0)
            sc = ax.scatter(
                pts_all[:, 0],
                pts_all[:, 1],
                pts_all[:, 2],
                c=force_all,
                cmap=custom_cmap,
                s=6,
                depthshade=False,
                zorder=0,
                alpha=0.8,
            )
            if phase_first_points:
                first_pts = np.array([p for p, _ in phase_first_points])
                ax.scatter(
                    first_pts[:, 0],
                    first_pts[:, 1],
                    first_pts[:, 2],
                    c="k",
                    s=28,
                    depthshade=False,
                    zorder=50,
                    edgecolors="w",
                    linewidths=0.6,
                    label="Phase start",
                )
            cb = fig.colorbar(sc, ax=ax, pad=0.1)
            cb.ax.set_facecolor("white")
            cb.set_label("Tether force (N)")
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            ax.set_zlabel("z (m)")
            plt.tight_layout()
            # plt.savefig(f"cycle_{cycle_id}_sim_trajectory_force.pdf")
            # ax.set_title(f"Cycle {cycle_id} sim trajectory colored by tether force")
            if phase_first_points:
                ax.legend()

    # Combined time-series across phases for key variables
    if combined_series["tension"]["sim"] or combined_series["speed_radial"]["sim"]:
        var_labels = {
            "tension": "Tether tension (N)",
            "speed_radial": "Radial speed (m/s)",
            "s_dot": "s_dot (rad/s)",
            "elevation": "Elevation (rad)",
            "azimuth": "Azimuth (rad)",
            "power": "Power (W)",
        }
        for var in [
            "tension",
            "speed_radial",
            "s_dot",
            "elevation",
            "azimuth",
            "power",
        ]:
            if not combined_series[var]["sim"] and not combined_series[var]["exp"]:
                continue
            plt.figure(figsize=(7, 3))
            for t_arr, vals in combined_series[var]["exp"]:
                plt.plot(t_arr, vals, label="Exp", linewidth=1.2)
            for t_arr, vals in combined_series[var]["sim"]:
                plt.plot(t_arr, vals, "--", label="Sim", linewidth=1.2)
            plt.xlabel("Time in cycle (s)")
            plt.ylabel(var_labels.get(var, var))
            plt.grid(True)
            plt.legend()
            plt.title(f"Cycle {cycle_id} {var_labels.get(var, var)}")


# -----------------------------------------------------------------------------
# Main routine
# -----------------------------------------------------------------------------


def main():
    # Configuration
    year, month, day = "2019", "10", "08"
    kite_model = "v3"
    path_to_main = "./data/LEI-V3-KITE"
    cycle_id = 68  # set to an int to run a single cycle, None to run all detected
    downloop_cfg_path = str(LEI_V3_DOWNLOOP_SPLINE_CONFIG)

    # Discretization level selector: choose one of ["coarse", "medium", "fine"]
    discretization_level = "fine"
    level_map = {
        "coarse": {"m_per_second": 0.5, "npoints_per_second": 1.0},
        "medium": {"m_per_second": 1.0, "npoints_per_second": 2.0},
        "fine": {"m_per_second": 1.0, "npoints_per_second": 6.0},
    }
    if discretization_level not in level_map:
        raise ValueError(
            f"Invalid discretization_level '{discretization_level}', choose from {list(level_map.keys())}"
        )
    level_cfg = level_map[discretization_level]

    ekf_df, flight_df, _ = read_results(
        year, month, day, kite_model, addition="", path_to_main=path_to_main
    )

    winch_models = fit_winch_models_full_flight(flight_df)

    if "flight_phase_index" not in flight_df:
        raise RuntimeError("flight_phase_index column is required to derive cycles")

    flight_df["cycle_by_phase"] = cycles_from_phases(flight_df["flight_phase_index"])
    available_cycles = [
        int(c) for c in np.unique(flight_df["cycle_by_phase"]) if c >= 0
    ]

    target_cycles = [cycle_id] if cycle_id is not None else available_cycles
    if not target_cycles:
        raise RuntimeError("No cycles detected from phase sequence 1-2-3-4")

    for cid in target_cycles:
        simulate_cycle(
            cycle_id=cid,
            ekf_df=ekf_df,
            flight_df=flight_df,
            path_to_main=path_to_main,
            downloop_cfg_path=downloop_cfg_path,
            m_per_second=level_cfg["m_per_second"],
            npoints_per_second=level_cfg["npoints_per_second"],
            wind_profile=None,
            winch_models=winch_models,
            use_dynamic=False,
        )

    plt.show()


if __name__ == "__main__":
    main()
