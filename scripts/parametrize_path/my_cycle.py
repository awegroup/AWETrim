import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import my_reel_in
import my_reelout_cst
from awetrim.utils.color_palette import set_plot_style
from awetrim.utils.defaults import PLOT_LABELS

REEL_OUT_CSV = my_reelout_cst.REEL_OUT_OUTPUT_PATH
REEL_IN_CSV = my_reel_in.REEL_IN_OUTPUT_PATH
CYCLE_CSV = Path("results/timeseries/cycle_timeseries.csv")

PLOT_VARIABLES = my_reel_in.PLOT_VARIABLES
BASE_VARIABLES = my_reel_in.BASE_VARIABLES
DERIVED_VARIABLES = my_reel_in.DERIVED_VARIABLES
CSV_HEADER = my_reel_in.CSV_HEADER
NUMERIC_FIELDS = ["time"] + BASE_VARIABLES + DERIVED_VARIABLES


def ensure_timeseries():
    if not REEL_OUT_CSV.exists():
        my_reelout_cst.main(run_plots=False, save_csv=True)
    if not REEL_IN_CSV.exists():
        my_reel_in.main(run_plots=False, save_csv=True)


def load_rows(path: Path):
    rows = []
    with path.open(newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            parsed = {key: row[key] for key in row.keys()}
            for field in NUMERIC_FIELDS:
                value = parsed.get(field)
                try:
                    parsed[field] = float(value)
                except (TypeError, ValueError):
                    parsed[field] = np.nan
            rows.append(parsed)
    return rows


def combine_rows(reel_out_rows, reel_in_rows):
    combined = list(reel_out_rows)
    if reel_out_rows:
        max_time = max(
            (row["time"] for row in reel_out_rows if np.isfinite(row["time"])),
            default=0.0,
        )
    else:
        max_time = 0.0
    for row in reel_in_rows:
        adjusted = row.copy()
        time_val = adjusted["time"]
        if not np.isfinite(time_val):
            time_val = 0.0
        adjusted["time"] = time_val + max_time
        combined.append(adjusted)
    return combined


def aggregate_rows(rows):
    aggregated = {
        "quasi_steady": {"t": [], "segment": []},
        "dynamic": {"t": [], "segment": []},
    }
    for var_name in BASE_VARIABLES + DERIVED_VARIABLES:
        aggregated["quasi_steady"][var_name] = []
        aggregated["dynamic"][var_name] = []

    for row in rows:
        sim_key = row.get("simulation")
        if sim_key not in aggregated:
            continue
        aggregated[sim_key]["t"].append(row.get("time", np.nan))
        aggregated[sim_key]["segment"].append(row.get("segment", ""))
        for var_name in BASE_VARIABLES + DERIVED_VARIABLES:
            aggregated[sim_key][var_name].append(row.get(var_name, np.nan))
    return aggregated


def plot_timeseries(aggregated):
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
            times = np.asarray(aggregated[sim_key]["t"], dtype=float)
            values = np.asarray(aggregated[sim_key][var_name], dtype=float)
            if times.size and values.size:
                ax.plot(times, values, label=sim_label)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.3)
    axes[-1].set_xlabel("Time [s]")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(loc="best")
    plt.tight_layout()
    plt.show()


def plot_trajectory(aggregated):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    plotted_any = False
    for sim_key, sim_label in [
        ("quasi_steady", "Quasi-Steady"),
        ("dynamic", "Dynamic"),
    ]:
        x_vals = np.asarray(aggregated[sim_key]["x_position"], dtype=float)
        y_vals = np.asarray(aggregated[sim_key]["y_position"], dtype=float)
        z_vals = np.asarray(aggregated[sim_key]["z_position"], dtype=float)
        finite_mask = np.isfinite(x_vals) & np.isfinite(y_vals) & np.isfinite(z_vals)
        if finite_mask.any():
            ax.plot(
                x_vals[finite_mask], y_vals[finite_mask], z_vals[finite_mask], label=sim_label
            )
            plotted_any = True
    if plotted_any:
        ax.set_xlabel(PLOT_LABELS.get("x", "x"))
        ax.set_ylabel(PLOT_LABELS.get("y", "y"))
        ax.set_zlabel(PLOT_LABELS.get("z", "z"))
        x_all = []
        y_all = []
        z_all = []
        for sim_key in ["quasi_steady", "dynamic"]:
            x_arr = np.asarray(aggregated[sim_key]["x_position"], dtype=float)
            y_arr = np.asarray(aggregated[sim_key]["y_position"], dtype=float)
            z_arr = np.asarray(aggregated[sim_key]["z_position"], dtype=float)
            finite = np.isfinite(x_arr) & np.isfinite(y_arr) & np.isfinite(z_arr)
            if finite.any():
                x_all.append(x_arr[finite])
                y_all.append(y_arr[finite])
                z_all.append(z_arr[finite])
        if x_all:
            x_stack = np.concatenate(x_all)
            y_stack = np.concatenate(y_all)
            z_stack = np.concatenate(z_all)
            ranges = np.array([np.ptp(x_stack), np.ptp(y_stack), np.ptp(z_stack)])
            overall = np.nanmax(ranges) if ranges.size else 0.0
            if overall > 0:
                mid_x = 0.5 * (np.nanmax(x_stack) + np.nanmin(x_stack))
                mid_y = 0.5 * (np.nanmax(y_stack) + np.nanmin(y_stack))
                mid_z = 0.5 * (np.nanmax(z_stack) + np.nanmin(z_stack))
                half = overall / 2.0
                ax.set_xlim(mid_x - half, mid_x + half)
                ax.set_ylim(mid_y - half, mid_y + half)
                ax.set_zlim(mid_z - half, mid_z + half)
                ax.set_box_aspect([1, 1, 1])
        ax.legend(loc="best")
        plt.tight_layout()
        plt.show()
    else:
        plt.close(fig)


def write_cycle_csv(rows):
    CYCLE_CSV.parent.mkdir(parents=True, exist_ok=True)
    with CYCLE_CSV.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(CSV_HEADER)
        for row in rows:
            data_row = [row.get(field, "") for field in CSV_HEADER]
            writer.writerow(data_row)
    print(f"Saved combined cycle timeseries to {CYCLE_CSV}")


def main(run_plots=True):
    ensure_timeseries()
    reel_out_rows = load_rows(REEL_OUT_CSV)
    reel_in_rows = load_rows(REEL_IN_CSV)
    combined_rows = combine_rows(reel_out_rows, reel_in_rows)

    aggregated = aggregate_rows(combined_rows)
    print(aggregated.keys())
    if run_plots:
        plot_timeseries(aggregated)
        plot_trajectory(aggregated)
    write_cycle_csv(combined_rows)

    return aggregated

if __name__ == "__main__":
    main()