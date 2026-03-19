"""Plot optimized trajectories for downloop, uploop, and helix patterns.

This script loads optimized reelout configurations and produces two figure types:
- Wind sweep: for a single shear (default z0 = 0.03) across wind speeds 6, 12, 18.
- Shear sweep: for a single wind speed (default 12) across shears 0.0002, 0.03, 0.3.
"""

from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np

from awetrim.environment.Wind import Wind
from awetrim.system.system_model import create_system_model_from_yaml
from awetrim.timeseries.reelout_phase import Reelout
from awetrim.utils.utils import load_cycle_config_from_yaml
from awetrim.utils.color_palette import set_plot_style, get_color_list


# Base configuration
KITE_CONFIG_PATH = Path("data/LEI-V3-KITE/v3_kite_input.yaml")
CONFIG_ROOT = Path("results/optimized_configs")
SHEARS = [0.0002, 0.03, 0.3]
WIND_SPEED = 12
WIND_SWEEP = (6, 12, 18, 22)
SHEAR_FOR_WIND_SWEEP = 0.03


PATTERN_FOLDERS = {
    "Down-loop": CONFIG_ROOT / "downloops",
    "Up-loop": CONFIG_ROOT / "uploops",
    "Helix": CONFIG_ROOT / "helix",
}


START_STATE = {
    "t": 0.0,
    "s": 0.0,
    "s_dot": 2,
    "speed_radial": 0,
    "input_steering": 0.0,
    "tension_tether_ground": 8.4e5,
}


def build_wind_model(speed_wind_at_100: float, z0: float) -> Wind:
    """Create a logarithmic wind model anchored at 100 m."""

    wind_model = Wind(wind_model="logarithmic", z0=z0)
    wind_model.speed_friction = 0.41 * speed_wind_at_100 / np.log(100 / wind_model.z0)
    return wind_model


def find_config(directory: Path, wind_speed: float, shear: float) -> Path:
    """Return the first matching optimized config path for the requested wind and shear."""

    shear_str = f"{shear:g}"
    # Use underscores as word boundaries to avoid matching e.g. wind_16 when looking for wind_6
    pattern = f"*wind_{wind_speed}_z0_{shear_str}*logarithmic_spline.yaml"
    matches = sorted(directory.glob(pattern))
    if not matches:
        # Fallback to older naming format without underscores
        pattern = f"*wind{wind_speed}z0{shear_str}*logarithmic_spline.yaml"
        matches = sorted(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No config found in {directory} for wind={wind_speed}, z0={shear_str}"
        )

    # Prefer config starting with "depower_" if it exists
    depower_matches = [m for m in matches if m.name.startswith("depower_")]
    if depower_matches:
        result = depower_matches[0]
    else:
        result = matches[0]

    print(f"Found config: {result.name}")
    return result


def simulate_phase(
    config_path: Path, wind_speed: float, shear: float, start_state: dict | None = None
):
    """Load a config, attach the requested wind model, and run a quasi-steady simulation."""

    pattern_config, _ = load_cycle_config_from_yaml(config_path)
    system_model = create_system_model_from_yaml(yaml_path=KITE_CONFIG_PATH)
    system_model.wind = build_wind_model(speed_wind_at_100=wind_speed, z0=shear)
    reelout = Reelout(
        system_model=system_model, pattern_config=pattern_config, depower=0
    )
    start_state["distance_radial"] = pattern_config["path_parameters"]["r0"]
    phase, _, start_state = reelout.run_simulation(
        run_plots=False,
        phase_sim=True,
        start_state=start_state,
        return_start_state=True,
    )
    return phase, start_state


def get_power_metrics(phase):
    """Return avg/min/max/delta power in kW using energy metrics and timeseries."""
    em = phase.energy_metrics()
    va = phase.return_variable("speed_apparent_wind")
    print("Average apparent wind speed (m/s):", max(va))
    avg_w = float(em.get("avg_power", 0.0))
    tension = phase.return_variable("tension_tether_ground")
    vr = phase.return_variable("speed_radial")
    p_ts = tension * vr
    min_w = float(np.min(p_ts))
    max_w = float(np.max(p_ts))
    x = phase.return_variable("x")
    y = phase.return_variable("y")
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    projected_area_m2 = float((x_max - x_min) * (y_max - y_min))
    ground_power_density = (
        (avg_w / projected_area_m2) if projected_area_m2 > 0 else float("nan")
    )

    return {
        "avg_kW": avg_w / 1000.0,
        "min_kW": min_w / 1000.0,
        "max_kW": max_w / 1000.0,
        "delta_kW": (max_w - min_w) / 1000.0,
        "ground_power_density_W_per_m2": ground_power_density,
    }


def plot_trajectories(
    shears: Iterable[float] = SHEARS,
    wind_speed: float = WIND_SPEED,
    start_state: dict = START_STATE,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot 2x3 grid: top row y-z, bottom row x-z for downloop, uploop, helix."""

    fig, axes = plt.subplots(2, 3, figsize=(8, 4))
    colors = get_color_list()[1::]

    # First pass: collect all data and compute global limits
    all_x, all_y, all_z = [], [], []
    trajectory_data = {}

    for col, (pattern_name, folder) in enumerate(PATTERN_FOLDERS.items()):
        trajectory_data[col] = []
        start_state = START_STATE.copy()  # Reset start state for each pattern
        for shear in shears:
            start_state = START_STATE.copy()  # Reset start state for each simulation
            if pattern_name == "Helix":
                start_state["s_dot"] = 2
            config_path = find_config(folder, wind_speed, shear)
            phase, start_state = simulate_phase(
                config_path, wind_speed, shear, start_state=start_state
            )
            x = phase.return_variable("x")
            y = phase.return_variable("y")
            z = phase.return_variable("z")
            p_metrics = get_power_metrics(phase)
            trajectory_data[col].append((x, y, z, shear, p_metrics))
            all_x.extend(x)
            all_y.extend(y)
            all_z.extend(z)

    # Compute global limits with small padding
    x_min, x_max = np.min(all_x), np.max(all_x)
    y_min, y_max = np.min(all_y), np.max(all_y)
    z_min, z_max = np.min(all_z), np.max(all_z)
    x_pad = 0.05 * (x_max - x_min)
    y_pad = 0.05 * (y_max - y_min)
    z_pad = 0.05 * (z_max - z_min)

    # Second pass: plot with consistent limits
    for col, (pattern_name, folder) in enumerate(PATTERN_FOLDERS.items()):
        for (x, y, z, shear, p_metrics), color in zip(trajectory_data[col], colors):
            # Top row: y-z (crosswind-altitude)
            axes[0, col].plot(y, z, label=f"z0={shear}", color=color)
            # Bottom row: x-z (downwind-altitude)
            axes[1, col].plot(x, z, label=f"z0={shear}", color=color)

        # Top row titles and labels
        axes[0, col].set_title(f"{pattern_name}")
        axes[0, col].set_xlabel("Crosswind y (m)")
        axes[0, col].grid(True, linestyle=":", linewidth=0.7)
        axes[0, col].set_xlim(y_min - y_pad, y_max + y_pad)
        axes[0, col].set_ylim(z_min - z_pad, z_max + z_pad)

        # Bottom row labels
        axes[1, col].set_xlabel("Downwind x (m)")
        axes[1, col].grid(True, linestyle=":", linewidth=0.7)
        axes[1, col].set_xlim(x_min - x_pad, x_max + x_pad)
        axes[1, col].set_ylim(z_min - z_pad, z_max + z_pad)

    # Plot discontinuous altitude lines at z=50m for reference
    for ax in axes.flatten():
        ax.axhline(y=50, color="gray", linestyle="--", linewidth=0.7)

    axes[0, 0].set_ylabel("Altitude z (m)")
    axes[1, 0].set_ylabel("Altitude z (m)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig("results/figures/torque2026/optimized_trajectories_shear.pdf")

    # Projected ground area (rectangle) in m^2
    projected_area_m2 = float((x_max - x_min) * (y_max - y_min))

    # Compact tables
    print("\n" + "=" * 90)
    print(f"Power Summary (wind={wind_speed} m/s) — Area={projected_area_m2:.0f} m^2")
    print("=" * 90)
    # Table 1: Power in kW: avg (min–max) Δ
    header = "Pattern".ljust(12) + "".join(
        f"z0={shear:<10}".ljust(22) for shear in shears
    )
    print(header)
    print("-" * 90)
    for col, (pattern_name, _) in enumerate(PATTERN_FOLDERS.items()):
        row = pattern_name.ljust(12)
        for _, _, _, _, pm in trajectory_data[col]:
            row += f"{pm['avg_kW']:.1f} ({pm['min_kW']:.1f}–{pm['max_kW']:.1f}) Δ{pm['delta_kW']:.1f}".ljust(
                22
            )
        print(row)
    # Table 2: Power density (W/m^2): avg/A
    print("\nPower density ρ = P_avg / Area [W/m^2]")
    header_pd = "Pattern".ljust(12) + "".join(
        f"z0={shear:<10}".ljust(16) for shear in shears
    )
    print(header_pd)
    print("-" * 90)
    for col, (pattern_name, _) in enumerate(PATTERN_FOLDERS.items()):
        row = pattern_name.ljust(12)
        for _, _, _, _, pm in trajectory_data[col]:
            rho = (
                (pm["ground_power_density_W_per_m2"])
                if projected_area_m2 > 0
                else float("nan")
            )
            row += f"{rho:.2f}".ljust(16)
        print(row)
    print("=" * 90 + "\n")

    return fig, axes


def plot_wind_sweep_for_shear(
    shear: float = SHEAR_FOR_WIND_SWEEP,
    wind_speeds: Iterable[float] = WIND_SWEEP,
    start_state: dict = START_STATE,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot 2x3 grid: top row y-z, bottom row x-z for multiple wind speeds."""

    fig, axes = plt.subplots(2, 3, figsize=(8, 4))
    colors = get_color_list()[1::]

    # First pass: collect all data and compute global limits
    all_x, all_y, all_z = [], [], []
    trajectory_data = {}

    for col, (pattern_name, folder) in enumerate(PATTERN_FOLDERS.items()):
        trajectory_data[col] = []
        start_state = START_STATE.copy()  # Reset start state for each pattern
        for wind_speed in wind_speeds:
            config_path = find_config(folder, wind_speed, shear)
            phase, start_state = simulate_phase(
                config_path, wind_speed, shear, start_state=start_state
            )
            get_power_metrics(phase)
            x = phase.return_variable("x")
            y = phase.return_variable("y")
            z = phase.return_variable("z")
            trajectory_data[col].append((x, y, z, wind_speed))
            all_x.extend(x)
            all_y.extend(y)
            all_z.extend(z)

    # Compute global limits with small padding
    x_min, x_max = np.min(all_x), np.max(all_x)
    y_min, y_max = np.min(all_y), np.max(all_y)
    z_min, z_max = np.min(all_z), np.max(all_z)
    x_pad = 0.05 * (x_max - x_min)
    y_pad = 0.05 * (y_max - y_min)
    z_pad = 0.05 * (z_max - z_min)

    # Second pass: plot with consistent limits
    for col, (pattern_name, folder) in enumerate(PATTERN_FOLDERS.items()):
        for (x, y, z, wind_speed), color in zip(trajectory_data[col], colors):
            # Top row: y-z (crosswind-altitude)
            axes[0, col].plot(y, z, label=f"wind={wind_speed}", color=color)
            # Bottom row: x-z (downwind-altitude)
            axes[1, col].plot(x, z, label=f"wind={wind_speed}", color=color)

        # Top row titles and labels
        axes[0, col].set_title(f"{pattern_name}")
        axes[0, col].set_xlabel("Crosswind y (m)")
        axes[0, col].grid(True, linestyle=":", linewidth=0.7)
        axes[0, col].set_xlim(y_min - y_pad, y_max + y_pad)
        axes[0, col].set_ylim(z_min - z_pad, z_max + z_pad)

        # Bottom row labels
        axes[1, col].set_xlabel("Downwind x (m)")
        axes[1, col].grid(True, linestyle=":", linewidth=0.7)
        axes[1, col].set_xlim(x_min - x_pad, x_max + x_pad)
        axes[1, col].set_ylim(z_min - z_pad, z_max + z_pad)

    axes[0, 0].set_ylabel("Altitude z (m)")
    axes[1, 0].set_ylabel("Altitude z (m)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig("results/figures/torque2026/optimized_trajectories_wind.pdf")
    return fig, axes


if __name__ == "__main__":
    set_plot_style()
    # plot_trajectories()
    # plt.show()
    plot_wind_sweep_for_shear()
    plt.show()
