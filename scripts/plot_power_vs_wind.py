"""Plot power and force metrics vs wind speed for optimized patterns at a fixed shear.

For each optimized config, run a quasi-steady simulation and compute:
- Power: mean, min, max (plotted with shaded min/max band)
- Tether force: mean, min, max
- Reeling speed: mean, min, max
"""

from pathlib import Path
import re
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

from awetrim.environment.Wind import Wind
from awetrim.system.system_model import create_system_model_from_yaml
from awetrim.timeseries.reelout_phase import Reelout
from awetrim.utils.utils import load_cycle_config_from_yaml
from awetrim.utils.color_palette import set_plot_style, get_color_list

# Configuration
KITE_CONFIG_PATH = Path("data/LEI-V3-KITE/v3_kite_input.yaml")
CONFIG_ROOT = Path("results/optimized_configs")
SHEAR = 0.03

PATTERN_FOLDERS: Dict[str, Path] = {
    "Down-loop": CONFIG_ROOT / "downloops",
    "Up-loop": CONFIG_ROOT / "uploops",
    "Helix": CONFIG_ROOT / "helix",
}


def build_wind_model(speed_wind_at_100: float, z0: float) -> Wind:
    wind_model = Wind(wind_model="logarithmic", z0=z0)
    wind_model.speed_friction = 0.41 * speed_wind_at_100 / np.log(100 / wind_model.z0)
    return wind_model


def extract_wind_speed(path: Path) -> float:
    match = re.search(r"wind_?([0-9]+(?:\.[0-9]+)?)", path.name)
    return float(match.group(1)) if match else None


def simulate_metrics(
    config_path: Path,
    wind_speed: float,
    shear: float,
    prev_state: Dict[str, float] | None = None,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    pattern_config, _ = load_cycle_config_from_yaml(config_path)
    # s_dot = 2 if "loop" in config_path.name else 4.0

    if prev_state is not None:
        start_state = prev_state
    else:
        start_state = {
            "t": 0.0,
            "s": 0.0,
            "s_dot": 4,
            "distance_radial": pattern_config["path_parameters"]["r0"],
            "speed_radial": -2.0,
            "input_steering": 0.0,
            "tension_tether_ground": 8.4e5,
        }
    system_model = create_system_model_from_yaml(yaml_path=KITE_CONFIG_PATH)
    system_model.wind = build_wind_model(speed_wind_at_100=wind_speed, z0=shear)
    reelout = Reelout(
        system_model=system_model, pattern_config=pattern_config, depower=0
    )
    phase, _, start_state = reelout.run_simulation(
        run_plots=False,
        phase_sim=True,
        start_state=start_state,
        return_start_state=True,
    )

    tension = phase.return_variable("tension_tether_ground")
    vr = phase.return_variable("speed_radial")
    depower = phase.return_variable("input_depower")
    depower = depower * 0.08 + 0.22  # rescale to [0.22, 0.3]
    power_ts = tension * vr
    vtau = phase.return_variable("speed_tangential")
    aoa_deg = phase.return_variable("angle_of_attack") * 180 / np.pi

    em = phase.energy_metrics()

    return {
        "avg_power": em.get("avg_power", float(np.mean(power_ts))) / 1000,
        "min_power": float(np.min(power_ts)) / 1000,
        "max_power": float(np.max(power_ts)) / 1000,
        "mean_tension": float(np.mean(tension)) / 1000,
        "min_tension": float(np.min(tension)) / 1000,
        "max_tension": float(np.max(tension)) / 1000,
        "mean_vr": float(np.mean(vr)),
        "min_vr": float(np.min(vr)),
        "max_vr": float(np.max(vr)),
        "mean_depower": float(np.mean(depower)),
        "min_depower": float(np.min(depower)),
        "max_depower": float(np.max(depower)),
        "mean_vtau": float(np.mean(vtau)),
        "min_vtau": float(np.min(vtau)),
        "max_vtau": float(np.max(vtau)),
        "mean_aoa": float(np.mean(aoa_deg)),
        "min_aoa": float(np.min(aoa_deg)),
        "max_aoa": float(np.max(aoa_deg)),
    }, start_state


def collect_metrics(folder: Path, shear: float) -> List[Tuple[float, Dict[str, float]]]:
    data: List[Tuple[float, Dict[str, float]]] = []
    prev_state: Dict[str, float] | None = None
    for config_path in sorted(
        folder.glob(f"*{shear}*logarithmic_spline.yaml"),
        key=lambda p: extract_wind_speed(p) or 0,
    ):
        wind_speed = extract_wind_speed(config_path)
        if wind_speed < 5:
            continue
        if wind_speed > 24:
            continue
        if wind_speed is None:
            continue
        print(f"Simulating {config_path.name}...")
        if "depower" in config_path.name:
            print("  (depower pattern)")
            print(f"  Previous state: {prev_state}")
            metrics, prev_state = simulate_metrics(
                config_path,
                wind_speed,
                shear,
                prev_state,
            )
            data.append((wind_speed, metrics))
    return sorted(data, key=lambda x: x[0])


def _plot_band(ax, x, y_mean, y_min, y_max, label: str, color: str):
    ax.plot(x, y_mean, marker="o", markersize=4, label=label, color=color)
    ax.fill_between(x, y_min, y_max, color=color, alpha=0.25, edgecolor=color)


def plot_metrics_vs_wind(shear: float = SHEAR) -> Tuple[plt.Figure, List[plt.Axes]]:
    fig, axes = plt.subplots(3, 2, figsize=(10, 9))
    axes = axes.flatten()
    colors = get_color_list()[1::]
    for pattern_idx, (pattern_name, folder) in enumerate(PATTERN_FOLDERS.items()):
        data = collect_metrics(folder, shear)
        if not data:
            print(f"No configs found for {pattern_name} at z0={shear}")
            continue
        wind_speeds, metrics_list = zip(*data)

        # Power [0,0]
        avg_power = [m["avg_power"] for m in metrics_list]
        min_power = [m["min_power"] for m in metrics_list]
        max_power = [m["max_power"] for m in metrics_list]
        _plot_band(
            axes[0],
            wind_speeds,
            avg_power,
            min_power,
            max_power,
            pattern_name,
            colors[pattern_idx],
        )

        # Tether force [0,1]
        mean_T = [m["mean_tension"] for m in metrics_list]
        min_T = [m["min_tension"] for m in metrics_list]
        max_T = [m["max_tension"] for m in metrics_list]
        _plot_band(
            axes[1],
            wind_speeds,
            mean_T,
            min_T,
            max_T,
            pattern_name,
            colors[pattern_idx],
        )

        # Reeling speed [1,0]
        mean_vr = [m["mean_vr"] for m in metrics_list]
        min_vr = [m["min_vr"] for m in metrics_list]
        max_vr = [m["max_vr"] for m in metrics_list]
        _plot_band(
            axes[2],
            wind_speeds,
            mean_vr,
            min_vr,
            max_vr,
            pattern_name,
            colors[pattern_idx],
        )

        # Tangential speed [1,1]
        mean_vtau = [m["mean_vtau"] for m in metrics_list]
        min_vtau = [m["min_vtau"] for m in metrics_list]
        max_vtau = [m["max_vtau"] for m in metrics_list]
        _plot_band(
            axes[3],
            wind_speeds,
            mean_vtau,
            min_vtau,
            max_vtau,
            pattern_name,
            colors[pattern_idx],
        )

        # Input depower [2,0]
        mean_depower = [m["mean_depower"] for m in metrics_list]
        min_depower = [m["min_depower"] for m in metrics_list]
        max_depower = [m["max_depower"] for m in metrics_list]
        _plot_band(
            axes[4],
            wind_speeds,
            mean_depower,
            min_depower,
            max_depower,
            pattern_name,
            colors[pattern_idx],
        )

        # Angle of attack [deg] [2,1]
        mean_aoa = [m["mean_aoa"] for m in metrics_list]
        min_aoa = [m["min_aoa"] for m in metrics_list]
        max_aoa = [m["max_aoa"] for m in metrics_list]
        _plot_band(
            axes[5],
            wind_speeds,
            mean_aoa,
            min_aoa,
            max_aoa,
            pattern_name,
            colors[pattern_idx],
        )

    # Labels and grids
    for ax in axes:
        ax.set_xlabel(r"$v_\mathrm{w,100m}$ (m s$^{-1}$)")
        ax.grid(True, linestyle=":", linewidth=0.7)
        ax.set_xlim(5, 24)
        ax.xaxis.set_major_locator(MultipleLocator(1.0))

    axes[0].set_ylabel("Power (kW)")
    axes[1].set_ylabel("Tether force (kN)")
    axes[2].set_ylabel(r"Reeling speed (m s$^{-1}$)")
    axes[3].set_ylabel(r"Tangential speed (m s$^{-1}$)")
    axes[4].set_ylabel("Input depower (-)")
    axes[5].set_ylabel(r"Angle of attack ($^\circ$)")

    axes[0].legend()
    fig.tight_layout()
    plt.savefig("results/figures/torque2026/metrics_vs_wind.pdf")

    return fig, axes


if __name__ == "__main__":
    set_plot_style()
    plot_metrics_vs_wind()

    plt.show()
