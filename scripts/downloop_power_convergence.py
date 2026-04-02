import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from awetrim.environment.Wind import Wind
from awetrim.system.system_model import create_system_model_from_yaml
from awetrim.timeseries.reelout_phase import Reelout
from awetrim.utils.utils import load_cycle_config_from_yaml

# ---------------------------------------------------------------------------
# Configuration files and defaults
# ---------------------------------------------------------------------------
KITE_CONFIG_PATH = Path("data/LEI-V3-KITE/v3_kite_input.yaml")
CYCLE_CONFIG_PATH = Path("data/LEI-V3-KITE/v3_downloop_config_example_spline.yaml")

REELOUT_CONFIG, _ = load_cycle_config_from_yaml(CYCLE_CONFIG_PATH)

WIND_CONFIG = {
    "speed_wind_at_100": 12,
    "z0": 0.03,
    "model_type": "logarithmic",
}

START_STATE_TEMPLATE = {
    "t": 0.0,
    "s": 0.0,
    "s_dot": 1.5,
    "input_steering": 0.0,
    "tension_tether_ground": 8.4e5,
    "speed_radial": -1.0,
    "distance_radial": float(REELOUT_CONFIG["path_parameters"]["r0"]),
    "input_depower": 0,
}


def build_wind_model(speed_wind_at_100: float, z0: float, model_type: str) -> Wind:
    """Create a wind model using the supplied parameters."""
    wind_model = Wind(wind_model=model_type, z0=z0, direction_wind=0)
    speed_friction = 0.41 * speed_wind_at_100 / np.log(100 / wind_model.z0)
    if model_type == "logarithmic":
        wind_model.speed_friction = speed_friction
    elif model_type == "uniform":
        wind_model.speed_wind_ref = speed_wind_at_100
    return wind_model


def simulate_downloop(
    n_points: int, quasi_steady: bool, run_plots: bool = False
) -> dict:
    """Run a single downloop simulation with a given discretization."""
    pattern_config = copy.deepcopy(REELOUT_CONFIG)
    pattern_config["sim_parameters"]["n_points"] = int(n_points)

    system_model = create_system_model_from_yaml(yaml_path=KITE_CONFIG_PATH)
    system_model.wind = build_wind_model(**WIND_CONFIG)

    reelout = Reelout(
        system_model=system_model,
        pattern_config=pattern_config,
        start_state=copy.deepcopy(START_STATE_TEMPLATE),
        quasi_steady=quasi_steady,
    )

    phase, _ = reelout.run_simulation(
        run_plots=run_plots,
        phase_sim=True,
        quasi_steady=quasi_steady,
    )
    return phase.energy_metrics()


def main():
    n_points_values = list(range(10, 301, 20))
    reference_n = 600

    print(f"Computing dynamic reference average power with n_points={reference_n}...")
    reference_metrics = simulate_downloop(reference_n, quasi_steady=False)
    reference_power = reference_metrics["avg_power"]
    print(
        f"Reference avg_power={reference_power:.2f} W | energy={reference_metrics['energy']:.2f} J | "
        f"total_time={reference_metrics['total_time']:.2f} s"
    )

    dyn_avg, dyn_err, dyn_rel = [], [], []
    qs_avg, qs_err, qs_rel = [], [], []

    for n_points in n_points_values:
        try:
            metrics_dyn = simulate_downloop(n_points, quasi_steady=False)
            avg_dyn = metrics_dyn["avg_power"]
            dyn_avg.append(avg_dyn)
            dyn_err.append(abs(avg_dyn - reference_power))
            dyn_rel.append(100.0 * dyn_err[-1] / abs(reference_power))

            metrics_qs = simulate_downloop(n_points, quasi_steady=True)
            avg_qs = metrics_qs["avg_power"]
            qs_avg.append(avg_qs)
            qs_err.append(abs(avg_qs - reference_power))
            qs_rel.append(100.0 * qs_err[-1] / abs(reference_power))
            print(
                f"n_points={n_points:3d} | dyn_avg={avg_dyn:.2f} W | qs_avg={avg_qs:.2f} W | "
                f"dyn_err={dyn_err[-1]:.3e} W ({dyn_rel[-1]:.3e} %) | "
                f"qs_err={qs_err[-1]:.3e} W ({qs_rel[-1]:.3e} %)"
            )
        except Exception as exc:
            dyn_avg.append(np.nan)
            dyn_err.append(np.nan)
            dyn_rel.append(np.nan)
            qs_avg.append(np.nan)
            qs_err.append(np.nan)
            qs_rel.append(np.nan)
            print(f"n_points={n_points:3d} failed: {exc}")

    plt.figure(figsize=(7, 4))
    plt.plot(n_points_values, dyn_avg, marker="o", label="dynamic")
    plt.plot(n_points_values, qs_avg, marker="s", label="quasi-steady")
    plt.xlabel("n_points")
    plt.ylabel("Average power [W]")
    plt.title("Downloop power vs discretization")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    dyn_mask = (~np.isnan(dyn_err)) & (np.array(dyn_err) > 0)
    qs_mask = (~np.isnan(qs_err)) & (np.array(qs_err) > 0)

    if np.any(dyn_mask) or np.any(qs_mask):
        plt.figure(figsize=(7, 4))
        if np.any(dyn_mask):
            plt.loglog(
                np.array(n_points_values)[dyn_mask],
                np.array(dyn_err)[dyn_mask],
                marker="o",
                label="dynamic",
            )
        if np.any(qs_mask):
            plt.loglog(
                np.array(n_points_values)[qs_mask],
                np.array(qs_err)[qs_mask],
                marker="s",
                label="quasi-steady",
            )
        plt.xlabel("n_points")
        plt.ylabel("|avg_power - dynamic_ref|")
        plt.title("Power convergence error (log-log)")
        plt.grid(True, which="both", linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()

        plt.figure(figsize=(7, 4))
        if np.any(dyn_mask):
            plt.loglog(
                np.array(n_points_values)[dyn_mask],
                np.array(dyn_rel)[dyn_mask],
                marker="o",
                label="dynamic",
            )
        if np.any(qs_mask):
            plt.loglog(
                np.array(n_points_values)[qs_mask],
                np.array(qs_rel)[qs_mask],
                marker="s",
                label="quasi-steady",
            )
        plt.xlabel("n_points")
        plt.ylabel("Relative error vs dynamic ref [%]")
        plt.title("Power convergence relative error (log-log)")
        plt.grid(True, which="both", linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
