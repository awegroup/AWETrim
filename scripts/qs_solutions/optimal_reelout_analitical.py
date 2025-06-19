"""
Standalone script for kite tether force optimization using CasADi.

This script computes and plots reeling forces for a kite system under different aerodynamic
performance conditions. It uses symbolic computation to find optimal reeling speeds
that maximize average force considering a reeling-in phase.

Author: Python Code Expert
"""

import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import logging
from picawe.utils.color_palette import set_plot_style_no_latex

set_plot_style_no_latex()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def define_force_function(
    C_L: float,
    C_D: float,
    phi: float = 0.0,
    beta: float = 0.0,
    rho: float = 1.225,
    S: float = 46.0,
) -> ca.Function:
    """
    Constructs the symbolic force function Ft(v_r, v_w) for given lift and drag coefficients.
    """
    v_r = ca.SX.sym("v_r")
    v_w = ca.SX.sym("v_w")

    C_R = ca.sqrt(C_L**2 + C_D**2)
    Ft = (
        0.5
        * rho
        * S
        * C_R
        * (v_w * ca.cos(phi) * ca.cos(beta) - v_r) ** 2
        * (1 + (C_L / C_D) ** 2)
    )
    return ca.Function("Ft", [v_r, v_w], [Ft])


def compute_optimal_reel_speeds_and_forces(
    aero_coefficients: list[tuple[float, float]],
    reel_out_length: float,
    reel_in_force: float,
    reel_in_velocity: float,
) -> tuple[dict[tuple[float, float], dict[str, np.ndarray]], np.ndarray]:
    """
    Compute the optimal reeling speeds and corresponding tether forces for given aerodynamic coefficients.

    Args:
        aero_coefficients (list of tuple): List of (C_L, C_D) aerodynamic performance pairs.
        reel_out_length (float): Length of the cable to reel out (in meters).
        reel_in_force (float): Constant force applied during reeling in (in Newtons).
        reel_in_velocity (float): Negative velocity for reeling in (in m/s).

    Returns:
        dict: Dictionary mapping each (C_L, C_D) pair to reeling speeds and tether forces.
        np.ndarray: Array of wind speeds used for simulation.
    """
    reel_in_duration = reel_out_length / abs(reel_in_velocity)
    reel_in_energy = reel_in_force * reel_in_velocity * reel_in_duration

    optimized_force_data = {}
    wind_speeds = np.linspace(1.5, 25.0, 100)

    phi = beta = 0.0  # The results are not sensitive to these angles in this context

    for lift_coefficient, drag_coefficient in aero_coefficients:
        force_function = define_force_function(
            lift_coefficient, drag_coefficient, phi, beta
        )
        v_r = ca.SX.sym("v_r")
        v_w = ca.SX.sym("v_w")

        tether_force_expr = force_function(v_r, v_w)
        power_expr = tether_force_expr * v_r

        average_energy_expr = (
            (power_expr * reel_out_length / v_r) + reel_in_energy
        ) / (reel_in_duration + (reel_out_length / v_r))
        energy_derivative = ca.gradient(average_energy_expr, v_r)

        root_solver = ca.rootfinder(
            "root_solver",
            "newton",
            ca.Function("energy_derivative", [v_r, v_w], [energy_derivative]),
        )

        optimal_reel_speeds = []
        corresponding_tether_forces = []

        for wind_speed in wind_speeds:
            initial_guess = wind_speed / 3 * np.cos(beta)
            try:
                optimal_speed = root_solver(initial_guess, wind_speed)
                optimal_reel_speeds.append(float(optimal_speed))
                corresponding_tether_forces.append(
                    float(force_function(optimal_speed, wind_speed))
                )
            except Exception as e:
                logger.warning(
                    f"Root solver failed for wind_speed={wind_speed:.2f} m/s: {e}"
                )
                optimal_reel_speeds.append(np.nan)
                corresponding_tether_forces.append(np.nan)

        optimized_force_data[(lift_coefficient, drag_coefficient)] = {
            "v_r_sol": np.array(optimal_reel_speeds),
            "ft_sol": np.array(corresponding_tether_forces),
        }

    return optimized_force_data, wind_speeds


def analytical_solution(
    wind_speeds, beta=0.0, phi=0.0, rho=1.225, S=46.0, C_L=0.75, C_D=0.15
):
    """
    Computes the analytical (non-optimized) force curve based on assumed optimal reeling speed.
    """
    C_R = np.sqrt(C_L**2 + C_D**2)
    v_r = wind_speeds / 3 * np.cos(beta)
    Ft = (
        0.5
        * rho
        * S
        * C_R
        * (wind_speeds * np.cos(phi) * np.cos(beta) - v_r) ** 2
        * (1 + (C_L / C_D) ** 2)
    )
    return v_r, Ft


def plot_results(results, wind_speeds, vr_opt, ft_opt):
    """
    Generates a comparison plot of numerical vs analytical reeling force results.
    """
    plt.figure(figsize=(10, 6))
    pairs = list(results.keys())

    plt.plot(
        results[pairs[0]]["v_r_sol"],
        results[pairs[0]]["ft_sol"] / 9.81,
        label="With reelin",
    )

    plt.fill_between(
        results[pairs[1]]["v_r_sol"],
        results[pairs[1]]["ft_sol"] / 9.81,
        results[pairs[2]]["ft_sol"] / 9.81,
        alpha=0.5,
        label="With reelin (min-max aero performance)",
    )

    plt.plot(vr_opt, ft_opt / 9.81, label="Without reelin (analytical-optimal)")
    plt.plot(
        [-0.2, 1.4, 10], [440, 2260, 2890], linestyle="--", label="Current curve KP"
    )

    plt.xlabel("Reeling speed [m/s]")
    plt.ylabel("Tether force [N]")
    plt.ylim([0, 3500])
    plt.legend()
    plt.grid(True)
    plt.title("Tether Force vs Reeling Speed")
    plt.tight_layout()
    plt.show()


def main():
    """Main entry point for the kite force optimization script."""
    logger.info("Starting kite reeling optimization...")

    cl_cd_pairs = [(0.75, 0.15), (0.9, 0.12), (0.68, 0.18)]
    deltaL = 70.0
    F_reelin = 4000.0
    v_r_reelin = -5.0

    results, wind_speeds = compute_optimal_reel_speeds_and_forces(
        cl_cd_pairs, deltaL, F_reelin, v_r_reelin
    )
    vr_opt, ft_opt = analytical_solution(wind_speeds)

    logger.info("Optimization complete. Plotting results...")
    plot_results(results, wind_speeds, vr_opt, ft_opt)


if __name__ == "__main__":
    main()
