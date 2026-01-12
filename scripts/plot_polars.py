"""Plot aerodynamic polars for V3 kite.

This script loads the V3 aerodynamic input and plots the CL and CD coefficients
as a function of angle of attack.
"""

import json
import yaml
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def load_aero_input_from_yaml(yaml_path: Path) -> dict:
    """Load aerodynamic input from v3_kite_input.yaml."""
    with yaml_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config["wing"].get("aerodynamics", {})


def load_aero_input_from_json(json_path: Path) -> dict:
    """Load aerodynamic input from v3_aero_input.json."""
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_coefficients(aero_input: dict) -> tuple:
    """Extract CL and CD as functions of angle of attack.

    Returns
    -------
    tuple
        (alpha_range, cl_values, cd_values) where each is a numpy array
    """
    if aero_input.get("model") != "coeffs":
        raise ValueError(
            f"Only 'coeffs' model is supported, got {aero_input.get('model')}"
        )

    # Extract base coefficients
    params = aero_input.get("params", {})
    cl0 = params.get("CL0", 0.0)
    cd0 = params.get("CD0", 0.0)

    # Get coefficient definitions
    coeffs_def = aero_input.get("coefficients", {})
    cl_terms = coeffs_def.get("CL", [])
    cd_terms = coeffs_def.get("CD", [])

    # Create alpha range
    alpha_range = np.linspace(-np.pi / 6, np.pi / 6, 200)  # -30 to +30 degrees

    # Compute CL and CD
    cl_values = np.zeros_like(alpha_range) + cl0
    cd_values = np.zeros_like(alpha_range) + cd0

    # Apply CL terms
    for term in cl_terms:
        if term.get("var") == "alpha":
            power = term.get("power", 1)
            coef = term.get("coef", 0.0)
            cl_values += coef * (alpha_range**power)

    # Apply CD terms
    for term in cd_terms:
        if term.get("var") == "alpha":
            power = term.get("power", 1)
            coef = term.get("coef", 0.0)
            cd_values += coef * np.abs(alpha_range) ** power  # Use abs for drag

    return alpha_range, cl_values, cd_values


def find_optimal_aoa(
    alpha_range: np.ndarray,
    cl_values: np.ndarray,
    cd_values: np.ndarray,
    cd_tether: float = 0.0,
) -> dict:
    """Find angle of attack for maximum CL/CD and CL³/CD².

    Parameters
    ----------
    alpha_range : np.ndarray
        Angle of attack range in radians.
    cl_values : np.ndarray
        Lift coefficient values.
    cd_values : np.ndarray
        Drag coefficient values.
    cd_tether : float, optional
        Additional tether drag coefficient to add to CD. Default is 0.0.

    Returns
    -------
    dict
        Dictionary with optimal angles and their corresponding metrics.
    """
    # Add tether drag to total drag
    cd_total = cd_values + cd_tether

    # Avoid division by zero
    cd_safe = np.where(cd_total > 1e-6, cd_total, 1e-6)

    # CL/CD (glide ratio)
    cl_cd = cl_values / cd_safe
    idx_max_cl_cd = np.argmax(cl_cd)
    aoa_max_cl_cd = alpha_range[idx_max_cl_cd]
    max_cl_cd = cl_cd[idx_max_cl_cd]

    # CL³/CD² (power metric for crosswind kites)
    cl3_cd2 = (cl_values**3) / (cd_safe**2)
    idx_max_cl3_cd2 = np.argmax(cl3_cd2)
    aoa_max_cl3_cd2 = alpha_range[idx_max_cl3_cd2]
    max_cl3_cd2 = cl3_cd2[idx_max_cl3_cd2]

    return {
        "aoa_max_cl_cd_rad": aoa_max_cl_cd,
        "aoa_max_cl_cd_deg": np.degrees(aoa_max_cl_cd),
        "max_cl_cd": max_cl_cd,
        "cl_at_max_cl_cd": cl_values[idx_max_cl_cd],
        "cd_at_max_cl_cd": cd_values[idx_max_cl_cd],
        "cd_total_at_max_cl_cd": cd_total[idx_max_cl_cd],
        "aoa_max_cl3_cd2_rad": aoa_max_cl3_cd2,
        "aoa_max_cl3_cd2_deg": np.degrees(aoa_max_cl3_cd2),
        "max_cl3_cd2": max_cl3_cd2,
        "cl_at_max_cl3_cd2": cl_values[idx_max_cl3_cd2],
        "cd_at_max_cl3_cd2": cd_values[idx_max_cl3_cd2],
        "cd_total_at_max_cl3_cd2": cd_total[idx_max_cl3_cd2],
        "cd_tether": cd_tether,
    }


def plot_polars(aero_input: dict, save_path: Path = None) -> None:
    """Plot CL and CD polars.

    Parameters
    ----------
    aero_input : dict
        Aerodynamic input dictionary with model and coefficients.
    save_path : Path, optional
        If provided, save the figure to this path.
    """
    alpha_range, cl_values, cd_values = extract_coefficients(aero_input)

    # Find optimal angles without tether drag
    optimal = find_optimal_aoa(alpha_range, cl_values, cd_values, cd_tether=0.0)

    # Find optimal angles with tether drag
    cd_tether = 0.03
    optimal_with_tether = find_optimal_aoa(
        alpha_range, cl_values, cd_values, cd_tether=cd_tether
    )

    print("\n" + "=" * 70)
    print("OPTIMAL ANGLE OF ATTACK ANALYSIS")
    print("=" * 70)

    print(f"\n{'WITHOUT TETHER DRAG (CD_tether = 0)':^70}")
    print("-" * 70)
    print(f"\nMaximum CL/CD (glide ratio):")
    print(
        f"  AoA: {optimal['aoa_max_cl_cd_deg']:.2f}° ({optimal['aoa_max_cl_cd_rad']:.4f} rad)"
    )
    print(f"  CL/CD: {optimal['max_cl_cd']:.2f}")
    print(f"  CL: {optimal['cl_at_max_cl_cd']:.3f}")
    print(f"  CD: {optimal['cd_at_max_cl_cd']:.3f}")

    print(f"\nMaximum CL³/CD² (power metric):")
    print(
        f"  AoA: {optimal['aoa_max_cl3_cd2_deg']:.2f}° ({optimal['aoa_max_cl3_cd2_rad']:.4f} rad)"
    )
    print(f"  CL³/CD²: {optimal['max_cl3_cd2']:.2f}")
    print(f"  CL: {optimal['cl_at_max_cl3_cd2']:.3f}")
    print(f"  CD: {optimal['cd_at_max_cl3_cd2']:.3f}")

    print(f"\n{'WITH TETHER DRAG (CD_tether = 0.03)':^70}")
    print("-" * 70)
    print(f"\nMaximum CL/CD (glide ratio):")
    print(
        f"  AoA: {optimal_with_tether['aoa_max_cl_cd_deg']:.2f}° ({optimal_with_tether['aoa_max_cl_cd_rad']:.4f} rad)"
    )
    print(f"  CL/CD: {optimal_with_tether['max_cl_cd']:.2f}")
    print(f"  CL: {optimal_with_tether['cl_at_max_cl_cd']:.3f}")
    print(f"  CD_wing: {optimal_with_tether['cd_at_max_cl_cd']:.3f}")
    print(f"  CD_total: {optimal_with_tether['cd_total_at_max_cl_cd']:.3f}")

    print(f"\nMaximum CL³/CD² (power metric):")
    print(
        f"  AoA: {optimal_with_tether['aoa_max_cl3_cd2_deg']:.2f}° ({optimal_with_tether['aoa_max_cl3_cd2_rad']:.4f} rad)"
    )
    print(f"  CL³/CD²: {optimal_with_tether['max_cl3_cd2']:.2f}")
    print(f"  CL: {optimal_with_tether['cl_at_max_cl3_cd2']:.3f}")
    print(f"  CD_wing: {optimal_with_tether['cd_at_max_cl3_cd2']:.3f}")
    print(f"  CD_total: {optimal_with_tether['cd_total_at_max_cl3_cd2']:.3f}")

    print("=" * 70 + "\n")

    # Convert to degrees for plotting
    alpha_deg = np.degrees(alpha_range)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot CL
    axes[0, 0].plot(alpha_deg, cl_values, "b-", linewidth=2, label="CL")
    axes[0, 0].axvline(
        x=optimal["aoa_max_cl_cd_deg"],
        color="orange",
        linestyle="--",
        alpha=0.7,
        label=f'Max CL/CD ({optimal["aoa_max_cl_cd_deg"]:.1f}°)',
    )
    axes[0, 0].axvline(
        x=optimal["aoa_max_cl3_cd2_deg"],
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f'Max CL³/CD² ({optimal["aoa_max_cl3_cd2_deg"]:.1f}°)',
    )
    axes[0, 0].axhline(y=0, color="k", linestyle="--", alpha=0.3)
    axes[0, 0].axvline(x=0, color="k", linestyle="--", alpha=0.3)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlabel("Angle of Attack [°]", fontsize=11)
    axes[0, 0].set_ylabel("Lift Coefficient [-]", fontsize=11)
    axes[0, 0].set_title("V3 Lift Coefficient (CL)", fontsize=12, fontweight="bold")
    axes[0, 0].legend(fontsize=9)

    # Plot CD
    axes[0, 1].plot(alpha_deg, cd_values, "r-", linewidth=2, label="CD")
    axes[0, 1].axvline(
        x=optimal["aoa_max_cl_cd_deg"],
        color="orange",
        linestyle="--",
        alpha=0.7,
        label=f'Max CL/CD ({optimal["aoa_max_cl_cd_deg"]:.1f}°)',
    )
    axes[0, 1].axvline(
        x=optimal["aoa_max_cl3_cd2_deg"],
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f'Max CL³/CD² ({optimal["aoa_max_cl3_cd2_deg"]:.1f}°)',
    )
    axes[0, 1].axhline(y=0, color="k", linestyle="--", alpha=0.3)
    axes[0, 1].axvline(x=0, color="k", linestyle="--", alpha=0.3)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlabel("Angle of Attack [°]", fontsize=11)
    axes[0, 1].set_ylabel("Drag Coefficient [-]", fontsize=11)
    axes[0, 1].set_title("V3 Drag Coefficient (CD)", fontsize=12, fontweight="bold")
    axes[0, 1].legend(fontsize=9)

    # Plot CL/CD
    cd_safe = np.where(cd_values > 1e-6, cd_values, 1e-6)
    cl_cd = cl_values / cd_safe
    axes[1, 0].plot(alpha_deg, cl_cd, "g-", linewidth=2, label="CL/CD")
    axes[1, 0].axvline(
        x=optimal["aoa_max_cl_cd_deg"],
        color="orange",
        linestyle="--",
        alpha=0.7,
        label=f'Max at {optimal["aoa_max_cl_cd_deg"]:.1f}°',
    )
    axes[1, 0].scatter(
        [optimal["aoa_max_cl_cd_deg"]],
        [optimal["max_cl_cd"]],
        color="orange",
        s=100,
        zorder=5,
    )
    axes[1, 0].axhline(y=0, color="k", linestyle="--", alpha=0.3)
    axes[1, 0].axvline(x=0, color="k", linestyle="--", alpha=0.3)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlabel("Angle of Attack [°]", fontsize=11)
    axes[1, 0].set_ylabel("CL/CD [-]", fontsize=11)
    axes[1, 0].set_title("Glide Ratio (CL/CD)", fontsize=12, fontweight="bold")
    axes[1, 0].legend(fontsize=9)

    # Plot CL³/CD²
    cl3_cd2 = (cl_values**3) / (cd_safe**2)
    axes[1, 1].plot(alpha_deg, cl3_cd2, "purple", linewidth=2, label="CL³/CD²")
    axes[1, 1].axvline(
        x=optimal["aoa_max_cl3_cd2_deg"],
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f'Max at {optimal["aoa_max_cl3_cd2_deg"]:.1f}°',
    )
    axes[1, 1].scatter(
        [optimal["aoa_max_cl3_cd2_deg"]],
        [optimal["max_cl3_cd2"]],
        color="red",
        s=100,
        zorder=5,
    )
    axes[1, 1].axhline(y=0, color="k", linestyle="--", alpha=0.3)
    axes[1, 1].axvline(x=0, color="k", linestyle="--", alpha=0.3)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlabel("Angle of Attack [°]", fontsize=11)
    axes[1, 1].set_ylabel("CL³/CD² [-]", fontsize=11)
    axes[1, 1].set_title("Power Metric (CL³/CD²)", fontsize=12, fontweight="bold")
    axes[1, 1].legend(fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    plt.show()


def main():
    """Main entry point."""
    # Try loading from YAML first (preferred)
    kite_config_path = Path("data/LEI-V3-KITE/v3_kite_input.yaml")
    json_config_path = Path("data/LEI-V3-KITE/v3_aero_input.json")

    if kite_config_path.exists():
        print(f"Loading aerodynamic data from {kite_config_path}")
        aero_input = load_aero_input_from_yaml(kite_config_path)
    elif json_config_path.exists():
        print(f"Loading aerodynamic data from {json_config_path}")
        aero_input = load_aero_input_from_json(json_config_path)
    else:
        raise FileNotFoundError(
            f"Neither {kite_config_path} nor {json_config_path} found"
        )

    # Plot the polars
    save_path = Path("results/figures/v3_polars.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plot_polars(aero_input, save_path=save_path)


if __name__ == "__main__":
    main()
