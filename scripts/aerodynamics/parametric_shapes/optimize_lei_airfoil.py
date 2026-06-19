"""Optimize a LEI airfoil with the Masure regression model.

The objective is to maximize max_alpha(CL^3 / CD^2) over a sweep of angle of attack.
The optimization stays inside a conservative parameter box close to the trained data.
"""

import contextlib
import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution

from VSM.core.AirfoilAerodynamics import AirfoilAerodynamics

from awetrim.aerodynamics.parametric_airfoil import LEI_airfoil, generate_profile

PROJECT_DIR = Path(__file__).resolve().parents[3]
ML_MODELS_DIR = PROJECT_DIR / "data" / "ml_models"
RESULTS_DIR = PROJECT_DIR / "results" / "TUDELFT_V3_KITE"

PARAMETER_NAMES = ["t", "eta", "kappa", "delta", "lambda", "phi"]
PARAMETER_BOUNDS = [
    (0.07, 0.10),
    (0.15, 0.26),
    (0.05, 0.11),
    (-5.0, 5.0),
    (0.0, 0.30),
    (0.50, 0.80),
]

# V3-KITE center section (airfoil_id 1, y=0) from
# data/TUDELFT_V3_KITE/CAD_derived_geometry/aero_geometry_CAD_masure_regression.yaml
BASELINE_PARAMS = {
    "t": 0.0782,
    "eta": 0.1753,
    "kappa": 0.0955,
    "delta": 0.0,
    "lambda": 0.16,
    "phi": 0.65,
}

ALPHA_RANGE = [-10, 31, 0.5]
REYNOLDS_NUMBER = 1e6

# Parameters held fixed during optimization (structural givens, e.g. the LE tube
# diameter set by the inflatable taper). The optimizer varies only the rest; the
# fixed values are taken from the baseline via set_fixed_params().
FIXED_PARAM_NAMES = ["t"]
_FIXED_VALUES = {}


def set_fixed_params(baseline):
    """Freeze FIXED_PARAM_NAMES at the given baseline's values for optimization."""
    global _FIXED_VALUES
    _FIXED_VALUES = {name: baseline[name] for name in FIXED_PARAM_NAMES}


def free_parameter_names():
    return [name for name in PARAMETER_NAMES if name not in FIXED_PARAM_NAMES]


def free_parameter_bounds():
    return [
        bound
        for name, bound in zip(PARAMETER_NAMES, PARAMETER_BOUNDS)
        if name not in FIXED_PARAM_NAMES
    ]


def assemble_params(free_x):
    """Combine optimized free values with the fixed values into a full param dict."""
    params = dict(zip(free_parameter_names(), free_x))
    params.update(_FIXED_VALUES)
    return params


def build_airfoil_aero(airfoil_params):
    return AirfoilAerodynamics.from_yaml_entry(
        airfoil_type="masure_regression",
        airfoil_params=airfoil_params,
        alpha_range=ALPHA_RANGE,
        reynolds=REYNOLDS_NUMBER,
        ml_models_dir=ML_MODELS_DIR,
    )


def compute_merit_curve(aero):
    cl = np.asarray(aero.CL, dtype=float)
    cd = np.asarray(aero.CD, dtype=float)
    cd_safe = np.clip(cd, 1e-6, None)
    merit_curve = cl**3 / cd_safe**2
    merit_curve = np.where(np.isfinite(merit_curve), merit_curve, -np.inf)

    best_index = int(np.argmax(merit_curve))
    best_merit = float(merit_curve[best_index])
    best_alpha_deg = float(np.rad2deg(np.asarray(aero.alpha)[best_index]))
    return best_merit, best_alpha_deg, merit_curve


# Quarter-chord reference for the moment coefficient (masure_regression convention).
CM_REF_X = 0.25
# Target front (LE) share of the section normal force, and the absolute band the
# constrained optimization must stay within (e.g. 0.67 +/- 0.05 -> 0.62..0.72).
TARGET_FRONT_FRACTION = 0.71
FRONT_FRACTION_TOL = 0.05


def compute_force_split(aero, index):
    """Split the section load into front (LE) and back (TE) forces at one alpha.

    Resolves CL/CD into a normal-force coefficient Cn and locates the center of
    pressure from the quarter-chord moment: x_cp/c = CM_REF_X - CM / Cn. The
    normal force is then distributed between the LE (x=0) and TE (x=1) so that
    BOTH the net force and the net pitching moment are conserved:

        Cn_back  = Cn * (x_cp/c)
        Cn_front = Cn * (1 - x_cp/c)

    Returns (cn, x_cp, cn_front, cn_back).
    """
    alpha = float(np.asarray(aero.alpha)[index])
    cl = float(np.asarray(aero.CL)[index])
    cd = float(np.asarray(aero.CD)[index])
    cm = float(np.asarray(aero.CM)[index])
    cn = cl * np.cos(alpha) + cd * np.sin(alpha)
    x_cp = CM_REF_X - cm / cn
    cn_front = cn * (1.0 - x_cp)
    cn_back = cn * x_cp
    return cn, x_cp, cn_front, cn_back


def design_front_fraction(aero):
    """Front (LE) share of the normal force at the peak-merit (design) alpha.

    Equals 1 - x_cp/c, i.e. the fraction of the section normal force reacted at
    the leading edge when the load is split between LE (x=0) and TE (x=1) so that
    force and moment are conserved.
    """
    _, _, merit_curve = compute_merit_curve(aero)
    design_index = int(np.argmax(merit_curve))
    cn, x_cp, cn_front, _ = compute_force_split(aero, design_index)
    return cn_front / cn


def front_fraction_curve(aero):
    """Front (LE) load fraction at every alpha in the sweep (1 - x_cp/c)."""
    cl = np.asarray(aero.CL, dtype=float)
    cd = np.asarray(aero.CD, dtype=float)
    cm = np.asarray(aero.CM, dtype=float)
    alpha = np.asarray(aero.alpha, dtype=float)
    cn = cl * np.cos(alpha) + cd * np.sin(alpha)
    cn_safe = np.where(np.abs(cn) < 1e-9, np.nan, cn)
    x_cp = CM_REF_X - cm / cn_safe
    return 1.0 - x_cp


def objective_constrained(x):
    """Maximize merit while keeping the front load fraction near the target.

    Uses a single aero evaluation and applies a soft penalty when the design-alpha
    front fraction leaves TARGET_FRONT_FRACTION +/- FRONT_FRACTION_TOL, so the
    optimum delivers the LE/TE load split the designer asked for.
    """
    airfoil_params = assemble_params(x)
    try:
        if not is_geometry_constructible(airfoil_params):
            return 1e12
        aero = build_airfoil_aero(airfoil_params)
        best_merit, _, _ = compute_merit_curve(aero)
        if not np.isfinite(best_merit):
            return 1e12
        front_fraction = design_front_fraction(aero)
        if not np.isfinite(front_fraction):
            return 1e12
        deviation = abs(front_fraction - TARGET_FRONT_FRACTION)
        excess = max(0.0, deviation - FRONT_FRACTION_TOL)
        return -best_merit + 1e9 * excess
    except Exception:
        return 1e12


def is_geometry_constructible(airfoil_params):
    """Return True if the analytic LEI geometry can actually be built.

    The ML aero model predicts CL/CD for any parameter combination in the box,
    but the geometry constructor (generate_profile/LEI_airfoil) is stricter and
    fails for some combinations (e.g. LE_seam_angle returns None). Gate the
    optimizer on this so the optimum is a real, drawable airfoil.
    """
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            generate_profile(
                t_val=airfoil_params["t"],
                eta_val=airfoil_params["eta"],
                kappa_val=airfoil_params["kappa"],
                delta_val=airfoil_params["delta"],
                lambda_val=airfoil_params["lambda"],
                phi_val=airfoil_params["phi"],
            )
        return True
    except Exception:
        return False


def objective(x):
    airfoil_params = assemble_params(x)
    try:
        if not is_geometry_constructible(airfoil_params):
            return 1e12
        aero = build_airfoil_aero(airfoil_params)
        best_merit, _, _ = compute_merit_curve(aero)
        if not np.isfinite(best_merit):
            return 1e12
        return -best_merit
    except Exception:
        return 1e12


def optimize_lei_airfoil(baseline, objective_func=objective):
    # Optimize only the free parameters; fixed ones come from set_fixed_params().
    x0 = [baseline[name] for name in free_parameter_names()]
    result = differential_evolution(
        objective_func,
        bounds=free_parameter_bounds(),
        x0=np.asarray(x0, dtype=float),
        seed=1,
        maxiter=35,
        popsize=10,
        tol=1e-3,
        polish=True,
        updating="deferred",
        workers=1,
        disp=True,
    )

    optimized_params = assemble_params(result.x)
    optimized_aero = build_airfoil_aero(optimized_params)
    optimized_merit, optimized_alpha_deg, optimized_merit_curve = compute_merit_curve(
        optimized_aero
    )
    return (
        result,
        optimized_params,
        optimized_aero,
        optimized_merit,
        optimized_alpha_deg,
        optimized_merit_curve,
    )


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Freeze the structural parameters (e.g. tube diameter t) at the baseline.
    set_fixed_params(BASELINE_PARAMS)
    print(
        f"Fixed parameters: {{{', '.join(f'{n}={BASELINE_PARAMS[n]:.4f}' for n in FIXED_PARAM_NAMES)}}}; "
        f"optimizing {free_parameter_names()}"
    )

    baseline_aero = build_airfoil_aero(BASELINE_PARAMS)
    baseline_merit, baseline_alpha_deg, baseline_merit_curve = compute_merit_curve(
        baseline_aero
    )

    print("Optimization 1: max(CL^3 / CD^2) (unconstrained)...")
    (
        optimization_result,
        optimized_params,
        optimized_aero,
        optimized_merit,
        optimized_alpha_deg,
        optimized_merit_curve,
    ) = optimize_lei_airfoil(baseline=BASELINE_PARAMS)

    print(
        "\nOptimization 2: max(CL^3 / CD^2) with front load fraction in "
        f"{TARGET_FRONT_FRACTION:.2f} +/- {FRONT_FRACTION_TOL:.2f}..."
    )
    (
        constrained_result,
        constrained_params,
        constrained_aero,
        constrained_merit,
        constrained_alpha_deg,
        constrained_merit_curve,
    ) = optimize_lei_airfoil(
        baseline=BASELINE_PARAMS,
        objective_func=objective_constrained,
    )

    baseline_front = design_front_fraction(baseline_aero)
    optimized_front = design_front_fraction(optimized_aero)
    constrained_front = design_front_fraction(constrained_aero)

    def _print_case(title, params, merit, alpha_deg, front_fraction):
        print(f"\n{title}:")
        for name in PARAMETER_NAMES:
            print(f"  {name:>6s} = {params[name]:.4f}")
        print(f"  Peak merit = {merit:.4f} at alpha = {alpha_deg:.2f} deg")
        print(
            f"  Front load fraction = {front_fraction:.3f} "
            f"(LE {front_fraction:.1%} / TE {1.0 - front_fraction:.1%})"
        )

    _print_case(
        "Baseline (V3 center)",
        BASELINE_PARAMS,
        baseline_merit,
        baseline_alpha_deg,
        baseline_front,
    )
    _print_case(
        "Optimized (merit only)",
        optimized_params,
        optimized_merit,
        optimized_alpha_deg,
        optimized_front,
    )
    _print_case(
        "Optimized (load-constrained)",
        constrained_params,
        constrained_merit,
        constrained_alpha_deg,
        constrained_front,
    )

    lo = TARGET_FRONT_FRACTION - FRONT_FRACTION_TOL
    hi = TARGET_FRONT_FRACTION + FRONT_FRACTION_TOL
    in_band = lo <= constrained_front <= hi
    print(
        f"\n  Target front fraction = {TARGET_FRONT_FRACTION:.2f} "
        f"(band {lo:.2f}..{hi:.2f}); constrained result {'inside' if in_band else 'OUTSIDE'} band"
    )
    print(
        f"  Opt1 success = {optimization_result.success}; "
        f"Opt2 success = {constrained_result.success}"
    )

    all_points, _, _ = generate_profile(
        t_val=optimized_params["t"],
        eta_val=optimized_params["eta"],
        kappa_val=optimized_params["kappa"],
        delta_val=optimized_params["delta"],
        lambda_val=optimized_params["lambda"],
        phi_val=optimized_params["phi"],
    )

    with contextlib.redirect_stdout(io.StringIO()):
        baseline_all_points, _, _ = generate_profile(
            t_val=BASELINE_PARAMS["t"],
            eta_val=BASELINE_PARAMS["eta"],
            kappa_val=BASELINE_PARAMS["kappa"],
            delta_val=BASELINE_PARAMS["delta"],
            lambda_val=BASELINE_PARAMS["lambda"],
            phi_val=BASELINE_PARAMS["phi"],
        )
        constrained_all_points, _, _ = generate_profile(
            t_val=constrained_params["t"],
            eta_val=constrained_params["eta"],
            kappa_val=constrained_params["kappa"],
            delta_val=constrained_params["delta"],
            lambda_val=constrained_params["lambda"],
            phi_val=constrained_params["phi"],
        )

    (
        LE_tube_points,
        P1,
        P11,
        P12,
        LE_points,
        TE_points,
        P2,
        P21,
        P22,
        P3,
        round_TE_points,
        P4,
        P5,
        P51,
        P52,
        TE_lower_points,
        P6,
        P61,
        P62,
        P63,
        fillet_points,
        Origin_LE_tube,
        round_TE_mid,
        seam_a_full,
        *_extras,
    ) = LEI_airfoil(
        tube_size=optimized_params["t"],
        c_x=optimized_params["eta"],
        c_y=optimized_params["kappa"],
        TE_angle=optimized_params["delta"],
        TE_cam_tension=optimized_params["lambda"],
        LE_tension=optimized_params["phi"],
    )

    alpha_deg = np.rad2deg(optimized_aero.alpha)
    baseline_alpha_deg_array = np.rad2deg(baseline_aero.alpha)
    constrained_alpha_deg_array = np.rad2deg(constrained_aero.alpha)

    fig = plt.figure(figsize=(18, 9))
    outer_gs = fig.add_gridspec(2, 1, height_ratios=[1.1, 1.0], hspace=0.3)
    top_gs = outer_gs[0].subgridspec(1, 2, wspace=0.15)
    bot_gs = outer_gs[1].subgridspec(1, 3, wspace=0.25)

    ax_detail = fig.add_subplot(top_gs[0, 0])
    ax_outline = fig.add_subplot(top_gs[0, 1])
    ax_cl = fig.add_subplot(bot_gs[0, 0])
    ax_cd = fig.add_subplot(bot_gs[0, 1])
    ax_merit = fig.add_subplot(bot_gs[0, 2])

    eta = np.linspace(0, 2 * np.pi, 100)
    radius = -np.min(LE_tube_points[:, 1])
    origin_circle = np.array([radius, 0.0])
    x_circ = origin_circle[0] + radius * np.cos(eta)
    y_circ = origin_circle[1] + radius * np.sin(eta)
    ax_detail.plot(
        x_circ, y_circ, "--", linewidth=2, color="#3776ab", label="Circular tube"
    )
    ax_detail.plot(
        LE_points[:, 0],
        LE_points[:, 1],
        "-",
        color="#ff7f0e",
        linewidth=2,
        label="Front spline",
    )
    ctrl_front = np.array([P1, P11, P12, P2])
    ax_detail.plot(ctrl_front[:, 0], ctrl_front[:, 1], "--", color="gray", linewidth=2)
    ax_detail.scatter(
        ctrl_front[:, 0], ctrl_front[:, 1], s=30, color="#ff7f0e", label="Control front"
    )
    ax_detail.plot(
        TE_points[:, 0],
        TE_points[:, 1],
        "-",
        color="#2CA02C",
        linewidth=2,
        label="Rear spline",
    )
    ctrl_rear = np.array([P2, P21, P22, P3])
    ax_detail.plot(ctrl_rear[:, 0], ctrl_rear[:, 1], "--", color="gray", linewidth=2)
    ax_detail.scatter(
        ctrl_rear[:, 0], ctrl_rear[:, 1], s=30, color="#2CA02C", label="Control rear"
    )
    ax_detail.plot(
        fillet_points[:, 0],
        fillet_points[:, 1],
        "-",
        color="#D62728",
        linewidth=2,
        label="LE fillet",
    )
    ctrl_fillet = np.array([P6, P61, P62, P63])
    ax_detail.plot(
        ctrl_fillet[:, 0], ctrl_fillet[:, 1], "--", color="gray", linewidth=2
    )
    ax_detail.scatter(
        ctrl_fillet[:, 0],
        ctrl_fillet[:, 1],
        s=30,
        color="#D62728",
        label="Control LE fillet",
    )
    ax_detail.plot(
        TE_lower_points[:, 0],
        TE_lower_points[:, 1],
        "-",
        color="teal",
        linewidth=2,
        label="TE lower",
    )
    ctrl_tel = np.array([P5, P51, P52, P4])
    ax_detail.plot(ctrl_tel[:, 0], ctrl_tel[:, 1], "--", color="gray", linewidth=2)
    ax_detail.scatter(
        ctrl_tel[:, 0], ctrl_tel[:, 1], s=30, color="teal", label="Control TE lower"
    )
    ax_detail.plot(
        round_TE_points[:, 0],
        round_TE_points[:, 1],
        "-",
        color="k",
        linewidth=1.5,
        label="Round TE",
    )
    ax_detail.scatter(
        origin_circle[0],
        origin_circle[1],
        marker="*",
        color="b",
        s=35,
        label="LE tube centre",
    )
    ax_detail.scatter(
        TE_points[-1, 0],
        TE_points[-1, 1],
        marker="*",
        color="r",
        s=35,
        label="TE position",
    )
    ax_detail.scatter(
        LE_points[0, 0],
        LE_points[0, 1],
        marker="*",
        color="g",
        s=35,
        label="Tube-canopy intersection",
    )
    ax_detail.scatter(
        LE_points[-1, 0],
        LE_points[-1, 1],
        marker="*",
        color="k",
        s=35,
        label="Max. camber position",
    )
    ax_detail.plot(
        baseline_all_points[:, 0],
        baseline_all_points[:, 1],
        "--",
        color="0.6",
        linewidth=1.5,
        alpha=0.8,
        label="Baseline (V3 center)",
    )
    ax_detail.set_title("Optimized LEI airfoil (construction geometry)")
    ax_detail.set_xlabel("x / c")
    ax_detail.set_ylabel("y / c")
    ax_detail.set_aspect("equal", "box")
    ax_detail.grid(True, linestyle="--", alpha=0.3)
    y_min = min(
        np.min(LE_tube_points[:, 1]),
        np.min(all_points[:, 1]),
        np.min(baseline_all_points[:, 1]),
    )
    y_max = max(
        np.max(LE_points[:, 1]),
        np.max(all_points[:, 1]),
        np.max(baseline_all_points[:, 1]),
    )
    ax_detail.set_xlim(-0.02, 1.02)
    ax_detail.set_ylim(1.5 * y_min, 1.2 * y_max)
    ax_detail.legend(
        loc="center left",
        bbox_to_anchor=(-0.1, -0.6),
        fontsize=8,
        ncol=5,
        frameon=False,
        borderaxespad=0.0,
    )

    ax_outline.plot(
        baseline_all_points[:, 0],
        baseline_all_points[:, 1],
        "--",
        color="0.6",
        linewidth=1.5,
        label="Baseline (V3 center)",
    )
    ax_outline.plot(
        all_points[:, 0],
        all_points[:, 1],
        linewidth=2.0,
        color="tab:orange",
        label="Optimized (merit only)",
    )
    ax_outline.plot(
        constrained_all_points[:, 0],
        constrained_all_points[:, 1],
        linewidth=2.0,
        color="tab:green",
        label="Optimized (load-constrained)",
    )
    ax_outline.set_aspect("equal", "box")
    ax_outline.set_xlim(-0.02, 1.02)
    ymin2 = float(
        min(
            all_points[:, 1].min(),
            baseline_all_points[:, 1].min(),
            constrained_all_points[:, 1].min(),
        )
    )
    ymax2 = float(
        max(
            all_points[:, 1].max(),
            baseline_all_points[:, 1].max(),
            constrained_all_points[:, 1].max(),
        )
    )
    pad2 = 0.1 * (ymax2 - ymin2 + 1e-6)
    ax_outline.set_ylim(ymin2 - pad2, ymax2 + pad2)
    ax_outline.set_title("Clean airfoil outline: baseline vs. optimized")
    ax_outline.set_xlabel("x / c")
    ax_outline.set_ylabel("y / c")
    ax_outline.grid(True, linestyle="--", alpha=0.3)
    ax_outline.legend(loc="best", fontsize=8)

    ax_cl.plot(
        baseline_alpha_deg_array,
        baseline_aero.CL,
        "--",
        linewidth=1.5,
        label="Baseline",
    )
    ax_cl.plot(
        alpha_deg,
        optimized_aero.CL,
        linewidth=1.8,
        color="tab:orange",
        label="Optimized (merit only)",
    )
    ax_cl.plot(
        constrained_alpha_deg_array,
        constrained_aero.CL,
        linewidth=1.8,
        color="tab:green",
        label="Optimized (load-constrained)",
    )
    ax_cl.set_title("CL vs. α")
    ax_cl.set_xlabel(r"$\alpha$ [deg]")
    ax_cl.set_ylabel("CL")
    ax_cl.grid(True, linestyle="--", alpha=0.4)
    ax_cl.legend()

    ax_cd.plot(
        baseline_alpha_deg_array,
        baseline_aero.CD,
        "--",
        linewidth=1.5,
        label="Baseline",
    )
    ax_cd.plot(
        alpha_deg,
        optimized_aero.CD,
        linewidth=1.8,
        color="tab:orange",
        label="Optimized (merit only)",
    )
    ax_cd.plot(
        constrained_alpha_deg_array,
        constrained_aero.CD,
        linewidth=1.8,
        color="tab:green",
        label="Optimized (load-constrained)",
    )
    ax_cd.set_title("CD vs. α")
    ax_cd.set_xlabel(r"$\alpha$ [deg]")
    ax_cd.set_ylabel("CD")
    ax_cd.grid(True, linestyle="--", alpha=0.4)
    ax_cd.legend()

    ax_merit.plot(
        baseline_alpha_deg_array,
        baseline_merit_curve,
        "--",
        linewidth=1.5,
        label="Baseline",
    )
    ax_merit.plot(
        alpha_deg,
        optimized_merit_curve,
        linewidth=1.8,
        color="tab:orange",
        label="Optimized (merit only)",
    )
    ax_merit.plot(
        constrained_alpha_deg_array,
        constrained_merit_curve,
        linewidth=1.8,
        color="tab:green",
        label="Optimized (load-constrained)",
    )
    ax_merit.scatter(
        [optimized_alpha_deg], [optimized_merit], color="tab:orange", zorder=3
    )
    ax_merit.scatter(
        [constrained_alpha_deg], [constrained_merit], color="tab:green", zorder=3
    )
    ax_merit.set_title(r"$C_L^3 / C_D^2$ vs. α")
    ax_merit.set_xlabel(r"$\alpha$ [deg]")
    ax_merit.set_ylabel(r"$C_L^3 / C_D^2$")
    ax_merit.grid(True, linestyle="--", alpha=0.4)
    ax_merit.legend()
    ax_merit.axvline(optimized_alpha_deg, color="tab:orange", alpha=0.2)
    ax_merit.axvline(constrained_alpha_deg, color="tab:green", alpha=0.2)

    fig.suptitle(
        f"LEI airfoil optimized for max(CL^3 / CD^2) at Re = {REYNOLDS_NUMBER:.0e}",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    save_path = RESULTS_DIR / "lei_airfoil_optimization.pdf"
    fig.savefig(save_path, bbox_inches="tight")
    print(f"\nSaved figure to {save_path}")

    # ---- Figure 2: front/back load split analysis ----
    fig2, (ax_ff, ax_bar) = plt.subplots(1, 2, figsize=(13, 5))

    lo = TARGET_FRONT_FRACTION - FRONT_FRACTION_TOL
    hi = TARGET_FRONT_FRACTION + FRONT_FRACTION_TOL

    ax_ff.plot(
        baseline_alpha_deg_array,
        front_fraction_curve(baseline_aero),
        "--",
        color="0.6",
        linewidth=1.5,
        label="Baseline (V3 center)",
    )
    ax_ff.plot(
        alpha_deg,
        front_fraction_curve(optimized_aero),
        color="tab:orange",
        linewidth=1.8,
        label="Optimized (merit only)",
    )
    ax_ff.plot(
        constrained_alpha_deg_array,
        front_fraction_curve(constrained_aero),
        color="tab:green",
        linewidth=1.8,
        label="Optimized (load-constrained)",
    )
    ax_ff.axhspan(lo, hi, color="tab:green", alpha=0.12, label="Target band")
    ax_ff.axhline(
        TARGET_FRONT_FRACTION, color="tab:green", linestyle=":", linewidth=1.0
    )
    # Mark the design (peak-merit) alpha for each case.
    ax_ff.scatter(
        [optimized_alpha_deg], [optimized_front], color="tab:orange", zorder=3
    )
    ax_ff.scatter(
        [constrained_alpha_deg], [constrained_front], color="tab:green", zorder=3
    )
    ax_ff.scatter([baseline_alpha_deg], [baseline_front], color="0.4", zorder=3)
    ax_ff.set_title("Front (LE) load fraction vs. α")
    ax_ff.set_xlabel(r"$\alpha$ [deg]")
    ax_ff.set_ylabel("Front load fraction  (1 − $x_{cp}/c$)")
    ax_ff.set_ylim(0.0, 1.0)
    ax_ff.grid(True, linestyle="--", alpha=0.4)
    ax_ff.legend(fontsize=8)

    # Stacked front/back bars at each case's design alpha.
    labels = ["Baseline", "Merit only", "Load-constrained"]
    fronts = np.array([baseline_front, optimized_front, constrained_front])
    backs = 1.0 - fronts
    ax_bar.bar(labels, fronts, color="tab:blue", label="Front (LE)")
    ax_bar.bar(labels, backs, bottom=fronts, color="tab:red", label="Back (TE)")
    for i, f in enumerate(fronts):
        ax_bar.text(i, f / 2, f"{f:.0%}", ha="center", va="center", color="white")
        ax_bar.text(
            i, f + (1 - f) / 2, f"{1 - f:.0%}", ha="center", va="center", color="white"
        )
    ax_bar.set_title("Load split at design α")
    ax_bar.set_ylabel("Fraction of normal force")
    ax_bar.set_ylim(0.0, 1.0)
    ax_bar.legend(fontsize=8)

    fig2.suptitle(
        "LE/TE load split (CM about quarter chord, force & moment conserved)",
        fontsize=13,
    )
    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    save_path2 = RESULTS_DIR / "lei_airfoil_load_split.pdf"
    fig2.savefig(save_path2, bbox_inches="tight")
    print(f"Saved figure to {save_path2}")

    plt.show()


if __name__ == "__main__":
    main()
