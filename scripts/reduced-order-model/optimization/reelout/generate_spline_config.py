"""Generate reelout B-spline YAML configs from simple initial curves.

Edit the constants in the "Defaults" section for quick experiments, or pass
the same values through the command line. The B-spline implementation lives in
``awetrim.kinematics.parametrized_patterns``; this script only writes configs.
"""

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from awetrim.kinematics.parametrized_patterns import (
    PeriodicBSpline,
    make_bspline_path_parameters_from_named_curve,
    named_curve_angles,
)
from awetrim.utils.config_paths import (
    LEI_V3_DOWNLOOP_SPLINE_CONFIG,
    LEI_V3_GENERATED_SPLINE_CONFIG,
)
from awetrim.utils.color_palette import get_color_list, set_plot_style_no_latex

# ---------------------------------------------------------------------------
# Defaults: edit these for quick config-generation experiments
# ---------------------------------------------------------------------------
TEMPLATE_PATH = LEI_V3_DOWNLOOP_SPLINE_CONFIG
OUTPUT_PATH = LEI_V3_GENERATED_SPLINE_CONFIG

CURVE_TYPE = "lissajous"  # "lissajous" or "helix"
SPLINE_TYPE = "periodic"  # "periodic" or "open"
DOWNLOOPS = True

M = 10
R0 = 230.0
S_INIT = 0.0
S_FINAL = 2.0 * np.pi
N_FIT = 400

AZ_AMP0 = 0.32
BETA0 = 0.3
BETA_AMP0 = 0.15


def write_cycle_config_from_template(
    template_path=TEMPLATE_PATH,
    output_path=OUTPUT_PATH,
    curve_type=CURVE_TYPE,
    spline_type=SPLINE_TYPE,
    M=M,
    r0=R0,
    s_init=S_INIT,
    s_final=S_FINAL,
    n_fit=N_FIT,
    az_amp0=AZ_AMP0,
    beta0=BETA0,
    beta_amp0=BETA_AMP0,
    downloops=DOWNLOOPS,
):
    """Write a cycle config with generated B-spline path parameters."""
    template_path = Path(template_path)
    output_path = Path(output_path)

    with template_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    path_parameters = make_bspline_path_parameters_from_named_curve(
        spline_type=spline_type,
        M=M,
        r0=r0,
        s_init=s_init,
        s_final=s_final,
        n_fit=n_fit,
        curve_type=curve_type,
        az_amp0=az_amp0,
        beta0=beta0,
        beta_amp0=beta_amp0,
        downloops=downloops,
    )

    reelout_config = config.setdefault("reelout", {})
    reelout_config["pattern_type"] = f"spline_{spline_type}"
    reelout_config["path_parameters"] = path_parameters

    sim_parameters = reelout_config.setdefault("sim_parameters", {})
    sim_parameters["start_angle"] = float(s_init)
    sim_parameters["end_angle"] = float(s_final)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False, default_flow_style=False)

    return output_path


def plot_generated_curve(
    curve_type=CURVE_TYPE,
    spline_type=SPLINE_TYPE,
    M=M,
    r0=R0,
    s_init=S_INIT,
    s_final=S_FINAL,
    n_fit=N_FIT,
    az_amp0=AZ_AMP0,
    beta0=BETA0,
    beta_amp0=BETA_AMP0,
    downloops=DOWNLOOPS,
):
    """Plot the target curve and generated periodic spline control points."""
    if spline_type != "periodic":
        raise ValueError(
            "plot_generated_curve currently supports only periodic splines."
        )

    set_plot_style_no_latex()
    colors = get_color_list()

    path_parameters = make_bspline_path_parameters_from_named_curve(
        spline_type=spline_type,
        M=M,
        r0=r0,
        s_init=s_init,
        s_final=s_final,
        n_fit=n_fit,
        curve_type=curve_type,
        az_amp0=az_amp0,
        beta0=beta0,
        beta_amp0=beta_amp0,
        downloops=downloops,
    )

    C_phi = np.asarray(path_parameters["C_phi"], dtype=float).reshape((M, 1))
    C_beta = np.asarray(path_parameters["C_beta"], dtype=float).reshape((M, 1))
    pattern = PeriodicBSpline(
        M=M,
        C_phi=C_phi,
        C_beta=C_beta,
        s_init=s_init,
        s_final=s_final,
        downloops=downloops,
    )

    s_plot = np.linspace(s_init, s_final, 200, endpoint=True)
    phi_target, beta_target = named_curve_angles(
        s_plot,
        curve_type=curve_type,
        az_amp0=az_amp0,
        beta0=beta0,
        beta_amp0=beta_amp0,
        downloops=downloops,
    )
    phi_fit = np.array([float(pattern.azimuth(r0, s_value)) for s_value in s_plot])
    beta_fit = np.array([float(pattern.elevation(r0, s_value)) for s_value in s_plot])

    phi_ctrl = np.r_[C_phi.flatten(), C_phi.flatten()[0]]
    beta_ctrl = np.r_[C_beta.flatten(), C_beta.flatten()[0]]

    plt.figure(figsize=(5, 4))
    plt.plot(np.degrees(phi_target), np.degrees(beta_target), label="target")
    plt.plot(np.degrees(phi_fit), np.degrees(beta_fit), ".", label="B-spline")
    plt.plot(
        np.degrees(phi_ctrl),
        np.degrees(beta_ctrl),
        "--o",
        color=colors[2],
        label="control points",
    )
    plt.grid(True)
    plt.legend()
    plt.xlabel(r"$\phi$ ($^{\circ}$)")
    plt.ylabel(r"$\beta$ ($^{\circ}$)")
    plt.tight_layout()
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate B-spline reelout YAML configs from helix or Lissajous initial curves."
    )
    parser.add_argument("--template", default=TEMPLATE_PATH)
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Modify the template file instead of writing --output.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot the generated initial curve after writing the config.",
    )
    parser.add_argument(
        "--curve-type",
        choices=["lissajous", "helix"],
        default=CURVE_TYPE,
    )
    parser.add_argument(
        "--spline-type",
        choices=["periodic", "open"],
        default=SPLINE_TYPE,
    )
    parser.add_argument("--M", type=int, default=M)
    parser.add_argument("--r0", type=float, default=R0)
    parser.add_argument("--s-init", type=float, default=S_INIT)
    parser.add_argument("--s-final", type=float, default=S_FINAL)
    parser.add_argument("--n-fit", type=int, default=N_FIT)
    parser.add_argument("--az-amp0", type=float, default=AZ_AMP0)
    parser.add_argument("--beta0", type=float, default=BETA0)
    parser.add_argument("--beta-amp0", type=float, default=BETA_AMP0)
    parser.add_argument(
        "--downloops",
        action=argparse.BooleanOptionalAction,
        default=DOWNLOOPS,
        help="Generate downloop pattern (--downloops) or uploop (--no-downloops). Defaults to the DOWNLOOPS constant.",
    )
    return parser.parse_args()


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "write-config":
        sys.argv.pop(1)

    args = parse_args()
    downloops = args.downloops
    output_path = args.template if args.in_place else args.output

    written_path = write_cycle_config_from_template(
        template_path=args.template,
        output_path=output_path,
        curve_type=args.curve_type,
        spline_type=args.spline_type,
        M=args.M,
        r0=args.r0,
        s_init=args.s_init,
        s_final=args.s_final,
        n_fit=args.n_fit,
        az_amp0=args.az_amp0,
        beta0=args.beta0,
        beta_amp0=args.beta_amp0,
        downloops=downloops,
    )
    print(f"Wrote {written_path}")

    if args.plot:
        plot_generated_curve(
            curve_type=args.curve_type,
            spline_type=args.spline_type,
            M=args.M,
            r0=args.r0,
            s_init=args.s_init,
            s_final=args.s_final,
            n_fit=args.n_fit,
            az_amp0=args.az_amp0,
            beta0=args.beta0,
            beta_amp0=args.beta_amp0,
            downloops=downloops,
        )


if __name__ == "__main__":
    main()
