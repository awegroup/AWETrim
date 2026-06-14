"""Solve one VSM aerodynamic trim state.

To trim on a deformed shape from an aerostructural run, pass
``--deformed-from <case_dir>`` (where ``case_dir`` contains the deformed
``aero_geometry.yaml`` and ``struc_geometry.yaml``). The deformed geometries
are used by VSM and for plotting, but mass, inertia tensor and centre of
gravity are still read from ``system.yaml`` in ``--config-folder`` -- this
script does NOT recompute inertia or CoG from the deformed shape.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common import (
    add_common_arguments,
    build_body,
    build_system_model,
    output_dir,
    parsed_common,
    print_trim_summary,
    save_figure,
    write_json,
)
from kite_tether_plot import draw_kite_tether, tether_info_str

from awetrim.aerodynamics.vsm_quasi_steady import (
    solve_vsm_qs_trim_with_williams_tether,
    solve_vsm_quasi_steady_trim,
)
from awetrim.system.williams_tether import WilliamsTether


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Solve one VSM aerodynamic trim state."
    )
    add_common_arguments(parser)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()
    values = parsed_common(args)
    out_dir = output_dir(args, "single_state")

    body, body_props = build_body(args)
    system_model = build_system_model(args)

    use_williams = isinstance(getattr(system_model, "tether", None), WilliamsTether)

    if use_williams:
        print(
            "Tether model: WilliamsTether -> running joint trim+tether least_squares."
        )
        result, _ = solve_vsm_qs_trim_with_williams_tether(
            body_aero=body,
            center_of_gravity=values["center_of_gravity"],
            reference_point=values["reference_point"],
            system_model=system_model,
            x_guess=values["x_guess"],
            bounds_lower=values["bounds_lower"],
            bounds_upper=values["bounds_upper"],
            include_gravity=args.include_gravity,
            moment_tolerance=args.moment_tolerance,
            max_nfev=args.max_nfev,
        )
    else:
        result, _ = solve_vsm_quasi_steady_trim(
            body_aero=body,
            center_of_gravity=values["center_of_gravity"],
            reference_point=values["reference_point"],
            system_model=system_model,
            x_guess=values["x_guess"],
            bounds_lower=values["bounds_lower"],
            bounds_upper=values["bounds_upper"],
            include_gravity=args.include_gravity,
            moment_tolerance=args.moment_tolerance,
            return_timing_breakdown=True,
            max_nfev=args.max_nfev,
        )

    print_trim_summary(result)

    if use_williams:
        print("--- Williams tether ---")
        print(f"  elevation_last [deg] : {result['williams_elevation_last_deg']:.4f}")
        print(f"  azimuth_last   [deg] : {result['williams_azimuth_last_deg']:.4f}")
        print(f"  tether_length  [m]   : {result['williams_tether_length']:.4f}")
        print(f"  ground residual [m]  : {result['williams_ground_residual']}")
        print(
            f"  |F_kite_resultant|   : {np.linalg.norm(result['force_kite_resultant']):.3f} N"
        )

    json_path = (
        Path(args.output_json) if args.output_json else out_dir / "trim_result.json"
    )
    write_json(json_path, result)

    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["cmx", "cmy", "cmz", "cfx", "cfy"]
    values_plot = np.r_[
        np.asarray(result["cm"], dtype=float), result["cfx"], result["cfy"]
    ]
    ax.bar(
        labels,
        values_plot,
        color=["#4C78A8", "#4C78A8", "#4C78A8", "#F58518", "#F58518"],
    )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("Residual coefficient [-]")
    ax.set_title("VSM aerodynamic trim residuals")
    fig.tight_layout()
    save_figure(fig, out_dir / "trim_residuals.pdf")
    if args.no_show:
        plt.close(fig)

    if use_williams:
        fig2 = plt.figure(figsize=(7, 6))
        ax2 = fig2.add_subplot(111, projection="3d")
        draw_kite_tether(
            ax2, result, system_model, body, body_props.get("struc_geometry_path")
        )
        ax2.set_title(
            f"Williams tether shape (kite -> ground)\n"
            f"[{tether_info_str(system_model)}]"
        )
        ax2.legend()
        fig2.tight_layout()
        save_figure(fig2, out_dir / "williams_tether_shape.pdf")
        if args.no_show:
            plt.close(fig2)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
