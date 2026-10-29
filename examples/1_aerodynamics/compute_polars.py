"""Lift and drag polars of the kite: C_L and C_D against angle of attack.

The wing is held at one shape (the built one, or a deformed one with
``--deformed-from <case_dir>``) and VSM is solved at each angle of attack for
a fixed apparent-wind speed, without sideslip or body rate and without a
re-trim (``run_frozen_geometry_alpha_sweep``). This is the wing's own
aerodynamic response, not a flyable state: the bridle pitch, the tether and
gravity do not enter.

Coefficients are given on two areas. VSM normalises by the wing's projected
area; the ROM states its own reference area (``aerodynamics.reference_area``
in the ROM config the kite's system.yaml selects), which is the area the
trajectory steps use. L/D does not depend on the area.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from common import add_common_arguments, build_body, csv_vector, output_dir, save_figure
from awetrim.aerodynamics.vsm_quasi_steady import default_vsm_solver
from awetrim.aerostructural.aerodynamic_vsm import run_frozen_geometry_alpha_sweep
from awetrim.system.factory import resolve_rom_config_path


def rom_reference_area(config_folder: Path) -> float | None:
    """The reference area the kite's ROM coefficients are normalised by."""
    rom_path = resolve_rom_config_path(config_folder / "system.yaml")
    if rom_path is None or not Path(rom_path).is_file():
        return None
    with open(rom_path, "r", encoding="utf-8") as f:
        area = ((yaml.safe_load(f) or {}).get("aerodynamics") or {}).get("reference_area")
    return float(area) if area is not None else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="C_L and C_D against angle of attack on a frozen wing shape."
    )
    add_common_arguments(parser)
    parser.add_argument("--apparent-wind-speed", type=float, default=20.0,
                        help="Apparent-wind speed held over the sweep [m/s].")
    parser.add_argument("--alpha-min", type=float, default=-6.0, help="[deg]")
    parser.add_argument("--alpha-max", type=float, default=25.0, help="[deg]")
    parser.add_argument("--alpha-step", type=float, default=1.0, help="[deg]")
    args = parser.parse_args()
    out_dir = output_dir(args, "polars")

    body, _ = build_body(args)
    solver = default_vsm_solver(
        csv_vector(args.reference_point, length=3, name="reference-point")
    )
    alphas = np.arange(args.alpha_min, args.alpha_max + 0.5 * args.alpha_step,
                       args.alpha_step)
    rows = run_frozen_geometry_alpha_sweep(
        body, solver, va_magnitude=args.apparent_wind_speed, alpha_values_deg=alphas
    )

    area_vsm = float(body.wings[0].compute_projected_area())
    area_rom = rom_reference_area(Path(args.config_folder))
    alpha_deg = np.rad2deg([r["alpha"] for r in rows])
    cl = np.array([r["cl"] for r in rows])
    cd = np.array([r["cd"] for r in rows])
    scale = area_vsm / area_rom if area_rom else np.nan

    csv_path = out_dir / "polars.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "alpha_deg", f"CL@VSM_{area_vsm:.2f}m2", f"CD@VSM_{area_vsm:.2f}m2",
            f"CL@ROM_{area_rom}m2", f"CD@ROM_{area_rom}m2", "L/D", "success",
        ])
        for r, a, l, d in zip(rows, alpha_deg, cl, cd):
            writer.writerow([f"{a:.3f}", f"{l:.5f}", f"{d:.5f}", f"{l * scale:.5f}",
                             f"{d * scale:.5f}", f"{l / d:.4f}", r["success"]])
    print(f"Wrote {csv_path}")

    print(f"\nV_a = {args.apparent_wind_speed:g} m/s; areas: VSM projected "
          f"{area_vsm:.2f} m2, ROM reference {area_rom} m2")
    print(f"{'alpha':>7} {'CL@VSM':>8} {'CD@VSM':>8} {'CL@ROM':>8} {'CD@ROM':>8} {'L/D':>6}")
    for a, l, d in zip(alpha_deg, cl, cd):
        print(f"{a:7.1f} {l:8.4f} {d:8.4f} {l * scale:8.4f} {d * scale:8.4f} {l / d:6.2f}")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    label = f"VSM projected area ({area_vsm:.2f} m$^2$)"
    axes[0].plot(alpha_deg, cl, marker="o", ms=3, label=label)
    axes[1].plot(alpha_deg, cd, marker="o", ms=3, label=label)
    if area_rom:
        label_rom = f"ROM reference area ({area_rom:g} m$^2$)"
        axes[0].plot(alpha_deg, cl * scale, marker="o", ms=3, label=label_rom)
        axes[1].plot(alpha_deg, cd * scale, marker="o", ms=3, label=label_rom)
    axes[2].plot(alpha_deg, cl / cd, marker="o", ms=3, color="k")
    axes[0].set_ylabel("$C_L$ (-)")
    axes[1].set_ylabel("$C_D$ (-)")
    axes[2].set_ylabel("$L/D$ (-)")
    for ax in axes:
        ax.set_xlabel(r"$\alpha$ (deg)")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Frozen-shape polars, $V_a$ = {args.apparent_wind_speed:g} m s$^{{-1}}$")
    save_figure(fig, out_dir / "polars.png")


if __name__ == "__main__":
    main()
