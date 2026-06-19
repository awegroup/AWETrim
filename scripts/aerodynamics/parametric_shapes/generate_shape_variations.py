"""Generate parametric 3D wing-shape variations from a baseline kite.

Loads a baseline ``aero_geometry.yaml`` and sweeps four planform degrees of
freedom -- aspect ratio, anhedral, taper, and twist -- writing one morphed
``aero_geometry.yaml`` per variation (quarter-chord anchored, area preserved by
default). A summary table of the resulting planform metrics is printed and
written to ``summary.csv``.

By default the sweep is **one factor at a time** (OAT): each parameter is varied
across its list while the others stay at the baseline, which keeps the case
count and the plots legible. Use ``--factorial`` for the full Cartesian product.

By default each variant is also evaluated with the VSM solver over an
angle-of-attack sweep (``--no-run-vsm`` to skip) and two comparison figures are
drawn -- a 3D overlay of the wing shapes and an aero comparison (lift curve,
drag polar, glide ratio) -- coloured by which parameter is being swept.

The morphing lives in the library:
    awetrim.aerodynamics.parametric_geometry  (WingSections, morph_wing_to)

Examples
--------
Default OAT sweep with VSM + comparison plots:

    python scripts/aerodynamics/parametric_shapes/generate_shape_variations.py

Custom taper/twist ranges, geometry only, headless:

    python scripts/aerodynamics/parametric_shapes/generate_shape_variations.py \
        --taper-ratios 0.7,1.0,1.4 --twist-degs -6,0,6 --no-run-vsm --no-show
"""

from __future__ import annotations

import argparse
import csv
import sys
from itertools import product
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DEFAULT_CONFIG_FOLDER = PROJECT_DIR / "data" / "LEI-V3-KITE"
DEFAULT_OUTPUT_ROOT = PROJECT_DIR / "results" / "aerodynamics" / "parametric_shapes"
DEFAULT_VSM_SRC = PROJECT_DIR.parent / "Vortex-Step-Method" / "src"

from awetrim.aerodynamics.parametric_geometry import WingSections, morph_wing_to

# Colour map per swept parameter group, for the comparison plots.
GROUP_CMAPS = {
    "aspect_ratio": "Blues",
    "anhedral": "Greens",
    "taper": "Oranges",
    "twist": "Purples",
    "factorial": "viridis",
}


def _float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-folder",
        default=str(DEFAULT_CONFIG_FOLDER),
        help="Baseline kite folder containing aero_geometry.yaml (and polar CSVs).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write the variant folders (default: "
        "results/aerodynamics/parametric_shapes/<kite>).",
    )
    parser.add_argument(
        "--aspect-ratio-scales",
        type=_float_list,
        default=[0.8, 0.9, 1.0, 1.1, 1.2],
        help="Comma-separated multipliers on the baseline aspect ratio.",
    )
    parser.add_argument(
        "--anhedral-scales",
        type=_float_list,
        default=[0.8, 0.9, 1.0, 1.1, 1.2],
        help="Comma-separated multipliers on the baseline anhedral angle.",
    )
    parser.add_argument(
        "--taper-ratios",
        type=_float_list,
        default=[0.7, 0.85, 1.0, 1.15, 1.3],
        help="Comma-separated multipliers on the baseline tip/root chord ratio.",
    )
    parser.add_argument(
        "--twist-degs",
        type=_float_list,
        default=[-6.0, -3.0, 0.0, 3.0, 6.0],
        help="Comma-separated tip washout angles to add [deg].",
    )
    parser.add_argument(
        "--factorial",
        action="store_true",
        help="Full Cartesian product of all four parameters (default: OAT).",
    )
    parser.add_argument(
        "--no-preserve-area",
        action="store_true",
        help="Do not hold flat area constant when changing aspect ratio.",
    )
    parser.add_argument(
        "--run-vsm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate each variant with VSM (AoA sweep) and draw comparison "
        "figures. Use --no-run-vsm to skip (default: on).",
    )
    parser.add_argument("--vsm-src", default=str(DEFAULT_VSM_SRC))
    parser.add_argument("--n-panels", type=int, default=18)
    parser.add_argument("--spanwise-panel-distribution", default="uniform")
    parser.add_argument("--umag", type=float, default=20.0, help="Apparent wind [m/s].")
    parser.add_argument(
        "--aoa-sweep",
        type=_float_list,
        default=[0.0, 4.0, 8.0, 12.0],
        help="Comma-separated angles of attack [deg] for --run-vsm.",
    )
    parser.add_argument(
        "--plot", action="store_true", help="Force the shape overlay even with --no-run-vsm."
    )
    parser.add_argument(
        "--no-show", action="store_true", help="Save figures without displaying them."
    )
    return parser.parse_args()


def build_combinations(args: argparse.Namespace) -> list[tuple[str, float, dict]]:
    """Return ``(group, value, morph-extra-kwargs)`` tuples for the sweep.

    ``group`` names the swept parameter (or ``"factorial"``/``"baseline"``),
    ``value`` is the swept value (used for plot shading), and the kwargs hold the
    non-baseline morph factors for that combination.
    """
    if args.factorial:
        combos = []
        for ar, anh, tp, tw in product(
            args.aspect_ratio_scales, args.anhedral_scales,
            args.taper_ratios, args.twist_degs,
        ):
            combos.append(("factorial", 0.0, _factors(ar, anh, tp, tw)))
        return combos

    # One factor at a time: baseline once, then each axis in turn.
    combos: list[tuple[str, float, dict]] = [("baseline", 1.0, _factors(1.0, 1.0, 1.0, 0.0))]
    for s in args.aspect_ratio_scales:
        if s != 1.0:
            combos.append(("aspect_ratio", s, _factors(s, 1.0, 1.0, 0.0)))
    for s in args.anhedral_scales:
        if s != 1.0:
            combos.append(("anhedral", s, _factors(1.0, s, 1.0, 0.0)))
    for r in args.taper_ratios:
        if r != 1.0:
            combos.append(("taper", r, _factors(1.0, 1.0, r, 0.0)))
    for t in args.twist_degs:
        if t != 0.0:
            combos.append(("twist", t, _factors(1.0, 1.0, 1.0, t)))
    return combos


def _factors(ar_scale: float, anh_scale: float, taper_ratio: float, twist_deg: float) -> dict:
    return {
        "ar_scale": ar_scale,
        "anh_scale": anh_scale,
        "taper_ratio": taper_ratio,
        "twist_deg": twist_deg,
    }


def morph_from_factors(
    baseline: WingSections,
    factors: dict,
    *,
    base_ar: float,
    base_anh: float,
    preserve_area: bool,
) -> WingSections:
    kwargs = {
        "target_aspect_ratio": base_ar * factors["ar_scale"],
        "preserve_area": preserve_area,
    }
    if factors["anh_scale"] != 1.0 and abs(base_anh) > 1e-6:
        kwargs["target_anhedral_deg"] = base_anh * factors["anh_scale"]
    if factors["taper_ratio"] != 1.0:
        kwargs["taper_ratio"] = factors["taper_ratio"]
    if factors["twist_deg"] != 0.0:
        kwargs["twist_deg"] = factors["twist_deg"]
    return morph_wing_to(baseline, **kwargs)


def variant_name(group: str, factors: dict) -> str:
    if group == "baseline":
        return "baseline"
    if group == "aspect_ratio":
        return f"AR_x{factors['ar_scale']:.2f}"
    if group == "anhedral":
        return f"anh_x{factors['anh_scale']:.2f}"
    if group == "taper":
        return f"taper_x{factors['taper_ratio']:.2f}"
    if group == "twist":
        return f"twist_{factors['twist_deg']:+.0f}deg"
    return (
        f"AR{factors['ar_scale']:.2f}_anh{factors['anh_scale']:.2f}"
        f"_tap{factors['taper_ratio']:.2f}_tw{factors['twist_deg']:+.0f}"
    )


def evaluate_with_vsm(
    variant_yaml: Path,
    *,
    vsm_src: str,
    n_panels: int,
    spanwise_panel_distribution: str,
    umag: float,
    aoa_sweep: list[float],
) -> dict[str, np.ndarray]:
    """Run an AoA sweep with VSM, returning the full CL/CD/L-over-D polar."""
    if vsm_src and str(Path(vsm_src).resolve()) not in sys.path:
        sys.path.insert(0, str(Path(vsm_src).resolve()))
    from VSM.core.BodyAerodynamics import BodyAerodynamics
    from VSM.core.Solver import Solver

    body = BodyAerodynamics.instantiate(
        n_panels=n_panels,
        file_path=str(variant_yaml),
        spanwise_panel_distribution=spanwise_panel_distribution,
        bridle_path=None,
    )
    solver = Solver()
    aoa = np.asarray(aoa_sweep, dtype=float)
    cl = np.full(aoa.shape, np.nan)
    cd = np.full(aoa.shape, np.nan)
    for i, a in enumerate(aoa):
        body.va_initialize(Umag=umag, angle_of_attack=float(a), side_slip=0.0)
        res = solver.solve(body)
        cl[i], cd[i] = float(res["cl"]), float(res["cd"])
    with np.errstate(divide="ignore", invalid="ignore"):
        ld = np.where(cd > 0, cl / cd, np.nan)
    return {"aoa_deg": aoa, "cl": cl, "cd": cd, "ld": ld}


def main() -> None:
    args = parse_args()
    config_folder = Path(args.config_folder).expanduser().resolve()
    baseline_yaml = config_folder / "aero_geometry.yaml"
    if not baseline_yaml.exists():
        raise FileNotFoundError(f"aero_geometry.yaml not found in {config_folder}")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else DEFAULT_OUTPUT_ROOT / config_folder.name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline = WingSections.from_yaml(baseline_yaml)
    base_ar = baseline.aspect_ratio
    base_anh = baseline.anhedral_angle_deg
    print(
        f"Baseline: AR={base_ar:.3f}  anhedral={base_anh:.3f} deg  "
        f"taper={baseline.taper_ratio:.3f}  span={baseline.projected_span:.3f} m  "
        f"area={baseline.area:.3f} m^2"
    )

    combos = build_combinations(args)
    mode = "factorial" if args.factorial else "OAT"
    print(f"Sweep mode: {mode}  ({len(combos)} variation(s))")

    rows = []
    records: list[dict] = []
    for group, value, factors in combos:
        variant = morph_from_factors(
            baseline, factors, base_ar=base_ar, base_anh=base_anh,
            preserve_area=not args.no_preserve_area,
        )
        name = variant_name(group, factors)
        variant_dir = output_dir / name
        variant_dir.mkdir(parents=True, exist_ok=True)
        variant.to_yaml(
            variant_dir / "aero_geometry.yaml",
            resolve_csv_paths_relative_to=config_folder,
        )

        record = {"name": name, "group": group, "value": value, "sections": variant}
        row = {
            "name": name,
            "group": group,
            **{k: round(v, 6) for k, v in variant.properties().items()},
        }

        if args.run_vsm:
            try:
                aero = evaluate_with_vsm(
                    variant_dir / "aero_geometry.yaml",
                    vsm_src=args.vsm_src,
                    n_panels=args.n_panels,
                    spanwise_panel_distribution=args.spanwise_panel_distribution,
                    umag=args.umag,
                    aoa_sweep=args.aoa_sweep,
                )
                record["aero"] = aero
                best = int(np.nanargmax(aero["ld"]))
                row.update(
                    {
                        "cl_bestLD": round(float(aero["cl"][best]), 4),
                        "cd_bestLD": round(float(aero["cd"][best]), 5),
                        "max_LD": round(float(aero["ld"][best]), 3),
                        "aoa_bestLD_deg": float(aero["aoa_deg"][best]),
                    }
                )
            except Exception as exc:  # noqa: BLE001 - report and continue
                print(f"  [VSM failed for {name}] {exc}")

        records.append(record)
        rows.append(row)
        print(f"  wrote {name}")

    write_summary(rows, output_dir)

    figures = []
    if args.plot or args.run_vsm:
        figures.append(plot_geometry_comparison(records, output_dir))
    if any("aero" in r for r in records):
        figures.append(plot_aero_comparison(records, output_dir))

    if figures and not args.no_show:
        import matplotlib.pyplot as plt

        plt.show()


def write_summary(rows: list[dict], output_dir: Path) -> None:
    summary_path = output_dir / "summary.csv"
    fieldnames = {key for row in rows for key in row}
    preferred = [
        "name", "group", "aspect_ratio", "anhedral_angle_deg", "taper_ratio",
        "tip_twist_deg", "projected_span", "flat_span", "area", "mean_chord",
        "cl_bestLD", "cd_bestLD", "max_LD", "aoa_bestLD_deg",
    ]
    ordered = [c for c in preferred if c in fieldnames] + [
        c for c in fieldnames if c not in preferred
    ]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} variation(s) and summary -> {summary_path}")


def _assign_colors(records: list[dict]) -> dict[str, tuple]:
    """Map variant name -> (color, linewidth), coloured by swept-parameter group."""
    import matplotlib.pyplot as plt

    by_group: dict[str, list[dict]] = {}
    for r in records:
        by_group.setdefault(r["group"], []).append(r)

    colors: dict[str, tuple] = {}
    for group, items in by_group.items():
        if group == "baseline":
            for r in items:
                colors[r["name"]] = ("black", 2.2)
            continue
        cmap = plt.get_cmap(GROUP_CMAPS.get(group, "viridis"))
        items_sorted = sorted(items, key=lambda r: r["value"])
        shades = np.linspace(0.4, 0.9, len(items_sorted))
        for r, s in zip(items_sorted, shades):
            colors[r["name"]] = (cmap(s), 1.3)
    return colors


def _group_legend_names(records: list[dict]) -> set[str]:
    """First variant name in each group (so the legend has one entry per group)."""
    seen, names = set(), set()
    for r in records:
        if r["group"] not in seen:
            seen.add(r["group"])
            names.add(r["name"])
    return names


def plot_geometry_comparison(records: list[dict], output_dir: Path):
    """3D overlay of the wing outlines (LE/TE), coloured by swept parameter."""
    import matplotlib.pyplot as plt  # local import: optional dependency

    colors = _assign_colors(records)
    legend_names = _group_legend_names(records)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    for r in records:
        sec = r["sections"]
        color, lw = colors[r["name"]]
        label = r["group"] if r["name"] in legend_names else "_nolegend_"
        ax.plot(sec.le[:, 0], sec.le[:, 1], sec.le[:, 2], color=color, lw=lw, label=label)
        ax.plot(sec.te[:, 0], sec.te[:, 1], sec.te[:, 2], color=color, lw=lw, ls="--")
        step = max(1, sec.n_sections // 8)
        for i in range(0, sec.n_sections, step):
            ax.plot(
                [sec.le[i, 0], sec.te[i, 0]],
                [sec.le[i, 1], sec.te[i, 1]],
                [sec.le[i, 2], sec.te[i, 2]],
                color=color, lw=0.4, alpha=0.4,
            )

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("Parametric wing shape variations (LE solid, TE dashed)")
    ax.legend(fontsize=8, loc="upper right", title="swept parameter")
    _set_axes_equal_3d(ax)
    fig.tight_layout()

    path = output_dir / "shape_comparison.pdf"
    fig.savefig(path, bbox_inches="tight")
    print(f"Wrote {path}")
    return fig


def plot_aero_comparison(records: list[dict], output_dir: Path):
    """CL-alpha, drag polar, and L/D-alpha comparison, coloured by swept parameter."""
    import matplotlib.pyplot as plt  # local import: optional dependency

    colors = _assign_colors(records)
    legend_names = _group_legend_names(records)
    aero_records = [r for r in records if "aero" in r]

    fig, (ax_cl, ax_polar, ax_ld) = plt.subplots(1, 3, figsize=(15, 4.5))
    for r in aero_records:
        aero = r["aero"]
        color, lw = colors[r["name"]]
        label = r["group"] if r["name"] in legend_names else "_nolegend_"
        ax_cl.plot(aero["aoa_deg"], aero["cl"], "-o", ms=3, color=color, lw=lw, label=label)
        ax_polar.plot(aero["cd"], aero["cl"], "-o", ms=3, color=color, lw=lw)
        ax_ld.plot(aero["aoa_deg"], aero["ld"], "-o", ms=3, color=color, lw=lw)

    ax_cl.set(xlabel="angle of attack [deg]", ylabel="CL", title="Lift curve")
    ax_polar.set(xlabel="CD", ylabel="CL", title="Drag polar")
    ax_ld.set(xlabel="angle of attack [deg]", ylabel="L/D", title="Glide ratio")
    for ax in (ax_cl, ax_polar, ax_ld):
        ax.grid(True, alpha=0.3)
    ax_cl.legend(fontsize=8, loc="best", title="swept parameter")
    fig.tight_layout()

    path = output_dir / "aero_comparison.pdf"
    fig.savefig(path, bbox_inches="tight")
    print(f"Wrote {path}")
    return fig


def _set_axes_equal_3d(ax) -> None:
    """Equal aspect ratio for a 3D axis (matplotlib has no native support)."""
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    centre = limits.mean(axis=1)
    radius = 0.5 * np.max(limits[:, 1] - limits[:, 0])
    ax.set_xlim3d(centre[0] - radius, centre[0] + radius)
    ax.set_ylim3d(centre[1] - radius, centre[1] + radius)
    ax.set_zlim3d(centre[2] - radius, centre[2] + radius)


if __name__ == "__main__":
    main()
