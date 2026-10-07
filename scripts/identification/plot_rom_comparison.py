"""Compare the LEI-V3 aerostructural ROM with the semi-empirical one.

Two figures, both against the exact aerostructural data the new ROM was
identified on (run identify_rom_aerostructural.py first):

    rom_comparison_polars    C_L(alpha), C_D(alpha) and the C_L-C_D polar at the
                             powered (l_dp 1.7 m) and depowered (2.1 m) setting,
                             centre of the window, u_s = 0, v_a = 19 m s^-1
    rom_comparison_theta_b   bridle pitch theta_b against the power-tape length

Drag is compared WITH the KCU in every curve: the semi-empirical C_D carries it
inside, the aerostructural ROM adds it explicitly from the system file's KCU
hardware (2019 flown: KCU + onboard turbine), and the coupled data carry their
own. The wing-only aerostructural C_D is drawn thin for reference. Every
coefficient is on the ROMs' reference area, 19.75 m^2.

theta_b depends on which force the bridle resultant F_b contains. With the KCU
drag inside C_D (semi-empirical) F_b is along the tether; with the KCU drag a
separate force at the bridle point (aerostructural) F_b also carries it. The
data are drawn in BOTH conventions, each next to the relation that uses it.

Usage (project root)
    python scripts/identification/plot_rom_comparison.py
"""

from __future__ import annotations

import contextlib
import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from awetrim.aerodynamics.kcu_drag import KcuDragModel
from awetrim.identification.aero_polynomial import fits_from_rom_aerodynamics
from awetrim.plotting.plotting import PALETTE, set_plot_style
from awetrim.system.factory import create_system_model_from_yaml
from awetrim.system.kite import canonical_aero_params, stall_blend
from awetrim.utils.config_paths import (
    LEI_V3_ROM_AEROSTRUCTURAL_CONFIG,
    LEI_V3_ROM_SEMI_EMPIRICAL_CONFIG,
    LEI_V3_SYSTEM_FLOWN_2019_CONFIG,
)

from identify_rom_aerostructural import OUT_DIR, S_REF, UP_BAND

#: (label, power-tape length [m]) of the polar rows.
SETTINGS = (("Powered", 1.7), ("Depowered", 2.1))
#: Target apparent speed of the chain whose polars are drawn [m s^-1].
VA_CHAIN = 19
#: Paper Table C1 (Cayon, van Deursen, Schmehl 2026), reel-out / reel-in.
PAPER_THETA_B = ((1.7, 6.88), (2.1, 15.47))

STYLE = {
    "data": dict(color=PALETTE["Black"], marker="o", ms=4, ls="none", mfc="none"),
    "new": dict(color=PALETTE["Blue"], lw=2.0, ls="-"),
    "old": dict(color=PALETTE["Vermillion"], lw=2.0, ls="--"),
}


def load_rom(path):
    aero = yaml.safe_load(path.read_text(encoding="utf-8"))["aerodynamics"]
    params = canonical_aero_params(aero.get("params", {}))
    return {
        "fits": fits_from_rom_aerodynamics({**aero, "params": params}),
        "theta_b": lambda up: params["angle_pitch_tether_0"]
        + params["slope_angle_pitch_tether_depower"] * up,
        "area": float(aero.get("reference_area", S_REF)),
        "kcu_in_cd": bool(aero.get("kcu_drag_in_coefficients", True)),
        "params": params,
    }


def kcu_2019() -> KcuDragModel:
    with contextlib.redirect_stdout(io.StringIO()):
        model = create_system_model_from_yaml(LEI_V3_SYSTEM_FLOWN_2019_CONFIG)
    return KcuDragModel.from_system_model(model)


def cd_kcu(kcu, alpha, theta_b, area):
    """KCU C_D at the centre of the window: the KCU hangs on the tether, and
    the apparent wind meets the tether at 90 deg - alpha_b, alpha_b = alpha +
    theta_b (paper Eq. 1). With the KCU drag separate, theta_b refers to F_b."""
    alpha_b = np.asarray(alpha) + theta_b
    va = np.stack([np.cos(alpha_b), np.zeros_like(alpha_b), np.sin(alpha_b)], -1)
    return np.array([kcu.drag_coefficient(v, [0.0, 0.0, 1.0], area) for v in va])


def evaluate(rom, alpha, up, va):
    data = {"alpha": alpha, "u_p": np.full_like(alpha, up),
            "u_s": np.zeros_like(alpha), "v_a": np.full_like(alpha, va)}
    params = rom["params"]
    if "angle_of_attack_stall" in params:
        data["stall"] = stall_blend(alpha, params["angle_of_attack_stall"],
                                    params["width_stall"], xp=np)
    return rom["fits"]["CL"].predict(data), rom["fits"]["CD"].predict(data)


def nearest_anchor(polar, up):
    rows = polar[(polar.source == "polar") & (polar.u_s == 0.0)
                 & polar.chain.str.contains(f"udp_continuation_billow9_va{VA_CHAIN}$")]
    best = rows.loc[(rows.u_p - up).abs().idxmin(), "anchor"]
    return rows[rows.anchor == best].sort_values("alpha")


def plot_polars(polar, trims, new, old, kcu):
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.6))
    for row, (label, up) in enumerate(SETTINGS):
        anchor = nearest_anchor(polar, up)
        up_a, va_a = float(anchor.u_p.iloc[0]), float(anchor.v_a.median())
        trim = trims[trims.anchor == anchor.anchor.iloc[0]].iloc[0]
        # The data's KCU drag at each polar step, with the coupled hardware.
        theta_data = float(trim.theta_b)
        alpha = anchor.alpha.to_numpy()
        cl_d = anchor.CL.to_numpy()
        cd_d = anchor.CD.to_numpy() + cd_kcu(kcu, alpha, theta_data, S_REF)

        grid = np.linspace(alpha.min(), alpha.max(), 200)
        cl_new, cd_new_wing = evaluate(new, grid, up_a, va_a)
        cd_new = cd_new_wing + cd_kcu(kcu, grid, new["theta_b"](up_a), new["area"])
        cl_old, cd_old = evaluate(old, grid, up_a, va_a)
        deg = np.rad2deg

        ax_cl, ax_cd, ax_pol = axes[row]
        ax_cl.plot(deg(alpha), cl_d, label="Aerostructural (frozen shape)", **STYLE["data"])
        ax_cl.plot(deg(grid), cl_new, label="Aerostructural ROM", **STYLE["new"])
        ax_cl.plot(deg(grid), cl_old, label="Semi-empirical ROM", **STYLE["old"])
        ax_cd.plot(deg(alpha), cd_d, **STYLE["data"])
        ax_cd.plot(deg(grid), cd_new, **STYLE["new"])
        ax_cd.plot(deg(grid), cd_new_wing, color=PALETTE["Blue"], lw=1.0, ls=":",
                   label="Aerostructural ROM, wing only")
        ax_cd.plot(deg(grid), cd_old, **STYLE["old"])
        ax_pol.plot(cd_d, cl_d, **STYLE["data"])
        ax_pol.plot(cd_new, cl_new, **STYLE["new"])
        ax_pol.plot(cd_old, cl_old, **STYLE["old"])
        for ax in (ax_cl, ax_cd):
            ax.axvline(np.rad2deg(trim.alpha_w), color=PALETTE["Black"], lw=0.8,
                       alpha=0.4)
        ax_cl.set_ylabel(f"{label}, $l_\\mathrm{{dp}}$ = {up_a:.2f} m\n$C_L$")
        ax_cd.set_ylabel("$C_D$")
        ax_pol.set_ylabel("$C_L$")
        if row == 1:
            ax_cl.set_xlabel(r"$\alpha_\mathrm{w}$ ($^\circ$)")
            ax_cd.set_xlabel(r"$\alpha_\mathrm{w}$ ($^\circ$)")
            ax_pol.set_xlabel("$C_D$")
    axes[0, 0].legend(loc="lower right", frameon=False, fontsize=9)
    axes[0, 1].legend(loc="upper left", frameon=False, fontsize=9)
    fig.suptitle(
        f"LEI-V3 ROMs at the centre of the wind window ($u_s$ = 0, "
        f"$v_a$ $\\approx$ {VA_CHAIN} m s$^{{-1}}$); $C_D$ includes the 2019 KCU; "
        f"reference area {S_REF} m$^2$; vertical line: coupled trim",
        fontsize=10,
    )
    fig.tight_layout()
    return fig


def plot_theta_b(trims, new, old):
    rows = trims[trims.u_s == 0.0].sort_values("u_p")
    up = np.linspace(rows.u_p.min(), rows.u_p.max(), 100)
    deg = np.rad2deg
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.axvspan(*UP_BAND, color=PALETTE["Black"], alpha=0.06, lw=0)
    ax.text(UP_BAND[0] + 0.01, 0.5, "aerostructural\nROM fit band", fontsize=8,
            va="bottom", color="0.35")
    ax.plot(rows.u_p, deg(rows.theta_b), label="Aerostructural, $F_b$ incl. KCU drag",
            **STYLE["data"])
    ax.plot(rows.u_p, deg(rows.alpha_b_tether - rows.alpha_w),
            label="Aerostructural, $F_b$ along the tether", color=PALETTE["Black"],
            marker="x", ms=4, ls="none")
    ax.plot(up, deg(new["theta_b"](up)), label="Aerostructural ROM (KCU drag separate)",
            **STYLE["new"])
    ax.plot(up, deg(old["theta_b"](up)), label="Semi-empirical ROM (KCU in $C_D$)",
            **STYLE["old"])
    ax.plot(*zip(*PAPER_THETA_B), ls="none", marker="s", ms=7,
            color=PALETTE["Vermillion"], mfc="none", label="Paper Table C1")
    ax.set_xlabel(r"$l_\mathrm{dp}$ (m)")
    ax.set_ylabel(r"$\theta_b$ ($^\circ$)")
    ax.set_title("Bridle pitch against power-tape length, all $v_a$, $u_s$ = 0",
                 fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    return fig


def main() -> None:
    set_plot_style()
    polar = pd.read_csv(OUT_DIR / "polar_samples.csv")
    trims = pd.read_csv(OUT_DIR / "trim_samples.csv")
    new = load_rom(LEI_V3_ROM_AEROSTRUCTURAL_CONFIG)
    old = load_rom(LEI_V3_ROM_SEMI_EMPIRICAL_CONFIG)
    assert not new["kcu_in_cd"] and old["kcu_in_cd"]
    assert new["area"] == old["area"] == S_REF, "compare on one reference area"
    kcu = kcu_2019()
    for name, fig in (("rom_comparison_polars", plot_polars(polar, trims, new, old, kcu)),
                      ("rom_comparison_theta_b", plot_theta_b(trims, new, old))):
        for ext in ("png", "pdf"):
            fig.savefig(OUT_DIR / f"{name}.{ext}", dpi=200)
        print(f"Wrote {OUT_DIR / name}.png/.pdf")


if __name__ == "__main__":
    main()
