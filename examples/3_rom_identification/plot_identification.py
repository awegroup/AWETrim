# Copyright (c) 2023-2026 Oriol Cayon, Delft University of Technology
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Figures of a ROM identification, from its dataset and fitted ROM only.

Run after identify_rom_aerostructural.py. Reads the sweep's ``trims.csv``, the
fit's tidy samples and its ``rom_config_aerostructural.yaml``; needs no flight
data. Coefficients are on the ROM's reference area.

    chains          every solved state in (u_p, u_s), per wind
    trim_vs_depower tension and angle of attack along the depower chains
    gust_margin     margin to the stall onset after a gust, along the same chains
    polars          lift and drag, the dataset against the ROM
    steering        course rate and tension against steering, per tape
    bridle_and_roll bridle pitch and roll gain, the dataset against the fit
    rom_check       the ROM against the coupled trims

Usage (project root)
    python examples/3_rom_identification/plot_identification.py --fidelity wireframe
    python examples/3_rom_identification/plot_identification.py --out docs/identification/figures
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import yaml

from awetrim.identification import aerostructural_dataset as ds
from awetrim.plotting.plotting import PALETTE, set_plot_style
from awetrim.system.kite import roll_gain, stall_blend

PROJECT_DIR = Path(__file__).resolve().parents[2]

GUST_SPEED = 1.0  # [m/s], the gust the depower limit is set against

# The onset measure reads a lift drop against a curve extrapolated from below the
# trim, so it only means something for states near their stall: from this far below
# the ROM's lower depower limit to this far above it [m].
GUST_MARGIN_TAPE_SPAN = (0.15, 0.30)

WIND_COLOURS = (PALETTE["Sky Blue"], PALETTE["Vermillion"], PALETTE["Blue"],
                PALETTE["Bluish Green"], PALETTE["Orange"])
MARKERS = ("o", "s", "^", "D", "v")
N_POLAR_TAPES = 4


def blues(n):
    """``n`` steps of one hue, light to dark, for an ordered quantity."""
    return plt.get_cmap("Blues")(np.linspace(0.4, 1.0, n))


def wind_styles(trims):
    """Colour and marker of each wind of the dataset, in order."""
    winds = sorted(trims.wind_speed.unique())
    return {w: (WIND_COLOURS[i % len(WIND_COLOURS)], MARKERS[i % len(MARKERS)])
            for i, w in enumerate(winds)}


def reference_wind(trims):
    """The middle wind among those with steering chains."""
    winds = sorted(trims[trims.chain.str.contains("steering")].wind_speed.unique())
    return winds[len(winds) // 2]


def wind_label(wind):
    return rf"$v_w={wind:g}$ m s$^{{-1}}$"


def rom_coefficient(rom, key, alpha, u_p, u_s, v_a):
    """C_L or C_D of the ROM file at numeric inputs (the terms of ``Kite``)."""
    aero = rom["aerodynamics"]
    params = aero["params"]
    variables = {
        "alpha": alpha, "u_p": u_p, "u_s": u_s, "v_a": v_a,
        "stall": stall_blend(alpha, params["angle_of_attack_stall"],
                             params["width_stall"], xp=np),
    }
    total = params[f"{key}0"] + 0.0 * alpha
    for term in aero["coefficients"][key]:
        powers = term.get("vars", {term.get("var"): term.get("power", 1)})
        value = term["coef"]
        for name, power in powers.items():
            value = value * variables[name] ** power
        total = total + value
    return total


def stall_onset_margin(polar, trims):
    """Depower-chain states with their margin to the stall onset after a gust.

    The gust of ``GUST_SPEED`` turns the inflow by atan(gust / v_a).
    """
    rows = []
    for _, trim in trims[trims.chain.str.contains("depower")].iterrows():
        samples = polar[polar.anchor == trim.anchor].sort_values("d_alpha_deg")
        onset = ds.stall_onset_increment(samples.d_alpha_deg, samples.CL)
        if not np.isfinite(onset):
            continue
        gust = np.degrees(np.arctan(GUST_SPEED / trim.v_a))
        rows.append({"wind_speed": trim.wind_speed, "u_p": trim.u_p, "margin": onset - gust})
    return pd.DataFrame(rows).sort_values(["wind_speed", "u_p"])


def mark_limits(ax, rom, fit_lower):
    """The ROM's lower depower limit and the fit's lower bound, as vertical lines."""
    ax.axvline(rom["validity"]["input_depower"][0], color="0.2", lw=1.0)
    ax.axvline(fit_lower, color="0.2", lw=1.0, ls=":")


def plot_chains(trims, rom, max_stalled_fraction, out):
    winds = sorted(trims[trims.chain.str.contains("steering")].wind_speed.unique())
    fig, axes = plt.subplots(1, len(winds), figsize=(9, 3.2), sharex=True, sharey=True, squeeze=False)
    us_max = rom["validity"]["input_steering_abs_max"]
    band = rom["validity"]["input_depower"]
    for ax, wind in zip(axes[0], winds):
        rows = trims[trims.wind_speed == wind]
        attached = rows.stalled_fraction <= max_stalled_fraction
        ax.plot(rows.u_p[attached], rows.u_s.abs()[attached], "o", ms=3.5, color=PALETTE["Blue"],
                label="solved")
        ax.plot(rows.u_p[~attached], rows.u_s.abs()[~attached], "o", ms=3.5, mfc="none",
                color=PALETTE["Vermillion"], label="past the fit's stall cut")
        ax.plot([band[0], band[1], band[1], band[0], band[0]], [0, 0, us_max, us_max, 0],
                color="0.3", lw=1.0, label="ROM limits")
        ax.set_title(wind_label(wind), fontsize=10)
        ax.set_xlabel(r"$u_p$ (m)")
    axes[0][0].set_ylabel(r"$|u_s|$")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out / "chains.pdf")
    plt.close(fig)


def plot_trim_vs_depower(trims, rom, fit_lower, out):
    rows = trims[trims.chain.str.contains("depower")]
    styles = wind_styles(trims)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharex=True)
    for wind, group in rows.groupby("wind_speed"):
        colour, marker = styles[wind]
        group = group.sort_values("u_p")
        axes[0].plot(group.u_p, group.tension / 1e3, "-", marker=marker, ms=4, color=colour,
                     label=wind_label(wind))
        axes[1].plot(group.u_p, np.degrees(group.alpha_w), "-", marker=marker, ms=4, color=colour)
    axes[0].set_ylabel(r"$F_t$ (kN)")
    axes[1].set_ylabel(r"$\alpha_w$ ($^\circ$)")
    for ax in axes:
        ax.set_xlabel(r"$u_p$ (m)")
        mark_limits(ax, rom, fit_lower)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "trim_vs_depower.pdf")
    plt.close(fig)


def plot_gust_margin(polar, trims, rom, fit_lower, out):
    lower = rom["validity"]["input_depower"][0]
    margins = stall_onset_margin(polar, trims)
    margins = margins[margins.u_p.between(lower - GUST_MARGIN_TAPE_SPAN[0],
                                          lower + GUST_MARGIN_TAPE_SPAN[1])]
    styles = wind_styles(trims)
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    for wind, group in margins.groupby("wind_speed"):
        colour, marker = styles[wind]
        ax.plot(group.u_p, group.margin, "-", marker=marker, ms=4, color=colour, label=wind_label(wind))
    ax.axhline(0.0, color="0.5", lw=0.8)
    mark_limits(ax, rom, fit_lower)
    ax.set_xlabel(r"$u_p$ (m)")
    ax.set_ylabel(r"margin to stall onset ($^\circ$)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "gust_margin.pdf")
    plt.close(fig)


def plot_polars(polar, trims, rom, out):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharex=True)
    reference = trims[(trims.wind_speed == reference_wind(trims)) & (trims.u_s == 0.0)]
    band = rom["validity"]["input_depower"]
    for tape, colour in zip(np.linspace(*band, N_POLAR_TAPES), blues(N_POLAR_TAPES)):
        trim = reference.iloc[(reference.u_p - tape).abs().argmin()]
        samples = polar[(polar.anchor == trim.anchor) & polar.converged].sort_values("alpha")
        alpha = np.linspace(samples.alpha.min(), samples.alpha.max(), 200)
        for ax, key in zip(axes, ("CL", "CD")):
            ax.plot(np.degrees(samples.alpha), samples[key], "o", ms=3, color=colour,
                    label=rf"$u_p={trim.u_p:.2f}$ m" if key == "CL" else None)
            ax.plot(np.degrees(alpha), rom_coefficient(rom, key, alpha, trim.u_p, 0.0, trim.v_a),
                    "-", lw=1.5, color=colour)
    axes[0].set_ylabel(r"$C_L$")
    axes[1].set_ylabel(r"$C_D$")
    for ax in axes:
        ax.set_xlabel(r"$\alpha_w$ ($^\circ$)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "polars.pdf")
    plt.close(fig)


def plot_steering(trims, out):
    rows = trims[(trims.wind_speed == reference_wind(trims)) & trims.chain.str.contains("steering")]
    tapes = sorted(rows.u_p.round(3).unique())
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharex=True)
    for tape, colour in zip(tapes, blues(len(tapes))):
        group = rows[np.isclose(rows.u_p, tape)].assign(us=lambda d: d.u_s.abs()).sort_values("us")
        axes[0].plot(group.us, np.abs(group.chi_dot), "-o", ms=3, color=colour,
                     label=rf"$u_p={tape:.2f}$ m")
        axes[1].plot(group.us, group.tension / 1e3, "-o", ms=3, color=colour)
    axes[0].set_ylabel(r"$|\dot\chi|$ (rad s$^{-1}$)")
    axes[1].set_ylabel(r"$F_t$ (kN)")
    for ax in axes:
        ax.set_xlabel(r"$|u_s|$")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "steering.pdf")
    plt.close(fig)


def plot_bridle_and_roll(trims, rom, band, out):
    params = rom["aerodynamics"]["params"]
    tape = np.linspace(*band, 100)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharex=True)
    steered = trims[trims.u_s.abs() > 0.01]
    theta = params["angle_pitch_tether_0"] + params["slope_angle_pitch_tether_depower"] * tape
    def gain(u_s):
        return roll_gain(params["gain_roll_steering"], params.get("gain_roll_steering_depower", []),
                         params.get("gain_roll_steering_squared", []), tape, u_s)

    us_max = rom["validity"]["input_steering_abs_max"]
    styles = wind_styles(trims)
    for wind, group in trims[trims.u_s == 0.0].groupby("wind_speed"):
        colour, marker = styles[wind]
        axes[0].plot(group.u_p, np.degrees(group.theta_b), marker, ms=3, color=colour,
                     label=wind_label(wind))
    points = axes[1].scatter(steered.u_p, steered.phi_a_w / steered.u_s, s=9, c=steered.u_s.abs(),
                             cmap=ListedColormap(blues(256)), vmin=0.0, edgecolors="none")
    fig.colorbar(points, ax=axes[1], label=r"$|u_s|$")
    axes[0].plot(tape, np.degrees(theta), "k-", lw=1.5, label="ROM")
    axes[1].plot(tape, gain(0.0), "k-", lw=1.5, label=r"ROM, $u_s=0$")
    axes[1].plot(tape, gain(us_max), "k--", lw=1.5, label=rf"ROM, $|u_s|={us_max:.2f}$")
    axes[1].legend(frameon=False, fontsize=8)
    axes[0].set_ylabel(r"$\theta_b$ ($^\circ$)")
    axes[1].set_ylabel(r"$k_{\phi,s}$")
    for ax in axes:
        ax.set_xlabel(r"$u_p$ (m)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "bridle_and_roll.pdf")
    plt.close(fig)


def plot_rom_check(polar, band, rom, out):
    rows = polar[(polar.source == "coupled_trim") & polar.u_p.between(*band)]
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.4))
    for ax, key, label in zip(axes, ("CL", "CD"), ("C_L", "C_D")):
        model = rom_coefficient(rom, key, rows.alpha.to_numpy(), rows.u_p.to_numpy(),
                                rows.u_s.to_numpy(), rows.v_a.to_numpy())
        data = rows[key].to_numpy()
        limits = [min(data.min(), model.min()), max(data.max(), model.max())]
        ax.plot(limits, limits, color="0.6", lw=0.8)
        ax.plot(data, model, "o", ms=3, color=PALETTE["Blue"])
        ax.set_xlabel(rf"${label}$, simulation")
        ax.set_ylabel(rf"${label}$, ROM")
        ax.set_aspect("equal")
        ax.text(0.05, 0.92, f"RMS {np.sqrt(np.mean((model - data) ** 2)):.4f}",
                transform=ax.transAxes, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "rom_check.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--kite", default="LEI-V3-KITE", help="folder under data/")
    parser.add_argument("--fidelity", choices=ds.FIDELITIES, default=None,
                        help="default: the fidelity of identification.yaml")
    parser.add_argument("--results", type=Path, default=None,
                        help="default: results/<kite>/identification/<fidelity>")
    parser.add_argument("--out", type=Path, default=None,
                        help="default: <results>/figures")
    args = parser.parse_args()

    fidelity, fit_spec = ds.identification_settings(
        PROJECT_DIR / "data" / args.kite / "identification.yaml", "fit", args.fidelity
    )
    results = args.results or PROJECT_DIR / "results" / args.kite / "identification" / fidelity
    fit_dir = results / "rom_aerostructural"
    out = args.out or results / "figures"
    out.mkdir(parents=True, exist_ok=True)

    polar = pd.read_csv(fit_dir / "polar_samples.csv")
    trims = pd.read_csv(fit_dir / "trim_samples.csv")
    raw, _ = ds.load_sweep(results / "sweep")
    trims = trims.merge(raw[["anchor", "stalled_fraction"]], on="anchor")
    rom = yaml.safe_load((fit_dir / "rom_config_aerostructural.yaml").read_text(encoding="utf-8"))
    band = rom["validity"]["input_depower"]
    fit_lower = fit_spec["depower_band"][0]

    set_plot_style()
    plot_chains(trims, rom, fit_spec["max_stalled_fraction"], out)
    plot_trim_vs_depower(trims, rom, fit_lower, out)
    plot_gust_margin(polar, trims, rom, fit_lower, out)
    plot_polars(polar, trims, rom, out)
    plot_steering(trims, out)
    plot_bridle_and_roll(trims, rom, band, out)
    plot_rom_check(polar, band, rom, out)
    print(f"figures in {out}")


if __name__ == "__main__":
    main()
