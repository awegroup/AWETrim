"""Interactive full-cycle path/depower parameter explorer.

This is a lightweight visual tuner for the synthetic full-cycle parameters in
``fit_periodic_cycle_config.py``. It does not forward-simulate the trim problem
or write YAML; it only shows how the geometry and depower warm start change.

Usage:
    python scripts/personal/reduced-order-model/optimization/cycle/interactive_cycle_parameters.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider

_here = Path(__file__).resolve()
_repo_root = next(p for p in _here.parents if (p / "src" / "awetrim").exists())
_src_dir = _repo_root / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from awetrim.identification.controls import (
    ROM_DEPOWERED_INPUT_DEPOWER,
    ROM_POWERED_INPUT_DEPOWER,
)
from awetrim.kinematics.parametrized_patterns import (
    PeriodicBSpline,
    full_cycle_angles,
    make_full_cycle_bspline_path_parameters,
    reelin_bump,
)

from fit_periodic_cycle_config import (
    ARTIFICIAL,
    DEPOWER_DEPTH,
    R0,
    curvature_from_angles,
    path_curvature_metrics,
)

SLIDER_SPECS = {
    "M": (12, 90, 1, int),
    "n_loops": (1, 8, 1, int),
    "reelout_fraction": (0.45, 0.85, 0.005, float),
    "beta0": (0.10, 0.80, 0.005, float),
    "beta_amp0": (0.00, 0.35, 0.005, float),
    "az_amp0": (0.05, 0.90, 0.005, float),
    "beta_reelin_peak": (0.50, 1.60, 0.005, float),
    "az_reelin_amp": (-1.00, 1.00, 0.005, float),
    "ramp_fraction": (0.02, 0.50, 0.005, float),
    "reelin_center": (0.00, 1.00, 0.005, float),
    "psi0": (0.00, 2.0 * np.pi, 0.005, float),
    # Handover phases (rad), active while "pin handover phases" is checked:
    # psi_entry = figure phase where the reel-in fade begins (0 = climbing
    # centre crossing), psi_exit = phase where the figures resume (pi = the
    # other crossing -> first lobe on the other side). psi0 is inert then.
    # With the "lobe" bow they are the FROZEN phases instead: psi_entry = the
    # figure point the climb peels off from (2*pi - 0.3 = a bit before the
    # centre, heading up), psi_exit = the point the descent lands on (3*pi/2
    # = left-lobe extreme heading down; pair it with a NEGATIVE az_reelin_amp).
    "psi_entry": (0.00, 2.0 * np.pi, 0.005, float),
    "psi_exit": (0.00, 2.0 * np.pi, 0.005, float),
    "depower_depth": (0.00, 1.00, 0.005, float),
}


def _clean_values(sliders):
    values = {}
    for key, slider in sliders.items():
        _, _, _, caster = SLIDER_SPECS[key]
        value = slider.val
        values[key] = caster(round(value)) if caster is int else float(value)
    return values


def _depower_profile(s, reelout_fraction, ramp_fraction, depth, reelin_center=0.5):
    bump = reelin_bump(
        s,
        reelout_fraction=reelout_fraction,
        ramp_fraction=ramp_fraction,
        reelin_center=reelin_center,
    )
    return ROM_POWERED_INPUT_DEPOWER + depth * bump * (
        ROM_DEPOWERED_INPUT_DEPOWER - ROM_POWERED_INPUT_DEPOWER
    )


def _shape_kwargs(values):
    """full_cycle_angles/spline knobs from slider values + toggle state."""
    pinned = values.get("pin_phases", False)
    return dict(
        n_loops=values["n_loops"],
        reelout_fraction=values["reelout_fraction"],
        beta0=values["beta0"],
        beta_amp0=values["beta_amp0"],
        az_amp0=values["az_amp0"],
        beta_reelin_peak=values["beta_reelin_peak"],
        az_reelin_amp=values["az_reelin_amp"],
        ramp_fraction=values["ramp_fraction"],
        reelin_center=values["reelin_center"],
        psi0=values["psi0"],
        psi_entry=values["psi_entry"] if pinned else None,
        psi_exit=values["psi_exit"] if pinned else None,
        bow_shape=values.get("bow_shape", "sym"),
        downloops=True,
    )


def _periodic_spline(values, s):
    path = make_full_cycle_bspline_path_parameters(
        M=values["M"],
        r0=R0,
        **_shape_kwargs(values),
    )
    pattern = PeriodicBSpline(
        M=path["M"],
        C_phi=np.asarray(path["C_phi"], dtype=float).reshape((path["M"], 1)),
        C_beta=np.asarray(path["C_beta"], dtype=float).reshape((path["M"], 1)),
        s_init=0.0,
        s_final=1.0,
        downloops=True,
    )
    phi = np.asarray([float(pattern.azimuth(R0, si)) for si in s])
    beta = np.asarray([float(pattern.elevation(R0, si)) for si in s])
    return path, phi, beta


def _print_values(values):
    pinned = values.get("pin_phases", False)
    print("\nARTIFICIAL = {")
    for key in (
        "M",
        "n_loops",
        "reelout_fraction",
        "beta0",
        "beta_amp0",
        "az_amp0",
        "beta_reelin_peak",
        "az_reelin_amp",
        "ramp_fraction",
        "reelin_center",
        "psi0",
    ):
        print(f'    "{key}": {values[key]!r},')
    print(f'    "psi_entry": {values["psi_entry"] if pinned else None!r},')
    print(f'    "psi_exit": {values["psi_exit"] if pinned else None!r},')
    print(f'    "bow_shape": {values.get("bow_shape", "sym")!r},')
    print("}")
    print(f"DEPOWER_DEPTH = {values['depower_depth']:.6g}\n")


def main(show_spline=True):
    s = np.linspace(0.0, 1.0, 700, endpoint=True)
    defaults = dict(ARTIFICIAL)
    defaults["depower_depth"] = float(DEPOWER_DEPTH)
    defaults.setdefault("reelin_center", 0.5)
    defaults.setdefault("psi0", 0.0)
    # Sliders need numbers even while unpinned; the toggles carry the state.
    pin_default = (
        defaults.get("psi_entry") is not None
        and defaults.get("psi_exit") is not None
    )
    bow_default = defaults.get("bow_shape", "sym")
    if defaults.get("psi_entry") is None:
        defaults["psi_entry"] = 0.0
    if defaults.get("psi_exit") is None:
        defaults["psi_exit"] = float(np.pi)

    fig = plt.figure(figsize=(13.5, 8.0))
    grid = fig.add_gridspec(
        2,
        2,
        left=0.07,
        right=0.97,
        top=0.93,
        bottom=0.36,
        width_ratios=(1.15, 1.0),
        hspace=0.28,
        wspace=0.25,
    )
    ax_path = fig.add_subplot(grid[:, 0])
    ax_angles = fig.add_subplot(grid[0, 1])
    ax_depower = fig.add_subplot(grid[1, 1])

    (target_line,) = ax_path.plot([], [], color="0.55", lw=1.0, label="target")
    (spline_line,) = ax_path.plot([], [], color="tab:blue", lw=1.7, label="B-spline")
    (control_line,) = ax_path.plot([], [], "--o", color="0.35", ms=3, lw=0.8)
    start_point = ax_path.scatter([], [], c="red", s=45, zorder=5, label="s = 0")

    (phi_line,) = ax_angles.plot([], [], color="tab:blue", lw=1.3, label="azimuth")
    (beta_line,) = ax_angles.plot([], [], color="tab:orange", lw=1.3, label="elevation")
    (depower_line,) = ax_depower.plot([], [], color="tab:green", lw=1.5)
    (bump_line,) = ax_depower.plot([], [], color="0.55", lw=1.0, ls="--")

    ax_path.set_xlabel("azimuth phi (deg)")
    ax_path.set_ylabel("elevation beta (deg)")
    ax_path.grid(True)
    ax_path.legend(loc="best")
    ax_path.set_title("Synthetic full-cycle path")

    ax_angles.set_xlabel("s")
    ax_angles.set_ylabel("angle (deg)")
    ax_angles.grid(True)
    ax_angles.legend(loc="best")

    ax_depower.set_xlabel("s")
    ax_depower.set_ylabel("input_depower")
    ax_depower.grid(True)

    slider_axes = {}
    sliders = {}
    y0 = 0.30
    dy = 0.020
    for i, (key, (vmin, vmax, step, _)) in enumerate(SLIDER_SPECS.items()):
        ax = fig.add_axes([0.18, y0 - i * dy, 0.58, 0.014])
        slider_axes[key] = ax
        sliders[key] = Slider(
            ax=ax,
            label=key,
            valmin=vmin,
            valmax=vmax,
            valinit=defaults[key],
            valstep=step,
        )

    button_reset_ax = fig.add_axes([0.80, 0.14, 0.08, 0.035])
    button_print_ax = fig.add_axes([0.89, 0.14, 0.08, 0.035])
    reset_button = Button(button_reset_ax, "Reset")
    print_button = Button(button_print_ax, "Print")

    check_ax = fig.add_axes([0.80, 0.265, 0.17, 0.07])
    spline_check = CheckButtons(
        check_ax,
        ["fit B-spline", "pin handover phases"],
        [show_spline, pin_default],
    )
    # Reel-in bow shape (see full_cycle_angles): "lobe" = the optimiser-like
    # giant lobe (frozen figure phase, climb on the meridian, land on the
    # lobe extreme); it needs the handover phases pinned.
    radio_ax = fig.add_axes([0.80, 0.19, 0.17, 0.07])
    radio_ax.set_title("reel-in bow", fontsize=9, loc="left", pad=2)
    bow_radio = RadioButtons(
        radio_ax, ["sym", "descent", "lobe"], active=["sym", "descent", "lobe"].index(bow_default)
    )

    status = fig.text(0.80, 0.115, "", fontsize=9, color="tab:red")
    curvature_status = fig.text(0.80, 0.085, "", fontsize=9, color="0.25")
    toggles = {
        "fit B-spline": bool(show_spline),
        "pin handover phases": bool(pin_default),
    }
    bow_state = {"bow_shape": bow_default}

    def redraw(_=None):
        values = _clean_values(sliders)
        values["pin_phases"] = toggles["pin handover phases"]
        values["bow_shape"] = bow_state["bow_shape"]
        try:
            phi_target, beta_target = full_cycle_angles(s, **_shape_kwargs(values))
        except ValueError as exc:  # e.g. pinned phases with ramp >= 0.5
            status.set_text(str(exc))
            fig.canvas.draw_idle()
            return
        depower = _depower_profile(
            s,
            values["reelout_fraction"],
            values["ramp_fraction"],
            values["depower_depth"],
            reelin_center=values["reelin_center"],
        )
        bump = reelin_bump(
            s,
            reelout_fraction=values["reelout_fraction"],
            ramp_fraction=values["ramp_fraction"],
            reelin_center=values["reelin_center"],
        )

        target_line.set_data(np.degrees(phi_target), np.degrees(beta_target))
        phi_for_profiles = phi_target
        beta_for_profiles = beta_target
        curvature = curvature_from_angles(phi_target, beta_target, R0)

        if toggles["fit B-spline"]:
            try:
                path, phi_spline, beta_spline = _periodic_spline(values, s)
                c_phi = np.degrees(np.asarray(path["C_phi"], dtype=float))
                c_beta = np.degrees(np.asarray(path["C_beta"], dtype=float))
                spline_line.set_data(np.degrees(phi_spline), np.degrees(beta_spline))
                control_line.set_data(np.r_[c_phi, c_phi[0]], np.r_[c_beta, c_beta[0]])
                phi_for_profiles = phi_spline
                beta_for_profiles = beta_spline
                curvature = path_curvature_metrics(path, n_samples=len(s))
                status.set_text("")
            except Exception as exc:  # pragma: no cover - interactive guardrail
                spline_line.set_data([], [])
                control_line.set_data([], [])
                status.set_text(f"Spline fit failed: {exc}")
        else:
            spline_line.set_data([], [])
            control_line.set_data([], [])
            status.set_text("")

        start_point.set_offsets(
            [[np.degrees(phi_for_profiles[0]), np.degrees(beta_for_profiles[0])]]
        )
        phi_line.set_data(s, np.degrees(phi_for_profiles))
        beta_line.set_data(s, np.degrees(beta_for_profiles))
        depower_line.set_data(s, depower)
        bump_line.set_data(s, ROM_POWERED_INPUT_DEPOWER + bump * 0.4)
        curvature_status.set_text(
            "max curvature: {max_physical:.6g} 1/m at s={s_at_max:.3f} "
            "({max_unit:.3g} unit-sphere)".format(**curvature)
        )

        for ax in (ax_path, ax_angles, ax_depower):
            ax.relim()
            ax.autoscale_view()
        ax_depower.set_ylim(
            ROM_POWERED_INPUT_DEPOWER - 0.03, ROM_DEPOWERED_INPUT_DEPOWER + 0.03
        )
        fig.canvas.draw_idle()

    def reset(_event):
        for slider in sliders.values():
            slider.reset()

    def print_current(_event):
        values = _clean_values(sliders)
        values["pin_phases"] = toggles["pin handover phases"]
        values["bow_shape"] = bow_state["bow_shape"]
        _print_values(values)

    def toggle_spline(label):
        toggles[label] = not toggles[label]
        redraw()

    def select_bow(label):
        bow_state["bow_shape"] = label
        redraw()

    for slider in sliders.values():
        slider.on_changed(redraw)
    reset_button.on_clicked(reset)
    print_button.on_clicked(print_current)
    spline_check.on_clicked(toggle_spline)
    bow_radio.on_clicked(select_bow)

    redraw()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-only",
        action="store_true",
        help="Start with the analytical target only; skip fitting the B-spline.",
    )
    args = parser.parse_args()
    main(show_spline=not args.target_only)



