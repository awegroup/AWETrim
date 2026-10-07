"""Tables, figures and numbers of the aerostructural-ROM identification note.

Writes everything ``docs/identification/rom_aerostructural.tex`` includes, so
the document follows the identification instead of restating it:

    docs/identification/generated/macros.tex          key numbers as \\newcommand
    docs/identification/generated/coefficients.tex    C_L / C_D terms +- 95 % CI
    docs/identification/generated/selection.tex       forward/backward path
    docs/identification/generated/fit_metrics.tex     RMSE, R^2, CV per alpha band
    docs/identification/generated/coupled_check.tex   ROM vs coupled trims
    docs/identification/generated/flight.tex          flight validation, both modes
    docs/identification/generated/flight_correction_*.tex  the flight correction
    docs/identification/figures/*.pdf                 the figures, copied

Uncertainty
    Polar samples are strongly correlated within one frozen shape (anchor), so
    ordinary least-squares standard errors are far too optimistic. The C_L/C_D
    coefficient intervals are an ANCHOR-CLUSTER BOOTSTRAP: anchors are drawn
    with replacement, the SELECTED terms are refitted on their samples, and
    the 2.5/97.5 % percentiles of each coefficient are reported. The intervals
    are conditional on the selected term set (selection is not re-run per
    resample). theta_b and the roll gain have one sample per coupled trim, so
    their standard errors are the ordinary least-squares ones.

Run after identify_rom_aerostructural.py, validate_rom_aerostructural.py and
the flight validations (plot_rom_flight_validation.py for both modes).

Usage (project root)
    python scripts/identification/report_rom_aerostructural.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from awetrim.identification import aero_polynomial as ap
from awetrim.identification.controls import FLIGHT_STEERING_ZERO_OFFSET_2019 as STEERING_ZERO_OFFSET
from awetrim.utils.config_paths import LEI_V3_ROM_AEROSTRUCTURAL_CONFIG, REPO_ROOT

import identify_rom_aerostructural as ident

DOC = REPO_ROOT / "docs" / "identification"
GEN = DOC / "generated"
FIG = DOC / "figures"
FLIGHT = REPO_ROOT / "results" / "LEI-V3-KITE" / "rom_validation"
CORRECTION = REPO_ROOT / "results" / "LEI-V3-KITE" / "identification" / "rom_flight_correction"
N_BOOT = 500
SEED = 0

#: Figures copied into the document folder (source -> name).
FIGURES = {
    ident.OUT_DIR / "rom_comparison_polars.pdf": "polars.pdf",
    ident.OUT_DIR / "rom_comparison_theta_b.pdf": "theta_b.pdf",
    FLIGHT / "rom_flight_validation_cycle64.pdf": "flight_solved.pdf",
    FLIGHT / "rom_flight_validation_cycle61.pdf": "flight_solved_cycle61.pdf",
    CORRECTION / "alpha_lift" / "flight_correction.pdf": "flight_correction.pdf",
    FLIGHT / "rom_flight_validation_cycle64_measured_steering.pdf": "flight_measured.pdf",
    FLIGHT / "semi_empirical" / "vdot_sweep.pdf": "vdot_sweep.pdf",
    FLIGHT / "semi_empirical" / "vdot_sweep_williams.pdf": "vdot_sweep_williams.pdf",
    REPO_ROOT / "results" / "LEI-V3-KITE" / "optimization" / "downloops" / "rom_comparison_loops.pdf":
        "optimisation_loops.pdf",
}


def tex_number(x, digits=4):
    """Number for a LaTeX table: fixed or scientific, minus sign as math."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    text = f"{x:.{digits}g}"
    if "e" in text:
        mant, exp = text.split("e")
        text = rf"{mant}\times10^{{{int(exp)}}}"
    return f"${text}$"


def term_tex(power_map) -> str:
    names = {"alpha": r"\alpha", "u_p": "u_p", "u_s": "u_s", "v_a": "v_a",
             "stall": r"\sigma"}
    parts = []
    for var, p in sorted(power_map.items(), key=lambda kv: list(names).index(kv[0])):
        parts.append(names[var] + (f"^{{{p}}}" if p != 1 else ""))
    return "$" + r"\,".join(parts) + "$" if parts else "$1$"


def polar_data():
    polar = pd.read_csv(ident.OUT_DIR / "polar_samples.csv")
    polar = polar[polar.converged & polar.lens_consistent & polar.u_p.between(*ident.UP_BAND)
                  & (polar.alpha <= np.deg2rad(ident.ALPHA_MAX_FIT_DEG))]
    return polar.reset_index(drop=True)


def bootstrap(polar, stall, fits):
    """Anchor-cluster bootstrap of the selected C_L / C_D terms."""
    rng = np.random.default_rng(SEED)
    data = ident.regressors(polar, stall)
    anchors = polar.anchor.to_numpy()
    unique = np.unique(anchors)
    index = {a: np.flatnonzero(anchors == a) for a in unique}
    out = {}
    for target, fit in fits.items():
        terms = [pm for pm, _ in fit.terms]
        y = polar[target].to_numpy()
        draws = []
        for _ in range(N_BOOT):
            rows = np.concatenate([index[a] for a in rng.choice(unique, len(unique))])
            sample = {k: v[rows] for k, v in data.items()}
            refit = ap.fit_terms(sample, y[rows], terms, target=target,
                                 regressors=tuple(data))
            draws.append([refit.intercept] + [c for _, c in refit.terms])
        out[target] = np.asarray(draws)
    return out


def write_coefficients(fits, boot):
    lines = [r"\begin{tabular}{llrrr}", r"\toprule",
             r"Coefficient & Term & Estimate & \multicolumn{2}{c}{95\,\% interval} \\",
             r"\midrule"]
    for target, fit in fits.items():
        draws = boot[target]
        entries = [({}, fit.intercept)] + list(fit.terms)
        for k, (pm, coef) in enumerate(entries):
            lo, hi = np.percentile(draws[:, k], [2.5, 97.5])
            name = target.replace("CL", "$C_L$").replace("CD", "$C_D$") if k == 0 else ""
            lines.append(f"{name} & {term_tex(pm)} & {tex_number(coef)} & "
                         f"{tex_number(lo)} & {tex_number(hi)} \\\\")
        lines.append(r"\midrule" if target == "CL" else r"\bottomrule")
    lines.append(r"\end{tabular}")
    (GEN / "coefficients.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_selection(report):
    lines = [r"\begin{tabular}{rlr@{\hspace{2.5em}}rlr}", r"\toprule",
             r"\multicolumn{3}{c}{$C_L$} & \multicolumn{3}{c}{$C_D$} \\",
             r"Step & Term & CV RMSE & Step & Term & CV RMSE \\", r"\midrule"]
    paths = [report["selection"][t]["path"] for t in ("CL", "CD")]
    for k in range(max(len(p) for p in paths)):
        cells = []
        for path in paths:
            if k < len(path):
                label, score = path[k]
                term = "intercept" if label == "1" else label
                term = term.replace("alpha", r"\alpha").replace("stall", r"\sigma") \
                    .replace("*", r"\,").replace("^", "^")
                term = term if label == "1" else f"${term}$"
                cells += [str(k), term, tex_number(score, 3)]
            else:
                cells += ["", "", ""]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "selection.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_fit_metrics(polar, stall, fits):
    data = ident.regressors(polar, stall)
    alpha_deg = np.rad2deg(polar.alpha.to_numpy())
    bands = (("all", -90, 90), (r"$\alpha<13^\circ$ (operating)", -90, 13),
             (r"$13^\circ\le\alpha<19^\circ$ (stall)", 13, 19),
             (r"$\alpha\ge19^\circ$ (post-stall)", 19, 90))
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Samples & $n$ & RMSE $C_L$ & RMSE $C_D$ & $R^2$ ($C_L$ / $C_D$) \\",
             r"\midrule"]
    for label, lo, hi in bands:
        m = (alpha_deg >= lo) & (alpha_deg < hi)
        cells = [label, str(int(m.sum()))]
        r2 = []
        for target in ("CL", "CD"):
            y = polar[target].to_numpy()[m]
            res = y - fits[target].predict({k: v[m] for k, v in data.items()})
            cells.append(tex_number(float(np.sqrt(np.mean(res**2))), 3))
            r2.append(1 - np.sum(res**2) / np.sum((y - y.mean())**2))
        cells.append(f"{r2[0]:.4f} / {r2[1]:.4f}")
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\midrule",
              f"\\multicolumn{{2}}{{l}}{{Grouped 5-fold CV RMSE}} & "
              f"{tex_number(fits['CL'].cv_rmse, 3)} & {tex_number(fits['CD'].cv_rmse, 3)} & \\\\",
              r"\bottomrule", r"\end{tabular}"]
    (GEN / "fit_metrics.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def theta_and_roll():
    trims = pd.read_csv(ident.OUT_DIR / "trim_samples.csv")
    rows = trims[(trims.u_s == 0.0) & trims.u_p.between(*ident.UP_BAND)]
    (slope, inter), cov = np.polyfit(rows.u_p, rows.theta_b, 1, cov=True)
    resid = rows.theta_b - (inter + slope * rows.u_p)
    steer = trims[trims.chain.str.contains("us_continuation")]
    k = float((steer.u_s @ steer.phi_a_w) / (steer.u_s @ steer.u_s))
    r = steer.phi_a_w - k * steer.u_s
    se_k = float(np.sqrt((r @ r) / (len(steer) - 1) / (steer.u_s @ steer.u_s)))
    return {
        "thetaZero": inter, "thetaZeroSE": float(np.sqrt(cov[1, 1])),
        "thetaSlope": slope, "thetaSlopeSE": float(np.sqrt(cov[0, 0])),
        "thetaRMSdeg": float(np.rad2deg(resid.std())), "thetaN": len(rows),
        "thetaPoweredDeg": float(np.rad2deg(inter + slope * 1.7)),
        "thetaDepoweredDeg": float(np.rad2deg(inter + slope * 2.1)),
        "rollGain": k, "rollGainSE": se_k, "rollRMSdeg": float(np.rad2deg(r.std())),
        "rollN": len(steer),
    }


def write_coupled_check():
    t = pd.read_csv(ident.OUT_DIR / "rom_vs_as_trims.csv")
    t = t[t.rom_ok & t.u_p.between(*ident.UP_BAND)].copy()
    t["dv"] = 100 * (t.rom_v_tau / t.as_v_tau - 1)
    t["dt"] = 100 * (t.rom_tension / t.as_tension - 1)
    t["da"] = np.rad2deg(t.rom_alpha_w - t.as_alpha_w)
    t["kind"] = np.where(t.u_s == 0, "unsteered", "steered")
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
             r"ROM & Trims & $n$ & $v_\tau$ error (\%) & $F_t$ error (\%) & $\alpha_w$ error (\si{\degree}) \\",
             r"\midrule"]
    names = {"aerostructural": "Aerostructural", "semi_empirical": "Semi-empirical"}
    for rom in ("aerostructural", "semi_empirical"):
        for kind in ("unsteered", "steered"):
            g = t[(t.rom == rom) & (t.kind == kind)]
            lines.append(
                f"{names[rom] if kind == 'unsteered' else ''} & {kind} & {len(g)} & "
                f"${g.dv.mean():+.1f}\\pm{g.dv.std():.1f}$ & "
                f"${g.dt.mean():+.1f}\\pm{g.dt.std():.1f}$ & "
                f"${g.da.mean():+.2f}\\pm{g.da.std():.2f}$ \\\\")
        lines.append(r"\midrule" if rom == "aerostructural" else r"\bottomrule")
    lines.append(r"\end{tabular}")
    (GEN / "coupled_check.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_flight():
    lines = [r"\begin{tabular}{llllrrrrrrr}", r"\toprule",
             r"Tether & Mode & ROM & Phase & Solved (\%) & Pre-stall (\%) & Speed-stable (\%) & $F_t$ error (\%) & $v_\tau$ error (\%) & $u_s$ slope (corr) & $|\Delta\dot\chi|$ (\si{\radian\per\second}) \\",
             r"\midrule"]
    runs = [(tether, t_suffix + m_suffix, label)
            for tether, t_suffix in (("rigid", ""), ("Williams", "_williams"))
            for m_suffix, label in (("", r"$\dot\chi$ in"),
                                    ("_measured_steering", r"$u_s$ in"))]
    found = False
    for tether, suffix, label in runs:
        path = FLIGHT / f"rom_flight_validation_errors{suffix}.csv"
        if not path.exists():
            continue
        found = True
        table = pd.read_csv(path)
        for k, row in table.iterrows():
            rom = row["rom"].replace("_", "-")
            chi = row.get("chi_dot_abs_err_median_rad_s", np.nan)
            slope = row.get("us_slope", np.nan)
            us_text = "--" if not np.isfinite(slope) else f"{slope:.2f} ({row['us_corr']:.2f})"
            lines.append(
                f"{tether if k == 0 else ''} & {label if k == 0 else ''} & {rom} & "
                f"{row['phase']} & {row['solved_%']:.0f} & "
                f"{row['pre_stall_%']:.1f} & {row['speed_stable_%']:.1f} & "
                f"{tex_number(row['tension_err_median_%'], 3)} "
                f"({row['tension_err_IQR_%']}) & "
                f"{tex_number(row['v_tau_err_median_%'], 3)} & {us_text} & {tex_number(chi, 2)} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    if found:
        (GEN / "flight.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_flight_correction() -> dict:
    """Parameter and training-error tables of the flight correction (steps 1-3).

    Step 1-2 values and their cycle-bootstrap intervals come from
    identify_rom_flight_correction.py, step 3 from
    refine_rom_flight_output_error.py: ``refinement_report.json`` (the run that
    wrote the ROM file: all six parameters jointly) and, when present,
    ``refinement_report_gain_fixed.json`` (the steps 1-2 roll gain held) and
    ``refinement_report_roll_gain.json`` (the gain against a steering error;
    rejected), each kept as a column. The drag
    terms are listed as corrections to the aerostructural C_D (which has a
    u_s^2 term but no plain u_p term). Returns macro values.
    """
    corr_path = CORRECTION / "alpha_lift" / "correction_report.json"
    final_path = CORRECTION / "output_error" / "refinement_report.json"


    if not (corr_path.exists() and final_path.exists()):
        print("  flight correction reports missing; tables skipped")
        return {}
    corr = json.loads(corr_path.read_text(encoding="utf-8"))
    final = json.loads(final_path.read_text(encoding="utf-8"))
    runs = [("step 3", final)]
    reelout_path = CORRECTION / "output_error" / "refinement_report_reelout_only.json"
    if reelout_path.exists():
        runs.append(("step 3, reel-out only", json.loads(reelout_path.read_text(encoding="utf-8"))))
    aero = yaml.safe_load(LEI_V3_ROM_AEROSTRUCTURAL_CONFIG.read_text(encoding="utf-8"))
    params = aero["aerodynamics"]["params"]
    base = {"theta0": params["angle_pitch_tether_0"],
            "theta1": params["slope_angle_pitch_tether_depower"],
            "cd0": params["CD0"], "cd_up": 0.0, "gain": params["gain_roll_steering"],
            "cd_us2": next(e["coef"] for e in aero["aerodynamics"]["coefficients"]["CD"]
                           if e.get("var") == "u_s" and int(e.get("power", 1)) == 2)}
    step12 = {**base, **runs[0][1]["start_values"]}
    boot = corr["bootstrap"]
    ci_keys = {"theta0": "angle_pitch_tether_0", "theta1": "slope_angle_pitch_tether_depower",
               "cd0": "dCD: 1", "cd_up": "dCD: u_p", "cd_us2": "dCD: u_s^2",
               "gain": "gain_roll_steering"}
    rows = (("theta0", r"$\theta_0$ (\si{\radian})", False),
            ("theta1", r"$\theta_1$ (\si{\radian\per\metre})", False),
            ("cd0", r"$\Delta C_{D,0}$", True),
            ("cd_up", r"$\Delta C_{D}$: $u_p$ (\si{\per\metre})", True),
            ("cd_us2", r"$\Delta C_{D}$: $u_s^2$", True),
            ("gain", r"$k_{\phi,s}$ (\si{\radian})", False))
    lines = [r"\begin{tabular}{lrr" + "r" * len(runs) + "}", r"\toprule",
             "Parameter & Aerostructural & Steps 1--2 (95\\,\\% CI) & "
             + " & ".join(name[0].upper() + name[1:] for name, _ in runs) + r" \\", r"\midrule"]
    for key, label, is_delta in rows:
        offset = base[key] if is_delta else 0.0
        a = 0.0 if is_delta else base[key]
        ci = boot.get(ci_keys.get(key, ""), {}).get("ci95")
        ci_text = "" if ci is None else f" [${ci[0]:+.3f}$, ${ci[1]:+.3f}$]"
        cells = [tex_number({**base, **run["refined_values"]}[key] - offset, 3)
                 for _, run in runs]
        lines.append(f"{label} & {tex_number(a, 3)} & {tex_number(step12[key] - offset, 3)}"
                     f"{ci_text} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "flight_correction_params.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Phase & Parameters & $F_t$ error (\%) & $v_\tau$ error (\%) & "
             r"$u_s$ rms & Solved (\%) \\", r"\midrule"]
    for ph in ("reel-out", "reel-in"):
        entries = [("steps 1--2", final["training_errors_before"][ph])]
        entries += [(name, run["training_errors_after"][ph]) for name, run in runs
                    if ph in run["training_errors_after"]]
        for k, (label, e) in enumerate(entries):
            lines.append(f"{ph if k == 0 else ''} & {label} & "
                         f"{tex_number(e['tension_median_%'], 3)} & "
                         f"{tex_number(e['v_tau_median_%'], 3)} & "
                         f"{tex_number(e.get('steering_rms'), 2)} & "
                         f"{100 * e.get('solved', e['n']) / e['n']:.0f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "flight_correction_training.tex").write_text("\n".join(lines) + "\n",
                                                         encoding="utf-8")
    return {"fcCycles": len(final["training_cycles"]), "fcSamples": final["samples"],
            "fcEvaluations": final["solver"]["evaluations"],
            "fcValidStartPct": 100 * final["solver"]["valid_start"],
            "fcValidPct": 100 * final["solver"]["valid"],
            "fcThetaCvDeg": corr["theta_b"]["cv_rmse_deg"],
            # Strings: formatted here, not by write_macros.
            "fcRollGain": f"{corr['roll_gain']['gain_roll_steering']:.3f}",
            "fcRollGainNoLag": f"{corr['roll_gain']['gain_without_lag']:.3f}",
            "fcRollRsq": f"{corr['roll_gain']['r2']:.2f}",
            "fcRollOffsetDeg": f"{corr['roll_gain']['offset_deg']:.2f}"}


def write_optimisations() -> None:
    """Reel-out downloop optimisation metrics per system and ROM (the metrics
    JSONs written by downloop_pattern.py --rom --system)."""
    root = REPO_ROOT / "results" / "LEI-V3-KITE" / "optimization" / "downloops"
    roms = (("semi_empirical", "semi-empirical"), ("aerostructural", "aerostructural"),
            ("aerostructural_flight", "aerostructural-flight"))
    systems = (("optimisation", r"optimisation hardware (KCU \SI{8.4}{\kilogram})"),
               ("flown_2019", r"flown 2019 hardware (KCU \SI{22}{\kilogram} + turbine)"))
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
             r"System & ROM & $\bar P$ (\si{\kilo\watt}) & $\bar F_t$ (\si{\kilo\newton}) & "
             r"$\bar v_\tau$ (\si{\metre\per\second}) & $T$ (\si{\second}) \\", r"\midrule"]
    found = False
    for sys_key, sys_label in systems:
        for k, (rom_key, rom_label) in enumerate(roms):
            files = list((root / f"{sys_key}_{rom_key}").glob("*_metrics.json"))
            if not files:
                continue
            found = True
            m = json.loads(files[0].read_text(encoding="utf-8"))
            lines.append(f"{sys_label if k == 0 else ''} & {rom_label} & "
                         f"{m['avg_power'] / 1e3:.2f} & {m['tension_mean'] / 1e3:.2f} & "
                         f"{m['vtau_mean']:.1f} & {m['total_time']:.1f} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    if found:
        (GEN / "optimisations.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_turn_law() -> dict:
    """Turn-rate law chi_dot = K v_a u_s + c_g (g cos(beta) sin(chi) / v_tau) on
    the 2019 reel-out (u_s with the zero offset and the 0.3 s lag), fitted to
    the flight and to each ROM's turn rate in the logged-steering mode: the
    check of the steering gain. Returns macro values."""
    import h5py
    ekf_path = REPO_ROOT / "results" / "LEI-V3-KITE" / "ekf" / "LEI-V3 Kite_2019-10-08.h5"
    if not ekf_path.exists():
        return {}
    with h5py.File(ekf_path, "r") as hf:
        ekf = pd.DataFrame({"time": np.asarray(hf["ekf_output"]["time"][()]),
                            "va": np.asarray(hf["ekf_output"]["kite_apparent_windspeed"][()]),
                            "course": np.asarray(hf["flight_data"]["kite_course"][()])})

    def regressors(path):
        d = pd.read_csv(path).merge(ekf, on="time")
        d = d[(d.speed_radial > 0.5) & (d.input_depower < 1.9) & (d.kite_elevation < 0.75)].copy()
        u = (d.measured_input_steering + STEERING_ZERO_OFFSET).shift(3)
        d["x_turn"] = d.va * u
        d["x_grav"] = 9.81 * np.cos(d.kite_elevation) * np.sin(d.course) / d.measured_speed_tangential
        return d.dropna(subset=["x_turn"])

    def fit(d, y):
        A = np.c_[d.x_turn, d.x_grav]
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
        return float(c[0]), float(c[1]), float(1 - np.var(y - A @ c) / np.var(y))

    roms = (("semi_empirical", "semi-empirical"), ("aerostructural", "aerostructural"),
            ("aerostructural_flight", "aerostructural-flight"))
    rows, macros = [], {}
    first = FLIGHT / roms[0][0] / "qs_validation_measured_steering.csv"
    if not first.exists():
        return {}
    d = regressors(first)
    k, cg, r2 = fit(d, d.measured_course_rate)
    rows.append(("flight", k, cg, r2))
    macros.update({"turnK": k, "turnG": cg})
    for key, label in roms:
        path = FLIGHT / key / "qs_validation_measured_steering.csv"
        if not path.exists():
            continue
        d = regressors(path)
        d = d[d.predicted_course_rate.notna()]
        k, cg, r2 = fit(d, d.predicted_course_rate)
        rows.append((label, k, cg, r2))
        macros[f"turnK{key.replace('_', '')}"] = k
    lines = [r"\begin{tabular}{lrrr}", r"\toprule",
             r"$\dot\chi$ of & $K$ (\si{\radian\per\metre}) & $c_g$ & $R^2$ \\", r"\midrule"]
    for label, k, cg, r2 in rows:
        lines.append(f"{label} & {k:.3f} & {cg:.2f} & {r2:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "turn_law.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return macros


def write_macros(values: dict):
    lines = ["% GENERATED by scripts/identification/report_rom_aerostructural.py"]
    for name, value in values.items():
        if not isinstance(value, float):
            text = str(value)
        elif name.endswith("SE"):
            text = f"{value:.2g}"  # uncertainties: two significant figures
        elif name.startswith("turn"):
            text = f"{value:.3f}" if name.startswith("turnK") else f"{value:.2f}"
        elif name.endswith("Pct"):
            text = f"{value:.0f}"
        elif "RMS" in name:
            text = f"{value:.2f}"
        elif name in ("thetaZero", "thetaSlope", "rollGain"):
            text = f"{value:.3f}"  # estimates to the precision their SE supports
        else:
            text = f"{value:.4g}"
        lines.append(f"\\newcommand{{\\{name}}}{{{text}}}")
    (GEN / "macros.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    GEN.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    aero = yaml.safe_load(LEI_V3_ROM_AEROSTRUCTURAL_CONFIG.read_text(encoding="utf-8"))
    params = aero["aerodynamics"]["params"]
    stall = {k: params[k] for k in ("angle_of_attack_stall", "width_stall")}
    fits = ap.fits_from_rom_aerodynamics(aero["aerodynamics"])
    report = json.loads((ident.OUT_DIR / "identification_report.json").read_text(encoding="utf-8"))
    for target in fits:
        fits[target].cv_rmse = report["selection"][target]["cv_rmse"]

    polar = polar_data()
    print(f"bootstrap: {N_BOOT} anchor resamples of {polar.anchor.nunique()} anchors")
    boot = bootstrap(polar, stall, fits)
    write_coefficients(fits, boot)
    write_selection(report)
    write_fit_metrics(polar, stall, fits)
    write_coupled_check()
    write_flight()
    correction = write_flight_correction()
    write_optimisations()
    turn_law = write_turn_law()
    numbers = theta_and_roll()
    trims = pd.read_csv(ident.OUT_DIR / "trim_samples.csv")
    write_macros({
        **{k: (float(v) if not isinstance(v, (int, np.integer)) else int(v))
           for k, v in numbers.items()},
        "nAnchorsAll": int(trims.anchor.nunique()),
        "nAnchorsFit": int(polar.anchor.nunique()),
        "nPolarSamples": int((polar.source == "polar").sum()),
        "nTrimSamples": int((polar.source == "coupled_trim").sum()),
        "stallDeg": float(np.rad2deg(stall["angle_of_attack_stall"])),
        "stallWidthDeg": float(np.rad2deg(stall["width_stall"])),
        "nBoot": N_BOOT, "refArea": float(aero["aerodynamics"]["reference_area"]),
        "upLow": ident.UP_BAND[0], "upHigh": ident.UP_BAND[1],
        "minImprovementPct": 100 * ident.MIN_RELATIVE_IMPROVEMENT,
        "lensTolPct": 100 * ident.LENS_CL_TOLERANCE,
        **correction,
        **turn_law,
    })
    for source, name in FIGURES.items():
        if source.exists():
            shutil.copyfile(source, FIG / name)
        else:
            print(f"  missing figure {source}")
    print(f"Wrote {GEN} and {FIG}")


if __name__ == "__main__":
    main()
