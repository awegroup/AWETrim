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

"""The aerostructural identification sweep: dataset format and derived samples.

A sweep (``examples/3_rom_identification/run_identification_sweep.py``) solves
coupled aerostructural states at the centre of the wind window along
continuation chains in depower and steering. Every converged state is an
ANCHOR, written as one row of ``trims.csv``; a frozen-shape angle-of-attack
polar about it (``aerodynamic_vsm.run_frozen_alpha_polar_about_trim``) gives
its rows of ``polars.csv``. Both files hold dimensional quantities only, so
the reference area and the control scaling are choices of the fit
(:func:`coefficient_tables`), not of the sweep.

Every vector is stored as three columns ``<name>_x/_y/_z`` in the trim (VSM)
frame: x chordwise toward the trailing edge, y to starboard, z up (along the
tether at the window centre).

``trims.csv``, one row per anchor:

    anchor, chain, step      identifiers ("<chain>/<step>"), step order
    wind_speed               wind at the reference height [m/s]
    power_tape_length        depower-tape rest length l_dp [m]
    steering_half_difference steering-tape half-difference [m] (+ shortens
                             the first steering tape, lengthens the second)
    rho                      air density [kg m-3]
    mass_kcu, mass_total     KCU and whole-kite mass of the trim [kg]
    va_*                     freestream apparent wind [m/s]
    omega_*                  rotational inflow rate of the trim [rad/s]
    force_aero_*             wing force, bridle-line drag included [N]
    force_kcu_*              KCU drag [N]
    force_inertial_*         d'Alembert force of the whole kite [N]
    body_z_*                 FRD body down axis (centre panel)
    tether_force             tether force at the kite [N]
    speed_tangential, roll_deg, pitch_deg, yaw_deg, course_rate
                             the trim states
    aoa_deg                  centre-chord angle of attack [deg]
    stalled_fraction         stalled share of the span
    converged                coupled solve met its gate
    success_physical         trim moments below tolerance
    residual_relative, n_iter, exit_reason
                             coupled-solve diagnostics
    polar_check              |F(d_alpha=0) - force_aero| / |force_aero| of
                             the anchor's polar (it re-solves the trim)

``polars.csv``, one row per polar sample:

    anchor                   the anchor it belongs to
    d_alpha_deg              angle-of-attack increment about the trim [deg]
    alpha_body_deg           FRD angle of attack of the freestream [deg]
    va_*                     freestream apparent wind [m/s]
    force_aero_*             wing force, bridle-line drag included [N]
    gamma_converged          circulation solve converged
    av_stage                 attached / stalled / latched / empty
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

TRIMS_FILE = "trims.csv"
POLARS_FILE = "polars.csv"
FIDELITIES = ("billow", "wireframe")

_AXES = ("x", "y", "z")


def _merged(base: Mapping, overrides: Mapping) -> dict:
    """``base`` with ``overrides`` applied, nested dicts merged key by key."""
    out = dict(base)
    for key, value in (overrides or {}).items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _merged(out[key], value)
        else:
            out[key] = value
    return out


def identification_settings(path, block: str, fidelity: str | None = None):
    """``(fidelity, settings)`` of one block (``sweep`` or ``fit``) of a kite's
    ``identification.yaml``: the block's common keys with its
    ``fidelities[<fidelity>]`` entry merged over them. ``fidelity`` defaults
    to the file's ``fidelity``."""
    import yaml

    spec = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    fidelity = fidelity or spec.get("fidelity", "billow")
    if fidelity not in FIDELITIES:
        raise ValueError(f"fidelity must be one of {FIDELITIES}, got {fidelity!r}")
    common = dict(spec[block])
    per = (common.pop("fidelities", None) or {}).get(fidelity) or {}
    return fidelity, _merged(common, per)


def vector_columns(name: str, vector) -> dict[str, float]:
    """``{name_x: .., name_y: .., name_z: ..}`` of a 3-vector."""
    v = np.asarray(vector, dtype=float).reshape(3)
    return {f"{name}_{axis}": float(v[i]) for i, axis in enumerate(_AXES)}


def vector(table, name: str) -> np.ndarray:
    """The ``(n, 3)`` array of a vector stored as ``name_x/_y/_z`` columns."""
    return np.column_stack([np.asarray(table[f"{name}_{a}"], dtype=float) for a in _AXES])


def trim_row(
    *,
    anchor: str,
    chain: str,
    step: int,
    wind_speed: float,
    power_tape_length: float,
    steering_half_difference: float,
    rho: float,
    mass_kcu: float,
    mass_total: float,
    trim: Mapping,
    meta: Mapping,
    body_z,
    polar_check: float,
) -> dict:
    """One ``trims.csv`` row from a coupled solve's final trim and ``meta``."""
    opt_x = np.asarray(trim["opt_x"], dtype=float).ravel()
    row = {
        "anchor": anchor, "chain": chain, "step": int(step),
        "wind_speed": float(wind_speed),
        "power_tape_length": float(power_tape_length),
        "steering_half_difference": float(steering_half_difference),
        "rho": float(rho), "mass_kcu": float(mass_kcu), "mass_total": float(mass_total),
        **vector_columns("va", trim["va_vel_world"]),
        **vector_columns("omega", trim.get("omega_c_vsm", np.zeros(3))),
        **vector_columns("force_aero", trim["total_aero_force_vec"]),
        **vector_columns("force_kcu", trim.get("kcu_drag_force_vsm", np.zeros(3))),
        **vector_columns("force_inertial", trim.get("inertial_force", np.zeros(3))),
        **vector_columns("body_z", body_z),
        "tether_force": float(trim["tether_force"]),
        "speed_tangential": float(opt_x[0]), "roll_deg": float(opt_x[1]),
        "pitch_deg": float(opt_x[2]), "yaw_deg": float(opt_x[3]),
        "course_rate": float(opt_x[4]),
        "aoa_deg": float(trim["aoa_deg"]),
        "stalled_fraction": float(trim.get("stalled_fraction", np.nan)),
        "converged": bool(meta["converged"]),
        "success_physical": bool(trim.get("success_physical", False)),
        "residual_relative": float(meta.get("residual_relative", np.nan)),
        "n_iter": int(meta["n_iter"]),
        "exit_reason": str(meta.get("exit_reason", "")),
        "polar_check": float(polar_check),
    }
    return row


def polar_rows(anchor: str, polar: Iterable[Mapping]) -> list[dict]:
    """``polars.csv`` rows from ``run_frozen_alpha_polar_about_trim``'s output."""
    return [
        {
            "anchor": anchor,
            "d_alpha_deg": float(sample["d_alpha_deg"]),
            "alpha_body_deg": float(sample["alpha_body_deg"]),
            **vector_columns("va", sample["va_world"]),
            **vector_columns("force_aero", sample["force_world"]),
            "gamma_converged": bool(sample["gamma_converged"]),
            "av_stage": sample.get("av_stage") or "",
        }
        for sample in polar
    ]


def append_rows(path, rows: list[dict]) -> None:
    """Append rows to a CSV, writing the header when the file is new."""
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        if is_new:
            writer.writeheader()
        writer.writerows(rows)


def load_sweep(folder):
    """``(trims, polars)`` DataFrames of a sweep folder."""
    import pandas as pd

    folder = Path(folder)
    trims = pd.read_csv(folder / TRIMS_FILE)
    polars = pd.read_csv(folder / POLARS_FILE, keep_default_na=False,
                         na_values=["nan", "NaN"])
    return trims, polars


# ───────────────────────────────────────────── derived samples (paper App. D)
def to_course_frame(v) -> np.ndarray:
    """Trim (VSM) frame -> paper C-frame, row-wise: the kite flies along
    -e_course there, so the course and normal axes flip (right-handedness is
    kept)."""
    v = np.asarray(v, dtype=float)
    return v * np.array([-1.0, -1.0, 1.0])


def _va_frame(va_c):
    """(e_n', e_r') of paper Eqs. D6-D8 for C-frame apparent winds (n, 3)."""
    va_chi, va_n, va_r = va_c.T
    va_tau = np.hypot(va_chi, va_n)
    va = np.linalg.norm(va_c, axis=1)
    e_n = np.column_stack([-va_n, va_chi, np.zeros_like(va)]) / va_tau[:, None]
    e_r = np.column_stack([-va_chi * va_r, -va_n * va_r, va_tau**2]) / (va * va_tau)[:, None]
    return e_n, e_r


def roll_about_va(force_c, va_c) -> np.ndarray:
    """Roll of a force about v_a, paper Eqs. D6-D8 (0 along e_r'), row-wise."""
    force_c, va_c = np.atleast_2d(force_c), np.atleast_2d(va_c)
    e_n, e_r = _va_frame(va_c)
    return np.arctan2(-np.einsum("ij,ij->i", force_c, e_n),
                      np.einsum("ij,ij->i", force_c, e_r))


def bridle_angle_of_attack(force_bridle_c, va_c) -> np.ndarray:
    """Paper Eq. D11: alpha_b = atan2(F_b . e_chi', -F_b . e_r'), e_chi' = -va/|va|."""
    force_bridle_c, va_c = np.atleast_2d(force_bridle_c), np.atleast_2d(va_c)
    _, e_r = _va_frame(va_c)
    e_chi = -va_c / np.linalg.norm(va_c, axis=1)[:, None]
    return np.arctan2(np.einsum("ij,ij->i", force_bridle_c, e_chi),
                      -np.einsum("ij,ij->i", force_bridle_c, e_r))


def lift_drag_coefficients(force, va, body_z, rho, area):
    """(C_L, C_D) of forces about their apparent winds, row-wise; lift signed
    by the body up axis (-body_z)."""
    force, va, body_z = (np.atleast_2d(np.asarray(x, dtype=float)) for x in (force, va, body_z))
    e_va = va / np.linalg.norm(va, axis=1)[:, None]
    drag = np.einsum("ij,ij->i", force, e_va)
    f_perp = force - drag[:, None] * e_va
    up = -(body_z - np.einsum("ij,ij->i", body_z, e_va)[:, None] * e_va)
    lift = np.sign(np.einsum("ij,ij->i", f_perp, up)) * np.linalg.norm(f_perp, axis=1)
    q = 0.5 * np.asarray(rho, dtype=float) * np.einsum("ij,ij->i", va, va)
    return lift / (q * area), drag / (q * area)


def coefficient_tables(trims, polars, *, reference_area: float,
                       steering_tape_per_unit_us: float):
    """``(polar_samples, trim_samples)``: the sweep in the ROM's variables.

    ``polar_samples`` has one row per coefficient sample -- the anchor's
    coupled trim (``source`` "coupled_trim") and its polar steps ("polar";
    the 0 step, which re-solves the trim, is the trim row) -- with ``u_p``
    (the power-tape length [m]), ``u_s`` (half-difference /
    ``steering_tape_per_unit_us``), ``v_a`` [m/s], ``alpha`` [rad] (the trim's
    centre-chord angle plus the polar increment), ``CL`` and ``CD`` on
    ``reference_area``, and ``converged``.

    ``trim_samples`` has one row per anchor with the quantities the ROM's
    bridle and steering laws are identified on: ``alpha_w`` (centre-chord
    angle), ``alpha_b`` (paper Eq. D11 angle of the bridle resultant F_b the
    ROM builds -- tether plus the KCU's drag and inertial share, i.e. minus
    the wing's force and inertial share), ``theta_b = alpha_b - alpha_w``,
    ``phi_a`` (roll of the lift about v_a), ``phi_a_b`` (roll of -F_b) and
    ``phi_a_w = phi_a - phi_a_b``, plus ``tension``, ``v_tau``, ``chi_dot``.
    """
    import pandas as pd

    t = trims.reset_index(drop=True)
    u_s = t["steering_half_difference"].to_numpy(float) / steering_tape_per_unit_us
    va = vector(t, "va")
    force = vector(t, "force_aero")
    body_z = vector(t, "body_z")
    cl, cd = lift_drag_coefficients(force, va, body_z, t["rho"], reference_area)
    alpha_w = np.deg2rad(t["aoa_deg"].to_numpy(float))
    by_anchor = pd.DataFrame({
        "anchor": t["anchor"], "chain": t["chain"],
        "u_p": t["power_tape_length"].to_numpy(float), "u_s": u_s,
        "v_a": np.linalg.norm(va, axis=1), "alpha_w": alpha_w,
        "rho": t["rho"], "body_z_x": body_z[:, 0], "body_z_y": body_z[:, 1],
        "body_z_z": body_z[:, 2],
    })

    trim_samples = by_anchor[["anchor", "chain", "u_p", "u_s", "v_a", "alpha_w"]].copy()
    va_c, f_wing_c = to_course_frame(va), to_course_frame(force)
    share_wing = 1.0 - t["mass_kcu"].to_numpy(float) / t["mass_total"].to_numpy(float)
    f_bridle_c = -(f_wing_c + share_wing[:, None] * to_course_frame(vector(t, "force_inertial")))
    e_va = va_c / np.linalg.norm(va_c, axis=1)[:, None]
    lift_c = f_wing_c - np.einsum("ij,ij->i", f_wing_c, e_va)[:, None] * e_va
    trim_samples["alpha_b"] = bridle_angle_of_attack(f_bridle_c, va_c)
    trim_samples["theta_b"] = trim_samples["alpha_b"] - alpha_w
    trim_samples["phi_a"] = roll_about_va(lift_c, va_c)
    trim_samples["phi_a_b"] = roll_about_va(-f_bridle_c, va_c)
    trim_samples["phi_a_w"] = trim_samples["phi_a"] - trim_samples["phi_a_b"]
    trim_samples["wind_speed"] = t["wind_speed"].to_numpy(float)
    trim_samples["tension"] = t["tether_force"].to_numpy(float)
    trim_samples["v_tau"] = t["speed_tangential"].to_numpy(float)
    trim_samples["chi_dot"] = t["course_rate"].to_numpy(float)

    coupled = by_anchor[["anchor", "chain", "u_p", "u_s", "v_a"]].copy()
    coupled["source"] = "coupled_trim"
    coupled["d_alpha_deg"] = 0.0
    coupled["alpha"] = alpha_w
    coupled["CL"], coupled["CD"] = cl, cd
    coupled["converged"] = True

    p = polars[polars["d_alpha_deg"] != 0.0].merge(by_anchor, on="anchor", how="inner")
    p_va = vector(p, "va")
    p_cl, p_cd = lift_drag_coefficients(
        vector(p, "force_aero"), p_va, vector(p, "body_z"), p["rho"], reference_area
    )
    stepped = pd.DataFrame({
        "anchor": p["anchor"], "chain": p["chain"], "u_p": p["u_p"], "u_s": p["u_s"],
        "v_a": np.linalg.norm(p_va, axis=1), "source": "polar",
        "d_alpha_deg": p["d_alpha_deg"].to_numpy(float),
        "alpha": p["alpha_w"].to_numpy(float) + np.deg2rad(p["d_alpha_deg"].to_numpy(float)),
        "CL": p_cl, "CD": p_cd,
        "converged": p["gamma_converged"].astype(bool).to_numpy(),
    })
    polar_samples = (
        pd.concat([coupled, stepped], ignore_index=True)
        .sort_values(["anchor", "alpha"])
        .reset_index(drop=True)
    )
    return polar_samples, trim_samples
