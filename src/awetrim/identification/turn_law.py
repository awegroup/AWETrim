"""The turn-rate law's inertial and gravity terms, measured on coupled states.

The ROM's turn-rate law (``aerodynamics.steering_law: turn_rate``,
:mod:`awetrim.system.kite`) is Eq. 33 of Cayon & Schmehl, *Quasi-steady
equilibrium and turn-rate law of kites* (in preparation), with the
coefficients of its App. A, divided by k2:

    w (v_a + b m v_tau) + b F_g,n + c5 F_g,chi + c6 F_g,r = a(u_p, u_s) v_a^2 u_s

with w the turn rate about the tether, F_g = -m g [cos chi cos beta,
sin chi cos beta, sin beta] the weight in the course frame [chi, n, r],
b = k3/k2 = k4/k2, c5 = k5/k2 and c6 = k6/k2 [1/kg].

The identification sweep solves every state at the centre of the window,
gravity off, at the kite's mass. There v_tau / v_a is fixed per tape, so the
sweep fits a / (1 + b m v_tau / v_a) only: a and b cannot be told apart, and
gravity-off states carry no c5, c6. They are measured here on a few extra
states of sweep anchors (``run_identification_sweep.py --turn-law``, written to
``<sweep>/turn_law/trims.csv``), tetherless:

* per anchor, the centre at mass scale 0 and 1, gravity off. Without mass,
  w = a v_a u_s gives a; the full-mass state then gives b. b(u_p) is a
  polynomial over the anchors' tapes.
* gravity-on states of an anchor away from the centre: with that anchor's a
  and b, what the b F_g,n term leaves is regressed on F_g,chi and F_g,r
  (least squares over every gravity-on state), giving c5 and c6.

The rows are the sweep's trim rows (``aerostructural_dataset.trim_row``) plus
``base`` (the sweep anchor), ``state`` (:func:`state_label`), ``mass_scale``,
``gravity``, ``elevation_deg``, ``azimuth_deg`` and ``course_deg``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from awetrim.identification import aerostructural_dataset as ds

#: Folder of the turn-law states inside the sweep folder.
TURN_LAW_DIR = "turn_law"
#: (elevation, azimuth, course) [deg] of the sweep's own states.
CENTRE = (0.0, 0.0, 90.0)
#: Mass scale 0 is solved at this (the inertial load still spreads over the nodes).
MASS_FLOOR = 1e-6
GRAVITY = 9.81


def state_label(elevation_deg: float, azimuth_deg: float, course_deg: float) -> str:
    return f"b{elevation_deg:g}_p{azimuth_deg:g}_c{course_deg:g}"


def state_anchor(base: str, label: str, mass_scale: float, gravity: bool) -> str:
    """The row's ``anchor``: the sweep anchor, the state, the mass scale, ``@g`` with gravity."""
    return f"{base}@{label}@m{mass_scale:g}" + ("@g" if gravity else "")


def turn_law_table(trims: pd.DataFrame, *, steering_tape_per_unit_us: float) -> pd.DataFrame:
    """The states with ``w`` (turn rate about the tether, -omega_z), ``v_a``,
    ``v_tau``, ``u_p`` (power-tape length), ``u_s``, ``mass``; mass scales at
    the floor read 0, ``gravity`` is boolean."""
    t = trims.copy()
    t["mass_scale"] = t.mass_scale.where(t.mass_scale > 10 * MASS_FLOOR, 0.0)
    t["gravity"] = t.get("gravity", pd.Series(False, index=t.index)).astype(str).str.lower() == "true"
    t["w"] = -t.omega_z.astype(float)
    t["v_a"] = np.linalg.norm(ds.vector(t, "va"), axis=1)
    t["v_tau"] = t.speed_tangential.astype(float)
    t["u_p"] = t.power_tape_length.astype(float)
    t["u_s"] = t.steering_half_difference.astype(float) / steering_tape_per_unit_us
    t["mass"] = t.mass_total.astype(float)
    return t


def inertia_terms(table: pd.DataFrame) -> pd.DataFrame:
    """a and b per anchor, from its centre at mass scale 0 and 1 (gravity off).

    Anchors without both states, or unsteered, are left out. ``inertia_share``
    is b m v_tau / (v_a + b m v_tau) of the full-mass state: the part of the
    yaw damping the inertia carries."""
    centre = state_label(*CENTRE)
    calm = table[(table.state == centre) & ~table.gravity]
    rows = []
    for base, g in calm.groupby("base"):
        zero, full = g[g.mass_scale == 0.0], g[g.mass_scale == 1.0]
        if zero.empty or full.empty or abs(zero.u_s.iloc[0]) < 1e-9:
            continue
        z, c = zero.iloc[0], full.iloc[0]
        a = z.w / (z.v_a * z.u_s)
        b = (a * c.v_a**2 * c.u_s / c.w - c.v_a) / (c.mass * c.v_tau)
        ratio = b * c.mass * c.v_tau / c.v_a
        rows.append({"base": base, "u_p": c.u_p, "u_s": c.u_s, "a": a, "b": b,
                     "inertia_share": ratio / (1.0 + ratio), "rho": c.rho})
    return pd.DataFrame(rows, columns=["base", "u_p", "u_s", "a", "b", "inertia_share", "rho"])


def fit_inertia_polynomial(inertia: pd.DataFrame, degree: int = 1) -> list[float]:
    """b(u_p) = b0 + b1 u_p + ... [1/kg] over the anchors' tapes (least
    squares; the degree is capped at one fewer than the distinct tapes)."""
    if inertia.empty:
        raise ValueError("no anchor has its centre at mass scales 0 and 1: b cannot be measured")
    degree = min(int(degree), inertia.u_p.round(3).nunique() - 1)
    return [float(c) for c in np.polynomial.polynomial.polyfit(inertia.u_p, inertia.b, degree)]


def gravity_terms(table: pd.DataFrame, inertia: pd.DataFrame) -> tuple[float, float, pd.DataFrame]:
    """``(c5, c6, rows)``: least squares over the gravity-on, full-mass states
    of the anchors in ``inertia``. ``rows`` holds, per state, what the b F_g,n
    term leaves per unit m g (``y``) and its regressors cos chi cos beta
    (``x5``) and sin beta (``x6``); c5 = c6 = 0 without gravity-on states."""
    terms = inertia.set_index("base")
    grav = table[table.gravity & (table.mass_scale == 1.0) & table.base.isin(terms.index)]
    if grav.empty:
        return 0.0, 0.0, pd.DataFrame(columns=["base", "state", "y", "x5", "x6"])
    a, b = grav.base.map(terms.a), grav.base.map(terms.b)
    beta, chi = np.radians(grav.elevation_deg), np.radians(grav.course_deg)
    mg = grav.mass * GRAVITY
    left = (grav.w * (grav.v_a + b * grav.mass * grav.v_tau) - a * grav.v_a**2 * grav.u_s
            - b * mg * np.sin(chi) * np.cos(beta))
    rows = pd.DataFrame({"base": grav.base, "state": grav.state, "y": left / mg,
                         "x5": np.cos(chi) * np.cos(beta), "x6": np.sin(beta)})
    (c5, c6), *_ = np.linalg.lstsq(rows[["x5", "x6"]].to_numpy(float), rows.y.to_numpy(float),
                                   rcond=None)
    return float(c5), float(c6), rows.reset_index(drop=True)


def measure(table: pd.DataFrame, inertia_degree: int = 1) -> dict:
    """b(u_p), c5 and c6 of :func:`turn_law_table`'s states.

    Returns ``inertia_coefficient`` [b0, b1, ...], ``gravity_course_coefficient``
    (c5), ``gravity_radial_coefficient`` (c6), ``density_reference`` (the
    states' density, at which they hold) and the tables behind them
    (``inertia``, ``gravity``)."""
    inertia = inertia_terms(table)
    b_coefs = fit_inertia_polynomial(inertia, inertia_degree)
    c5, c6, gravity = gravity_terms(table, inertia)
    return {"inertia_coefficient": b_coefs, "gravity_course_coefficient": c5,
            "gravity_radial_coefficient": c6, "density_reference": float(inertia.rho.median()),
            "inertia": inertia, "gravity": gravity}


def load_turn_law_states(sweep_folder) -> pd.DataFrame | None:
    """``<sweep>/turn_law/trims.csv``, converged rows only; None when absent."""
    path = Path(sweep_folder) / TURN_LAW_DIR / ds.TRIMS_FILE
    if not path.is_file():
        return None
    t = pd.read_csv(path)
    return t[t.converged.astype(str).str.lower() == "true"].reset_index(drop=True)
