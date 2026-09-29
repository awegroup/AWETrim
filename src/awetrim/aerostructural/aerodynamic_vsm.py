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
#
# Portions of this file are adapted from ASKITE
# (https://github.com/awegroup/ASKITE), licensed under the MIT License,
# Copyright (c) 2024 jellepoland (Jelle Poland, Patrick Roeleveld, TU Delft).
# See the NOTICE file at the repository root for the full MIT licence text.

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import copy
import dataclasses
import time
from VSM.core.BodyAerodynamics import BodyAerodynamics
from VSM.core.WingGeometry import Wing
from VSM.core.Solver import Solver
from VSM.plot_geometry_matplotlib import plot_geometry
from awetrim.aerodynamics.kcu_drag import KcuDragModel
from awetrim.aerodynamics.line_drag import (
    apply_bridle_drag_diameters,
    bridle_drag_diameters,
    settings_from_config,
)
from awetrim.aerodynamics.vsm_quasi_steady import (
    extend_polar_past_onset,
    solve_quasi_steady_state,
    solve_vsm_qs_trim_with_williams_tether,
    AV_ATTACHED_FIRST as _AV_ATTACHED_FIRST_DEFAULT,
    DEFAULT_TRANSFORMATION_C_FROM_VSM,
)

# Bounds and defaults (aoa, sideslip, course_rate_body)
kite_speed_bounds = (-2.0, 50.0)  # m/s
pitch_bounds = (-10, 10)  # deg
yaw_bounds = (-10, 10)  # deg
course_rate_bounds = (
    -3,
    3,
)  # rad/s, small course rate allowed for numerical reasons; not a physical course rate
roll_bounds = (
    -10,
    10,
)  # deg, small roll allowed for numerical reasons; not a physical roll

DEFAULT_GUESS_QS = np.array(
    [30.0, 0.0, 0.0, 0.0, -0.0]
)  # [kite_speed, roll, pitch, yaw, course_rate_body]


#: Trim-state names, for the on-bound diagnostic below.
_QS_STATE_NAMES = ("kite_speed", "roll", "pitch", "yaw", "course_rate")


def _warn_if_trim_on_bounds(opt_x, bounds_lower, bounds_upper, rel_tol=1e-6) -> list:
    """Warn for each trim unknown that ended on its search bound.

    ``least_squares`` reports ``success=True`` for a constrained optimum, so a
    trim pinned against a bound looks converged while being no equilibrium at
    all -- the force balance wanted a value the box forbade. Returns the list of
    ``(name, value, "lower"/"upper")`` that are pinned (empty if none).
    """
    if opt_x is None:
        return []
    x = np.asarray(opt_x, dtype=float).ravel()
    lo = np.asarray(bounds_lower, dtype=float).ravel()
    hi = np.asarray(bounds_upper, dtype=float).ravel()
    if x.size != lo.size or x.size != hi.size:
        return []
    pinned = []
    for i, name in enumerate(_QS_STATE_NAMES[: x.size]):
        span = max(abs(hi[i] - lo[i]), 1e-12)
        if abs(x[i] - lo[i]) <= rel_tol * span:
            pinned.append((name, float(x[i]), "lower"))
        elif abs(x[i] - hi[i]) <= rel_tol * span:
            pinned.append((name, float(x[i]), "upper"))
    for name, value, which in pinned:
        logging.warning(
            "Quasi-steady trim is pinned on its %s bound: %s = %.6g. This is a "
            "constrained optimum, NOT an equilibrium -- widen the bound "
            "(quasi_steady.%s_bounds* in the config) or the deformed shape and "
            "everything downstream describe a state the kite cannot fly.",
            which,
            name,
            value,
            name,
        )
    return pinned


def _alpha_stall_from_polar(polar_data: np.ndarray) -> float:
    """Return the stall angle of attack [rad] from a 2-D polar table.

    Stall is defined as the **first local Cl maximum** in the positive-Cl
    region of the polar (the onset of stall, not any post-stall Cl recovery).
    The polar table must have columns [alpha_rad, Cl, Cd, Cm].

    Returns ``math.inf`` if no peak is found within the table range, meaning
    stall is not detectable from this polar and the panel is never flagged.
    """
    import math

    polar = np.asarray(polar_data)
    cl = polar[:, 1]
    alpha = polar[:, 0]

    # Only search in the positive-Cl region to ignore noise at negative alpha.
    pos = np.where(cl > 0)[0]
    if len(pos) == 0:
        return math.inf

    # Find the first local maximum within the positive-Cl rows.
    for k in pos[1:-1]:
        if cl[k] > cl[k - 1] and cl[k] > cl[k + 1]:
            return float(alpha[k])

    # No interior peak found — polar does not cover stall.
    return math.inf


def check_panel_stall(
    alpha_at_ac: np.ndarray,
    panel_polar_data: list,
) -> np.ndarray:
    """Return a boolean mask flagging panels whose local AoA exceeds stall.

    Args:
        alpha_at_ac: Local angle of attack per panel [rad], shape (n_panels,).
        panel_polar_data: List of polar arrays (one per panel), each shaped
            (N, 4) with columns [alpha_rad, Cl, Cd, Cm].

    Returns:
        stall_mask: Boolean array of shape (n_panels,); True where panel stalls.
    """
    alpha = np.ravel(alpha_at_ac)
    n = len(alpha)
    stall_mask = np.zeros(n, dtype=bool)
    for i, polar in enumerate(panel_polar_data):
        if i >= n:
            break
        alpha_stall = _alpha_stall_from_polar(polar)
        stall_mask[i] = alpha[i] > alpha_stall
    return stall_mask


def _run_vsm_direct_fallback(body_aero, solver, system_model, current_guess):
    """
    Fallback used when quasi-steady trim fails.

    This path bypasses trim optimization and runs a direct aerodynamic solve on the
    current body geometry. The returned dictionary mirrors the keys expected by the
    coupled solver pipeline.
    """
    res = solver.solve(body_aero)

    # Convert AWETrim vectors into the same course-frame convention used by QSM outputs.
    trans = np.asarray(DEFAULT_TRANSFORMATION_C_FROM_VSM, dtype=float)
    # Total kite mass (wing + KCU); mass_wing/mass_kcu live on system_model.kite.
    mass_total = float(system_model.kite.mass_wing) + float(
        getattr(system_model.kite, "mass_kcu", 0.0)
    )
    # SystemModel does not expose these as plain numeric attributes (they live
    # in the CasADi expression registry); a missing one must not crash the
    # fallback -- this path already reports success=False, and the coupled loop
    # only needs F_distribution to keep stepping. Zero is the QS assumption.
    def _course_vector(attribute):
        value = getattr(system_model, attribute, None)
        if value is None:
            print(
                f"WARNING: system_model has no numeric {attribute!r}; "
                "fallback uses zeros."
            )
            return np.zeros(3)
        return np.asarray(
            trans @ np.asarray(value, dtype=float), dtype=float
        ).reshape(3)

    inertial_force = -mass_total * _course_vector("acceleration_course_body")
    gravity_force = _course_vector("force_gravity")

    guess = np.asarray(current_guess, dtype=float).reshape(5)
    fallback_opt_x = np.array(DEFAULT_GUESS_QS, dtype=float)
    print(
        f"Falling back to direct VSM solve with guess {guess} (quasi-steady optimization failed)."
    )
    print(
        f"Direct VSM results: {res.get('F_distribution')}, cmx: {res.get('cmx')}, cmy: {res.get('cmy')}, cmz: {res.get('cmz')}, side_slip_deg: {res.get('side_slip_deg')}, side_slip_course_deg: {res.get('side_slip_course_deg')}"
    )
    return {
        "opt_x": fallback_opt_x,
        "success": False,
        "inertial_force": inertial_force,
        "gravity_force": gravity_force,
        "panel_cp_locations": res.get("panel_cp_locations"),
        "F_distribution": res.get("F_distribution"),
        "alpha_at_ac": res.get("alpha_at_ac"),
        "stall_mask": None,
    }


def _solve_trim_casadi(
    *,
    body_aero,
    solver,
    system_model,
    center_of_gravity,
    reference_point,
    x_guess,
    bounds_lower,
    bounds_upper,
    include_gravity,
    kcu_drag,
    tether_model,
    gamma_seed,
    tolerance,
):
    """The quasi-steady trim as ONE CasADi root-finding problem
    (``awetrim.aerodynamics.trim_casadi``), in the contract of the NumPy
    trims: ``(results, body)`` with the per-panel loads and the rotated body
    the coupled loop consumes.

    The VSM settings come off the coupled solver's own ``Solver`` (artificial
    viscosity and its factor, polar interpolation, rho, core radius, model
    type), so the trim solves the same circulation problem as the
    ``least_squares`` path's inner loop. ``tether_model`` is ``None``
    (tetherless) or ``"williams"``; ``rigid_lumped`` is not covered by the
    CasADi trim and is refused upstream.

    Attached-first branch rule, at the TRIM level: with the artificial
    viscosity on and ``attached_first`` set on the solver (the default,
    ``solve_vsm_attached_first``), the trim is first solved on every polar
    continued linearly past its stall onset (no stalled-tip fixed point to
    lock into) and kept when every panel comes out below its ORIGINAL onset
    -- that state is an exact solution of the true model, ``av_stage =
    "attached"``. Otherwise the true polars + AV problem is solved from the
    caller's seed, ``av_stage = "stalled"``. The NumPy trims apply the same
    rule per VSM evaluation; the trim is either attached or not, so once per
    trim is the consistent place, and it costs one extra graph build.
    """
    from awetrim.aerodynamics.trim_casadi import CasadiTrim, CasadiTrimOptions
    from awetrim.aerodynamics.vsm_quasi_steady import (
        _attached_polars_installed,
        _panel_stall_onsets_rad,
    )

    options = CasadiTrimOptions(
        tether_model=tether_model,
        polar_interpolation=str(getattr(solver, "polar_interpolation", "linear")),
        is_with_artificial_viscosity=bool(
            getattr(solver, "is_with_artificial_viscosity", False)
        ),
        artificial_viscosity_factor=float(
            getattr(solver, "artificial_viscosity_factor", 0.035)
        ),
        # Refused inside CasadiTrim when True: the kernels are the
        # freestream-direction law, so a config asking for the corrected
        # directions must run the least_squares trims.
        is_aoa_corrected=bool(getattr(solver, "is_aoa_corrected", False)),
        include_gravity=bool(include_gravity),
        tolerance=float(tolerance),
        rho=float(getattr(solver, "rho", 1.225)),
        core_radius_fraction=float(getattr(solver, "core_radius_fraction", 0.05)),
        aerodynamic_model_type=str(getattr(solver, "aerodynamic_model_type", "VSM")),
    )

    # One graph per (polars, counts, options, system model), cached on the
    # coupled solver's own Solver object and re-used across the coupled
    # iterations with the deformed shape swapped in as parameters: the build
    # (~0.5 s at 45 panels) is paid once per coupled solve, not per
    # iteration. The attached-first stage swaps polars, so it gets its own
    # entry. Keyed on the system model's identity too (its wind, position and
    # masses are baked into the graph numerically).
    cache = getattr(solver, "_awetrim_casadi_trims", None)
    if cache is None:
        cache = {}
        try:
            solver._awetrim_casadi_trims = cache
        except AttributeError:  # a slotted or mock solver: no cache
            pass
    def run_once():
        # The key is taken HERE, with the polars actually installed: the
        # attached-first stage swaps in continued polars, so a key computed
        # once up front filed that stage's graph under the TRUE polars' entry,
        # and every stalled-stage solve then tripped update_geometry's polar
        # check and fell back to an untrimmed direct VSM solve (2026-09-15:
        # every coupled iteration of the deep-depower reel-in grid points).
        key = (
            CasadiTrim.geometry_signature(body_aero),
            dataclasses.astuple(options),
            None if kcu_drag is None else dataclasses.astuple(kcu_drag),
            tuple(np.asarray(bounds_lower, dtype=float).ravel().tolist()),
            tuple(np.asarray(bounds_upper, dtype=float).ravel().tolist()),
            tuple(np.asarray(reference_point, dtype=float).ravel().tolist()),
        )
        entry = cache.get(key)
        if entry is not None and entry[0] is system_model:
            trim = entry[1]
            trim.update_geometry(body_aero, center_of_gravity)
        else:
            trim = CasadiTrim(
                body_aero,
                system_model,
                center_of_gravity,
                reference_point,
                kcu_drag=kcu_drag,
                bounds_lower=bounds_lower,
                bounds_upper=bounds_upper,
                options=options,
            )
            if len(cache) >= 8:
                cache.pop(next(iter(cache)))
            cache[key] = (system_model, trim)
        return trim.solve_with_body(x_guess, gamma_seed=gamma_seed)

    av_first = options.is_with_artificial_viscosity and bool(
        getattr(solver, "_awetrim_attached_first", _AV_ATTACHED_FIRST_DEFAULT)
    )
    if not av_first:
        results, body = run_once()
        results["av_stage"] = None
        return results, body

    # Stage 1: the attached predictor. The graph reads the polars at
    # construction, so building it inside the swap bakes the continued polars
    # in; the returned body is a deep copy made during the swap and is given
    # the original tables back, so nothing downstream inherits the extension.
    onsets = _panel_stall_onsets_rad(body_aero)
    with _attached_polars_installed(body_aero):
        results, body = run_once()
    for panel_out, panel_in in zip(body.panels, body_aero.panels):
        original = getattr(panel_in, "_panel_polar_data", None)
        if original is not None:
            panel_out._panel_polar_data = original
    alpha = results.get("alpha_at_ac")
    attached = bool(results.get("converged", False))
    if attached and alpha is not None and onsets.size:
        alpha = np.asarray(alpha, dtype=float).ravel()
        if alpha.shape == onsets.shape:
            finite = np.isfinite(alpha) & np.isfinite(onsets)
            attached = bool(np.all(alpha[finite] <= onsets[finite])) and bool(
                np.all(np.isfinite(alpha))
            )
        else:
            attached = False
    if attached:
        results["av_stage"] = "attached"
        return results, body

    # Stage 2: the true model, from the caller's seed.
    predictor_converged = bool(results.get("converged", False))
    results, body = run_once()
    results["av_stage"] = "stalled"
    results["attached_predictor_converged"] = predictor_converged
    return results, body


def initialize(
    aero_geometry_path,
    config,
    n_panels_aero: int,
    bridle_path=None,
) -> BodyAerodynamics:
    """
    Initialize aerodynamic model and VSM solver.

    Args:
        aero_geometry_path: Path to aerodynamic geometry file.
        config (dict): Main ASKITE configuration dictionary.
        n_panels_aero (int): Number of aerodynamic panels.
        bridle_path: Optional structural geometry path used by VSM to build bridle lines.

    Returns:
        tuple: (body_aero, vsm_solver, vel_app, initial_polar_data)
    """
    body_aero = BodyAerodynamics.instantiate(
        n_panels=int(n_panels_aero),
        file_path=aero_geometry_path,
        spanwise_panel_distribution=config["aerodynamic"][
            "spanwise_panel_distribution"
        ],
        bridle_path=bridle_path,
    )
    # The bridle-line drag law reads one diameter per segment, and the
    # structural table stores flat tapes as an AREA-equivalent circle --
    # right for mass/EA, meaningless for drag. Swap in the drag-equivalent
    # diameters before anything solves (awetrim.aerodynamics.line_drag).
    if bridle_path is not None:
        import yaml as _yaml

        with open(bridle_path, "r", encoding="utf-8") as f:
            _struc_geometry = _yaml.safe_load(f)
        _roll_model, _cd_cable = settings_from_config(config)
        _n = apply_bridle_drag_diameters(
            body_aero, _struc_geometry, _roll_model, _cd_cable
        )
        if _n:
            logging.info(
                "flat-tape drag applied to %d bridle segments (roll model %r)",
                _n,
                _roll_model,
            )

    aero_cfg = config["aerodynamic"]
    vsm_solver = Solver(
        max_iterations=aero_cfg["max_iterations"],
        allowed_error=aero_cfg["allowed_error"],
        relaxation_factor=aero_cfg["relaxation_factor"],
        reference_point=aero_cfg["reference_point"],
        mu=config["mu"],
        rho=config["rho"],
        # Optional post-stall stabilization: the parameter-free Li/Gaunaa
        # spanwise artificial viscosity (TORQUE 2026), applied in the base
        # gamma loop. Off by default to preserve historical behaviour.
        is_with_artificial_viscosity=aero_cfg.get(
            "is_with_artificial_viscosity", False
        ),
        artificial_viscosity_factor=aero_cfg.get("artificial_viscosity_factor", 0.035),
        # Inner circulation solver. "base" is the relaxed-Picard loop and stays
        # the default. "anderson" is Anderson-accelerated (O(10s) rather than
        # O(100s) of iterations) but terminates on a superlinear, non-smooth
        # residual, which corrupts the finite-difference Jacobian of the QSM
        # trim this solver calls unless `allowed_error` is ~1e-8. Set the two
        # together or not at all.
        gamma_loop_type=aero_cfg.get("gamma_loop_type", "base"),
        # Force directions from the freestream (VSM ``is_aoa_corrected``):
        # False is the model every AWETrim result is built on and what the
        # CasADi trim kernels reproduce. VSM develop flipped its own default
        # to True on 2026-09-17 (quarter-chord flow), so it is stated here.
        is_aoa_corrected=bool(aero_cfg.get("is_aoa_corrected", False)),
        # Anderson headroom instead of the Picard fallback (2026-09-03):
        # across the 2019+2025 steering campaigns the base-loop fallback
        # rescued 99 of ~92,400 Anderson failures (0.1%) while costing up to
        # two 1500-iteration base loops per failure. 1000 / False are the
        # VSM defaults too; the keys exist so a kite config can dial back.
        anderson_max_iterations=int(aero_cfg.get("anderson_max_iterations", 1000)),
        anderson_fallback_to_base=bool(
            aero_cfg.get("anderson_fallback_to_base", False)
        ),
    )

    # For QSM, wind speed comes from system model configuration (wind_speed_wind_ref).
    # Kite velocity is computed by the optimizer, so we initialize with wind direction.
    wind_speed_ref = float(config.get("wind_speed_wind_ref", 6.0))
    vel_app = np.array([wind_speed_ref, 0.0, 0.0])
    body_aero.va = vel_app
    wing = body_aero.wings[0]
    new_sections = wing.refine_aerodynamic_mesh()
    initial_polar_data = []
    for new_section in new_sections:
        initial_polar_data.append(new_section.polar_data)
    # ATTACHED-BRANCH FINDER (aerodynamic.attached_polars, off by default):
    # every section polar continued linearly past its stall onset, so the
    # coupled solve cannot lock into the AV stalled-tip family. Only a state
    # with every panel below its ORIGINAL onset is a solution of the true
    # model; the caller (the sweep's attached guard) validates that with a
    # trim on the true polars and discards the rest. The stall mask this
    # adapter logs is built from the extended polars and reports nothing --
    # the validation is the caller's, not this log line.
    # ATTACHED-FIRST branch finder, per run. Defaults to the module's own
    # AV_ATTACHED_FIRST (True since 2026-09-12): try the attached branch first
    # on every VSM evaluation and keep it when the whole wing comes out below
    # its stall onset, falling through to the true post-stall problem when it
    # does not. Set `aerodynamic.attached_first: false` to reproduce an older
    # result. Stored on the solver rather than mutating the module global, so
    # two runs in one process cannot fight over it.
    vsm_solver._awetrim_attached_first = bool(
        aero_cfg.get("attached_first", _AV_ATTACHED_FIRST_DEFAULT)
    )

    if bool(aero_cfg.get("attached_polars", False)):
        initial_polar_data = [
            extend_polar_past_onset(polar) for polar in initial_polar_data
        ]
        body_aero.update_from_points(
            *(
                np.array([[s.LE_point for s in new_sections]][0]),
                np.array([[s.TE_point for s in new_sections]][0]),
            ),
            aero_input_type="reuse_initial_polar_data",
            initial_polar_data=initial_polar_data,
        )
        logging.warning(
            "attached_polars: %d section polars continued linearly past their "
            "stall onset -- the coupled solve targets the ATTACHED family; "
            "validate the result on the true polars",
            len(initial_polar_data),
        )

    return body_aero, vsm_solver, vel_app, initial_polar_data


def plot_vsm_geometry(body_aero):
    """
    Plot the VSM geometry using the provided aerodynamic body.

    Args:
        body_aero (BodyAerodynamics): Aerodynamic body object.

    Returns:
        None. Displays a 3D plot.
    """
    plot_geometry(
        body_aero,
        title="VSM Geometry",
        data_type=None,
        save_path=None,
        is_save=False,
        is_show=True,
        view_elevation=15,
        view_azimuth=-120,
    )


def parse_bridle_line_specs(struc_geometry: dict, config: dict = None) -> list:
    """``(node_i, node_j, diameter)`` rows of the VSM bridle-line system.

    Replicates exactly the parse ``BodyAerodynamics.instantiate(bridle_path=...)``
    performs (``bridle_connections`` rows against the ``bridle_lines`` table,
    one extra segment for 3-node pulley rows), but keeps NODE INDICES instead
    of baked-in coordinates -- so the segments can be rebuilt from the live
    ``struc_nodes`` as the structure deforms. Node ids match the structural
    array (0 = KCU/bridle point).

    The diameter is the DRAG-equivalent one, matching what ``initialize``
    put in the body: flat tapes carry their projected width, every other
    line its structural diameter (awetrim.aerodynamics.line_drag). Without
    ``config`` the module defaults apply.
    """
    if "bridle_connections" not in struc_geometry or "bridle_lines" not in struc_geometry:
        return []
    headers = struc_geometry["bridle_lines"]["headers"][1:]
    lines = {
        row[0]: dict(zip(headers, row[1:]))
        for row in struc_geometry["bridle_lines"]["data"]
    }
    diameters = bridle_drag_diameters(struc_geometry, *settings_from_config(config))
    specs = []
    for row in struc_geometry["bridle_connections"]["data"]:
        specs.append((int(row[1]), int(row[2]), diameters[len(specs)]))
        if len(row) == 4:
            specs.append((int(row[2]), int(row[3]), diameters[len(specs)]))
    return specs


def rebuild_bridle_line_system(body_aero, struc_nodes, bridle_line_specs) -> None:
    """Point the body's bridle-line drag system at the CURRENT node positions.

    ``BodyAerodynamics.update_from_points`` refreshes the wings only; the
    bridle segments built at ``instantiate`` keep their initial coordinates.
    On an actuated (steered/depowered) kite the deformed bridle is asymmetric,
    and its drag carries a roll moment the static initial bridle misses -- the
    coupled trim then converges to a slightly wrong roll state. Rebuilding the
    segment list per aero call keeps the bridle drag consistent with the
    structure the loads come from.
    """
    body_aero._bridle_line_system = [
        [
            np.asarray(struc_nodes[i], dtype=float).copy(),
            np.asarray(struc_nodes[j], dtype=float).copy(),
            diameter,
        ]
        for i, j, diameter in bridle_line_specs
    ]


def run_vsm_package(
    body_aero,
    solver,
    system_model,
    center_of_gravity,
    le_arr,
    te_arr,
    # va_vector,
    aero_input_type="reuse_initial_polar_data",
    initial_polar_data=None,
    reference_point=[0.0, 0.0, 0.0],
    include_gravity=False,
    is_with_plot=False,
    current_guess=None,
    config=None,
    struc_nodes=None,
    bridle_line_specs=None,
    gamma_seed=None,
):
    """
    Run quasi-steady aerodynamic solve for the current structural geometry.

    Args:
        body_aero (BodyAerodynamics): Aerodynamic body object.
        solver (Solver): VSM solver object.
        system_model: AWETrim system model used by quasi-steady trim.
        center_of_gravity (np.ndarray): Current center of gravity in solver frame.
        le_arr (np.ndarray): Leading edge points (n,3).
        te_arr (np.ndarray): Trailing edge points (n,3).
        aero_input_type (str): Type of aerodynamic input.
        initial_polar_data (list or None): Initial polar data for panels.
        reference_point (list[float]): Reference point for moments and rotations.
        include_gravity (bool): Include gravity in quasi-steady force/moment balance.
        is_with_plot (bool): If True, plot the geometry.
        current_guess (np.ndarray or None): Initial guess for quasi-steady optimizer.

    Returns:
        tuple: (F_distribution, body_aero, results)
    """
    # Update aerodynamic mesh from the latest structural leading/trailing-edge points.
    body_aero.update_from_points(
        le_arr,
        te_arr,
        aero_input_type=aero_input_type,
        initial_polar_data=initial_polar_data,
    )
    # update_from_points refreshes wings only; keep the bridle-line drag
    # segments tracking the deforming structure too (see
    # rebuild_bridle_line_system). Only when the body was built WITH bridles.
    if (
        bridle_line_specs
        and struc_nodes is not None
        and getattr(body_aero, "_bridle_line_system", None) is not None
    ):
        rebuild_bridle_line_system(body_aero, struc_nodes, bridle_line_specs)
    # set again where velocity vector is coming from
    # The VSM va setter accepts keyword arguments but properties don't support that in Python
    # So we call the underlying setter method directly using the descriptor protocol
    # type(body_aero).va.fset(body_aero, va_vector)

    # Trim search bounds. The module constants above are the defaults; a
    # ``config`` with a ``quasi_steady`` block overrides them. Those YAML keys
    # (kite_speed_bounds_ms, roll_bounds_deg, ...) already existed in
    # as_config.yaml and read as if they configured this solve, but nothing
    # consulted them -- so the hard-coded values silently won. That matters for
    # near-stationary states: a kite whose trim speed is below
    # kite_speed_bounds[0] cannot be represented at all, and the optimiser
    # returns a bound-constrained point that is NOT an equilibrium.
    # The kite configs name this block ``quasi_steady_trim``; ``quasi_steady``
    # is accepted as an alias so either spelling works.
    _cfg = config or {}
    qs_cfg = _cfg.get("quasi_steady_trim") or _cfg.get("quasi_steady") or {}
    speed_b = tuple(qs_cfg.get("kite_speed_bounds_ms", kite_speed_bounds))
    roll_b = tuple(qs_cfg.get("roll_bounds_deg", roll_bounds))
    pitch_b = tuple(qs_cfg.get("pitch_bounds_deg", pitch_bounds))
    yaw_b = tuple(qs_cfg.get("yaw_bounds_deg", yaw_bounds))
    rate_b = tuple(qs_cfg.get("course_rate_bounds_rad_s", course_rate_bounds))

    bounds_lower = np.array(
        [speed_b[0], roll_b[0], pitch_b[0], yaw_b[0], rate_b[0]]
    )
    bounds_upper = np.array(
        [speed_b[1], roll_b[1], pitch_b[1], yaw_b[1], rate_b[1]]
    )
    # Trim evaluation cap (opt-in, quasi_steady_trim.max_nfev). Inside a
    # coupled loop a FULLY converged trim on an intermediate geometry is
    # wasted work -- the geometry moves right after. Capping least_squares to
    # a few evaluations per coupled iteration lets trim and geometry converge
    # TOGETHER (measured 2026-09-09: cap 8 cut VSM calls ~30% at identical
    # outer iteration count and final state). Warm-started trims near the
    # coupled fixed point terminate inside the cap on their own tolerances,
    # so the final accepted state is a genuinely converged trim. Absent key =
    # None = historical behaviour.
    trim_max_nfev = qs_cfg.get("max_nfev")
    trim_max_nfev = int(trim_max_nfev) if trim_max_nfev else None
    if current_guess is None:
        current_guess = DEFAULT_GUESS_QS
    # Which trim solver (quasi_steady_trim.solver, 2026-09-15):
    #   'least_squares' -- the NumPy trims (scipy least_squares over the five
    #                      trim states, a full inner VSM solve per residual
    #                      evaluation). The default; every stored result.
    #   'casadi'        -- awetrim.aerodynamics.trim_casadi: trim states and
    #                      circulations as ONE Newton problem with the exact
    #                      Jacobian. Same physics, no max_nfev (a converged
    #                      trim per coupled iteration costs ~0.5 s at 45
    #                      panels, most of it the graph build).
    trim_solver_name = str(qs_cfg.get("solver", "least_squares")).lower()
    if trim_solver_name not in ("least_squares", "casadi"):
        raise ValueError(
            f"quasi_steady_trim.solver must be 'least_squares' or 'casadi', "
            f"got {trim_solver_name!r}"
        )
    # Newton tolerance of the CasADi trim on its stacked (nondimensional)
    # residual; the NumPy trims' inner loop uses the solver's allowed_error.
    trim_casadi_tolerance = float(qs_cfg.get("casadi_tolerance", 1e-8))

    # Tether in the trim (opt-in). The default trim is TETHERLESS: its residuals
    # are the tangential force balance only, so the tether is implicitly
    # massless, dragless and perfectly radial. With
    # ``config["tether"]["include_in_trim"]`` true the tether-aware trim is used
    # instead, which carries the tether's off-radial drag and weight:
    #   'rigid_lumped' -- lumped drag + half weight at the kite, 6th unknown is
    #                     the radial tension. Needs only diameter/density.
    #   'williams'     -- full distributed shape with ground closure, 6th
    #                     unknown is the tether length. Needs a WilliamsTether
    #                     on the system model, and costs a nested tether solve
    #                     per residual evaluation.
    # Default False keeps every existing aerostructural run bit-identical.
    tether_cfg = (config or {}).get("tether", {}) or {}
    include_tether_in_trim = bool(tether_cfg.get("include_in_trim", False))
    tether_model = str(tether_cfg.get("model", "rigid_lumped")).lower()

    # KCU bluff-body drag: on unless a config switches it off. Reads the
    # KCU envelope off the system model, so a system.yaml without a
    # control_system.structure block yields None = no term. ``.get`` (not
    # ``[]``) because stored snapshot configs predate the key.
    kcu_drag = (
        KcuDragModel.from_system_model(system_model)
        if bool((config or {}).get("is_with_kcu_drag", True))
        else None
    )

    if trim_solver_name == "casadi" and include_tether_in_trim and tether_model != "williams":
        raise ValueError(
            "quasi_steady_trim.solver 'casadi' covers the tetherless and the "
            f"'williams' tether trims; tether.model {tether_model!r} needs "
            "'least_squares'."
        )

    # Primary path: quasi-steady trim solve.
    t_trim_start = time.perf_counter()
    try:
        # Preserve the pre-trim body state in case we need direct-solve fallback.
        body_fallback = copy.deepcopy(body_aero)
        if trim_solver_name == "casadi":
            results, body_aero = _solve_trim_casadi(
                body_aero=body_aero,
                solver=solver,
                system_model=system_model,
                center_of_gravity=center_of_gravity,
                reference_point=reference_point,
                x_guess=current_guess,
                bounds_lower=bounds_lower,
                bounds_upper=bounds_upper,
                include_gravity=include_gravity,
                kcu_drag=kcu_drag,
                tether_model="williams" if include_tether_in_trim else None,
                gamma_seed=gamma_seed,
                tolerance=trim_casadi_tolerance,
            )
            # A trim on a search bound is reported through ``trim_on_bounds``
            # below (the coupled loop's runaway stop counts it), as with
            # least_squares, which also returns success there; only a Newton
            # failure takes the direct-solve fallback.
            results["success"] = bool(results.get("converged", False))
        elif include_tether_in_trim:
            results, body_aero = solve_vsm_qs_trim_with_williams_tether(
                kcu_drag=kcu_drag,
                body_aero=body_aero,
                center_of_gravity=center_of_gravity,
                reference_point=reference_point,
                system_model=system_model,
                x_guess=current_guess,
                solver=solver,
                bounds_lower=bounds_lower,
                bounds_upper=bounds_upper,
                include_gravity=include_gravity,
                tether_model=tether_model,
                # Warm continuation across coupling iterations: the previous
                # iteration's converged circulation seeds every inner solve of
                # this trim. In deep stall a cold-started gamma loop is noisy
                # enough that least_squares parks on a residual plateau
                # (qs_cm ~ 3e-2 observed) and burns minutes per aero call; the
                # fixed per-solve seed removes both while staying FD-smooth.
                gamma_seed=gamma_seed,
                max_nfev=trim_max_nfev,
            )
        else:
            results, body_aero = solve_quasi_steady_state(
                kcu_drag=kcu_drag,
                body_aero=body_aero,
                center_of_gravity=center_of_gravity,
                reference_point=reference_point,
                system_model=system_model,
                x_guess=current_guess,
                solver=solver,
                bounds_lower=bounds_lower,
                bounds_upper=bounds_upper,
                include_gravity=include_gravity,
                max_nfev=trim_max_nfev,
            )
        if not results.get("success", False) and trim_max_nfev is not None:
            # A capped trim reports success=False (status 0) while it is still
            # mid-convergence -- that partial step IS the point of the cap, so
            # keep it rather than discarding it through the direct-solve
            # fallback below. The real failure modes (non-finite residuals)
            # raise and take the except path regardless of the cap.
            results["trim_truncated"] = True
            results["success"] = True
        if not results.get("success", False):
            print(
                "Quasi-steady optimization did not converge to a valid trim state. "
                "Falling back to direct VSM solver.solve(body_aero)."
            )
            results = _run_vsm_direct_fallback(
                body_aero=body_fallback,
                solver=solver,
                system_model=system_model,
                current_guess=current_guess,
            )
        else:
            # A trim sitting ON a search bound is a constrained optimum, not an
            # equilibrium: the solution the force balance wants lies outside the
            # box. ``success`` is still True there, so this would otherwise pass
            # silently and every downstream result would inherit a state the
            # kite cannot actually fly.
            pinned = _warn_if_trim_on_bounds(
                results.get("opt_x"), bounds_lower, bounds_upper
            )
            # Names of the trim unknowns sitting on a bound (empty when none):
            # the coupled loop's runaway stop reads ``"kite_speed"`` here.
            results["trim_on_bounds"] = [name for name, _value, _which in pinned]
        results["trim_solver"] = trim_solver_name
    except ValueError as exc:
        # Typical case: non-finite residual in initial optimizer point.
        print(
            f"QSM failed ({type(exc).__name__}: {exc}). "
            "Falling back to direct VSM solver.solve(body_aero)."
        )
        results = _run_vsm_direct_fallback(
            body_aero=body_aero,
            solver=solver,
            system_model=system_model,
            current_guess=current_guess,
        )
    results["trim_time_s"] = time.perf_counter() - t_trim_start
    if is_with_plot:
        plot_vsm_geometry(body_aero)

    # Stall detection: compare local AoA against each panel's polar Cl-peak angle.
    alpha_at_ac = results.get("alpha_at_ac")
    if alpha_at_ac is not None and initial_polar_data:
        stall_mask = check_panel_stall(alpha_at_ac, initial_polar_data)
        results["stall_mask"] = stall_mask
        n_stalled = int(stall_mask.sum())
        if n_stalled > 0:
            stalled_indices = np.where(stall_mask)[0].tolist()
            logging.warning(
                "STALL detected: %d/%d panels stalled (indices: %s)",
                n_stalled,
                len(stall_mask),
                stalled_indices,
            )
            print(
                f"  *** STALL: {n_stalled}/{len(stall_mask)} panels stalled "
                f"(indices: {stalled_indices})"
            )
    else:
        results["stall_mask"] = None

    return np.array(results["F_distribution"]), body_aero, results


def run_frozen_geometry_alpha_sweep(
    body_aero,
    solver,
    *,
    va_magnitude: float,
    alpha_values_deg,
    side_slip_deg: float = 0.0,
    body_rates: float | np.ndarray = 0.0,
    body_axis: np.ndarray | None = None,
    reference_point: np.ndarray | None = None,
) -> list[dict]:
    """Direct VSM alpha sweep at a fixed (frozen) deformed geometry.

    For each requested angle of attack the apparent wind is imposed on the frozen
    mesh with the same inflow mechanism as the quasi-steady trim — through
    ``body_aero.va_initialize`` with sideslip, body-rate, body axis and
    reference point, rates given in world components — and a single
    ``solver.solve(body_aero)`` is run on the unchanged geometry (no re-trim).

    The freestream apparent wind is
    ``va = |va| * (cos a cos b, sin b, sin a)`` so that ``atan2(va_z, va_x) == a``
    and the sideslip is ``b``; the rigid-body rate is added as rotational inflow
    about ``body_axis`` by the VSM ``va`` setter, and ``phi_a`` is the tilt of
    the total aero force about that freestream apparent wind. Note the trim
    solvers evaluate the aero at the full course-frame transport rate
    ``Omega_C``; to reproduce a trim anchor exactly, pass the anchor's
    ``|Omega_C|`` and axis (world frame). The default ``body_axis``
    ``-radial = (0, 0, -1)`` is the radial-only reduction.

    With the defaults (``side_slip_deg=0``, ``body_rates=0``) the sweep recovers
    the pure symmetric ``C_L(alpha)`` / ``C_D(alpha)`` / ``phi_a(alpha)`` response.
    To sweep *around* a turning anchor — so the swept row at the anchor's angle
    of attack reproduces the anchor state, including its sideslip-driven
    aerodynamic roll — pass the anchor's ``side_slip_deg`` and ``body_rates``.

    Args:
        body_aero: VSM ``BodyAerodynamics`` (already updated to the deformed
            leading/trailing-edge points); its inflow is overwritten per sample.
        solver: VSM ``Solver`` whose ``solve(body_aero)`` returns a results dict
            with wing ``"cl"``/``"cd"`` and a per-panel ``"F_distribution"``.
        va_magnitude: apparent-wind speed magnitude [m/s] held constant over the
            sweep.
        alpha_values_deg: iterable of angles of attack [deg].
        side_slip_deg: sideslip angle [deg] held constant over the sweep.
        body_rates: rigid-body rate(s) [rad/s] held constant over the sweep
            (scalar or per-axis), added as rotational inflow about ``body_axis``.
        body_axis: rotation axis (or axes) for ``body_rates``, in world
            components; defaults to ``-radial = (0, 0, -1)`` (the radial-only
            reduction). Pass the anchor's full ``Omega_C`` axis to match the
            trim exactly.
        reference_point: reference point r0 for the rotational inflow
            ``v_rot(r) = omega x (r - r0)``; defaults to the origin.

    Returns:
        One dict per requested alpha with keys ``alpha`` [rad], ``cl``, ``cd``,
        ``phi_a`` [rad], ``v_a`` [m/s] and ``success``.  A failed solve yields a
        row with ``success=False`` and ``nan`` coefficients rather than raising,
        so one bad angle never aborts the sweep.
    """
    # phi_a uses the same force-vector definition as the identification dataset;
    # imported locally to avoid a module-load cycle (mirrors the local import in
    # ``plot_aero_forces_with_frames``).
    from awetrim.identification.aero_dataset import aerodynamic_roll

    # Lift/side reference axis for the aerodynamic-roll decomposition (the radial
    # axis, +z): a force purely along +z has zero aerodynamic roll. This is the
    # lift reference and is distinct from ``body_axis`` (the rigid-body rate axis).
    radial_axis = np.array([0.0, 0.0, 1.0])
    if body_axis is None:
        body_axis = -radial_axis

    beta_rad = float(np.radians(side_slip_deg))
    rows: list[dict] = []
    for alpha_deg in np.asarray(alpha_values_deg, dtype=float).ravel():
        alpha_rad = float(np.radians(alpha_deg))
        # Freestream apparent wind (before the per-panel rotational inflow), used
        # as the aerodynamic-roll reference exactly as the trim does.
        va_free = float(va_magnitude) * np.array(
            [
                np.cos(alpha_rad) * np.cos(beta_rad),
                np.sin(beta_rad),
                np.sin(alpha_rad),
            ]
        )
        try:
            # rates_in_body_frame=True: body_axis is given in world components
            # and must be used as-is. False would multiply the rate vector by
            # the body's accumulated ``geometry_rotation`` — a silent tilt for
            # any body that has been ``rotate()``d (e.g. a steering-rolled
            # turning anchor); identity-rotation bodies are unaffected.
            body_aero.va_initialize(
                Umag=float(va_magnitude),
                angle_of_attack=float(alpha_deg),
                side_slip=float(side_slip_deg),
                body_rates=body_rates,
                body_axis=body_axis,
                reference_point=reference_point,
                rates_in_body_frame=True,
            )
            results = solver.solve(body_aero)
            cl = float(np.mean(np.asarray(results["cl"], dtype=float)))
            cd = float(np.mean(np.asarray(results["cd"], dtype=float)))
            total_force = np.asarray(results["F_distribution"], dtype=float).sum(axis=0)
            phi_a = float(aerodynamic_roll(total_force, va_free, radial_axis))
            success = True
        except Exception:
            logging.exception(
                "Frozen alpha sweep: VSM solve failed at alpha=%.2f deg.", alpha_deg
            )
            cl = cd = phi_a = float("nan")
            success = False
        rows.append(
            {
                "alpha": alpha_rad,
                "cl": cl,
                "cd": cd,
                "phi_a": phi_a,
                "v_a": float(va_magnitude),
                "success": success,
            }
        )
    return rows


def plot_aero_forces_with_frames(
    struc_nodes: np.ndarray,
    kite_connectivity_arr,
    m_arr: np.ndarray,
    panel_cp_locations: np.ndarray,
    f_aero_panel: np.ndarray,
    title: str = "Aero forces, body frame and course frame",
) -> plt.Figure:
    """3-D plot of the deformed kite structure with:

    - structural connectivity (thin grey lines)
    - total aerodynamic force arrow at each panel aerodynamic centre
    - course frame triad at the origin (dashed)
    - body frame triad (principal inertia axes) at the CG (solid)

    All coordinates are in the structural/VSM frame.  The course-frame unit
    vectors in that frame are X_C=[-1,0,0], Y_C=[0,-1,0], Z_C=[0,0,1].
    """
    from awetrim.identification.rigid_body_axes import compute_rigid_body_axes

    # ── body axes ────────────────────────────────────────────────────────────
    rba = compute_rigid_body_axes(struc_nodes, m_arr)
    cg = rba.cg
    body_axes = rba.body_axes  # rows: x_K, y_K, z_K in structural frame

    # ── arrow scale: 20 % of the bounding-box diagonal ───────────────────────
    bbox_diag = np.linalg.norm(struc_nodes.max(axis=0) - struc_nodes.min(axis=0))
    frame_len = 0.20 * bbox_diag

    # Scale force arrows so the largest force == frame_len
    f_mags = np.linalg.norm(f_aero_panel, axis=1)
    f_max = f_mags.max() if f_mags.max() > 0 else 1.0
    force_scale = frame_len / f_max

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    # ── structural connectivity ───────────────────────────────────────────────
    for conn in kite_connectivity_arr:
        i, j = int(conn[0]), int(conn[1])
        xs = [struc_nodes[i, 0], struc_nodes[j, 0]]
        ys = [struc_nodes[i, 1], struc_nodes[j, 1]]
        zs = [struc_nodes[i, 2], struc_nodes[j, 2]]
        ax.plot(xs, ys, zs, color="dimgrey", linewidth=0.8, alpha=0.6)

    ax.scatter(
        struc_nodes[:, 0],
        struc_nodes[:, 1],
        struc_nodes[:, 2],
        s=10,
        c="dimgrey",
        alpha=0.5,
        zorder=2,
    )

    # ── aerodynamic force arrows at panel ACs ─────────────────────────────────
    cp = np.asarray(panel_cp_locations)
    for k in range(len(cp)):
        fvec = f_aero_panel[k] * force_scale
        ax.quiver(
            cp[k, 0],
            cp[k, 1],
            cp[k, 2],
            fvec[0],
            fvec[1],
            fvec[2],
            color="tab:orange",
            linewidth=1.2,
            arrow_length_ratio=0.2,
        )
    # single legend proxy for forces
    ax.quiver(
        [],
        [],
        [],
        [],
        [],
        [],
        color="tab:orange",
        linewidth=1.2,
        label=f"Aero force (max={f_max:.1f} N)",
    )

    # ── frame triads: shared colours (x=red, y=green, z=blue) ────────────────
    frame_colors = ["tab:red", "tab:green", "tab:blue"]

    # course frame at origin (dashed, thinner)
    course_axes = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
    course_labels = ["$X_C$", "$Y_C$", "$Z_C$"]
    for k, (col, lbl) in enumerate(zip(frame_colors, course_labels)):
        tip = course_axes[k] * frame_len * 0.6
        ax.quiver(
            0,
            0,
            0,
            tip[0],
            tip[1],
            tip[2],
            color=col,
            linewidth=1.0,
            arrow_length_ratio=0.15,
            linestyle="dashed",
            alpha=0.55,
        )
        ax.text(*(tip * 1.08), lbl, color=col, fontsize=7)
    ax.plot(
        [],
        [],
        color="grey",
        linestyle="dashed",
        linewidth=1.0,
        label="Course frame $C$",
    )

    # body frame at CG (solid, thicker)
    body_labels = ["$x_K$", "$y_K$", "$z_K$"]
    for k, (col, lbl) in enumerate(zip(frame_colors, body_labels)):
        tip = body_axes[k] * frame_len
        ax.quiver(
            *cg,
            tip[0],
            tip[1],
            tip[2],
            color=col,
            linewidth=2.2,
            arrow_length_ratio=0.15,
        )
        ax.text(*(cg + tip * 1.08), lbl, color=col, fontsize=7)
    ax.plot(
        [],
        [],
        color="grey",
        linestyle="solid",
        linewidth=2.0,
        label="Body frame $K$ (inertia)",
    )

    ax.scatter(*cg, s=80, c="black", marker="*", zorder=5, label="CG")

    # ── equal aspect ratio ────────────────────────────────────────────────────
    all_pts = np.vstack([struc_nodes, cp])
    mid = (all_pts.max(axis=0) + all_pts.min(axis=0)) / 2
    half = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2 * 1.15
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_zlim(mid[2] - half, mid[2] + half)

    ax.set_xlabel("$x_{struc}$ (m)")
    ax.set_ylabel("$y_{struc}$ (m)")
    ax.set_zlabel("$z_{struc}$ (m)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)

    return fig
