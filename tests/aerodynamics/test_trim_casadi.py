"""CasADi trim (``awetrim.aerodynamics.trim_casadi``) against the NumPy trims.

On the LEI-V3 (27 panels, the repo data) the CasADi trim must reproduce
``solve_vsm_quasi_steady_trim`` and ``solve_vsm_qs_trim_with_williams_tether``
when it uses their bridle convention, converge in a handful of Newton
iterations, carry an exact Jacobian, and honour a pinned roll. Slow-ish
(~30 s): every case builds the graph and runs a NumPy reference trim.
"""

from __future__ import annotations

import numpy as np
import pytest

casadi = pytest.importorskip("casadi")
pytest.importorskip("VSM")

from awetrim.aerodynamics.trim_casadi import CasadiTrim, CasadiTrimOptions, solve_vsm_trim_casadi  # noqa: E402
from awetrim.aerodynamics.vsm_quasi_steady import (  # noqa: E402
    solve_vsm_qs_trim_with_williams_tether,
    solve_vsm_quasi_steady_trim,
)
from conftest import step1_trim_env  # noqa: E402


@pytest.fixture(scope="module")
def lei_v3(lei_v3_step1_kite):
    return step1_trim_env(lei_v3_step1_kite)


def _numpy_reference(env, tether: bool, aoa: bool = True, **extra):
    v = env["values"]
    kw = dict(
        body_aero=env["body"], center_of_gravity=v["center_of_gravity"], reference_point=v["reference_point"],
        system_model=env["system_model"], x_guess=v["x_guess"], gamma_loop="casadi_newton", gamma_tolerance=1e-10,
        kcu_drag=env["kcu"], include_gravity=False, is_aoa_corrected=aoa, **extra,
    )
    res, _ = (solve_vsm_qs_trim_with_williams_tether if tether else solve_vsm_quasi_steady_trim)(**kw)
    return res


def _casadi(env, tether: bool, bridle_rotates: bool = True, aoa: bool = True, **extra):
    v = env["values"]
    trim = CasadiTrim(
        env["body"], env["system_model"], v["center_of_gravity"], v["reference_point"], kcu_drag=env["kcu"],
        options=CasadiTrimOptions(tether_model="williams" if tether else None, bridle_rotates_with_body=bridle_rotates,
                                  is_aoa_corrected=aoa),
        **extra,
    )
    return trim, trim.solve(v["x_guess"])


@pytest.mark.parametrize("aoa", [True, False])
@pytest.mark.parametrize("tether", [False, True])
def test_legacy_bridle_reproduces_numpy_trim(lei_v3, tether, aoa):
    """With the NumPy trims' bridle convention the CasADi trim lands on the
    same trim state (to the NumPy loop's own 1e-10 circulation tolerance),
    the same forces and the same tether length, for both force laws. The
    reported coefficients share one normalisation factor that differs by up
    to ~3e-4 (VSM's attitude-dependent projected area), so they are held to
    that common factor rather than to an absolute tolerance."""
    ref = _numpy_reference(lei_v3, tether, aoa)
    trim, res = _casadi(lei_v3, tether, bridle_rotates=False, aoa=aoa)
    assert res["success"] and res["success_physical"]
    dx = np.abs(np.asarray(res["opt_x"]) - np.asarray(ref["opt_x"]))
    assert dx[0] < 5e-3  # m/s
    assert np.all(dx[1:4] < 1e-4)  # deg
    assert dx[4] < 1e-6  # rad/s
    for key in ("total_aero_force_vec", "tether_force"):
        if key not in res or key not in ref:
            continue
        a, b = np.asarray(res[key], float), np.asarray(ref[key], float)
        assert np.max(np.abs(a - b)) < 1e-5 * np.max(np.abs(b)), key
    ratio_cl, ratio_cd = res["cl"] / ref["cl"], res["cd"] / ref["cd"]
    assert abs(ratio_cl - ratio_cd) < 1e-5  # one normalisation, not a force difference
    assert abs(ratio_cl - 1.0) < 5e-4
    if tether:
        assert abs(res["williams_tether_length"] - ref["williams_tether_length"]) < 1e-3
        assert np.linalg.norm(res["williams_ground_residual"]) < 1e-4
        assert res["williams_positions"].shape[1] == 3
    assert res["casadi_trim"]["iterations"] <= 40
    # the solver's promise: below the wake-rebuild tolerance after the last
    # rebuild (each Newton pass itself stops at ``tolerance``)
    assert res["casadi_trim"]["residual_max"] < trim.options.wake_tolerance


def test_rigid_bridle_differs_only_slightly(lei_v3):
    """The default (bridle rotating with the wing) stays within the size of
    that modelling difference on the LEI-V3 at its ~1 deg trim pitch."""
    ref = _numpy_reference(lei_v3, False)
    _, res = _casadi(lei_v3, False, bridle_rotates=True)
    dx = np.abs(np.asarray(res["opt_x"]) - np.asarray(ref["opt_x"]))
    assert dx[0] < 0.1 and dx[2] < 0.01
    assert abs(res["cl"] - ref["cl"]) < 2e-3


@pytest.mark.parametrize("aoa", [True, False])
def test_jacobian_matches_finite_differences(lei_v3, aoa):
    trim, res = _casadi(lei_v3, True, aoa=aoa)
    x, gamma, length = res["opt_x"], res["gamma_distribution"], res["williams_tether_length"]
    aic = trim.compute_aic(x)
    jac = trim.jacobian(x, gamma, aic, length, exact=True)
    u0 = np.concatenate([x, gamma, [length]])
    eps = 1e-6
    fd = np.zeros_like(jac)
    for j in range(u0.size):
        e = np.zeros(u0.size)
        e[j] = eps
        rp, _ = trim.residual(*trim._split(u0 + e, x)[:2], aic, trim._split(u0 + e, x)[2])
        rm, _ = trim.residual(*trim._split(u0 - e, x)[:2], aic, trim._split(u0 - e, x)[2])
        fd[:, j] = (rp - rm) / (2 * eps)
    np.testing.assert_allclose(jac, fd, atol=2e-6, rtol=1e-6)
    assert res["jacobian"].shape == (5 + trim.n + 1, 5 + trim.n + 1)
    assert res["jacobian_layout"]["columns"][-1] == "tether_length"


def test_prescribed_roll_pins_the_dof_and_reports_the_reaction(lei_v3):
    ref = _numpy_reference(lei_v3, False, prescribed_roll_deg=1.0)
    trim, res = _casadi(lei_v3, False, bridle_rotates=False, prescribed_roll_deg=1.0)
    assert res["opt_x"][1] == pytest.approx(1.0)
    assert len(trim.free_x) == 4 and "cmx" not in res["jacobian_layout"]["rows"]
    dx = np.abs(np.asarray(res["opt_x"]) - np.asarray(ref["opt_x"]))
    assert dx[0] < 5e-3 and dx[2] < 1e-4 and dx[3] < 1e-4
    assert res["reaction_roll_moment_nm"] == pytest.approx(ref["reaction_roll_moment_nm"], rel=1e-2, abs=1.0)


def test_applied_moment_turns_the_kite(lei_v3):
    _, res0 = _casadi(lei_v3, False)
    _, res1 = _casadi(lei_v3, False, applied_moment_nm=np.array([150.0, 0.0, 0.0]))
    assert abs(res1["opt_x"][4]) > 1e-3 and abs(res0["opt_x"][4]) < 1e-5
    assert res1["opt_x"][1] != pytest.approx(res0["opt_x"][1], abs=1e-3)


def test_one_shot_wrapper_and_seed(lei_v3):
    v = lei_v3["values"]
    res = solve_vsm_trim_casadi(
        lei_v3["body"], v["center_of_gravity"], v["reference_point"], lei_v3["system_model"], v["x_guess"],
        kcu_drag=lei_v3["kcu"],
    )
    assert res["success"]
    trim, res2 = _casadi(lei_v3, False)
    res3 = trim.solve(res2["opt_x"], gamma_seed=res2["gamma_distribution"])
    assert res3["casadi_trim"]["iterations"] <= 2
    np.testing.assert_allclose(res3["opt_x"], res2["opt_x"], atol=1e-8)


def test_rejects_unsupported_tether_model(lei_v3):
    v = lei_v3["values"]
    with pytest.raises(ValueError, match="tether_model"):
        CasadiTrim(lei_v3["body"], lei_v3["system_model"], v["center_of_gravity"], v["reference_point"],
                   options=CasadiTrimOptions(tether_model="rigid_lumped"))


# ----------------------------------------------------------------------------
# Geometry as parameters (2026-09-15): one build, many shapes.
# ----------------------------------------------------------------------------


def _coupled_style_bodies(body, dz: float):
    """Two bodies the way the aerostructural coupling produces them:
    ``update_from_points`` with the section polars REUSED (so the panel
    tables are byte-identical and the graph can be shared), the second one
    lifted by ``dz`` -- a different geometry as far as the parameters are
    concerned."""
    import copy

    wing = body.wings[0]
    sections = wing.refine_aerodynamic_mesh()
    le = np.array([sec.LE_point for sec in sections], dtype=float)
    te = np.array([sec.TE_point for sec in sections], dtype=float)
    polars = [sec.polar_data for sec in sections]
    base = copy.deepcopy(body)
    base.update_from_points(le, te, aero_input_type="reuse_initial_polar_data", initial_polar_data=polars)
    moved = copy.deepcopy(body)
    moved.update_from_points(le + [0.0, 0.0, dz], te + [0.0, 0.0, dz], aero_input_type="reuse_initial_polar_data", initial_polar_data=polars)
    return base, moved


def test_update_geometry_matches_fresh_build(lei_v3):
    """A graph built on one shape and handed another through update_geometry
    gives the fresh build's answer on that shape, and the shape really
    changed."""
    v = lei_v3["values"]
    body_a, body_b = _coupled_style_bodies(lei_v3["body"], 0.3)
    assert CasadiTrim.geometry_signature(body_a) == CasadiTrim.geometry_signature(body_b)
    kw = dict(kcu_drag=lei_v3["kcu"], options=CasadiTrimOptions())

    base = CasadiTrim(body_a, lei_v3["system_model"], v["center_of_gravity"], v["reference_point"], **kw)
    res_base = base.solve(v["x_guess"])
    fresh = CasadiTrim(body_b, lei_v3["system_model"], v["center_of_gravity"], v["reference_point"], **kw)
    res_fresh = fresh.solve(v["x_guess"])
    base.update_geometry(body_b, v["center_of_gravity"])
    res_reused = base.solve(v["x_guess"])

    assert res_fresh["success"] and res_reused["success"]
    np.testing.assert_allclose(res_reused["opt_x"], res_fresh["opt_x"], rtol=0, atol=1e-8)
    np.testing.assert_allclose(res_reused["gamma_distribution"], res_fresh["gamma_distribution"], rtol=1e-9, atol=1e-10)
    assert abs(res_reused["cl"] - res_fresh["cl"]) < 1e-10
    # the wing moved 0.3 m along the tether axis: the moment balance shifted
    assert abs(res_fresh["opt_x"][2] - res_base["opt_x"][2]) > 1e-3


def test_update_geometry_polar_check(lei_v3, monkeypatch):
    """With the polar tables as graph inputs (linear polars, one shared alpha
    grid) other polar VALUES are numbers of the same graph -- the
    attached-first stage's continued polars swap in -- and only another
    alpha grid is refused. The embedded-table fallback refuses other polars."""
    import copy

    from VSM.core.Solver import Solver

    from awetrim.aerodynamics.vsm_quasi_steady import _attached_polars_installed

    v = lei_v3["values"]

    def make():
        return CasadiTrim(lei_v3["body"], lei_v3["system_model"], v["center_of_gravity"], v["reference_point"],
                          kcu_drag=lei_v3["kcu"])

    trim = make()
    assert trim.table_inputs
    with _attached_polars_installed(lei_v3["body"]):
        trim.update_geometry(lei_v3["body"], v["center_of_gravity"])  # same grid: accepted
        attached_key = CasadiTrim.geometry_signature(lei_v3["body"])
    assert attached_key == CasadiTrim.geometry_signature(lei_v3["body"])  # one cache entry
    # another alpha grid is another graph
    other = copy.deepcopy(lei_v3["body"])
    table = np.array(other.panels[0].panel_polar_data, dtype=float)
    table[:, 0] += 1e-3
    other.panels[0]._panel_polar_data = table  # the property's storage, as _attached_polars_installed does
    with pytest.raises(ValueError, match="grid"):
        trim.update_geometry(other, v["center_of_gravity"])
    assert CasadiTrim.geometry_signature(other) != CasadiTrim.geometry_signature(lei_v3["body"])

    # embedded tables (e.g. bspline polars): other polars need a new graph
    def refuse(self, alpha_tables):
        raise ValueError("forced: embedded tables")

    monkeypatch.setattr(Solver, "_casadi_table_input_key", refuse)
    embedded = make()
    assert not embedded.table_inputs
    with _attached_polars_installed(lei_v3["body"]):
        with pytest.raises(ValueError, match="polars"):
            embedded.update_geometry(lei_v3["body"], v["center_of_gravity"])
        attached_key = CasadiTrim.geometry_signature(lei_v3["body"], "bspline")
    assert attached_key != CasadiTrim.geometry_signature(lei_v3["body"], "bspline")


def test_solve_with_body_contract(lei_v3):
    """The aerostructural coupling's vocabulary on a body rotated to the trim
    attitude with the world-frame inflow set: per-panel loads that sum to the
    trim's own aerodynamic resultant, VSM's aerodynamic-centre alpha, the
    stalled fraction and the KCU fields."""
    from awetrim.aerodynamics.apparent_wind import inflow_state_of

    v = lei_v3["values"]
    trim = CasadiTrim(lei_v3["body"], lei_v3["system_model"], v["center_of_gravity"], v["reference_point"], kcu_drag=lei_v3["kcu"])
    res, body = trim.solve_with_body(v["x_guess"])
    n = trim.n
    forces = np.asarray(res["F_distribution"], dtype=float)
    points = np.asarray(res["panel_cp_locations"], dtype=float)
    alpha = np.asarray(res["alpha_at_ac"], dtype=float).ravel()
    assert forces.shape == (n, 3) and points.shape == (n, 3) and alpha.shape == (n,)
    assert np.all(np.isfinite(alpha))
    bridle = np.asarray(res.get("bridle_line_forces", np.zeros((0, 3))), dtype=float).reshape(-1, 3)
    total = forces.sum(axis=0) + bridle.sum(axis=0)
    # wing panels + the (rotated) bridle segments ARE the trim's resultant
    np.testing.assert_allclose(total, res["total_aero_force_vec"], rtol=1e-6, atol=1e-3)
    state = inflow_state_of(body)
    np.testing.assert_allclose(state.velocity_apparent_free, res["va_vel_world"], rtol=1e-12)
    np.testing.assert_allclose(state.velocity_rotation, res["omega_c_vsm"], rtol=0, atol=1e-12)
    assert "stalled_fraction" in res and "kcu_drag_coefficient" in res
    assert body.geometry_rotation.shape == (3, 3)


@pytest.mark.parametrize("tether", [False, True])
def test_solve_with_body_reports_like_the_numpy_trim(lei_v3, tether):
    """The aerostructural dataset reads aoa_deg as alpha_w and stores
    tether_force: both must mean what the NumPy trims report."""
    numpy_res = _numpy_reference(lei_v3, tether)
    trim, _ = _casadi(lei_v3, tether, bridle_rotates=False)
    res, _ = trim.solve_with_body(lei_v3["values"]["x_guess"])
    for key in ("aoa_deg", "side_slip_deg", "aoa_course_deg"):
        assert res[key] == pytest.approx(numpy_res[key], abs=1e-3), key
    assert res["tether_force"] == pytest.approx(numpy_res["tether_force"], rel=1e-4)


def _deformed_copy(body):
    """The body with every other section's chord 8 % longer: the panel polars
    are blended from the section polars with geometry-dependent weights, so
    the deformed shape has new polar VALUES on the same alpha grid."""
    import copy

    body_b = copy.deepcopy(body)
    sections = body_b.wings[0].sections
    le = np.array([s.LE_point for s in sections], dtype=float)
    te = np.array([s.TE_point for s in sections], dtype=float)
    te[1::2] = le[1::2] + 1.08 * (te[1::2] - le[1::2])
    body_b.update_from_points(le, te, "reuse_initial_polar_data", [s.polar_data for s in sections])
    return body_b


@pytest.mark.parametrize("aoa", [True, False])
def test_one_graph_serves_a_deformed_shape(lei_v3, aoa, monkeypatch):
    """Built once on shape A, re-used through update_geometry on a deformed
    shape B whose blended polars differ: each solve equals a fresh CasadiTrim
    built for that shape, and the reused graph is never rebuilt."""
    import copy

    v = lei_v3["values"]
    body_a = copy.deepcopy(lei_v3["body"])
    body_b = _deformed_copy(body_a)
    tables_a = np.array([p.panel_polar_data for p in body_a.panels])
    tables_b = np.array([p.panel_polar_data for p in body_b.panels])
    assert np.array_equal(tables_a[:, :, 0], tables_b[:, :, 0])  # same alpha grid
    assert np.max(np.abs(tables_a[:, :, 1:] - tables_b[:, :, 1:])) > 1e-4  # other polar values

    builds = []
    original = CasadiTrim._build_graph

    def counting(self):
        builds.append(self)
        return original(self)

    monkeypatch.setattr(CasadiTrim, "_build_graph", counting)

    def make(body):
        return CasadiTrim(body, lei_v3["system_model"], v["center_of_gravity"], v["reference_point"],
                          kcu_drag=lei_v3["kcu"], options=CasadiTrimOptions(tether_model=None, is_aoa_corrected=aoa))

    reused = make(body_a)
    assert reused.table_inputs
    res_a = reused.solve(v["x_guess"])
    reused.update_geometry(body_b, v["center_of_gravity"])
    res_b = reused.solve(v["x_guess"])
    assert len(builds) == 1  # one graph for both shapes

    for body, res in ((body_a, res_a), (body_b, res_b)):
        fresh = make(body).solve(v["x_guess"])
        np.testing.assert_allclose(res["opt_x"], fresh["opt_x"], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(np.ravel(res["gamma_distribution"]), np.ravel(fresh["gamma_distribution"]),
                                   rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(res["total_aero_force_vec"], fresh["total_aero_force_vec"], rtol=1e-10, atol=1e-8)
    assert abs(res_a["opt_x"][0] - res_b["opt_x"][0]) > 1e-3  # the shapes do trim differently


@pytest.mark.parametrize("av", [False, True])
@pytest.mark.parametrize("aoa", [True, False])
def test_table_inputs_equal_the_embedded_polars(lei_v3, aoa, av, monkeypatch):
    """The polar tables as graph inputs (VSM PR #167) give the trim of the
    graph with the tables embedded as constants, to round-off."""
    from VSM.core.Solver import Solver

    v = lei_v3["values"]

    def solve():
        trim = CasadiTrim(lei_v3["body"], lei_v3["system_model"], v["center_of_gravity"], v["reference_point"],
                          kcu_drag=lei_v3["kcu"],
                          options=CasadiTrimOptions(tether_model=None, is_aoa_corrected=aoa,
                                                    is_with_artificial_viscosity=av))
        res = trim.solve(v["x_guess"])
        return trim, res

    trim_in, res_in = solve()
    assert trim_in.table_inputs

    def refuse(self, alpha_tables):
        raise ValueError("forced: embedded tables")

    monkeypatch.setattr(Solver, "_casadi_table_input_key", refuse)
    trim_emb, res_emb = solve()
    assert not trim_emb.table_inputs
    np.testing.assert_allclose(res_in["opt_x"], res_emb["opt_x"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.ravel(res_in["gamma_distribution"]), np.ravel(res_emb["gamma_distribution"]),
                               rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(res_in["total_aero_force_vec"], res_emb["total_aero_force_vec"], rtol=1e-12, atol=1e-9)
