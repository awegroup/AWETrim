"""``aerodynamic_vsm``: the CasADi trim inside the coupled adapter.

``_solve_trim_casadi`` must return the coupled loop's vocabulary (per-panel
loads on a rotated body, ``av_stage``), apply the attached-first rule at the
trim level, and re-use one built graph across calls on the same polars
(cached on the VSM solver object). LEI-V3, 27 panels; ~20 s.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("casadi")
pytest.importorskip("VSM")

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR / "examples" / "1_aerodynamics"))
from common import add_common_arguments, build_body, build_system_model, parsed_common  # noqa: E402

from awetrim.aerodynamics.kcu_drag import KcuDragModel  # noqa: E402
from awetrim.aerostructural import aerodynamic_vsm  # noqa: E402


@pytest.fixture(scope="module")
def lei_v3():
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    args = parser.parse_args([])
    body, _ = build_body(args)
    values = parsed_common(args)
    system_model = build_system_model(args)
    return dict(body=body, values=values, system_model=system_model, kcu=KcuDragModel.from_system_model(system_model))


def _solver(av: bool):
    from awetrim.aerodynamics.vsm_settings import make_vsm_solver

    solver = make_vsm_solver(
        gamma_loop_type="anderson", allowed_error=1e-8, is_with_artificial_viscosity=av,
        artificial_viscosity_factor=0.035, reference_point=[0.0, 0.0, 0.0],
    )
    solver._awetrim_attached_first = True
    return solver


def _call(env, solver, **extra):
    v = env["values"]
    kw = dict(
        body_aero=env["body"], solver=solver, system_model=env["system_model"],
        center_of_gravity=v["center_of_gravity"], reference_point=v["reference_point"],
        x_guess=v["x_guess"], bounds_lower=v["bounds_lower"], bounds_upper=v["bounds_upper"],
        include_gravity=False, kcu_drag=env["kcu"], tether_model=None, gamma_seed=None, tolerance=1e-8,
    )
    kw.update(extra)
    return aerodynamic_vsm._solve_trim_casadi(**kw)


def test_casadi_trim_returns_the_coupled_vocabulary(lei_v3):
    results, body = _call(lei_v3, _solver(av=False))
    n = len(lei_v3["body"].panels)
    assert results["converged"] and results["success"]
    assert np.asarray(results["F_distribution"]).shape == (n, 3)
    assert np.asarray(results["panel_cp_locations"]).shape == (n, 3)
    assert np.asarray(results["alpha_at_ac"]).ravel().shape == (n,)
    assert results["av_stage"] is None  # AV off: no attached-first stage
    assert np.asarray(body.va).shape == (3,)


def test_attached_first_at_the_trim_level(lei_v3):
    """AV on + attached_first: the unsteered LEI-V3 trims attached, the
    predictor is accepted, and the returned body carries the ORIGINAL polars
    (not the continued ones the predictor was built on)."""
    from awetrim.aerodynamics.vsm_quasi_steady import _panel_stall_onsets_rad

    results, body = _call(lei_v3, _solver(av=True))
    assert results["av_stage"] == "attached"
    onsets = _panel_stall_onsets_rad(lei_v3["body"])
    alpha = np.asarray(results["alpha_at_ac"], dtype=float).ravel()
    assert np.all(alpha <= onsets)
    for panel_out, panel_in in zip(body.panels, lei_v3["body"].panels):
        assert panel_out._panel_polar_data is panel_in._panel_polar_data


def test_attached_and_true_polar_graphs_get_separate_cache_entries(lei_v3):
    """The attached-first stage builds its graph on CONTINUED polars; a later
    true-polar solve on the same solver must get its own graph, not trip the
    cached one's polar check (the key was once taken before the swap, so every
    stalled-stage trim raised and fell back to an untrimmed direct solve)."""
    solver = _solver(av=True)
    _call(lei_v3, solver)  # attached-first: graph on the continued polars
    solver._awetrim_attached_first = False  # true polars, same solver object
    results, _ = _call(lei_v3, solver)
    assert results["converged"]
    assert len(solver._awetrim_casadi_trims) == 2


def test_graph_is_cached_on_the_solver_and_reused(lei_v3):
    solver = _solver(av=False)
    first, _ = _call(lei_v3, solver)
    cache = solver._awetrim_casadi_trims
    assert len(cache) == 1
    (system_model, trim), = cache.values()
    assert system_model is lei_v3["system_model"]
    second, _ = _call(lei_v3, solver, gamma_seed=first["gamma_distribution"], x_guess=first["opt_x"])
    assert len(cache) == 1 and next(iter(cache.values()))[1] is trim
    np.testing.assert_allclose(second["opt_x"], first["opt_x"], atol=1e-7)
    # a warm re-solve on the cached graph is far cheaper than the build
    assert second["casadi_trim"]["time_total_s"] < 0.5


def test_pre_loop_trim_config_overrides_only_its_keys():
    config = {
        "is_with_gravity": True,
        "quasi_steady_trim": {
            "max_nfev": 8,
            "yaw_bounds_deg": [-4.0, 4.0],
            "pre_loop": {"yaw_bounds_deg": [-15.0, 15.0]},
        },
    }
    pre = aerodynamic_vsm.pre_loop_trim_config(config)
    assert pre["quasi_steady_trim"]["yaw_bounds_deg"] == [-15.0, 15.0]
    assert pre["quasi_steady_trim"]["max_nfev"] == 8
    assert pre["is_with_gravity"] is True
    # the loop keeps its own bounds
    assert config["quasi_steady_trim"]["yaw_bounds_deg"] == [-4.0, 4.0]


def test_pre_loop_trim_config_without_block_is_unchanged():
    config = {"quasi_steady_trim": {"yaw_bounds_deg": [-4.0, 4.0]}}
    assert aerodynamic_vsm.pre_loop_trim_config(config) is config
    assert aerodynamic_vsm.pre_loop_trim_config(None) is None


def test_trim_just_inside_a_bound_counts_as_pinned():
    """TRF stops strictly inside the box: yaw 3.9987 on [-4, 4] (40 deg
    elevation, 2026-10-22) is pinned, an interior 3.0 is not."""
    lo = [-2.0, -4.0, -4.0, -4.0, -2.5]
    hi = [50.0, 4.0, 4.0, 4.0, 2.5]
    pinned = aerodynamic_vsm._warn_if_trim_on_bounds([20.0, 0.0, 0.0, 3.9987, 0.0], lo, hi)
    assert [(name, which) for name, _value, which in pinned] == [("yaw", "upper")]
    assert aerodynamic_vsm._warn_if_trim_on_bounds([20.0, 0.0, 0.0, 3.0, 0.0], lo, hi) == []


# --- the cold trim seed -------------------------------------------------------
class _StubKite:
    """Apparent wind of a kite at tangential speed v in a 3 m/s radial wind,
    course frame: what the seed reads off the system model."""

    speed_tangential = 0.0

    @property
    def velocity_apparent_wind(self):
        return np.array([-self.speed_tangential, 0.0, 3.0])


def _one_chord_body(te_rise):
    """A wing of three identical panels, chord 1 m along +x, its trailing
    edge ``te_rise`` m above the leading edge (VSM frame: nose down)."""
    from types import SimpleNamespace

    le, te = np.zeros(3), np.array([1.0, 0.0, te_rise])
    panel = SimpleNamespace(LE_point_1=le, LE_point_2=le, TE_point_1=te, TE_point_2=te)
    return SimpleNamespace(panels=[panel] * 3)


def test_mid_span_chord_angle_of_attack_is_signed_nose_up_positive():
    body = _one_chord_body(0.0)
    assert np.degrees(aerodynamic_vsm.mid_span_chord_angle_of_attack(body, [10.0, 0.0, 1.0])) == pytest.approx(
        np.degrees(np.arctan2(1.0, 10.0))
    )
    assert aerodynamic_vsm.mid_span_chord_angle_of_attack(body, [10.0, 0.0, -1.0]) < 0.0


def test_cold_seed_is_the_default_when_its_angle_of_attack_is_sensible():
    guess = aerodynamic_vsm.feasible_initial_guess(_one_chord_body(0.0), _StubKite(), (-2.0, 50.0))
    np.testing.assert_array_equal(guess, aerodynamic_vsm.DEFAULT_GUESS_QS)  # 30 m/s: 5.7 deg


def test_cold_seed_slows_down_until_the_wing_meets_the_wind_mid_band():
    body, kite = _one_chord_body(0.5), _StubKite()  # nose down: -20.9 deg at 30 m/s
    guess = aerodynamic_vsm.feasible_initial_guess(body, kite, (-2.0, 50.0), aoa_band_deg=(0.0, 15.0))
    np.testing.assert_array_equal(guess[1:], aerodynamic_vsm.DEFAULT_GUESS_QS[1:])
    kite.speed_tangential = guess[0]
    va = np.asarray(aerodynamic_vsm.DEFAULT_TRANSFORMATION_C_FROM_VSM) @ kite.velocity_apparent_wind
    aoa = np.degrees(aerodynamic_vsm.mid_span_chord_angle_of_attack(body, va))
    assert 6.0 < aoa < 9.0 and guess[0] < 10.0  # mid-band 7.5 deg


@pytest.mark.parametrize(
    "relative",
    ["structure.yml", "deformed_results/powered_2019/structure.yml"],
)
def test_bridle_line_specs_index_the_body_line_system(relative):
    """The node-index specs rebuild exactly the line system ``initialize`` gives
    the body: same segments, same order, the drag-equivalent diameters."""
    from awetrim.aerodynamics.line_drag import attach_bridle_lines
    from awetrim.aerostructural.coupled.awesio_structure import load_structure

    path = PROJECT_DIR / "data" / "LEI-V3-KITE" / relative
    if not path.exists():  # pragma: no cover - shipped with the repo
        pytest.skip(f"{path} not available")
    structure = load_structure(path)
    specs = aerodynamic_vsm.parse_bridle_line_specs(structure)

    class Body:
        pass

    body = Body()
    attach_bridle_lines(body, structure)
    rebuilt = Body()
    rebuilt._bridle_line_system = []
    aerodynamic_vsm.rebuild_bridle_line_system(rebuilt, structure.positions, specs)
    assert len(rebuilt._bridle_line_system) == len(body._bridle_line_system) > 0
    for got, want in zip(rebuilt._bridle_line_system, body._bridle_line_system):
        np.testing.assert_array_equal(got[0], want[0])
        np.testing.assert_array_equal(got[1], want[1])
        assert got[2] == want[2]
