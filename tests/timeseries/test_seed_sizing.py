"""The kite-sized seed of a reel-out path, on a stubbed turn-radius table."""
from types import SimpleNamespace

import numpy as np
import pytest

from awetrim.kinematics.parametrized_patterns import make_bspline_path_parameters_from_named_curve
from awetrim.kinematics.steering_limited_eight import spline_turn_geometry
from awetrim.timeseries import seed_sizing

R0 = 230.0


def _helix_config():
    path = make_bspline_path_parameters_from_named_curve(
        spline_type="periodic", M=10, r0=R0, s_init=0.0, s_final=2.0 * np.pi, n_fit=200,
        curve_type="helix", az_amp0=0.1, beta0=0.35, beta_amp0=0.1, downloops=True,
    )
    return {
        "pattern_type": "spline_periodic",
        "path_parameters": path,
        "radial_parameters": {},
        "sim_parameters": {"input_depower": 1.9},
    }


@pytest.fixture
def stub_table(monkeypatch):
    def set_radius(radius):
        table = SimpleNamespace(radius=lambda course, sign: np.full_like(course, radius, dtype=float))
        monkeypatch.setattr(seed_sizing, "steering_limited_turn_radius", lambda *a, **k: table)
        monkeypatch.setattr(seed_sizing, "Winch", lambda pattern_config: None)

    return set_radius


def test_a_wide_enough_path_is_left_alone(stub_table):
    stub_table(1.0)
    config = _helix_config()
    before = dict(config["path_parameters"])
    assert not seed_sizing.size_reelout_seed(None, config, mirror=False, self_crossings=0)
    assert config["path_parameters"] == before


@pytest.mark.slow
def test_a_tight_helix_is_widened_and_keeps_its_span_and_coefficients(stub_table):
    stub_table(25.0)
    config = _helix_config()
    margin = 1.4
    assert seed_sizing.size_reelout_seed(
        None, config, margin=margin, mirror=False, self_crossings=0
    )
    path = config["path_parameters"]
    assert path["M"] == 10
    assert path["s_final"] == pytest.approx(2.0 * np.pi)
    _, radius, _ = spline_turn_geometry(path, R0)
    assert radius.min() >= 0.99 * margin * 25.0


def test_a_fit_with_the_wrong_crossings_keeps_the_seed(stub_table, monkeypatch):
    stub_table(40.0)
    report = {"converged": True, "ratio": 1.0, "self_crossings": 1, "status": "ok"}
    monkeypatch.setattr(seed_sizing, "fit_spline_to_turn_limit", lambda *a, **k: ({"M": 16}, report))
    config = _helix_config()
    assert not seed_sizing.size_reelout_seed(None, config, mirror=False, self_crossings=0)
    assert config["path_parameters"]["M"] == 10


def test_a_config_it_cannot_size_is_skipped():
    assert not seed_sizing.size_reelout_seed(None, {"pattern_type": "x", "path_parameters": {}})
