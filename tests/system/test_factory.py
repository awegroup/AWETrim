import math

import casadi as ca

from awetrim.system import create_system_model_from_yaml


def _write_config(path, wind):
    path.write_text(
        f"""
wing:
  mass: 15
  area: 19.75
  aerodynamics:
    model: coeffs
    params:
      CD0: 0.1
      CL0: 0.0
      angle_pitch_depower_0: -0.1
      delta_pitch_depower: -0.2
    coefficients:
      CL: []
      CD: []
      CS:
        - var: u_s
          power: 1
          coef: 0.15
kcu:
  mass: 10
tether:
  diameter: 0.01
wind:
{wind}
""",
        encoding="utf-8",
    )


def test_factory_builds_logarithmic_wind_from_yaml(tmp_path):
    config_path = tmp_path / "kite.yaml"
    _write_config(
        config_path,
        """  model: logarithmic
  z0: 0.03
  direction_wind: 0.25
  speed_wind_at_100: 10
""",
    )

    model = create_system_model_from_yaml(config_path)

    assert model.wind.wind_model == "logarithmic"
    assert model.wind.z0 == 0.03
    assert model.wind.direction_wind == 0.25
    assert math.isclose(
        float(model.wind.speed_friction),
        0.41 * 10 / math.log(100 / 0.03),
    )


def test_factory_builds_uniform_wind_from_yaml(tmp_path):
    config_path = tmp_path / "kite.yaml"
    _write_config(
        config_path,
        """  model: uniform
  direction_wind: 0
  speed_wind_ref: 8
""",
    )

    model = create_system_model_from_yaml(config_path)

    assert model.wind.wind_model == "uniform"
    assert model.wind.speed_wind_ref == 8


def test_system_model_exposes_named_expressions(tmp_path):
    config_path = tmp_path / "kite.yaml"
    _write_config(
        config_path,
        """  model: uniform
  direction_wind: 0
  speed_wind_ref: 8
""",
    )

    model = create_system_model_from_yaml(config_path)

    assert "angle_of_attack" in model.available_expressions()
    func = model.extract_function("angle_of_attack")
    assert isinstance(func, ca.Function)
