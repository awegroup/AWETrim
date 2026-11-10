"""Per-segment aerodynamic verification for ``WilliamsTether``.

The tether resolves the apparent wind on each segment into a drag (along the
apparent wind) and a lift (perpendicular to it) using the cross-flow +
skin-friction decomposition:

    cd_t = C_N * sin^3(theta) + pi * cf * cos^3(theta)
    cl_t = C_N * sin^2(theta) * cos(theta) - pi * cf * sin(theta) * cos^2(theta)

where ``theta`` is the angle between the apparent wind and the segment axis.
For UNIT drag/lift directions, ``drag_n + lift_n`` must reproduce the physical
force exactly: a normal (cross-flow) component of magnitude ``q * C_N * sin^2``
and a tangential (skin-friction) component of magnitude ``q * pi * cf * cos^2``,
with ``q = 0.5 * rho * V^2 * L * d``. These tests pin that invariant down and
guard against the lift direction being left unnormalised.
"""

from __future__ import annotations

from dataclasses import dataclass

import casadi as ca
import numpy as np
import pytest

from awetrim.system.williams_tether import WilliamsTether


@dataclass
class _UniformWind:
    """Constant-speed wind exposing the interface the tether reads."""

    speed: float
    z0: float = 0.01

    def speed_wind_at_height(self, z):  # noqa: D401 - simple stub
        return self.speed


@dataclass
class _Env:
    rho: float
    g: float
    wind: object


def _evaluate_shape(tether, env, *, omega):
    """Evaluate the symbolic tether shape at a fully numeric configuration.

    All decision symbols are overridden with concrete values so the resulting
    CasADi graph is constant and can be evaluated with ``ca.evalf``.
    """
    length = 228.0
    shape = tether.tether_shape_symbolic(
        env=env,
        r_kite=ca.DM([100.0, 0.0, 200.0]),
        tension_kite=5000.0,
        omega=ca.DM(omega),
        tether_length=length,
        elevation_last=float(np.arctan2(200.0, 100.0)),
        azimuth_last=0.0,
    )
    keys = (
        "drag_per_segment",
        "lift_per_segment",
        "drag_per_node",
        "lift_per_node",
        "gravity_per_node",
        "velocities_apparent_wind",
        "angle_va_tether",
        "tensions",
    )
    out = {k: np.asarray(ca.evalf(shape[k])) for k in keys}
    out["length"] = length
    return out


def test_node_aero_matches_crossflow_plus_friction():
    """drag_n + lift_n reproduces the closed-form cross-flow + friction force.

    The magnitude is independent of direction, so this directly catches an
    unnormalised lift direction (which would scale the lift by sin(theta)).
    """
    cf = 0.02
    n = 10
    diameter = 0.01
    tether = WilliamsTether(
        diameter=diameter, density=950.0, n_elements=n, elastic=False, cf=cf
    )
    env = _Env(rho=1.225, g=9.81, wind=_UniformWind(speed=12.0))

    res = _evaluate_shape(tether, env, omega=[0.0, 0.1, 0.0])

    c_n = tether.drag_coefficient_tether
    l_unstrained = res["length"] / n

    for k in range(n):
        speed = float(res["velocities_apparent_wind"][k, 0])
        theta = float(res["angle_va_tether"][k, 0])
        force = res["drag_per_segment"][k] + res["lift_per_segment"][k]

        q = 0.5 * env.rho * speed**2 * l_unstrained * diameter
        expected = q * np.hypot(
            c_n * np.sin(theta) ** 2, np.pi * cf * np.cos(theta) ** 2
        )
        assert float(np.linalg.norm(force)) == pytest.approx(
            expected, rel=1e-6, abs=1e-9
        )


def test_node_aero_is_normal_to_segment_without_friction():
    """With cf = 0 the force is pure cross-flow drag: perpendicular to the
    segment axis. An unnormalised lift direction would leave a spurious
    tangential component even here."""
    n = 12
    tether = WilliamsTether(
        diameter=0.01, density=950.0, n_elements=n, elastic=False, cf=0.0
    )
    env = _Env(rho=1.225, g=9.81, wind=_UniformWind(speed=12.0))

    res = _evaluate_shape(tether, env, omega=[0.0, 0.1, 0.0])
    tensions = res["tensions"]  # row k is the tension in segment k (axis dir)

    for k in range(n):
        ej = tensions[k] / (np.linalg.norm(tensions[k]) + 1e-12)
        force = res["drag_per_segment"][k] + res["lift_per_segment"][k]
        magnitude = np.linalg.norm(force)
        # The load is evaluated once on the direction before the last correction.
        assert abs(float(np.dot(force, ej))) <= 1e-3 * (magnitude + 1e-9)


def test_every_segment_load_reaches_exactly_two_nodes():
    """Half of each segment to each end node: nothing lost, the kite holds a half."""
    n = 8
    tether = WilliamsTether(
        diameter=0.01, density=950.0, n_elements=n, elastic=False, cf=0.01
    )
    env = _Env(rho=1.225, g=9.81, wind=_UniformWind(speed=12.0))
    res = _evaluate_shape(tether, env, omega=[0.0, 0.1, 0.0])

    segments = res["drag_per_segment"].sum(axis=0)
    assert res["drag_per_node"].sum(axis=0) == pytest.approx(segments)
    assert res["drag_per_node"][n] == pytest.approx(0.5 * res["drag_per_segment"][n - 1])
    assert res["drag_per_node"][0] == pytest.approx(0.5 * res["drag_per_segment"][0])
    weight = 950.0 * np.pi * 0.01**2 / 4 * res["length"] * 9.81
    assert res["gravity_per_node"].sum(axis=0)[2] == pytest.approx(-weight)
    assert res["gravity_per_node"][1, 2] == pytest.approx(-weight / n)
    assert res["gravity_per_node"][n, 2] == pytest.approx(-weight / n / 2)


def test_kite_drag_share_converges_at_second_order():
    """The solved pull of a straight rotating tether in still air.

    The tether is pinned at the anchor, so the transverse part of the pull balances
    the moment of the drag about it: 1/4 of the dynamic pressure on the whole tether
    in the limit, 1/4 - 1/(8 n^2) with the load at each segment's centre.
    """
    length, elevation, tension = 300.0, np.deg2rad(30.0), 1.0e7
    rho, diameter, speed = 1.225, 0.01, 40.0
    r_kite = length * np.array([np.cos(elevation), 0.0, np.sin(elevation)])
    omega = ca.DM([0.0, speed / length, 0.0])  # rotates in the plane of the tether
    env = _Env(rho=rho, g=0.0, wind=None)
    for n in (2, 4, 8, 16):
        tether = WilliamsTether(
            diameter=diameter, density=1e-9, n_elements=n, elastic=False, cf=0.0
        )
        q_total = 0.5 * rho * speed**2 * length * diameter * tether.drag_coefficient_tether
        kwargs = dict(
            env=env,
            force_kite_resultant=ca.DM(tension * r_kite / length),
            omega=omega,
        )
        residual, _ = tether.residual_function(**kwargs)
        jacobian, _ = tether.residual_jacobian_function(**kwargs)
        x = np.array([elevation, 0.0, length])
        for _ in range(8):
            x = x - np.linalg.solve(
                np.asarray(jacobian(x=x, p=r_kite)["jac"]),
                np.asarray(residual(x=x, p=r_kite)["residual"]).ravel(),
            )
        share = tension * abs(np.sin(x[0] - elevation)) / q_total
        assert share == pytest.approx(0.25 - 1.0 / (8.0 * n**2), rel=2e-3)


def test_veering_wind_turns_segment_drag_sideways():
    """A veering wind pushes the segments out of the kite's vertical plane.

    The kite sits in the x-z plane with no rotation, so without veer the
    apparent wind (and the drag) has no y component; with veer the segments
    below the reference height see the wind turned anticlockwise (backed, +y)
    and those above it turned clockwise (-y).
    """
    from awetrim.environment.Wind import Wind

    tether = WilliamsTether(diameter=0.01, density=950.0, n_elements=10, elastic=False)
    straight = Wind(wind_model="uniform", direction_wind=0.0, speed_wind_ref=12.0)
    veering = Wind(
        wind_model="uniform",
        direction_wind=0.0,
        speed_wind_ref=12.0,
        veer_rate=np.radians(0.2),
        height_direction_ref=100.0,
    )
    res0 = _evaluate_shape(tether, _Env(rho=1.225, g=9.81, wind=straight), omega=[0, 0, 0])
    res1 = _evaluate_shape(tether, _Env(rho=1.225, g=9.81, wind=veering), omega=[0, 0, 0])

    assert np.allclose(res0["drag_per_segment"][:, 1], 0.0, atol=1e-9)
    y = res1["drag_per_segment"][:, 1]
    assert y[0] > 0.0  # bottom segment, below 100 m: backed wind
    assert y[-1] < 0.0  # top segment, above 100 m: veered wind
