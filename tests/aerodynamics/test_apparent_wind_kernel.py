# Copyright (c) 2023-2026 Oriol Cayon, Delft University of Technology
#
# SPDX-License-Identifier: Apache-2.0

"""Apparent wind on a rigidly rotating body: ``va(r) = va_free - omega x (r - r0)``.

Covers ``awetrim.aerodynamics.apparent_wind``. Named ``..._kernel`` rather than
after the module because ``tests/`` carries no ``__init__.py``, so basenames
must be unique across it and ``tests/system/test_apparent_wind.py`` (the
point-mass ``v_wind - v_kite``, a different quantity) already holds that one.

The invariance test below is the one that matters. Two AWETrim bugs were a
missing or misplaced ``r - r0`` (the bridle segments charged the WING's inflow;
the KCU's drag and its moment arm encoded the KCU's station differently), and
both were invisible because every shipped configuration puts ``r0`` at the
geometry origin, where the error is exactly zero.
"""

from pathlib import Path

import numpy as np
import pytest
import yaml

from awetrim.aerodynamics.apparent_wind import (
    RigidInflowState,
    apparent_wind_at,
    inflow_state_of,
)
from awetrim.aerodynamics.vsm_quasi_steady import DEFAULT_STATION_KCU

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_no_rotation_returns_the_freestream():
    va = np.array([12.0, -1.0, 3.0])
    for station in ([0.0, 0.0, 0.0], [4.0, -7.0, 11.0]):
        assert np.allclose(apparent_wind_at(va, np.zeros(3), station), va)


def test_at_the_reference_point_the_rotational_term_vanishes():
    va, omega, r0 = np.array([12.0, 0.0, 1.0]), np.array([0.1, -0.3, 0.7]), [1.0, 2.0, 3.0]
    assert np.allclose(apparent_wind_at(va, omega, r0, r0), va)


def test_sign_matches_the_vsm_convention():
    """VSM ``va_distribution += -cross(body_rates, control_points - r0)``."""
    va, omega = np.array([15.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.5])
    station, r0 = np.array([0.0, 2.0, 0.0]), np.zeros(3)
    assert np.allclose(
        apparent_wind_at(va, omega, station, r0), va - np.cross(omega, station - r0)
    )
    # omega = +z, r = +y  ->  omega x r = -x  ->  va gains +x
    assert apparent_wind_at(va, omega, station, r0)[0] == pytest.approx(16.0)


def test_inflow_is_invariant_to_where_the_reference_point_is_put():
    """The physics cannot depend on the bookkeeping origin.

    A rigid field is fixed by (va at some anchor B, omega). Re-expressing it
    about any other r0 and evaluating at the same station must return the same
    vector -- which is what lets the KCU drag and the KCU moment arm be driven
    from one station symbol instead of two disagreeing conventions.
    """
    rng = np.random.default_rng(0)
    anchor = np.array([0.0, 0.0, 0.0])
    va_at_anchor = np.array([15.6, 0.3, 1.6])
    omega = np.array([0.0, 0.125, -0.714])
    station = np.array([0.0, 0.0, 0.0])  # the KCU, at the bridle point

    expected = va_at_anchor - np.cross(omega, station - anchor)
    for _ in range(8):
        r0 = rng.uniform(-12.0, 12.0, 3)
        va_free = va_at_anchor - np.cross(omega, r0 - anchor)
        assert np.allclose(apparent_wind_at(va_free, omega, station, r0), expected)


def test_ignoring_the_offset_breaks_that_invariance():
    """Guards the regression: the pre-fix KCU path charged va_free directly."""
    omega = np.array([0.0, 0.125, -0.714])
    anchor, station = np.zeros(3), np.zeros(3)
    va_at_anchor = np.array([15.6, 0.3, 1.6])
    r0 = np.array([0.0, 4.0, 10.5])  # reference point moved off the KCU
    va_free = va_at_anchor - np.cross(omega, r0 - anchor)
    assert not np.allclose(va_free, apparent_wind_at(va_free, omega, station, r0))


def test_rejects_non_three_vectors():
    with pytest.raises(ValueError, match="velocity_rotation"):
        apparent_wind_at([1.0, 0.0, 0.0], [0.0, 1.0], [0.0, 0.0, 0.0])


class _Body:
    def __init__(self, va, body_rates, reference_point):
        self.va, self.body_rates, self.reference_point = va, body_rates, reference_point


def test_inflow_state_reads_the_triple_off_a_body():
    state = inflow_state_of(_Body([1.0, 2.0, 3.0], [0.0, 0.0, 0.5], [0.0, 1.0, 0.0]))
    assert isinstance(state, RigidInflowState)
    assert np.allclose(state.velocity_apparent_free, [1.0, 2.0, 3.0])
    assert np.allclose(state.velocity_rotation, [0.0, 0.0, 0.5])
    assert np.allclose(state.reference_point, [0.0, 1.0, 0.0])


def test_inflow_state_refuses_to_invent_a_freestream_from_a_distribution():
    """A per-panel field has no single freestream; ``None`` says so."""
    distribution = np.tile([10.0, 0.0, 0.0], (7, 1))
    assert inflow_state_of(_Body(distribution, [0.0, 0.0, 0.5], np.zeros(3)))[0] is None


def test_inflow_state_defaults_a_bare_object_to_rest_at_the_origin():
    va_free, omega, r0 = inflow_state_of(object())
    assert va_free is None
    assert np.allclose(omega, 0.0) and np.allclose(r0, 0.0)


@pytest.mark.parametrize(
    "geometry_path",
    sorted((PROJECT_ROOT / "data").glob("*-KITE/struc_geometry*.yaml")),
    ids=lambda p: p.parent.name + "/" + p.name,
)
def test_every_shipped_kite_hangs_its_kcu_at_the_assumed_station(geometry_path):
    """``DEFAULT_STATION_KCU`` is only right while this holds.

    The trim reads the KCU's station from that constant, not from the geometry,
    so a kite whose bridle point is elsewhere would be charged drag at the wrong
    place AND given the wrong moment arm. Fail here rather than there.
    """
    geometry = yaml.safe_load(geometry_path.read_text(encoding="utf-8"))
    bridle_point = np.asarray(geometry.get("bridle_point_node", [0.0, 0.0, 0.0]), float)
    assert np.allclose(bridle_point, DEFAULT_STATION_KCU), (
        f"{geometry_path.name} puts the KCU at {bridle_point}, but the trim "
        f"assumes DEFAULT_STATION_KCU = {DEFAULT_STATION_KCU}"
    )
