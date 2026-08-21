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

"""Tests for the KCU bluff-body drag model.

Pure numpy: no VSM installation, no solver, no CasADi. They pin the handbook
tables, the two drag areas of the shipped kites, the invariances of the force
law (even in the axis, quadratic in speed) and the closed forms of the two
coefficients -- the last of which is what would fail if the axial and
crossflow pairs were ever crossed (see the module's pairing note).
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from awetrim.aerodynamics.kcu_drag import (
    CD_AXIAL,
    CD_BROADSIDE,
    FINENESS_AXIAL,
    FINENESS_BROADSIDE,
    KcuDragModel,
    cd_area_axial_kcu,
    cd_area_broadside_kcu,
    force_drag_kcu,
)

#: LEI-V3 KCU envelope (data/LEI-V3-KITE/system.yaml control_system.structure).
LEI_V3 = dict(length_kcu=1.0, diameter_kcu=0.48)
#: LEI-V9 KCU envelope.
LEI_V9 = dict(length_kcu=1.2, diameter_kcu=0.62)


def _model_v3() -> KcuDragModel:
    model = KcuDragModel.from_dimensions(**LEI_V3)
    assert model is not None
    return model


# ------------------------------------------------------------ coefficients
def test_tables_are_evaluated_at_their_own_abscissae():
    """At a tabulated L/D the interpolation returns the tabulated value."""
    diameter = 0.5
    for fineness, cd in zip(FINENESS_AXIAL[1:], CD_AXIAL[1:]):
        area = np.pi * (diameter / 2.0) ** 2
        assert cd_area_axial_kcu(fineness * diameter, diameter) == pytest.approx(
            cd * area
        )
    for fineness, cd in zip(FINENESS_BROADSIDE[:-1], CD_BROADSIDE[:-1]):
        length = fineness * diameter
        assert cd_area_broadside_kcu(length, diameter) == pytest.approx(
            cd * diameter * length
        )


def test_lei_v3_drag_areas():
    """Regression pin for the shipped LEI-V3 KCU (L/D = 2.083)."""
    assert cd_area_axial_kcu(**LEI_V3) == pytest.approx(0.1505, rel=3e-3)
    assert cd_area_broadside_kcu(**LEI_V3) == pytest.approx(0.3294, rel=3e-3)


def test_lei_v9_drag_areas():
    """Regression pin for the shipped LEI-V9 KCU (L/D = 1.935)."""
    assert cd_area_axial_kcu(**LEI_V9) == pytest.approx(0.2514, rel=3e-3)
    assert cd_area_broadside_kcu(**LEI_V9) == pytest.approx(0.5046, rel=3e-3)


def test_linear_interpolation_tracks_the_reference_spline():
    """np.interp stands in for the reference implementation's cubic spline.

    The repo's in-house pattern is np.interp; the model this one is compared
    against (awes_ekf.setup.kcu) uses scipy splrep/splev. AT THE FINENESS
    RATIOS OF THE SHIPPED KITES the two agree to a few parts in a thousand;
    across the rest of the tables the cubic undershoots the steep first
    segment, so the honest bound there is 2 %.
    """
    interpolate = pytest.importorskip("scipy.interpolate")
    spline_axial = interpolate.splrep(FINENESS_AXIAL, CD_AXIAL, s=0)
    spline_broadside = interpolate.splrep(FINENESS_BROADSIDE, CD_BROADSIDE, s=0)

    for envelope in (LEI_V3, LEI_V9):
        fineness = envelope["length_kcu"] / envelope["diameter_kcu"]
        assert float(np.interp(fineness, FINENESS_AXIAL, CD_AXIAL)) == pytest.approx(
            float(interpolate.splev(fineness, spline_axial)), rel=5e-3
        )
        assert float(
            np.interp(fineness, FINENESS_BROADSIDE, CD_BROADSIDE)
        ) == pytest.approx(
            float(interpolate.splev(fineness, spline_broadside)), rel=5e-3
        )

    # Whole interpolation range (the broadside table starts at L/D = 1; below
    # it np.interp clamps while the spline extrapolates, which
    # test_fineness_outside_the_tables_is_clamped pins instead).
    grid = np.linspace(1.0, 5.0, 200)
    assert (
        np.max(
            np.abs(
                np.interp(grid, FINENESS_AXIAL, CD_AXIAL)
                / interpolate.splev(grid, spline_axial)
                - 1.0
            )
        )
        < 2e-2
    )
    assert (
        np.max(
            np.abs(
                np.interp(grid, FINENESS_BROADSIDE, CD_BROADSIDE)
                / interpolate.splev(grid, spline_broadside)
                - 1.0
            )
        )
        < 2e-2
    )


def test_fineness_outside_the_tables_is_clamped():
    """A degenerate or huge fineness holds the endpoint, never raises."""
    assert cd_area_axial_kcu(1e-6, 1.0) == pytest.approx(
        CD_AXIAL[0] * np.pi * 0.25, rel=1e-3
    )
    assert cd_area_broadside_kcu(1e7, 1.0) == pytest.approx(CD_BROADSIDE[-1] * 1e7)


# -------------------------------------------------------------- force law
def test_pure_axial_flow_uses_the_axial_area():
    model = _model_v3()
    va = np.array([0.0, 0.0, 20.0])
    force = model.force(va, [0.0, 0.0, 1.0], 1.225)
    assert force[0] == pytest.approx(0.0)
    assert force[1] == pytest.approx(0.0)
    assert force[2] == pytest.approx(0.5 * 1.225 * 400.0 * model.cd_area_axial)


def test_pure_crossflow_uses_the_broadside_area():
    model = _model_v3()
    va = np.array([20.0, 0.0, 0.0])
    force = model.force(va, [0.0, 0.0, 1.0], 1.225)
    assert force[0] == pytest.approx(0.5 * 1.225 * 400.0 * model.cd_area_broadside)
    assert force[2] == pytest.approx(0.0)


def test_force_is_even_in_the_axis_and_scale_free():
    model = _model_v3()
    va = np.array([18.0, 3.0, 7.0])
    reference = model.force(va, [0.0, 0.0, 1.0], 1.225)
    np.testing.assert_allclose(model.force(va, [0.0, 0.0, -1.0], 1.225), reference)
    np.testing.assert_allclose(model.force(va, [0.0, 0.0, 42.0], 1.225), reference)


def test_force_is_quadratic_in_speed_and_reverses_with_the_flow():
    model = _model_v3()
    va = np.array([18.0, 3.0, 7.0])
    reference = model.force(va, [0.0, 0.0, 1.0], 1.225)
    np.testing.assert_allclose(
        model.force(2.0 * va, [0.0, 0.0, 1.0], 1.225), 4.0 * reference
    )
    np.testing.assert_allclose(
        model.force(-va, [0.0, 0.0, 1.0], 1.225), -reference, atol=1e-12
    )


def test_zero_inputs_give_zero_force():
    model = _model_v3()
    np.testing.assert_allclose(model.force(np.zeros(3), [0, 0, 1], 1.225), np.zeros(3))
    np.testing.assert_allclose(
        force_drag_kcu([10.0, 0.0, 0.0], [0, 0, 1], 1.225, 0.0, 0.0), np.zeros(3)
    )


def test_force_rejects_non_3_vectors():
    with pytest.raises(ValueError):
        force_drag_kcu([1.0, 2.0], [0, 0, 1], 1.225, 0.1, 0.3)


# ----------------------------------------------------------- coefficients
@pytest.mark.parametrize("theta_deg", [0.0, 15.0, 45.0, 75.0, 90.0])
def test_coefficient_closed_forms(theta_deg):
    """CD and |F|/qS follow the analytic split; crossing the pairs breaks this."""
    model = _model_v3()
    area_reference = 17.2119
    theta = np.deg2rad(theta_deg)
    va = 22.0 * np.array([np.sin(theta), 0.0, np.cos(theta)])  # theta from the axis
    expected_cd = (
        model.cd_area_axial * abs(np.cos(theta)) ** 3
        + model.cd_area_broadside * np.sin(theta) ** 3
    ) / area_reference
    expected_cf = (
        np.hypot(
            model.cd_area_axial * np.cos(theta) ** 2,
            model.cd_area_broadside * np.sin(theta) ** 2,
        )
        / area_reference
    )
    assert model.drag_coefficient(va, [0, 0, 1], area_reference) == pytest.approx(
        expected_cd
    )
    assert model.force_coefficient(va, [0, 0, 1], area_reference) == pytest.approx(
        expected_cf
    )


def test_representative_crosswind_state():
    """LEI-V3 at a flown reel-out state: CD_kcu of order 0.017 on 17.2 m^2."""
    model = _model_v3()
    va = np.array([20.78, 4.30, 6.12])  # a solved S1 trim's apparent wind
    assert model.drag_coefficient(va, [0, 0, 1], 17.2119) == pytest.approx(
        0.0172, rel=5e-2
    )
    assert float(np.linalg.norm(model.force(va, [0, 0, 1], 1.225))) == pytest.approx(
        91.0, rel=5e-2
    )


def test_coefficients_are_zero_without_flow_or_area():
    model = _model_v3()
    assert model.drag_coefficient(np.zeros(3), [0, 0, 1], 17.0) == 0.0
    assert model.force_coefficient([10.0, 0, 0], [0, 0, 1], 0.0) == 0.0


# ------------------------------------------------------ model construction
@pytest.mark.parametrize(
    "length,diameter", [(0.0, 0.48), (1.0, 0.0), (0.0, 0.0), (-1.0, 0.48)]
)
def test_missing_geometry_gives_no_model(length, diameter):
    assert KcuDragModel.from_dimensions(length, diameter) is None


def test_from_system_model_reads_the_kite_envelope():
    system_model = SimpleNamespace(
        kite=SimpleNamespace(mass_kcu=8.4, length_kcu=1.0, diameter_kcu=0.48)
    )
    model = KcuDragModel.from_system_model(system_model)
    assert model is not None
    assert model.cd_area_broadside == pytest.approx(0.3294, rel=3e-3)


def test_from_system_model_without_envelope_is_none():
    """A system model carrying only the KCU mass yields no drag term."""
    system_model = SimpleNamespace(kite=SimpleNamespace(mass_kcu=8.4))
    assert KcuDragModel.from_system_model(system_model) is None


def test_from_trim_result_round_trip():
    model = _model_v3()
    recovered = KcuDragModel.from_trim_result(
        {
            "kcu_cd_area_axial_m2": model.cd_area_axial,
            "kcu_cd_area_broadside_m2": model.cd_area_broadside,
        }
    )
    assert recovered is not None
    assert recovered.cd_area_axial == pytest.approx(model.cd_area_axial)
    assert recovered.cd_area_broadside == pytest.approx(model.cd_area_broadside)


def test_from_trim_result_of_a_kcu_free_trim_is_none():
    assert KcuDragModel.from_trim_result({}) is None
    assert (
        KcuDragModel.from_trim_result(
            {"kcu_cd_area_axial_m2": 0.0, "kcu_cd_area_broadside_m2": 0.0}
        )
        is None
    )


def test_public_signatures():
    assert list(inspect.signature(force_drag_kcu).parameters) == [
        "velocity_apparent",
        "axis_kcu",
        "density_air",
        "cd_area_axial",
        "cd_area_broadside",
    ]
    assert list(inspect.signature(cd_area_axial_kcu).parameters) == [
        "length_kcu",
        "diameter_kcu",
    ]
    assert list(inspect.signature(KcuDragModel.force).parameters) == [
        "self",
        "velocity_apparent",
        "axis_kcu",
        "density_air",
    ]
