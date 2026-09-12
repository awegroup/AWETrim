# Copyright (c) 2023-2026 Oriol Cayon, Delft University of Technology
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared coupled driver's steering actuation.

The driver dispatches over three structural backends; the steering walk goes
through one getter/setter pair for all of them. The stepping logic is tested
on a minimal stand-in for the PSS particle system, and the Billow route on the
toy wing ``test_billow`` builds.
"""

import numpy as np
import pytest

from awetrim.aerostructural.coupled import coupled_solver as driver
from test_billow import build, toy_geometry  # noqa: F401  (fixture)


class _RestLengths:
    """The two calls the driver makes on a PSS particle system."""

    def __init__(self, lengths):
        self.extract_rest_length = np.asarray(lengths, dtype=float)

    def update_rest_length(self, index, delta):
        self.extract_rest_length[index] += delta


def _step(psystem, target, step, indices=(1, 2), initial=(1.6, 1.6)):
    return driver.update_steering_tape_actuation(
        config={"structural_solver": "wireframe"},
        psystem=psystem,
        billow_structure=None,
        kite_connectivity_arr=None,
        steering_tape_indices=list(indices),
        initial_lengths_steering=list(initial),
        steering_tape_final_extension=target,
        steering_tape_extension_step=step,
    )


def test_steering_shortens_the_first_tape_and_lengthens_the_second():
    psystem = _RestLengths([9.0, 1.6, 1.6])
    half, finalized, updated = _step(psystem, 0.05, 0.0)
    assert updated and finalized
    assert half == pytest.approx(0.05)
    np.testing.assert_allclose(psystem.extract_rest_length, [9.0, 1.55, 1.65])


def test_steering_walks_in_steps_without_overshoot():
    psystem = _RestLengths([9.0, 1.6, 1.6])
    history = []
    for _ in range(4):
        half, finalized, updated = _step(psystem, 0.05, 0.02)
        history.append((round(half, 9), finalized, updated))
    assert history == [
        (0.02, False, True),
        (0.04, False, True),
        (0.05, True, True),
        (0.05, True, False),      # arrived: nothing more to do
    ]
    np.testing.assert_allclose(psystem.extract_rest_length[1:], [1.55, 1.65])


def test_steering_resumes_from_the_live_rest_lengths():
    """A walk interrupted part-way continues, rather than restarting."""
    psystem = _RestLengths([9.0, 1.57, 1.63])
    half, _, updated = _step(psystem, 0.05, 0.01)
    assert updated
    assert half == pytest.approx(0.04)


def test_negative_steering_turns_the_other_way():
    psystem = _RestLengths([9.0, 1.6, 1.6])
    _step(psystem, -0.03, 0.0)
    np.testing.assert_allclose(psystem.extract_rest_length[1:], [1.63, 1.57])


def test_remaining_drift_of_a_geometric_tail_is_exact():
    limit, ratio = 0.33, 0.6
    values = limit - 0.1 * ratio ** np.arange(8)
    assert driver.remaining_drift(values) == pytest.approx(limit - values[-1])


def test_remaining_drift_is_not_fooled_by_a_fast_transient_onto_a_slow_tail():
    """The measured steered Billow course rate, iterations 6-11.

    A two-point ratio reads 0.0007 / 0.0489 ~ 0.01 and calls it settled; the
    tail was contracting at ~0.97 with ~0.02 rad/s still to come.
    """
    course_rate = [0.0128, 0.2571, 0.3060, 0.3067, 0.3074, 0.3081]
    assert driver.remaining_drift(course_rate[:5]) == np.inf   # transient in window
    assert driver.remaining_drift(course_rate) > 0.01


def test_remaining_drift_of_a_growing_or_short_sequence_is_unsettled():
    assert driver.remaining_drift([0.0, 1e-6, 1e-5, 1e-4, 1e-3]) == np.inf
    assert driver.remaining_drift([0.1, 0.2, 0.3]) == np.inf


def test_remaining_drift_of_a_flat_sequence_is_zero():
    assert driver.remaining_drift([0.3] * 6) == 0.0


def test_billow_backend_takes_the_steering_through_its_parameters(toy_geometry):  # noqa: F811
    structure = build(toy_geometry)
    cable = toy_geometry["cable_index"]
    pulley = toy_geometry["pulley_first"]
    config = {"structural_solver": "billow"}
    initial = [
        driver._rest_length(config, index, None, structure, None)
        for index in (cable, pulley)
    ]
    half, finalized, updated = driver.update_steering_tape_actuation(
        config=config,
        psystem=None,
        billow_structure=structure,
        kite_connectivity_arr=None,
        steering_tape_indices=[cable, pulley],
        initial_lengths_steering=initial,
        steering_tape_final_extension=0.1,
        steering_tape_extension_step=0.0,
    )
    assert updated and finalized and half == pytest.approx(0.1)
    assert driver._rest_length(config, cable, None, structure, None) == (
        pytest.approx(initial[0] - 0.1)
    )
    assert driver._rest_length(config, pulley, None, structure, None) == (
        pytest.approx(initial[1] + 0.1)
    )


# --- bridle-line drag: the trim's apparent wind, not the freestream ---------
#
# The trim carries this drag inside its own balance (the VSM body is built with
# bridle lines), so the structure has to receive the same load or the
# difference is an unbalanced moment about the pinned bridle point. Evaluating
# it at `vel_app` left -326 N m of pitch and a 0.76 deg rigid swing per
# iteration on the LEI-V3 Billow case.

BRIDLE_CONFIG = {
    "is_with_aero_bridle": True,
    "rho": 1.225,
    "aerodynamic_bridle": {"cd_cable": 1.1, "cf_cable": 0.01},
}
#: one 1 m segment, 2 mm, standing normal to a flow along +x
BRIDLE_NODES = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
BRIDLE_LINES = np.array([[0, 1]])
BRIDLE_DIAMETERS = np.array([0.002])


def _bridle_force(results_aero, vel_app, config=BRIDLE_CONFIG):
    return driver._bridle_line_drag(
        config, BRIDLE_NODES, BRIDLE_LINES, BRIDLE_DIAMETERS, results_aero,
        np.asarray(vel_app, dtype=float),
    )


def test_bridle_drag_is_taken_at_the_trims_apparent_wind():
    force = _bridle_force({"va_vel_world": np.array([20.0, 0.0, 0.0])},
                          [5.0, 0.0, 0.0])
    # cross-flow segment: cd_t = cd_cable, all of it along va
    expected = 0.5 * 1.225 * 20.0**2 * 1.0 * 0.002 * 1.1
    assert force.sum(axis=0)[0] == pytest.approx(expected, rel=1e-9)
    np.testing.assert_allclose(force[0], force[1])  # split over both ends


def test_bridle_drag_scales_with_the_apparent_wind_not_the_freestream():
    slow = _bridle_force({"va_vel_world": np.array([5.0, 0.0, 0.0])},
                         [5.0, 0.0, 0.0])
    fast = _bridle_force({"va_vel_world": np.array([20.0, 0.0, 0.0])},
                         [5.0, 0.0, 0.0])
    assert np.linalg.norm(fast.sum(axis=0)) == pytest.approx(
        16.0 * np.linalg.norm(slow.sum(axis=0)), rel=1e-9
    )


@pytest.mark.parametrize(
    "results_aero",
    [{}, {"va_vel_world": np.full(3, np.nan)}, {"va_vel_world": np.zeros(3)},
     {"va_vel_world": np.zeros((4, 3))}],
)
def test_bridle_drag_falls_back_to_vel_app_without_a_usable_trim_wind(results_aero):
    fallback = _bridle_force(results_aero, [5.0, 0.0, 0.0])
    reference = _bridle_force({"va_vel_world": np.array([5.0, 0.0, 0.0])},
                              [5.0, 0.0, 0.0])
    np.testing.assert_allclose(fallback, reference)


def test_bridle_drag_is_off_when_the_config_says_so():
    # The pre-loop call ignored this gate, so a run with the bridle drag OFF
    # still got it once, on the state every later iteration starts from.
    config = {**BRIDLE_CONFIG, "is_with_aero_bridle": False}
    force = _bridle_force({"va_vel_world": np.array([20.0, 0.0, 0.0])},
                          [5.0, 0.0, 0.0], config=config)
    np.testing.assert_array_equal(force, np.zeros((2, 3)))


# --- and on the segments the structure carries it on -----------------------
#
# The VSM bakes its bridle in as coordinates, so without rebuilding it the trim
# charges the drag to the BUILT shape while the structure deforms -- 24 N m
# about the bridle point on a steered LEI-V3 state, where the deformed bridle
# is asymmetric and the built one cannot be.

class _BridleBody:
    def __init__(self, lines):
        self._bridle_line_system = lines


def test_bridle_specs_pair_reader_nodes_with_the_bodys_drag_diameters():
    # The diameters have to come from the BODY: initialize() already replaced
    # them with the drag-equivalent ones (a flat tape's projected width).
    body = _BridleBody([
        [np.zeros(3), np.ones(3), 0.004],
        [np.ones(3), 2 * np.ones(3), 0.012],
    ])
    specs = driver._bridle_line_specs_for_vsm(body, np.array([[0, 5], [5, 9]]))
    assert specs == [(0, 5, 0.004), (5, 9, 0.012)]


def test_bridle_specs_refuse_a_segment_count_the_structure_disagrees_with():
    # Pairing them up regardless would draw bridle lines between wrong nodes.
    body = _BridleBody([[np.zeros(3), np.ones(3), 0.004]])
    assert driver._bridle_line_specs_for_vsm(body, np.array([[0, 5], [5, 9]])) is None


def test_bridle_specs_are_none_when_the_body_carries_no_bridle():
    assert driver._bridle_line_specs_for_vsm(_BridleBody(None), np.array([[0, 5]])) is None
    assert driver._bridle_line_specs_for_vsm(_BridleBody([]), np.array([[0, 5]])) is None
