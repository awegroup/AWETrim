import numpy as np

from awetrim.aerostructural.wireframe.actuation import (
    compute_power_tape_increment,
    steering_half_differences,
    update_power_tape_actuation,
    update_steering_tape_actuation,
)


class FakePssSystem:
    def __init__(self, rest_lengths):
        self._rest_lengths = np.asarray(rest_lengths, dtype=float)

    @property
    def extract_rest_length(self):
        return self._rest_lengths

    def update_rest_length(self, element_index, delta_length):
        self._rest_lengths[element_index] += delta_length


def test_compute_power_tape_increment_moves_toward_target_without_overshoot():
    increment, should_update = compute_power_tape_increment(
        delta_power_tape=0.08,
        power_tape_final_extension=0.10,
        power_tape_extension_step=0.05,
    )

    assert should_update is True
    np.testing.assert_allclose(increment, 0.02)


def test_update_power_tape_actuation_updates_single_rest_length():
    system = FakePssSystem([1.0, 2.0])

    delta, is_finalized, did_update = update_power_tape_actuation(
        system,
        power_tape_index=0,
        power_tape_extension_step=0.05,
        initial_length_power_tape=1.0,
        power_tape_final_extension=0.2,
        should_apply_update=True,
        n_power_tape_steps=4,
    )

    assert did_update is True
    assert is_finalized is False
    np.testing.assert_allclose(delta, 0.05)
    np.testing.assert_allclose(system.extract_rest_length, [1.05, 2.0])


def test_update_steering_tape_actuation_shortens_left_and_lengthens_right():
    system = FakePssSystem([1.0, 2.0, 3.0])

    did_update = update_steering_tape_actuation(
        system,
        steering_tape_indices=(1, 2),
        steering_tape_extension_step=0.05,
        initial_length_steering_left=2.0,
        initial_length_steering_right=3.0,
        steering_tape_final_extension=0.1,
    )

    assert did_update is True
    np.testing.assert_allclose(system.extract_rest_length, [1.0, 1.9, 3.1])


def test_steering_half_differences_commanded_vs_realised():
    # Node 0 is the KCU; tapes 0 (left, shortened) and 1 (right, lengthened)
    # run from the knots 1 and 2 to it. Rest lengths already walked to a
    # half-difference of 0.075 m while the knots have only moved to 0.060 m,
    # the way a slowly relaxing coupled loop leaves them.
    nodes = np.array([[0.0, 0.0, 0.0], [0.0, -0.5, 1.46], [0.0, 0.5, 1.58]])
    connectivity = [[1, 0], [2, 0]]
    rest = np.array([1.6 - 0.075, 1.6 + 0.075])
    left_len = np.linalg.norm(nodes[1]); right_len = np.linalg.norm(nodes[2])
    commanded, realised = steering_half_differences(
        rest, nodes, connectivity, [0, 1], 1.6, 1.6
    )
    np.testing.assert_allclose(commanded, 0.075)
    np.testing.assert_allclose(
        realised, 0.5 * ((1.6 - left_len) + (right_len - 1.6))
    )
    assert realised < commanded - 5e-3  # the walk is not realised yet
    # Move the knots onto the rest lengths: realised meets the command.
    nodes[1] *= (1.6 - 0.075) / left_len
    nodes[2] *= (1.6 + 0.075) / right_len
    _, realised_on = steering_half_differences(
        rest, nodes, connectivity, [0, 1], 1.6, 1.6
    )
    np.testing.assert_allclose(realised_on, 0.075, atol=1e-12)

