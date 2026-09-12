import numpy as np

from awetrim.aerostructural.wireframe.actuation import (
    compute_power_tape_increment,
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
