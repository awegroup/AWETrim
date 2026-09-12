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

"""Tape actuation helpers for PSS aerostructural simulations."""

from __future__ import annotations

import logging

import numpy as np


def compute_power_tape_increment(
    delta_power_tape: float,
    power_tape_final_extension: float,
    power_tape_extension_step: float,
    tol: float = 1e-9,
) -> tuple[float, bool]:
    """Return the signed rest-length increment needed to move toward target."""
    remaining = power_tape_final_extension - delta_power_tape
    if np.abs(remaining) <= tol:
        return 0.0, False
    if np.abs(power_tape_extension_step) <= tol:
        return 0.0, False

    increment = np.sign(remaining) * min(
        np.abs(power_tape_extension_step), np.abs(remaining)
    )
    return increment, True


def update_steering_tape_actuation(
    psystem,
    steering_tape_indices,
    steering_tape_extension_step,
    initial_length_steering_left,
    initial_length_steering_right,
    steering_tape_final_extension,
) -> bool:
    """Apply steering by shortening left tape and lengthening right tape."""
    if np.abs(steering_tape_extension_step) <= 1e-9:
        return False
    if np.abs(steering_tape_final_extension) <= 1e-9:
        return False

    if steering_tape_indices is None or len(steering_tape_indices) < 2:
        raise ValueError("steering_tape_indices must contain at least two entries.")

    left_index = int(steering_tape_indices[0])
    right_index = int(steering_tape_indices[1])
    desired_left = initial_length_steering_left - steering_tape_final_extension
    desired_right = initial_length_steering_right + steering_tape_final_extension

    psystem.update_rest_length(
        left_index,
        desired_left - float(psystem.extract_rest_length[left_index]),
    )
    psystem.update_rest_length(
        right_index,
        desired_right - float(psystem.extract_rest_length[right_index]),
    )

    logging.info(
        "Steering tapes updated: left %.3fm -> %.3fm, right %.3fm -> %.3fm",
        initial_length_steering_left,
        desired_left,
        initial_length_steering_right,
        desired_right,
    )
    return True


def update_power_tape_actuation(
    psystem,
    power_tape_index,
    power_tape_extension_step,
    initial_length_power_tape,
    power_tape_final_extension,
    should_apply_update,
    n_power_tape_steps,
) -> tuple[float, bool, bool]:
    """Progressively update depower tape rest length."""
    did_update = False
    current_length = float(psystem.extract_rest_length[power_tape_index])
    delta_power_tape = current_length - initial_length_power_tape
    _, should_update = compute_power_tape_increment(
        delta_power_tape=delta_power_tape,
        power_tape_final_extension=power_tape_final_extension,
        power_tape_extension_step=power_tape_extension_step,
    )
    is_actuation_finalized = not should_update

    if should_apply_update and should_update:
        increment, _ = compute_power_tape_increment(
            delta_power_tape=delta_power_tape,
            power_tape_final_extension=power_tape_final_extension,
            power_tape_extension_step=power_tape_extension_step,
        )
        psystem.update_rest_length(power_tape_index, increment)
        did_update = True
        current_length = float(psystem.extract_rest_length[power_tape_index])
        delta_power_tape = current_length - initial_length_power_tape
        _, should_update_after = compute_power_tape_increment(
            delta_power_tape=delta_power_tape,
            power_tape_final_extension=power_tape_final_extension,
            power_tape_extension_step=power_tape_extension_step,
        )
        is_actuation_finalized = not should_update_after
        logging.info(
            "||--- delta l_d: %.3fm | new l_d: %.3fm | Steps required: %s",
            delta_power_tape,
            current_length,
            n_power_tape_steps,
        )

    return delta_power_tape, is_actuation_finalized, did_update


def update_steering_tape_actuation_progressive(
    psystem,
    steering_tape_indices,
    steering_tape_extension_step,
    initial_length_steering_left,
    initial_length_steering_right,
    steering_tape_final_extension,
    should_apply_update,
) -> tuple[float, bool, bool]:
    """Progressively apply steering actuation, mirroring depower stepping."""
    if steering_tape_indices is None or len(steering_tape_indices) < 2:
        return 0.0, True, False

    target = float(steering_tape_final_extension)
    if np.abs(target) <= 1e-9:
        return 0.0, True, False

    left_idx = int(steering_tape_indices[0])
    right_idx = int(steering_tape_indices[1])
    current_left = float(psystem.extract_rest_length[left_idx])
    current_right = float(psystem.extract_rest_length[right_idx])

    current_delta_left = float(initial_length_steering_left) - current_left
    current_delta_right = current_right - float(initial_length_steering_right)
    current_delta = 0.5 * (current_delta_left + current_delta_right)

    if np.abs(target - current_delta) <= 1e-9:
        return current_delta, True, False
    if not should_apply_update:
        return current_delta, False, False

    step = float(steering_tape_extension_step)
    if np.abs(step) <= 1e-9:
        increment = target - current_delta
    else:
        remaining = target - current_delta
        increment = np.sign(remaining) * min(np.abs(step), np.abs(remaining))

    next_delta = current_delta + increment
    update_steering_tape_actuation(
        psystem=psystem,
        steering_tape_indices=steering_tape_indices,
        steering_tape_extension_step=step,
        initial_length_steering_left=initial_length_steering_left,
        initial_length_steering_right=initial_length_steering_right,
        steering_tape_final_extension=next_delta,
    )

    is_finalized = np.abs(target - next_delta) <= 1e-9
    return next_delta, is_finalized, True


__all__ = [
    "compute_power_tape_increment",
    "update_power_tape_actuation",
    "update_steering_tape_actuation",
    "update_steering_tape_actuation_progressive",
]
