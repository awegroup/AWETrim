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

"""Scalar metrics of a flown / optimized control history.

Pure NumPy, so the cycle scripts and their tests share one definition.
"""

import numpy as np

# Standardised steering below which a node counts as "centred": the hardware
# saturates at +-0.175 (``DEFAULT_OPTI_LIMITS["input_steering"]``), so 0.02
# is ~11% of the stroke -- small enough to keep genuine lobe-to-lobe
# reversals, large enough to ignore the trim chatter around zero.
STEERING_DEADBAND = 0.02


def steering_reversal_pairs(u_s, deadband=STEERING_DEADBAND, periodic=True):
    """Node index pairs ``(i, j)`` between which the steering reverses sign.

    Nodes with ``|u_s| <= deadband`` are dropped (they carry no sense); the
    remaining nodes are compared neighbour to neighbour and each sign change
    yields the pair of kept nodes it happens between. With ``periodic`` the
    last and first kept nodes are compared too, so a closed cycle reports the
    reversal across its seam (as the pair ``(i_last, i_first)``).
    """
    u = np.asarray(u_s, dtype=float).ravel()
    kept = np.flatnonzero(np.abs(u) > float(deadband))
    signs = np.sign(u[kept])
    if signs.size < 2:
        return []
    pairs = [
        (int(kept[k]), int(kept[k + 1]))
        for k in np.flatnonzero(signs[1:] != signs[:-1])
    ]
    if periodic and signs[0] != signs[-1]:
        pairs.append((int(kept[-1]), int(kept[0])))
    return pairs


def count_steering_reversals(u_s, deadband=STEERING_DEADBAND, periodic=True):
    """Number of sign changes of the steering outside a deadband (see
    :func:`steering_reversal_pairs`). An all-centred history has none."""
    return len(steering_reversal_pairs(u_s, deadband=deadband, periodic=periodic))
