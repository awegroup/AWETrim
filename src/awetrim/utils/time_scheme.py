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
"""Time differences on a path-parameter grid, the one source for every caller.

The forward march (``PhaseParameterized.run_simulation_phase``), its node
solver (``residual_solver``), the cycle NLP (``opti_phase``) and the winch
drivetrain step a path whose nodes are fixed in the path parameter s, so the
time step follows from the path rate s_dot. Two schemes (``time_scheme``):

- ``euler``: the step is ``ds / s_dot_prev`` (left rule) and a rate is the
  backward difference over it: first order.
- ``bdf2``: the step is ``2 ds / (s_dot_prev + s_dot)`` (trapezoidal) and a
  rate is the variable-step BDF2 difference over the last two steps: second
  order, and like backward Euler it damps stiff modes (the winch drum's);
  the trapezoidal rule would ring on them.

Every function takes plain numbers or CasADi expressions (``xp``: ``numpy``
or ``casadi``), so the march records exactly the differences its nodes were
solved with.
"""

import numpy as np

# Floor on the previous step in the BDF2 ratio, for a first node that has none.
H_PREV_MIN = 1e-9


def left_step(ds, s_dot_prev):
    """``ds / s_dot_prev``: the first-order step over ``ds`` in s."""
    return ds / s_dot_prev


def trapezoid_step(ds, s_dot_prev, s_dot):
    """``2 ds / (s_dot_prev + s_dot)``: the second-order step over ``ds``."""
    return 2 * ds / (s_dot_prev + s_dot)


def backward_rate(y, y_prev, step):
    """``(y - y_prev) / step``: the backward (Euler) difference."""
    return (y - y_prev) / step


def bdf2_rate(y, y_prev, y_prev2, step, step_prev, second_order=1.0, xp=np):
    """Variable-step BDF2 difference at the newest of three points.

    ``step`` is the time from ``y_prev`` to ``y``, ``step_prev`` from
    ``y_prev2`` to ``y_prev``. With ``w = step / step_prev``:

        y' = [(1 + 2w)/(1 + w) y - (1 + w) y_prev + w^2/(1 + w) y_prev2] / step

    ``second_order`` blends to the backward difference (0: a node with only
    one previous point); a number or a CasADi expression.
    """
    w = step / xp.fmax(step_prev, H_PREV_MIN)
    second = ((1 + 2 * w) / (1 + w) * y - (1 + w) * y_prev + w**2 / (1 + w) * y_prev2) / step
    euler = backward_rate(y, y_prev, step)
    return euler + second_order * (second - euler)


def trapezoid_increment(step, y_prev, y):
    """``step (y_prev + y) / 2``: the second-order increment of an integral."""
    return step * (y_prev + y) / 2
