"""Unmocked reel-out server solves on the kite's DEFAULT ROM (IPOPT, slow).

The state-machine tests stub the solve; this one runs it, so a change of the
default ROM or of the server's seed that leaves the documented flow unsolvable
fails here rather than at a client. (On 2026-10-29 the LEI-V3 default became
the simulation-only ROM, whose roll gain cannot march the built-in down-loop
seed: the seed is sized to the kite at /init, kinematics.steering_limited_eight,
and with the configured seed the cold solve needs awetrim.timeseries.roll_assist.)
"""

import numpy as np
import pytest

pytest.importorskip("pydantic")

from awetrim.server.session import ReeloutSession, SessionState

pytestmark = pytest.mark.slow

INFLOW = {"wind_speed": 8.0, "wind_direction": 270.0, "profile_law": 2, "z0": 0.03}


def test_cold_and_warm_step_converge_on_the_default_rom():
    sess = ReeloutSession()
    sess.init({"name": "slow-test", "length": 200.0, "inflow_conditions": INFLOW})

    reply = sess.step_blocking(distance_radial=200.0)
    assert sess.state == SessionState.CONVERGED
    assert reply["metrics"]["avg_power_W"] > 0.0
    # the reply is the optimum at the ROM's own roll gain
    assert sess.system_model.kite.gain_roll_scale == 1.0
    steering = np.asarray(sess.trajectory()["table"]["input_steering"], float)
    rom_limit = 0.1429  # validity.input_steering_abs_max of the LEI-V3 ROMs
    assert np.abs(steering).max() <= rom_limit + 1e-3

    # a later warm step (longer tether) converges too
    sess.step_blocking(distance_radial=220.0)
    assert sess.state == SessionState.CONVERGED
    assert sess.successful_steps == 2


@pytest.mark.parametrize("wind", [4.0, 6.0])
def test_sized_seed_trims_at_the_kite_gain_on_the_flown_root(wind):
    # 6 m/s is the case that failed: seed turns tighter than the kite holds,
    # and a march that started on the low-lift root (|u_s| ~0.5).
    sess = ReeloutSession()
    sess.init({"name": "slow-test", "length": 200.0,
               "inflow_conditions": dict(INFLOW, wind_speed=wind)})
    assert sess.init_reply()["kite_sized_seed"] is True
    assert sess._march_seed() == 1.0
    sess.system_model.kite.gain_roll_scale = 1.0

    sess.step_blocking(distance_radial=200.0)
    assert sess.state == SessionState.CONVERGED
    assert sess._seed_sized is True  # no retry from the configured seed
