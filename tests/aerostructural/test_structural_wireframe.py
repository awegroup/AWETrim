# Copyright (c) 2023-2026 Oriol Cayon, Delft University of Technology
#
# SPDX-License-Identifier: Apache-2.0

"""The wireframe structural backend: Billow's line system behind the QSM contract.

These cover the adapter, not the physics -- the cable and pulley force laws are
Billow's and are tested there against closed-form statics. What is asserted here
is what the adapter is responsible for: the call contract the coupled drivers
use, the reader's line numbering surviving the trip into Billow's element
ordering, the pulley rest-length convention, and state that persists across
solves so an actuated run works.
"""

import numpy as np
import pytest

from awetrim.aerostructural.wireframe import structural_wireframe


def _make_config():
    return {
        "is_with_initial_point_velocity": False,
        "structural_pss": {
            "fixed_point_indices": [0],
            "nlp_tolerance": 1e-8,
            "nlp_max_iterations": 1000,
        },
    }


def _instantiate(nodes, conn, l0, k, linktypes, pulley_dict=None, m=1.0):
    config = _make_config()
    n = len(nodes)
    structure, *_ = structural_wireframe.instantiate(
        config,
        np.array(nodes, dtype=float),
        np.full(n, m),
        np.asarray(conn, dtype=int),
        np.asarray(l0, dtype=float),
        np.asarray(k, dtype=float),
        np.zeros(len(conn)),
        np.asarray(linktypes),
        pulley_dict or {},
    )
    return structure, config


def _residual_at_free_nodes(f_int, f_ext, fixed=(0,)):
    res = np.asarray(f_int).reshape(-1, 3) + np.asarray(f_ext).reshape(-1, 3)
    res = res.copy()
    for i in fixed:
        res[i] = 0.0
    return res


# -- the call contract the drivers rely on --------------------------------


def test_contract_and_force_balance_with_slack_line():
    # chain: 0 (fixed) --default-- 1 --noncompressive-- 2 --default-- 3,
    # plus a noncompressive 1-3 with a LONG rest length that must end slack.
    nodes = [(0, 0, 0), (0.2, 0.1, 1), (0.5, 0.6, 1.8), (-0.4, -0.1, 2.4)]
    conn = [(0, 1), (1, 2), (2, 3), (1, 3)]
    l0 = [1.0, 1.0, 0.9, 3.0]
    k = [1e4, 5e3, 5e3, 5e3]
    linktypes = ["default", "noncompressive", "default", "noncompressive"]
    structure, config = _instantiate(nodes, conn, l0, k, linktypes)

    f_ext = np.zeros((4, 3))
    f_ext[2] = [3.0, 12.0, 20.0]
    f_ext[3] = [-2.0, -5.0, 15.0]

    out = structural_wireframe.run_wireframe(
        structure, f_ext.reshape(-1), config["structural_pss"]
    )
    assert isinstance(out, tuple) and len(out) == 4
    structure_out, converged, struc_nodes, f_int = out
    assert converged
    assert struc_nodes.shape == (4, 3)
    assert np.asarray(f_int).shape == (12,)

    # equilibrium at the free nodes (anchor term is 1e-3 N/m -> sub-mN here)
    res = _residual_at_free_nodes(f_int, f_ext)
    assert np.max(np.abs(res)) < 5e-3

    # fixed node did not move; solved positions are the structure's state
    assert np.allclose(struc_nodes[0], nodes[0])
    assert np.allclose(
        np.array([p.x for p in structure_out.particles]), struc_nodes
    )

    # the long noncompressive line must end slack, and read zero tension
    assert np.linalg.norm(struc_nodes[1] - struc_nodes[3]) < 3.0
    assert structure.tensions()[3] == pytest.approx(0.0, abs=1e-9)


def test_particles_are_live_views_the_driver_can_write_back_to():
    """Aitken relaxation writes relaxed positions back; the next solve must see them."""
    nodes = [(0, 0, 0), (0.0, 0.0, 1.0)]
    structure, config = _instantiate(
        nodes, [(0, 1)], [1.0], [1e4], ["default"]
    )

    relaxed = np.array([0.1, 0.2, 1.3])
    structure.particles[1].update_pos(relaxed)
    structure.particles[1].update_vel(np.zeros(3))  # no-op, but the driver calls it

    np.testing.assert_allclose(structure.positions[1], relaxed)
    np.testing.assert_allclose(structure.particles[1].x, relaxed)


def test_kinetic_damping_is_gone_and_says_so():
    """A caller reaching for the removed relaxation must fail loudly."""
    structure, _ = _instantiate(
        [(0, 0, 0), (0, 0, 1)], [(0, 1)], [1.0], [1e4], ["default"]
    )
    with pytest.raises(NotImplementedError, match="kinetic damping was removed"):
        structure.kin_damp_sim(np.zeros(6))


# -- pulleys, and the rest-length convention -------------------------------


def _pulley_case():
    """Rope 0-1-2 over a sheave at node 1, in the geometry reader's format."""
    nodes = [(0, 0, 0), (0.3, 0.5, 1.0), (-0.2, 1.2, 0.2), (0.4, 2.0, 1.0)]
    conn = [(0, 1), (1, 2), (2, 3), (1, 3)]
    total_l0 = 2.2
    len_01 = np.linalg.norm(np.subtract(nodes[1], nodes[0]))
    len_12 = np.linalg.norm(np.subtract(nodes[2], nodes[1]))
    share = total_l0 / (len_01 + len_12)
    # The reader stores the WHOLE rope on both arm rows of l0_arr, and the
    # proportional per-arm split at index [3] of the pulley entry.
    pulley_dict = {
        "0": np.array([1, 2, share * len_12, share * len_01, 0]),
        "1": np.array([1, 0, share * len_01, share * len_12, 0]),
    }
    l0 = [total_l0, total_l0, 1.1, 1.45]
    k = [4e3, 4e3, 6e3, 5e3]
    linktypes = ["pulley", "pulley", "default", "default"]
    return nodes, conn, l0, k, linktypes, pulley_dict, total_l0


def test_pulley_pair_shares_one_stretch():
    nodes, conn, l0, k, linktypes, pulley_dict, _ = _pulley_case()
    structure, config = _instantiate(nodes, conn, l0, k, linktypes, pulley_dict)

    f_ext = np.zeros((4, 3))
    f_ext[1] = [0.0, 2.0, -8.0]
    f_ext[2] = [0.0, 5.0, -3.0]
    f_ext[3] = [0.0, 9.0, 6.0]

    _, converged, struc_nodes, f_int = structural_wireframe.run_wireframe(
        structure, f_ext.reshape(-1), config["structural_pss"]
    )
    assert converged
    res = _residual_at_free_nodes(f_int, f_ext)
    assert np.max(np.abs(res)) < 5e-3

    # One rope, one tension: a frictionless sheave cannot hold a difference.
    tensions = structure.tensions()
    assert tensions[0] == pytest.approx(tensions[1], rel=1e-9)


def test_extract_rest_length_reports_the_per_arm_split():
    """The drivers and the actuation read this; it must stay the reader's convention.

    Billow's PulleyKernel holds the WHOLE rope, so the adapter has to translate.
    Reading the total back as an arm length would put every rope into artificial
    tension.
    """
    nodes, conn, l0, k, linktypes, pulley_dict, total_l0 = _pulley_case()
    structure, _ = _instantiate(nodes, conn, l0, k, linktypes, pulley_dict)

    rest = structure.extract_rest_length
    assert rest[0] == pytest.approx(float(pulley_dict["0"][3]))
    assert rest[1] == pytest.approx(float(pulley_dict["1"][3]))
    # ... and the two arms still sum to the whole rope.
    assert rest[0] + rest[1] == pytest.approx(total_l0)
    # The non-pulley rows are untouched.
    assert rest[2] == pytest.approx(l0[2])


def test_actuating_one_pulley_arm_changes_the_whole_rope():
    nodes, conn, l0, k, linktypes, pulley_dict, total_l0 = _pulley_case()
    structure, _ = _instantiate(nodes, conn, l0, k, linktypes, pulley_dict)

    structure.update_rest_length(0, -0.1)

    assert structure.extract_rest_length[0] == pytest.approx(
        float(pulley_dict["0"][3]) - 0.1
    )
    # The Billow element holds the rope total, which moved by the same amount.
    assert structure.system.rest_length(0) == pytest.approx(total_l0 - 0.1)
    assert structure.system.rest_length(1) == pytest.approx(total_l0 - 0.1)


def test_a_slack_pulley_rope_does_not_push():
    """The regression that motivated the swap.

    The NLP inner solver this backend replaces omitted the tension-only cut on
    pulley ropes, so a slack rope PUSHED -- 9.27 N on the LEI-V3 PSM geometry,
    moving nodes up to 72 mm. PSS itself cut compression on pulleys
    (``SpringDamper.force_value``, PULLEY branch), so the NLP disagreed with the
    solver it was written to reproduce. A rope cannot push.

    Asserted on the built geometry rather than a solved one on purpose: a rope
    made longer in a structure that HANGS from it simply drops until it is taut
    again, which tests nothing.
    """
    nodes, conn, l0, k, linktypes, pulley_dict, total_l0 = _pulley_case()
    # Make the rope far longer than any path its end nodes can be at.
    for key in pulley_dict:
        pulley_dict[key][3] *= 3.0
    structure, _ = _instantiate(nodes, conn, l0, k, linktypes, pulley_dict)

    path = np.linalg.norm(
        structure.positions[1] - structure.positions[0]
    ) + np.linalg.norm(structure.positions[2] - structure.positions[1])
    assert path < structure.system.rest_length(0)  # genuinely slack

    tensions = structure.tensions()
    assert tensions[0] == pytest.approx(0.0, abs=1e-9)
    assert tensions[1] == pytest.approx(0.0, abs=1e-9)


def test_no_pulley_rope_is_ever_in_compression_after_a_solve():
    """The same property, on whatever configuration the solve actually lands on."""
    nodes, conn, l0, k, linktypes, pulley_dict, _ = _pulley_case()
    structure, config = _instantiate(nodes, conn, l0, k, linktypes, pulley_dict)

    f_ext = np.zeros((4, 3))
    f_ext[1] = [0.0, 2.0, -8.0]
    f_ext[3] = [0.0, 9.0, 6.0]
    structural_wireframe.run_wireframe(
        structure, f_ext.reshape(-1), config["structural_pss"]
    )

    pulley_lines = [i for i, name in enumerate(structure.system.line_set)
                    if name == "pulleys"]
    assert pulley_lines
    assert (structure.tensions()[pulley_lines] >= -1e-9).all()


def test_pulley_arms_must_be_consecutive():
    nodes = [(0, 0, 0), (0.3, 0.5, 1.0), (-0.2, 1.2, 0.2), (0.4, 2.0, 1.0)]
    conn = [(0, 1), (2, 3), (1, 2)]
    pulley_dict = {
        "0": np.array([1, 2, 0.5, 0.5, 0]),
        "2": np.array([1, 0, 0.5, 0.5, 0]),
    }
    with pytest.raises(ValueError, match="consecutive"):
        _instantiate(
            nodes, conn, [1.0] * 3, [1e3] * 3,
            ["pulley", "default", "pulley"], pulley_dict,
        )


def test_unpaired_pulley_arms_are_rejected():
    nodes = [(0, 0, 0), (0.3, 0.5, 1.0)]
    with pytest.raises(ValueError, match="pair up"):
        _instantiate(
            nodes, [(0, 1)], [1.0], [1e3], ["pulley"],
            {"0": np.array([1, 0, 0.5, 0.5, 0])},
        )


# -- state across solves ---------------------------------------------------


def test_stiffness_round_trips_in_the_readers_line_order():
    nodes, conn, l0, k, linktypes, pulley_dict, _ = _pulley_case()
    structure, _ = _instantiate(nodes, conn, l0, k, linktypes, pulley_dict)

    values = structural_wireframe.get_stiffnesses(structure)
    np.testing.assert_allclose(values, k)

    structural_wireframe.set_stiffnesses(structure, np.asarray(k) * 1.5)
    np.testing.assert_allclose(
        structural_wireframe.get_stiffnesses(structure), np.asarray(k) * 1.5
    )


def test_consecutive_solves_continue_from_the_last_answer():
    nodes = [(0, 0, 0), (0.2, 0.1, 1), (0.5, 0.6, 1.8)]
    structure, config = _instantiate(
        nodes, [(0, 1), (1, 2)], [1.0, 0.9], [1e4, 1e4], ["default", "default"]
    )
    f_ext = np.zeros((3, 3))
    f_ext[2] = [0.0, 5.0, -20.0]

    _, _, first, _ = structural_wireframe.run_wireframe(
        structure, f_ext.reshape(-1), config["structural_pss"]
    )
    _, converged, second, f_int = structural_wireframe.run_wireframe(
        structure, f_ext.reshape(-1), config["structural_pss"]
    )

    assert converged
    # Already at equilibrium, so the second solve only polishes: it must not
    # restart from the built geometry, which is ~1 m away.
    np.testing.assert_allclose(second, first, atol=1e-3)
    assert np.max(np.abs(_residual_at_free_nodes(f_int, f_ext))) < 5e-3


# -- the stiffness-bound helpers (pure, moved across unchanged) -------------


def test_modulus_ceiling_is_a_multiple_of_each_elements_own_stiffness():
    initial = np.array([1e3, 5e3, 2e4])
    ceiling = structural_wireframe.modulus_stiffness_ceiling(initial, 9.0)
    np.testing.assert_allclose(ceiling, initial * 9.0)


def test_adapt_stiffnesses_stiffens_only_what_is_over_the_bound():
    stiffness = np.array([1e3, 1e3, 1e3])
    elongations = np.array([0.02, 0.005, 0.03])

    updated, n_updated, max_seen, n_pinned = structural_wireframe.adapt_stiffnesses(
        stiffness, elongations, max_elongation=0.01, factor=1.5
    )

    np.testing.assert_allclose(updated, [1.5e3, 1e3, 1.5e3])
    assert n_updated == 2
    assert max_seen == pytest.approx(0.03)
    assert n_pinned == 0


def test_adapt_stiffnesses_stops_at_the_ceiling_and_counts_it():
    stiffness = np.array([1e3, 9e3])
    elongations = np.array([0.05, 0.05])
    ceiling = structural_wireframe.modulus_stiffness_ceiling([1e3, 1e3], 9.0)

    updated, n_updated, _, n_pinned = structural_wireframe.adapt_stiffnesses(
        stiffness, elongations, max_elongation=0.01, factor=1.5, max_stiffness=ceiling
    )

    np.testing.assert_allclose(updated, [1.5e3, 9e3])
    assert n_updated == 1
    assert n_pinned == 1
