# Copyright (c) 2023-2026 Oriol Cayon, Delft University of Technology
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Billow structural backend of the aero-structural coupling.

The physics of the elements themselves is covered in ``tests/structural/``.
What is tested here is the *adapter*: that the reader's arrays are translated
into the right Billow elements, that the canopy/bridle split is exact, that the
call contract the coupled driver relies on holds, and that actuation reaches
the parameters it is supposed to reach.
"""

import numpy as np
import pytest

from awetrim.aerostructural.billow import structural_billow as sb


# --------------------------------------------------------------------------
# A small synthetic wing: 3 spanwise sections x 3 chordwise nodes, one strut
# chain along the leading edge, a bridle cable and a pulley below it.
# --------------------------------------------------------------------------


@pytest.fixture
def toy_geometry():
    """Reader-shaped arrays for a minimal but complete Billow model."""
    # Wing grid: sections at y = 0, 1, 2; chord along x at 0, 0.5, 1.
    nodes, grid = [], []
    for section in range(3):
        row = []
        for chord in range(3):
            row.append(len(nodes))
            nodes.append([0.5 * chord, float(section), 0.0])
        grid.append(row)
    kcu = len(nodes)
    nodes.append([0.5, 1.0, -2.0])          # KCU, below the wing
    knot = len(nodes)
    nodes.append([0.5, 1.0, -1.0])          # bridle knot
    pulley = len(nodes)
    nodes.append([0.5, 0.5, -0.5])          # pulley sheave
    nodes = np.asarray(nodes, dtype=float)

    connectivity, rest, stiff, damp, link = [], [], [], [], []

    def add(a, b, l0, k, c, kind):
        connectivity.append([a, b])
        rest.append(l0)
        stiff.append(k)
        damp.append(c)
        link.append(kind)

    # Leading-edge tube along the grid's first column: k holds the diameter and
    # c the inflation pressure, which is how the reader hands tubes over.
    for section in range(2):
        add(grid[section][0], grid[section + 1][0], 1.0, 0.1, 0.3, "inflatable_beam")

    # Canopy springs: every grid edge, which the adapter must drop.
    canopy_edges = []
    for i in range(3):
        for j in range(2):
            canopy_edges.append((grid[i][j], grid[i][j + 1]))
    for i in range(2):
        for j in range(3):
            if j > 0:  # the j == 0 column is the tube, already added
                canopy_edges.append((grid[i][j], grid[i + 1][j]))
        for j in range(2):
            canopy_edges.append((grid[i][j], grid[i + 1][j + 1]))
            canopy_edges.append((grid[i][j + 1], grid[i + 1][j]))
    for a, b in canopy_edges:
        add(a, b, float(np.linalg.norm(nodes[b] - nodes[a])), 5000.0, 0.0,
            "noncompressive")

    # Bridle: a cable KCU -> knot, then three legs from the knot to the wing --
    # two plain cables and one pulley rope. Three non-collinear attachments are
    # the minimum that holds the wing against rigid-body rotation; with fewer,
    # the model is a mechanism and no solver can converge it.
    def length(a, b):
        return float(np.linalg.norm(nodes[b] - nodes[a]))

    cable_index = len(connectivity)
    add(kcu, knot, length(kcu, knot), 2.0e4, 0.0, "noncompressive")
    add(knot, grid[0][0], length(knot, grid[0][0]), 2.0e4, 0.0, "noncompressive")
    add(knot, grid[2][0], length(knot, grid[2][0]), 2.0e4, 0.0, "noncompressive")
    pulley_first = len(connectivity)
    rope = length(knot, pulley) + length(pulley, grid[0][2])
    add(knot, pulley, rope, 1.0e4, 0.0, "pulley")
    add(pulley, grid[0][2], rope, 1.0e4, 0.0, "pulley")

    struc_geometry = {
        "fixed_point_indices": [kcu],
        "leading_edge_tubes": {
            "headers": ["name", "ci", "cj", "diameter"],
            "data": [
                ["le_1", grid[0][0], grid[1][0], 0.1],
                ["le_2", grid[1][0], grid[2][0], 0.1],
            ],
        },
    }
    return {
        "struc_geometry": struc_geometry,
        "nodes": nodes,
        "connectivity": np.asarray(connectivity, dtype=int),
        "rest": np.asarray(rest, dtype=float),
        "stiffness": np.asarray(stiff, dtype=float),
        "damping": np.asarray(damp, dtype=float),
        "masses": np.full(len(nodes), 0.1),
        "link_types": np.asarray(link),
        "pulley_line_indices": [pulley_first, pulley_first + 1],
        # No strut chains: the leading-edge column is the only beam chain, and
        # the other two columns are pure canopy.
        "canopy_sections": [grid[0], grid[1], grid[2]],
        "strut_sections": [],
        "grid": np.asarray(grid, dtype=int),
        "kcu": kcu,
        "knot": knot,
        "cable_index": cable_index,
        "pulley_first": pulley_first,
    }


def build(toy_geometry, **settings):
    defaults = {"relax_bridles": False}
    defaults.update(settings)
    return sb.instantiate(
        {"structural_billow": defaults},
        toy_geometry["struc_geometry"],
        toy_geometry["nodes"],
        toy_geometry["connectivity"],
        toy_geometry["rest"],
        toy_geometry["stiffness"],
        toy_geometry["damping"],
        toy_geometry["masses"],
        toy_geometry["link_types"],
        toy_geometry["pulley_line_indices"],
        toy_geometry["canopy_sections"],
        toy_geometry["strut_sections"],
    )


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------


def test_canopy_grid_orders_sections_by_leading_edge(toy_geometry):
    grid = sb.canopy_grid(
        list(reversed(toy_geometry["canopy_sections"])), toy_geometry["strut_sections"]
    )
    np.testing.assert_array_equal(grid, toy_geometry["grid"])


def test_canopy_grid_rejects_a_ragged_wing():
    with pytest.raises(ValueError, match="structured grid"):
        sb.canopy_grid([[0, 1, 2], [3, 4]], [])


def test_grid_edges_covers_every_canopy_spring(toy_geometry):
    """The edge set is the classifier, so it must match the reader exactly."""
    edges = sb.grid_edges(toy_geometry["grid"])
    link = toy_geometry["link_types"]
    springs = [
        frozenset((int(a), int(b)))
        for (a, b), kind in zip(toy_geometry["connectivity"], link)
        if kind == "noncompressive"
    ]
    # The split must be exactly "on the grid or not": a spring is a canopy
    # spring precisely when both its ends are grid nodes, and every one of those
    # must be a grid EDGE. Anything else would be silently eaten by the membrane.
    on_grid = set(toy_geometry["grid"].ravel().tolist())
    canopy = [pair for pair in springs if pair in edges]
    both_ends_on_grid = [pair for pair in springs if set(pair) <= on_grid]
    assert canopy == both_ends_on_grid
    assert len(canopy) < len(springs)  # the bridle legs are not grid edges

    kcu, knot = toy_geometry["kcu"], toy_geometry["knot"]
    assert frozenset((kcu, knot)) not in edges


def test_canopy_triangles_tile_every_quad(toy_geometry):
    triangles = sb.canopy_triangles(toy_geometry["grid"])
    rows, columns = toy_geometry["grid"].shape
    assert len(triangles) == 2 * (rows - 1) * (columns - 1)
    # Each triangle has three distinct nodes and every grid node is used.
    assert all(len(set(map(int, tri))) == 3 for tri in triangles)
    assert set(triangles.ravel().tolist()) == set(toy_geometry["grid"].ravel().tolist())


def test_leading_edge_chain_walks_the_tube(toy_geometry):
    chain = sb.leading_edge_chain(toy_geometry["struc_geometry"])
    np.testing.assert_array_equal(chain, toy_geometry["grid"][:, 0])


def test_leading_edge_chain_rejects_a_broken_tube():
    geometry = {
        "leading_edge_tubes": {
            "headers": ["name", "ci", "cj", "diameter"],
            "data": [["a", 0, 1, 0.1], ["b", 5, 6, 0.1]],
        }
    }
    with pytest.raises(ValueError, match="one open chain"):
        sb.leading_edge_chain(geometry)


# --------------------------------------------------------------------------
# Element mapping
# --------------------------------------------------------------------------


def test_element_sets_split_the_reader_arrays_exactly(toy_geometry):
    structure = build(toy_geometry)
    cables = structure.model.element_set(sb.CABLES)
    pulleys = structure.model.element_set(sb.PULLEYS)
    tubes = structure.model.element_set(sb.TUBES)
    canopy = structure.model.element_set(sb.CANOPY)

    assert len(cables.connectivity) == 3          # KCU leg plus two wing legs
    assert len(pulleys.connectivity) == 1         # two arms -> one rope
    assert len(tubes.connectivity) == 2           # the leading-edge tube
    assert len(canopy.connectivity) == 8          # 2 quads x 2 triangles x 2 bays

    # Nothing is counted twice and nothing is lost: every reader element is a
    # cable, a pulley arm, a tube, or a canopy spring the membrane replaced.
    rows = np.stack(
        [structure.cable_row, structure.pulley_row, structure.tube_row]
    )
    assigned = (rows >= 0).sum(axis=0)
    assert set(np.unique(assigned)) <= {0, 1}


def test_pulley_takes_the_whole_rope_and_the_right_triplet(toy_geometry):
    structure = build(toy_geometry)
    pulleys = structure.model.element_set(sb.PULLEYS)
    first = toy_geometry["pulley_first"]
    ci, cj = toy_geometry["connectivity"][first]
    _, ck = toy_geometry["connectivity"][first + 1]
    np.testing.assert_array_equal(pulleys.connectivity[0], [ci, cj, ck])
    # kite_fem and this reader store the TOTAL rope length on each arm; Billow's
    # kernel wants that total, not a per-arm split.
    assert pulleys.params[0, 0] == pytest.approx(toy_geometry["rest"][first])


def test_pulley_arms_must_pair_up_and_share_a_node(toy_geometry):
    with pytest.raises(ValueError, match="pair up"):
        sb._pulley_triplets(toy_geometry["connectivity"], [3])
    with pytest.raises(ValueError, match="consecutive"):
        sb._pulley_triplets(toy_geometry["connectivity"], [3, 7])


def test_tube_diameter_and_pressure_come_from_k_and_c(toy_geometry):
    """The reader smuggles diameter through k_arr and pressure through c_arr."""
    structure = build(toy_geometry)
    expected = sb.InflatableTubeLaw.from_fit(0.1, 0.3)
    assert structure.tube_laws[0].bending_stiffness == pytest.approx(
        expected.bending_stiffness
    )
    assert structure.tube_laws[0].moment_max == pytest.approx(expected.moment_max)


def test_tube_stiffness_factor_scales_the_curve_without_reshaping_it(toy_geometry):
    plain = build(toy_geometry)
    scaled = build(toy_geometry, tube_stiffness_factor=0.36)
    a, b = plain.tube_laws[0], scaled.tube_laws[0]
    assert b.bending_stiffness == pytest.approx(0.36 * a.bending_stiffness)
    assert b.moment_max == pytest.approx(0.36 * a.moment_max)
    # M_max / EI_0 is the curvature scale; holding it fixed is what makes the
    # factor a magnitude knob rather than a refit.
    assert b.moment_max / b.bending_stiffness == pytest.approx(
        a.moment_max / a.bending_stiffness
    )


def test_canopy_thickness_only_splits_the_stiffness_product(toy_geometry):
    """E*t is the physical quantity; how it is split into E and t is free."""
    expected = sb.DEFAULTS["canopy_stiffness"]
    thin = build(toy_geometry, canopy_thickness=1e-4)
    thick = build(toy_geometry, canopy_thickness=1e-3)
    for structure in (thin, thick):
        params = structure.model.element_set(sb.CANOPY).params
        np.testing.assert_allclose(params[:, 1] * params[:, 2], expected)


def test_default_canopy_is_a_real_fabric_not_a_spring_net_equivalent():
    """Guards the 2026-09-11 change away from the kite_fem net's 5000 N/m.

    E*t = 5000 is E = 20 MPa at any plausible sailcloth thickness -- rubber --
    and a canopy that soft billows instead of carrying load, which bends the
    struts. The default is the 170 g/m2 polyester the geometry YAML records.
    """
    modulus = sb.DEFAULTS["canopy_stiffness"] / sb.DEFAULTS["canopy_thickness"]
    assert 1e9 < modulus < 1.5e10, f"E = {modulus:.3g} Pa is not a woven polyester"
    # Thickness consistent with 170 g/m2 of polyester, within a factor of two.
    assert 6e-5 < sb.DEFAULTS["canopy_thickness"] < 2.5e-4


def test_a_bridle_line_between_two_grid_nodes_is_rejected(toy_geometry):
    """Ambiguity here would silently delete a bridle line into the membrane."""
    geometry = dict(toy_geometry)
    grid = toy_geometry["grid"]
    geometry["connectivity"] = np.vstack(
        [toy_geometry["connectivity"], [int(grid[0][1]), int(grid[2][2])]]
    )
    geometry["rest"] = np.append(toy_geometry["rest"], 1.0)
    geometry["stiffness"] = np.append(toy_geometry["stiffness"], 1.0e4)
    geometry["damping"] = np.append(toy_geometry["damping"], 0.0)
    geometry["link_types"] = np.append(toy_geometry["link_types"], "noncompressive")
    with pytest.raises(RuntimeError, match="canopy split is ambiguous"):
        build(geometry)


# --------------------------------------------------------------------------
# Frames
# --------------------------------------------------------------------------


def test_frames_are_orthonormal_and_tangent_to_the_beam(toy_geometry):
    structure = build(toy_geometry)
    frames = structure.model.node_frames
    for frame in frames:
        np.testing.assert_allclose(frame.T @ frame, np.eye(3), atol=1e-12)
        assert np.linalg.det(frame) == pytest.approx(1.0)

    chain = toy_geometry["grid"][:, 0]
    nodes = toy_geometry["nodes"]
    tangent = nodes[chain[-1]] - nodes[chain[0]]
    tangent /= np.linalg.norm(tangent)
    for node in chain:
        assert abs(frames[node][:, 0] @ tangent) == pytest.approx(1.0, abs=1e-9)


def test_tree_transport_beats_per_chain_seeding_on_relative_rotation(toy_geometry):
    """A straight chain should carry no relative rotation at all."""
    structure = build(toy_geometry)
    frames = structure.model.node_frames
    tubes = structure.model.element_set(sb.TUBES)
    for node_a, node_b in tubes.connectivity:
        relative = frames[node_b] @ frames[node_a].T
        np.testing.assert_allclose(relative, np.eye(3), atol=1e-9)


def test_junction_frame_must_be_a_known_choice(toy_geometry):
    with pytest.raises(ValueError, match="junction_frame"):
        sb.build_frames(
            toy_geometry["nodes"],
            toy_geometry["connectivity"][:2],
            toy_geometry["struc_geometry"],
            toy_geometry["strut_sections"],
            junction_frame="whatever",
        )


# --------------------------------------------------------------------------
# The driver's call contract
# --------------------------------------------------------------------------


def wing_load(toy_geometry, total=20.0):
    """Lift on the wing, directed away from the KCU -- what tensions a bridle.

    The KCU sits below the wing, so a load on the wing in +z pulls the bridle
    taut. Loading the knot instead leaves the wing hanging on slack rope, which
    is a mechanism rather than a structure.
    """
    forces = np.zeros((len(toy_geometry["nodes"]), 3))
    wing = np.unique(toy_geometry["grid"].ravel())
    forces[wing, 2] = total / len(wing)
    return forces


def test_run_billow_returns_the_run_kite_fem_contract(toy_geometry):
    structure = build(toy_geometry)
    forces = wing_load(toy_geometry)

    returned, converged, nodes, f_int = sb.run_billow(
        structure, forces.flatten(), {"relax_bridles": False}
    )
    assert returned is structure
    assert isinstance(converged, bool)
    assert nodes.shape == (len(toy_geometry["nodes"]), 3)
    assert f_int.shape == (3 * len(toy_geometry["nodes"]),)


def test_solved_state_satisfies_force_balance_at_the_free_nodes(toy_geometry):
    """f_res = f_int + f_ext is the convention every driver residual assumes."""
    structure = build(toy_geometry)
    forces = wing_load(toy_geometry)

    _, converged, _, f_int = sb.run_billow(
        structure, forces.flatten(), {"relax_bridles": False, "force_tolerance": 1e-6}
    )
    residual = f_int.reshape(-1, 3) + forces
    free = np.ones(len(forces), bool)
    free[list(structure.fixed_node_indices)] = False
    assert converged
    assert np.linalg.norm(residual[free], axis=1).max() < 1e-5


def test_fixed_node_residual_is_zeroed_like_the_fem_backend(toy_geometry):
    structure = build(toy_geometry)
    forces = wing_load(toy_geometry)
    _, _, _, f_int = sb.run_billow(structure, forces.flatten(), {"relax_bridles": False})
    kcu = toy_geometry["kcu"]
    np.testing.assert_allclose(f_int.reshape(-1, 3)[kcu], -forces[kcu], atol=1e-12)


def test_solving_twice_reuses_the_state_rather_than_restarting(toy_geometry):
    """The second call must continue from the first, not cold-start.

    Asserted on the residual rather than on identical positions: a stiff model
    may not reach force balance inside one call's round budget, and then the
    second call legitimately moves further. What must never happen is the
    residual jumping back towards its cold-start value, which is what a lost
    state would look like.
    """
    reference = build(toy_geometry).model.nodes
    structure = build(toy_geometry)
    forces = wing_load(toy_geometry)
    settings = {"relax_bridles": False}

    _, _, first, _ = sb.run_billow(structure, forces.flatten(), settings)
    first_residual = structure.last_solution.residual_norm
    _, _, second, _ = sb.run_billow(structure, forces.flatten(), settings)
    second_residual = structure.last_solution.residual_norm

    assert second_residual <= first_residual + 1e-9
    # And it really did start from the solved shape, not the built one.
    assert np.abs(second - first).max() < np.abs(first - reference).max()


# --------------------------------------------------------------------------
# Rest lengths, actuation and stiffness
# --------------------------------------------------------------------------


def test_rest_lengths_map_back_to_the_reader_ordering(toy_geometry):
    structure = build(toy_geometry)
    lengths = sb.get_rest_lengths(structure, toy_geometry["connectivity"])
    assert lengths.shape == (len(toy_geometry["connectivity"]),)

    cable = toy_geometry["cable_index"]
    assert lengths[cable] == pytest.approx(toy_geometry["rest"][cable])
    # Canopy springs have no Billow counterpart and come back NaN, the same
    # convention the FEM backend uses for unmatched elements.
    canopy_like = np.isnan(lengths)
    expected = sum(
        1
        for pair, kind in zip(toy_geometry["connectivity"], toy_geometry["link_types"])
        if kind == "noncompressive"
        and frozenset(map(int, pair)) in sb.grid_edges(toy_geometry["grid"])
    )
    assert canopy_like.sum() == expected
    assert not canopy_like[cable]


def test_actuating_a_rest_length_changes_parameters_not_topology(toy_geometry):
    structure = build(toy_geometry)
    cable = toy_geometry["cable_index"]
    before = structure.model.layout.n_dof
    sb.update_rest_length(structure, cable, 0.2)
    assert sb.get_rest_length(structure, cable) == pytest.approx(
        toy_geometry["rest"][cable] + 0.2
    )
    assert structure.model.layout.n_dof == before
    # The solver was compiled against the topology, so it must still accept it.
    forces = np.zeros((len(toy_geometry["nodes"]), 3))
    sb.run_billow(structure, forces.flatten(), {"relax_bridles": False})


def test_actuating_either_pulley_arm_moves_the_same_rope(toy_geometry):
    structure = build(toy_geometry)
    first = toy_geometry["pulley_first"]
    sb.set_rest_length(structure, first, 2.0)
    assert sb.get_rest_length(structure, first + 1) == pytest.approx(2.0)


def test_actuating_a_tube_or_a_canopy_spring_is_refused(toy_geometry):
    structure = build(toy_geometry)
    tube = int(np.flatnonzero(structure.tube_row >= 0)[0])
    with pytest.raises(KeyError, match="not an actuatable"):
        sb.set_rest_length(structure, tube, 1.0)


def test_stiffness_round_trips_through_the_reader_ordering(toy_geometry):
    structure = build(toy_geometry)
    n = len(toy_geometry["connectivity"])
    target = np.full(n, 1234.0)
    sb.set_stiffnesses(structure, target)
    stiffness = sb.get_stiffnesses(structure, n)
    assert stiffness[toy_geometry["cable_index"]] == pytest.approx(1234.0)
    assert stiffness[toy_geometry["pulley_first"]] == pytest.approx(1234.0)
    assert np.isnan(stiffness[int(np.flatnonzero(structure.tube_row >= 0)[0])])


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


def test_unknown_config_keys_are_rejected_not_ignored():
    """Silently ignored keys are how a setting ends up doing nothing for months."""
    with pytest.raises(ValueError, match="unknown structural_billow keys"):
        sb.resolve_config({"canopy_stifness": 1.0})


def test_resolve_config_fills_defaults():
    resolved = sb.resolve_config({"canopy_stiffness": 7.0})
    assert resolved["canopy_stiffness"] == 7.0
    assert resolved["junction_frame"] == sb.DEFAULTS["junction_frame"]


def test_bridle_relaxation_removes_the_built_in_pre_stress(toy_geometry):
    """The YAML rest lengths and node positions disagree; relaxation fixes that."""
    geometry = dict(toy_geometry)
    # Shorten the bridle cable so it is 20% overstretched as built.
    rest = toy_geometry["rest"].copy()
    rest[toy_geometry["cable_index"]] *= 0.8
    geometry["rest"] = rest

    tense = build(geometry, relax_bridles=False)
    relaxed = build(geometry, relax_bridles=True)

    def peak_tension(structure):
        cables = structure.model.element_set(sb.CABLES)
        nodes = structure.model.nodes
        length = np.linalg.norm(
            nodes[cables.connectivity[:, 1]] - nodes[cables.connectivity[:, 0]], axis=1
        )
        return np.maximum(0.0, cables.params[:, 1] * (length - cables.params[:, 0])).max()

    assert peak_tension(relaxed) < 0.01 * peak_tension(tense)
    # Only bridle nodes move; the wing is held, so the canopy and tube reference
    # configurations are untouched up to the rigid re-centring.
    wing = toy_geometry["grid"].ravel()
    shift = relaxed.model.nodes[wing] - tense.model.nodes[wing]
    np.testing.assert_allclose(shift, np.tile(shift[0], (len(shift), 1)), atol=1e-8)
