"""Traction-integral load transfer: aerodynamic load onto the canopy elements.

The transfer must be conservative -- the resultant the aerodynamic solver
computed has to be the resultant the structure receives -- and it must load
every canopy node through the elements around it. These are identities of the
construction, so they are checked to roundoff on small synthetic wings.
"""

import numpy as np
import pytest

from awetrim.aerostructural.coupled.aero2struc import (
    canopy_surface_coordinates,
    consistent_chordwise_density,
    map_aero_traction_to_membrane,
)

STATIONS = np.linspace(0.0, 1.0, 10)
MIRROR = np.array([1.0, -1.0, 1.0])


def flat_wing(rows=5, chordwise=(0.0, 0.3, 0.7, 1.0), half_span=2.0):
    """Planar wing: straight parallel edges, so position is affine in (s, xi)."""
    ys = np.linspace(half_span, -half_span, rows)
    nodes = np.array([[x, y, 0.0] for y in ys for x in chordwise])
    grid = np.arange(len(nodes)).reshape(rows, len(chordwise))
    return nodes, grid


def cross_triangles(grid, nodes):
    """Quad-centre triangulation; returns (nodes with centres, triangles)."""
    triangles, centres = [], []
    nxt = len(nodes)
    for i in range(grid.shape[0] - 1):
        for j in range(grid.shape[1] - 1):
            a, b = grid[i, j], grid[i, j + 1]
            c, d = grid[i + 1, j + 1], grid[i + 1, j]
            centres.append(nodes[[a, b, c, d]].mean(axis=0))
            triangles += [[a, b, nxt], [b, c, nxt], [c, d, nxt], [d, a, nxt]]
            nxt += 1
    return np.vstack([nodes, centres]), np.asarray(triangles)


def union_triangles(grid):
    """Both diagonal splits superposed: the surface is covered twice."""
    triangles = []
    for i in range(grid.shape[0] - 1):
        for j in range(grid.shape[1] - 1):
            a, b = grid[i, j], grid[i, j + 1]
            c, d = grid[i + 1, j + 1], grid[i + 1, j]
            triangles += [[a, b, c], [a, c, d], [a, b, d], [b, c, d]]
    return np.asarray(triangles)


def panel_loads(n_panels, seed=0):
    rng = np.random.default_rng(seed)
    forces = rng.normal(size=(n_panels, 3)) * 50.0
    weights = rng.uniform(0.2, 1.0, size=(n_panels, len(STATIONS)))
    return forces, weights / weights.sum(axis=1, keepdims=True)


def chord_line_moment(nodes, grid, forces, weights):
    """The moment the aerodynamic side computes: loads on each panel's chord line."""
    le, te = nodes[grid[:, 0]], nodes[grid[:, -1]]
    n_panels = len(forces)
    s_edges = np.linspace(0.0, grid.shape[0] - 1.0, n_panels + 1)
    rows = np.arange(grid.shape[0])
    moment = np.zeros(3)
    for k in range(n_panels):
        s_mid = 0.5 * (s_edges[k] + s_edges[k + 1])
        le_mid = np.array([np.interp(s_mid, rows, le[:, a]) for a in range(3)])
        te_mid = np.array([np.interp(s_mid, rows, te[:, a]) for a in range(3)])
        for xi, w in zip(STATIONS, weights[k]):
            moment += np.cross(le_mid + xi * (te_mid - le_mid), w * forces[k])
    return moment


def test_chordwise_density_keeps_force_and_centre_of_pressure():
    rng = np.random.default_rng(3)
    weights = rng.uniform(0.0, 1.0, len(STATIONS))
    weights /= weights.sum()
    q = consistent_chordwise_density(weights, STATIONS)

    # q is linear per interval, xi*q quadratic: trapezoid and Simpson are exact
    a, b = STATIONS[:-1], STATIONS[1:]
    zeroth = np.sum((b - a) * (q[:-1] + q[1:]) / 2)
    mid = 0.5 * (a + b)
    first = np.sum((b - a) / 6 * (a * q[:-1] + 4 * mid * 0.5 * (q[:-1] + q[1:]) + b * q[1:]))
    assert zeroth == pytest.approx(weights.sum(), abs=1e-13)
    assert first == pytest.approx(STATIONS @ weights, abs=1e-13)


def test_traction_on_a_flat_wing_conserves_force_and_moment():
    nodes, grid = flat_wing()
    nodes, triangles = cross_triangles(grid, nodes)
    forces, weights = panel_loads(2 * (grid.shape[0] - 1))

    nodal = map_aero_traction_to_membrane(forces, weights, STATIONS, nodes, grid, triangles)

    np.testing.assert_allclose(nodal.sum(axis=0), forces.sum(axis=0), atol=1e-11)
    # Planar, affine wing: the surface IS the chord plane, so the moment must
    # match the chord-line loads exactly -- nothing is offset.
    np.testing.assert_allclose(
        np.cross(nodes, nodal).sum(axis=0),
        chord_line_moment(nodes, grid, forces, weights),
        atol=1e-10,
    )


def test_traction_loads_every_canopy_node():
    nodes, grid = flat_wing(rows=7, chordwise=(0.0, 0.1, 0.25, 0.5, 0.8, 1.0))
    nodes, triangles = cross_triangles(grid, nodes)
    forces, weights = panel_loads(2 * (grid.shape[0] - 1), seed=1)

    nodal = map_aero_traction_to_membrane(forces, weights, STATIONS, nodes, grid, triangles)

    canopy = np.unique(triangles)
    assert np.all(np.linalg.norm(nodal[canopy], axis=1) > 0.0)
    # the quad-centre nodes above all: they are the ones point loads missed
    centres = np.arange(grid.size, len(nodes))
    assert np.all(np.linalg.norm(nodal[centres], axis=1) > 0.0)


def test_traction_force_is_exact_on_a_curved_wing():
    nodes, grid = flat_wing()
    rng = np.random.default_rng(7)
    nodes[:, 2] += 0.15 * np.sin(np.pi * nodes[:, 0]) + 0.02 * rng.normal(size=len(nodes))
    nodes, triangles = cross_triangles(grid, nodes)
    forces, weights = panel_loads(2 * (grid.shape[0] - 1), seed=2)

    nodal = map_aero_traction_to_membrane(forces, weights, STATIONS, nodes, grid, triangles)

    np.testing.assert_allclose(nodal.sum(axis=0), forces.sum(axis=0), atol=1e-11)


def test_traction_is_mirror_symmetric():
    nodes, grid = flat_wing(rows=6)
    nodes[:, 2] = 0.1 * nodes[:, 1] ** 2              # symmetric anhedral
    nodes, triangles = cross_triangles(grid, nodes)
    n_panels = 2 * (grid.shape[0] - 1)
    forces, weights = panel_loads(n_panels, seed=4)
    # a symmetric load: panel k and its mirror partner carry mirrored forces
    half = n_panels // 2
    forces[half:] = (forces[:half] * MIRROR)[::-1]
    weights[half:] = weights[:half][::-1]

    nodal = map_aero_traction_to_membrane(forces, weights, STATIONS, nodes, grid, triangles)

    reflected = nodes * MIRROR
    partner = np.array([np.argmin(np.linalg.norm(nodes - r, axis=1)) for r in reflected])
    np.testing.assert_allclose(nodal, nodal[partner] * MIRROR, atol=1e-11)


def test_superposed_sheets_are_not_double_counted():
    nodes, grid = flat_wing()
    forces, weights = panel_loads(2 * (grid.shape[0] - 1), seed=5)

    nodal = map_aero_traction_to_membrane(
        forces, weights, STATIONS, nodes, grid, union_triangles(grid)
    )

    np.testing.assert_allclose(nodal.sum(axis=0), forces.sum(axis=0), atol=1e-11)


def test_quad_centres_sit_at_the_middle_of_their_quad():
    nodes, grid = flat_wing()
    nodes, triangles = cross_triangles(grid, nodes)

    coordinates = canopy_surface_coordinates(nodes, grid, triangles)

    first_centre = coordinates[grid.size]
    assert first_centre[0] == pytest.approx(0.5)
    assert first_centre[1] == pytest.approx(0.5 * (0.0 + 0.3))


def test_curled_trailing_edge_is_loaded_exactly():
    """A slack trailing edge curls aft; the transfer must stay conservative.

    Projected on the curled shape's chord, the fabric ahead of the trailing
    edge sits aft of it. Unrepaired, that turns elements inside out in the
    surface coordinates and part of the load is counted twice; the rows are
    made strictly increasing, so the curled canopy is loaded exactly, every
    node included.
    """
    nodes, grid = flat_wing(rows=6, chordwise=(0.0, 0.3, 0.7, 0.9, 1.0))
    nodes, triangles = cross_triangles(grid, nodes)
    forces, weights = panel_loads(2 * (grid.shape[0] - 1), seed=6)

    curled = nodes.copy()
    curled[grid[2, -2]] += [0.25, 0.0, 0.05]      # 0.9 c pushed aft of the TE

    nodal = map_aero_traction_to_membrane(forces, weights, STATIONS, curled, grid, triangles)

    np.testing.assert_allclose(nodal.sum(axis=0), forces.sum(axis=0), atol=1e-11)
    assert np.all(np.linalg.norm(nodal[np.unique(triangles)], axis=1) > 0.0)


def test_order_repair_moves_only_the_violating_node():
    nodes, grid = flat_wing(rows=6, chordwise=(0.0, 0.3, 0.7, 0.9, 1.0))
    nodes, triangles = cross_triangles(grid, nodes)
    curled = nodes.copy()
    curled[grid[2, -2]] += [0.25, 0.0, 0.05]

    before = canopy_surface_coordinates(nodes, grid, triangles)
    after = canopy_surface_coordinates(curled, grid, triangles)

    moved = np.flatnonzero(np.abs(after - before).max(axis=1) > 1e-12)
    folded = grid[2, -2]
    # the curled node, and the quad centres that average it -- nothing else
    around = np.unique(triangles[(triangles == folded).any(axis=1)])
    centres = around[around >= grid.size]
    np.testing.assert_array_equal(moved, np.sort(np.r_[folded, centres]))
    assert before[folded, 1] < after[folded, 1] < 1.0     # clamped ahead of the TE
    rows = after[grid][:, :, 1]
    assert np.all(np.diff(rows, axis=1) > 0.0)


def test_a_broken_tiling_is_refused():
    nodes, grid = flat_wing()
    nodes, triangles = cross_triangles(grid, nodes)
    forces, weights = panel_loads(2 * (grid.shape[0] - 1), seed=7)
    folded = canopy_surface_coordinates(nodes, grid, triangles)
    folded[grid[1, 1], 1] = 0.95                  # past its aft neighbour at 0.7

    with pytest.raises(ValueError, match="fold"):
        map_aero_traction_to_membrane(forces, weights, STATIONS, nodes, grid,
                                      triangles, surface_coordinates=folded)
