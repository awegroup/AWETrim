import numpy as np

from awetrim.aerostructural.mapping import (
    BilinearAeroToStructuralLoadMapper,
    LinearStructuralToAeroMapper,
    check_moment_preservation,
    interpolate_points,
)


class Panel:
    def __init__(self, center):
        self.aerodynamic_center = np.asarray(center, dtype=float)


def test_interpolate_points_adds_inner_points_per_section():
    points = np.asarray([[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]])

    interpolated = interpolate_points(points, n_panels_per_section=2)

    np.testing.assert_allclose(
        interpolated,
        np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 2.0, 0.0],
            ]
        ),
    )


def test_linear_structural_to_aero_mapper_returns_le_and_te_points():
    nodes = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [1.0, 2.0, 0.0],
        ]
    )

    update = LinearStructuralToAeroMapper().map(
        nodes,
        le_node_indices=np.asarray([0, 2]),
        te_node_indices=np.asarray([1, 3]),
        n_panels_per_section=2,
    )

    assert update.leading_edge_points.shape == (3, 3)
    assert update.trailing_edge_points.shape == (3, 3)
    np.testing.assert_allclose(update.leading_edge_points[1], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(update.trailing_edge_points[1], [1.0, 1.0, 0.0])


def test_bilinear_aero_to_structural_mapper_preserves_total_force():
    nodes = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
    )
    mapper = BilinearAeroToStructuralLoadMapper()
    mapping = mapper.initialize(
        [Panel([0.5, 0.5, 0.0])],
        nodes,
        le_node_indices=np.asarray([0, 2]),
        te_node_indices=np.asarray([1, 3]),
    )
    panel_forces = np.asarray([[4.0, -2.0, 8.0]])
    panel_points = np.asarray([[0.5, 0.5, 0.0]])

    nodal_forces = mapper.map_loads(panel_forces, panel_points, nodes, mapping)

    np.testing.assert_allclose(np.sum(nodal_forces, axis=0), panel_forces[0])
    np.testing.assert_allclose(
        nodal_forces[[0, 1, 2, 3]],
        np.tile(panel_forces[0] / 4.0, (4, 1)),
    )


def test_moment_preservation_report_contains_force_and_moment_errors():
    nodes = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
    )
    panel_forces = np.asarray([[0.0, 0.0, 4.0]])
    panel_points = np.asarray([[0.5, 0.5, 0.0]])
    nodal_forces = np.full((4, 3), [0.0, 0.0, 1.0])

    report = check_moment_preservation(panel_forces, panel_points, nodal_forces, nodes)

    assert report["dF_norm"] == 0.0
    assert report["dM_norm"] == 0.0


# --- SkinStructuralToAeroMapper ------------------------------------------------

from awetrim.aerostructural.mapping import SkinStructuralToAeroMapper as _Skin


def _skin_case():
    """Three ribs (front/rear nodes) on a curved arc; aero points off the ribs."""
    ribs_y = np.array([2.0, 0.0, -2.0])
    nodes = [np.zeros(3)]
    for y in ribs_y:
        z = 8.0 - 0.1 * y**2
        nodes += [np.array([0.0, y, z]), np.array([1.5, y, z - 0.2])]
    nodes = np.asarray(nodes)
    sections_y = np.linspace(2.2, -2.2, 7)
    le = np.array([[-0.05, y, 8.0 - 0.1 * y**2 + 0.05] for y in sections_y])
    te = np.array([[1.7, y, 7.75 - 0.1 * y**2] for y in sections_y])
    return nodes, le, te


def test_skin_reproduces_the_reference_edges():
    nodes, le, te = _skin_case()
    skin = _Skin().initialize(nodes, [1, 3, 5], [2, 4, 6], le, te, n_panels_per_section=1)
    update = skin.map(nodes)
    np.testing.assert_allclose(update.leading_edge_points, le, atol=1e-10)
    np.testing.assert_allclose(update.trailing_edge_points, te, atol=1e-10)


def test_skin_moves_rigidly_with_the_structure():
    nodes, le, te = _skin_case()
    skin = _Skin().initialize(nodes, [1, 3, 5], [2, 4, 6], le, te, n_panels_per_section=1)
    angle = 0.3
    rotation = np.array([[np.cos(angle), 0.0, np.sin(angle)], [0.0, 1.0, 0.0],
                         [-np.sin(angle), 0.0, np.cos(angle)]])
    shift = np.array([0.4, -0.1, 2.0])
    update = skin.map(nodes @ rotation.T + shift)
    np.testing.assert_allclose(update.leading_edge_points, le @ rotation.T + shift, atol=1e-9)
    np.testing.assert_allclose(update.trailing_edge_points, te @ rotation.T + shift, atol=1e-9)


def test_skin_subdivides_like_the_reference_mesh():
    nodes, le, te = _skin_case()
    skin = _Skin().initialize(nodes, [1, 3, 5], [2, 4, 6], le, te, n_panels_per_section=2)
    assert skin.map(nodes).leading_edge_points.shape == (2 * (len(le) - 1) + 1, 3)


def test_bilinear_mapper_is_continuous_on_a_near_vertical_le_segment():
    """A wing-tip LE that curls down: the segment runs along z with almost no
    y extent. Moving a node by a millimetre across y must not throw the
    panel load from one LE node to the other."""
    mapper = BilinearAeroToStructuralLoadMapper()
    mapping = mapper.initialize(
        [Panel([0.5, 4.0005, 0.25])],
        np.asarray(
            [[0.0, 4.0, 0.0], [1.0, 4.0, 0.0], [0.0, 4.001, 0.5], [1.0, 4.001, 0.5]]
        ),
        le_node_indices=np.asarray([0, 2]),
        te_node_indices=np.asarray([1, 3]),
    )
    panel_forces = np.asarray([[0.0, 0.0, 10.0]])
    panel_points = np.asarray([[0.5, 4.0005, 0.25]])
    loads = []
    for dy in (-1e-3, 1e-3):
        nodes = np.asarray(
            [[0.0, 4.0, 0.0], [1.0, 4.0, 0.0], [0.0, 4.0 + dy, 0.5], [1.0, 4.0 + dy, 0.5]]
        )
        loads.append(mapper.map_loads(panel_forces, panel_points, nodes, mapping))
    np.testing.assert_allclose(loads[0], loads[1], atol=0.05)
    np.testing.assert_allclose(np.sum(loads[1], axis=0), panel_forces[0])


# -- strut-lever (rigid-body) load transfer -----------------------------------
from awetrim.aerostructural.mapping import StrutLeverAeroToStructuralLoadMapper as _StrutLever


def _warped_ribs(n_ribs=5, seed=0):
    """Front/rear rib nodes of a warped wing: struts tilted, ribs off-plane."""
    rng = np.random.default_rng(seed)
    front = np.array([[0.1 * rng.normal(), 1.3 * r, 0.3 * rng.normal()] for r in range(n_ribs)])
    rear = front + [2.4, 0.0, 0.0] + rng.normal(scale=0.3, size=(n_ribs, 3))
    return front, rear


def test_strut_lever_conserves_force_and_moment_on_a_warped_wing():
    rng = np.random.default_rng(1)
    front, rear = _warped_ribs()
    nodes = np.vstack([front, rear])
    n = 150
    bays = rng.integers(0, len(front) - 1, size=n)
    points = 0.5 * (front[bays] + rear[bays + 1]) + rng.normal(scale=0.3, size=(n, 3))
    forces = rng.normal(scale=200.0, size=(n, 3))
    moments = rng.normal(scale=30.0, size=(n, 3))
    mapper = _StrutLever()
    mapping = mapper.initialize(bays, np.arange(len(front)), np.arange(len(front), 2 * len(front)))
    nodal = mapper.map_loads(forces, points, nodes, mapping, panel_moments=moments)
    report = check_moment_preservation(forces, points, nodal, nodes, ref_point=[0.3, -2.0, 1.0],
                                       panel_moments=moments)
    assert report["dF_norm"] < 1e-9
    assert report["dM_norm"] < 1e-9


def test_strut_lever_is_the_bilinear_split_on_a_flat_bay():
    front = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    rear = np.array([[2.0, 0.0, 0.0], [2.0, 1.0, 0.0]])
    mapper = _StrutLever()
    mapping = mapper.initialize([0], [0, 1], [2, 3])
    force = np.array([[10.0, -3.0, 100.0]])
    nodal = mapper.map_loads(force, [[0.5, 0.3, 0.0]], np.vstack([front, rear]), mapping)
    xi, eta = 0.25, 0.3
    weights = np.array([(1 - xi) * (1 - eta), (1 - xi) * eta, xi * (1 - eta), xi * eta])
    np.testing.assert_allclose(nodal, weights[:, None] * force, atol=1e-12)


def test_strut_lever_carries_a_twist_across_the_span():
    """A moment about the chord axis cannot sit on one strut: it becomes a
    couple between the two struts."""
    front = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    rear = np.array([[2.0, 0.0, 0.0], [2.0, 1.0, 0.0]])
    mapper = _StrutLever()
    mapping = mapper.initialize([0], [0, 1], [2, 3])
    nodal = mapper.map_loads(np.zeros((1, 3)), [[0.5, 0.5, 0.0]], np.vstack([front, rear]),
                             mapping, panel_moments=[[40.0, 0.0, 0.0]])
    strut_a, strut_b = nodal[[0, 2]].sum(axis=0), nodal[[1, 3]].sum(axis=0)
    np.testing.assert_allclose(strut_a, [0.0, 0.0, -40.0], atol=1e-12)
    np.testing.assert_allclose(strut_b, [0.0, 0.0, 40.0], atol=1e-12)


def test_strut_lever_zero_load_is_zero():
    front, rear = _warped_ribs(3)
    mapper = _StrutLever()
    mapping = mapper.initialize([0, 1], [0, 1, 2], [3, 4, 5])
    nodal = mapper.map_loads(np.zeros((2, 3)), front[:2] + 0.5, np.vstack([front, rear]), mapping,
                             panel_moments=np.zeros((2, 3)))
    assert np.all(nodal == 0.0)


def test_strut_lever_is_mirror_symmetric():
    front = np.array([[0.1, 1.4, 0.3], [0.0, 0.0, 0.5], [0.1, -1.4, 0.3]])
    rear = np.array([[2.4, 1.5, 0.2], [2.5, 0.0, 0.4], [2.4, -1.5, 0.2]])
    nodes = np.vstack([front, rear])
    S = np.diag([1.0, -1.0, 1.0])
    point, force, moment = np.array([0.7, 0.6, 0.4]), np.array([5.0, 20.0, 300.0]), np.array([3.0, -40.0, 2.0])
    mapper = _StrutLever()
    mapping = mapper.initialize([0, 1], [0, 1, 2], [3, 4, 5])
    nodal = mapper.map_loads(
        np.array([force, S @ force]), np.array([point, S @ point]), nodes, mapping,
        panel_moments=np.array([moment, np.diag([-1.0, 1.0, -1.0]) @ moment]),
    )
    # panel 0 in bay 0, its mirror image in bay 1: the nodal forces mirror too
    single = mapper.map_loads(force[None], point[None], nodes, mapper.initialize([0], [0, 1, 2], [3, 4, 5]),
                              panel_moments=moment[None])
    mirror = (S @ single.T).T[[2, 1, 0, 5, 4, 3]]
    np.testing.assert_allclose(nodal - single, mirror, atol=1e-10)


def test_strut_lever_rejects_a_bay_outside_the_ribs():
    import pytest

    with pytest.raises(ValueError):
        _StrutLever().initialize([0, 2], [0, 1, 2], [3, 4, 5])
