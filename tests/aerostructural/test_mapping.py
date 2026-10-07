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
