import numpy as np

from awetrim.aerostructural import (
    PssQsmCoupler,
    QsmCouplingRequest,
    QsmCouplingSettings,
    StructuralGeometry,
    TapeActuationState,
)


class Particle:
    def __init__(self, position):
        self.x = np.asarray(position, dtype=float)

    def update_pos(self, position):
        self.x = np.asarray(position, dtype=float)

    def update_vel(self, velocity):
        self.v = np.asarray(velocity, dtype=float)


class FakePssSystem:
    def __init__(self, nodes, rest_lengths):
        self._particles = [Particle(node) for node in nodes]
        self._rest_lengths = np.asarray(rest_lengths, dtype=float).copy()
        self._f_int = np.zeros(3 * len(nodes))

    @property
    def particles(self):
        return self._particles

    @property
    def extract_rest_length(self):
        return self._rest_lengths

    @property
    def f_int(self):
        return self._f_int

    @property
    def x_v_current(self):
        return (
            np.asarray([particle.x for particle in self._particles]).reshape(-1),
            np.zeros(3 * len(self._particles)),
        )

    def update_rest_length(self, element_index, delta_length):
        self._rest_lengths[element_index] += delta_length

    def kin_damp_sim(self, external_force):
        self._f_int = -np.asarray(external_force, dtype=float)


class FakeStructuralSolver:
    def __init__(self):
        self.system = None

    def instantiate(self, geometry, settings):
        self.system = FakePssSystem(geometry.nodes, geometry.rest_lengths)
        return self.system

    def solve(self, system, external_force):
        system.kin_damp_sim(external_force)
        nodes = np.asarray([particle.x for particle in system.particles])
        return system, True, nodes, system.f_int


class Panel:
    def __init__(self, center):
        self.aerodynamic_center = np.asarray(center, dtype=float)


class FakeBody:
    def __init__(self):
        self.panels = [Panel([0.5, 0.5, 0.0])]
        self.updated = False

    def update_from_points(
        self,
        leading_edge_points,
        trailing_edge_points,
        *,
        aero_input_type,
        initial_polar_data,
    ):
        self.updated = True
        self.leading_edge_points = np.asarray(leading_edge_points)
        self.trailing_edge_points = np.asarray(trailing_edge_points)


def fake_trim_solver(**kwargs):
    return (
        {
            "opt_x": np.asarray([12.0, 0.0, 0.0, 0.0, 0.1]),
            "F_distribution": np.asarray([[0.0, 0.0, -8.0]]),
            "panel_cp_locations": np.asarray([[0.5, 0.5, 0.0]]),
            "inertial_force": np.zeros(3),
            "gravity_force": np.zeros(3),
            "success": True,
            "success_physical": True,
            "aoa_deg": 4.0,
            "side_slip_deg": 0.0,
        },
        kwargs["body_aero"],
    )


def geometry():
    return StructuralGeometry(
        nodes=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ]
        ),
        masses=np.ones(4),
        connectivity=np.asarray([[0, 1], [2, 3]]),
        rest_lengths=np.asarray([1.0, 1.0]),
        stiffness=np.asarray([100.0, 100.0]),
        damping=np.asarray([0.0, 0.0]),
        link_types=["default", "default"],
        le_node_indices=np.asarray([0, 2]),
        te_node_indices=np.asarray([1, 3]),
        fixed_node_indices=[],
        pulley_line_indices=[],
        pulley_line_to_other_node_pair={},
    )


def settings():
    return QsmCouplingSettings(
        max_iter=3,
        residual_tolerance=1e-9,
        residual_stagnation_window=3,
        residual_stagnation_tolerance=0.0,
        relaxation_factor=1.0,
        use_aitken_relaxation=False,
        n_aero_panels_per_structural_section=1,
    )


def test_qsm_coupler_runs_with_fake_protocol_implementations():
    structural_solver = FakeStructuralSolver()
    request = QsmCouplingRequest(
        structural_geometry=geometry(),
        system_model=object(),
        body_aero=FakeBody(),
        vsm_solver=object(),
        center_of_gravity=None,
        reference_point=np.zeros(3),
        x_guess=np.asarray([10.0, 0.0, 0.0, 0.0, 0.0]),
        bounds_lower=np.asarray([1.0, -1.0, -1.0, -1.0, -1.0]),
        bounds_upper=np.asarray([20.0, 1.0, 1.0, 1.0, 1.0]),
        settings=settings(),
    )

    result = PssQsmCoupler(
        structural_solver=structural_solver,
        trim_solver=fake_trim_solver,
    ).solve(request)

    assert result.converged is True
    assert len(result.iteration_records) == 1
    np.testing.assert_allclose(result.final_residual, np.zeros(12))
    np.testing.assert_allclose(result.metadata["opt_x"], [12.0, 0.0, 0.0, 0.0, 0.1])
    assert result.metadata["moment_preservation"]["dF_norm"] == 0.0


def test_fixed_node_residual_components_are_removed():
    residual = np.arange(12, dtype=float)

    cleaned = PssQsmCoupler._residual_without_fixed_nodes(residual, [1])

    np.testing.assert_allclose(cleaned[3:6], np.zeros(3))
    np.testing.assert_allclose(cleaned[:3], residual[:3])
    np.testing.assert_allclose(cleaned[6:], residual[6:])


def test_tape_actuation_updates_only_target_rest_lengths():
    system = FakePssSystem(np.zeros((3, 3)), np.asarray([1.0, 2.0, 3.0]))
    actuation = TapeActuationState(
        power_tape_index=0,
        steering_tape_indices=(1, 2),
        initial_power_tape_length=1.0,
        initial_steering_left_length=2.0,
        initial_steering_right_length=3.0,
        power_tape_final_extension=0.2,
        power_tape_extension_step=0.05,
        steering_tape_final_extension=0.1,
        steering_tape_extension_step=0.05,
    )

    PssQsmCoupler._apply_actuation(
        system,
        actuation,
        iteration=0,
        steering_interval=1,
        power_interval=1,
    )

    np.testing.assert_allclose(system.extract_rest_length, [1.05, 1.95, 3.05])
