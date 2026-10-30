"""The rigid-tether warm start must not leave its node functions on the phase.

``residual_solver`` stores ``_node_alpha_function`` (and the Williams tension
function) for the solver it builds. The Williams phase builds its own solver
first, so the rigid warm start that follows used to overwrite them with
functions over the rigid solver's shorter decision vector, and the Williams
march failed at its first node with a CasADi shape error
(examples/5_flight_data/validate_spline_v3.py).
"""

from awetrim.timeseries.phase_parametrized import PhaseParameterized


class _Tether:
    diameter_tether = 0.01
    density_tether = 970.0


class _KiteModel:
    tether = _Tether()

    def reset_solver(self):
        pass


class _Phase:
    """The attributes ``_rigid_warm_start`` touches, nothing more."""

    quasi_steady = False  # the dynamic branch returns right after the build

    def __init__(self):
        self.kite_model = _KiteModel()

    def substitute_parametrized_kinematics(self, pattern):
        return None

    def residual_solver(self, km_copy=None, quasi_steady=None):
        self._node_alpha_function = "rigid alpha"
        self._williams_tension_ground_function = "rigid tension"
        return None


def test_rigid_warm_start_restores_the_williams_node_functions():
    phase = _Phase()
    phase._node_alpha_function = "williams alpha"
    phase._williams_tension_ground_function = "williams tension"
    model = phase.kite_model
    assert PhaseParameterized._rigid_warm_start(phase, None, None, [0.0], [1.5], [6.0]) is None
    assert phase._node_alpha_function == "williams alpha"
    assert phase._williams_tension_ground_function == "williams tension"
    assert phase.kite_model is model


def test_rigid_warm_start_leaves_no_functions_it_did_not_find():
    phase = _Phase()
    PhaseParameterized._rigid_warm_start(phase, None, None, [0.0], [1.5], [6.0])
    assert "_node_alpha_function" not in vars(phase)
    assert "_williams_tension_ground_function" not in vars(phase)
