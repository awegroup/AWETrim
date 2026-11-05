"""sim_parameters.slow_path_rate: IPOPT keeps the march's warm start, only when asked."""
import copy

import casadi as ca
import pytest

from awetrim.system.factory import create_system_model_from_yaml
from awetrim.timeseries.phase_parametrized import PhaseParameterized
from awetrim.utils.config_paths import LEI_V3_HELIX_SPLINE_CONFIG, LEI_V3_SYSTEM_CONFIG
from awetrim.utils.utils import load_cycle_config_from_yaml

pytestmark = pytest.mark.skipif(
    not LEI_V3_SYSTEM_CONFIG.exists() or not LEI_V3_HELIX_SPLINE_CONFIG.exists(),
    reason="LEI-V3 kite data not available",
)


@pytest.mark.parametrize("slow, expected", [(True, 1e-8), (False, None)])
def test_residual_solver_bound_push_is_opt_in(monkeypatch, slow, expected):
    config, _ = load_cycle_config_from_yaml(LEI_V3_HELIX_SPLINE_CONFIG)
    config = copy.deepcopy(config)
    config["sim_parameters"]["slow_path_rate"] = slow
    system_model = create_system_model_from_yaml(yaml_path=LEI_V3_SYSTEM_CONFIG)
    phase = PhaseParameterized(system_model, quasi_steady=True, pattern_config=config)
    seen = {}

    def nlpsol(name, plugin, nlp, options):
        seen.update(options["ipopt"])
        return "solver"

    monkeypatch.setattr(ca, "nlpsol", nlpsol)

    assert phase.residual_solver() == "solver"
    assert seen.get("bound_push") == expected
    assert seen.get("bound_frac") == expected
