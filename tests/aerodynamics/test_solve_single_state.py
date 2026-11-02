"""Tests for the step-1 scripts' kite (examples/1_aerodynamics/common.py).

``load_kite`` builds the kite as the coupled solver does, everything physical
from the data folder: the trim mass and the CG/inertia must be one body (the
v1.0.1 mass bug), the mesh and the moment reference come from as_config and
the structure, and a shape the case cannot have is refused rather than
silently replaced.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("VSM")

from conftest import lei_v3_case, load_step1_common  # noqa: E402

common = load_step1_common()


@pytest.fixture(scope="module")
def kite():
    """The scripts' default kite: built shape, as_config's mesh."""
    return common.load_kite("LEI-V3-KITE", lei_v3_case())


def _mock_trim_result():
    return {
        "opt_x": np.array([20.0, 0.1, -0.05, 0.0, 0.0]),
        "cm": np.array([1e-4, 2e-4, 3e-4]),
        "cfx": 1e-3,
        "cfy": 2e-3,
        "success": True,
        "success_physical": True,
        "aoa_deg": 5.0,
        "side_slip_deg": 0.5,
        "cl": 1.2,
        "cd": 0.05,
    }


class TestOneBody:
    def test_trim_mass_is_wing_plus_kcu_once(self, kite):
        """The trim mass is the system model's wing + KCU, counted once, and
        the CG and inertia belong to that same mass (the aggregate's)."""
        model = kite.system_model.kite
        assert kite.mass == pytest.approx(float(model.mass_wing) + float(model.mass_kcu))
        assert kite.inertia_cg.shape == (3, 3)

    def test_refuses_a_mass_its_cg_does_not_belong_to(self, kite):
        """The pre-v1.0.1 stability script: the aggregate (KCU included) as the
        wing mass, so the system model added the KCU a second time."""
        aggregate = {
            "mass": 2.0 * kite.mass,
            "center_of_mass": kite.center_of_gravity,
            "inertia_tensor": kite.inertia_cg,
        }
        with pytest.raises(ValueError, match="structure aggregate"):
            common.kite_rigid_body(kite.system_model, aggregate)


class TestKiteFromTheDataFolder:
    def test_mesh_is_as_configs(self, kite):
        """5 panels per FEM section on the 28 FEM sections: 135."""
        per_section = kite.config["aerodynamic"]["n_aero_panels_per_struc_section"]
        assert len(kite.body.panels) == 27 * per_section

    def test_moment_reference_is_the_kcu_point(self, kite):
        from awetrim.aerostructural.coupled.awesio_structure import load_structure

        structure = load_structure(kite.structure_path)
        np.testing.assert_allclose(kite.reference_point, structure.positions[structure.kcu])

    def test_model_choices_are_as_configs(self, kite):
        assert kite.kcu_drag is not None
        assert kite.tether_in_trim == "williams"

    def test_step1_vsm_loop_is_base_without_av(self, kite):
        """The one deliberate difference from step 2 (common.STEP1_VSM_SETTINGS):
        the least_squares trims stop at their seed with as_config's AV loop."""
        assert not kite.solver.is_with_artificial_viscosity
        assert kite.solver.gamma_loop_type == "base"

    def test_the_case_sets_the_system_model(self, kite):
        assert kite.system_model.distance_radial == pytest.approx(300.0)
        assert kite.system_model.speed_radial == pytest.approx(1.5)
        assert kite.system_model.timeder_speed_tangential == 0.0

    def test_zero_billowing_is_the_built_wing(self):
        """Contraction 0 gives the structure's own wing: its FEM sections,
        except the two tip leading-edge points (``wing_tip.correct_tip_bays``)."""
        from awetrim.aerostructural.coupled.awesio_structure import load_structure
        from awetrim.aerostructural.wireframe.driver import wing_edges

        structure = load_structure(common.PROJECT_DIR / "data" / "LEI-V3-KITE" / "structure.yml")
        reader = structure.resolved()
        le, te = wing_edges(structure, 0.0, 3)
        built_le = reader[0][np.asarray(reader[2])]
        np.testing.assert_allclose(le[::3][1:-1], built_le[1:-1])
        assert np.linalg.norm(le[0] - built_le[0]) > 0.1
        np.testing.assert_allclose(te[::3], reader[0][np.asarray(reader[3])])


class TestShapeSelection:
    def test_a_tape_command_needs_a_deformed_shape(self):
        with pytest.raises(ValueError, match="tape command"):
            common.load_kite("LEI-V3-KITE", lei_v3_case(power_tape_extension=0.2))

    def test_a_missing_step2_run_names_the_script(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="run_simulation_BILLOW.py"):
            common.load_kite("LEI-V3-KITE", lei_v3_case(), shape="billow", project_dir=tmp_path)

    def test_the_shape_folder_is_the_cases_step2_result(self):
        folder = common.deformed_shape_folder("LEI-V3-KITE", lei_v3_case(), "wireframe")
        assert folder.parent.name == "wireframe"
        assert folder.name == lei_v3_case().folder_name()

    def test_unknown_shape_raises(self):
        with pytest.raises(ValueError, match="shape must be one of"):
            common.load_kite("LEI-V3-KITE", lei_v3_case(), shape="cad")


class TestOutput:
    def test_output_dir_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = common.output_dir("test_script", Path(tmpdir) / "out")
            assert path.is_dir()

    def test_write_json_handles_numpy_and_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "result.json"
            payload = {
                "array": np.array([1.0, 2.0]),
                "scalar": np.float64(42.0),
                "tuple": (1.0, 2.0),
                "path": Path(tmpdir),
                "trim": _mock_trim_result(),
            }
            with patch("builtins.print"):
                common.write_json(out_path, payload)
            data = json.loads(out_path.read_text(encoding="utf-8"))
            assert isinstance(data["array"], list)
            assert isinstance(data["tuple"], list)
            assert isinstance(data["scalar"], float)
            assert data["path"] == str(Path(tmpdir))
            assert data["trim"]["cfx"] == pytest.approx(1e-3)

    def test_print_trim_summary_handles_a_result(self):
        with patch("builtins.print"):
            common.print_trim_summary(_mock_trim_result())

    def test_properties_are_json_ready(self, kite):
        with tempfile.TemporaryDirectory() as tmpdir, patch("builtins.print"):
            common.write_json(Path(tmpdir) / "kite.json", kite.properties())
