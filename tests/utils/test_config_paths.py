"""Unit tests for awetrim.utils.config_paths.

These are derived path constants; the tests pin the repo-root anchoring and the
canonical filenames so a refactor of the data layout can't silently break the
paths scripts rely on. Existence of the files is intentionally NOT asserted.
"""

from awetrim.utils import config_paths as cp


def test_repo_root_anchors_the_package():
    assert cp.REPO_ROOT.is_dir()
    assert (cp.REPO_ROOT / "src" / "awetrim").is_dir()


def test_data_dir_under_repo_root():
    assert cp.DATA_DIR == cp.REPO_ROOT / "data"
    assert cp.LEI_V3_DATA_DIR == cp.DATA_DIR / "LEI-V3-KITE"


def test_canonical_filenames():
    assert cp.LEI_V3_SYSTEM_CONFIG.name == "system.yaml"
    assert cp.LEI_V3_ROM_AERO_CONFIG == cp.LEI_V3_ROM_AEROSTRUCTURAL_FLIGHT_CONFIG
    assert cp.LEI_V3_ROM_AEROSTRUCTURAL_FLIGHT_CONFIG.exists()
    # Archived in git history, not on disk.
    assert not cp.LEI_V3_ROM_SEMI_EMPIRICAL_CONFIG.exists()
    assert cp.LEI_V3_DOWNLOOP_SPLINE_CONFIG.name == "downloop_spline.yaml"
    assert cp.LEI_V3_UPLOOP_SPLINE_CONFIG.name == "uploop_spline.yaml"
    assert cp.LEI_V3_HELIX_SPLINE_CONFIG.name == "helix_spline.yaml"


def test_cycle_configs_live_under_kite_dir():
    assert cp.LEI_V3_CYCLE_CONFIG_DIR == cp.LEI_V3_DATA_DIR / "cycle_configs"
    assert cp.LEI_V3_DOWNLOOP_SPLINE_CONFIG.parent == cp.LEI_V3_CYCLE_CONFIG_DIR
