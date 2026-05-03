from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"

LEI_V3_DATA_DIR = DATA_DIR / "LEI-V3-KITE"
LEI_V3_SYSTEM_CONFIG = LEI_V3_DATA_DIR / "lei_v3_system_config.yaml"
LEI_V3_CYCLE_CONFIG_DIR = LEI_V3_DATA_DIR / "cycle_configs"
LEI_V3_DOWNLOOP_SPLINE_CONFIG = LEI_V3_CYCLE_CONFIG_DIR / "downloop_spline.yaml"
LEI_V3_UPLOOP_SPLINE_CONFIG = LEI_V3_CYCLE_CONFIG_DIR / "uploop_spline.yaml"
LEI_V3_HELIX_SPLINE_CONFIG = LEI_V3_CYCLE_CONFIG_DIR / "helix_spline.yaml"
LEI_V3_GENERATED_SPLINE_CONFIG = LEI_V3_CYCLE_CONFIG_DIR / "generated_spline.yaml"
