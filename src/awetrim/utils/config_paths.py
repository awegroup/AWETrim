# Copyright (c) 2023-2026 Oriol Cayon, Delft University of Technology
#
# Licensed under the EUPL, Version 1.2 or - as soon they will be approved by
# the European Commission - subsequent versions of the EUPL (the "Licence");
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at:
#
#     https://joinup.ec.europa.eu/software/page/eupl
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the Licence is distributed on an "AS IS" basis,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the Licence for the specific language governing permissions and
# limitations under the Licence.
#
# SPDX-License-Identifier: EUPL-1.2

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"

LEI_V3_DATA_DIR = DATA_DIR / "LEI-V3-KITE"
# Primary system config in awesIO format (physical system parameters)
LEI_V3_SYSTEM_CONFIG = LEI_V3_DATA_DIR / "system.yaml"
# As-flown system (KCU 22.75 kg) used by validation; system.yaml is the
# optimization config (KCU 8.4 kg).
LEI_V3_SYSTEM_FLOWN_CONFIG = LEI_V3_DATA_DIR / "system_flown.yaml"
LEI_V3_ROM_AERO_CONFIG = LEI_V3_DATA_DIR / "rom_config.yaml"
LEI_V3_CYCLE_CONFIG_DIR = LEI_V3_DATA_DIR / "cycle_configs"
LEI_V3_DOWNLOOP_SPLINE_CONFIG = LEI_V3_CYCLE_CONFIG_DIR / "downloop_spline.yaml"
LEI_V3_UPLOOP_SPLINE_CONFIG = LEI_V3_CYCLE_CONFIG_DIR / "uploop_spline.yaml"
LEI_V3_HELIX_SPLINE_CONFIG = LEI_V3_CYCLE_CONFIG_DIR / "helix_spline.yaml"
LEI_V3_GENERATED_SPLINE_CONFIG = LEI_V3_CYCLE_CONFIG_DIR / "generated_spline.yaml"
