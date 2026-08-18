# Copyright (c) 2023-2026 Oriol Cayon, Delft University of Technology
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"

LEI_V3_DATA_DIR = DATA_DIR / "LEI-V3-KITE"
# Primary system config in awesIO format (physical system parameters)
LEI_V3_SYSTEM_CONFIG = LEI_V3_DATA_DIR / "system.yaml"
# As-flown systems, one per flight campaign, used by validation; system.yaml is
# the optimization config (KCU 8.4 kg).
#   2019-10-08: KCU 22.75 kg, 10 mm tether
#   2025-10-09: KCU 23.3 kg, 13.5 mm tether
LEI_V3_SYSTEM_FLOWN_2019_CONFIG = LEI_V3_DATA_DIR / "system_flown_2019.yaml"
LEI_V3_SYSTEM_FLOWN_2025_CONFIG = LEI_V3_DATA_DIR / "system_flown_2025.yaml"
# Unqualified alias, kept for callers that predate the per-flight split; it is
# the 2019 hardware, which every existing validation script assumed.
LEI_V3_SYSTEM_FLOWN_CONFIG = LEI_V3_SYSTEM_FLOWN_2019_CONFIG
LEI_V3_ROM_AERO_CONFIG = LEI_V3_DATA_DIR / "rom_config.yaml"
LEI_V3_CYCLE_CONFIG_DIR = LEI_V3_DATA_DIR / "cycle_configs"
LEI_V3_DOWNLOOP_SPLINE_CONFIG = LEI_V3_CYCLE_CONFIG_DIR / "downloop_spline.yaml"
LEI_V3_UPLOOP_SPLINE_CONFIG = LEI_V3_CYCLE_CONFIG_DIR / "uploop_spline.yaml"
LEI_V3_HELIX_SPLINE_CONFIG = LEI_V3_CYCLE_CONFIG_DIR / "helix_spline.yaml"
LEI_V3_GENERATED_SPLINE_CONFIG = LEI_V3_CYCLE_CONFIG_DIR / "generated_spline.yaml"
