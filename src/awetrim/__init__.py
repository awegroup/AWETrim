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

from .system import SystemModel, State, create_system_model_from_yaml
from .timeseries import Cycle
from .utils import defaults, reference_frames, color_palette, utils
import logging

# Configure logging (usually done once in your main module or __init__.py)
logging.basicConfig(level=logging.INFO)  # or DEBUG for more detail
logger = logging.getLogger(__name__)  # Use module-specific logger
