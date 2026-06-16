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

"""Shared plotting helpers and styling."""

from .plotting import (
    PALETTE,
    main,
    plot_aerodynamic_forces_chordwise_distributed,
    plot_normalized_elongation,
    set_plot_style,
)

__all__ = [
    "PALETTE",
    "main",
    "plot_aerodynamic_forces_chordwise_distributed",
    "plot_normalized_elongation",
    "set_plot_style",
]
