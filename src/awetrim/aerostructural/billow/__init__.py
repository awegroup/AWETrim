# Copyright (c) 2023-2026 Oriol Cayon, Delft University of Technology
#
# SPDX-License-Identifier: Apache-2.0

"""Billow-based aero-structural coupling.

``structural_billow`` adapts :mod:`awetrim.structural` to the structural-backend
contract the coupled drivers use; ``aerostructural_coupled_solver`` is the
Billow/QSM driver. The geometry reader, the load mapping and the VSM trim are
shared with the FEM path.
"""

from .structural_billow import (
    BillowStructure,
    canopy_grid,
    canopy_triangles,
    get_rest_length,
    get_rest_lengths,
    get_stiffnesses,
    grid_edges,
    instantiate,
    resolve_config,
    run_billow,
    set_rest_length,
    set_stiffnesses,
    update_rest_length,
)

__all__ = [
    "BillowStructure",
    "canopy_grid",
    "canopy_triangles",
    "get_rest_length",
    "get_rest_lengths",
    "get_stiffnesses",
    "grid_edges",
    "instantiate",
    "resolve_config",
    "run_billow",
    "set_rest_length",
    "set_stiffnesses",
    "update_rest_length",
]
