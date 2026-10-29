# Copyright (c) 2023-2026 Oriol Cayon, Delft University of Technology
#
# SPDX-License-Identifier: Apache-2.0

"""The package's public surface, and the dependencies it must NOT drag in."""

import sys

import pytest


def test_aerostructural_exposes_the_wireframe_coupler():
    import awetrim.aerostructural as aerostructural

    assert hasattr(aerostructural, "WireframeQsmCoupler")


@pytest.mark.parametrize("banned", ["PSS", "kite_fem", "pyfe3d"])
def test_importing_aerostructural_does_not_pull_in_a_removed_solver(banned):
    """The Particle System Simulator and kite_fem were removed on 2026-09-12.

    Billow is the only structural solver. This is a real check and not a
    formality: both were once imported at module scope, so a stray import would
    silently reinstate a dependency the project no longer declares.
    """
    for name in list(sys.modules):
        if name == banned or name.startswith(f"{banned}."):
            del sys.modules[name]

    import awetrim.aerostructural  # noqa: F401
    import awetrim.aerostructural.wireframe.structural_wireframe  # noqa: F401
    import awetrim.aerostructural.billow.structural_billow  # noqa: F401

    assert banned not in sys.modules


def test_the_wireframe_backend_is_billow_backed():
    from awetrim.aerostructural.wireframe import structural_wireframe

    assert structural_wireframe.build_line_system.__module__.startswith("billow")
