"""awetrim.aerostructural.symmetry: the readers' mirror-symmetry check."""

import copy
import logging
from pathlib import Path

import numpy as np
import pytest
import yaml

from awetrim.aerostructural.symmetry import check_mirror_symmetry, warn_if_asymmetric

KITE_DIR = Path(__file__).resolve().parents[2] / "data" / "LEI-V3-KITE"


def _toy():
    """A centre node and two mirrored pairs, a mirrored element pair and a
    centre-crossing element."""
    nodes = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, -1.0, 1.0], [0.5, 2.0, 1.0], [0.5, -2.0, 1.0]])
    conn = np.array([[0, 1], [0, 2], [1, 3], [2, 4], [1, 2]])
    l0 = np.array([1.4, 1.4, 1.1, 1.1, 2.0])
    k = np.full(5, 1e5)
    types = ["noncompressive"] * 5
    masses = np.array([10.0, 0.2, 0.2, 0.1, 0.1])
    return nodes, conn, l0, k, types, masses


def test_a_symmetric_geometry_has_no_issues():
    assert check_mirror_symmetry(*_toy()) == []


def test_each_kind_of_asymmetry_is_reported():
    nodes, conn, l0, k, types, masses = _toy()
    nodes[3, 2] += 0.003  # 3 mm
    nodes[0, 1] = 0.001  # centre node off the plane
    l0[3] = 1.07  # one side's element 30 mm short
    masses[4] = 0.3
    issues = check_mirror_symmetry(nodes, conn, l0, k, types, masses)
    text = "\n".join(issues)
    assert "nodes 3 / 4: positions 3.00 mm" in text
    assert "node 0: 1.00 mm off the symmetry plane" in text
    assert "rest lengths 1.1 / 1.07" in text
    assert "masses 0.1 / 0.3" in text


def test_a_node_without_mirror_partner_is_reported():
    nodes, conn, l0, k, types, masses = _toy()
    nodes[4] = [3.0, -0.5, 4.0]
    assert any("no mirror partner" in i for i in check_mirror_symmetry(nodes, conn, l0, k, types, masses))


def test_the_warning_can_be_switched_off(caplog):
    nodes, conn, l0, k, types, masses = _toy()
    l0[3] = 1.07
    with caplog.at_level(logging.WARNING):
        assert warn_if_asymmetric(nodes, conn, l0, k, types, masses, config={"is_with_symmetry_check": False}) == []
        assert warn_if_asymmetric(nodes, conn, l0, k, types, masses, config={}) != []
    assert "not mirror-symmetric" in caplog.text


def _psm():
    """The LEI-V3 PSM-format wireframe: the reduction of the FEM geometry."""
    from awetrim.aerostructural.wireframe.reduce_fem import reduce_fem_geometry

    from awetrim.aerostructural.coupled.awesio_structure import load_structure

    return reduce_fem_geometry(load_structure(KITE_DIR / "structure.yml"))


def test_the_lei_v3_geometries_are_symmetric():
    from awetrim.aerostructural.utils import resolve_structure
    from awetrim.aerostructural.wireframe import structural_geometry_io

    r = structural_geometry_io.main(_psm())
    assert check_mirror_symmetry(r[0], r[7], r[10], r[11], r[13], r[1]) == []
    system = yaml.safe_load((KITE_DIR / "system.yaml").read_text(encoding="utf-8"))
    r = resolve_structure(KITE_DIR / "structure.yml", system_config=system)
    assert check_mirror_symmetry(r[0], r[10], r[13], r[14], r[16], r[1]) == []


def test_pulley_rope_mass_does_not_depend_on_the_row_order():
    """[end, pulley, end] and [end, pulley, end] reversed are the same rope."""
    from awetrim.aerostructural.wireframe import structural_geometry_io

    geometry = _psm()
    reversed_rows = copy.deepcopy(geometry)
    for row in reversed_rows["bridle_connections"]["data"]:
        if len(row) == 4:
            row[1], row[3] = row[3], row[1]
    m = structural_geometry_io.main(geometry)[1]
    m_reversed = structural_geometry_io.main(reversed_rows)[1]
    np.testing.assert_allclose(m_reversed, m, rtol=0, atol=1e-12)
    assert m.sum() == pytest.approx(m_reversed.sum())
