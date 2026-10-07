"""Tests for awetrim.aerostructural.wireframe.reduce_fem (FEM_full -> wireframe).

The reference is ``struc_geometry_PSM_reduced.yaml``, an exact reduction of the
full geometry's bridle: same wing nodes (up to a rigid shift), same line set,
same fixed lengths, the tip lines absorbed into their pulleys.
"""

from pathlib import Path

import numpy as np
import pytest
import yaml

from awetrim.aerostructural.wireframe.reduce_fem import reduce_fem_geometry

DATA = Path(__file__).resolve().parents[2] / "data" / "LEI-V3-KITE"


def _load(name):
    with (DATA / name).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@pytest.fixture(scope="module")
def reduced():
    return reduce_fem_geometry(_load("struc_geometry_FEM_full.yaml"))


@pytest.fixture(scope="module")
def psm():
    return _load("struc_geometry_PSM_reduced.yaml")


def _wing(geometry):
    return {int(r[0]): np.array(r[1:4], float) for r in geometry["wing_particles"]["data"]}


def _l0(geometry, key):
    return {r[0]: float(r[1]) for r in geometry[key]["data"]}


def test_wing_nodes_are_the_reduced_files_up_to_a_rigid_shift(reduced, psm):
    ours, theirs = _wing(reduced), _wing(psm)
    assert sorted(ours) == sorted(theirs) == list(range(1, 21))
    shift = np.mean([theirs[i] - ours[i] for i in ours], axis=0)
    assert max(np.linalg.norm(theirs[i] - ours[i] - shift) for i in ours) < 1e-6


def test_bridle_line_set_matches(reduced, psm):
    ours = {r[0] for r in reduced["bridle_connections"]["data"]}
    theirs = {r[0] for r in psm["bridle_connections"]["data"]}
    # PSM's extra 0.07 m Br_main_e link before M is the one known difference.
    assert theirs - ours == {"Br_main_e"}
    assert ours - theirs == set()


def test_fixed_lengths_and_absorbed_tip_pulleys(reduced, psm):
    ours, theirs = _l0(reduced, "bridle_lines"), _l0(psm, "bridle_elements")
    for name in ("AI", "AII", "A_main", "br_1", "br_2", "br_3", "br_4",
                 "Br_I", "Br_II", "Br_main_1", "M", "steering_tape"):
        assert ours[name] == pytest.approx(theirs[name]), name
    assert ours["a5_equiv"] == pytest.approx(11.928)
    assert ours["br_5_equiv"] == pytest.approx(13.818)


def test_strut_fans_become_one_equivalent_line_each(reduced):
    rows = [r for r in reduced["bridle_connections"]["data"] if r[0].startswith("A") and "_equiv" in r[0]]
    assert len(rows) == 8
    assert all(len(r) == 3 and 1 <= r[1] <= 20 and r[1] % 2 == 1 for r in rows)


def test_wing_topology_and_compressive_tubes(reduced):
    names = [r[0] for r in reduced["wing_elements"]["data"]]
    assert sum(n.startswith("strut_") for n in names) == 10
    assert sum(n.startswith("le_") for n in names) == 9
    assert sum(n.startswith("te_") for n in names) == 9
    assert sum(n.startswith("dia_") for n in names) == 18
    linktype = {r[0]: r[5] for r in reduced["wing_elements"]["data"]}
    assert linktype["le_1"] == linktype["strut_1"] == "default"
    assert linktype["te_1"] == linktype["dia_1a"] == "noncompressive"


def test_masses_are_conserved():
    fem = _load("struc_geometry_FEM_full.yaml")
    ids = [0] + [int(r[0]) for k in ("wing_particles", "bridle_particles") for r in fem[k]["data"]]
    nodes = np.zeros((max(ids) + 1, 3))
    for k in ("wing_particles", "bridle_particles"):
        for r in fem[k]["data"]:
            nodes[int(r[0])] = r[1:4]
    masses = np.full(len(nodes), 0.1)
    masses[0] = 8.0
    reduced = reduce_fem_geometry(fem, nodes=nodes, masses=masses)
    total = reduced["reduction"]["wing_mass"] + reduced["reduction"]["bridle_mass"]
    assert total == pytest.approx(masses[1:].sum(), rel=1e-6)
