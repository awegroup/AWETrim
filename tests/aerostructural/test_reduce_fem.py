"""Tests for awetrim.aerostructural.wireframe.reduce_fem (structure.yml -> wireframe).

The reference numbers are frozen from ``struc_geometry_PSM_reduced.yaml``
(removed 2026-10-29; in the v0.1.0 release), an exact reduction of the full
geometry's bridle: same wing nodes (the file sat 0.0466 m lower, a rigid
shift), same line set, same fixed lengths, the tip lines absorbed into their
pulleys.
"""

from pathlib import Path

import numpy as np
import pytest

from awetrim.aerostructural.coupled.awesio_structure import load_structure
from awetrim.aerostructural.wireframe.reduce_fem import reduce_fem_geometry

DATA = Path(__file__).resolve().parents[2] / "data" / "LEI-V3-KITE"


@pytest.fixture(scope="module")
def structure():
    return load_structure(DATA / "structure.yml")


@pytest.fixture(scope="module")
def reduced(structure):
    return reduce_fem_geometry(structure)


#: struc_geometry_PSM_reduced.yaml's wing nodes [m], shifted by +0.0466 m in z
#: onto the built FEM geometry: front (odd) and rear (even) attachment per rib.
WING_NODES = {
    1: (-0.183243, 4.085799, 8.602814),
    2: (0.789240, 4.140976, 8.378346),
    3: (-0.692027, 3.967853, 9.239621),
    4: (0.839743, 3.956753, 9.236237),
    5: (-0.906897, 3.134159, 10.244571),
    6: (1.086881, 3.118511, 10.224542),
    7: (-1.050409, 1.959598, 10.846579),
    8: (1.169142, 1.947901, 10.813494),
    9: (-1.122026, 0.664217, 11.106687),
    10: (1.205432, 0.661074, 11.066604),
    11: (-1.122026, -0.664217, 11.106687),
    12: (1.205432, -0.661074, 11.066604),
    13: (-1.050409, -1.959598, 10.846579),
    14: (1.169142, -1.947901, 10.813494),
    15: (-0.906897, -3.134159, 10.244571),
    16: (1.086881, -3.118511, 10.224542),
    17: (-0.692027, -3.967853, 9.239621),
    18: (0.839743, -3.956753, 9.236237),
    19: (-0.183243, -4.085799, 8.602814),
    20: (0.789240, -4.140976, 8.378346),
}

#: struc_geometry_PSM_reduced.yaml's bridle line set (less its extra 0.07 m
#: Br_main_e link before M, the one known difference).
LINES = {
    "A1_equiv", "A2_equiv", "A3_equiv", "A4_equiv", "AI", "AII", "A_main",
    "Br_I", "Br_II", "Br_main_1", "M", "a5_equiv", "br_1", "br_2", "br_3",
    "br_4", "br_5_equiv", "depower_tape", "steering_tape",
}

#: Its fixed rest lengths [m].
FIXED_LENGTHS = {
    "AI": 3.67, "AII": 3.36, "A_main": 3.903, "br_1": 4.407, "br_2": 4.18,
    "br_3": 4.094, "br_4": 3.55, "Br_I": 2.333, "Br_II": 2.045,
    "Br_main_1": 0.61, "M": 4.856, "steering_tape": 1.6,
}


def test_wing_nodes_are_the_reduced_files(reduced):
    ours = {int(r[0]): np.array(r[1:4], float) for r in reduced["wing_particles"]["data"]}
    assert sorted(ours) == list(range(1, 21))
    for i, expected in WING_NODES.items():
        np.testing.assert_allclose(ours[i], expected, atol=1e-6, err_msg=str(i))


def test_bridle_line_set_matches(reduced):
    assert {r[0] for r in reduced["bridle_connections"]["data"]} == LINES


def test_fixed_lengths_and_absorbed_tip_pulleys(reduced):
    ours = {r[0]: float(r[1]) for r in reduced["bridle_lines"]["data"]}
    for name, length in FIXED_LENGTHS.items():
        assert ours[name] == pytest.approx(length), name
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


def test_masses_are_conserved(structure):
    masses = np.full(len(structure.positions), 0.1)
    masses[structure.kcu] = 8.0
    reduced = reduce_fem_geometry(structure, nodes=structure.positions, masses=masses)
    total = reduced["reduction"]["wing_mass"] + reduced["reduction"]["bridle_mass"]
    assert total == pytest.approx(masses.sum() - 8.0, rel=1e-6)


def test_without_masses_the_structure_is_weighed(structure, reduced):
    """The wireframe weighs what the full structure does, KCU aside."""
    total = reduced["reduction"]["wing_mass"] + reduced["reduction"]["bridle_mass"]
    masses = structure.masses()
    assert total == pytest.approx(masses.sum() - masses[structure.kcu], rel=1e-6)


def test_tube_stiffness_comes_from_the_tubes(structure, reduced):
    """Strut members at the strut's mean diameter: the length-weighted mean of
    its elements, which is the retired file's (d_le + d_te) / 2 exactly."""
    from awetrim.aerostructural.billow.structural_billow import tube_axial_stiffness

    # ribs run +y to -y like the struts: the 5th (structure "strut_4", after
    # the tip rib "le_tip_start") is the wireframe's strut_5
    strut = structure.struts[4]
    assert strut["name"] == "strut_4"
    old_mean = 0.5 * (0.140834 + 0.105972)  # strut_diam_le / _te of the old file
    nodes = strut["nodes"]
    lengths = np.linalg.norm(np.diff(structure.positions[nodes], axis=0), axis=1)
    assert np.dot(lengths, strut["diameters"]) / lengths.sum() == pytest.approx(old_mean, abs=1e-9)
    k = {r[0]: r[2] for r in reduced["wing_elements"]["data"]}
    ends = {r[0]: (r[1], r[2]) for r in reduced["wing_connections"]["data"]}
    pos = {r[0]: np.array(r[1:4]) for r in reduced["wing_particles"]["data"]}
    a, b = ends["strut_5"]
    ea = tube_axial_stiffness([old_mean], structure.pressure_bar)[0]
    assert k["strut_5"] == pytest.approx(ea / np.linalg.norm(pos[a] - pos[b]), rel=1e-9)


def test_a_retired_fem_file_still_reduces_the_same(reduced):
    """The old struc_geometry_FEM_full.yaml converts on the fly (with a warning)."""
    import yaml

    legacy = Path(__file__).resolve().parents[1] / "data" / "legacy" / "struc_geometry_FEM_full.yaml"
    with legacy.open(encoding="utf-8") as handle:
        fem = yaml.safe_load(handle)
    with pytest.warns(DeprecationWarning):
        again = reduce_fem_geometry(fem)
    assert again["bridle_connections"] == reduced["bridle_connections"]
    np.testing.assert_allclose(
        [r[1:] for r in again["wing_particles"]["data"]],
        [r[1:] for r in reduced["wing_particles"]["data"]],
        atol=1e-9,
    )
