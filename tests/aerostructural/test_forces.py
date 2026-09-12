"""Unit tests for awetrim.aerostructural.forces

distribute_total_force_by_particle_mass spreads a single 3D resultant (inertial
or gravity) over the structural nodes in proportion to their mass. The split
must conserve the total force exactly.
"""

import numpy as np
import pytest

from awetrim.aerostructural.forces import distribute_total_force_by_particle_mass
from awetrim.aerostructural.utils import calculate_cg


class TestDistributeTotalForce:
    def test_output_shape(self):
        out = distribute_total_force_by_particle_mass([0.0, 0.0, -10.0], [1.0, 1.0, 2.0])
        assert out.shape == (3, 3)

    def test_conserves_total_force(self):
        # Sum the nodal forces with an elementwise add rather than
        # ``out.sum(axis=0)``: the reduction form forwards numpy's ``_NoValue``
        # sentinel, which breaks when casadi has re-imported numpy under some
        # test/coverage import orders.
        out = distribute_total_force_by_particle_mass([1.0, -2.0, -10.0], [1.0, 1.0, 2.0])
        total_out = out[0] + out[1] + out[2]
        assert total_out == pytest.approx([1.0, -2.0, -10.0])

    def test_force_proportional_to_mass(self):
        out = distribute_total_force_by_particle_mass([0.0, 0.0, -8.0], [1.0, 3.0])
        # Node masses 1:3 -> z-forces split -2 and -6.
        assert out[0, 2] == pytest.approx(-2.0)
        assert out[1, 2] == pytest.approx(-6.0)

    def test_equal_masses_split_evenly(self):
        out = distribute_total_force_by_particle_mass([6.0, 0.0, 0.0], [2.0, 2.0])
        assert out[0, 0] == pytest.approx(3.0)
        assert out[1, 0] == pytest.approx(3.0)

    def test_zero_total_mass_raises(self):
        with pytest.raises(ValueError, match="mass must be positive"):
            distribute_total_force_by_particle_mass([0.0, 0.0, -1.0], [0.0, 0.0])

    def test_moment_matches_the_cg_the_trim_applied_the_force_at(self):
        """The split has to reproduce the resultant at the CG, not just its sum.

        The trim applies the inertial and gravity resultants as point loads at
        ``calculate_cg``; the drivers hand the structure this split instead.
        The two are the same load only if both weigh the nodes the same way --
        including nodes whose lumped mass came out NEGATIVE, as 96 tube nodes
        of the LEI-V3 FEM_full geometry do. Weighting them at zero instead
        moved the implied CG 0.64 m in z there, a 20% error on every inertial
        moment the structure received.
        """
        nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 2.0], [0.0, 1.0, 4.0]])
        masses = np.array([2.0, 1.5, -0.25])
        force = np.array([0.0, 300.0, 0.0])

        nodal = distribute_total_force_by_particle_mass(force, masses)
        moment = np.cross(nodes, nodal).sum(axis=0)
        np.testing.assert_allclose(
            moment, np.cross(calculate_cg(nodes, masses), force), atol=1e-9
        )
