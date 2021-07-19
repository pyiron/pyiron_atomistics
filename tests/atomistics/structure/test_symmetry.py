# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import unittest
import numpy as np
from pyiron_atomistics.atomistics.structure.atoms import Atoms, CrystalStructure
from pyiron_atomistics.atomistics.structure.factory import StructureFactory

class TestAtoms(unittest.TestCase):
    def test_get_arg_equivalent_sites(self):
        a_0 = 4.0
        structure = StructureFactory().ase.bulk('Al', cubic=True, a=a_0).repeat(2)
        sites = structure.get_wrapped_coordinates(structure.positions+np.array([0, 0, 0.5*a_0]))
        v_position = structure.positions[0]
        del structure[0]
        pairs = np.stack((
            structure.get_symmetry().get_arg_equivalent_sites(sites),
            np.unique(np.round(structure.get_distances_array(v_position, sites), decimals=2), return_inverse=True)[1]
        ), axis=-1)
        unique_pairs = np.unique(pairs, axis=0)
        self.assertEqual(len(unique_pairs), len(np.unique(unique_pairs[:,0])))

    def test_generate_equivalent_points(self):
        a_0 = 4
        structure = StructureFactory().ase.bulk('Al', cubic=True, a=a_0)
        self.assertEqual(
            len(structure),
            len(structure.get_symmetry().generate_equivalent_points([0, 0, 0.5*a_0]))
        )


if __name__ == "__main__":
    unittest.main()
