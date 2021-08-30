# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from pyiron_base._tests import PyironTestCase
import pyiron_atomistics.atomistics.structure.factories.compound as compound_module
from pyiron_atomistics.atomistics.structure.factories.compound import CompoundFactory
from pyiron_atomistics.atomistics.structure.factory import StructureFactory
import numpy as np


class TestCompoundFactory(PyironTestCase):

    @classmethod
    def setUpClass(cls):
        cls.sf = StructureFactory()
        cls.compound = CompoundFactory()

    @property
    def docstring_module(self):
        return compound_module

    def test_B2(self):
        structure = self.compound.B2('Fe', 'Al')
        self.assertAlmostEqual(self.sf.bulk('Fe', cubic=True).cell[0, 0], structure.cell[0, 0],
                         msg="Docstring claims lattice constant defaults to primary species")
        self.assertEqual(2, len(structure))
        neigh = structure.get_neighbors(num_neighbors=8)
        symbols = structure.get_chemical_symbols()
        self.assertEqual(8, np.sum(symbols[neigh.indices[0]] == 'Al'),
                         msg="Expected the primary atom to have all secondary neighbors")
        self.assertEqual(8, np.sum(symbols[neigh.indices[1]] == 'Fe'),
                         msg="Expected the secondary atom to have all primary neighbors")
        structure = self.compound.B2('Fe', 'Al', a=1)
        self.assertTrue(np.allclose(np.diag(structure.cell.array), 1), "Expected cubic cell with specified size.")

    def test_C14(self):
        with self.assertRaises(NotImplementedError):
            self.compound.C14()

    def test_C15(self):
        """
        Tests based on Xie et al., JMR 2021 (DOI:10.1557/s43578-021-00237-y).
        """

        a_type = 'Mg'
        b_type = 'Cu'
        structure = self.compound.C15(a_type, b_type)

        a_type_nn_distance = StructureFactory().bulk(a_type).get_neighbors(num_neighbors=1).distances[0, 0]
        self.assertAlmostEqual((4 / np.sqrt(3)) * a_type_nn_distance, structure.cell.array[0, 0],
                               msg="Default lattice constant should relate to NN distance of A-type element.")

        unique_ids = np.unique(structure.get_symmetry()['equivalent_atoms'])
        self.assertEqual(2, len(unique_ids), msg="Expected only A- and B1-type sites.")

        csa = structure.analyse.pyscal_centro_symmetry()[unique_ids]
        self.assertLess(1, csa[0], msg="Primary A site should be significantly non-symmetric.")
        self.assertAlmostEqual(0, csa[1], msg="Secondary B1 site should be nearly symmetric.")

        num_a_neighs = 16
        num_b_neighs = 12
        neigh = structure.get_neighbors(num_neighbors=num_a_neighs)
        a_neighs = neigh.indices[unique_ids[0]]
        b_neighs = neigh.indices[unique_ids[1], :num_b_neighs]
        symbols = structure.get_chemical_symbols()
        self.assertEqual(4, np.sum(symbols[a_neighs] == a_type))
        self.assertEqual(12, np.sum(symbols[a_neighs] == b_type))
        self.assertEqual(6, np.sum(symbols[b_neighs] == a_type))
        self.assertEqual(6, np.sum(symbols[b_neighs] == b_type))

    def test_C36(self):
        with self.assertRaises(NotImplementedError):
            self.compound.C36()

    def test_D03(self):
        element_a, element_b = 'Al', 'Fe'
        structure = self.compound.D03(element_a, element_b)
        symbols = structure.get_chemical_symbols()
        neigh = structure.get_neighbors(num_neighbors=8)

        a_neighbors = neigh.indices[symbols == element_a]
        self.assertTrue(np.all(symbols[a_neighbors] == element_b), msg="A-type should only have B-type neighbors.")

        b_neighbors = neigh.indices[symbols == element_b]
        sorted_vals, counts = np.unique(np.mean(symbols[b_neighbors] == 'Al', axis=1), return_counts=True)
        self.assertAlmostEqual(0, sorted_vals[0], msg="Shared sub-lattice has no A-type neighbors.")
        self.assertAlmostEqual(0.5, sorted_vals[1], msg="B-type sub-lattice has half A-type neighbors.")
        self.assertEqual(4, counts[0], msg="Shared sub-lattice should be only 1/4 of atoms.")
        self.assertEqual(8, counts[1], msg="Pure-B sub-lattice should be 1/2 of atoms.")

    def test_Heusler(self):
        with self.assertRaises(NotImplementedError):
            self.compound.Heusler('Mg', 'Ca', 'Al')
