# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from pyiron_atomistics.lammps.potentials import Library, Morse, CustomPotential, LammpsPotentials
import unittest


class TestPotentials(unittest.TestCase):
    def test_harmonize_species(self):
        pot = LammpsPotentials()
        self.assertEqual(pot._harmonize_species(("Al",)), ["Al", "Al"])
        for i in [2, 3, 4]:
            self.assertEqual(pot._harmonize_species(i * ("Al",)), i * ["Al"])
        self.assertRaises(ValueError, pot._harmonize_species, tuple())


if __name__ == "__main__":
    unittest.main()
