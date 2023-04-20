# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from pyiron_atomistics.lammps.potentials import Library, Morse, CustomPotential, LammpsPotentials
import unittest
import pandas as pd


class TestPotentials(unittest.TestCase):
    def test_harmonize_species(self):
        pot = LammpsPotentials()
        self.assertEqual(pot._harmonize_species(("Al",)), ["Al", "Al"])
        for i in [2, 3, 4]:
            self.assertEqual(pot._harmonize_species(i * ("Al",)), i * ["Al"])
        self.assertRaises(ValueError, pot._harmonize_species, tuple())

    def test_set_df(self):
        pot = LammpsPotentials()
        self.assertEqual(pot.df, None)
        required_keys = [
            "pair_style",
            "interacting_species",
            "pair_coeff",
            "preset_species",
        ]
        arg_dict = {k: [] for k in required_keys}
        pot.set_df(pd.DataFrame(arg_dict))
        self.assertIsInstance(pot.df, pd.DataFrame)
        for key in required_keys:
            arg_dict = {k: [] for k in required_keys if k != key}
            self.assertRaises(ValueError, pot.set_df, pd.DataFrame(arg_dict))


if __name__ == "__main__":
    unittest.main()
