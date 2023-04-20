# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from pyiron_atomistics.lammps.potentials import Library, Morse, CustomPotential, LammpsPotentials
import unittest
import pandas as pd
import numpy as np


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

    def test_initialize_df(self):
        pot = LammpsPotentials()
        pot._initialize_df(
            pair_style=["some_potential"],
            interacting_species=[["Al", "Al"]],
            pair_coeff=["something"],
        )
        self.assertIsInstance(pot.df, pd.DataFrame)
        self.assertEqual(len(pot.df), 1)
        self.assertEqual(pot.df.iloc[0].pair_style, "some_potential")
        self.assertEqual(pot.df.iloc[0].interacting_species, ["Al", "Al"])
        self.assertEqual(pot.df.iloc[0].pair_coeff, "something")
        self.assertEqual(pot.df.iloc[0].preset_species, [])
        self.assertEqual(pot.df.iloc[0].cutoff, 0)
        self.assertEqual(pot.df.iloc[0].model, "some_potential")
        self.assertEqual(pot.df.iloc[0].citations, [])
        self.assertEqual(pot.df.iloc[0].filename, "")
        self.assertEqual(pot.df.iloc[0].potential_name, "some_potential")
        with self.assertRaises(ValueError):
            pot._initialize_df(
                pair_style=["some_potential", "one_too_many"],
                interacting_species=[["Al", "Al"]],
                pair_coeff=["something"],
            )

    def test_custom_potential(self):
        pot = CustomPotential("lj/cut", "Al", "Ni", epsilon=0.5, sigma=1, cutoff=3)
        self.assertEqual(pot.df.iloc[0].pair_coeff, "0.5 1 3")

    def test_copy(self):
        pot_1 = CustomPotential("lj/cut", "Al", "Ni", epsilon=0.5, sigma=1, cutoff=3)
        pot_2 = pot_1.copy()
        self.assertTrue(np.all(pot_1.df == pot_2.df))
        pot_2.df.cutoff = 1
        self.assertFalse(np.all(pot_1.df == pot_2.df))

if __name__ == "__main__":
    unittest.main()
