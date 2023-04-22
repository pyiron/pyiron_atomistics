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

    def test_model(self):
        pot = CustomPotential("lj/cut", "Al", "Ni", epsilon=0.5, sigma=1, cutoff=3)
        self.assertEqual(pot.model, "lj/cut")
        pot = LammpsPotentials()
        pot._initialize_df(
            pair_style=["a", "b"],
            interacting_species=[["Al"], ["Ni"]],
            pair_coeff=["one", "two"],
            model=["first", "second"]
        )
        self.assertEqual(pot.model, "first_and_second")

    def test_potential_name(self):
        pot = CustomPotential("lj/cut", "Al", "Ni", epsilon=0.5, sigma=1, cutoff=3)
        self.assertEqual(pot.potential_name, "lj/cut")
        pot = LammpsPotentials()
        pot._initialize_df(
            pair_style=["a", "b"],
            interacting_species=[["Al"], ["Ni"]],
            pair_coeff=["one", "two"],
            potential_name=["first", "second"]
        )
        self.assertEqual(pot.potential_name, "first_and_second")

    def test_is_scaled(self):
        pot = CustomPotential("lj/cut", "Al", "Ni", epsilon=0.5, sigma=1, cutoff=3)
        self.assertFalse(pot.is_scaled)

    def test_unique(self):
        pot = LammpsPotentials()
        self.assertEqual(pot._unique([1, 0, 2, 1, 3]).tolist(), [1, 0, 2, 3])

    def test_pair_style(self):
        pot = LammpsPotentials()
        pot._initialize_df(
            pair_style=["a"],
            interacting_species=[["Al"]],
            pair_coeff=["one"],
            potential_name=["first"]
        )
        self.assertEqual(pot.pair_style, "pair_style a\n")
        pot._initialize_df(
            pair_style=["a", "b"],
            interacting_species=[["Al"], ["Ni"]],
            pair_coeff=["one", "two"],
            potential_name=["first", "second"]
        )
        self.assertEqual(pot.pair_style, "pair_style hybrid a b\n")
        pot._initialize_df(
            pair_style=["a", "b"],
            interacting_species=[["Al"], ["Ni"]],
            pair_coeff=["one", "two"],
            potential_name=["first", "second"],
            cutoff=[1, 1],
        )
        self.assertEqual(pot.pair_style, "pair_style hybrid a 1 b 1\n")
        pot._initialize_df(
            pair_style=["a", "b"],
            interacting_species=[["Al"], ["Ni"]],
            pair_coeff=["one", "two"],
            potential_name=["first", "second"],
            scale=[1, 1],
            cutoff=[1, 1],
        )
        self.assertEqual(pot.pair_style, "pair_style hybrid/overlay a 1 b 1\n")
        pot._initialize_df(
            pair_style=["a", "b"],
            interacting_species=[["Al"], ["Ni"]],
            pair_coeff=["one", "two"],
            potential_name=["first", "second"],
            scale=[1, 0.5],
            cutoff=[1, 1],
        )
        self.assertEqual(pot.pair_style, "pair_style hybrid/scaled 1.0 a 1 0.5 b 1\n")
        pot._initialize_df(
            pair_style=["a", "a"],
            interacting_species=[["Al"], ["Ni"]],
            pair_coeff=["one", "two"],
            potential_name=["first", "second"],
        )
        self.assertEqual(pot.pair_style, "pair_style a\n")

    def test_PairCoeff(self):
        pot = LammpsPotentials()
        pc = pot._PairCoeff(
            is_hybrid=False,
            pair_style=["my_style"],
            interacting_species=[["Al", "Fe"]],
            pair_coeff=["some arguments"],
            species=["Al", "Fe"],
            preset_species=[],
        )
        self.assertEqual(pc.counter, [""])


if __name__ == "__main__":
    unittest.main()
