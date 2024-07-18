# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import numpy as np
import unittest
from pathlib import Path

from pyiron_atomistics.sphinx.output_parser import (
    SphinxLogParser,
    collect_energy_dat,
    collect_residue_dat,
    collect_spins_dat,
    collect_relaxed_hist,
    collect_energy_struct,
    collect_eps_dat,
    SphinxWavesReader,
)


class TestSphinx(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = Path(__file__).parent.absolute() / "../static/sphinx"

    def get_path(self, name):
        return self.directory / f"{name}_hdf5" / f"{name}"

    def test_energy(self):
        E_dict = collect_energy_dat(
            file_name="energy.dat", cwd=self.get_path("sphinx_test_2_5")
        )
        self.assertTrue(
            np.all(E_dict["scf_energy_free"][-1] < E_dict["scf_energy_int"][-1])
        )
        E_dict = collect_energy_dat(
            file_name="energy.dat", cwd=self.get_path("sphinx_test_2_3")
        )
        self.assertTrue("scf_energy_free" not in E_dict)

    def test_waves(self):
        test_path = Path(__file__).parents[2 - 1]
        waves_path = Path.joinpath(test_path, "static/sphinx/sphinx_test_waves")
        waves = SphinxWavesReader("waves.sxb", cwd=waves_path)
        self.assertTrue(waves.nk == 1)
        self.assertAlmostEqual(
            np.linalg.norm(waves.get_psi_rec(0, 0, 0)), 0.9984249664645706
        )
        self.assertEqual(waves.get_psi_rec(0, 0, 0).shape, (10, 10, 10))


if __name__ == "__main__":
    unittest.main()
