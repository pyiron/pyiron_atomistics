# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import os
import numpy as np
import unittest
from pathlib import Path

from sphinx_parser.output import collect_energy_dat


class TestSphinx(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = Path(__file__).parent.absolute() / "../static/sphinx"

    def get_path(self, name):
        return self.directory / f"{name}_hdf5" / f"{name}"

    def test_energy(self):
        E_dict = collect_energy_dat(
            file_name=os.path.join(self.get_path("sphinx_test_2_5"), "energy.dat"),
        )
        self.assertTrue(
            np.all(E_dict["scf_energy_free"][-1] < E_dict["scf_energy_int"][-1])
        )


if __name__ == "__main__":
    unittest.main()
