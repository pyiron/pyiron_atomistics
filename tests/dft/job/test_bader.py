# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import unittest
import numpy as np
import os
from pyiron_atomistics.dft.job.bader import parse_charge_vol_file
from pyiron_atomistics.vasp.structure import read_atoms


class TestBader(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.file_location = os.path.dirname(os.path.abspath(__file__))

    def test_parse_charge_vol(self):
        filename = os.path.join(
            self.file_location, "../../static/dft/bader_files/ACF.dat")
        struct = read_atoms(os.path.join(self.file_location, "../../static/vasp_test_files/bader_test/POSCAR"))
        charges, volumes = parse_charge_vol_file(structure=struct, filename=filename)
        self.assertTrue(np.array_equal(charges, [0.438202, 0.438197, 7.143794]))
        self.assertTrue(np.array_equal(volumes, [287.284690, 297.577878, 415.155432]))
