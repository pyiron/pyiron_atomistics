# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import unittest
import os
from pyiron_atomistics.vasp.potential import get_enmax_among_potentials, strip_xc_from_potential_name
from pyiron_base import state


class TestPotential(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        state.update({'resource_paths': os.path.join(os.path.dirname(os.path.abspath(__file__)), "../static")})

    @classmethod
    def tearDownClass(cls) -> None:
        state.update()

    def test_get_enmax_among_potentials(self):
        float_out = get_enmax_among_potentials('Fe', return_list=False)
        self.assertTrue(isinstance(float_out, float))

        tuple_out = get_enmax_among_potentials('Fe', return_list=True)
        self.assertTrue(isinstance(tuple_out, tuple))

        self.assertRaises(ValueError, get_enmax_among_potentials, 'X')
        self.assertRaises(ValueError, get_enmax_among_potentials, 'Fe', xc='FOO')

    def test_strip_xc_from_potential_name(self):
        self.assertEqual(strip_xc_from_potential_name('X_pv-gga-pbe'), 'X_pv')
