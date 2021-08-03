# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import unittest
import numpy as np
import time
from pyiron_atomistics.atomistics.structure.atoms import Atoms
from pyiron_atomistics.atomistics.structure.periodic_table import element, PeriodicTable


class TestAtoms(unittest.TestCase):
    def test_cached_speed(self):
        """
        Creating atoms should be faster after the first time, due to caches in periodictable/mendeleev.
        """
        pos, cell = generate_fcc_lattice()
        expected_speedup_factor = 15
        n_timing_loop = 5
        t1, t2, t3, t4, t5, t6, t7 = [np.array([0.0]*n_timing_loop) for _ in range(7)]
        for i in range(n_timing_loop):
            element.cache_clear()
            PeriodicTable._get_periodic_table_df.cache_clear()
            t1[i] = time.perf_counter()
            Atoms(symbols="Al", positions=pos, cell=cell)
            t2[i] = time.perf_counter()
            Atoms(symbols="Al", positions=pos, cell=cell)
            t3[i] = time.perf_counter()
            Atoms(symbols="Cu", positions=pos, cell=cell)
            t4[i] = time.perf_counter()
            Atoms(symbols="CuAl", positions=[[0., 0., 0.], [0.5, 0.5, 0.5]], cell=cell)
            t5[i] = time.perf_counter()
            Atoms(symbols="MgO", positions=[[0., 0., 0.], [0.5, 0.5, 0.5]], cell=cell)
            t6[i] = time.perf_counter()
            Atoms(symbols="AlMgO", positions=[[0., 0., 0.], [0.5, 0.5, 0.5], [0.5, 0.5, 0.]], cell=cell)
            t7[i] = time.perf_counter()
        dt21 = np.mean(t2 - t1)
        dt32 = np.mean(t3 - t2)
        # check the simple case of structures with one element type
        self.assertGreater(dt21, dt32, "Atom creation not speed up by caches!")
        self.assertGreater(dt21 / dt32, expected_speedup_factor,
                           "Atom creation not speed up to the required level by caches!")
        dt43 = np.mean(t4 - t3)
        dt54 = np.mean(t5 - t4)
        # check that speed up also holds when creating structures with multiple elements, but all the elements have been
        # seen before
        self.assertGreater(dt43 / dt54, expected_speedup_factor,
                            "Atom creation not speed up to the required level by caches!")
        dt65 = np.mean(t6 - t5)
        dt76 = np.mean(t7 - t6)
        # check that again with three elements
        self.assertGreater(dt65 / dt76, expected_speedup_factor,
                            "Atom creation not speed up to the required level by caches!")


def generate_fcc_lattice(a=4.2):
    positions = [[0, 0, 0]]
    cell = (np.ones((3, 3)) - np.eye(3)) * 0.5 * a
    return positions, cell


if __name__ == "__main__":
    unittest.main()
