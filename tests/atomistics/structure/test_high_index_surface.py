import unittest
from pyiron_atomistics.atomistics.structure.factory import StructureFactory


class TestHighIndexSurface(unittest.TestCase):
    def test_high_index_surface(self):
        slab = StructureFactory().high_index_surface(element='Ni', crystal_structure='fcc',
                                                     lattice_constant=3.526,
                                                     terrace_orientation=[1, 1, 1],
                                                     step_orientation=[1, 1, 0],
                                                     kink_orientation=[1, 0, 1],
                                                     step_down_vector=[1, 1, 0], length_step=2,
                                                     length_terrace=3,
                                                     length_kink=1, layers=60,
                                                     vacuum=10)

        self.assertEqual(len(slab), 60)

    def test_high_index_surface_info(self):
        h, s, k = StructureFactory().high_index_surface_info(element='Ni', crystal_structure='fcc',
                                                             lattice_constant=3.526,
                                                             terrace_orientation=[1, 1, 1],
                                                             step_orientation=[1, 1, 0],
                                                             kink_orientation=[1, 0, 1],
                                                             step_down_vector=[1, 1, 0], length_step=2,
                                                             length_terrace=3,
                                                             length_kink=1)
        self.assertEqual(len(h), 3)
        self.assertEqual(h[0], -9)
        self.assertEqual(len(k), 3)
        self.assertEqual(len(s), 3)
        with self.assertRaises(ValueError):
            StructureFactory().high_index_surface_info(element='Ni', crystal_structure='fcc',
                                                       lattice_constant=3.526,
                                                       terrace_orientation=[1, 1, 1],
                                                       step_orientation=[1, 0, 0],
                                                       kink_orientation=[1, 0, 0],
                                                       step_down_vector=[1, 1, 0], length_step=2,
                                                       length_terrace=3,
                                                       length_kink=1)


if __name__ == '__main__':
    unittest.main()
