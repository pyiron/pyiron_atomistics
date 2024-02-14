from unittest import TestCase
from pyiron_atomistics.atomistics.structure.atoms import Atoms
from pyiron_atomistics.atomistics.structure.pyxtal import pyxtal
from pyiron_atomistics.atomistics.structure.structurestorage import StructureStorage

class TestPyxtal(TestCase):

    def test_args_raised(self):
        """pyxtal should raise appropriate errors when called with wrong arguments"""

        with self.assertRaises(ValueError, msg="No error raised when num_ions and species do not match!"):
            pyxtal(1, species=['Fe'], num_ions=[1,2])

        with self.assertRaises(ValueError, msg="No error raised when num_ions and species do not match!"):
            pyxtal(1, species=['Fe', 'Cr'], num_ions=[1])

        try:
            pyxtal([193, 194], ['Mg'], num_ions=[1], allow_exceptions=True)
        except ValueError:
            self.fail("Error raised even though allow_exceptions=True was passed!")

        with self.assertRaises(ValueError, msg="No error raised even though allow_exceptions=False was passed!"):
            pyxtal(194, ['Mg'], num_ions=[1], allow_exceptions=False)

    def test_return_value(self):
        """pyxtal should either return Atoms or StructureStorage, depending on arguments"""

        self.assertIsInstance(pyxtal(1, species=['Fe'], num_ions=[1]), Atoms,
                              "returned not an Atoms with scalar arguments")
        self.assertIsInstance(pyxtal([1,2], species=['Fe'], num_ions=[1]), StructureStorage,
                              "returned not a StructureStorage with multiple groups")
        self.assertIsInstance(pyxtal(1, species=['Fe'], num_ions=[1], repeat=5), StructureStorage,
                              "returned not a StructureStorage with repeat given")
        self.assertEqual(pyxtal(1, species=['Fe'], num_ions=[1], repeat=5).number_of_structures, 5,
                         "returned number of structures did not match given repeat")
