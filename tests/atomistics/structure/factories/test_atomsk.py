# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from pyiron_atomistics._tests import TestWithProject
from pyiron_atomistics.atomistics.structure.factories.atomsk import AtomskFactory, AtomskError, _ATOMSK_EXISTS

if _ATOMSK_EXISTS is not None:
    class TestAtomskFactory(TestWithProject):
        @classmethod
        def setUpClass(cls):
            super().setUpClass()
            cls.atomsk = AtomskFactory()

        def test_create_fcc_cubic(self):
            """Should create cubic fcc cell."""

            try:
                structure = self.atomsk.create('fcc', 3.6, 'Cu').build()
            except Exception as e:
                self.fail(f"atomsk create fails with {e}.")

            self.assertEqual(len(structure), 4, "Wrong number of atoms.")
            self.assertTrue(all(structure.cell[i, i] == 3.6 for i in range(3)),
                            "Wrong lattice parameters {structure.cell} != 3.6.")

            try:
                duplicate = self.atomsk.create('fcc', 3.6, 'Cu').duplicate(2, 1, 1).build()
            except Exception as e:
                self.fail(f"chaining duplicate after atomsk create fails with {e}.")

            self.assertEqual(len(duplicate), 8, "Wrong number of atoms.")
            self.assertEqual(duplicate.cell[0, 0], 7.2,
                            "Wrong lattice parameter in duplicate direction.")

        def test_modify(self):
            """Should correctly modify passed in structures."""

            structure = self.atomsk.create('fcc', 3.6, 'Cu').build()
            try:
                duplicate = self.atomsk.modify(structure).duplicate(2, 1, 1).build()
            except Exception as e:
                self.fail(f"atomsk modify fails with {e}.")

            self.assertEqual(len(structure) * 2, len(duplicate), "Wrong number of atoms.")

        def test_error(self):
            """Should raise AtomskError on errors during call to atomsk."""

            try:
                self.atomsk.create('fcc', 3.6, 'Cu').foo_bar(42).build()
            except AtomskError:
                pass
            except Exception as e:
                self.fail(f"Wrong error {e} raised on non-existing option passed.")
            else:
                self.fail("No error raised on non-existing option passed.")
