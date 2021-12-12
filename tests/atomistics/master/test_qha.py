# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from pyiron_atomistics._tests import TestWithCleanProject
from pyiron_atomistics.atomistics.master.qha import Hessian
from pyiron_atomistics.atomistics.structure.atoms import CrystalStructure
import unittest


class TestQuasi(TestWithCleanProject):
    def test_validate_ready_to_run(self):
        lmp = self.project.create.job.Lammps('lmp')
        lmp.structure = self.project.create.structure.bulk('Al', cubic=True)
        qha = lmp.create_job('QuasiHarmonicApproximation', 'qha')
        self.assertEqual(qha.validate_ready_to_run(), None)
        lmp.calc_minimize()
        self.assertRaises(ValueError, qha.validate_ready_to_run)

# class TestHessian(unittest.TestCase):
#     def test_n_snapshots(self):
#         hessian = Hessian(CrystalStructure('Al', 'bcc', 4.0))
#         self.assertEqual(hessian.displacements.shape, ((2,) + hessian.structure.positions.shape))


if __name__ == "__main__":
    unittest.main()
