# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from pyiron_atomistics._tests import TestWithCleanProject
import unittest


class TestQuasi(TestWithCleanProject):
    def test_validate_ready_to_run(self):
        lmp = self.project.create.job.Lammps('lmp_phono')
        lmp.structure = self.project.create.structure.bulk('Al', cubic=True)
        phono = lmp.create_job('PhonopyJob', 'phono')
        qha = phono.create_job('QuasiHarmonicJob', 'quasi')
        self.assertEqual(qha.validate_ready_to_run(), None)


if __name__ == "__main__":
    unittest.main()
