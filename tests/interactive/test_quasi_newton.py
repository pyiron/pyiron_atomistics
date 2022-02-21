import os
import numpy as np
import unittest
from pyiron_atomistics.project import Project
from pyiron_atomistics.interactive.quasi_newton import QuasiNewtonInteractive, run_qn


class TestQuasiNewton(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.file_location = os.path.dirname(os.path.abspath(__file__))
        cls.project = Project(os.path.join(cls.file_location, 'qn'))

    @classmethod
    def tearDownClass(cls):
        cls.file_location = os.path.dirname(os.path.abspath(__file__))
        cls.project = Project(os.path.join(cls.file_location, 'qn'))
        cls.project.remove_jobs_silently(recursive=True)
        cls.project.remove(enable=True)

    def test_run_qn_regularization(self):
        lj = self.project.create.job.AtomisticExampleJob('exp_reg')
        lj.structure = self.project.create.structure.bulk('Al', cubic=True).repeat(3)
        lj.structure.positions[-1, -1] += 0.
        lj.interactive_open()
        ionic_force_tolerance = 0.01
        qn = run_qn(
            lj, mode='PSB', ionic_force_tolerance=ionic_force_tolerance, max_displacement=0.01
        )
        self.assertLess(lj.output.force_max[-1], ionic_force_tolerance)
        dx = qn.get_dx(np.random.randn(*lj.structure.positions.shape))
        self.assertLess(np.absolute(dx).max(), 0.01)

    def test_job(self):
        lj = self.project.create.job.AtomisticExampleJob('exp')
        lj.structure = self.project.create.structure.bulk('Al', cubic=True).repeat(3)
        qn = lj.create_job('QuasiNewton', 'qn')
        self.assertTrue(qn.input.symmetrize)


if __name__ == '__main__':
    unittest.main()
