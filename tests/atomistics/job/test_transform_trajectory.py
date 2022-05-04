import os
import shutil
import unittest
from pathlib import Path
from pyiron_base._tests import TestWithCleanProject
from pyiron_base import ProjectHDFio
from pyiron_atomistics.atomistics.job.atomistic import Trajectory
from pyiron_atomistics.atomistics.structure.atoms import CrystalStructure
from test_atomistic import ToyAtomisticJob


class TestTransformTrajectory(TestWithCleanProject):

    @classmethod
    def tearDownClass(cls):
        cls.project.remove(enable=True)

    def setUp(self) -> None:
        # super().setUp()
        self.job = ToyAtomisticJob(
            project=ProjectHDFio(project=self.project, file_name="test_job"),
            job_name="test_job",
        )
        self.job.structure = CrystalStructure(
            element="Al", bravais_basis="fcc", lattice_constants=4
        ).repeat(4)
        self.job.run()

    def test_transform_trajectory(self):
        cwd = os.curdir
        tests_loc = Path(__file__).parents[2]
        shutil.copy(os.path.join(tests_loc, "static/lammps_test_files/get_structure_test.tar.gz"), cwd)
        shutil.copy(os.path.join(tests_loc, "static/lammps_test_files/export.csv"), cwd)
        self.project.unpack("get_structure_test")
        traj = self.job.trajectory()
        self.assertIsInstance(traj, Trajectory)
        self.assertEqual(len(traj), len(self.job.output.positions))
        traj = self.job.trajectory()
        traj = traj.transform(lambda s: s.repeat(1))
        self.assertEqual(len(traj.get_structure()), 1*len(self.job.get_structure()))
        traj = self.job.trajectory()
        traj = traj.transform(lambda s: s.repeat(2))
        for s1, s2 in zip(traj.iter_structures(), self.job.iter_structures()):
            self.assertEqual(len(s1), 8 * len(s2))


if __name__ == '__main__':
    unittest.main()
