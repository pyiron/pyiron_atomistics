import os
import shutil
import unittest
from abc import ABC
import numpy as np
from pathlib import Path
from pyiron_base._tests import TestWithCleanProject
from pyiron_base import ProjectHDFio
from pyiron_atomistics.atomistics.job.atomistic import AtomisticGenericJob, Trajectory, TransformTrajectory
from pyiron_atomistics.atomistics.structure.atoms import Atoms, CrystalStructure


class ToyAtomisticJob(AtomisticGenericJob, ABC):

    def _check_if_input_should_be_written(self):
        return False

    def run_static(self):
        self.save()
        self.status.running = True
        self.status.finished = True
        self.to_hdf()

    def to_hdf(self, hdf=None, group_name=None):
        super().to_hdf(hdf=hdf, group_name=group_name)
        # create some dummy output
        n_steps = 10
        with self.project_hdf5.open("output/generic") as h_out:
            h_out["positions"] = np.array([self.structure.positions + 0.5 * i for i in range(n_steps)])
            h_out["cells"] = np.array([self.structure.cell] * n_steps)
            h_out["indices"] = np.zeros((n_steps, len(self.structure)), dtype=int)


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
        traj = traj.transform(lambda s: s.repeat(1))
        self.assertIsInstance(traj, TransformTrajectory)
        self.assertEqual(len(traj), 1*len(self.job.output.positions))
        traj = self.job.trajectory()
        traj = traj.transform(lambda s: s.repeat(2))
        self.assertEqual(len(traj.get_structure()), 8*len(self.job.get_structure()))
        traj = self.job.trajectory()
        traj = traj.transform(lambda s: s.repeat(1))
        self.assertEqual(len(traj.get_structure()), 1*len(self.job.get_structure()))
        traj = self.job.trajectory()
        traj = traj.transform(lambda s: s.repeat(2))
        for i in traj.iter_structures():
            self.assertEqual(len(i), 8*len(self.job.get_structure()))


if __name__ == '__main__':
    unittest.main()
