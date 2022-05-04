# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import unittest
import numpy as np
import os
import shutil
from pathlib import Path
from pyiron_base import ProjectHDFio
from pyiron_base._tests import TestWithCleanProject
from pyiron_atomistics.atomistics.job.atomistic import AtomisticGenericJob, Trajectory
from pyiron_atomistics.atomistics.structure.atoms import Atoms, CrystalStructure
import warnings


class ToyAtomisticJob(AtomisticGenericJob):

    def _check_if_input_should_be_written(self):
        return False

    def run_static(self):
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


class TestAtomisticGenericJob(TestWithCleanProject):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.execution_path = os.path.dirname(os.path.abspath(__file__))

    def setUp(self) -> None:
        super().setUp()
        self.job = ToyAtomisticJob(
            project=ProjectHDFio(project=self.project, file_name="test_job"),
            job_name="test_job",
        )
        self.job.structure = CrystalStructure(
            element="Al", bravais_basis="fcc", lattice_constants=4
        ).repeat(4)
        self.job.run()

    def test_attributes(self):
        self.assertIsInstance(self.job.structure, Atoms)

    def test_get_displacements(self):
        n_steps = 10
        # increasing each position by 0.5 at each step
        positions = np.array([self.job.structure.positions + 0.5 * i for i in range(n_steps)])
        # constant cell
        cells = np.array([self.job.structure.cell] * n_steps)
        disp = self.job.output.get_displacements(self.job.structure, positions=positions, cells=cells)
        disp_ref = np.ones_like(positions) * 0.5
        disp_ref[0] *= 0.0
        self.assertTrue(np.allclose(disp, disp_ref))
        # varying cell
        cells = np.array([self.job.structure.cell * ((i+1) / 10) for i in range(n_steps)])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            disp = self.job.output.get_displacements(self.job.structure,
                                                     positions=positions, cells=cells)
            self.assertEqual(len(w), 1)
            self.assertIsInstance(w[-1].message, UserWarning)
        self.assertFalse(np.allclose(disp, disp_ref))
        dummy_struct = self.job.structure.copy()
        disp_ref = list()
        for pos, cell in zip(positions, cells):
            pos_init = dummy_struct.get_scaled_positions().copy()
            dummy_struct.set_cell(cell, scale_atoms=False)
            dummy_struct.positions = pos
            dummy_struct.center_coordinates_in_unit_cell()
            diff = dummy_struct.get_scaled_positions()-pos_init
            diff[diff >= 0.5] -= 1.0
            diff[diff <= -0.5] += 1.0
            disp_ref.append(np.dot(diff, cell))
        self.assertTrue(np.allclose(disp, disp_ref))

    @unittest.skipIf(os.name == 'nt', "Runs forever on Windows")
    def test_get_structure(self):
        """get_structure() should return structures with the exact values from the HDF files even if the size of
        structures changes."""

        # tested here with lammps as a concrete instantiation of AtomisticGenericJob
        # have to do extra tango because Project.unpack is weird right now
        cwd = os.curdir
        tests_loc = Path(__file__).parents[2]
        shutil.copy(os.path.join(tests_loc, "static/lammps_test_files/get_structure_test.tar.gz"), cwd)
        shutil.copy(os.path.join(tests_loc, "static/lammps_test_files/export.csv"), cwd)
        self.project.unpack("get_structure_test")
        job = self.project.load("inter_calculator")
        os.unlink("export.csv")
        os.unlink("get_structure_test.tar.gz")

        for i, struct in enumerate(job.iter_structures()):
            # breakpoint()
            self.assertTrue(np.allclose(job.output.positions[i], struct.positions))
            self.assertTrue(np.allclose(job.output.cells[i], struct.cell.array))
            self.assertTrue(np.allclose(job.output.indices[i], struct.indices))

    def test_animate_structure(self):
        traj = self.job.trajectory()
        self.assertIsInstance(traj, Trajectory)
        self.assertEqual(len(traj), len(self.job.output.positions))
        traj = self.job.trajectory(atom_indices=[3, 5], snapshot_indices=[3, 4])
        self.assertEqual(len(traj), 2)
