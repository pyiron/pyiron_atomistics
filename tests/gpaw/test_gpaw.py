# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import os
import numpy as np
from pyiron_base import Project, ProjectHDFio
from pyiron_atomistics.gpaw.gpaw import Gpaw
from pyiron_atomistics.atomistics.structure.atoms import Atoms
from pyiron_base._tests import TestWithProject


class TestGpaw(TestWithProject):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.execution_path = os.path.dirname(os.path.abspath(__file__))
        cls.project = Project(os.path.join(cls.execution_path, "gpaw"))
        atoms = Atoms("Fe1", positions=np.zeros((1, 3)), cell=np.eye(3))
        job = Gpaw(
            project=ProjectHDFio(project=cls.project, file_name=cls.project.path),
            job_name="gpaw",
        )
        job.structure = atoms
        job.encut = 300
        job.set_kpoints([5, 5, 5])
        cls.job = job

    def test_serialization(self):
        self.job.to_hdf()
        loaded = Gpaw(
            project=ProjectHDFio(project=self.project, file_name=self.project.path),
            job_name="gpaw",
        )
        loaded.from_hdf()
        self.assertEqual(self.job.encut, loaded.encut)

    def test_encut(self):
        self.assertEqual(self.job.encut, 300)

    def test_kpoint_mesh(self):
        self.assertEqual(self.job.input["kpoints"], [5, 5, 5])
