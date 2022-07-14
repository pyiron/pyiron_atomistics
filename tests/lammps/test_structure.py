# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import numpy as np
import os
from pyiron_base import state, ProjectHDFio
from pyiron_atomistics.lammps.lammps import Lammps
from pyiron_base._tests import TestWithCleanProject
from pyiron_atomistics.project import Creator

# Lammps and pyiron structure clearly require more tests
class TestLammpsStructure(TestWithCleanProject):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.execution_path = os.path.dirname(os.path.abspath(__file__))
        state.update(
            {
                "resource_paths": os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "../static"
                )
            }
        )

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        state.update()

    def setUp(self) -> None:
        super().setUp()
        self.job = Lammps(
            project=ProjectHDFio(project=self.project, file_name="lammps"),
            job_name="lammps",
        )
        self.ref = Lammps(
            project=ProjectHDFio(project=self.project, file_name="ref"),
            job_name="ref",
        )
        # Creating jobs this way puts them at the right spot, but decouples them from our self.project instance.
        # I still don't understand what's happening as deeply as I'd like (at all!) but I've been fighting with it too
        # long, so for now I will just force the issue by redefining the project attribute(s). -Liam Huber
        self.project = self.job.project
        self.ref_project = self.ref.project

    def tearDown(self) -> None:
        super().tearDown()
        self.ref_project.remove_jobs_silently(recursive=True)  # cf. comment in setUp

    def test_velocity_basics(self):
        creator = Creator(self.project)
        self.job.structure = creator.structure.ase.bulk("Cu")
        self.assertTrue(
            self.job.structure.velocities is None,
            msg="Initial velocties of structure are not None",
        )
        with self.assertRaises(
            ValueError,
            msg="Setting velocities with a different shape than positions should raise",
        ):
            self.job.structure.velocities = np.array(
                [
                    [1.0, 1.0, -1.0],
                    [3.0, 2.0, -1.0],
                ],
            )
        vels = np.array([[1.0, 1.0, 1.0]])
        self.job.structure.velocities = vels
        self.assertTrue(
            np.allclose(
                self.job.structure.velocities,
                vels
            ),
            msg="Velocties of structure are not correctly set",
        )
