# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from pyiron_base._tests import TestWithProject


class TestGpaw(TestWithProject):
    def test_interactive_run(self):
        """
        Gpaw should run interactively, even if you update the structure to a new object.
        """
        job = self.project.create.job.Gpaw("gpaw", delete_existing_job=True)
        job.input["encut"] = 100
        job.input["kpoints"] = 3 * [1]

        s1 = self.project.atomistics.structure.bulk("Al", cubic=True)
        s2 = self.project.atomistics.structure.bulk("Al", cubic=True)
        s2.positions[0, 0] += 0.2

        job.structure = s1
        with job.interactive_open() as ijob:
            ijob.run()
            ijob.structure = s2
            ijob.run()

    def test_interface_initialization(self):
        """
        Make sure that you can initialize the interactive interface without having
        already run the code.
        """
        job = self.project.create.job.Gpaw("gpaw", delete_existing_job=True)
        job.input["encut"] = 100
        job.input["kpoints"] = 3 * [1]
        job.structure = self.project.atomistics.structure.bulk("Al", cubic=True)
        job.interactive_open()
        job.interactive_initialize_interface()
