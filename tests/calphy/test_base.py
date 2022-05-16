# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import os
import unittest
import shutil

from pyiron_atomistics.project import Project
from pyiron_atomistics.calphy.job import Calphy
from pyiron_base import state, ProjectHDFio


class TestCalphy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        state.update(
            {
                "resource_paths": os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "../static/calphy_test_files"
                )
            }
        )
        cls.execution_path = os.path.dirname(os.path.abspath(__file__))
        # cls.file_location = os.path.dirname(os.path.abspath(__file__))
        cls.project = Project(os.path.join(cls.execution_path, "test_calphy"))
        cls.job = Calphy(
            project=ProjectHDFio(project=cls.project, file_name="test_calphy"),
            job_name="test_calphy",
        )
        filepath = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../static/"
        )
        cls.output_project = Project(os.path.join(filepath, "test_files"))

    @classmethod
    def tearDownClass(cls):
        cls.execution_path = os.path.dirname(os.path.abspath(__file__))
        project = Project(os.path.join(cls.execution_path, "test_calphy"))
        project.remove_jobs(silently=True, recursive=True)
        project.remove(enable=True)
        filepath = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../static/"
        )
        out_project = Project(os.path.join(filepath, "test_files"))
        out_project.remove_jobs(silently=True, recursive=True)
        out_project.remove(enable=True)
        state.update()

    def test_potentials(self):
        self.job.set_potentials(
            ["2001--Mishin-Y--Cu-1--LAMMPS--ipr5", "2001--Mishin-Y--Cu-1--LAMMPS--ipr5"]
        )
        # print(self.job.input)
        # pint(self.job.input.potential_initial_name)
        self.assertEqual(
            self.job.input.potential_initial_name, "2001--Mishin-Y--Cu-1--LAMMPS--ipr5"
        )
        self.assertEqual(
            self.job.input.potential_final_name, "2001--Mishin-Y--Cu-1--LAMMPS--ipr5"
        )
        self.assertRaises(
            ValueError,
            self.job.set_potentials,
            [
                "2001--Mishin-Y--Cu-1--LAMMPS--ipr5",
                "2001--Mishin-Y--Cu-1--LAMMPS--ipr5",
                "2001--Mishin-Y--Cu-1--LAMMPS--ipr5",
            ],
        )

    def test_prepare_pair_styles(self):
        pair_style, pair_coeff = self.job._prepare_pair_styles()
        self.assertEqual(pair_style[0], "eam/alloy")

    def test_modes(self):
        self.job.potential = "2001--Mishin-Y--Cu-1--LAMMPS--ipr5"
        self.job.calc_free_energy(temperature=100, pressure=0, reference_phase="solid")
        self.assertEqual(self.job.input.mode, "fe")

        self.job.calc_free_energy(
            temperature=[100, 200], pressure=0, reference_phase="solid"
        )
        self.assertEqual(self.job.input.mode, "ts")

        self.job.calc_free_energy(
            temperature=100, pressure=[0, 1000], reference_phase="solid"
        )
        self.assertEqual(self.job.input.mode, "pscale")

        self.job.potential = [
            "2001--Mishin-Y--Cu-1--LAMMPS--ipr5",
            "2001--Mishin-Y--Cu-1--LAMMPS--ipr5",
        ]
        self.job.calc_free_energy(temperature=100, pressure=0, reference_phase="solid")
        self.assertEqual(self.job.input.mode, "alchemy")

    def test_output(self):
        filepath = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../static/"
        )
        print(filepath)
        shutil.copy(
            os.path.join(filepath, "calphy_test_files/test_files.tar.gz"), os.getcwd()
        )
        shutil.copy(os.path.join(filepath, "calphy_test_files/export.csv"), os.getcwd())
        self.output_project.unpack("test_files")
        self.assertEqual(
            float(
                self.output_project["calphy_unittest/solid_job"].output.spring_constant
            ),
            1.45,
        )
        self.assertEqual(
            self.output_project["calphy_unittest/solid_job"].output.energy_free[0],
            -4.0002701274424295,
        )
        self.assertEqual(
            int(self.output_project["calphy_unittest/solid_job"].output.temperature[0]),
            1100,
        )
        self.assertEqual(
            int(
                self.output_project["calphy_unittest/solid_job"].output.temperature[-1]
            ),
            1400,
        )
