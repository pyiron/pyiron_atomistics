# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import os
import unittest
import shutil
import pandas as pd

from pyiron_atomistics.project import Project
from pyiron_atomistics.calphy.job import Calphy
from pyiron_base import state, ProjectHDFio


class TestCalphy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        state.update(
            {
                "resource_paths": os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "../static/calphy_test_files",
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
            ["2001--Mishin-Y--Cu-1--LAMMPS--ipr1", "2001--Mishin-Y--Cu-1--LAMMPS--ipr1"]
        )
        # print(self.job.input)
        # pint(self.job.input.potential_initial_name)
        self.assertEqual(
            self.job.input.potential_initial_name, "2001--Mishin-Y--Cu-1--LAMMPS--ipr1"
        )
        self.assertEqual(
            self.job.input.potential_final_name, "2001--Mishin-Y--Cu-1--LAMMPS--ipr1"
        )
        self.assertRaises(
            ValueError,
            self.job.set_potentials,
            [
                "2001--Mishin-Y--Cu-1--LAMMPS--ipr1",
                "2001--Mishin-Y--Cu-1--LAMMPS--ipr1",
                "2001--Mishin-Y--Cu-1--LAMMPS--ipr1",
            ],
        )
        self.assertEqual(len(self.job.potential), 2)

    def test_potentials_df(self):
        filepath = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../static/calphy_test_files/lammps/2001--Mishin-Y--Cu-1--LAMMPS--ipr1/Cu01.eam.alloy",
        )
        pot_eam = pd.DataFrame(
            {
                "Name": ["2001--Mishin-Y--Cu-1--LAMMPS--ipr1"],
                "Filename": [[filepath]],
                "Model": ["EAM"],
                "Species": [["Cu"]],
                "Config": [
                    ["pair_style eam/alloy\n", "pair_coeff * * Cu01.eam.alloy Cu\n"]
                ],
            }
        )
        self.job.set_potentials([pot_eam, pot_eam])
        # print(self.job.input)
        # pint(self.job.input.potential_initial_name)
        self.assertEqual(
            self.job.input.potential_initial_name, "2001--Mishin-Y--Cu-1--LAMMPS--ipr1"
        )
        self.assertEqual(
            self.job.input.potential_final_name, "2001--Mishin-Y--Cu-1--LAMMPS--ipr1"
        )

    def test_view_potentials(self):
        structure = self.project.create.structure.ase.bulk("Cu", cubic=True).repeat(5)
        self.job.structure = structure
        self.assertEqual(isinstance(self.job.view_potentials(), pd.DataFrame), True)
        self.assertEqual(isinstance(self.job.list_potentials(), list), True)

    def test_prepare_pair_styles(self):
        pair_style, pair_coeff = self.job._prepare_pair_styles()
        self.assertEqual(pair_style[0], "eam/alloy")

    def test_modes(self):
        self.job.potential = "2001--Mishin-Y--Cu-1--LAMMPS--ipr1"
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
            "2001--Mishin-Y--Cu-1--LAMMPS--ipr1",
            "2001--Mishin-Y--Cu-1--LAMMPS--ipr1",
        ]
        self.job.calc_free_energy(temperature=100, pressure=0, reference_phase="solid")
        self.assertEqual(self.job.input.mode, "alchemy")

    def test_get_element_list(self):
        structure = self.project.create.structure.ase.bulk("Cu", cubic=True).repeat(5)
        structure[0] = "Li"
        self.job.potential = "2001--Mishin-Y--Cu-1--LAMMPS--ipr1"
        self.job.structure = structure
        self.assertEqual(self.job._get_element_list(), ["Cu"])
        pm, pl = self.job._get_masses()
        self.assertEqual(pm, [63.546])
        self.assertEqual(pl, 0)

    def test_write_structure(self):
        structure = self.project.create.structure.ase.bulk("Cu", cubic=True).repeat(5)
        self.job.potential = "2001--Mishin-Y--Cu-1--LAMMPS--ipr1"
        self.job.structure = structure
        self.job.write_structure(structure, "test.dump", ".")
        self.assertEqual(os.path.exists("test.dump"), True)
        self.assertEqual(self.job._number_of_structures(), 2)

    def test_publication(self):
        self.assertEqual(self.job.publication["calphy"]["calphy"]["number"], "10")

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
            2.33,
        )
        self.assertEqual(
            self.output_project["calphy_unittest/solid_job"].output.energy_free[0],
            -3.996174590429723,
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
