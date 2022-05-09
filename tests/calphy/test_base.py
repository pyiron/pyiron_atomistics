# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import os
import numpy as np
import unittest
import warnings
import shutil

from pyiron_atomistics.project import Project
from pyiron_atomistics.calphy.calphy import Calphy
from pyiron_base import state, ProjectHDFio

class TestCalphy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        state.update({'resource_paths': os.path.join(os.path.dirname(os.path.abspath(__file__)), "../static")})
        cls.execution_path = os.path.dirname(os.path.abspath(__file__))
        #cls.file_location = os.path.dirname(os.path.abspath(__file__))
        cls.project = Project(os.path.join(cls.execution_path, "test_calphy"))
        cls.job = Calphy(
            project=ProjectHDFio(project=cls.project, file_name="test_calphy"),
            job_name="test_calphy",
        )
        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../static/")
        cls.output_project = Project(os.path.join(filepath, "calphy_test_files"))

    @classmethod
    def tearDownClass(cls):
        cls.execution_path = os.path.dirname(os.path.abspath(__file__))
        project = Project(os.path.join(cls.execution_path, "test_calphy"))
        project.remove_jobs_silently(recursive=True)
        project.remove(enable=True)
        state.update()
    
    def test_potentials(self):
        self.job.set_potentials(["2001--Mishin-Y--Cu-1--LAMMPS--ipr1", "2001--Mishin-Y--Cu-1--LAMMPS--ipr1"])
        #print(self.job.input)
        #pint(self.job.input.potential_initial_name)
        self.assertEqual(self.job.input.potential_initial_name, "2001--Mishin-Y--Cu-1--LAMMPS--ipr1")
        self.assertEqual(self.job.input.potential_final_name, "2001--Mishin-Y--Cu-1--LAMMPS--ipr1")
        self.assertRaises(ValueError, self.job.set_potentials, ["2001--Mishin-Y--Cu-1--LAMMPS--ipr1", "2001--Mishin-Y--Cu-1--LAMMPS--ipr1", "2001--Mishin-Y--Cu-1--LAMMPS--ipr1"])
        
    def test_prepare_pair_styles(self):
        pair_style, pair_coeff = self.job.prepare_pair_styles()
        self.assertEqual(pair_style[0], "eam/alloy")
        
    def test_modes(self):
        self.job.potential = "2001--Mishin-Y--Cu-1--LAMMPS--ipr1"
        self.job.calc_free_energy(temperature=100, pressure=0,
                                 reference_phase="solid")
        self.assertEqual(self.job.input.mode, "fe")
        
        self.job.calc_free_energy(temperature=[100, 200], pressure=0,
                                 reference_phase="solid")
        self.assertEqual(self.job.input.mode, "ts")

        self.job.calc_free_energy(temperature=100, pressure=[0, 1000],
                                 reference_phase="solid")
        self.assertEqual(self.job.input.mode, "pscale")
        
        self.job.potential = ["2001--Mishin-Y--Cu-1--LAMMPS--ipr1", "2001--Mishin-Y--Cu-1--LAMMPS--ipr1"]
        self.job.calc_free_energy(temperature=100, pressure=0,
                                 reference_phase="solid")
        self.assertEqual(self.job.input.mode, "alchemy")
        
    
    def test_output(self):
        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../static/")
        shutil.copy(os.path.join(filepath, "calphy_test_files/tm_fcc.tar.gz"), cwd)
        shutil.copy(os.path.join(filepath, "calphy_test_files/export.csv"), cwd)
        self.output_project.unpack("tm_fcc")
        self.output_project["copper_demo/tm_fcc"].output
        self.assertEqual(float(self.output_project["copper_demo/tm_fcc"].output.spring_constant), 1.51)    
        self.assertEqual(self.output_project["copper_demo/tm_fcc"].output.energy_free[0], -4.002465158959863)
        self.assertEqual(int(self.output_project["copper_demo/tm_fcc"].output.temperature[0]), 1100)
        self.assertEqual(int(self.output_project["copper_demo/tm_fcc"].output.temperature[-1]), 1400)


        