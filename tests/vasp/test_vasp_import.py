# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import unittest
import numpy as np
import os
from pyiron_atomistics.project import Project
from pyiron_atomistics.vasp.vasp import Vasp
from pyiron_atomistics.vasp.volumetric_data import VaspVolumetricData
from pyiron_vasp.vasp.vasprun import VasprunWarning
import warnings


class TestVaspImport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.file_location = os.path.dirname(os.path.abspath(__file__))
        cls.project = Project(os.path.join(cls.file_location, "vasp_import_testing"))

    @classmethod
    def tearDownClass(cls):
        cls.file_location = os.path.dirname(os.path.abspath(__file__))
        project = Project(os.path.join(cls.file_location, "vasp_import_testing"))
        project.remove_jobs(recursive=True, silently=True)
        project.remove(enable=True)

    def test_import(self):
        folder_path = os.path.join(
            self.file_location, "../static/vasp_test_files/full_job_sample"
        )
        self.project.import_from_path(path=folder_path, recursive=False)
        ham = self.project.load("full_job_sample")
        self.assertTrue(ham.status.finished)
        self.assertTrue(isinstance(ham, Vasp))
        self.assertEqual(ham.get_nelect(), 16)
        self.assertTrue(
            np.array_equal(ham.structure.get_initial_magnetic_moments(), [-1, -1])
        )
        self.assertIsInstance(ham.output.unwrapped_positions, np.ndarray)

        self.assertEqual(
            ham["output/structure/positions"][1, 2],
            2.7999999999999997 * 0.4999999999999999,
        )
        self.assertEqual(ham["output/generic/dft/e_fermi_list"][-1], 5.9788)
        self.assertEqual(ham["output/generic/dft/vbm_list"][-1], 6.5823)
        self.assertEqual(ham["output/generic/dft/cbm_list"][-1], 6.7396)
        folder_path = os.path.join(
            self.file_location, "../static/vasp_test_files/full_job_minor_glitch"
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("ignore")
            warnings.simplefilter("always", category=VasprunWarning)
            self.project.import_from_path(path=folder_path, recursive=False)
            self.assertEqual(len(w), 3)
        ham = self.project.load("full_job_minor_glitch")
        self.assertTrue(ham.status.finished)
        self.assertTrue(isinstance(ham, Vasp))
        self.assertEqual(ham.get_nelect(), 16)
        self.assertIsInstance(ham.output.unwrapped_positions, np.ndarray)
        self.assertEqual(ham["output/generic/dft/scf_energy_free"][0][1], 0.0)
        self.assertEqual(
            ham["output/electronic_structure/occ_matrix"].shape, (1, 4, 12)
        )
        self.assertEqual(
            ham["output/electronic_structure/eig_matrix"].shape, (1, 4, 12)
        )
        self.assertEqual(
            ham._generic_input["reduce_kpoint_symmetry"], ham.reduce_kpoint_symmetry
        )
        folder_path = os.path.join(
            self.file_location, "../static/vasp_test_files/full_job_corrupt_potcar"
        )
        self.project.import_from_path(path=folder_path, recursive=False)
        ham = self.project.load("full_job_corrupt_potcar")
        self.assertTrue(ham.status.finished)

    def test_import_bader(self):
        folder_path = os.path.join(
            self.file_location, "../static/vasp_test_files/bader_test"
        )
        self.project.import_from_path(path=folder_path, recursive=False)
        ham = self.project.load("bader_test")
        self.assertTrue(
            np.allclose(ham["output/generic/dft/valence_charges"], [1.0, 1.0, 6.0])
        )
        # Only check if Bader is installed!
        if os.system("bader") == 0:
            self.assertTrue(
                np.allclose(
                    ham["output/generic/dft/bader_charges"],
                    [0.928831, 1.018597, -8.403403],
                )
            )

    def test_incar_import(self):
        file_path = os.path.join(
            self.file_location, "../static/vasp_test_files/incar_samples/INCAR_1"
        )
        ham = self.project.create_job(self.project.job_type.Vasp, "incar_import")
        ham.input.incar.read_input(file_path, ignore_trigger="!")
        self.assertTrue(ham.input.incar["LWAVE"])
        self.assertTrue(ham.input.incar["LCHARG"])
        self.assertTrue(ham.input.incar["LVTOT"])
        self.assertFalse(ham.input.incar["LDIPOL"])
        self.assertFalse(ham.input.incar["LVHAR"])
        self.assertFalse(ham.input.incar["LORBIT"])
        self.assertTrue(ham.input.incar["LCORE"])
        self.assertFalse(ham.input.incar["LTEST"])
        self.assertEqual(ham.input.incar["POTIM"], 0.5)

    def test_output(self):
        ham = self.project.inspect("full_job_sample")
        self.assertEqual(ham["output/generic/dft/energy_free"][-1], -17.7379867884)
        self.assertIsInstance(
            ham["output/charge_density"].to_object(), VaspVolumetricData
        )
        with ham.project_hdf5.open("output/generic") as h_gen:
            for node in h_gen.list_nodes():
                if h_gen[node] is not None:
                    self.assertIsInstance(
                        h_gen[node],
                        np.ndarray,
                        f"output/generic/{node} is not stored as a numpy array",
                    )


if __name__ == "__main__":
    unittest.main()
