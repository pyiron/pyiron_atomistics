# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import unittest
import os
from pyiron_atomistics.atomistics.structure.atoms import CrystalStructure
from pyiron_atomistics.vasp.base import Input, Output
from pyiron_atomistics import Project
from pyiron_base import state, ProjectHDFio
from pyiron_atomistics.vasp.potential import VaspPotentialSetter
from pyiron_atomistics.vasp.vasp import Vasp
from pyiron_atomistics.vasp.metadyn import VaspMetadyn
from pyiron_atomistics.vasp.structure import read_atoms
import numpy as np
import warnings

__author__ = "Sudarsan Surendralal"


class TestVasp(unittest.TestCase):
    """
    Tests the pyiron_atomistics.objects.hamilton.dft.vasp.Vasp class
    """

    @classmethod
    def setUpClass(cls):
        state.update(
            {
                "resource_paths": os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "..", "static"
                )
            }
        )
        cls.execution_path = os.path.dirname(os.path.abspath(__file__))
        cls.project = Project(os.path.join(cls.execution_path, "test_vasp"))
        cls.job = cls.project.create_job("Vasp", "trial")
        cls.job_spin = cls.project.create_job("Vasp", "spin")
        cls.job_spin.structure = CrystalStructure("Fe", BravaisBasis="bcc", a=2.83)
        cls.job_spin.structure = cls.job_spin.structure.repeat(2)
        cls.job_spin.structure[2] = "Se"
        cls.job_spin.structure[3] = "O"
        cls.job_metadyn = cls.project.create_job("VaspMetadyn", "trial_metadyn")
        cls.job_complete = Vasp(
            project=ProjectHDFio(project=cls.project, file_name="vasp_complete"),
            job_name="vasp_complete",
        )
        poscar_file = os.path.join(
            cls.execution_path,
            "..",
            "static",
            "vasp_test_files",
            "full_job_sample",
            "POSCAR",
        )
        cls.job_complete.structure = read_atoms(poscar_file, species_from_potcar=True)
        poscar_file = os.path.join(
            cls.execution_path,
            "..",
            "static",
            "vasp_test_files",
            "poscar_samples",
            "POSCAR_metadyn",
        )
        cls.job_metadyn.structure = read_atoms(poscar_file)

    @classmethod
    def tearDownClass(cls):
        cls.execution_path = os.path.dirname(os.path.abspath(__file__))
        project = Project(os.path.join(cls.execution_path, "test_vasp"))
        project.remove_jobs(recursive=True, silently=True)
        project.remove(enable=True)
        state.update()

    def setUp(self):
        self.job.structure = None

    def test_potential_set(self):
        job_pot = self.project.create_job("Vasp", "potential")
        job_pot.structure = CrystalStructure("Fe", BravaisBasis="bcc", a=2.83)
        job_pot.potential["Fe"] = "Fe_sv_GW"
        structure = CrystalStructure("Fe", BravaisBasis="bcc", a=2.9)
        job_pot.structure = structure
        self.assertEqual(str(job_pot.potential), str({"Fe": "Fe_sv_GW"}))
        structure[0] = "Al"
        job_pot.structure = structure
        self.assertEqual(str(job_pot.potential), str({"Fe": "Fe_sv_GW", "Al": None}))
        structure[:] = "Al"
        job_pot.structure = structure
        self.assertEqual(str(job_pot.potential), str({"Fe": "Fe_sv_GW", "Al": None}))

    def test_list_potentials(self):
        self.assertRaises(ValueError, self.job.list_potentials)
        self.assertEqual(
            sorted(
                [
                    "Fe",
                    "Fe_GW",
                    "Fe_pv",
                    "Fe_sv",
                    "Fe_sv_GW",
                    "Se",
                    "Se_GW",
                    "O",
                    "O_GW",
                    "O_GW_new",
                    "O_h",
                    "O_s",
                    "O_s_GW",
                ]
            ),
            sorted(self.job_spin.list_potentials()),
        )
        self.assertEqual(
            sorted(["Fe", "Fe_GW", "Fe_pv", "Fe_sv", "Fe_sv_GW"]),
            sorted(self.job_complete.list_potentials()),
        )
        self.job_spin.potential["Fe"] = "Fe_sv_GW"
        self.job_complete.potential.Fe = "Fe_sv_GW"
        self.assertEqual(
            "Fe_sv_GW", list(self.job_spin.potential.to_dict().values())[0]
        )
        self.assertEqual(
            "Fe_sv_GW", list(self.job_complete.potential.to_dict().values())[0]
        )
        self.job_complete.potential["Fe"] = "Fe"
        self.job_spin.potential.Fe = "Fe"

    def test_init(self):
        self.assertEqual(self.job.__name__, "Vasp")
        self.assertEqual(self.job._sorted_indices, None)
        self.assertIsInstance(self.job.input, Input)
        self.assertIsInstance(self.job._output_parser, Output)
        self.assertIsInstance(self.job._potential, VaspPotentialSetter)
        self.assertTrue(self.job._compress_by_default)
        self.assertEqual(self.job.get_eddrmm_handling(), "warn")
        self.assertIsInstance(self.job_metadyn, Vasp)
        self.assertIsInstance(self.job_metadyn, VaspMetadyn)
        self.assertTrue(self.job_metadyn.input.incar["LBLUEOUT"])

    def test_eddrmm(self):
        self.job.set_eddrmm_handling("ignore")
        self.assertEqual(self.job.get_eddrmm_handling(), "ignore")
        self.job.set_eddrmm_handling("restart")
        self.assertEqual(self.job.get_eddrmm_handling(), "restart")
        self.job.set_eddrmm_handling()
        self.assertEqual(self.job.get_eddrmm_handling(), "warn")
        self.assertRaises(ValueError, self.job.set_eddrmm_handling, status="blah")

    def test_rwigs(self):
        rwigs_dict = {"Fe": 1.1, "Se": 2.2, "O": 3.3, "N": 4.4}
        rwigs_dict_wrong_1 = {"Fe": "not a float", "Se": 2.2, "O": 3.3, "N": 4.4}
        rwigs_dict_wrong_2 = {"Fe": 1.1}

        self.assertIsNone(self.job_spin.get_rwigs())
        self.assertRaises(
            AssertionError, self.job_spin.set_rwigs, rwigs_dict="not a dict"
        )
        self.assertRaises(
            ValueError, self.job_spin.set_rwigs, rwigs_dict=rwigs_dict_wrong_1
        )
        self.assertRaises(
            ValueError, self.job_spin.set_rwigs, rwigs_dict=rwigs_dict_wrong_2
        )

        self.job_spin.set_rwigs(rwigs_dict)
        rwigs_dict_out = self.job_spin.get_rwigs()
        for key in rwigs_dict_out.keys():
            self.assertEqual(rwigs_dict_out[key], rwigs_dict[key])

    def test_spin_constraints(self):
        self.job_spin.spin_constraints = 1
        self.assertTrue(self.job_spin.spin_constraints)

        self.job_spin.spin_constraints = 2
        self.assertTrue(self.job_spin.spin_constraints)

        del self.job_spin.input.incar["I_CONSTRAINED_M"]
        self.assertFalse(self.job_spin.spin_constraints)

    def test_spin_constraint(self):
        rwigs_dict = {"Fe": 1.1, "Se": 2.2, "O": 3.3, "N": 4.4}

        self.assertRaises(
            AssertionError,
            self.job_spin.set_spin_constraint,
            lamb=0.5,
            rwigs_dict=rwigs_dict,
            direction="not a bool",
            norm=False,
        )
        self.assertRaises(
            AssertionError,
            self.job_spin.set_spin_constraint,
            lamb=0.5,
            rwigs_dict=rwigs_dict,
            direction=True,
            norm="not a bool",
        )
        self.assertRaises(
            AssertionError,
            self.job_spin.set_spin_constraint,
            lamb="not a float",
            rwigs_dict=rwigs_dict,
            direction=True,
            norm=False,
        )
        self.assertRaises(
            ValueError,
            self.job_spin.set_spin_constraint,
            lamb=0.5,
            rwigs_dict=rwigs_dict,
            direction=False,
            norm=False,
        )
        self.assertRaises(
            ValueError,
            self.job_spin.set_spin_constraint,
            lamb=0.5,
            rwigs_dict=rwigs_dict,
            direction=False,
            norm=True,
        )

        self.job_spin.set_spin_constraint(
            lamb=0.5, rwigs_dict=rwigs_dict, direction=True, norm=False
        )
        self.assertEqual(self.job_spin.input.incar["LAMBDA"], 0.5)
        self.assertEqual(self.job_spin.input.incar["I_CONSTRAINED_M"], 1)
        rwigs_dict_out = self.job_spin.get_rwigs()
        for key in rwigs_dict_out.keys():
            self.assertEqual(rwigs_dict_out[key], rwigs_dict[key])

        self.job_spin.set_spin_constraint(
            lamb=0.5, rwigs_dict=rwigs_dict, direction=True, norm=True
        )
        self.assertEqual(self.job_spin.input.incar["I_CONSTRAINED_M"], 2)

    def test_potential(self):
        self.assertEqual(self.job.potential, self.job._potential)

    def test_plane_wave_cutoff(self):
        self.assertIsInstance(self.job.plane_wave_cutoff, (float, int, type(None)))
        # self.assertIsInstance(self.job.plane_wave_cutoff, (float, int))
        self.job.plane_wave_cutoff = 350
        self.assertEqual(self.job.input.incar["ENCUT"], 350)
        self.assertEqual(self.job.plane_wave_cutoff, 350)
        self.assertEqual(self.job.plane_wave_cutoff, self.job.encut)
        self.job.encut = 450
        self.assertEqual(self.job.encut, 450)
        self.assertEqual(self.job.input.incar["ENCUT"], 450)
        self.assertEqual(self.job.plane_wave_cutoff, 450)

    def test_exchange_correlation_functional(self):
        self.assertEqual(self.job.exchange_correlation_functional, "GGA")
        self.assertEqual(self.job.input.potcar["xc"], "GGA")
        self.job.exchange_correlation_functional = "LDA"
        self.assertEqual(self.job.exchange_correlation_functional, "LDA")
        self.assertEqual(self.job.input.potcar["xc"], "LDA")

    def test_get_nelect(self):
        atoms = CrystalStructure("Pt", BravaisBasis="fcc", a=3.98)
        self.job.structure = atoms
        self.assertEqual(self.job.get_nelect(), 10)

    def test_write_magmoms(self):
        magmom = np.arange(8.0)
        magmom_ncl = np.zeros([8, 3])
        magmom_ncl[:, 0] = magmom / 2
        magmom_ncl[:, 1] = magmom
        magmom_ncl[:, 2] = magmom**2

        magmom_str = "0.0   1.0   2.0   3.0   4.0   5.0   6.0   7.0"
        magmom_ncl_str = (
            "0.0 0.0 0.0   0.5 1.0 1.0   1.0 2.0 4.0   1.5 3.0 9.0   "
            "2.0 4.0 16.0   2.5 5.0 25.0   3.0 6.0 36.0   3.5 7.0 49.0"
        )

        self.job.structure = CrystalStructure("Fe", BravaisBasis="bcc", a=2.83)
        self.job.structure = self.job.structure.repeat(2)

        self.job.structure.set_initial_magnetic_moments(magmom)

        self.job.input.incar["ISPIN"] = 1
        self.job.write_magmoms()
        self.assertIsNone(self.job.input.incar["MAGMOM"])
        self.assertEqual(self.job.input.incar["ISPIN"], 1)

        del self.job.input.incar["ISPIN"]
        self.job.write_magmoms()
        self.assertEqual(self.job.input.incar["ISPIN"], 2)
        self.assertEqual(self.job.input.incar["MAGMOM"], magmom_str)

        del self.job.input.incar["MAGMOM"]
        self.job.structure.set_initial_magnetic_moments(magmom_ncl)
        self.job.set_spin_constraint(
            lamb=1.0, rwigs_dict={"Fe": 2.5}, direction=True, norm=True
        )
        self.job.write_magmoms()
        self.assertEqual(self.job.input.incar["LNONCOLLINEAR"], True)
        self.assertEqual(self.job.input.incar["MAGMOM"], magmom_ncl_str)
        self.assertEqual(self.job.input.incar["M_CONSTR"], magmom_ncl_str)

        del self.job.input.incar["MAGMOM"]
        del self.job.input.incar["M_CONSTR"]
        del self.job.input.incar["LNONCOLLINEAR"]
        del self.job.input.incar["RWIGS"]

        self.assertRaises(ValueError, self.job.write_magmoms)

        self.job.input.incar["RWIGS"] = "2.5"
        del self.job.input.incar["LAMBDA"]

        self.assertRaises(ValueError, self.job.write_magmoms)

    def test_set_empty_states(self):
        atoms = CrystalStructure("Pt", BravaisBasis="fcc", a=3.98)
        self.job.structure = atoms
        self.job.set_empty_states(n_empty_states=10)
        self.assertEqual(self.job.input.incar["NBANDS"], 15)
        self.job.structure = atoms.repeat([3, 1, 1])
        self.job.set_empty_states(n_empty_states=10)
        self.assertEqual(self.job.input.incar["NBANDS"], 25)

    def test_set_occpuancy_smearing(self):
        job_smear = self.project.create_job("Vasp", "smearing")
        self.assertIsNone(job_smear.input.incar["ISMEAR"])
        self.assertIsNone(job_smear.input.incar["SIGMA"])
        job_smear.set_occupancy_smearing(smearing="methfessel_paxton")
        self.assertEqual(job_smear.input.incar["ISMEAR"], 1)
        job_smear.set_occupancy_smearing(smearing="methfessel_paxton", order=2)
        self.assertEqual(job_smear.input.incar["ISMEAR"], 2)
        job_smear.set_occupancy_smearing(smearing="Fermi", width=0.1)
        self.assertEqual(job_smear.input.incar["ISMEAR"], -1)
        self.assertEqual(job_smear.input.incar["SIGMA"], 0.1)
        job_smear.set_occupancy_smearing(smearing="Gaussian", width=0.1)
        self.assertEqual(job_smear.input.incar["ISMEAR"], 0)
        self.assertEqual(job_smear.input.incar["SIGMA"], 0.1)
        with warnings.catch_warnings(record=True) as w:
            job_smear.set_occupancy_smearing(smearing="Gaussian", ismear=10)
            self.assertEqual(job_smear.input.incar["ISMEAR"], 10)
            self.assertEqual(len(w), 1)
        self.assertRaises(
            ValueError, job_smear.set_occupancy_smearing, smearing="gibberish"
        )

    def test_calc_static(self):
        self.job.calc_static(
            electronic_steps=90,
            retain_charge_density=True,
            retain_electrostatic_potential=True,
        )
        self.assertEqual(self.job.input.incar["IBRION"], -1)
        self.assertEqual(self.job.input.incar["NELM"], 90)
        self.assertEqual(self.job.input.incar["LVTOT"], True)
        self.assertEqual(self.job.input.incar["LCHARG"], True)

    def test_set_structure(self):
        self.assertEqual(self.job.structure, None)
        atoms = CrystalStructure("Pt", BravaisBasis="fcc", a=3.98)
        self.job.structure = atoms
        self.assertEqual(self.job.structure, atoms)
        self.job.structure = None
        self.assertEqual(self.job.structure, None)
        self.job.structure = atoms
        self.assertEqual(self.job.structure, atoms)

    def test_run_complete(self):
        self.job_complete.exchange_correlation_functional = "PBE"
        self.job_complete.set_occupancy_smearing(smearing="fermi", width=0.2)
        self.job_complete.calc_static()
        self.job_complete.set_convergence_precision(electronic_energy=1e-7)
        self.job_complete.write_electrostatic_potential = False
        self.assertEqual(self.job_complete.input.incar["SIGMA"], 0.2)
        self.assertEqual(self.job_complete.input.incar["LVTOT"], False)
        self.assertEqual(self.job_complete.input.incar["EDIFF"], 1e-7)
        file_directory = os.path.join(
            self.execution_path, "..", "static", "vasp_test_files", "full_job_sample"
        )
        self.job_complete.restart_file_list.append(
            os.path.join(file_directory, "vasprun.xml")
        )
        self.job_complete.restart_file_list.append(
            os.path.join(file_directory, "OUTCAR")
        )
        self.job_complete.restart_file_list.append(
            os.path.join(file_directory, "CHGCAR")
        )
        self.job_complete.restart_file_list.append(
            os.path.join(file_directory, "WAVECAR")
        )
        self.job_complete.run(run_mode="manual")
        self.job_complete.status.collect = True
        self.job_complete.run()
        nodes = [
            "positions",
            "temperature",
            "energy_tot",
            "steps",
            "positions",
            "forces",
            "cells",
            "pressures",
        ]
        with self.job_complete.project_hdf5.open("output/generic") as h_gen:
            hdf_nodes = h_gen.list_nodes()
            self.assertTrue(all([node in hdf_nodes for node in nodes]))
        nodes = [
            "energy_free",
            "energy_int",
            "energy_zero",
            "final_magmoms",
            "magnetization",
            "n_elect",
            "scf_dipole_mom",
            "scf_energy_free",
            "scf_energy_int",
            "scf_energy_zero",
        ]
        with self.job_complete.project_hdf5.open("output/generic/dft") as h_dft:
            hdf_nodes = h_dft.list_nodes()
            self.assertTrue(all([node in hdf_nodes for node in nodes]))
        nodes = ["efermi", "eig_matrix", "k_points", "k_weights", "occ_matrix"]
        with self.job_complete.project_hdf5.open(
            "output/electronic_structure"
        ) as h_dft:
            hdf_nodes = h_dft.list_nodes()
            self.assertTrue(all([node in hdf_nodes for node in nodes]))
        job_chg_den = self.job_complete.restart_from_charge_density(job_name="chg")
        self.assertEqual(job_chg_den.structure, self.job_complete.get_structure(-1))
        working_directory = os.path.join(
            self.execution_path, "test_vasp", "vasp_complete_hdf5", "vasp_complete"
        )
        self.assertTrue(
            os.path.join(working_directory, "CHGCAR")
            in [os.path.abspath(f) for f in job_chg_den.restart_file_list]
        )

        def check_group_is_empty(example_job, group_name):
            with example_job.project_hdf5.open(group_name) as h_gr:
                self.assertTrue(h_gr.list_nodes() == [])
                self.assertTrue(h_gr.list_groups() == [])

        check_group_is_empty(job_chg_den, "output")

        job_chg_wave = self.job_complete.restart_from_wave_and_charge(
            job_name="chg_wave"
        )
        self.assertEqual(job_chg_wave.structure, self.job_complete.get_structure(-1))
        self.assertTrue(
            os.path.join(working_directory, "WAVECAR")
            in [os.path.abspath(f) for f in job_chg_wave.restart_file_list]
        )
        self.assertTrue(
            os.path.join(working_directory, "CHGCAR")
            in [os.path.abspath(f) for f in job_chg_wave.restart_file_list]
        )
        for key, val in job_chg_wave.restart_file_dict.items():
            self.assertTrue(key, val)
        check_group_is_empty(job_chg_wave, "output")

        job = self.job_complete.restart()
        job.restart_file_list.append(os.path.join(file_directory, "vasprun.xml"))
        job.restart_file_list.append(os.path.join(file_directory, "OUTCAR"))
        job.run(run_mode="manual")
        job.status.collect = True
        job.run()
        # Check if error raised if the files don't exist
        self.assertRaises(
            FileNotFoundError, job.restart_from_wave_functions, "wave_restart"
        )
        self.assertRaises(
            FileNotFoundError, job.restart_from_charge_density, "chg_restart"
        )
        self.assertRaises(
            FileNotFoundError, job.restart_from_wave_and_charge, "wave_chg_restart"
        )

    def test_vasp_metadyn(self):
        self.job_metadyn.set_primitive_constraint(
            "bond_1", "bond", atom_indices=[0, 2], increment=1e-4
        )
        self.job_metadyn.set_primitive_constraint(
            "bond_2", "bond", atom_indices=[0, 3], increment=1e-4
        )
        self.job_metadyn.set_complex_constraint(
            "combine", "linear_combination", {"bond_1": 1, "bond_2": -1}, increment=1e-4
        )
        self.job_metadyn.write_constraints()
        constraints = self.job_metadyn.input.iconst._dataset["Value"]
        for val in ["R 1 6 0", "R 1 2 0", "S 1 -1 0"]:
            self.assertTrue(val in constraints)

    def test_setting_input(self):
        self.job.set_convergence_precision(
            electronic_energy=1e-7, ionic_force_tolerance=0.1
        )
        self.assertAlmostEqual(self.job.input.incar["EDIFF"], 1e-7)
        self.assertAlmostEqual(self.job.input.incar["EDIFFG"], -0.1)
        self.job.set_convergence_precision(ionic_force_tolerance=0.001)
        self.assertAlmostEqual(self.job.input.incar["EDIFFG"], -0.001)
        self.job.calc_minimize()
        self.assertAlmostEqual(self.job.input.incar["EDIFFG"], -0.01)
        self.job.calc_minimize(ionic_energy_tolerance=1e-5)
        self.assertAlmostEqual(
            self.job.input.incar["EDIFFG"],
            1e-5,
            "ionic energy tolerance not set correctly by calc_minimize",
        )
        self.job.calc_minimize(ionic_force_tolerance=1e-3)
        self.assertAlmostEqual(
            self.job.input.incar["EDIFFG"],
            -0.001,
            "ionic force tolerance not set correctly by calc_minimize",
        )
        self.assertAlmostEqual(self.job.input.incar["EDIFF"], 1e-7)

    def test_mixing_parameter(self):
        job = self.project.create_job("Vasp", "mixing_parameter")
        job.set_mixing_parameters(density_mixing_parameter=0.1)
        self.assertEqual(job.input.incar["IMIX"], 4)
        with self.assertRaises(NotImplementedError):
            job.set_mixing_parameters(density_residual_scaling=0.1)

    def test_potentials(self):
        # Assert that no warnings are raised
        with warnings.catch_warnings(record=True) as w:
            structure = self.project.create.structure.bulk("Al", cubic=True)
            element = self.project.create.structure.element(
                new_element_name="Al_GW", parent_element="Al", potential_file="Al_GW"
            )
            structure[:] = element
            job = self.project.create.job.Vasp("test")
            job.structure = structure
            job.run(run_mode="manual")
            self.assertTrue(
                len(w) <= 1,
                msg=f"Expected one warnings but got {[warn.message for warn in w]}.",
            )

    def test_kspacing(self):
        job_kspace = self.project.create_job("Vasp", "job_kspacing")
        job_kspace.structure = self.project.create.structure.ase.bulk("Fe")
        job_kspace.input.incar["KSPACING"] = 0.5
        with warnings.catch_warnings(record=True) as w:
            job_kspace.run(run_mode="manual")
            self.assertNotIn(
                "KPOINTS",
                job_kspace.files.list(),
                "'KPOINTS' file written even when KPACING tag is present in INCAR",
            )

            self.assertTrue(len(w) <= 2)
            self.assertEqual(
                str(w[0].message), "'KSPACING' found in INCAR, no KPOINTS file written"
            )


if __name__ == "__main__":
    unittest.main()
