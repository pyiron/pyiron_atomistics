# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import numpy as np
import pandas as pd
import os
import re
from io import StringIO
from pyiron_base import state, ProjectHDFio
from pyiron_atomistics.atomistics.structure.atoms import Atoms
from pyiron_atomistics.lammps.lammps import Lammps
from pyiron_atomistics.lammps.base import LammpsStructure, UnfoldingPrism
from pyiron_atomistics.lammps.units import LAMMPS_UNIT_CONVERSIONS, UnitConverter
import ase.units as units
from pyiron_base._tests import TestWithCleanProject


class TestLammps(TestWithCleanProject):
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

    def test_selective_dynamics(self):
        atoms = Atoms("Fe8", positions=np.zeros((8, 3)), cell=np.eye(3))
        atoms.add_tag(selective_dynamics=[True, True, True])
        self.job.structure = atoms
        self.job._set_selective_dynamics()
        self.assertFalse("group" in self.job.input.control._dataset["Parameter"])
        atoms.add_tag(selective_dynamics=None)
        atoms.selective_dynamics[1] = [True, True, False]
        atoms.selective_dynamics[2] = [True, False, True]
        atoms.selective_dynamics[3] = [False, True, True]
        atoms.selective_dynamics[4] = [False, True, False]
        atoms.selective_dynamics[5] = [False, False, True]
        atoms.selective_dynamics[6] = [True, False, False]
        atoms.selective_dynamics[7] = [False, False, False]
        self.job.structure = atoms
        self.job._set_selective_dynamics()
        for constraint in ["x", "y", "z", "xy", "yz", "xz", "xyz"]:
            self.assertTrue(
                f"group___constraint{constraint}"
                in self.job.input.control._dataset["Parameter"],
                msg=f"Failed to find group___constraint{constraint} in control",
            )

    def test_structure_atomic(self):
        atoms = Atoms("Fe1", positions=np.zeros((1, 3)), cell=np.eye(3))
        lmp_structure = LammpsStructure()
        lmp_structure._el_eam_lst = ["Fe"]
        lmp_structure.structure = atoms
        self.assertEqual( 
            lmp_structure._dataset["Value"],
            [
                "Start File for LAMMPS",
                "1 atoms",
                "1 atom types",
                "",
                "0. 1.000000000000000 xlo xhi",
                "0. 1.000000000000000 ylo yhi",
                "0. 1.000000000000000 zlo zhi",
                "",
                "Masses",
                "",
                "1 55.845000",
                "",
                "Atoms",
                "",
                "1 1 0.000000000000000 0.000000000000000 0.000000000000000",
                "",
            ],
        )

    def test_structure_charge(self):
        atoms = Atoms("Fe1", positions=np.zeros((1, 3)), cell=np.eye(3))
        atoms.set_initial_charges(charges=np.ones(len(atoms)) * 2.0)
        lmp_structure = LammpsStructure()
        lmp_structure.atom_type = "charge"
        lmp_structure._el_eam_lst = ["Fe"]
        lmp_structure.structure = atoms
        self.assertEqual(
            lmp_structure._dataset["Value"],
            [
                "Start File for LAMMPS",
                "1 atoms",
                "1 atom types",
                "",
                "0. 1.000000000000000 xlo xhi",
                "0. 1.000000000000000 ylo yhi",
                "0. 1.000000000000000 zlo zhi",
                "",
                "Masses",
                "",
                "1 55.845000",
                "",
                "Atoms",
                "",
                "1 1 2.000000 0.000000000000000 0.000000000000000 0.000000000000000",
                "",
            ],
        )

    def test_avilable_versions(self):
        self.job.executable = os.path.abspath(
            os.path.join(
                self.execution_path,
                "..",
                "static",
                "lammps",
                "bin",
                "run_lammps_2018.03.16.sh",
            )
        )
        self.assertTrue([2018, 3, 16] == self.job._get_executable_version_number())
        self.job.executable = os.path.abspath(
            os.path.join(
                self.execution_path,
                "..",
                "static",
                "lammps",
                "bin",
                "run_lammps_2018.03.16_mpi.sh",
            )
        )
        self.assertTrue([2018, 3, 16] == self.job._get_executable_version_number())

    def _build_water(self, y0_shift=0.0):
        density = 1.0e-24  # g/A^3
        n_mols = 27
        mol_mass_water = 18.015  # g/mol
        # Determining the supercell size size
        mass = mol_mass_water * n_mols / units.mol  # g
        vol_h2o = mass / density  # in A^3
        a = vol_h2o ** (1.0 / 3.0)  # A
        # Constructing the unitcell
        n = int(round(n_mols ** (1.0 / 3.0)))
        dx = 0.7
        r_O = [0, 0, 0]
        r_H1 = [dx, dx, 0]
        r_H2 = [-dx, dx, 0]
        unit_cell = (a / n) * np.eye(3)
        unit_cell[0][1] += y0_shift
        water = Atoms(
            elements=["H", "H", "O"],
            positions=[r_H1, r_H2, r_O],
            cell=unit_cell,
            pbc=True,
        )
        water.set_repeat([n, n, n])
        return water

    def test_lammps_water(self):
        self.job.structure = self._build_water()
        with self.assertWarns(UserWarning):
            self.job.potential = "H2O_tip3p"
        with self.assertRaises(ValueError):
            self.job.calc_md(temperature=350, seed=0)
        with self.assertRaises(ValueError):
            self.job.calc_md(temperature=[0, 100])
        with self.assertRaises(ValueError):
            self.job.calc_md(pressure=0)
        with self.assertRaises(ValueError):
            self.job.calc_md(temperature=[0, 100, 200])
        self.job.calc_md(
            temperature=350,
            initial_temperature=350,
            time_step=1,
            n_ionic_steps=1000,
            n_print=200,
        )
        file_directory = os.path.join(
            self.execution_path, "..", "static", "lammps_test_files"
        )
        self.job.restart_file_list.append(os.path.join(file_directory, "dump.out"))
        self.job.restart_file_list.append(os.path.join(file_directory, "log.lammps"))
        self.job.run(run_mode="manual")
        self.job.status.collect = True
        self.job.run()
        nodes = [
            "positions",
            "temperature",
            "energy_tot",
            "energy_pot",
            "steps",
            "positions",
            "forces",
            "cells",
            "pressures",
            "unwrapped_positions",
        ]
        with self.job.project_hdf5.open("output/generic") as h_gen:
            hdf_nodes = h_gen.list_nodes()
            self.assertTrue(all([node in hdf_nodes for node in nodes]))
        self.assertTrue(
            np.array_equal(self.job["output/generic/positions"].shape, (6, 81, 3))
        )
        self.assertTrue(
            np.array_equal(
                self.job["output/generic/positions"].shape,
                self.job["output/generic/forces"].shape,
            )
        )
        self.assertEqual(len(self.job["output/generic/steps"]), 6)

    def test_dump_parser_water(self):
        water = self._build_water(y0_shift=0.01)
        self.job.structure = water
        with self.assertWarns(UserWarning):
            self.job.potential = "H2O_tip3p"
        self.job.calc_md(
            temperature=350,
            initial_temperature=350,
            time_step=1,
            n_ionic_steps=1000,
            n_print=200,
            pressure=0,
        )
        self.assertFalse("nan" in self.job.input.control["fix___ensemble"])
        file_directory = os.path.join(
            self.execution_path, "..", "static", "lammps_test_files"
        )
        self.job.restart_file_list.append(os.path.join(file_directory, "log.lammps"))
        self.job.restart_file_list.append(os.path.join(file_directory, "dump.out"))
        self.job.run(run_mode="manual")
        self.job.status.collect = True
        self.job.run()
        positions = np.loadtxt(os.path.join(file_directory, "positions_water.dat"))
        positions = positions.reshape(len(positions), -1, 3)
        forces = np.loadtxt(os.path.join(file_directory, "forces_water.dat"))
        forces = forces.reshape(len(forces), -1, 3)
        self.assertTrue(
            np.allclose(self.job["output/generic/unwrapped_positions"], positions)
        )
        uc = UnitConverter(self.job.input.control["units"])
        self.assertTrue(
            np.allclose(
                self.job["output/generic/forces"],
                uc.convert_array_to_pyiron_units(forces, "forces"),
            )
        )
        self.assertEqual(
            self.job["output/generic/energy_tot"][-1],
            -5906.46836142123 * uc.lammps_to_pyiron("energy"),
        )
        self.assertEqual(
            self.job["output/generic/energy_pot"][-1],
            -5982.82004785158 * uc.lammps_to_pyiron("energy"),
        )

        self.assertAlmostEqual(
            self.job["output/generic/pressures"][-2][0, 0],
            515832.570508186 / uc.pyiron_to_lammps("pressure"),
            2,
        )
        self.job.write_traj(filename="test.xyz", file_format="xyz")
        atom_indices = self.job.structure.select_index("H")
        snap_indices = [1, 3, 4]
        orig_pos = self.job.output.positions
        self.job.write_traj(
            filename="test.xyz",
            file_format="xyz",
            atom_indices=atom_indices,
            snapshot_indices=snap_indices,
        )
        self.job.write_traj(
            filename="test.xyz",
            file_format="xyz",
            atom_indices=atom_indices,
            snapshot_indices=snap_indices,
            overwrite_positions=np.zeros_like(orig_pos),
        )
        self.assertRaises(
            ValueError,
            self.job.write_traj,
            filename="test.xyz",
            file_format="xyz",
            atom_indices=atom_indices,
            snapshot_indices=snap_indices,
            overwrite_positions=np.zeros_like(orig_pos)[:-1],
        )

        self.job.write_traj(
            filename="test.xyz",
            file_format="xyz",
            atom_indices=atom_indices,
            snapshot_indices=snap_indices,
            overwrite_positions=np.zeros_like(orig_pos),
            overwrite_cells=self.job.trajectory()._cells,
        )
        self.job.write_traj(
            filename="test.xyz",
            file_format="xyz",
            atom_indices=atom_indices,
            snapshot_indices=snap_indices,
            overwrite_positions=np.zeros_like(orig_pos)[:-1],
            overwrite_cells=self.job.trajectory()._cells[:-1],
        )
        self.assertRaises(
            ValueError,
            self.job.write_traj,
            filename="test.xyz",
            file_format="xyz",
            atom_indices=atom_indices,
            snapshot_indices=snap_indices,
            overwrite_positions=np.zeros_like(orig_pos),
            overwrite_cells=self.job.trajectory()._cells[:-1],
        )
        os.remove("test.xyz")
        self.assertTrue(np.array_equal(self.job.trajectory()._positions, orig_pos))
        self.assertTrue(
            np.array_equal(self.job.trajectory(stride=2)._positions, orig_pos[::2])
        )
        self.assertTrue(
            np.array_equal(
                self.job.trajectory(
                    atom_indices=atom_indices, snapshot_indices=snap_indices
                )._positions,
                orig_pos[snap_indices][:, atom_indices, :],
            )
        )

        nx, ny, nz = orig_pos.shape
        random_array = np.random.rand(nx, ny, nz)
        random_cell = np.random.rand(nx, 3, 3)
        self.assertTrue(
            np.array_equal(
                self.job.trajectory(
                    atom_indices=atom_indices,
                    snapshot_indices=snap_indices,
                    overwrite_positions=random_array,
                )._positions,
                random_array[snap_indices][:, atom_indices, :],
            )
        )
        self.assertTrue(
            np.array_equal(
                self.job.trajectory(
                    atom_indices=atom_indices,
                    snapshot_indices=snap_indices,
                    overwrite_positions=random_array,
                    overwrite_cells=random_cell,
                )._cells,
                random_cell[snap_indices],
            )
        )
        self.assertIsInstance(self.job.get_structure(-1), Atoms)
        # Test for clusters
        with self.job.project_hdf5.open("output/generic") as h_out:
            h_out["cells"] = None
        self.assertTrue(
            np.array_equal(
                self.job.trajectory(
                    atom_indices=atom_indices, snapshot_indices=snap_indices
                )._positions,
                orig_pos[snap_indices][:, atom_indices, :],
            )
        )
        with self.job.project_hdf5.open("output/generic") as h_out:
            h_out["cells"] = np.repeat(
                [np.array(water.cell)], len(h_out["positions"]), axis=0
            )
        self.assertTrue(
            np.array_equal(
                self.job.trajectory(
                    atom_indices=atom_indices, snapshot_indices=snap_indices
                )._positions,
                orig_pos[snap_indices][:, atom_indices, :],
            )
        )
        neigh_traj_obj = self.job.get_neighbors()
        self.assertTrue(
            np.allclose(
                np.linalg.norm(neigh_traj_obj.vecs, axis=-1), neigh_traj_obj.distances
            )
        )
        h_indices = self.job.structure.select_index("H")
        o_indices = self.job.structure.select_index("O")
        self.assertLessEqual(neigh_traj_obj.distances[:, o_indices, :2].max(), 1.2)
        self.assertGreaterEqual(neigh_traj_obj.distances[:, o_indices, :2].min(), 0.8)
        self.assertTrue(
            np.alltrue(
                [
                    np.in1d(np.unique(ind_mat.flatten()), h_indices)
                    for ind_mat in neigh_traj_obj.indices[:, o_indices, :2]
                ]
            )
        )
        neigh_traj_obj_snaps = self.job.get_neighbors_snapshots(
            snapshot_indices=[2, 3, 4]
        )
        self.assertTrue(np.allclose(neigh_traj_obj.vecs[2:], neigh_traj_obj_snaps.vecs))
        neigh_traj_obj.to_hdf(self.job.project_hdf5)
        neigh_traj_obj_loaded = self.job["neighbors_traj"].to_object()
        # self.assertEqual(neigh_traj_obj._init_structure, neigh_traj_obj_loaded._init_structure)
        self.assertEqual(
            neigh_traj_obj._num_neighbors, neigh_traj_obj_loaded._num_neighbors
        )
        self.assertTrue(
            np.allclose(neigh_traj_obj.indices, neigh_traj_obj_loaded.indices)
        )
        self.assertTrue(
            np.allclose(neigh_traj_obj.distances, neigh_traj_obj_loaded.distances)
        )
        self.assertTrue(np.allclose(neigh_traj_obj.vecs, neigh_traj_obj_loaded.vecs))
        self.assertTrue(self.job.units, "real")

    def test_dump_parser(self):
        structure = Atoms(
            elements=2 * ["Fe"],
            cell=2.78 * np.eye(3),
            positions=2.78 * np.outer(np.arange(2), np.ones(3)) * 0.5,
        )
        self.job.structure = structure
        self.job.potential = self.job.list_potentials()[0]
        file_directory = os.path.join(
            self.execution_path, "..", "static", "lammps_test_files"
        )
        self.job.collect_dump_file(cwd=file_directory, file_name="dump_static.out")
        self.assertTrue(
            np.array_equal(self.job["output/generic/forces"].shape, (1, 2, 3))
        )
        self.assertTrue(
            np.array_equal(self.job["output/generic/positions"].shape, (1, 2, 3))
        )
        self.assertTrue(
            np.array_equal(self.job["output/generic/cells"].shape, (1, 3, 3))
        )
        self.assertTrue(
            np.array_equal(self.job["output/generic/indices"].shape, (1, 2))
        )

    def test_vcsgc_input(self):
        unit_cell = Atoms(
            elements=["Al", "Al", "Al", "Mg"],
            positions=[
                [0.0, 0.0, 0.0],
                [0.0, 2.0, 2.0],
                [2.0, 0.0, 2.0],
                [2.0, 2.0, 0.0],
            ],
            cell=4 * np.eye(3),
        )
        self.job.structure = unit_cell
        self.job.potential = self.job.list_potentials()[0]
        symbols = self.job.input.potential.get_element_lst()

        with self.subTest("Fail when elements outside the periodic table are used"):
            bad_element = {s: 0.0 for s in symbols}
            bad_element.update({"X": 1.0})  # Non-existant chemical symbol
            self.assertRaises(
                ValueError, self.job.calc_vcsgc, mu=bad_element, temperature_mc=300.0
            )
            self.assertRaises(
                ValueError,
                self.job.calc_vcsgc,
                target_concentration=bad_element,
                temperature_mc=300.0,
            )

        with self.subTest("Fail when concentrations don't add to 1"):
            bad_conc = {s: 0.0 for s in symbols}
            bad_conc["Al"] = 0.99
            self.assertRaises(
                ValueError,
                self.job.calc_vcsgc,
                target_concentration=bad_conc,
                temperature_mc=300.0,
            )

        with self.subTest("Check window definitions"):
            for bad_window in [-1, 1.1]:
                self.assertRaises(
                    ValueError,
                    self.job.calc_vcsgc,
                    window_moves=bad_window,
                    temperature_mc=300.0,
                )
            self.assertRaises(
                ValueError, self.job.calc_vcsgc, window_size=0.3, temperature_mc=300.0
            )

        with self.subTest("Temperature can't be None"):
            mu = {s: 0.0 for s in symbols}
            mu[symbols[0]] = 1.0
            self.assertRaises(
                ValueError,
                self.job.calc_vcsgc,
                mu=mu,
                temperature_mc=None,
                temperature=None,
            )

        args = dict(
            mu=mu,
            target_concentration=None,
            kappa=1000.0,
            mc_step_interval=100,
            swap_fraction=0.1,
            temperature_mc=None,
            window_size=None,
            window_moves=None,
            seed=1,
            temperature=300.0,
        )
        input_string = "all sgcmc {0} {1} {2} {3} randseed {4}".format(
            args["mc_step_interval"],
            args["swap_fraction"],
            args["temperature"],
            " ".join(
                [
                    str(args["mu"][symbol] - args["mu"][symbols[0]])
                    for symbol in symbols[1:]
                ]
            ),
            args["seed"],
        )
        self.job.calc_vcsgc(**args)
        self.assertEqual(
            self.job.input.control["fix___vcsgc"],
            input_string,
            msg="Parser did not reproduce expected lammps control syntax",
        )

        args["temperature_mc"] = 100.0
        input_string = "all sgcmc {0} {1} {2} {3} randseed {4}".format(
            args["mc_step_interval"],
            args["swap_fraction"],
            args["temperature_mc"],
            " ".join(
                [
                    str(args["mu"][symbol] - args["mu"][symbols[0]])
                    for symbol in symbols[1:]
                ]
            ),
            args["seed"],
        )
        self.job.calc_vcsgc(**args)
        self.assertEqual(
            self.job.input.control["fix___vcsgc"],
            input_string,
            msg="Parser did not reproduce expected lammps control syntax",
        )

        conc = {s: 0.0 for s in symbols}
        conc[symbols[0]] = 0.5
        conc[symbols[-1]] = 0.5
        args["target_concentration"] = conc
        input_string += " variance {0} {1}".format(
            args["kappa"], " ".join([str(conc[symbol]) for symbol in symbols[1:]])
        )
        self.job.calc_vcsgc(**args)
        self.assertEqual(
            self.job.input.control["fix___vcsgc"],
            input_string,
            msg="Parser did not reproduce expected lammps control syntax",
        )

        args["window_moves"] = 10
        input_string += " window_moves {0}".format(args["window_moves"])
        self.job.calc_vcsgc(**args)
        self.assertEqual(
            self.job.input.control["fix___vcsgc"],
            input_string,
            msg="Parser did not reproduce expected lammps control syntax",
        )

        args["window_size"] = 0.75
        input_string += " window_size {0}".format(args["window_size"])
        self.job.calc_vcsgc(**args)
        self.assertEqual(
            self.job.input.control["fix___vcsgc"],
            input_string,
            msg="Parser did not reproduce expected lammps control syntax",
        )

        self.job.to_hdf()
        for k, v in args.items():
            if k not in (
                "mu",
                "target_concentration",
                "mc_step_interval",
                "swap_fraction",
                "temperature_mc",
            ):
                continue
            self.assertEqual(
                self.job._generic_input[k],
                v,
                msg=f"Wrong value stored in generic input for parameter {k}!",
            )
            # decode saved GenericParameters manually...
            data = self.job["input/generic/data_dict"]
            self.assertEqual(
                data["Value"][data["Parameter"].index(k)],
                str(v),
                msg=f"Wrong value stored in HDF for parameter {k}!",
            )

    def test_calc_minimize_input(self):
        # Ensure that defaults match control defaults
        atoms = Atoms("Fe8", positions=np.zeros((8, 3)), cell=np.eye(3))
        self.ref.structure = atoms
        self.ref.input.control.calc_minimize()

        self.job.sturcture = atoms
        self.job._prism = UnfoldingPrism(atoms.cell)
        self.job.calc_minimize()
        for k in self.job.input.control.keys():
            self.assertEqual(self.job.input.control[k], self.ref.input.control[k])

        # Ensure that pressure inputs are being parsed OK
        self.ref.calc_minimize(pressure=0)
        self.assertEqual(
            self.ref.input.control["fix___ensemble"], "all box/relax iso 0.0"
        )

        self.ref.calc_minimize(pressure=[0.0, 0.0, 0.0])
        self.assertEqual(
            self.ref.input.control["fix___ensemble"],
            "all box/relax x 0.0 y 0.0 z 0.0 couple none",
        )

        cnv = LAMMPS_UNIT_CONVERSIONS[self.ref.input.control["units"]]["pressure"]

        self.ref.calc_minimize(pressure=-2.0)
        m = re.match(
            r"all +box/relax +iso +([-\d.]+)$",
            self.ref.input.control["fix___ensemble"].strip(),
        )
        self.assertTrue(m)
        self.assertTrue(np.isclose(float(m.group(1)), -2.0 * cnv))

        self.ref.calc_minimize(pressure=[1, 2, None, 3.0, 0.0, None])
        m = re.match(
            r"all +box/relax +x +([\d.]+) +y ([\d.]+) +xy +([\d.]+) +xz +([\d.]+) +couple +none$",
            self.ref.input.control["fix___ensemble"].strip(),
        )
        self.assertTrue(m)
        self.assertTrue(np.isclose(float(m.group(1)), 1.0 * cnv))
        self.assertTrue(np.isclose(float(m.group(2)), 2.0 * cnv))
        self.assertTrue(np.isclose(float(m.group(3)), 3.0 * cnv))
        self.assertTrue(np.isclose(float(m.group(4)), 0.0 * cnv))

    def test_calc_md_input(self):
        # Ensure that defaults match control defaults
        atoms = Atoms("Fe8", positions=np.zeros((8, 3)), cell=np.eye(3))
        self.ref.structure = atoms
        self.ref.input.control.calc_md()

        self.job.sturcture = atoms
        self.job._prism = UnfoldingPrism(atoms.cell)
        self.job.calc_md()
        for k in self.job.input.control.keys():
            self.assertEqual(self.job.input.control[k], self.ref.input.control[k])

        # Ensure that pressure inputs are being parsed OK
        self.ref.calc_md(temperature=300.0, pressure=0)
        self.assertEqual(
            self.ref.input.control["fix___ensemble"],
            "all npt temp 300.0 300.0 0.1 iso 0.0 0.0 1.0",
        )

        self.ref.calc_md(temperature=300.0, pressure=[0.0, 0.0, 0.0])
        self.assertEqual(
            self.ref.input.control["fix___ensemble"],
            "all npt temp 300.0 300.0 0.1 x 0.0 0.0 1.0 y 0.0 0.0 1.0 z 0.0 0.0 1.0",
        )

        cnv = LAMMPS_UNIT_CONVERSIONS[self.ref.input.control["units"]]["pressure"]

        self.ref.calc_md(temperature=300.0, pressure=-2.0)
        m = re.match(
            r"all +npt +temp +300.0 +300.0 +0.1 +iso +([-\d.]+) +([-\d.]+) 1.0$",
            self.ref.input.control["fix___ensemble"].strip(),
        )
        self.assertTrue(m)
        self.assertTrue(np.isclose(float(m.group(1)), -2.0 * cnv))
        self.assertTrue(np.isclose(float(m.group(2)), -2.0 * cnv))

        self.ref.calc_md(temperature=300.0, pressure=[1, 2, None, 3.0, 0.0, None])
        m = re.match(
            r"all +npt +temp +300.0 +300.0 +0.1 +"
            r"x +([\d.]+) +([\d.]+) +1.0 +y +([\d.]+) +([\d.]+) +1.0 +"
            r"xy +([\d.]+) +([\d.]+) +1.0 +xz +([\d.]+) +([\d.]+) +1.0$",
            self.ref.input.control["fix___ensemble"].strip(),
        )
        self.assertTrue(m)
        self.assertTrue(np.isclose(float(m.group(1)), 1.0 * cnv))
        self.assertTrue(np.isclose(float(m.group(2)), 1.0 * cnv))
        self.assertTrue(np.isclose(float(m.group(3)), 2.0 * cnv))
        self.assertTrue(np.isclose(float(m.group(4)), 2.0 * cnv))
        self.assertTrue(np.isclose(float(m.group(5)), 3.0 * cnv))
        self.assertTrue(np.isclose(float(m.group(6)), 3.0 * cnv))
        self.assertTrue(np.isclose(float(m.group(7)), 0.0 * cnv))
        self.assertTrue(np.isclose(float(m.group(8)), 0.0 * cnv))

    def test_read_restart_file(self):
        self.job.read_restart_file()
        self.assertIsNone(self.job["dimension"])

    def test_write_restart(self):
        self.job.write_restart_file()
        self.assertEqual(self.job.input.control["write_restart"], "restart.out")

    def test_average(self):
        a_0 = 2.855312531
        atoms = Atoms("Fe2", positions=[3 * [0], 3 * [0.5 * a_0]], cell=a_0 * np.eye(3))
        self.job.structure = atoms
        self.job.potential = "Fe_C_Becquart_eam"
        file_directory = os.path.join(
            self.execution_path, "..", "static", "lammps_test_files"
        )
        self.job.collect_dump_file(cwd=file_directory, file_name="dump_average.out")
        self.job.collect_output_log(cwd=file_directory, file_name="log_average.lammps")

    def test_validate(self):
        with self.assertRaises(ValueError):
            self.job.validate_ready_to_run()
        a_0 = 2.855312531
        atoms = Atoms(
            "Fe2", positions=[3 * [0], 3 * [0.5 * a_0]], cell=a_0 * np.eye(3), pbc=False
        )
        self.job.structure = atoms
        # with self.assertRaises(ValueError):
        #     self.job.validate_ready_to_run()
        self.job.potential = self.job.list_potentials()[-1]
        self.job.validate_ready_to_run()
        self.job.structure.positions[0, 0] -= 2.855
        with self.assertRaises(ValueError):
            self.job.validate_ready_to_run()
        self.job.structure.pbc = True
        self.job.validate_ready_to_run()
        self.job.structure.pbc = [True, True, False]
        self.job.validate_ready_to_run()
        self.job.structure.pbc = [False, True, True]
        with self.assertRaises(ValueError):
            self.job.validate_ready_to_run()

    def test_potential_check(self):
        """
        Verifies that job.potential accepts only potentials that contain the
        species in the set structure.
        """

        self.job.structure = Atoms("Al1", positions=[3 * [0]], cell=np.eye(3))
        with self.assertRaises(ValueError):
            self.job.potential = "Fe_C_Becquart_eam"

        potential = pd.DataFrame(
            {
                "Name": ["Fe Morse"],
                "Filename": [[]],
                "Model": ["Morse"],
                "Species": [["Fe"]],
                "Config": [
                    [
                        "atom_style full\n",
                        "pair_coeff 1 2 morse 0.019623 1.8860 3.32833\n",
                    ]
                ],
            }
        )

        with self.assertRaises(ValueError):
            self.job.potential = potential

        potential["Species"][0][0] = "Al"
        self.job.potential = potential  # shouldn't raise ValueError

    def test_units(self):
        self.assertTrue(self.job.units, "metal")
        self.job.units = "real"
        self.assertTrue(self.job.units, "real")

        def setter(x):
            self.job.units = x

        self.assertRaises(ValueError, setter, "nonsense")

    def test_bonds_input(self):
        potential = pd.DataFrame(
            {
                "Name": ["Morse"],
                "Filename": [[]],
                "Model": ["Morse"],
                "Species": [["Al"]],
                "Config": [
                    [
                        "atom_style bond\n",
                        "bond_style morse\n",
                        "bond_coeff 1 0.1 1.5 2.0\n",
                        "bond_coeff 2 0.1 1.5 2.0",
                    ]
                ],
            }
        )
        cell = Atoms(
            elements=4 * ["Al"],
            positions=[
                [0.0, 0.0, 0.0],
                [0.0, 2.0, 2.0],
                [2.0, 0.0, 2.0],
                [2.0, 2.0, 0.0],
            ],
            cell=4 * np.eye(3),
        )
        self.job.structure = cell.repeat(2)
        self.job.structure.bonds = [[1, 2, 1], [1, 3, 2]]
        self.job.potential = potential
        self.job.calc_static()
        self.job.run(run_mode="manual")

        bond_str = "2 bond types\n"
        self.assertTrue(self.job["structure.inp"][4][-1], bond_str)

    def test_dump_parsing(self):
        self.job.structure = Atoms("Al1", positions=[3 * [0]], cell=np.eye(3)).repeat(2)
        potential = pd.DataFrame(
            {
                "Name": ["Al Morse"],
                "Filename": [[]],
                "Model": ["Morse"],
                "Species": [["Al"]],
                "Config": [
                    [
                        "atom_style full\n",
                        "pair_coeff 1 2 morse 0.019623 1.8860 3.32833\n",
                    ]
                ],
            }
        )
        self.job.potential = potential 
        self.job.calc_md(n_ionic_steps=3, n_print=1)
        self.job.run()
        self.job.decompress()
        old_output = collect_dump_file_old(self.job.working_directory)
        with open(self.job.project_hdf5.open("output/generic")) as hdf_out:
            for k, v in old_output.items():
                self.assertTrue(
                    np.all(v==hdf_out[k])
                )


def collect_dump_file_old(self, file_name="dump.out", cwd=None):
    """
    general purpose routine to extract static from a lammps dump file

    Args:
        file_name:
        cwd:

    Returns:

    """
    uc = UnitConverter(self.units)
    file_name = self.job_file_name(file_name=file_name, cwd=cwd)
    output = {}
    with open(file_name, "r") as ff:
        dump = ff.readlines()

    steps = np.genfromtxt(
        [
            dump[nn]
            for nn in np.where(
                [ll.startswith("ITEM: TIMESTEP") for ll in dump]
            )[0]
            + 1
        ],
        dtype=int,
    )
    steps = np.array([steps]).flatten()
    output["steps"] = steps

    natoms = np.genfromtxt(
        [
            dump[nn]
            for nn in np.where(
                [ll.startswith("ITEM: NUMBER OF ATOMS") for ll in dump]
            )[0]
            + 1
        ],
        dtype=int,
    )
    natoms = np.array([natoms]).flatten()

    prism = self._prism
    rotation_lammps2orig = self._prism.R.T
    cells = np.genfromtxt(
        " ".join(
            (
                [
                    " ".join(dump[nn : nn + 3])
                    for nn in np.where(
                        [ll.startswith("ITEM: BOX BOUNDS") for ll in dump]
                    )[0]
                    + 1
                ]
            )
        ).split()
    ).reshape(len(natoms), -1)
    lammps_cells = np.array([to_amat(cc) for cc in cells])
    unfolded_cells = np.array(
        [prism.unfold_cell(cell) for cell in lammps_cells]
    )
    output["cells"] = unfolded_cells

    l_start = np.where([ll.startswith("ITEM: ATOMS") for ll in dump])[0]
    l_end = l_start + natoms + 1
    content = [
        pd.read_csv(
            StringIO("\n".join(dump[llst:llen]).replace("ITEM: ATOMS ", "")),
            delim_whitespace=True,
        ).sort_values(by="id", ignore_index=True)
        for llst, llen in zip(l_start, l_end)
    ]

    indices = np.array([cc["type"] for cc in content], dtype=int)
    output["indices"] = self.remap_indices(indices)

    forces = np.array(
        [np.stack((cc["fx"], cc["fy"], cc["fz"]), axis=-1) for cc in content]
    )
    output["forces"] = np.matmul(forces, rotation_lammps2orig)

    if "f_mean_forces[1]" in content[0].keys():
        forces = np.array(
            [
                np.stack(
                    (
                        cc["f_mean_forces[1]"],
                        cc["f_mean_forces[2]"],
                        cc["f_mean_forces[3]"],
                    ),
                    axis=-1,
                )
                for cc in content
            ]
        )
        output["mean_forces"] = np.matmul(forces, rotation_lammps2orig)

    if np.all(
        [flag in content[0].columns.values for flag in ["vx", "vy", "vz"]]
    ):
        velocities = np.array(
            [
                np.stack((cc["vx"], cc["vy"], cc["vz"]), axis=-1)
                for cc in content
            ]
        )
        output["velocities"] = np.matmul(velocities, rotation_lammps2orig)

    if "f_mean_velocities[1]" in content[0].keys():
        velocities = np.array(
            [
                np.stack(
                    (
                        cc["f_mean_velocities[1]"],
                        cc["f_mean_velocities[2]"],
                        cc["f_mean_velocities[3]"],
                    ),
                    axis=-1,
                )
                for cc in content
            ]
        )
        output["mean_velocities"] = np.matmul(velocities, rotation_lammps2orig)
    direct_unwrapped_positions = np.array(
        [np.stack((cc["xsu"], cc["ysu"], cc["zsu"]), axis=-1) for cc in content]
    )
    unwrapped_positions = np.matmul(direct_unwrapped_positions, lammps_cells)
    output["unwrapped_positions"] = np.matmul(
        unwrapped_positions, rotation_lammps2orig
    )
    if "f_mean_positions[1]" in content[0].keys():
        direct_unwrapped_positions = np.array(
            [
                np.stack(
                    (
                        cc["f_mean_positions[1]"],
                        cc["f_mean_positions[2]"],
                        cc["f_mean_positions[3]"],
                    ),
                    axis=-1,
                )
                for cc in content
            ]
        )
        unwrapped_positions = np.matmul(
            direct_unwrapped_positions, lammps_cells
        )
        output["mean_unwrapped_positions"] = np.matmul(
            unwrapped_positions, rotation_lammps2orig
        )

    direct_positions = direct_unwrapped_positions - np.floor(
        direct_unwrapped_positions
    )
    positions = np.matmul(direct_positions, lammps_cells)
    output["positions"] = np.matmul(positions, rotation_lammps2orig)

    keys = content[0].keys()
    for kk in keys[keys.str.startswith("c_")]:
        output[kk.replace("c_", "")] = np.array(
            [cc[kk] for cc in content], dtype=float
        )
    
    return output