# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import unittest
import numpy as np
from pyiron_atomistics.atomistics.structure.atoms import Atoms
from pyiron_atomistics.atomistics.structure.factory import StructureFactory

class TestAtoms(unittest.TestCase):
    def test_get_arg_equivalent_sites(self):
        a_0 = 4.0
        structure = StructureFactory().ase.bulk('Al', cubic=True, a=a_0).repeat(2)
        sites = structure.get_wrapped_coordinates(structure.positions+np.array([0, 0, 0.5*a_0]))
        v_position = structure.positions[0]
        del structure[0]
        pairs = np.stack((
            structure.get_symmetry().get_arg_equivalent_sites(sites),
            np.unique(np.round(structure.get_distances_array(v_position, sites), decimals=2), return_inverse=True)[1]
        ), axis=-1)
        unique_pairs = np.unique(pairs, axis=0)
        self.assertEqual(len(unique_pairs), len(np.unique(unique_pairs[:,0])))
        with self.assertRaises(ValueError):
            structure.get_symmetry().get_arg_equivalent_sites([0, 0, 0])

    def test_generate_equivalent_points(self):
        a_0 = 4
        structure = StructureFactory().ase.bulk('Al', cubic=True, a=a_0)
        self.assertEqual(
            len(structure),
            len(structure.get_symmetry().generate_equivalent_points([0, 0, 0.5*a_0]))
        )

    def test_get_symmetry(self):
        cell = 2.2 * np.identity(3)
        Al = Atoms("AlAl", positions=[(0, 0, 0), (0.5, 0.5, 0.5)], cell=cell, pbc=True).repeat(2)
        self.assertEqual(len(set(Al.get_symmetry()["equivalent_atoms"])), 1)
        self.assertEqual(len(Al.get_symmetry()["translations"]), 96)
        self.assertEqual(
            len(Al.get_symmetry()["translations"]), len(Al.get_symmetry()["rotations"])
        )
        cell = 2.2 * np.identity(3)
        Al = Atoms("AlAl", scaled_positions=[(0, 0, 0), (0.5, 0.5, 0.5)], cell=cell, pbc=True)
        with self.assertRaises(ValueError):
            Al.get_symmetry().symmetrize_vectors(1)
        v = np.random.rand(6).reshape(-1, 3)
        self.assertAlmostEqual(np.linalg.norm(Al.get_symmetry().symmetrize_vectors(v)), 0)
        Al.positions[0,0] += 0.01
        w = Al.get_symmetry().symmetrize_vectors(v)
        self.assertAlmostEqual(np.absolute(w[:,0]).sum(), np.linalg.norm(w, axis=-1).sum())

    def test_get_symmetry_dataset(self):
        cell = 2.2 * np.identity(3)
        Al_sc = Atoms("AlAl", scaled_positions=[(0, 0, 0), (0.5, 0.5, 0.5)], cell=cell)
        Al_sc.set_repeat([2, 2, 2])
        self.assertEqual(Al_sc.get_symmetry().info["number"], 229)

    def test_get_ir_reciprocal_mesh(self):
        cell = 2.2 * np.identity(3)
        Al_sc = Atoms("AlAl", scaled_positions=[(0, 0, 0), (0.5, 0.5, 0.5)], cell=cell)
        self.assertEqual(len(Al_sc.get_symmetry().get_ir_reciprocal_mesh([3, 3, 3])[0]), 27)

    def test_get_primitive_cell(self):
        cell = 2.2 * np.identity(3)
        basis = Atoms("AlFe", scaled_positions=[(0, 0, 0), (0.5, 0.5, 0.5)], cell=cell)
        structure = basis.repeat([2, 2, 2])
        sym = structure.get_symmetry()
        self.assertEqual(len(basis), len(sym.get_primitive_cell(standardize=True)))
        self.assertEqual(len(sym.primitive_cell), len(sym.get_primitive_cell(standardize=False)))
        self.assertEqual(len(sym.refine_cell()), len(sym.get_primitive_cell(standardize=True)))
        self.assertEqual(sym.get_primitive_cell().get_symmetry().spacegroup["Number"], 221)

    def test_get_equivalent_points(self):
        basis = Atoms("FeFe", positions=[[0.01, 0, 0], [0.5, 0.5, 0.5]], cell=np.identity(3))
        arr = basis.get_symmetry().generate_equivalent_points([0, 0, 0.5])
        self.assertAlmostEqual(np.linalg.norm(arr-np.array([0.51, 0.5, 0]), axis=-1).min(), 0)

    def test_get_space_group(self):
        cell = 2.2 * np.identity(3)
        Al_sc = Atoms("AlAl", scaled_positions=[(0, 0, 0), (0.5, 0.5, 0.5)], cell=cell)
        self.assertEqual(Al_sc.get_symmetry().spacegroup["InternationalTableSymbol"], "Im-3m")
        self.assertEqual(Al_sc.get_symmetry().spacegroup["Number"], 229)
        cell = 4.2 * (0.5 * np.ones((3, 3)) - 0.5 * np.eye(3))
        Al_fcc = Atoms("Al", scaled_positions=[(0, 0, 0)], cell=cell)
        self.assertEqual(Al_fcc.get_symmetry().spacegroup["InternationalTableSymbol"], "Fm-3m")
        self.assertEqual(Al_fcc.get_symmetry().spacegroup["Number"], 225)
        a = 3.18
        c = 1.623 * a
        cell = np.eye(3)
        cell[0, 0] = a
        cell[2, 2] = c
        cell[1, 0] = -a / 2.0
        cell[1, 1] = np.sqrt(3) * a / 2.0
        pos = np.array([[0.0, 0.0, 0.0], [1.0 / 3.0, 2.0 / 3.0, 1.0 / 2.0]])
        Mg_hcp = Atoms("Mg2", scaled_positions=pos, cell=cell)
        self.assertEqual(Mg_hcp.get_symmetry().spacegroup["Number"], 194)
        cell = np.eye(3)
        cell[0, 0] = a
        cell[2, 2] = c
        cell[1, 1] = np.sqrt(3) * a
        pos = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.5, 0.0],
                [0.5, 1/6, 0.5],
                [0.0, 2/3, 0.5],
            ]
        )
        Mg_hcp = Atoms("Mg4", scaled_positions=pos, cell=cell)
        self.assertEqual(Mg_hcp.get_symmetry().spacegroup["Number"], 194)

    def test_permutations(self):
        structure = StructureFactory().ase.bulk('Al', cubic=True).repeat(2)
        x_vacancy = structure.positions[0]
        del structure[0]
        neigh = structure.get_neighborhood(x_vacancy)
        vec = np.zeros_like(structure.positions)
        vec[neigh.indices[0]] = neigh.vecs[0]
        sym = structure.get_symmetry()
        all_vectors = np.einsum('ijk,ink->inj', sym.rotations, vec[sym.permutations])
        for i, v in zip(neigh.indices, neigh.vecs):
            vec = np.zeros_like(structure.positions)
            vec[i] = v
            self.assertAlmostEqual(np.linalg.norm(all_vectors - vec, axis=(-1, -2)).min(), 0,)

    def test_arg_equivalent_vectors(self):
        structure = StructureFactory().ase.bulk('Al', cubic=True).repeat(2)
        self.assertEqual(np.unique(structure.get_symmetry().arg_equivalent_vectors).squeeze(), 0)
        x_v = structure.positions[0]
        del structure[0]
        arg_v = structure.get_symmetry().arg_equivalent_vectors
        dx = structure.get_distances_array(structure.positions, x_v, vectors=True)
        dx_round = np.round(np.absolute(dx), decimals=3)
        self.assertEqual(len(np.unique(dx_round + arg_v)), len(np.unique(arg_v)))


if __name__ == "__main__":
    unittest.main()
