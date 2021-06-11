# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import unittest
import numpy as np
from pyiron_atomistics.atomistics.structure.atoms import Atoms, CrystalStructure
from pyiron_atomistics.atomistics.structure.factory import StructureFactory


class TestAtoms(unittest.TestCase):
    def test_get_layers(self):
        a_0 = 4
        struct = CrystalStructure('Al', lattice_constants=a_0, bravais_basis='fcc').repeat(10)
        layers = struct.analyse.get_layers()
        self.assertAlmostEqual(np.linalg.norm(layers-np.rint(2*struct.positions/a_0).astype(int)), 0)
        struct.append(Atoms(elements=['C'], positions=np.random.random((1,3))))
        self.assertEqual(
            np.linalg.norm(layers-struct.analyse.get_layers(id_list=struct.select_index('Al'))), 0
        )
        self.assertEqual(
            np.linalg.norm(layers-struct.analyse.get_layers(
                id_list=struct.select_index('Al'),
                wrap_atoms=False
            )), 0
        )
        with self.assertRaises(ValueError):
            _ = struct.analyse.get_layers(distance_threshold=0)
        with self.assertRaises(ValueError):
            _ = struct.analyse.get_layers(id_list=[])

    def test_get_layers_other_planes(self):
        structure = CrystalStructure('Fe', bravais_basis='fcc', lattice_constants=3.5).repeat(2)
        layers = structure.analyse.get_layers(planes=[1,1,1])
        self.assertEqual(np.unique(layers).tolist(), [0,1,2,3,4])

    def test_get_layers_with_strain(self):
        structure = CrystalStructure('Fe', bravais_basis='bcc', lattice_constants=2.8).repeat(2)
        layers = structure.analyse.get_layers().tolist()
        structure.apply_strain(0.1*(np.random.random((3,3))-0.5))
        self.assertEqual(
            layers, structure.analyse.get_layers(planes=np.linalg.inv(structure.cell).T).tolist()
        )

    def test_get_layers_across_pbc(self):
        structure = CrystalStructure('Fe', bravais_basis='bcc', lattice_constants=2.8).repeat(2)
        layers = structure.analyse.get_layers()
        structure.cell[1,0] += 0.01
        structure.center_coordinates_in_unit_cell()
        self.assertEqual(len(np.unique(layers[structure.analyse.get_layers()[:,0]==0,0])), 1)

    def test_pyscal_cna_adaptive(self):
        basis = Atoms(
            "FeFe", scaled_positions=[(0, 0, 0), (0.5, 0.5, 0.5)], cell=np.identity(3)
        )
        self.assertTrue(
            basis.analyse.pyscal_cna_adaptive()["bcc"] == 2
        )

    def test_pyscal_centro_symmetry(self):
        basis = CrystalStructure('Fe', bravais_basis='bcc', lattice_constants=2.8)
        self.assertTrue(
            all([np.isclose(v, 0.0) for v in basis.analyse.pyscal_centro_symmetry(num_neighbors=8)])
        )

    def test_get_voronoi_vertices(self):
        basis = CrystalStructure('Al', bravais_basis='fcc', lattice_constants=4)
        self.assertEqual(len(basis.analyse.get_voronoi_vertices()), 12)
        self.assertEqual(len(basis.analyse.get_voronoi_vertices(distance_threshold=2)), 1)

    def test_get_interstitials_bcc(self):
        bcc = StructureFactory().ase.bulk('Fe', cubic=True)
        x_octa_ref = bcc.positions[:,None,:]+0.5*bcc.cell[None,:,:]
        x_octa_ref = x_octa_ref.reshape(-1, 3)
        x_octa_ref = bcc.get_wrapped_coordinates(x_octa_ref)
        int_octa = bcc.analyse.get_interstitials(num_neighbors=6)
        self.assertEqual(len(int_octa.positions), len(x_octa_ref))
        self.assertAlmostEqual(
            np.linalg.norm(
                x_octa_ref[:,None,:]-int_octa.positions[None,:,:], axis=-1
            ).min(axis=0).sum(), 0
        )
        int_tetra = bcc.analyse.get_interstitials(num_neighbors=4)
        x_tetra_ref = bcc.get_wrapped_coordinates(bcc.analyse.get_voronoi_vertices())
        self.assertEqual(len(int_tetra.positions), len(x_tetra_ref))
        self.assertAlmostEqual(
            np.linalg.norm(
                x_tetra_ref[:,None,:]-int_tetra.positions[None,:,:], axis=-1
            ).min(axis=0).sum(), 0
        )

    def test_get_interstitials_fcc(self):
        fcc = StructureFactory().ase.bulk('Al', cubic=True)
        a_0 = fcc.cell[0,0]
        x_tetra_ref = 0.25*a_0*np.ones(3)*np.array([[1],[-1]])+fcc.positions[:,None,:]
        x_tetra_ref = fcc.get_wrapped_coordinates(x_tetra_ref).reshape(-1, 3)
        int_tetra = fcc.analyse.get_interstitials(num_neighbors=4)
        self.assertEqual(len(int_tetra.positions), len(x_tetra_ref))
        self.assertAlmostEqual(
            np.linalg.norm(
                x_tetra_ref[:,None,:]-int_tetra.positions[None,:,:], axis=-1
            ).min(axis=0).sum(), 0
        )
        x_octa_ref = 0.5*a_0*np.array([1, 0, 0])+fcc.positions
        x_octa_ref = fcc.get_wrapped_coordinates(x_octa_ref)
        int_octa = fcc.analyse.get_interstitials(num_neighbors=6)
        self.assertEqual(len(int_octa.positions), len(x_octa_ref))
        self.assertAlmostEqual(
            np.linalg.norm(x_octa_ref[:,None,:]-int_octa.positions[None,:,:], axis=-1).min(axis=0).sum(), 0
        )
        self.assertTrue(
            np.allclose(int_octa.get_areas(), a_0**2*np.sqrt(3)),
            msg='Convex hull area comparison with analytical value failed'
        )
        self.assertTrue(
            np.allclose(int_octa.get_volumes(), a_0**3/6),
            msg='Convex hull volume comparison with analytical value failed'
        )
        self.assertTrue(
            np.allclose(int_octa.get_distances(), a_0/2),
            msg='Distance comparison with analytical value failed'
        )
        self.assertTrue(
            np.all(int_octa.get_steinhardt_parameters(4)>0),
            msg='Illegal Steinhardt parameter'
        )
        self.assertAlmostEqual(
            int_octa.get_variances().sum(), 0,
            msg='Distance variance in FCC must be 0'
        )


if __name__ == "__main__":
    unittest.main()
