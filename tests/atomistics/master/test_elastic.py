# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import os
import unittest
from pyiron_atomistics.atomistics.structure.atoms import CrystalStructure
from pyiron_base import Project
from pyiron_atomistics.atomistics.master.elastic import (
    calc_elastic_tensor,
    _get_higher_order_strains,
    calc_elastic_constants,
    get_elastic_tensor_by_orientation,
    _convert_to_voigt,
)
import numpy as np


class TestElasticTensor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.file_location = os.path.dirname(os.path.abspath(__file__))
        cls.project = Project(os.path.join(cls.file_location, "test_elast"))
        cls.basis = CrystalStructure(
            element="Fe", bravais_basis="bcc", lattice_constant=2.83
        )
        cls.project.remove_jobs_silently(recursive=True)

    @classmethod
    def tearDownClass(cls):
        cls.file_location = os.path.dirname(os.path.abspath(__file__))
        cls.project.remove(enable=True, enforce=True)

    def test_run(self):
        job = self.project.create_job(
            "HessianJob", "job_test"
        )
        job.set_reference_structure(self.basis)
        job.set_force_constants(1)
        job.set_elastic_moduli(100, 100)
        job.server.run_mode.interactive = True
        elast = job.create_job('ElasticTensor', 'elast_job')
        elast.input['min_num_measurements'] = 10
        elast.validate_ready_to_run()
        self.assertEqual(len(elast.input['rotations']), 48)
        self.assertEqual(len(elast.input['strain_matrices']), 21)
        elast.run()

    def test_get_higher_order_strains(self):
        epsilon = np.random.random((3,3))-0.5
        epsilon += epsilon.T
        epsilon /= np.linalg.norm(epsilon.flatten())
        epsilon = np.einsum('n,ij->nij', np.linspace(-0.01, 0.01, 5), epsilon)
        self.assertEqual(_get_higher_order_strains(
            strain_lst=epsilon,
            rotations=np.eye(3).reshape(1,3,3)
        ).shape, (5, 1))

    def test_calc_elastic_tensor(self):
        strain = [[[0.005242761019305993, -0.0012053606628952052, -0.0032546722513198236],
                   [-0.0012053606628952052, -0.007127371135828069, 3.041718158272068e-06],
                   [-0.0032546722513198236, 3.041718158272068e-06, 0.006586071210292139]],
                  [[0.0024605481734605596, 0.0016778559303512886, -0.0016403042799169243],
                   [0.0016778559303512886, -8.266496915246613e-05, -0.002503657361580728],
                   [-0.0016403042799169243, -0.002503657361580728, -0.003929702111752978]]]
        stress = [[[1.1937765570713004, -0.2760427230548313, -0.7461461292223824],
                   [-0.2760427230548312, -0.0019061490643724585, -0.00241807297859032],
                   [-0.7461461292223824, -0.0024180729785903197, 1.336480507679297]],
                  [[0.012575219368485248, 0.38833608463518243, -0.3793653414598678],
                   [0.3883360846351825, -0.24167276177709596, -0.5818057844834313],
                   [-0.3793653414598678, -0.5818057844834315, -0.6091472161106105]]]
        C = calc_elastic_tensor(strain=strain, rotations=self.basis.get_symmetry()['rotations'], stress=stress)
        self.assertGreater(np.min(C[:3,:3].diagonal()), 200)
        self.assertLess(np.min(C[:3,:3].diagonal()), 300)
        energy = [-8.02446818, -8.02538938]
        volume = [23.38682061, 23.24219739]
        C = calc_elastic_tensor(strain=strain,
                                rotations=self.basis.get_symmetry()['rotations'],
                                energy=energy,
                                volume=volume)

    def test_calc_elastic_constants(self):
        C = np.zeros((6,6))
        K = np.random.random()
        mu = np.random.random()
        C[:3,:3] = K-2*mu/3
        C[:3,:3] += 2*np.eye(3)*mu
        C[3:,3:] = mu*np.eye(3)
        output = calc_elastic_constants(C)
        self.assertAlmostEqual(K, output['bulk_modulus'])
        self.assertAlmostEqual(mu, output['shear_modulus'])
        self.assertAlmostEqual(
            np.linalg.norm(C-get_elastic_tensor_by_orientation([[1, 0, 0], [0, 0, -1], [0, 1, 0]], C)), 0
        )

    def test_convert_to_voigt(self):
        voigt = _convert_to_voigt(np.arange(3**4).reshape(*4 * (3,)))
        self.assertTrue(np.array_equal(voigt[0], [ 0,  4,  8,  5,  2,  1]))
        self.assertTrue(np.all(np.diff(voigt, axis=0) == 9))


if __name__ == "__main__":
    unittest.main()
