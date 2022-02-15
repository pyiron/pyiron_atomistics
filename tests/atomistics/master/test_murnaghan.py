# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import unittest
import matplotlib
import numpy as np
from pyiron_atomistics.atomistics.structure.atoms import CrystalStructure
from pyiron_base._tests import TestWithProject


def convergence_goal(self, **qwargs):
    import numpy as np
    eps = 0.2
    if "eps" in qwargs:
        eps = qwargs["eps"]
    erg_lst = self.get_from_childs("output/generic/energy")
    var = 1000 * np.var(erg_lst)
    # print(var / len(erg_lst))
    if var / len(erg_lst) < eps:
        return True
    job_prev = self[-1]
    job_name = self.first_child_name() + "_" + str(len(self))
    job_next = job_prev.restart(job_name=job_name)
    return job_next


class TestMurnaghan(TestWithProject):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.basis = CrystalStructure(
            element="Fe", bravais_basis="fcc", lattice_constant=3.5
        )

    def test_interactive_run(self):
        job = self.project.create_job('HessianJob', 'hessian')
        job.set_reference_structure(self.basis)
        job.set_elastic_moduli(1, 1)
        job.set_force_constants(1)
        job.server.run_mode.interactive = True
        murn = job.create_job('Murnaghan', 'murn_hessian')
        murn.input['num_points'] = 5
        murn.input['vol_range'] = 1e-5
        murn.run()
        self.assertAlmostEqual(self.basis.get_volume(), murn['output/equilibrium_volume'])

        optimal = murn.get_structure()
        self.assertAlmostEqual(optimal.get_volume(), murn['output/equilibrium_volume'],
                               msg="Output of get_structure should have equilibrium volume")

    def test_run(self):
        job = self.project.create_job(
            'AtomisticExampleJob', "job_test"
        )
        job.structure = self.basis
        job_ser = self.project.create_job(
            self.project.job_type.SerialMaster, "murn_iter"
        )
        job_ser.append(job)
        job_ser.set_goal(convergence_goal, eps=0.4)
        murn = self.project.create_job("Murnaghan", "murnaghan")
        murn.ref_job = job_ser
        murn.input['num_points'] = 3
        murn.run()
        self.assertTrue(murn.status.finished)

        murn.remove()
        job_ser.remove()

    def test_fitting_routines(self):
        ref_job = self.project.create.job.Lammps('ref')
        murn = ref_job.create_job('Murnaghan', 'murn')
        murn.structure = self.basis
        # mock murnaghan run with data from:
        #   ref_job = pr.create.job.Lammps('Lammps')
        #   ref_job.structure = pr.create_structure('Al','fcc', 4.0).repeat(3)
        #   ref_job.potential = '1995--Angelo-J-E--Ni-Al-H--LAMMPS--ipr1'
        #   murn = ref_job.create_job(ham.job_type.Murnaghan, 'murn')
        #   murn.run()
        energies = np.array([-88.23691773, -88.96842984, -89.55374317, -90.00642629,
                             -90.33875009, -90.5618246, -90.68571886, -90.71957679,
                             -90.67170222, -90.54964935, -90.36029582])
        volume = np.array([388.79999999, 397.44, 406.08, 414.71999999,
                           423.35999999, 431.99999999, 440.63999999, 449.27999999,
                           457.92, 466.55999999, 475.19999999])
        murn._hdf5["output/volume"] = volume
        murn._hdf5["output/energy"] = energies
        murn._hdf5["output/equilibrium_volume"] = 448.4033384110422
        murn.status.finished = True

        self.assertIsInstance(murn.plot(plt_show=True), matplotlib.axes.Axes)
        with self.subTest(msg="standard polynomial fit"):
            self.assertAlmostEqual(-90.71969974284912, murn.equilibrium_energy)
            self.assertAlmostEqual(448.1341230545222, murn.equilibrium_volume)

        with self.subTest(msg="polynomial fit with fit_order = 2"):
            murn.fit_polynomial(fit_order=2)
            self.assertAlmostEqual(-90.76380033222287, murn.equilibrium_energy)
            self.assertAlmostEqual(449.1529040727273, murn.equilibrium_volume)

        with self.subTest(msg='birchmurnaghan'):
            murn.fit_birch_murnaghan()
            self.assertAlmostEqual(-90.72005405262217, murn.equilibrium_energy)
            self.assertAlmostEqual(448.41909755611437, murn.equilibrium_volume)

        with self.subTest(msg="vinet"):
            murn.fit_vinet()
            self.assertAlmostEqual(-90.72000006839492, murn.equilibrium_energy)
            self.assertAlmostEqual(448.40333840970357, murn.equilibrium_volume)

        with self.subTest(msg='murnaghan'):
            murn.fit_murnaghan()
            self.assertAlmostEqual(-90.72018572197015, murn.equilibrium_energy)
            self.assertAlmostEqual(448.4556825322108, murn.equilibrium_volume)


if __name__ == "__main__":
    unittest.main()
