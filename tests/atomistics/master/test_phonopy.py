# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from pyiron_atomistics._tests import TestWithCleanProject


class TestPhonopy(TestWithCleanProject):

    def test_run(self):
        job = self.project.create.job.HessianJob('job_test')
        basis = self.project.create.structure.bulk('Fe')
        basis.set_initial_magnetic_moments([2]*len(basis))
        job.set_reference_structure(basis)
        phono = self.project.create.job.PhonopyJob('phono')
        phono.ref_job = job
        structure = phono.list_structures()[0]
        magmoms = structure.get_initial_magnetic_moments()
        self.assertAlmostEqual(sum(magmoms-2), 0)
        rep = phono._phonopy_supercell_matrix().diagonal().astype(int)
        job._reference_structure.set_repeat(rep)
        job.structure.set_repeat(rep)
        job.set_force_constants(1)
        # phono.run() # removed because somehow it's extremely slow

    def test_non_static_ref_job(self):
        phon_ref_job = self.project.create.job.Lammps('ref_job')
        phon_ref_job.structure = self.project.create.structure.bulk('Al', a=4.5)
        phon_ref_job.potential = phon_ref_job.list_potentials()[0]
        phon_ref_job.calc_minimize()
        phonopy_job = self.project.create.job.PhonopyJob('phonopy_job')
        phonopy_job.ref_job = phon_ref_job
        with self.assertRaises(ValueError):
            phonopy_job.validate_ready_to_run()
