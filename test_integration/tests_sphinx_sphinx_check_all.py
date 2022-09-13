from pyiron_atomistics import Project
import numpy as np
import unittest
import os


class TestSphinx(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.file_location = os.path.dirname(os.path.abspath(__file__))
        cls.project = Project('SPX_CHECK_ALL')
        cls.a_Fe = 2.83
        cls.a_Al = 4.024

    @classmethod
    def tearDownClass(cls):
        cls.project.remove(enable=True)

    def test_Fe_nonmag(self):
        job = self.project.create_job(self.project.job_type.Sphinx, 'spx_Fe_nonmag')
        job.structure = self.project.create_structure('Fe', 'bcc', self.a_Fe)
        job.calc_static()
        job.run()
        if np.linalg.norm(job['output/generic/forces'])>1.0e-4:
            raise ValueError('Forces wrong')
        if np.linalg.norm(job.structure.positions-job['output/generic/positions'][-1])>1.0e-4:
            raise ValueError('Positions not correctly parsed')
        if np.linalg.norm(job.structure.cell-job['output/generic/cells'][-1])>1.0e-4:
            raise ValueError('Cells not correctly parsed')
        if 'atom_spins' in job['output/generic/dft'].list_nodes():
            raise AssertionError('spins present')
        if np.abs(job['output/generic/volume']-np.linalg.det(job.structure.cell)) > 1.0e-4:
            raise ValueError('Volume wrong')
        if np.linalg.norm(job.structure.positions-job['output/generic/positions'][0])>1.0e-4:
            raise ValueError('Positions not parsed properly')
        job = self.project.create_job(self.project.job_type.Sphinx, 'spx_Fe_ferro')
        job.structure = self.project.create_structure('Fe', 'bcc', self.a_Fe)
        job.structure.set_initial_magnetic_moments([2, 2])
        job.calc_static()
        job.run()
        self.assertGreater(
            np.diff([
                self.project.load(job_name)['output/generic/energy_tot'][0]
                for job_name in ['spx_Fe_ferro', 'spx_Fe_nonmag']]
            )[0],
            0,
            'BCC Fe erromagnetic state has lower energy than nonmagnetic state'
        )

    def test_Fe_ferro_C(self):
        job = self.project.create_job(self.project.job_type.Sphinx, 'spx_Fe_ferro_C')
        job.structure = self.project.create_structure('Fe', 'bcc', self.a_Fe)
        job.structure.set_initial_magnetic_moments([2, 2])
        job.structure += self.project.create_atoms(elements=['C'], positions=[[0, 0, 0.5*self.a_Fe]], magmoms=[0])
        job.calc_static()
        job.run()
        if np.linalg.norm(job.structure.positions-job['output/generic/positions'][-1])>1.0e-4:
            raise ValueError('Positions not correctly parsed')
        job = self.project.create_job(self.project.job_type.Sphinx, 'spx_Al')
        job.structure = self.project.create_structure('Al', 'fcc', self.a_Al)
        job.calc_static()
        job.run()
        job = job.restart(from_charge_density=False, from_wave_functions=False)
        job.run()
        if 'spx_Al_restart' not in list(self.project.job_table().job):
            raise AssertionError('restart job not found')
        if np.abs(self.project.load('spx_Al')['output/generic/energy_tot'][-1]-self.project.load('spx_Al_restart')['output/generic/energy_tot'][-1])>1.0e-3:
            raise ValueError('Energy value after restart too different')

    def test_Al_minimize(self):
        job = self.project.create_job(self.project.job_type.Sphinx, 'spx_Al_minimize')
        job.structure = self.project.create_structure('Al', 'fcc', self.a_Al)
        job.structure.positions[0,0] += 0.01
        job.calc_minimize()
        job.run()
        E = job['output/generic/energy_tot']
        if E[0]-E[1]<0:
            raise AssertionError('Energy not decreased')

    def test_check_overlap(self):
        job = self.project.create_job(self.project.job_type.Sphinx, 'spx_check_overlap')
        job.structure = self.project.create_structure('Fe', 'bcc', 2.832)
        job.set_check_overlap(False)
        job.calc_static()
        job.run()
        job = self.project.create_job(self.project.job_type.Sphinx, 'spx_symmetry')
        job.structure = self.project.create_structure('Fe', 'bcc', 2.832)
        job.fix_symmetry = False
        job.calc_static()
        job.run()
        job = self.project.create_job(self.project.job_type.Sphinx, 'spx_Fe_ferro_constraint')
        job.structure = self.project.create_structure('Fe', 'bcc', self.a_Fe)
        job.structure.set_initial_magnetic_moments([2, 2])
        job.fix_spin_constraint = True
        job.calc_static()
        job.run()
        if np.linalg.norm(job['output/generic/dft/atom_spins']-job.structure.get_initial_magnetic_moments())>1.0e-4:
            raise AssertionError('Magnetic moments either not properly parsed or constraining not working')

    def test_Al_save_memory(self):
        job = self.project.create_job(self.project.job_type.Sphinx, 'spx_Al_save_memory')
        job.structure = self.project.create_structure('Al', 'fcc', self.a_Al)
        job.input['SaveMemory'] = True
        job.calc_static()
        job.run()

    def test_Al_interactive(self):
        job = self.project.create_job(self.project.job_type.Sphinx, 'spx_Al_interactive')
        job.structure = self.project.create_structure('Al', 'fcc', self.a_Al)
        job.structure.positions[0,0] += 0.01
        job.server.run_mode.interactive = True
        job.calc_static()
        minim = job.create_job(self.project.job_type.SxExtOptInteractive, 'sxextopt_Al')
        minim.run()

    def test_nonmodal2(self):
        job = self.project.create_job(self.project.job_type.Sphinx, 'nonmodal2')
        job.structure = self.project.create_structure('Al', 'fcc', self.a_Al)
        job.calc_static()
        job.save()
        job_reload = self.project.load(job.job_name)
        job_reload.run()
        self.assertTrue(job['output/generic/dft/bands_e_fermi'] is not None)
        self.assertTrue(job_reload.status.finished)

    def test_sxextopt_Fe(self):
        spx = self.project.create_job('Sphinx', 'spx_sxextopt_Fe')
        spx.structure = self.project.create_structure('Fe', 'bcc', 2)
        spx.structure.set_initial_magnetic_moments([2, 2])
        spx.server.run_mode.interactive = True
        spx.calc_static()
        sxextopt = self.project.create_job('SxExtOptInteractive', 'sxextopt_Fe')
        sxextopt.ref_job = spx
        sxextopt.save()
        sxextopt = self.project.load('sxextopt_Fe')
        sxextopt.run()

if __name__ == "__main__":
    unittest.main()
