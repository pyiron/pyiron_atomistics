from pyiron_atomistics import Project
import numpy as np
pr = Project('SPX_CHECK_ALL')
a_Fe = 2.83
a_Al = 4.024
job = pr.create_job(pr.job_type.Sphinx, 'spx_Fe_nonmag')
job.structure = pr.create_structure('Fe', 'bcc', a_Fe)
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
job = pr.create_job(pr.job_type.Sphinx, 'spx_Fe_ferro')
job.structure = pr.create_structure('Fe', 'bcc', a_Fe)
job.structure.set_initial_magnetic_moments([2, 2])
job.calc_static()
job.run()
if pr.load('spx_Fe_ferro')['output/generic/energy_tot'][0]-pr.load('spx_Fe_nonmag')['output/generic/energy_tot'][0]>0:
    raise ValueError('BCC Fe erromagnetic state has lower energy than nonmagnetic state')
job = pr.create_job(pr.job_type.Sphinx, 'spx_Fe_ferro_C')
job.structure = pr.create_structure('Fe', 'bcc', a_Fe)
job.structure.set_initial_magnetic_moments([2, 2])
job.structure += pr.create_atoms(elements=['C'], positions=[[0, 0, 0.5*a_Fe]], magmoms=[0])
job.calc_static()
job.run()
if np.linalg.norm(job.structure.positions-job['output/generic/positions'][-1])>1.0e-4:
    raise ValueError('Positions not correctly parsed')
job = pr.create_job(pr.job_type.Sphinx, 'spx_Al')
job.structure = pr.create_structure('Al', 'fcc', a_Al)
job.calc_static()
job.run()
job = job.restart(from_charge_density=False, from_wave_functions=False)
job.run()
if 'spx_Al_restart' not in list(pr.job_table().job):
    raise AssertionError('restart job not found')
if np.abs(pr.load('spx_Al')['output/generic/energy_tot'][-1]-pr.load('spx_Al_restart')['output/generic/energy_tot'][-1])>1.0e-3:
    raise ValueError('Energy value after restart too different')
job = pr.create_job(pr.job_type.Sphinx, 'spx_Al_minimize')
job.structure = pr.create_structure('Al', 'fcc', a_Al)
job.structure.positions[0,0] += 0.01
job.calc_minimize()
job.run()
E = job['output/generic/energy_tot']
if E[0]-E[1]<0:
    raise AssertionError('Energy not decreased')
job = pr.create_job(pr.job_type.Sphinx, 'spx_check_overlap')
job.structure = pr.create_structure('Fe', 'bcc', 2.832)
job.set_check_overlap(False)
job.calc_static()
job.run()
job = pr.create_job(pr.job_type.Sphinx, 'spx_symmetry')
job.structure = pr.create_structure('Fe', 'bcc', 2.832)
job.fix_symmetry = False
job.calc_static()
job.run()
job = pr.create_job(pr.job_type.Sphinx, 'spx_Fe_ferro_constraint')
job.structure = pr.create_structure('Fe', 'bcc', a_Fe)
job.structure.set_initial_magnetic_moments([2, 2])
job.fix_spin_constraint = True
job.calc_static()
job.run()
if np.linalg.norm(job['output/generic/dft/atom_spins']-job.structure.get_initial_magnetic_moments())>1.0e-4:
    raise AssertionError('Magnetic moments either not properly parsed or constraining not working')
job = pr.create_job(pr.job_type.Sphinx, 'spx_Al_submit')
job.structure = pr.create_structure('Al', 'fcc', a_Al)
job.calc_static()
job.run()
job = pr.create_job(pr.job_type.Sphinx, 'spx_Al_save_memory')
job.structure = pr.create_structure('Al', 'fcc', a_Al)
job.input['SaveMemory'] = True
job.calc_static()
job.run()
job = pr.create_job(pr.job_type.Sphinx, 'spx_Al_interactive')
job.structure = pr.create_structure('Al', 'fcc', a_Al)
job.structure.positions[0,0] += 0.01
job.server.run_mode.interactive = True
job.calc_static()
minim = job.create_job(pr.job_type.SxExtOptInteractive, 'sxextopt_Al')
minim.run()
job = pr.create_job(pr.job_type.Sphinx, 'nonmodal2')
job.structure = pr.create_structure('Al', 'fcc', a_Al)
job.calc_static()
job.save()
job_reload = pr.load(job.job_name)
job_reload.run()
job['output/generic/dft/bands_e_fermi']
spx = pr.create_job('Sphinx', 'spx_sxextopt_Fe')
spx.structure = pr.create_structure('Fe', 'bcc', 2)
spx.structure.set_initial_magnetic_moments([2, 2])
spx.server.run_mode.interactive = True
spx.calc_static()
sxextopt = pr.create_job('SxExtOptInteractive', 'sxextopt_Fe')
sxextopt.ref_job = spx

sxextopt.save()
sxextopt = pr.load('sxextopt_Fe')
sxextopt.run()
if not all(pr.job_table().status=='finished'):
    jt = pr.job_table()
    print(jt[jt.status!='finished'])
    raise AssertionError('Some simulations failed')
pr.remove(enable=True)
