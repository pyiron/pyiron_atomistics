# FAQ

## How to cite pyiron?
To cite pyiron and the corresponding codes, please follow the instructions on the publication page.

## What units does pyiron use?

* mass = atomic mass units
* distance = Angstroms
* time = femtoseconds
* energy = eV
* velocity = Angstroms/femtoseconds
* force = eV/Angstrom
* temperature = Kelvin
* pressure = GPa
* charge = multiple of electron charge (1.0 is a proton)

## How to import existing calculation?
Importing existing calculations is currently only supported for VASP. A tutorial how to import existing calculations is
available in the tutorial section. 

## How to import structures from files or existing databases?
To read structure formats you can use ASE and then convert the structure to a pyiron structure using:
```python
from ase.build import bulk
from pyiron_atomistics import ase_to_pyiron
pyiron_structure = ase_to_pyiron(bulk("Al")) 
```

## How to install pyiron?
pyiron is designed to be installed as centralized service on your local computer cluster, rather than a local installation
on each individual workstation. To test pyiron online or with a local installation, please follow the instructions on the
[installation page](https://pyiron-atomistics.readthedocs.io/en/jupyterbook/installation.html).

## How do I install additional codes for pyiron?
When installing pyiron via conda it is possible to install most opensource codes via conda as well: 

| code         | `job_type` | How to install ?                            |
|--------------|------------|---------------------------------------------|
| GAUSSIAN     | `Gaussian` | Compile on your own (commercial code)       |
| Gpaw         | `Gpaw`     | `conda install -c conda-forge gpaw`         |
| LAMMPS       | `Lammps`   | `conda install -c conda-forge lammps`       |
| S/PHI/nX     | `Sphinx`   | `conda install -c conda-forge sphinxdft`    |
| sqsgenerator | `SQSJob`   | `conda install -c conda-forge sqsgenerator` |
| VASP         | `Vasp`     | Compile on your own (commercial code)       |

## How to use a custom Pseudo potential in VASP?
There are two ways to assign custom potentials in VASP, either you can change the pseudo potential for all atoms of one 
species: 

```python
job_vasp.potential.Fe = "~/resources/vasp/potentials/potpaw_PBE/Fe/POTCAR"
```

Or alternatively you can change the pseudo potential of a single atom by creating a new element: 

```python
my_fe = pr.create_element(
    new_element_name="Fe", 
    parent_element="Fe", 
    potential_file="~/resources/vasp/potentials/potpaw_PBE/Fe/POTCAR"
) 
job_vasp.structure[0] = my_fe
```

## How to use VASP tags which are not supported by pyiron?
The underlying input of any simulation code in pyiron can be directly accessed. For VASP you can change the `INCAR` 
parameters using the VASP specific syntax: 

```python
job_vasp.input.incar["ENCUT"] = 320.0  # eV 
```

## How to use a custom potential in LAMMPS?
A custom empirical potential (here, a hybrid potential) can be defined in the following format:
```python
custom_potential = pd.DataFrame({    
    'Name': ['SrTiO3_Pedone'],
    'Filename': [[]],
    'Model': ['Custom'],
    'Species': [['O', 'Sr', 'Ti']],
    'Config': [['atom_style full\n',  # I use 'full' here as atom_style 'charge' gives the same result
                '## create groups ###\n', 
                'group O type 1\n', 
                'group Sr type 2\n', 
                'group Ti type 3\n', 
                '\n', 
                '## set charges - beside manually ###\n', 
                'set group O charge -1.2000\n',  
                'set group Sr charge 1.2000\n',
                'set group Ti charge 2.4000\n',
                '\n', 
                'pair_style hybrid/overlay morse 15.0 mie/cut 15.0 coul/long 15.0 beck 15.0\n', 
                'pair_coeff * * coul/long\n', 
                'pair_coeff 1 2 beck 3.0 0 0 0 0\n', 
                'pair_coeff 1 3 beck 1.0 0 0 0 0\n', 
                'pair_coeff 1 1 beck 22.0 0 0 0 0\n', 
                'pair_coeff 1 2 mie/cut 3.0 1.0 12.0 0\n', 
                'pair_coeff 1 3 mie/cut 1.0 1.0 12.0 0\n', 
                'pair_coeff 1 1 mie/cut 22.0 1.0 12.0 0\n', 
                'pair_coeff 1 2 morse 0.019623 1.8860 3.32833\n', 
                'pair_coeff 1 3 morse 0.024235 2.2547 2.708943\n', 
                'pair_coeff 1 1 morse 0.042395 1.3793 3.618701\n', 
                'kspace_style ewald 1.0e-8\n']]
  })
```

The lines in `Config` will be written to the LAMMPS `potential.inp` file. Make sure that the arrangement of the species
in `Species` is the same as the group types `create groups` within `Config`. Otherwise, a mixup or the species may occur
in the LAMMPS `structure.inp` file.

The potential can then be used by assigning `job.potential = custom_potential`.

## How to extend the potential database inside pyiron?
By default pyiron provides access to the OpenKIM and NIST databases for interatomic potentials and individual potentials
can be added as discussed above. While there was an option to extend the default database this option was disabled as it
decreased the reproducibility of simulation protocols. 

## How to link your own executable?
The linking of executables is explained as part of the installation in the section of advanced configuarion options. By
default pyiron links to the executables provided by conda but you can accelerate you calculation by compiling your own 
version of a given simulation code which is optimized for your hardware. 

## How to send a calculation to the background ?
While most examples execute calculations inline or in modal mode, it is also possible to send calculation in the 
background. 

```python
job.server.run_mode.non_modal = True
job.run()
print("execute other commands while the job is running.")
pr.wait_for_job(job)
```
  
In this example the job is executed in the background, while the print command is already executed. Afterwards the 
project object waits for the execution of the job in the background to be finished. 

## How to submit a calculation to the queuing system?
Just like executing calculation in the background it is also possible to submit calculation to the queuing system: 

```python
job.server.list_queues()  # returns a list of queues available on the system
job.server.view_queues()  # returns a DataFrame listing queues and their settings 
job.server.queue = "my_queue"  # select a queue 
job.server.cores = 80          # set the number of cores 
job.server.run_time = 3600     # set the run time in seconds
job.run()
```

For the queuing system to be available in pyiron it is necessary to configure it. The configuration of different queuing
systems is explained in the installation. 

## How to setup spin constraint calculation?
pyiron supports setting up spin constrained calculations for VASP using the generic spin_constraint property: 

```python
job_vasp.spin_constraints = 1
```

## What is the meaning of the name - pyiron?
pyiron is the combination of **py** + **iron** connecting Python, the programming language with iron as pyiron was
initially developed at the Max Planck Institut für Eisenforschung (iron research).

## Which output quantities are stored in pyiron?
Generic

| tag                 | dimension                                   | description                                                                 |   VASP    | SPHInX  | LAMMPS |
|---------------------|---------------------------------------------|-----------------------------------------------------------------------------|:---------:|:-------:|:------:|
| time                | N<sub>step</sub>                            | simulation time ( fs )                                                      |           |         |   x    |
| steps               | N<sub>step</sub>                            | time steps                                                                  |           |         |   x    |
| unwrapped_positions | N<sub>step</sub> x N<sub>atom</sub> x 3     | unwrapped atom coordinates ( Angstrom )                                     |     x     |    x    |   x    |
| positions           | N<sub>step</sub> x N<sub>atom</sub> x 3     | wrapped atom coordinates ( Angstrom )                                       |     x     |    x    |   x    |
| velocities          | N<sub>step</sub> x N<sub>atom</sub> x 3     | velocity of each atom ( Angstrom/fs )                                       |           |         |        |
| forces              | N<sub>step</sub> x N<sub>atom</sub> x 3     | force on each atom ( eV/ Angstrom )                                         |     x     |    x    |   x    |
| cells               | N<sub>step</sub> x 3 x 3                    | cell dimensions (cf. VASP website) ( Angstrom )                             |     x     |    x    |   x    |
| energy_tot          | N<sub>step</sub>                            | total energy of the system ( eV )                                           |     x     |    x    |   x    |
| energy_kin          | N<sub>step</sub>                            | kinetic energy of the system ( eV )                                         |     x     |         |        |
| energy_pot          | N<sub>step</sub>                            | potential energy of the system ( eV )                                       |     x     |         |        |
| pressures           | N<sub>step</sub>  x 3 x 3                   | pressures ( GPa )                                                           |           |         |   x    |
| temperature         | N<sub>step</sub>                            | temperature ( K )                                                           |     x     |         |   x    |
| volume              | N<sub>step</sub>                            | supercell volume ( Angstrom<sup>3</sup> )                                   |     x     |    x    |   x    |
| atom_voronoi        | N<sub>step</sub> x N<sub>atom</sub>         | Voronoi volume of each atom ( Angstrom<sup>3</sup> )                        |           |         |        |
| atom_stress         | N<sub>step</sub> x N<sub>atom</sub> x 3 x 3 | stress per atom x atomic volume ( eV )                                      |           |         |   x    |
| atom_centro         | N<sub>step</sub> x N<sub>atom</sub>         | centro-symmetry parameter ( Angstrom<sup>2</sup> )                          |           |         |        |
| atom_displace       | N<sub>step</sub> x N<sub>atom</sub> x 3     | displacement of each atom with respect to the initial position ( Angstrom ) |           |         |        |
| computation_time    | N<sub>step</sub>                            | computation time of the simulation ( s )                                    |           |    x    |        |

Density Functional Theory (DFT)

| tag                   | dimension                                                     | description                                       | VASP | SPHInX | LAMMPS |
|-----------------------|---------------------------------------------------------------|---------------------------------------------------|:----:|:------:|:------:|
| (scf_)energy_int      | N<sub>step</sub>                                              | internal energy ( eV )                            |      |   x    |        |
| (scf_)energy_free     | N<sub>step</sub>                                              | free energy, same as energy_tot in generic ( eV ) | x    |   x    |        |
| (scf_)energy_zero     | N<sub>step</sub>                                              | extrapolated energy, sigma to 0 ( eV )            | x    |   x    |        |
| (scf_)energy_band     | N<sub>step</sub>                                              | band gap energy ( eV )                            |      |   x    |        |
| (scf_)residue         | N<sub>step</sub> ( x 2 )                                      | energy residue ( eV )                             |      |   x    |        |
| atoms_(scf_)spins     | N<sub>step</sub> x N<sub>atom</sub>                           | spin moment of each atom ( Bohr magneton )        |      |   x    |        |
| (scf_)magnetic_forces | N<sub>step</sub> x N<sub>atom</sub>                           | spin forces ? ( eV/Bohr magneton )                |      |   x    |        |
| atom_spin_constraints | N<sub>step</sub> x N<sub>atom</sub>                           | spin constraints ( Bohr magneton )                |      |   x    |        |
| bands_e_fermi         | N<sub>step</sub>                                              | fermi energy ( eV )                               |      |   x    |        |
| bands_occ             | N<sub>step</sub> ( x 2 ) x N<sub>k</sub> x N<sub>states</sub> | occupancy                                         |      |   x    |        |
| bands_k_weights       | N<sub>k</sub>                                                 | weight of each k point                            |      |   x    |        |
| bands_eigen_values    | N<sub>step</sub> ( x 2 ) x N<sub>k</sub> x N<sub>states</sub> | eigenspectrums ( eV )                             |      |   x    |        |
| scf_convergence       | N<sub>step</sub>                                              | convergence of each ionic step                    |      |   x    |        |

* N<sub>step</sub> refers to ionic steps and not electronic steps
* properties preceded by scf_ contain the values of each electronic step except for scf_convergence
* ( x 2 ) refers to the additional column which appears only in magnetic calculations
* if the crosses under VASP, SPHInX or LAMMPS are missing, the corresponding properties are not implemented

