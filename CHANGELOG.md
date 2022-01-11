# Unreleased

# pyiron_atomistics-0.2.34

# pyiron_atomistics-0.2.33
- Some test modernization (#425)
- Decrease convergence goal (#479)
- revert error (#474)
- Write POSCAR in direct coordinate when selective dynamics is on (#448)
- Strain (#465)
- Tessellation neighbors (#413)
- add error message in qha (#466)
- add assertion test to symmetry permutation (#463)
- Refactor symmetrize_vectors (#462)
- Add callback function for LAMMPS (#458)
- Use numpy.all in NeighborsTrajectory (#461)
- dependency updates: #472, #471, #469, #470, #480

# pyiron_atomistics-0.2.32
- More structures (#259)
- Add proper smearing for VASP  (#454)
- Try Python 3.10 (#450)
- Remove unused properties in NeighborsTrajectory (#457)
- Do not polute resource_path (#453)
- dependencies: #459

# pyiron_atomistics-0.2.31
- Make NeighborTrajectory simpler (#445)
- Use state instead of Settings (#424)
- use unwrapped_positions in displacements (#451)
- Store shells in Neighbors Trajectory (#444)
- replace full output info in __str__ by chemical formula (#439)
- Fix Lammps h5md parser (#446)
- Use FlattenedStorage in NeighborsTrajectory (#387)
- Clear up TypeError in StructureContainer.append (#441)
- Add clear error when calling animate_structure on empty job (#442)
- add get_primitive_cell from spglib (#433)
- dependencies: #449, #432, #437, #430, #456
- GitHub infrastructure: #438 

# pyiron_atomistics-0.2.30
- Parse irreductible kpoints properly (#423)
- Cluster (atom) positions (#419) 
- Make Methfessel Paxton default in SPHInX (#416)
- Pyscal solid liquid (#414)
- dependency updates: #410, #411, #431
- Infrastructure: #429

The update to the new `pyiron_base` version > 0.4 (#431) fixes some bugs in the writing of numerical data to our storage backend (hdf), see [pyiron_base release notes](https://github.com/pyiron/pyiron_base/releases/tag/pyiron_base-0.4.0).

# pyiron_atomistics-0.2.29
- VASP doesn't save stresses to HDF when run on ISIF=2 (default) (#354) 
- Allow hcp 4-axes indices as well (#390)
- [minor] clean up find_mic (#403)
- Fix scaled test (#389)
- dependency updates: #398, #406, #404, #408 

# pyiron_atomistics-0.2.28
- Fix structure check in `restart()` (#392)

# pyiron_atomistics-0.2.27
- Adapt to removal of load_object  (#386)
- Don't conflict with HasHDF in StructureStorage  (#376)

# pyiron_atomistics-0.2.26
- pep8 for sphinx (#373)
- Bind creator at import (#347)
- Lammps style full - fix numpy warnings (#367)
- Water interactive fix  (#366)
- Update base.py (#346)
- Support full style without bonds (#365)
- Strain (#364)
- Use StructureStorage in StructureContainer (#344)
- Explicitly specify water potential (#353)
- Add automatic labeling for integration tests (#361)
- Consistent atoms order (#338)
- Add Wrapper for Atomsk Structure Creation (#260)
- replace strain by master job name (#355)
- change pointer to copy (#350)
- Fixes for pyiron table (#329)
- Don't specify force tolerance if only optimizing cell (#339)

# pyiron_atomistics-0.2.25
- Strain (#328)
- Properly implement `HasStructure` for `StructureContainer`  (#331)
- Move StructureStorage from pyiron_contrib  (#327)
- Use correct role name for sphinx  (#330)

# pyiron_atomistics-0.2.24
Update to pyiron_base-0.3.0

# pyiron_atomistics-0.2.23
- Hotfix: Handle empty indices  (#284)
- Interactive units (#295)

# pyiron_atomistics-0.2.22
- Outcar bands (#299)
- Add error message if `Murnaghan.plot` is called on unfinished jobs  (#291)
- Make Atoms() faster via caches  (#224)
- Create new Atoms in `get_structure` if size changes  (#241) 
- add mode in `get_neighborhood` (#285)
- Equivalent points (#280)

# pyiron_atomistics-0.2.21
- Neighbor analysis for the entire trajectory (#251)
- Get neighbors (#239)
- Save all VCSGC parameters in generic input (#262)
- Only force skew when tensions are applied(#263)

# pyiron_atomistics-0.2.20
Bugfix:
- Estimate width (#272)

# pyiron_atomistics-0.2.19
* Extending the units class (#271)
* interactive_prepare  (#235)
* Centrosymmetry (#261)
* Add HasStructure to Trajectory  (#270)
* scf_residue unit corrected (#266)

# pyiron_atomistics-0.2.18
- Analysis of Atomic structure updated
- Update to vasp job: raise error for non-zero pressure

# pyiron_atomistics-0.2.17
- Enable `use_pressure = False` (#237)

# pyiron_atomistics-0.2.16
- Get interstitials: New feature to find interstitial sites (#219)
- Phonopy options: Expose additional argument `number_of_snapshots` (#213)
- VASP - change NSW if interactive (#220)
- Use new copy hook (#201)


# pyiron_atomistics-0.2.15
- Proper conversion of LAMMPS output into pyiron units on parsing
- Fix to stop convergence checks for interactive jobs! 
- Support newest LAMMPS version 

# pyiron_atomistics-0.2.14
* Add optional method to allow non-integer frames
* Bump to pyiron_base-0.2.15 to avoid breaking copy bug, see [here](https://github.com/pyiron/pyiron_atomistics/issues/223)

# pyiron_atomistics-0.2.13
- Steinhardt interface
- remove tree-internal periodic boundary detection
- Make ´delete_existing_job´ available to the ´AtomisticGenericJob´ class


# pyiron_atomistics-0.2.12
- getting structure faster
- updating atoms symbols via atoms.symbols
- Steinhardt parameter implementation
- Badr analysis
