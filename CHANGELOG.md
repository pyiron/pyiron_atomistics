# Unreleased

# pyiron_atomistics-0.2.37

- Remove _QhullUser import ([#527](https://github.com/pyiron/pyiron_atomistics/pull/527))
- Add pint as dependency ([#529](https://github.com/pyiron/pyiron_atomistics/pull/529))
- Some docstrings for phono ([#447](https://github.com/pyiron/pyiron_atomistics/pull/447))
- Dependency updates:  [#522](https://github.com/pyiron/pyiron_atomistics/pull/522), [#523](https://github.com/pyiron/pyiron_atomistics/pull/523), [#524](https://github.com/pyiron/pyiron_atomistics/pull/524)

# pyiron_atomistics-0.2.36
- Update Readthedocs ([#519](https://github.com/pyiron/pyiron_atomistics/pull/519))
- Selected animation ([#513](https://github.com/pyiron/pyiron_atomistics/pull/513))
- Spx refactor ([#505](https://github.com/pyiron/pyiron_atomistics/pull/505))
- Make codebase black ([#507](https://github.com/pyiron/pyiron_atomistics/pull/507) and [#517](https://github.com/pyiron/pyiron_atomistics/pull/517))
- Dependency updates: [#494](https://github.com/pyiron/pyiron_atomistics/pull/494), [#515](https://github.com/pyiron/pyiron_atomistics/pull/515), [#516](https://github.com/pyiron/pyiron_atomistics/pull/516), [#510](https://github.com/pyiron/pyiron_atomistics/pull/510), [#503](https://github.com/pyiron/pyiron_atomistics/pull/503), [#509](https://github.com/pyiron/pyiron_atomistics/pull/509), [#520](https://github.com/pyiron/pyiron_atomistics/pull/520)

# pyiron_atomistics-0.2.35
- High index surfaces  ([#400](https://github.com/pyiron/pyiron_atomistics/pull/400))
- Consistent indices for StructureStorage.get_structures ([#482](https://github.com/pyiron/pyiron_atomistics/pull/482))
- Dependency updates: [#504](https://github.com/pyiron/pyiron_atomistics/pull/504)

# pyiron_atomistics-0.2.34
Since pyiron_atomistics-0.2.29 stresses are by default calculated and stored for VASP calculations. However, the stored values were stored in an undocumented order and had the wrong sign. [#497](https://github.com/pyiron/pyiron_atomistics/pull/497) provides a fix by storing the stress as an unambiguous matrix. 

- Update to pyiron_base-0.5.0 and add changelog ([#501](https://github.com/pyiron/pyiron_atomistics/pull/501))
- Save VASP stresses as matrix ([#497](https://github.com/pyiron/pyiron_atomistics/pull/497))
- Drop python3.7 support ([#500](https://github.com/pyiron/pyiron_atomistics/pull/500))
- Add cell_only to Vasp.calc_minimize  ([#498](https://github.com/pyiron/pyiron_atomistics/pull/498))
- Use current structure instead of output to update previous structure ([#483](https://github.com/pyiron/pyiron_atomistics/pull/483))
- give possibility of setting log_file in LAMMPS ([#488](https://github.com/pyiron/pyiron_atomistics/pull/488))
- Replace Random Atomistics by Lennard Jones in testing ([#478](https://github.com/pyiron/pyiron_atomistics/pull/478))
- Dependency updates: [#485](https://github.com/pyiron/pyiron_atomistics/pull/485), [#496](https://github.com/pyiron/pyiron_atomistics/pull/496), [#493](https://github.com/pyiron/pyiron_atomistics/pull/493), [#489](https://github.com/pyiron/pyiron_atomistics/pull/489)

# pyiron_atomistics-0.2.33
- Some test modernization ([#425](https://github.com/pyiron/pyiron_atomistics/pull/425))
- Decrease convergence goal ([#479](https://github.com/pyiron/pyiron_atomistics/pull/479))
- revert error ([#474](https://github.com/pyiron/pyiron_atomistics/pull/474))
- Write POSCAR in direct coordinate when selective dynamics is on ([#448](https://github.com/pyiron/pyiron_atomistics/pull/448))
- Strain ([#465](https://github.com/pyiron/pyiron_atomistics/pull/465))
- Tessellation neighbors ([#413](https://github.com/pyiron/pyiron_atomistics/pull/413))
- add error message in qha ([#466](https://github.com/pyiron/pyiron_atomistics/pull/466))
- add assertion test to symmetry permutation ([#463](https://github.com/pyiron/pyiron_atomistics/pull/463))
- Refactor symmetrize_vectors ([#462](https://github.com/pyiron/pyiron_atomistics/pull/462))
- Add callback function for LAMMPS ([#458](https://github.com/pyiron/pyiron_atomistics/pull/458))
- Use numpy.all in NeighborsTrajectory ([#461](https://github.com/pyiron/pyiron_atomistics/pull/461))
- dependency updates: [#472](https://github.com/pyiron/pyiron_atomistics/pull/472), [#471](https://github.com/pyiron/pyiron_atomistics/pull/471), [#469](https://github.com/pyiron/pyiron_atomistics/pull/469), [#470](https://github.com/pyiron/pyiron_atomistics/pull/470), [#480](https://github.com/pyiron/pyiron_atomistics/pull/480)

# pyiron_atomistics-0.2.32
- More structures ([#259](https://github.com/pyiron/pyiron_atomistics/pull/259))
- Add proper smearing for VASP  ([#454](https://github.com/pyiron/pyiron_atomistics/pull/454))
- Try Python 3.10 ([#450](https://github.com/pyiron/pyiron_atomistics/pull/450))
- Remove unused properties in NeighborsTrajectory ([#457](https://github.com/pyiron/pyiron_atomistics/pull/457))
- Do not polute resource_path ([#453](https://github.com/pyiron/pyiron_atomistics/pull/453))
- dependencies: [#459](https://github.com/pyiron/pyiron_atomistics/pull/459)

# pyiron_atomistics-0.2.31
- Make NeighborTrajectory simpler ([#445](https://github.com/pyiron/pyiron_atomistics/pull/445))
- Use state instead of Settings ([#424](https://github.com/pyiron/pyiron_atomistics/pull/424))
- use unwrapped_positions in displacements ([#451](https://github.com/pyiron/pyiron_atomistics/pull/451))
- Store shells in Neighbors Trajectory ([#444](https://github.com/pyiron/pyiron_atomistics/pull/444))
- replace full output info in __str__ by chemical formula ([#439](https://github.com/pyiron/pyiron_atomistics/pull/439))
- Fix Lammps h5md parser ([#446](https://github.com/pyiron/pyiron_atomistics/pull/446))
- Use FlattenedStorage in NeighborsTrajectory ([#387](https://github.com/pyiron/pyiron_atomistics/pull/387))
- Clear up TypeError in StructureContainer.append ([#441](https://github.com/pyiron/pyiron_atomistics/pull/441))
- Add clear error when calling animate_structure on empty job ([#442](https://github.com/pyiron/pyiron_atomistics/pull/442))
- add get_primitive_cell from spglib ([#433](https://github.com/pyiron/pyiron_atomistics/pull/433))
- dependencies: [#449](https://github.com/pyiron/pyiron_atomistics/pull/449), [#432](https://github.com/pyiron/pyiron_atomistics/pull/432), [#437](https://github.com/pyiron/pyiron_atomistics/pull/437), [#430](https://github.com/pyiron/pyiron_atomistics/pull/430), [#456](https://github.com/pyiron/pyiron_atomistics/pull/456)
- GitHub infrastructure: [#438](https://github.com/pyiron/pyiron_atomistics/pull/438)

# pyiron_atomistics-0.2.30
- Parse irreductible kpoints properly ([#423](https://github.com/pyiron/pyiron_atomistics/pull/423))
- Cluster (atom) positions ([#419](https://github.com/pyiron/pyiron_atomistics/pull/419)) 
- Make Methfessel Paxton default in SPHInX ([#416](https://github.com/pyiron/pyiron_atomistics/pull/416))
- Pyscal solid liquid ([#414](https://github.com/pyiron/pyiron_atomistics/pull/414))
- dependency updates: [#410](https://github.com/pyiron/pyiron_atomistics/pull/410), [#411](https://github.com/pyiron/pyiron_atomistics/pull/411), [#431](https://github.com/pyiron/pyiron_atomistics/pull/431)
- Infrastructure: [#429](https://github.com/pyiron/pyiron_atomistics/pull/429)

The update to the new `pyiron_base` version > 0.4 ([#431](https://github.com/pyiron/pyiron_atomistics/pull/431)) fixes some bugs in the writing of numerical data to our storage backend (hdf), see [pyiron_base release notes](https://github.com/pyiron/pyiron_base/releases/tag/pyiron_base-0.4.0).

# pyiron_atomistics-0.2.29
- VASP doesn't save stresses to HDF when run on ISIF=2 (default) ([#354](https://github.com/pyiron/pyiron_atomistics/pull/354)) 
- Allow hcp 4-axes indices as well ([#390](https://github.com/pyiron/pyiron_atomistics/pull/390))
- [minor] clean up find_mic ([#403](https://github.com/pyiron/pyiron_atomistics/pull/403))
- Fix scaled test ([#389](https://github.com/pyiron/pyiron_atomistics/pull/389))
- dependency updates: [#398](https://github.com/pyiron/pyiron_atomistics/pull/398), [#406](https://github.com/pyiron/pyiron_atomistics/pull/406), [#404](https://github.com/pyiron/pyiron_atomistics/pull/404), [#408](https://github.com/pyiron/pyiron_atomistics/pull/408)

# pyiron_atomistics-0.2.28
- Fix structure check in `restart()` ([#392](https://github.com/pyiron/pyiron_atomistics/pull/392))

# pyiron_atomistics-0.2.27
- Adapt to removal of load_object  ([#386](https://github.com/pyiron/pyiron_atomistics/pull/386))
- Don't conflict with HasHDF in StructureStorage  ([#376](https://github.com/pyiron/pyiron_atomistics/pull/376))

# pyiron_atomistics-0.2.26
- pep8 for sphinx ([#373](https://github.com/pyiron/pyiron_atomistics/pull/373))
- Bind creator at import ([#347](https://github.com/pyiron/pyiron_atomistics/pull/347))
- Lammps style full - fix numpy warnings ([#367](https://github.com/pyiron/pyiron_atomistics/pull/367))
- Water interactive fix  ([#366](https://github.com/pyiron/pyiron_atomistics/pull/366))
- Update base.py ([#346](https://github.com/pyiron/pyiron_atomistics/pull/346))
- Support full style without bonds ([#365](https://github.com/pyiron/pyiron_atomistics/pull/365))
- Strain ([#364](https://github.com/pyiron/pyiron_atomistics/pull/364))
- Use StructureStorage in StructureContainer ([#344](https://github.com/pyiron/pyiron_atomistics/pull/344))
- Explicitly specify water potential ([#353](https://github.com/pyiron/pyiron_atomistics/pull/353))
- Add automatic labeling for integration tests ([#361](https://github.com/pyiron/pyiron_atomistics/pull/361))
- Consistent atoms order ([#338](https://github.com/pyiron/pyiron_atomistics/pull/338))
- Add Wrapper for Atomsk Structure Creation ([#260](https://github.com/pyiron/pyiron_atomistics/pull/260))
- replace strain by master job name ([#355](https://github.com/pyiron/pyiron_atomistics/pull/355))
- change pointer to copy ([#350](https://github.com/pyiron/pyiron_atomistics/pull/350))
- Fixes for pyiron table ([#329](https://github.com/pyiron/pyiron_atomistics/pull/329))
- Don't specify force tolerance if only optimizing cell ([#339](https://github.com/pyiron/pyiron_atomistics/pull/339))

# pyiron_atomistics-0.2.25
- Strain ([#328](https://github.com/pyiron/pyiron_atomistics/pull/328))
- Properly implement `HasStructure` for `StructureContainer`  ([#331](https://github.com/pyiron/pyiron_atomistics/pull/331))
- Move StructureStorage from pyiron_contrib  ([#327](https://github.com/pyiron/pyiron_atomistics/pull/327))
- Use correct role name for sphinx  ([#330](https://github.com/pyiron/pyiron_atomistics/pull/330))

# pyiron_atomistics-0.2.24
Update to pyiron_base-0.3.0

# pyiron_atomistics-0.2.23
- Hotfix: Handle empty indices  ([#284](https://github.com/pyiron/pyiron_atomistics/pull/284))
- Interactive units ([#295](https://github.com/pyiron/pyiron_atomistics/pull/295))

# pyiron_atomistics-0.2.22
- Outcar bands ([#299](https://github.com/pyiron/pyiron_atomistics/pull/299))
- Add error message if `Murnaghan.plot` is called on unfinished jobs  ([#291](https://github.com/pyiron/pyiron_atomistics/pull/291))
- Make Atoms() faster via caches  ([#224](https://github.com/pyiron/pyiron_atomistics/pull/224))
- Create new Atoms in `get_structure` if size changes  ([#241](https://github.com/pyiron/pyiron_atomistics/pull/241)) 
- add mode in `get_neighborhood` ([#285](https://github.com/pyiron/pyiron_atomistics/pull/285))
- Equivalent points ([#280](https://github.com/pyiron/pyiron_atomistics/pull/280))

# pyiron_atomistics-0.2.21
- Neighbor analysis for the entire trajectory ([#251](https://github.com/pyiron/pyiron_atomistics/pull/251))
- Get neighbors ([#239](https://github.com/pyiron/pyiron_atomistics/pull/239))
- Save all VCSGC parameters in generic input ([#262](https://github.com/pyiron/pyiron_atomistics/pull/262))
- Only force skew when tensions are applied([#263](https://github.com/pyiron/pyiron_atomistics/pull/263))

# pyiron_atomistics-0.2.20
Bugfix:
- Estimate width ([#272](https://github.com/pyiron/pyiron_atomistics/pull/272))

# pyiron_atomistics-0.2.19
* Extending the units class ([#271](https://github.com/pyiron/pyiron_atomistics/pull/271))
* interactive_prepare  ([#235](https://github.com/pyiron/pyiron_atomistics/pull/235))
* Centrosymmetry ([#261](https://github.com/pyiron/pyiron_atomistics/pull/261))
* Add HasStructure to Trajectory  ([#270](https://github.com/pyiron/pyiron_atomistics/pull/270))
* scf_residue unit corrected ([#266](https://github.com/pyiron/pyiron_atomistics/pull/266))

# pyiron_atomistics-0.2.18
- Analysis of Atomic structure updated
- Update to vasp job: raise error for non-zero pressure

# pyiron_atomistics-0.2.17
- Enable `use_pressure = False` ([#237](https://github.com/pyiron/pyiron_atomistics/pull/237))

# pyiron_atomistics-0.2.16
- Get interstitials: New feature to find interstitial sites ([#219](https://github.com/pyiron/pyiron_atomistics/pull/219))
- Phonopy options: Expose additional argument `number_of_snapshots` ([#213](https://github.com/pyiron/pyiron_atomistics/pull/213))
- VASP - change NSW if interactive ([#220](https://github.com/pyiron/pyiron_atomistics/pull/220))
- Use new copy hook ([#201](https://github.com/pyiron/pyiron_atomistics/pull/201))

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

# pyiron_atomistics-0.2.11


# pyiron_atomistics-0.2.10


# pyiron_atomistics-0.2.7
Last release didn't have consistent requirements (aimsgb missing).

# pyiron_atomistics-0.2.2


# pyiron_atomistics-0.2.1


