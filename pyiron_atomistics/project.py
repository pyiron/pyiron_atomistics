# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from __future__ import print_function

import os
import posixpath
from shutil import copyfile

# import warnings
from string import punctuation

from pyiron_base import (
    Creator as CreatorCore,
)
from pyiron_base import (
    JobType,
    JobTypeChoice,
    LocalMaintenance,
    Maintenance,
    ProjectHDFio,
)
from pyiron_base import (
    Project as ProjectCore,
)
from pyiron_snippets.deprecate import deprecate
from pyiron_snippets.logger import logger

try:
    from pyiron_base import ProjectGUI
except (ImportError, TypeError, AttributeError):
    pass
import numpy as np

import pyiron_atomistics.atomistics.structure.pyironase as ase
from pyiron_atomistics.atomistics.generic.object_type import (
    ObjectType,
    ObjectTypeChoice,
)
from pyiron_atomistics.atomistics.master.parallel import pipe
from pyiron_atomistics.atomistics.structure.atoms import Atoms
from pyiron_atomistics.atomistics.structure.factory import StructureFactory
from pyiron_atomistics.atomistics.structure.periodic_table import PeriodicTable
from pyiron_atomistics.lammps.potential import LammpsPotentialFile
from pyiron_atomistics.vasp.base import _vasp_generic_energy_free_affected
from pyiron_atomistics.vasp.potential import VaspPotential

__author__ = "Joerg Neugebauer, Jan Janssen"
__copyright__ = (
    "Copyright 2021, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "1.0"
__maintainer__ = "Jan Janssen"
__email__ = "janssen@mpie.de"
__status__ = "production"
__date__ = "Sep 1, 2017"

if not (isinstance(ase.__file__, str)):
    raise AssertionError()


def _vasp_energy_kin_affected(job):
    e_kin = job["output/generic/dft/scf_energy_kin"]
    return e_kin is not None and not isinstance(e_kin, np.ndarray)


class AtomisticsLocalMaintenance(LocalMaintenance):
    def vasp_energy_pot_as_free_energy(
        self, recursive: bool = True, progress: bool = True, **kwargs
    ):
        """
        Ensure generic potential energy is the electronic free energy.

        This ensures that the energy is consistent with the forces and stresses.
        In version 0.1.0 of the Vasp job (pyiron_atomistics<=0.3.10) a combination of bugs in Vasp and pyiron caused the
        potential energy reported to be the internal energy of electronic system extrapolated to zero smearing instead.

        Args:
            recursive (bool): search subprojects [True/False] - True by default
            progress (bool): if True (default), add an interactive progress bar to the iteration
            **kwargs (dict): Optional arguments for filtering with keys matching the project database column name
                            (eg. status="finished"). Asterisk can be used to denote a wildcard, for zero or more
                            instances of any character
        """
        kwargs["hamilton"] = "Vasp"
        kwargs["status"] = "finished"

        found_energy_kin_job = False

        for job in self._project.iter_jobs(
            recursive=recursive, progress=progress, convert_to_object=False, **kwargs
        ):
            if _vasp_generic_energy_free_affected(job):
                job.project_hdf5["output/generic/energy_pot"] = np.array(
                    [
                        e[-1]
                        for e in job.project_hdf5["output/generic/dft/scf_energy_free"]
                    ]
                )
                found_energy_kin_job |= _vasp_energy_kin_affected(job)

        if found_energy_kin_job:
            logger.warn(
                "Found at least one Vasp MD job with wrong kinetic energy.  Apply vasp_correct_energy_kin to fix!"
            )

    def vasp_correct_energy_kin(
        self, recursive: bool = True, progress: bool = True, **kwargs
    ):
        """
        Ensure kinetic and potential energy are correctly parsed for AIMD Vasp jobs.

        Version 0.1.0 of the Vasp job (pyiron_atomistics<=0.3.10) incorrectly parsed the kinetic energy during MD runs,
        such that it only reported the kinetic energy of the final ionic step and subtracted it from the electronic
        energy instead of adding it.

        Args:
            recursive (bool): search subprojects [True/False] - True by default
            progress (bool): if True (default), add an interactive progress bar to the iteration
            **kwargs (dict): Optional arguments for filtering with keys matching the project database column name
                            (eg. status="finished"). Asterisk can be used to denote a wildcard, for zero or more
                            instances of any character
        """
        kwargs["hamilton"] = "Vasp"
        kwargs["status"] = "finished"
        for job in self._project.iter_jobs(
            recursive=recursive, progress=progress, convert_to_object=False, **kwargs
        ):
            # only Vasp jobs of version 0.1.0 were affected
            if job["HDF_VERSION"] != "0.1.0":
                continue
            if not _vasp_energy_kin_affected(job):
                continue

            job.decompress()
            job = job.to_object()
            job.status.collect = True
            job.run()


class AtomisticsMaintenance(Maintenance):
    def __init__(self, project):
        """
        Args:
            (project): pyiron project to do maintenance on
        """
        super().__init__(project=project)
        self._local = AtomisticsLocalMaintenance(project)


class Project(ProjectCore):
    """
    Welcome to pyiron! The project is the central class in pyiron, all other objects can be
    created from the project object.

    Your first steps in pyiron:

    >>> pr = Project("EXAMPLE")
    >>> job = pr.create.job.Lammps(job_name="lmp_example")

    Replace `Lammps` by the job type of your choice - you can look at the list of all available
    jobs in the list of auto-complete in `pr.create.job`. After you create a job, look up the
    DocString of your job by `job?` to find out what are the next steps!

    Args:
        path (GenericPath, str): path of the project defined by GenericPath, absolute or relative (with respect to
                                     current working directory) path
        user (str): current pyiron user
        sql_query (str): SQL query to only select a subset of the existing jobs within the current project
        default_working_directory (bool): Access default working directory, for ScriptJobs this equals the project
                                     directory of the ScriptJob for regular projects it falls back to the current
                                     directory.

    Attributes:

        .. attribute:: root_path

            the pyiron user directory, defined in the .pyiron configuration

        .. attribute:: project_path

            the relative path of the current project / folder starting from the root path
            of the pyiron user directory

        .. attribute:: path

            the absolute path of the current project / folder

        .. attribute:: base_name

            the name of the current project / folder

        .. attribute:: history

            previously opened projects / folders

        .. attribute:: parent_group

            parent project - one level above the current project

        .. attribute:: user

            current unix/linux/windows user who is running pyiron

        .. attribute:: sql_query

            an SQL query to limit the jobs within the project to a subset which matches the SQL query.

        .. attribute:: db

            connection to the SQL database

        .. attribute:: job_type

            Job Type object with all the available job types: ['StructureContainer’, ‘StructurePipeline’, ‘AtomisticExampleJob’,
                                             ‘ExampleJob’, ‘Lammps’, ‘KMC’, ‘Sphinx’, ‘Vasp’, ‘GenericMaster’,
                                             ‘ParallelMaster’, ‘KmcMaster’,
                                             ‘ThermoLambdaMaster’, ‘RandomSeedMaster’, ‘MeamFit’, ‘Murnaghan’,
                                             ‘MinimizeMurnaghan’, ‘ElasticMatrix’,
                                             ‘ConvergenceKpointParallel’, ’PhonopyMaster’,
                                             ‘DefectFormationEnergy’, ‘LammpsASE’, ‘PipelineMaster’,
                                             ’TransformationPath’, ‘ThermoIntEamQh’, ‘ThermoIntDftEam’, ‘ScriptJob’,
                                             ‘ListMaster']
    """

    def __init__(
        self, path="", user=None, sql_query=None, default_working_directory=False
    ):
        super(Project, self).__init__(
            path=path,
            user=user,
            sql_query=sql_query,
            default_working_directory=default_working_directory,
        )
        self.job_type = JobTypeChoice()
        self.object_type = ObjectTypeChoice()
        self._creator = Creator(self)
        # TODO: instead of re-initialzing, auto-update pyiron_base creator with factories, like we update job class
        #  creation

    @property
    def maintenance(self):
        if self._maintenance is None:
            self._maintenance = AtomisticsMaintenance(self)
        return self._maintenance

    def create_job(self, job_type, job_name, delete_existing_job=False):
        """
        Create one of the following jobs:
        - 'StructureContainer’:
        - ‘StructurePipeline’:
        - ‘AtomisticExampleJob’: example job just generating random number
        - ‘ExampleJob’: example job just generating random number
        - ‘Lammps’:
        - ‘KMC’:
        - ‘Sphinx’:
        - ‘Vasp’:
        - ‘GenericMaster’:
        - ‘ParallelMaster’: series of jobs run in parallel
        - ‘KmcMaster’:
        - ‘ThermoLambdaMaster’:
        - ‘RandomSeedMaster’:
        - ‘MeamFit’:
        - ‘Murnaghan’:
        - ‘MinimizeMurnaghan’:
        - ‘ElasticMatrix’:
        - ‘ConvergenceEncutParallel’:
        - ‘ConvergenceKpointParallel’:
        - ’PhonopyMaster’:
        - ‘DefectFormationEnergy’:
        - ‘LammpsASE’:
        - ‘PipelineMaster’:
        - ’TransformationPath’:
        - ‘ThermoIntEamQh’:
        - ‘ThermoIntDftEam’:
        - ‘ScriptJob’: Python script or jupyter notebook job container
        - ‘ListMaster': list of jobs

        Args:
            job_type (str): job type can be ['StructureContainer’, ‘StructurePipeline’, ‘AtomisticExampleJob’,
                                             ‘ExampleJob’, ‘Lammps’, ‘KMC’, ‘Sphinx’, ‘Vasp’, ‘GenericMaster’,
                                             ‘ParallelMaster’, ‘KmcMaster’,
                                             ‘ThermoLambdaMaster’, ‘RandomSeedMaster’, ‘MeamFit’, ‘Murnaghan’,
                                             ‘MinimizeMurnaghan’, ‘ElasticMatrix’,
                                             ‘ConvergenceEncutParallel’, ‘ConvergenceKpointParallel’, ’PhonopyMaster’,
                                             ‘DefectFormationEnergy’, ‘LammpsASE’, ‘PipelineMaster’,
                                             ’TransformationPath’, ‘ThermoIntEamQh’, ‘ThermoIntDftEam’, ‘ScriptJob’,
                                             ‘ListMaster']
            job_name (str): name of the job

        Returns:
            GenericJob: job object depending on the job_type selected
        """
        job = JobType(
            job_type,
            project=ProjectHDFio(project=self.copy(), file_name=job_name),
            job_name=job_name,
            job_class_dict=self.job_type.job_class_dict,
            delete_existing_job=delete_existing_job,
        )
        if self.user is not None:
            job.user = self.user
        return job

    @staticmethod
    def create_object(object_type):
        """

        Args:
            object_type:

        Returns:

        """
        obj = ObjectType(object_type, project=None, job_name=None)
        return obj

    def load_from_jobpath(self, job_id=None, db_entry=None, convert_to_object=True):
        """
        Internal function to load an existing job either based on the job ID or based on the database entry dictionary.

        Args:
            job_id (int): Job ID - optional, but either the job_id or the db_entry is required.
            db_entry (dict): database entry dictionary - optional, but either the job_id or the db_entry is required.
            convert_to_object (bool): convert the object to an pyiron object or only access the HDF5 file - default=True
                                      accessing only the HDF5 file is about an order of magnitude faster, but only
                                      provides limited functionality. Compare the GenericJob object to JobCore object.

        Returns:
            GenericJob, JobCore: Either the full GenericJob object or just a reduced JobCore object
        """
        job = super(Project, self).load_from_jobpath(
            job_id=job_id, db_entry=db_entry, convert_to_object=convert_to_object
        )
        job.project_hdf5._project = self.__class__(path=job.project_hdf5.file_path)
        return job

    def load_from_jobpath_string(self, job_path, convert_to_object=True):
        """
        Internal function to load an existing job either based on the job ID or based on the database entry dictionary.

        Args:
            job_path (str): string to reload the job from an HDF5 file - '/root_path/project_path/filename.h5/h5_path'
            convert_to_object (bool): convert the object to an pyiron object or only access the HDF5 file - default=True
                                      accessing only the HDF5 file is about an order of magnitude faster, but only
                                      provides limited functionality. Compare the GenericJob object to JobCore object.

        Returns:
            GenericJob, JobCore: Either the full GenericJob object or just a reduced JobCore object
        """
        job = super(Project, self).load_from_jobpath_string(
            job_path=job_path, convert_to_object=convert_to_object
        )
        job.project_hdf5._project = Project(path=job.project_hdf5.file_path)
        return job

    def import_single_calculation(
        self,
        project_to_import_from,
        rel_path=None,
        job_type="Vasp",
        copy_raw_files=False,
    ):
        """
        A method to import a single calculation jobs into pyiron. Currently, it suppor
        ts VASP and KMC calculations.
        Args:
            rel_path:
            project_to_import_from:
            job_type (str): Type of the calculation which is going to be imported.
            copy_raw_files (bool): True if raw files are to be imported.
        """
        if job_type not in ["Vasp", "KMC"]:
            raise ValueError("The job_type is not supported.")
        job_name = project_to_import_from.split("/")[-1]
        if job_name[0].isdigit():
            pyiron_job_name = "job_" + job_name
        else:
            pyiron_job_name = job_name
        for ch in list(punctuation):
            if ch in pyiron_job_name:
                pyiron_job_name = pyiron_job_name.replace(ch, "_")
        print(job_name, pyiron_job_name)
        if rel_path:
            rel_path_lst = [pe for pe in rel_path.split("/")[:-1] if pe != ".."]
            pr_import = self.open("/".join(rel_path_lst))
        else:
            pr_import = self.open("/".join(project_to_import_from.split("/")[:-1]))
        if self.db.get_items_dict(
            {"job": pyiron_job_name, "project": pr_import.project_path}
        ):
            print("The job exists already - skipped!")
        else:
            ham = pr_import.create_job(job_type=job_type, job_name=pyiron_job_name)
            ham._job_id = self.db.add_item_dict(ham.db_entry())
            ham.refresh_job_status()
            print("job was stored with the job ID ", str(ham._job_id))
            if not os.path.abspath(project_to_import_from):
                project_to_import_from = os.path.join(self.path, project_to_import_from)
            try:
                ham.from_directory(project_to_import_from.replace("\\", "/"))
            except:
                ham.status.aborted = True
            else:
                ham._import_directory = None
                del ham["import_directory"]
                if copy_raw_files:
                    os.makedirs(ham.working_directory, exist_ok=True)
                    for f in os.listdir(project_to_import_from):
                        src = os.path.join(project_to_import_from, f)
                        if os.path.isfile(src):
                            copyfile(
                                src=src, dst=os.path.join(ham.working_directory, f)
                            )
                    ham.compress()

    def import_from_path(self, path, recursive=True, copy_raw_files=False):
        """
        A method to import jobs into pyiron. Currently, it supports VASP and
        KMC calculations.

        Args:
            path (str): The path of the directory to import
            recursive (bool): True if sub-directories to be imported.
            copy_raw_files (bool): True if the raw files are to be copied.

        """
        if os.path.abspath(path):
            search_path = posixpath.normpath(path.replace("//", "/"))
        else:
            search_path = posixpath.normpath(
                posixpath.join(self.path, path.replace("//", "/"))
            )
        if recursive:
            for x in os.walk(search_path):
                self._calculation_validation(
                    x[0],
                    x[2],
                    rel_path=posixpath.relpath(x[0], search_path),
                    copy_raw_files=copy_raw_files,
                )
        else:
            abs_path = "/".join(search_path.replace("\\", "/").split("/")[:-1])
            rel_path = posixpath.relpath(abs_path, self.path)
            self._calculation_validation(
                search_path,
                os.listdir(search_path),
                rel_path=rel_path,
                copy_raw_files=copy_raw_files,
            )

    def get_structure(self, job_specifier, iteration_step=-1, wrap_atoms=True):
        """
        Gets the structure from a given iteration step of the simulation (MD/ionic relaxation). For static calculations
        there is only one ionic iteration step
        Args:
            job_specifier (str, int): name of the job or job ID
            iteration_step (int): Step for which the structure is requested
            wrap_atoms (bool): True if the atoms are to be wrapped back into the unit cell

        Returns:
            atomistics.structure.atoms.Atoms object
        """
        job = self.inspect(job_specifier)
        snapshot = Atoms().from_hdf(job["input"], "structure")
        if "output" in job.project_hdf5.list_groups() and iteration_step != 0:
            snapshot.cell = job.get("output/generic/cells")[iteration_step]
            snapshot.positions = job.get("output/generic/positions")[iteration_step]
            if "indices" in job.get("output/generic").list_nodes():
                snapshot.set_array(
                    "indices", job.get("output/generic/indices")[iteration_step]
                )
            if (
                "dft" in job["output/generic"].list_groups()
                and "atom_spins" in job["output/generic/dft"].list_nodes()
            ):
                snapshot.set_initial_magnetic_moments(
                    job.get("output/generic/dft/atom_spins")[iteration_step]
                )
        if wrap_atoms:
            return snapshot.center_coordinates_in_unit_cell()
        else:
            return snapshot

    def _calculation_validation(
        self, path, files_available, rel_path=None, copy_raw_files=False
    ):
        """

        Args:
            path:
            files_available:
            rel_path:
            copy_raw_files (bool):
        """
        if (
            "OUTCAR" in files_available
            or "vasprun.xml" in files_available
            or "OUTCAR.gz" in files_available
            or "vasprun.xml.bz2" in files_available
            or "vasprun.xml.gz" in files_available
        ):
            self.import_single_calculation(
                path, rel_path=rel_path, job_type="Vasp", copy_raw_files=copy_raw_files
            )
        if (
            "incontrol.dat" in files_available
            and "lattice.out" in files_available
            and "lattice.inp" in files_available
        ):
            self.import_single_calculation(
                path, rel_path=rel_path, job_type="KMC", copy_raw_files=copy_raw_files
            )

    @staticmethod
    def inspect_periodic_table():
        return PeriodicTable()

    @staticmethod
    def inspect_empirical_potentials():
        return LammpsPotentialFile()

    @staticmethod
    @deprecate("Use inspect_empirical_potentials instead!")
    def inspect_emperical_potentials():
        """
        For backwards compatibility, calls inspect_empirical_potentials()
        """
        return LammpsPotentialFile()

    @staticmethod
    def inspect_pseudo_potentials():
        return VaspPotential()

    # Graphical user interfaces
    def gui(self):
        """

        Returns:

        """
        ProjectGUI(self)

    def create_pipeline(self, job, step_lst, delete_existing_job=False):
        """
        Create a job pipeline

        Args:
            job (AtomisticGenericJob): Template for the calculation
            step_lst (list): List of functions which create calculations

        Returns:
            FlexibleMaster:
        """
        return pipe(
            project=self,
            job=job,
            step_lst=step_lst,
            delete_existing_job=delete_existing_job,
        )

    # Deprecated methods

    @deprecate("Use create.structure.bulk instead")
    def create_ase_bulk(
        self,
        name,
        crystalstructure=None,
        a=None,
        c=None,
        covera=None,
        u=None,
        orthorhombic=False,
        cubic=False,
    ):
        """
        Creating bulk systems using ASE bulk module. Crystal structure and lattice constant(s) will be guessed if not
        provided.

        name (str): Chemical symbol or symbols as in 'MgO' or 'NaCl'.
        crystalstructure (str): Must be one of sc, fcc, bcc, hcp, diamond, zincblende,
                                rocksalt, cesiumchloride, fluorite or wurtzite.
        a (float): Lattice constant.
        c (float): Lattice constant.
        c_over_a (float): c/a ratio used for hcp.  Default is ideal ratio: sqrt(8/3).
        u (float): Internal coordinate for Wurtzite structure.
        orthorhombic (bool): Construct orthorhombic unit cell instead of primitive cell which is the default.
        cubic (bool): Construct cubic unit cell if possible.

        Returns:

            pyiron.atomistics.structure.atoms.Atoms: Required bulk structure
        """
        return self.create.structure.ase.bulk(
            name=name,
            crystalstructure=crystalstructure,
            a=a,
            c=c,
            covera=covera,
            u=u,
            orthorhombic=orthorhombic,
            cubic=cubic,
        )

    @deprecate("Use create.structure.* methods instead")
    def create_structure(self, element, bravais_basis, lattice_constant):
        """
        Create a crystal structure using pyiron's native crystal structure generator

        Args:
            element (str): Element name
            bravais_basis (str): Basis type
            lattice_constant (float/list): Lattice constants

        Returns:
            pyiron.atomistics.structure.atoms.Atoms: The required crystal structure

        """
        # warnings.warn(
        #     "Project.create_structure is deprecated as of v0.3. Please use Project.create.structure.structure.",
        #     DeprecationWarning
        # )
        return self.create.structure.crystal(
            element=element,
            bravais_basis=bravais_basis,
            lattice_constant=lattice_constant,
        )

    @deprecate("Use create.structure.surface instead")
    def create_surface(
        self,
        element,
        surface_type,
        size=(1, 1, 1),
        vacuum=1.0,
        center=False,
        pbc=None,
        **kwargs,
    ):
        """
        Generate a surface based on the ase.build.surface module.

        Args:
            element (str): Element name
            surface_type (str): The string specifying the surface type generators available through ase (fcc111,
            hcp0001 etc.)
            size (tuple): Size of the surface
            vacuum (float): Length of vacuum layer added to the surface along the z direction
            center (bool): Tells if the surface layers have to be at the center or at one end along the z-direction
            pbc (list/numpy.ndarray): List of booleans specifying the periodic boundary conditions along all three
                                      directions. If None, it is set to [True, True, True]
            **kwargs: Additional, arguments you would normally pass to the structure generator like 'a', 'b',
            'orthogonal' etc.

        Returns:
            pyiron_atomistics.atomistics.structure.atoms.Atoms instance: Required surface

        """
        # warnings.warn(
        #     "Project.create_surface is deprecated as of v0.3. Please use Project.create.structure.surface.",
        #     DeprecationWarning
        # )
        return self.create.structure.surface(
            element=element,
            surface_type=surface_type,
            size=size,
            vacuum=vacuum,
            center=center,
            pbc=pbc,
            **kwargs,
        )

    @deprecate("Use create.structure.atoms instead")
    def create_atoms(
        self,
        symbols=None,
        positions=None,
        numbers=None,
        tags=None,
        momenta=None,
        masses=None,
        magmoms=None,
        charges=None,
        scaled_positions=None,
        cell=None,
        pbc=None,
        celldisp=None,
        constraint=None,
        calculator=None,
        info=None,
        indices=None,
        elements=None,
        dimension=None,
        species=None,
        **qwargs,
    ):
        """
        Creates a atomistics.structure.atoms.Atoms instance.

        Args:
            elements (list/numpy.ndarray): List of strings containing the elements or a list of
                                atomistics.structure.periodic_table.ChemicalElement instances
            numbers (list/numpy.ndarray): List of atomic numbers of elements
            symbols (list/numpy.ndarray): List of chemical symbols
            positions (list/numpy.ndarray): List of positions
            scaled_positions (list/numpy.ndarray): List of scaled positions (relative coordinates)
            pbc (boolean): Tells if periodic boundary conditions should be applied
            cell (list/numpy.ndarray): A 3x3 array representing the lattice vectors of the structure
            momenta (list/numpy.ndarray): List of momentum values
            tags (list/numpy.ndarray): A list of tags
            masses (list/numpy.ndarray): A list of masses
            magmoms (list/numpy.ndarray): A list of magnetic moments
            charges (list/numpy.ndarray): A list of point charges
            celldisp:
            constraint (list/numpy.ndarray): A list of constraints
            calculator: ASE calculator
            info (list/str): ASE compatibility
            indices (list/numpy.ndarray): The list of species indices
            dimension (int): Dimension of the structure
            species (list): List of species

        Returns:
            pyiron.atomistics.structure.atoms.Atoms: The required structure instance
        """
        # warnings.warn(
        #     "Project.create_atoms is deprecated as of v0.3. Please use Project.create.structure.atoms.",
        #     DeprecationWarning
        # )
        return self.create.structure.atoms(
            symbols=symbols,
            positions=positions,
            numbers=numbers,
            tags=tags,
            momenta=momenta,
            masses=masses,
            magmoms=magmoms,
            charges=charges,
            scaled_positions=scaled_positions,
            cell=cell,
            pbc=pbc,
            celldisp=celldisp,
            constraint=constraint,
            calculator=calculator,
            info=info,
            indices=indices,
            elements=elements,
            dimension=dimension,
            species=species,
            **qwargs,
        )

    @deprecate("Use create.structure.element instead")
    def create_element(
        self, parent_element, new_element_name=None, spin=None, potential_file=None
    ):
        """
        Args:
            parent_element (str, int): The parent element eq. "N", "O", "Mg" etc.
            new_element_name (str): The name of the new parent element (can be arbitrary)
            spin (float): Value of the magnetic moment (with sign)
            potential_file (str): Location of the new potential file if necessary

        Returns:
            atomistics.structure.periodic_table.ChemicalElement instance
        """
        # warnings.warn(
        #     "Project.create_element is deprecated as of v0.3. Please use Project.create.structure.element.",
        #     DeprecationWarning
        # )
        return self.create.structure.element(
            parent_element=parent_element,
            new_element_name=new_element_name,
            spin=spin,
            potential_file=potential_file,
        )


class Creator(CreatorCore):
    def __init__(self, project):
        super().__init__(project)
        self._structure = StructureFactory()

    @property
    def structure(self):
        return self._structure
