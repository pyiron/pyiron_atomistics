# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import codecs
import pickle

import numpy as np
import posixpath
import scipy.constants
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.units import VaspToTHz
from phonopy.file_IO import write_FORCE_CONSTANTS

from pyiron_atomistics.atomistics.structure.atoms import Atoms
from pyiron_atomistics.atomistics.master.parallel import AtomisticParallelMaster
from pyiron_atomistics.atomistics.structure.phonopy import publication as phonopy_publication
from pyiron_base import JobGenerator, Settings

__author__ = "Jan Janssen, Yury Lysogorskiy"
__copyright__ = (
    "Copyright 2021, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "1.0"
__maintainer__ = "Jan Janssen"
__email__ = "janssen@mpie.de"
__status__ = "development"
__date__ = "Sep 1, 2017"

s = Settings()


class Thermal:
    """
    Holds thermal properties that are the results of the phonopy calculation.

    Args:
        temperatures (ndarray): temperatures at which the other quantities are evaluated, units of Kelvin
        free_energies (ndarray): free energies in the quasi-harmonic approximation from phonon contributions, units of
                                 electron volts
        entropy (ndarray): vibrational entropy calculated from the above free energy
        cv (ndarray): heat capacity at constant volume, units of kJ/K/mol
    """

    def __init__(self, temperatures, free_energies, entropy, cv):

        self.temperatures = temperatures
        self.free_energies = free_energies
        self.entropy = entropy
        self.cv = cv


def phonopy_to_atoms(ph_atoms):
    """
    Convert Phonopy Atoms to ASE-like Atoms
    Args:
        ph_atoms: Phonopy Atoms object

    Returns: ASE-like Atoms object

    """
    return Atoms(
        symbols=list(ph_atoms.get_chemical_symbols()),
        positions=list(ph_atoms.get_positions()),
        cell=list(ph_atoms.get_cell()), pbc=True
    )


def atoms_to_phonopy(atom):
    """
    Convert ASE-like Atoms to Phonopy Atoms
    Args:
        atom: ASE-like Atoms

    Returns:
        Phonopy Atoms

    """
    return PhonopyAtoms(
        symbols=list(atom.get_chemical_symbols()),
        scaled_positions=list(atom.get_scaled_positions()),
        cell=list(atom.get_cell()),
    )


class PhonopyJobGenerator(JobGenerator):
    @property
    def parameter_list(self):
        """

        Returns:
            (list)
        """
        supercells = self._master.phonopy.get_supercells_with_displacements()
        return [
            ["{}_{}".format(self._master.ref_job.job_name, ind), self._restore_magmoms(phonopy_to_atoms(sc))]
            for ind, sc in enumerate(supercells)
        ]

    def _restore_magmoms(self, structure):
        """
        Args:
            structure (pyiron.atomistics.structure.atoms): input structure

        Returns:
            structure (pyiron_atomistics.atomistics.structure.atoms): output structure with magnetic moments
        """
        if any(self._master.structure.get_initial_magnetic_moments() != None):
            magmoms = self._master.structure.get_initial_magnetic_moments()
            magmoms = np.tile(magmoms, np.prod(np.diagonal(self._master._phonopy_supercell_matrix())).astype(int))
            structure.set_initial_magnetic_moments(magmoms)
        return structure

    @staticmethod
    def job_name(parameter):
        return parameter[0]

    def modify_job(self, job, parameter):
        job.structure = parameter[1]
        return job


class PhonopyJob(AtomisticParallelMaster):
    """
    Phonopy wrapper for the calculation of free energy in the framework of quasi harmonic approximation.

    Example:

    >>> from pyiron_atomistics import Project
    >>> pr = Project('my_project')
    >>> lmp = pr.create_job('Lammps', 'lammps')
    >>> lmp.structure = pr.create_structure('Fe', 'bcc', 2.832)
    >>> phono = lmp.create_job('PhonopyJob', 'phonopy')
    >>> phono.run()

    Get output via `get_thermal_properties()`.

    Note:

    - This class does not consider the thermal expansion. For this, use `QuasiHarmonicJob` (find more in its
        docstring)
    - Depending on the value given in `job.input['interaction_range']`, this class automatically changes the number of
        atoms. The input parameters of the reference job might have to be set appropriately (e.g. use `k_mesh_density`
        for DFT instead of setting k-points directly).
    - The structure used in the reference job should be a relaxed structure.
    - Theory behind it: https://en.wikipedia.org/wiki/Quasi-harmonic_approximation
    """

    def __init__(self, project, job_name):
        super(PhonopyJob, self).__init__(project, job_name)
        self.__name__ = "PhonopyJob"
        self.__version__ = "0.0.1"
        self.input["interaction_range"] = (10.0, "Minimal size of supercell, Ang")
        self.input["factor"] = (
            VaspToTHz,
            "Frequency unit conversion factor (default for VASP)",
        )
        self.input["displacement"] = (0.01, "atoms displacement, Ang")
        self.input["dos_mesh"] = (20, "mesh size for DOS calculation")
        self.input["primitive_matrix"] = None

        self.phonopy = None
        self._job_generator = PhonopyJobGenerator(self)
        self._disable_phonopy_pickle = False
        s.publication_add(phonopy_publication())

    @property
    def phonopy_pickling_disabled(self):
        return self._disable_phonopy_pickle

    @phonopy_pickling_disabled.setter
    def phonopy_pickling_disabled(self, disable):
        self._disable_phonopy_pickle = disable

    @property
    def _phonopy_unit_cell(self):
        if self.structure is not None:
            return atoms_to_phonopy(self.structure)
        else:
            return None

    def _enable_phonopy(self):
        if self.phonopy is None:
            if self.structure is not None:
                self.phonopy = Phonopy(
                    unitcell=self._phonopy_unit_cell,
                    supercell_matrix=self._phonopy_supercell_matrix(),
                    primitive_matrix=self.input["primitive_matrix"],
                    factor=self.input["factor"],
                )
                self.phonopy.generate_displacements(distance=self.input["displacement"])
                self.to_hdf()
            else:
                raise ValueError("No reference job/ No reference structure found.")

    def list_structures(self):
        if self.structure is not None:
            self._enable_phonopy()
            return [struct for _, struct in self._job_generator.parameter_list]
        else:
            return []

    def _phonopy_supercell_matrix(self):
        if self.structure is not None:
            supercell_range = np.ceil(
                self.input["interaction_range"]
                / np.array(
                    [np.linalg.norm(vec) for vec in self._phonopy_unit_cell.get_cell()]
                )
            )
            return np.eye(3) * supercell_range
        else:
            return np.eye(3)

    def run_static(self):
        # Initialise the phonopy object before starting the first calculation.
        self._enable_phonopy()
        super(PhonopyJob, self).run_static()

    def run_if_interactive(self):
        self._enable_phonopy()
        super(PhonopyJob, self).run_if_interactive()

    def to_hdf(self, hdf=None, group_name=None):
        """
        Store the PhonopyJob in an HDF5 file

        Args:
            hdf (ProjectHDFio): HDF5 group object - optional
            group_name (str): HDF5 subgroup name - optional
        """
        super(PhonopyJob, self).to_hdf(hdf=hdf, group_name=group_name)
        if self.phonopy is not None and not self._disable_phonopy_pickle:
            with self.project_hdf5.open("output") as hdf5_output:
                hdf5_output["phonopy_pickeled"] = codecs.encode(
                    pickle.dumps(self.phonopy), "base64"
                ).decode()

    def from_hdf(self, hdf=None, group_name=None):
        """
        Restore the PhonopyJob from an HDF5 file

        Args:
            hdf (ProjectHDFio): HDF5 group object - optional
            group_name (str): HDF5 subgroup name - optional
        """
        super(PhonopyJob, self).from_hdf(hdf=hdf, group_name=group_name)
        with self.project_hdf5.open("output") as hdf5_output:
            if "phonopy_pickeled" in hdf5_output.list_nodes():
                self.phonopy = pickle.loads(
                    codecs.decode(hdf5_output["phonopy_pickeled"].encode(), "base64")
                )
                if "dos_total" in hdf5_output.list_nodes():
                    self._dos_total = hdf5_output["dos_total"]
                if "dos_energies" in hdf5_output.list_nodes():
                    self._dos_energies = hdf5_output["dos_energies"]

    def collect_output(self):
        """

        Returns:

        """
        if self.ref_job.server.run_mode.interactive:
            forces_lst = self.project_hdf5.inspect(self.child_ids[0])[
                "output/generic/forces"
            ]
        else:
            pr_job = self.project_hdf5.project.open(self.job_name + "_hdf5")
            forces_lst = [
                pr_job.inspect(job_name)["output/generic/forces"][-1]
                for job_name in self._get_jobs_sorted()
            ]
        self.phonopy.set_forces(forces_lst)
        self.phonopy.produce_force_constants()
        self.phonopy.run_mesh(mesh=[self.input["dos_mesh"]] * 3)
        mesh_dict = self.phonopy.get_mesh_dict()
        self.phonopy.run_total_dos()
        dos_dict = self.phonopy.get_total_dos_dict()

        self.to_hdf()

        with self.project_hdf5.open("output") as hdf5_out:
            hdf5_out["dos_total"] = dos_dict['total_dos']
            hdf5_out["dos_energies"] = dos_dict['frequency_points']
            hdf5_out["qpoints"] = mesh_dict['qpoints']
            hdf5_out["supercell_matrix"] = self._phonopy_supercell_matrix()
            hdf5_out["displacement_dataset"] = self.phonopy.get_displacement_dataset()
            hdf5_out[
                "dynamical_matrix"
            ] = self.phonopy.dynamical_matrix.get_dynamical_matrix()
            hdf5_out["force_constants"] = self.phonopy.force_constants

    def write_phonopy_force_constants(self, file_name="FORCE_CONSTANTS", cwd=None):
        """

        Args:
            file_name:
            cwd:

        Returns:

        """
        if cwd is not None:
            file_name = posixpath.join(cwd, file_name)
        write_FORCE_CONSTANTS(
            force_constants=self.phonopy.force_constants, filename=file_name
        )

    def get_hesse_matrix(self):
        """

        Returns:

        """
        unit_conversion = (
            scipy.constants.physical_constants["Hartree energy in eV"][0]
            / scipy.constants.physical_constants["Bohr radius"][0] ** 2
            * scipy.constants.angstrom ** 2
        )
        force_shape = np.shape(self.phonopy.force_constants)
        force_reshape = force_shape[0] * force_shape[2]
        return (
            np.transpose(self.phonopy.force_constants, (0, 2, 1, 3)).reshape(
                (force_reshape, force_reshape)
            )
            / unit_conversion
        )

    def get_thermal_properties(self, t_min=1, t_max=1500, t_step=50, temperatures=None):
        """
        Returns thermal properties at constant volume in the given temperature range.  Can only be called after job
        successfully ran.

        Args:
            t_min (float): minimum sample temperature
            t_max (float): maximum sample temperature
            t_step (int):  tempeature sample interval
            temperatures (array_like, float):  custom array of temperature samples, if given t_min, t_max, t_step are
                                               ignored.

        Returns:
            :class:`Thermal`: thermal properties as returned by Phonopy
        """
        self.phonopy.run_thermal_properties(
            t_step=t_step, t_max=t_max, t_min=t_min, temperatures=temperatures
        )
        tp_dict = self.phonopy.get_thermal_properties_dict()
        kJ_mol_to_eV = 1000/scipy.constants.Avogadro/scipy.constants.electron_volt
        return Thermal(tp_dict['temperatures'],
                       tp_dict['free_energy'] * kJ_mol_to_eV,
                       tp_dict['entropy'],
                       tp_dict['heat_capacity'])

    @property
    def dos_total(self):
        """

        Returns:

        """
        return self["output/dos_total"]

    @property
    def dos_energies(self):
        """

        Returns:

        """
        return self["output/dos_energies"]

    @property
    def dynamical_matrix(self):
        """

        Returns:

        """
        return np.real_if_close(
            self.phonopy.get_dynamical_matrix().get_dynamical_matrix()
        )

    def dynamical_matrix_at_q(self, q):
        """

        Args:
            q:

        Returns:

        """
        return np.real_if_close(self.phonopy.get_dynamical_matrix_at_q(q))

    def plot_dos(self, ax=None, *args, **qwargs):
        """

        Args:
            *args:
            ax:
            **qwargs:

        Returns:

        """
        try:
            import pylab as plt
        except ImportError:
            import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        ax.plot(self["output/dos_energies"], self["output/dos_total"], *args, **qwargs)
        ax.set_xlabel("Frequency [THz]")
        ax.set_ylabel("DOS")
        ax.set_title("Phonon DOS vs Energy")
        return ax

    def validate_ready_to_run(self):
        if self.ref_job._generic_input["calc_mode"] != "static":
            raise ValueError("Phonopy reference jobs should be static calculations, but got {}".format(
                self.ref_job._generic_input["calc_mode"]
            ))
        super().validate_ready_to_run()
