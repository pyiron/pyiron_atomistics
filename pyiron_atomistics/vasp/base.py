# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from __future__ import print_function

import os
import posixpath
import subprocess
import warnings
from typing import Optional

import numpy as np
from pyiron_base import GenericParameters, state
from pyiron_snippets.deprecate import deprecate
from pyiron_vasp.vasp.output import (
    VaspCollectError,
    get_final_structure_from_file,
    parse_vasp_output,
)
from pyiron_vasp.vasp.structure import vasp_sorter
from pyiron_vasp.vasp.vasprun import VasprunError

from pyiron_atomistics.atomistics.structure.atoms import Atoms, CrystalStructure
from pyiron_atomistics.dft.bader import Bader, get_valence_and_total_charge_density
from pyiron_atomistics.dft.job.generic import GenericDFTJob
from pyiron_atomistics.dft.waves.bandstructure import Bandstructure
from pyiron_atomistics.dft.waves.electronic import ElectronicStructure
from pyiron_atomistics.vasp.output import Output, output_dict_to_hdf
from pyiron_atomistics.vasp.potential import (
    Potcar,
    VaspPotential,
    VaspPotentialFile,
    VaspPotentialSetter,
    get_enmax_among_potentials,
    strip_xc_from_potential_name,
)
from pyiron_atomistics.vasp.structure import (
    get_poscar_content,
    read_atoms,
)
from pyiron_atomistics.vasp.vasprun import Vasprun as Vr
from pyiron_atomistics.vasp.volumetric_data import VaspVolumetricData

__author__ = "Sudarsan Surendralal, Felix Lochner"
__copyright__ = (
    "Copyright 2021, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "1.0"
__maintainer__ = "Sudarsan Surendralal"
__email__ = "surendralal@mpie.de"
__status__ = "production"
__date__ = "Sep 1, 2017"


def _vasp_generic_energy_free_affected(job):
    """
    Checks whether the value saved in output/generic/energy_pot matches the electronic free energy.
    """
    if job.project_hdf5.get("HDF_VERSION", "0.1.0") == "0.1.0":
        energy_free = np.array(
            [e[-1] for e in job.project_hdf5["output/generic/dft/scf_energy_free"]]
        )
        energy_pot = job.project_hdf5["output/generic/energy_pot"]
        return not np.allclose(energy_free, energy_pot)
    else:
        return False


class VaspBase(GenericDFTJob):
    """
    Class to setup and run and analyze VASP simulations which is a derivative of pyiron_atomistics.objects.job.generic.GenericJob.
    The functions in these modules are written in such the function names and attributes are very generic
    (get_structure(), molecular_dynamics(), version) but the functions are written to handle VASP specific input/output.

    Args:
        project (pyiron_atomistics.project.Project instance):  Specifies the project path among other attributes
        job_name (str): Name of the job

    Attributes:
        input (pyiron_atomistics.vasp.vasp.Input): Instance which handles the input

    Examples:
        Let's say you need to run a vasp simulation where you would like to control the input parameters manually. To
        set up a static dft run with Gaussian smearing and a k-point MP mesh of [6, 6, 6]. You would have to set it up
        as shown below:

        >>> ham = VaspBase(job_name="trial_job")
        >>> ham.input.incar[IBRION] = -1
        >>> ham.input.incar[ISMEAR] = 0
        >>> ham.input.kpoints.set_kpoints_file(size_of_mesh=[6, 6, 6])

        However, the according to pyiron's philosophy, it is recommended to avoid using code specific tags like IBRION,
        ISMEAR etc. Therefore the recommended way to set this calculation is as follows:

        >>> ham = VaspBase(job_name="trial_job")
        >>> ham.calc_static()
        >>> ham.set_occupancy_smearing(smearing="gaussian")
        >>> ham.set_kpoints(mesh=[6, 6, 6])
        The exact same tags as in the first examples are set automatically.

    """

    def __init__(self, project, job_name):
        super(VaspBase, self).__init__(project, job_name)
        self._sorted_indices = None
        self.input = Input()
        self.input.incar["SYSTEM"] = self.job_name
        self._output_parser = Output()
        self._potential = VaspPotentialSetter([])
        self._compress_by_default = True
        self._job_with_calculate_function = True
        self._collect_output_funct = parse_vasp_output
        self.get_enmax_among_species = get_enmax_among_potentials
        state.publications.add(self.publication)
        self.__hdf_version__ = "0.2.0"

    @property
    def structure(self):
        """

        Returns:

        """
        return GenericDFTJob.structure.fget(self)

    @structure.setter
    def structure(self, structure):
        """

        Args:
            structure:

        Returns:

        """
        GenericDFTJob.structure.fset(self, structure)
        self._reinit_potential_setter(structure=structure)

    @property
    def potential(self):
        return self._potential

    @property
    def plane_wave_cutoff(self):
        """
        Plane wave energy cutoff in eV
        """
        return self.input.incar["ENCUT"]

    @plane_wave_cutoff.setter
    def plane_wave_cutoff(self, val):
        self.input.incar["ENCUT"] = val

    @property
    def exchange_correlation_functional(self):
        """
        The exchange correlation functional used (LDA or GGA)
        """
        return self.input.potcar["xc"]

    @exchange_correlation_functional.setter
    def exchange_correlation_functional(self, val):
        if val in ["PBE", "pbe", "GGA", "gga"]:
            self.input.potcar["xc"] = "PBE"
        elif val in ["LDA", "lda"]:
            self.input.potcar["xc"] = "LDA"
        else:
            self.input.potcar["xc"] = val

    @property
    def spin_constraints(self):
        """
        Returns True if the calculation is spin constrained
        """
        if "I_CONSTRAINED_M" in self.input.incar._dataset["Parameter"]:
            return self.input.incar["I_CONSTRAINED_M"] >= 1
        else:
            return False

    @spin_constraints.setter
    def spin_constraints(self, val):
        self.input.incar["I_CONSTRAINED_M"] = val

    @property
    def write_electrostatic_potential(self):
        """
        True if the local potential or electrostatic potential LOCPOT file is/should be written
        """
        return bool(self.input.incar["LVTOT"])

    @write_electrostatic_potential.setter
    def write_electrostatic_potential(self, val):
        self.input.incar["LVTOT"] = bool(val)
        if bool(val):
            self.input.incar["LVHAR"] = True

    @property
    def write_charge_density(self):
        """
        True if the charge density file CHGCAR file is/should be written
        """
        return bool(self.input.incar["LCHARG"])

    @write_charge_density.setter
    def write_charge_density(self, val):
        self.input.incar["LCHARG"] = bool(val)

    @property
    def write_wave_funct(self):
        """
        True if the wave function file WAVECAR file is/should be written
        """
        return self.input.incar["LWAVE"]

    @write_wave_funct.setter
    def write_wave_funct(self, write_wave):
        if not isinstance(write_wave, bool):
            raise ValueError("write_wave_funct, can either be True or False.")
        self.input.incar["LWAVE"] = write_wave

    @property
    def write_resolved_dos(self):
        """
        True if the resolved DOS should be written (in the vasprun.xml file)
        """
        return self.input.incar["LORBIT"]

    @write_resolved_dos.setter
    def write_resolved_dos(self, resolved_dos):
        if not isinstance(resolved_dos, bool) and not isinstance(resolved_dos, int):
            raise ValueError(
                "write_resolved_dos, can either be True, False or 0, 1, 2, 5, 10, 11, 12."
            )
        self.input.incar["LORBIT"] = resolved_dos

    @property
    def sorted_indices(self):
        """
        How the original atom indices are ordered in the vasp format (species by species)
        """
        if self._sorted_indices is None:
            self._sorted_indices = vasp_sorter(self.structure)
        return self._sorted_indices

    @sorted_indices.setter
    def sorted_indices(self, val):
        """
        Setter for the sorted indices
        """
        self._sorted_indices = val

    @property
    def fix_spin_constraint(self):
        """
        bool: Tells if the type of constraints the spins have for this calculation
        """
        return self.spin_constraints

    @fix_spin_constraint.setter
    def fix_spin_constraint(self, boolean):
        raise NotImplementedError(
            "The fix_spin_constraint property is not implemented for this code. "
            "Instead use ham.spin_constraints - I_CONSTRAINED_M."
        )

    @property
    def fix_symmetry(self):
        if "ISYM" in self.input.incar._dataset["Parameter"]:
            return (
                self.input.incar["ISYM"] == 1
                or self.input.incar["ISYM"] == 2
                or self.input.incar["ISYM"] == 3
            )
        else:
            return True

    @fix_symmetry.setter
    def fix_symmetry(self, boolean):
        raise NotImplementedError(
            "The fix_symmetry property is not implemented for this code. "
            "Instead use ham.input.incar['ISYM']."
        )

    @property
    def potential_available(self):
        if self.structure is not None:
            return VaspPotential(
                selected_atoms=self.structure.get_species_symbols().tolist()
            )
        else:
            return VaspPotential()

    @property
    def potential_view(self):
        if self.structure is None:
            raise ValueError("Can't list potentials unless a structure is set")
        else:
            df = VaspPotentialFile(xc=self.input.potcar["xc"]).find(
                self.structure.get_species_symbols().tolist()
            )
            if len(df) > 0:
                df["Name"] = [
                    strip_xc_from_potential_name(n) for n in df["Name"].values
                ]
            return df

    @property
    def potential_list(self):
        return list(self.potential_view["Name"].values)

    @property
    def publication(self):
        return {
            "vasp": {
                "Kresse1993": {
                    "title": "Ab initio molecular dynamics for liquid metals",
                    "author": ["Kresse, G.", "Hafner, J."],
                    "journal": "Phys. Rev. B",
                    "volume": "47",
                    "issue": "1",
                    "pages": "558--561",
                    "numpages": "0",
                    "month": "jan",
                    "publisher": "American Physical Society",
                    "doi": "10.1103/PhysRevB.47.558",
                    "url": "https://link.aps.org/doi/10.1103/PhysRevB.47.558",
                },
                "Kresse1996a": {
                    "title": "Efficiency of ab-initio total energy calculations for metals and "
                    "semiconductors using a plane-wave basis set",
                    "journal": "Computational Materials Science",
                    "volume": "6",
                    "number": "1",
                    "pages": "15-50",
                    "year": "1996",
                    "issn": "0927-0256",
                    "doi": "10.1016/0927-0256(96)00008-0",
                    "url": "http://www.sciencedirect.com/science/article/pii/0927025696000080",
                    "author": ["Kresse, G.", "Furthmüller, J."],
                },
                "Kresse1996b": {
                    "title": "Efficient iterative schemes for ab initio total-energy calculations "
                    "using a plane-wave basis set",
                    "author": ["Kresse, G.", "Furthmüller, J."],
                    "journal": "Phys. Rev. B",
                    "volume": "54",
                    "issue": "16",
                    "pages": "11169--11186",
                    "numpages": "0",
                    "year": "1996",
                    "month": "oct",
                    "publisher": "American Physical Society",
                    "doi": "10.1103/PhysRevB.54.11169",
                    "url": "https://link.aps.org/doi/10.1103/PhysRevB.54.11169",
                },
            }
        }

    def collect_output(self):
        self.save_output(
            output_dict=self._collect_output_funct(
                working_directory=self.working_directory,
                **self.get_output_parameter_dict(),
            )
        )

    def get_kpoints(self):
        return [int(v) for v in self.input.kpoints[3].split()]

    def set_input_to_read_only(self):
        """
        This function enforces read-only mode for the input classes, but it has to be implement in the individual
        classes.
        """
        super(VaspBase, self).set_input_to_read_only()
        self.input.incar.read_only = True
        self.input.kpoints.read_only = True
        self.input.potcar.read_only = True

    # Compatibility functions
    def get_input_parameter_dict(self) -> dict:
        """
        Get an hierarchical dictionary of input files. On the first level the dictionary is divided in file_to_create
        and files_to_copy. Both are dictionaries use the file names as keys. In file_to_create the values are strings
        which represent the content which is going to be written to the corresponding file. In files_to_copy the values
        are the paths to the source files to be copied.

        The get_input_file_dict() function is called before the write_input() function to convert the input specified on
        the job object to strings which can be written to the working directory as well as files which are copied to the
        working directory. After the write_input() function wrote the input files the executable is called.

        Returns:
            dict: hierarchical dictionary of input files
        """
        if self.input.incar["SYSTEM"] == "pyiron_jobname":
            self.input.incar["SYSTEM"] = self.job_name
        modified_elements = {
            key: value
            for key, value in self._potential.to_dict().items()
            if value is not None
        }
        self.write_magmoms()
        self.set_coulomb_interactions()
        if "CONTCAR" in self.restart_file_dict.keys():
            if self.restart_file_dict["CONTCAR"] == "POSCAR":
                if self.server.run_mode.modal:
                    warnings.warn(
                        "The POSCAR file will be overwritten by the CONTCAR file specified in restart_file_list."
                    )
                else:
                    self.logger.info(
                        "The POSCAR file will be overwritten by the CONTCAR file specified in restart_file_list."
                    )
        input_file_dict = super().get_input_parameter_dict()
        input_file_dict["files_to_create"].update(
            self.input.get_input_parameter_dict(
                structure=self.structure,
                modified_elements=modified_elements,
            )
        )
        return input_file_dict

    def save_output(
        self, output_dict: Optional[dict] = None, shell_output: Optional[str] = None
    ):
        """
        Internal helper function to store the hierarchical output dictionary in the HDF5 file of the pyiron job object

        Args:
            output_dict (dict): hierarchical output dictionary
        """
        _ = shell_output
        output_dict_to_hdf(
            data_dict=output_dict,
            hdf=self._hdf5,
            group_name="output",
        )
        if len(self._exclude_groups_hdf) > 0 or len(self._exclude_nodes_hdf) > 0:
            self.project_hdf5.rewrite_hdf5()

    # define routines that collect all output files
    def get_output_parameter_dict(self):
        return {
            "structure": self.structure,
            "sorted_indices": self.sorted_indices,
            "read_atoms_funct": read_atoms,
            "bader_class": Bader,
            "es_class": ElectronicStructure,
            "output_parser_class": Output,
        }

    def convergence_check(self):
        """
        Checks for electronic and ionic convergence according to the user specified tolerance

        Returns:

            bool: True if converged

        """
        # Checks if sufficient empty states are present
        if not self.nbands_convergence_check():
            return False
        if "IBRION" in self["input/incar/data_dict"]["Parameter"]:
            ind = self["input/incar/data_dict"]["Parameter"].index("IBRION")
            ibrion = int(self["input/incar/data_dict"]["Value"][ind])
        else:
            ibrion = 0
        if "NELM" in self["input/incar/data_dict"]["Parameter"]:
            ind = self["input/incar/data_dict"]["Parameter"].index("NELM")
            max_e_steps = int(self["input/incar/data_dict"]["Value"][ind])
        else:
            max_e_steps = 60
        if "NSW" in self["input/incar/data_dict"]["Parameter"]:
            ind = self["input/incar/data_dict"]["Parameter"].index("NSW")
            max_i_steps = int(self["input/incar/data_dict"]["Value"][ind])
        else:
            max_i_steps = 0
        if "ALGO" in self["input/incar/data_dict"]["Parameter"]:
            ind = self["input/incar/data_dict"]["Parameter"].index("ALGO")
            algo = str(self["input/incar/data_dict"]["Value"][ind])
            if algo.upper() in ["EIGENVAL", "EXACT"]:
                if max_e_steps == 1:
                    return True
        scf_energies = self["output/generic/dft/scf_energy_free"]
        if scf_energies is None:
            scf_energies = self["output/outcar/scf_energies"]
        e_steps_converged = [len(step) < max_e_steps for step in scf_energies]
        # For calc_md() we do not care about convergence.
        if ibrion == 0 and max_i_steps != 0:
            return True
        # For calc_static only the electronic convergence matters.
        elif max_i_steps == 0 and np.all(e_steps_converged):
            return True
        # For calc_minimize only the last ionic step has to be converged!
        elif (
            0 < max_i_steps
            and len(scf_energies) < max_i_steps
            and e_steps_converged[-1]
        ):
            return True
        else:
            return False

    def cleanup(self, files_to_remove=("WAVECAR", "CHGCAR", "CHG", "vasprun.xml")):
        """
        Removes excess files (by default: WAVECAR, CHGCAR, CHG)
        """
        for file in self.files.list():
            if file in files_to_remove:
                abs_file_path = os.path.join(self.working_directory, file)
                os.remove(abs_file_path)

    def collect_logfiles(self):
        """
        Collect errors and warnings.
        """
        self.collect_errors()
        self.collect_warnings()

    def collect_warnings(self):
        """
        Collects warnings from the VASP run
        """
        # TODO: implement for VASP
        self._logger.info("collect_warnings() is not yet implemented for VASP")

    def collect_errors(self):
        """
        Collects errors from the VASP run
        """

        # error messages by VASP
        eddrmm_error_str = "WARNING in EDDRMM: call to ZHEGV failed, returncode ="
        zbrent_error_str = "ZBRENT: fatal error in bracketing"

        # warning messages for pyiron
        eddrmm_warning_str = "EDDRMM warnings occured {} times, first in ionic step {}."
        zbrent_warning_str = "'ZBRENT: fatal error in bracketing' occured. Please check VASP manual for details."
        warning_status_str = "Status is switched to 'warning'."
        aborted_status_str = "Status is switched to 'aborted'."

        # collecting errors
        num_eddrmm = 0
        snap_eddrmm = None

        zbrent_status = False

        file_name = os.path.join(self.working_directory, "error.out")
        if os.path.exists(file_name):
            with open(file_name, "r") as f:
                lines = f.readlines()

            # EDDRMM
            # If the wrong convergence algorithm is chosen, we get the following error.
            # https://cms.mpi.univie.ac.at/vasp-forum/viewtopic.php?f=4&t=17071
            lines_where_eddrmm = np.argwhere(
                [eddrmm_error_str in l for l in lines]
            ).flatten()
            num_eddrmm = len(lines_where_eddrmm)
            if num_eddrmm > 0:
                snap_eddrmm = len(
                    np.argwhere(
                        ["E0=" in l for l in lines[: lines_where_eddrmm[0]]]
                    ).flatten()
                )

            # ZBRENT
            for l in lines:
                if zbrent_error_str in l:
                    zbrent_status = True
                    break

        # handling and logging
        if zbrent_status is True:
            self.status.aborted = True
            self._logger.warning(zbrent_warning_str + aborted_status_str)
        elif snap_eddrmm is not None:
            if self.get_eddrmm_handling() == "ignore":
                self._logger.warning(eddrmm_warning_str.format(num_eddrmm, snap_eddrmm))
            elif self.get_eddrmm_handling() == "warn":
                self.status.warning = True
                self._logger.warning(
                    eddrmm_warning_str.format(num_eddrmm, snap_eddrmm)
                    + warning_status_str
                )
            elif self.get_eddrmm_handling() == "restart":
                self.status.warning = True
                self._logger.warning(
                    eddrmm_warning_str.format(num_eddrmm, snap_eddrmm)
                    + warning_status_str
                )
                if not self.input.incar["ALGO"].lower() == "normal":
                    ham_new = self.copy_hamiltonian(self.name + "_normal")
                    ham_new.input.incar["ALGO"] = "Normal"
                    ham_new.set_eddrmm_handling()
                    ham_new.run()
                    self._logger.info(
                        "Job was restarted with 'ALGO' = 'Normal' to avoid EDDRMM warning."
                    )

    def copy_hamiltonian(self, job_name):
        """
        Copies a job to new one with a different name.

        Args:
            job_name (str): Job name

        Returns:
            pyiron.vasp.vasp.Vasp: New job
        """
        ham_new = self.restart(job_name=job_name)
        ham_new.structure = self.structure
        return ham_new

    @staticmethod
    def _decompress_files_in_directory(directory):
        files = os.listdir(directory)
        for file_compressed, file, mode in [
            ["OUTCAR.gz", "OUTCAR", "gzip"],
            ["vasprun.xml.bz2", "vasprun.xml", "bzip2"],
            ["vasprun.xml.gz", "vasprun.xml", "gzip"],
        ]:
            if file_compressed in files and file not in files:
                _ = subprocess.check_output(
                    [mode, "-d", file_compressed],
                    cwd=directory,
                    shell=False,
                    universal_newlines=True,
                )
                files = os.listdir(directory)
        return files

    def from_directory(self, directory):
        """
        The Vasp instance is created by parsing the input and output from the specified directory

        Args:
            directory (str): Path to the directory
        """
        if not self.status.finished:
            # _ = s.top_path(directory)
            files = self._decompress_files_in_directory(directory)
            vp_new = Vr()
            try:
                if not ("OUTCAR" in files or "vasprun.xml" in files):
                    raise IOError("This file isn't present")
                    # raise AssertionError("OUTCAR/vasprun.xml should be present in order to import from directory")
                if "vasprun.xml" in files:
                    vp_new.from_file(filename=posixpath.join(directory, "vasprun.xml"))
                    self.structure = vp_new.get_initial_structure()
            except (IOError, VasprunError):  # except AssertionError:
                pass
                # raise AssertionError("OUTCAR/vasprun.xml should be present in order to import from directory")
            if "INCAR" in files:
                try:
                    self.input.incar.read_input(
                        posixpath.join(directory, "INCAR"), ignore_trigger="!"
                    )
                except (IndexError, TypeError, ValueError):
                    pass
            if "KPOINTS" in files:
                try:
                    self.input.kpoints.read_input(
                        posixpath.join(directory, "KPOINTS"), ignore_trigger="!"
                    )
                except (IndexError, TypeError, ValueError):
                    pass
            if "POSCAR" in files:
                if "POTCAR" in files:
                    try:
                        structure = read_atoms(
                            posixpath.join(directory, "POSCAR"),
                            species_from_potcar=True,
                        )
                    # In order to handle cases where the species info. is corrputed in POTCAR files
                    except KeyError:
                        structure = read_atoms(posixpath.join(directory, "POSCAR"))
                else:
                    structure = read_atoms(posixpath.join(directory, "POSCAR"))
            elif "CONTCAR" in files:
                structure = read_atoms(posixpath.join(directory, "CONTCAR"))
            elif "vasprun.xml" in files:
                structure = vp_new.get_initial_structure()
            else:
                raise ValueError("Unable to import job because structure not present")
            self.structure = structure
            # Always set the sorted_indices to the original order when importing from jobs
            self.sorted_indices = np.arange(len(self.structure), dtype=int)
            # Read initial magnetic moments from the INCAR file and set it to the structure
            magmom_loc = np.array(self.input.incar._dataset["Parameter"]) == "MAGMOM"
            if any(magmom_loc):
                init_moments = list()
                try:
                    value = np.array(self.input.incar._dataset["Value"])[magmom_loc][0]
                    if "*" not in value:
                        init_moments = np.array([float(val) for val in value.split()])
                    else:
                        # Values given in "number_of_atoms*value" format
                        init_moments = np.hstack(
                            (
                                [
                                    int(val.split("*")[0]) * [float(val.split("*")[1])]
                                    for val in value.split()
                                ]
                            )
                        )
                except (ValueError, IndexError, TypeError):
                    self.logger.warning(
                        "Unable to parse initial magnetic moments from the INCAR file"
                    )
                if len(init_moments) == len(self.structure):
                    self.structure.set_initial_magnetic_moments(init_moments)
                else:
                    self.logger.warning(
                        "Inconsistency during parsing initial magnetic moments from the INCAR file"
                    )

            self._write_chemical_formular_to_database()
            self._import_directory = directory
            self.status.collect = True
            self.collect_output()
            self.to_hdf()
            self.status.finished = True
        else:
            return

    def stop_calculation(self, next_electronic_step=False):
        """
        Call to stop the VASP calculation

        Args:
            next_electronic_step (bool): True if the next electronic step should be calculated

        """
        filename = os.path.join(self.working_directory, "STOPCAR")
        with open(filename, "w") as f:
            if not next_electronic_step:
                f.write("LSTOP = .TRUE.\n")
            else:
                f.write("LABORT =.TRUE.\n")

    def to_dict(self):
        job_dict = super().to_dict()
        job_dict.update({"input/" + k: v for k, v in self._structure_to_dict().items()})
        job_dict.update({"input/" + k: v for k, v in self.input.to_dict().items()})
        job_dict["input/potential_dict"] = self._potential.to_dict()
        return job_dict

    def from_dict(self, obj_dict):
        super().from_dict(obj_dict=obj_dict)
        self._structure_from_dict(obj_dict=obj_dict)
        self.input.from_dict(obj_dict=obj_dict["input"])
        if "potential_dict" in obj_dict["input"].keys():
            self._potential.from_dict(obj_dict=obj_dict["input"]["potential_dict"])

    def to_hdf(self, hdf=None, group_name=None):
        """
        Stores the instance attributes into the hdf5 file

        Args:
            hdf (pyiron_base.generic.hdfio.ProjectHDFio): The HDF file/path to write the data to
            group_name (str): The name of the group under which the data must be stored as

        """
        super(VaspBase, self).to_hdf(hdf=hdf, group_name=group_name)
        self._output_parser.to_hdf(self._hdf5)
        if _vasp_generic_energy_free_affected(self):
            self.logger.warn(
                "Generic energy_pot does not match electronic free energy! "
                "Generic energies is not consistent to generic forces and stress, "
                "call project.maintenance.local.vasp_energy_pot_as_free_energy() "
                "to correct generic energy!"
            )

    def from_hdf(self, hdf=None, group_name=None):
        """
        Recreates instance from the hdf5 file

        Args:
            hdf (pyiron_base.generic.hdfio.ProjectHDFio): The HDF file/path to read the data from
            group_name (str): The name of the group under which the data must be stored as

        """
        super(VaspBase, self).from_hdf(hdf=hdf, group_name=group_name)
        if (
            "output" in self.project_hdf5.list_groups()
            and "structure" in self["output"].list_groups()
        ):
            self._output_parser.from_hdf(self._hdf5)

    def reset_output(self):
        """
        Resets the output instance
        """
        self._output_parser = Output()

    def get_final_structure_from_file(self, cwd, filename="CONTCAR"):
        """
        Get the final structure of the simulation usually from the CONTCAR file

        Args:
            filename (str): Path to the CONTCAR file in VASP

        Returns:
            pyiron.atomistics.structure.atoms.Atoms: The final structure
        """
        return get_final_structure_from_file(
            working_directory=cwd,
            filename=filename,
            structure=self.structure,
            sorted_indices=self.sorted_indices,
        )

    def write_magmoms(self):
        """
        Write the magnetic moments in INCAR from that assigned to the species
        """
        if self.structure.has("initial_magmoms"):
            if "ISPIN" not in self.input.incar._dataset["Parameter"]:
                self.input.incar["ISPIN"] = 2

                # LORBIT MUST BE SET TO WRITE PER-ATOM MAGNETISATIONS
                # Check if LORBIT is in the INCAR parameters
                if "LORBIT" not in self.input.incar._dataset["Parameter"]:
                    # If LORBIT is not set, set it to 10
                    self.input.incar["LORBIT"] = 10
                    self.logger.warning(
                        "We have set LORBIT = 10 to write magmoms to OUTCAR! This is a spin-polarized calculation."
                    )
                else:
                    # If LORBIT is set but not in the valid range, set it to 10 and warn
                    if self.input.incar["LORBIT"] not in [
                        0,
                        1,
                        2,
                        5,
                        10,
                        11,
                        12,
                        13,
                        14,
                    ]:
                        self.logger.warning(
                            "Invalid LORBIT tag. We have set LORBIT = 10 to write magmoms to OUTCAR! This is a spin-polarized calculation."
                        )
                        self.input.incar["LORBIT"] = 10

            if self.input.incar["ISPIN"] != 1:
                final_cmd = "   ".join(
                    [
                        (
                            " ".join([str(spinmom) for spinmom in spin])
                            if isinstance(spin, (list, np.ndarray))
                            else str(spin)
                        )
                        for spin in self.structure.get_initial_magnetic_moments()[
                            self.sorted_indices
                        ]
                    ]
                )
                state.logger.debug("Magnetic Moments are: {0}".format(final_cmd))
                if "MAGMOM" not in self.input.incar._dataset["Parameter"]:
                    self.input.incar["MAGMOM"] = final_cmd
                if any(
                    [
                        isinstance(spin, (list, np.ndarray))
                        for spin in self.structure.get_initial_magnetic_moments()
                    ]
                ):
                    self.input.incar["LNONCOLLINEAR"] = True
                    if (
                        self.spin_constraints
                        and "M_CONSTR" not in self.input.incar._dataset["Parameter"]
                    ):
                        self.input.incar["M_CONSTR"] = final_cmd
                    if (
                        self.spin_constraints
                        or "M_CONSTR" in self.input.incar._dataset["Parameter"]
                    ):
                        if "ISYM" not in self.input.incar._dataset["Parameter"]:
                            self.input.incar["ISYM"] = 0
                    if (
                        self.spin_constraints
                        and "LAMBDA" not in self.input.incar._dataset["Parameter"]
                    ):
                        raise ValueError(
                            "LAMBDA is not specified but it is necessary for non collinear calculations."
                        )
                    if (
                        self.spin_constraints
                        and "RWIGS" not in self.input.incar._dataset["Parameter"]
                    ):
                        raise ValueError(
                            "Parameter RWIGS has to be set for spin constraint calculations"
                        )
                if self.spin_constraints and not self.input.incar["LNONCOLLINEAR"]:
                    raise ValueError(
                        "Spin constraints are only avilable for non collinear calculations."
                    )
                # LORBIT MUST BE SET TO WRITE PER-ATOM MAGNETISATIONS
                # Check if LORBIT is in the INCAR parameters
                if "LORBIT" not in self.input.incar._dataset["Parameter"]:
                    # If LORBIT is not set, set it to 10
                    self.input.incar["LORBIT"] = 10
                    self.logger.warning(
                        "We have set LORBIT = 10 to write magmoms to OUTCAR! This is a spin-polarized calculation."
                    )
                else:
                    # If LORBIT is set but not in the valid range, set it to 10 and warn
                    if self.input.incar["LORBIT"] not in [
                        0,
                        1,
                        2,
                        5,
                        10,
                        11,
                        12,
                        13,
                        14,
                    ]:
                        self.logger.warning(
                            "Invalid LORBIT tag. We have set LORBIT = 10 to write magmoms to OUTCAR! This is a spin-polarized calculation."
                        )
                        self.input.incar["LORBIT"] = 10
            else:
                state.logger.debug(
                    "Spin polarized calculation is switched off by the user. No magnetic moments are written."
                )
        else:
            state.logger.debug("No magnetic moments")

    def set_eddrmm_handling(self, status="warn"):
        """
        Sets the way, how EDDRMM warning is handled.

        Args:
            status (str): new status of EDDRMM handling (can be 'warn', 'ignore', or 'restart')
        """
        if status == "warn" or status == "ignore" or status == "restart":
            self.input._eddrmm = status
        else:
            raise ValueError

    def get_eddrmm_handling(self):
        """
        Returns:
            str: status of EDDRMM handling
        """
        return self.input._eddrmm

    def set_coulomb_interactions(self, interaction_type=2, ldau_print=True):
        """
        Write the on-site Coulomb interactions in the INCAR file

        Args:
            interaction_type (int): Type of Coulombic interaction
                1 - Asimov method
                2 - Dudarev method
            ldau_print (boolean): True/False
        """
        obj_lst = self.structure.get_species_objects()
        ldaul = []
        ldauu = []
        ldauj = []
        needed = False
        for el_obj in obj_lst:
            conditions = []
            if isinstance(el_obj.tags, dict):
                for tag in ["ldauu", "ldaul", "ldauj"]:
                    conditions.append(tag in el_obj.tags.keys())
                if not any(conditions):
                    ldaul.append("-1")
                    ldauu.append("0")
                    ldauj.append("0")
                if any(conditions) and not all(conditions):
                    raise ValueError(
                        "All three tags ldauu,ldauj and ldaul have to be specified"
                    )
                if all(conditions):
                    needed = True
                    ldaul.append(str(el_obj.tags["ldaul"]))
                    ldauu.append(str(el_obj.tags["ldauu"]))
                    ldauj.append(str(el_obj.tags["ldauj"]))
        if needed:
            self.input.incar["LDAU"] = True
            self.input.incar["LDAUTYPE"] = interaction_type
            self.input.incar["LDAUL"] = " ".join(ldaul)
            self.input.incar["LDAUU"] = " ".join(ldauu)
            self.input.incar["LDAUJ"] = " ".join(ldauj)
            if ldau_print:
                self.input.incar["LDAUPRINT"] = 2
        else:
            state.logger.debug("No on site coulomb interactions")

    def set_algorithm(self, algorithm="Fast", ialgo=None):
        """
        Sets the type of electronic minimization algorithm

        Args:
            algorithm (str): Algorithm defined by VASP (Fast, Normal etc.)
            ialgo (int): Sets the IALGO tag in VASP. If not none, this overwrites algorithm
        """
        algorithm_list = ["Fast", "Accurate", "Normal", "Very Fast"]
        if ialgo is not None:
            self.input.incar["IALGO"] = int(ialgo)
        else:
            self.input.incar["ALGO"] = str(algorithm)
            if algorithm not in algorithm_list:
                state.logger.warning(
                    msg="Algorithm {} is unusual for VASP. "
                    "I hope you know what you are up to".format(algorithm)
                )

    def calc_minimize(
        self,
        electronic_steps=60,
        ionic_steps=100,
        max_iter=None,
        pressure=None,
        algorithm=None,
        retain_charge_density=False,
        retain_electrostatic_potential=False,
        ionic_energy_tolerance=None,
        ionic_force_tolerance=None,
        volume_only=False,
        cell_only=False,
    ):
        """
        Function to setup the hamiltonian to perform ionic relaxations using DFT. The ISIF tag has to be supplied
        separately.

        Args:
            electronic_steps (int): Maximum number of electronic steps
            ionic_steps (int): Maximum number of ionic
            max_iter (int): Maximum number of iterations
            pressure (float): External pressure to be applied
            algorithm (str): Type of VASP algorithm to be used "Fast"/"Accurate"
            retain_charge_density (bool): True if the charge density should be written
            retain_electrostatic_potential (boolean): True if the electrostatic potential should be written
            ionic_energy_tolerance (float): Ionic energy convergence criteria (eV)
            ionic_force_tolerance (float): Ionic forces convergence criteria (overwrites ionic energy) (ev/A)
            volume_only (bool): Option to relax only the volume (keeping the relative coordinates fixed)
            cell_only (bool): Option to relax only the cell parameters (keeping the relative coordinates fixed)
        """
        super(VaspBase, self).calc_minimize(
            electronic_steps=electronic_steps,
            ionic_steps=ionic_steps,
            max_iter=max_iter,
            pressure=pressure,
            ionic_energy_tolerance=ionic_energy_tolerance,
            ionic_force_tolerance=ionic_force_tolerance,
            volume_only=volume_only,
        )
        if volume_only:
            self.input.incar["ISIF"] = 7
        elif cell_only:
            self.input.incar["ISIF"] = 6
        else:
            if pressure == 0.0:
                self.input.incar["ISIF"] = 3
            elif pressure is None:
                self.input.incar["ISIF"] = 2
            else:
                raise ValueError("Non-zero pressure not supported!")

        if max_iter:
            electronic_steps = max_iter
            ionic_steps = max_iter

        self.input.incar["IBRION"] = 2
        self.input.incar["NELM"] = electronic_steps
        self.input.incar["NSW"] = ionic_steps
        if algorithm is not None:
            self.set_algorithm(algorithm=algorithm)
        self.write_charge_density = retain_charge_density
        self.write_electrostatic_potential = retain_electrostatic_potential
        self.set_convergence_precision(
            ionic_force_tolerance=ionic_force_tolerance,
            ionic_energy_tolerance=ionic_energy_tolerance,
            electronic_energy=None,
        )

    def calc_static(
        self,
        electronic_steps=100,
        algorithm=None,
        retain_charge_density=False,
        retain_electrostatic_potential=False,
    ):
        """
        Function to setup the hamiltonian to perform static SCF DFT runs.

        Args:
            electronic_steps (int): Maximum number of electronic steps
            algorithm (str): Type of VASP algorithm to be used "Fast"/"Accurate"
            retain_charge_density (bool): True if
            retain_electrostatic_potential (bool): True/False
        """
        super().calc_static(electronic_steps=electronic_steps)
        self.input.incar["IBRION"] = -1
        self.input.incar["NELM"] = electronic_steps
        # Make sure vasp runs only 1 ionic step
        self.input.incar["NSW"] = 0
        if algorithm is not None:
            if algorithm is not None:
                self.set_algorithm(algorithm=algorithm)
        self.write_charge_density = retain_charge_density
        self.write_electrostatic_potential = retain_electrostatic_potential

    def calc_md(
        self,
        temperature=None,
        n_ionic_steps=1000,
        n_print=1,
        time_step=1.0,
        retain_charge_density=False,
        retain_electrostatic_potential=False,
        **kwargs,
    ):
        """
        Sets appropriate tags for molecular dynamics in VASP

        Args:
            temperature (int/float/list): Temperature/ range of temperatures in Kelvin
            n_ionic_steps (int): Maximum number of ionic steps
            n_print (int): Prints outputs every n_print steps
            time_step (float): time step (fs)
            retain_charge_density (bool): True id the charge density should be written
            retain_electrostatic_potential (bool): True if the electrostatic potential should be written
        """
        super(VaspBase, self).calc_md(
            temperature=temperature,
            n_ionic_steps=n_ionic_steps,
            n_print=n_print,
            time_step=time_step,
            **kwargs,
        )
        if temperature is not None:
            # NVT ensemble
            self.input.incar["SMASS"] = 3
            if isinstance(temperature, (int, float)):
                self.input.incar["TEBEG"] = temperature
            else:
                self.input.incar["TEBEG"] = temperature[0]
                self.input.incar["TEEND"] = temperature[-1]
        else:
            # NVE ensemble
            self.input.incar["SMASS"] = -3
        self.input.incar["NSW"] = n_ionic_steps
        self.input.incar["NBLOCK"] = int(n_print)
        self.input.incar["POTIM"] = time_step
        if "ISYM" not in self.input.incar.keys():
            self.input.incar["ISYM"] = 0
        self.write_charge_density = retain_charge_density
        self.write_electrostatic_potential = retain_electrostatic_potential
        for key in kwargs.keys():
            self.logger.warning("Tag {} not relevant for vasp".format(key))

    def set_for_band_structure_calc(
        self, num_points, structure=None, read_charge_density=True
    ):
        """
        Sets up the input for a non self-consistent bandstructure calculation

        Args:
            num_points (int): Number of k-points along the total BZ path
            structure (atomistics.structure.atoms.Atoms instance): Structure for which the bandstructure is to be
                                                                       generated. (default is the input structure)
            read_charge_density (boolean): If True, a charge density from a previous SCF run is used (recommended)
        """
        if read_charge_density:
            self.input.incar["ICHARG"] = 11
        if structure is None:
            if not (self._output_parser.structure is not None):
                raise AssertionError()
            structure = self._output_parser.structure
        bs_obj = Bandstructure(structure)
        _, q_point_list, [_, _] = bs_obj.get_path(
            num_points=num_points, path_type="full"
        )
        q_point_list = np.array(q_point_list)
        self._set_kpoints(
            scheme="Manual",
            symmetry_reduction=False,
            manual_kpoints=q_point_list,
            weights=None,
            reciprocal=False,
        )

    def set_convergence_precision(
        self,
        ionic_energy_tolerance=1.0e-3,
        electronic_energy=1.0e-7,
        ionic_force_tolerance=1.0e-2,
    ):
        """
        Sets the electronic and ionic convergence precision. For ionic convergence either the energy or the force
        precision is required

        Args:
            ionic_energy_tolerance (float): Ionic energy convergence precision (eV)
            electronic_energy (float/NoneType): Electronic energy convergence precision (eV)
            ionic_force_tolerance (float): Ionic force convergence precision (eV/A)
        """
        if ionic_force_tolerance is not None:
            self.input.incar["EDIFFG"] = -1.0 * abs(ionic_force_tolerance)
        elif ionic_energy_tolerance is not None:
            self.input.incar["EDIFFG"] = abs(ionic_energy_tolerance)
        else:
            # Using default convergence criterion
            self.input.incar["EDIFFG"] = (
                -0.01 if self.input.incar["ISIF"] not in (5, 6, 7) else 0.01
            )
        if electronic_energy is not None:
            self.input.incar["EDIFF"] = electronic_energy

    def set_dipole_correction(self, direction=2, dipole_center=None):
        """
        Apply a dipole correction using the dipole layer method proposed by `Neugebauer & Scheffler`_

        Args:
            direction (int): Direction along which the field has to be applied (0, 1, or 2)
            dipole_center (list/numpy.ndarray): Position of the center of the dipole (not the center of the vacuum) in
                                                relative coordinates

        .. _Neugebauer & Scheffler: https://doi.org/10.1103/PhysRevB.46.16067
        """
        self.set_electric_field(
            e_field=0, direction=direction, dipole_center=dipole_center
        )

    def set_electric_field(self, e_field=0.1, direction=2, dipole_center=None):
        """
        Set an external electric field using the dipole layer method proposed by `Neugebauer & Scheffler`_

        Args:
            e_field (float): Magnitude of the external electric field (eV/A)
            direction (int): Direction along which the field has to be applied (0, 1, or 2)
            dipole_center (list/numpy.ndarray): Position of the center of the dipole (not the center of the vacuum) in
                                                relative coordinates

        .. _Neugebauer & Scheffler: https://doi.org/10.1103/PhysRevB.46.16067

        """
        if not (direction in range(3)):
            raise AssertionError()
        self.input.incar["ISYM"] = 0
        self.input.incar["LORBIT"] = 11
        self.input.incar["IDIPOL"] = direction + 1
        self.input.incar["LDIPOL"] = True
        self.input.incar["EFIELD"] = e_field
        if dipole_center is not None:
            self.input.incar["DIPOL"] = " ".join(str(val) for val in dipole_center)

    @deprecate(
        ismear="Preferably use parameters `smearing` and `order` "
        "to set the type of smearing you want"
    )
    def set_occupancy_smearing(
        self,
        smearing: str = None,
        width: float = None,
        order: int = 1,
        ismear: int = None,
    ) -> None:
        """
        Set how the finite temperature smearing is applied in determining partial occupancies

        Args:
            smearing (str): Type of smearing (Fermi, Gaussian, or Methfessel-Paxton)
            width (float): Smearing width (eV)
            order (int): order (int): Smearing order (only for Methfessel-Paxton)
            ismear (int): (Deprecated) Directly sets the ISMEAR tag. Overwrites the smearing tag
        """
        if ismear is not None:
            self.input.incar["ISMEAR"] = int(ismear)
        elif smearing.lower().startswith("meth") or smearing.lower().startswith("mp"):
            self.input.incar["ISMEAR"] = int(order)
        elif smearing.lower().startswith("fermi"):
            self.input.incar["ISMEAR"] = -1
        elif smearing.lower().startswith("gauss"):
            self.input.incar["ISMEAR"] = 0
        else:
            raise ValueError(
                f"Smearing scheme {smearing} is not available. Only types 'Fermi', 'Gaussian', "
                f"and 'Methfessel-Paxton'"
            )
        if width is not None:
            self.input.incar["SIGMA"] = width

    def set_fft_mesh(self, nx=None, ny=None, nz=None):
        """
        Set the number of points in the respective directions for the 3D FFT mesh used for computing the charge density
        or electrostatic potentials. In VASP, using PAW potentials, this refers to the "finer fft mesh". If no values
        are set, the default settings from Vasp are used to set the number of grid points.

        Args:
            nx (int): Number of points on the x-grid
            ny (int): Number of points on the y-grid
            nz (int): Number of points on the z-grid
        """
        if nx is not None:
            self.input.incar["NGXF"] = int(nx)
        if ny is not None:
            self.input.incar["NGYF"] = int(ny)
        if nz is not None:
            self.input.incar["NGZF"] = int(nz)

    def set_mixing_parameters(
        self,
        method=None,
        n_pulay_steps=None,
        density_mixing_parameter=None,
        spin_mixing_parameter=None,
        density_residual_scaling=None,
        spin_residual_scaling=None,
    ):
        if density_residual_scaling is not None or spin_residual_scaling is not None:
            raise NotImplementedError("Residual scaling is not implemented in VASP")
        if method is None:
            method = "PULAY"
        if method.upper() == "PULAY":
            self.input.incar["IMIX"] = 4
        if method.upper() == "KERKER":
            self.input.incar["IMIX"] = 1
        if n_pulay_steps is not None:
            self.input.incar["MAXMIX"] = n_pulay_steps
        if density_mixing_parameter is not None:
            self.input.incar["AMIX"] = density_mixing_parameter

    set_mixing_parameters.__doc__ = GenericDFTJob.set_mixing_parameters.__doc__

    def set_empty_states(self, n_empty_states=None):
        """
        Sets the number of empty states in the calculation
        Args:
            n_empty_states (int): Required number of empty states

        """
        n_elect = self.get_nelect()
        if n_empty_states is not None:
            if n_empty_states < 0:
                raise ValueError(
                    f"Number of empty states must be a positive integer or zero, not {n_empty_states}!"
                )
            self.input.incar["NBANDS"] = int(round(n_elect / 2)) + int(n_empty_states)

    def get_nelect(self):
        """
        Returns the number of electrons in the systems

        Returns:
            float: Number of electrons in the system

        """
        if not self.status.finished and self.structure is not None:
            potential = VaspPotentialFile(xc=self.input.potcar["xc"])
            return sum(
                [
                    potential.find_default(el).n_elect.values[-1] * n_atoms
                    for el, n_atoms in self.structure.get_parent_basis()
                    .get_number_species_atoms()
                    .items()
                ]
            )
        else:
            return self["output/generic/dft/n_elect"]

    def get_magnetic_moments(self, iteration_step=-1):
        """
        Gives the magnetic moments of a calculation for each iteration step.

        Args:
            iteration_step (int): Step for which the structure is requested

        Returns:
            numpy.ndarray/None: array of final magmetic moments or None if no magnetic moment is given
        """
        spins = self["output/generic/dft/final_magmoms"]
        if spins is not None and len(spins) > 0:
            return spins[iteration_step]
        else:
            return None

    def get_charge_density(self):
        """
        Gets the charge density from the hdf5 file. This value is normalized by the volume

        Returns:
                atomistics.volumetric.generic.VolumetricData instance
        """
        if not self.status.finished:
            return
        else:
            with self.project_hdf5.open("output") as ho:
                cd_obj = VaspVolumetricData()
                cd_obj.from_hdf(ho, "charge_density")
            return cd_obj

    def get_valence_and_total_charge_density(self):
        """
        Gives the valence and total charge densities

        Returns:
            tuple: The required charge densities
        """
        return get_valence_and_total_charge_density(
            working_directory=self.working_directory,
        )

    def get_electrostatic_potential(self):
        """
        Gets the electrostatic potential from the hdf5 file.

        Returns:
                atomistics.volumetric.generic.VolumetricData instance
        """
        if not self.status.finished:
            return
        else:
            with self.project_hdf5.open("output") as ho:
                es_obj = VaspVolumetricData()
                es_obj.from_hdf(ho, "electrostatic_potential")
            return es_obj

    def restart(self, job_name=None, job_type=None):
        """
        Creates a "restart" job from an existing Vasp calculation.
        (Default behaviour is to copy CONTCAR -> POSCAR, all other job inputs are copied from original job)

        Usage: job.restart().run() restarts the job with job_name: "$(original_job_name)_restart"

        Args:
            job_name (str): Job name
            job_type (str): Job type. If not specified a Vasp job type is assumed

        Returns:
            new_ham (vasp.vasp.Vasp instance): New job
        """
        new_ham = super(VaspBase, self).restart(job_name=job_name, job_type=job_type)
        if not self.is_compressed():
            try:
                self.save_output(
                    output_dict=self._collect_output_funct(
                        working_directory=self.working_directory,
                        **self.get_output_parameter_dict(),
                    ),
                )
                self.compress()
            except VaspCollectError:
                self.logger.warn(
                    "Tried to automatically recollect job in case it timed out during collection, but it failed."
                )
        if new_ham.__name__ == self.__name__:
            new_ham.input.potcar["xc"] = self.input.potcar["xc"]
        if new_ham.input.incar["MAGMOM"] is not None:
            del new_ham.input.incar["MAGMOM"]
        if new_ham.input.incar["M_CONSTR"] is not None:
            del new_ham.input.incar["M_CONSTR"]
        if new_ham.input.incar["LNONCOLLINEAR"] is not None:
            del new_ham.input.incar["LNONCOLLINEAR"]
        return new_ham

    def restart_for_band_structure_calculations(self, job_name=None):
        """
        Restart a new job created from an existing Vasp calculation by reading the charge density
        for band structure calculations.

        Args:
            job_name (str/None): Job name

        Returns:
            new_ham (vasp.vasp.Vasp instance): New job
        """
        return self.restart_from_charge_density(
            job_name=job_name, job_type=None, icharg=11, self_consistent_calc=None
        )

    def get_icharg_value(self, icharg=None, self_consistent_calc=None):
        """
        Gives the correct ICHARG value for the restart calculation.

        Args:
            icharg (int/None): If given, this value will be checked for validity and returned.
            self_consistent_calc (bool/None): If 'True' returns 1, if 'False' returns 11,
                if 'None' returns based on the job either 1 or 11.

        Returns:
            int: the icharg tag

        """
        if icharg is None:
            if self_consistent_calc is True:
                return 1
            if self_consistent_calc is False:
                return 11
            if (
                "ICHARG" in self.input.incar.keys()
                and int(self.input.incar["ICHARG"]) > 9
            ):
                return 11
            return 1
        if icharg not in [0, 1, 2, 4, 10, 11, 12]:
            raise ValueError(
                "The value '{}' is not a proper input for 'icharg'. Look at VASP manual.".format(
                    icharg
                )
            )
        return icharg

    def restart_from_charge_density(
        self,
        job_name=None,
        job_type=None,
        icharg=None,
        self_consistent_calc=None,
    ):
        """
        Restart a new job created from an existing Vasp calculation by reading the charge density.

        Args:
            job_name (str/None): Job name
            job_type (str/None): Job type. If not specified a Vasp job type is assumed
            icharg (int/None): If given, this value will be checked for validity and returned.
            self_consistent_calc (bool/None): If 'True' returns 1, if 'False' returns 11,
                if 'None' returns based on the job either 1 or 11.

        Returns:
            new_ham (vasp.vasp.Vasp instance): New job
        """
        new_ham = self.restart(job_name=job_name, job_type=job_type)

        if new_ham.__name__ == self.__name__:
            new_ham.restart_file_list.append(self.files.CHGCAR)
            new_ham.input.incar["ICHARG"] = self.get_icharg_value(
                icharg=icharg,
                self_consistent_calc=self_consistent_calc,
            )
        return new_ham

    def append_charge_density(self, job_specifier=None, path=None):
        """
        Append charge density file (CHGCAR)

        Args:
            job_specifier (str/int): name of the job or job ID
            path (str): path to CHGCAR file
        """
        if job_specifier is None and path is None:
            raise ValueError("Either 'job_specifier' or 'path' has to be given!")
        elif job_specifier is not None:
            self.restart_file_list.append(
                self.project.inspect(job_specifier=job_specifier).files.CHGCAR
            )
        elif os.path.basename(path) == "CHGCAR":
            self.restart_file_list.append(path)
        else:
            self.restart_file_list.append(os.path.join(path, "CHGCAR"))

    def restart_from_wave_and_charge(
        self,
        job_name=None,
        job_type=None,
        icharg=None,
        self_consistent_calc=None,
        istart=1,
    ):
        """
        Restart a new job created from an existing Vasp calculation by reading the charge density and the wave
        function.

        Args:
            job_name (str/None): Job name
            job_type (str/None): Job type. If not specified a Vasp job type is assumed
            icharg (int/None): If given, this value will be checked for validity and returned.
            self_consistent_calc (bool/None): If 'True' returns 1, if 'False' returns 11,
                if 'None' returns based on the job either 1 or 11.
            istart (int): Vasp ISTART tag

        Returns:
            new_ham (vasp.vasp.Vasp instance): New job
        """
        new_ham = self.restart(job_name=job_name, job_type=job_type)
        if new_ham.__name__ == self.__name__:
            new_ham.restart_file_list.append(self.files.CHGCAR)
            new_ham.restart_file_list.append(self.files.WAVECAR)
            new_ham.input.incar["ISTART"] = istart
            new_ham.input.incar["ICHARG"] = self.get_icharg_value(
                icharg=icharg,
                self_consistent_calc=self_consistent_calc,
            )
        return new_ham

    def compress(self, files_to_compress=None):
        """
        Compress the output files of a job object.

        Args:
            files_to_compress (list): A list of files to compress (optional)
        """
        if files_to_compress is None:
            files_to_compress = [
                f
                for f in self.files.list()
                if f
                not in [
                    "CHGCAR",
                    "CONTCAR",
                    "WAVECAR",
                    "STOPCAR",
                    "AECCAR0",
                    "AECCAR1",
                    "AECCAR2",
                ]
            ]
        # delete empty files
        for f in self.files.list():
            filename = os.path.join(self.working_directory, f)
            if (
                f not in files_to_compress
                and os.path.exists(filename)
                and os.stat(filename).st_size == 0
            ):
                os.remove(filename)
        super(VaspBase, self).compress(files_to_compress=files_to_compress)

    def restart_from_wave_functions(self, job_name=None, job_type=None, istart=1):
        """
        Restart a new job created from an existing Vasp calculation by reading the wave functions.

        Args:
            job_name (str/None): Job name
            job_type (str/None): Job type. If not specified a Vasp job type is assumed
            istart (int): Vasp ISTART tag

        Returns:
            new_ham (vasp.vasp.Vasp instance): New job
        """
        new_ham = self.restart(job_name=job_name, job_type=job_type)
        if new_ham.__name__ == self.__name__:
            new_ham.restart_file_list.append(self.files.WAVECAR)
            new_ham.input.incar["ISTART"] = istart
        return new_ham

    def append_wave_function(self, job_specifier=None, path=None):
        """
        Append wave function file (WAVECAR)

        Args:
            job_specifier (str/int): name of the job or job ID
            path (str): path to WAVECAR file
        """
        if job_specifier is None and path is None:
            raise ValueError("Either 'job_specifier' or 'path' has to be given!")
        elif job_specifier is not None:
            self.restart_file_list.append(
                self.project.inspect(job_specifier=job_specifier).files.WAVECAR
            )
        elif os.path.basename(path) == "WAVECAR":
            self.restart_file_list.append(path)
        else:
            self.restart_file_list.append(posixpath.join(path, "WAVECAR"))

    def set_rwigs(self, rwigs_dict):
        """
        Sets the radii of Wigner-Seitz cell. (RWIGS tag)

        Args:
            rwigs_dict (dict): Dictionary of species and corresponding radii.
                (structure has to be defined before)
        """
        if not isinstance(rwigs_dict, dict):
            raise AssertionError("'rwigs_dict' has to be a dict!")
        if not all([isinstance(val, (int, float)) for val in rwigs_dict.values()]):
            raise ValueError("The values of 'rwigs_dict' has to be floats!")
        species_keys = self.structure.get_number_species_atoms().keys()
        rwigs_keys = rwigs_dict.keys()
        for k in species_keys:
            if k not in list(rwigs_keys):
                raise ValueError("'{}' is not in rwigs_dict!".format(k))

        rwigs = [rwigs_dict[i] for i in species_keys]
        self.input.incar["RWIGS"] = " ".join(map(str, rwigs))

    def get_rwigs(self):
        """
        Gets the radii of Wigner-Seitz cell. (RWIGS tag)

        Returns:
            dict: dictionary of radii
        """
        if "RWIGS" in self.input.incar._dataset["Parameter"]:
            species_keys = self.structure.get_number_species_atoms().keys()
            rwigs = [float(i) for i in self.input.incar["RWIGS"].split()]
            rwigs_dict = dict()
            for i, k in enumerate(species_keys):
                rwigs_dict.update({k: rwigs[i]})
            return rwigs_dict
        else:
            return None

    def set_spin_constraint(self, lamb, rwigs_dict, direction=False, norm=False):
        """
        Sets spin constrains including 'LAMBDA' and 'RWIGS'.

        Args:
            lamb (float): LAMBDA tag
            rwigs_dict (dict): Dictionary of species and corresponding radii.
                (structure has to be defined before)
            direction (bool): (True/False) constrain spin direction.
            norm (bool): (True/False) constrain spin norm (magnitude).
        """
        if not isinstance(direction, bool):
            raise AssertionError("'direction' has to be a bool!")
        if not isinstance(norm, bool):
            raise AssertionError("'lamb' has to be a bool!")
        if not isinstance(lamb, float):
            raise AssertionError("'lamb' has to be a float!")
        if direction and norm:
            self.input.incar["I_CONSTRAINED_M"] = 2
        elif direction:
            self.input.incar["I_CONSTRAINED_M"] = 1
        elif norm:
            raise ValueError("Constraining norm only is not possible.")
        else:
            raise ValueError(
                "You have to constrain either direction or norm and direction."
            )

        self.input.incar["LAMBDA"] = lamb
        self.set_rwigs(rwigs_dict)

    def validate_ready_to_run(self):
        super(VaspBase, self).validate_ready_to_run()
        if "spin_constraint" in self.structure.arrays.keys():
            raise NotImplementedError(
                "The spin_constraint tag is not supported by VASP."
            )

    def list_potentials(self):
        """
        Lists all the possible POTCAR files for the elements in the structure depending on the XC functional

        Returns:
           list: a list of available potentials
        """
        return self.potential_list

    def _set_kpoints(
        self,
        mesh=None,
        scheme="MP",
        center_shift=None,
        symmetry_reduction=True,
        manual_kpoints=None,
        weights=None,
        reciprocal=True,
        n_path=None,
        path_name=None,
    ):
        """
        Function to setup the k-points for the VASP job

        Args:
            mesh (list): Size of the mesh (in the MP scheme)
            scheme (str): Type of k-point generation scheme (MP/GC(gamma centered)/GP(gamma point)/Manual/Line)
            center_shift (list): Shifts the center of the mesh from the gamma point by the given vector
            symmetry_reduction (boolean): Tells if the symmetry reduction is to be applied to the k-points
            manual_kpoints (list/numpy.ndarray): Manual list of k-points
            weights(list/numpy.ndarray): Manually supplied weights to each k-point in case of the manual mode
            reciprocal (bool): Tells if the supplied values are in reciprocal (direct) or cartesian coordinates (in
            reciprocal space)
            n_path (int): Number of points per trace part for line mode
            path_name (str): Name of high symmetry path used for band structure calculations.
        """
        if not symmetry_reduction:
            self.input.incar["ISYM"] = -1
        scheme_list = ["MP", "GC", "GP", "Line", "Manual"]
        if not (scheme in scheme_list):
            raise AssertionError()
        if scheme == "MP":
            if mesh is None:
                mesh = [int(val) for val in self.input.kpoints[3].split()]
            self.input.kpoints.set_kpoints_file(size_of_mesh=mesh, shift=center_shift)
        if scheme == "GC":
            if mesh is None:
                mesh = [int(val) for val in self.input.kpoints[3].split()]
            self.input.kpoints.set_kpoints_file(
                size_of_mesh=mesh, shift=center_shift, method="Gamma centered"
            )
        if scheme == "GP":
            self.input.kpoints.set_kpoints_file(
                size_of_mesh=[1, 1, 1], method="Gamma Point"
            )
        if scheme == "Line":
            if n_path is None and self.input.kpoints._n_path is None:
                raise ValueError("n_path has to be defined")
            high_symmetry_points = self.structure.get_high_symmetry_points()
            if high_symmetry_points is None:
                raise ValueError("high_symmetry_points has to be defined")

            if path_name is None and self.input.kpoints._path_name is None:
                raise ValueError("path_name has to be defined")
            if path_name not in self.structure.get_high_symmetry_path().keys():
                raise ValueError("path_name is not a valid key of high_symmetry_path")

            if path_name is not None:
                self.input.kpoints._path_name = path_name
            if n_path is not None:
                self.input.kpoints._n_path = n_path

            self.input.kpoints.set_kpoints_file(
                method="Line",
                n_path=self.input.kpoints._n_path,
                path=self._get_path_for_kpoints(self.input.kpoints._path_name),
            )
        if scheme == "Manual":
            if manual_kpoints is None:
                raise ValueError(
                    "For the manual mode, the kpoints list should be specified"
                )
            else:
                if weights is not None:
                    if not (len(manual_kpoints) == len(weights)):
                        raise AssertionError()
                self.input.kpoints.set_value(line=1, val=str(len(manual_kpoints)))
                if reciprocal:
                    self.input.kpoints.set_value(line=2, val="Reciprocal")
                else:
                    self.input.kpoints.set_value(line=2, val="Cartesian")
                for i, kpt in enumerate(manual_kpoints):
                    if weights is not None:
                        wt = weights[i]
                    else:
                        wt = 1.0
                    self.input.kpoints.set_value(
                        line=3 + i,
                        val=" ".join([str(kpt[0]), str(kpt[1]), str(kpt[2]), str(wt)]),
                    )

    def _reinit_potential_setter(self, structure):
        if structure is not None:
            self._potential.to_dict().update(
                {
                    el: None
                    for el in set(structure.get_chemical_symbols())
                    if el not in self._potential.to_dict().keys()
                }
            )

    def _get_path_for_kpoints(self, path_name):
        """
        gets the trace for k-points line mode in a VASP readable form.

        Args:
            path_name (str): Name of the path used for band structure calculation from structure instance.

        Returns:
            list: list of tuples of position and path name
        """
        path = self.structure.get_high_symmetry_path()[path_name]

        k_trace = []
        for t in path:
            k_trace.append((self.structure.get_high_symmetry_points()[t[0]], t[0]))
            k_trace.append((self.structure.get_high_symmetry_points()[t[1]], t[1]))

        return k_trace

    def __del__(self):
        pass


class Input:
    """
    Handles setting the input parameters for a VASP job.

    Attributes:
        incar: .vasp.vasp.Incar instance to handle the INCAR file inputs in VASP
        kpoints: vasp.vasp.Kpoints instance to handle the KPOINTS file inputs in VASP
        potcar: vasp.vasp.Potcar instance to set the appropriate POTCAR files for the simulation

    Ideally, the user would not have to access the Input instance unless the user wants to set an extremely specific
    VASP tag which can't se set using functions in Vasp().

    Examples:

        >>> atoms =  CrystalStructure("Pt", BravaisBasis="fcc", a=3.98)
        >>> ham = VaspBase("trial")
        >>> ham.structure = atoms
        >>> ham.calc_static()
        >>> assert(atoms==ham.structure)
        >>> assert(ham.input.incar["ISIF"]==-1)
    """

    def __init__(self):
        self.incar = Incar(table_name="incar")
        self.kpoints = Kpoints(table_name="kpoints")
        self.potcar = Potcar(table_name="potcar")

        # "official" recommendation of VASP devs is to just ignore this warning
        # https://www.vasp.at/forum/viewtopic.php?f=3&t=17822
        self._eddrmm = "ignore"

    def write(self, structure, modified_elements, directory=None):
        """
        Writes all the input files to a specified directory

        Args:
            structure (atomistics.structure.atoms.Atoms instance): Structure to be written
            directory (str): The working directory for the VASP run
        """
        files_to_create_dict = self.get_input_file_dict(
            structure=structure, modified_elements=modified_elements
        )
        for file_name, content in files_to_create_dict.items():
            with open(os.path.join(directory, file_name), "w") as f:
                f.writelines(content)

    def get_input_parameter_dict(
        self, structure: Atoms, modified_elements: list
    ) -> dict:
        """
        Get an hierarchical dictionary of input files. On the first level the dictionary is divided in file_to_create
        and files_to_copy. Both are dictionaries use the file names as keys. In file_to_create the values are strings
        which represent the content which is going to be written to the corresponding file. In files_to_copy the values
        are the paths to the source files to be copied.

        Args:
            structure (Atoms):
            modified_elements (list):

        Returns:
            dict: hierarchical dictionary of input files
        """
        self.potcar.potcar_set_structure(
            structure=structure, modified_elements=modified_elements
        )
        # Write the species info in the POSCAR file only if there are no user defined species
        is_user_defined = list()
        for species in structure.get_species_objects():
            is_user_defined.append(species.Parent is not None)
        do_not_write_species = any(is_user_defined)
        files_to_create = {
            "INCAR": "".join(self.incar.get_string_lst()),
            "POTCAR": "".join(self.potcar.get_file_content()),
            "POSCAR": "".join(
                get_poscar_content(
                    structure=structure,
                    write_species=not do_not_write_species,
                    cartesian=True,
                )
            ),
        }
        if "KSPACING" in self.incar.keys():
            warnings.warn("'KSPACING' found in INCAR, no KPOINTS file written")
        else:
            files_to_create["KPOINTS"] = "".join(self.kpoints.get_string_lst())
        return files_to_create

    def to_dict(self):
        input_dict = {"vasp_dict/eddrmm_handling": self._eddrmm}
        input_dict.update({"incar/" + k: v for k, v in self.incar.to_dict().items()})
        input_dict.update(
            {"kpoints/" + k: v for k, v in self.kpoints.to_dict().items()}
        )
        input_dict.update({"potcar/" + k: v for k, v in self.potcar.to_dict().items()})
        return input_dict

    def from_dict(self, obj_dict):
        self.incar.from_dict(obj_dict=obj_dict["incar"])
        self.kpoints.from_dict(obj_dict=obj_dict["kpoints"])
        self.potcar.from_dict(obj_dict=obj_dict["potcar"])
        self._eddrmm = "ignore"
        if "vasp_dict" in obj_dict.keys():
            vasp_dict = obj_dict["vasp_dict"]
            if "eddrmm_handling" in vasp_dict.keys():
                self._eddrmm = self._eddrmm_backwards_compatibility(
                    vasp_dict["eddrmm_handling"]
                )

    @staticmethod
    def _eddrmm_backwards_compatibility(eddrmm_value):
        """On 9-03-2020, the EDDRMM flag 'not_converged' was switched to 'warn'."""
        if eddrmm_value == "not_converged":
            return "warn"
        else:
            return eddrmm_value


class Incar(GenericParameters):
    """
    Class to control the INCAR file of a vasp simulation
    """

    def __init__(self, input_file_name=None, table_name="incar"):
        super(Incar, self).__init__(
            input_file_name=input_file_name,
            table_name=table_name,
            comment_char="#",
            separator_char="=",
        )
        self._bool_dict = {True: ".TRUE.", False: ".FALSE."}

    def load_default(self):
        """
        Loads the default file content
        """
        file_content = """\
SYSTEM =  ToDo  # jobname
PREC = Accurate
ALGO = Fast
LREAL = False
LWAVE = False
LORBIT = 0
"""
        self.load_string(file_content)

    def _bool_str_to_bool(self, val):
        val = super(Incar, self)._bool_str_to_bool(val)
        extra_bool = {True: "T", False: "F"}
        for key, value in extra_bool.items():
            if val == value:
                return key
        extra_bool = {True: ".True.", False: ".False."}
        for key, value in extra_bool.items():
            if val == value:
                return key
        return val


class Kpoints(GenericParameters):
    """
    Class to control the KPOINTS file of a vasp simulation
    """

    def __init__(self, input_file_name=None, table_name="kpoints"):
        super(Kpoints, self).__init__(
            input_file_name=input_file_name,
            table_name=table_name,
            val_only=True,
            comment_char="!",
        )
        self._path_name = None
        self._n_path = None

    def set_kpoints_file(
        self, method=None, size_of_mesh=None, shift=None, n_path=None, path=None
    ):
        """
        Sets appropriate tags and values in the KPOINTS file
        Args:
            method (str): Type of meshing scheme (Gamma, MP, Manual or Line)
            size_of_mesh (list/numpy.ndarray): List of size 1x3 specifying the required mesh size
            shift (list): List of size 1x3 specifying the user defined shift from the Gamma point
            n_path (int): Number of points per trace for line mode
            path (list): List of tuples including path coorinate and name.
        """
        if n_path is not None:
            if path is None:
                raise ValueError("trace have to be defined")

            self.set_value(line=1, val=n_path)
            self.set_value(line=3, val="rec")

            for i, t in enumerate(path):
                val = " ".join([str(ii) for ii in t[0]])
                val = val + " !" + t[1]
                self.set_value(line=i + 4, val=val)
        if method is not None:
            self.set_value(line=2, val=method)
        if size_of_mesh is not None:
            val = " ".join([str(i) for i in size_of_mesh])
            self.set_value(line=3, val=val)
        if shift is not None:
            val = " ".join([str(i) for i in shift])
            self.set_value(line=4, val=val)

    def load_default(self):
        """
        Loads the default file content
        """
        file_content = """\
Kpoints file generated with pyiron_atomistics
0
Monkhorst_Pack
4 4 4
0 0 0
"""
        self.load_string(file_content)

    def set_kmesh_by_density(self, structure):
        if (
            "density_of_mesh" in self._dataset
            and self._dataset["density_of_mesh"] is not None
        ):
            if self._dataset["density_of_mesh"] != 0.0:
                k_mesh = get_k_mesh_by_cell(
                    structure.get_cell(),
                    kspace_per_in_ang=self._dataset["density_of_mesh"],
                )
                self.set_kpoints_file(size_of_mesh=k_mesh)

    def to_hdf(self, hdf, group_name=None):
        """
        Store the GenericParameters in an HDF5 file

        Args:
            hdf (ProjectHDFio): HDF5 group object
            group_name (str): HDF5 subgroup name - optional
        """
        super(Kpoints, self).to_hdf(hdf=hdf, group_name=group_name)
        if self._path_name is not None:
            line_dict = {"path_name": self._path_name, "n_path": self._n_path}
            with hdf.open("kpoints") as hdf_kpoints:
                hdf_kpoints["line_dict"] = line_dict

    def from_hdf(self, hdf, group_name=None):
        """
        Restore the GenericParameters from an HDF5 file

        Args:
            hdf (ProjectHDFio): HDF5 group object
            group_name (str): HDF5 subgroup name - optional
        """
        super(Kpoints, self).from_hdf(hdf=hdf, group_name=group_name)
        self._path_name = None
        self._n_path = None
        with hdf.open("kpoints") as hdf_kpoints:
            if "line_dict" in hdf_kpoints.list_nodes():
                self._path_name = hdf_kpoints["line_dict"]["path_name"]
                self._n_path = hdf_kpoints["line_dict"]["n_path"]


def get_k_mesh_by_cell(cell, kspace_per_in_ang=0.10):
    """
    Args:
        cell:
        kspace_per_in_ang:
    Returns:
    """
    latlens = [np.linalg.norm(lat) for lat in cell]
    kmesh = np.ceil(np.array([2 * np.pi / ll for ll in latlens]) / kspace_per_in_ang)
    kmesh[kmesh < 1] = 1
    return kmesh
