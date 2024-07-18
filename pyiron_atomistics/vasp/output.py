from __future__ import print_function

import os
import posixpath
import warnings

import numpy as np
from pyiron_base import state

from pyiron_atomistics.atomistics.structure.atoms import (
    Atoms,
    dict_group_to_hdf,
    structure_dict_to_hdf,
)
from pyiron_atomistics.dft.bader import Bader
from pyiron_atomistics.dft.waves.electronic import (
    ElectronicStructure,
    electronic_structure_dict_to_hdf,
)
from pyiron_atomistics.vasp.parser.oszicar import Oszicar
from pyiron_atomistics.vasp.parser.outcar import Outcar, OutcarCollectError
from pyiron_atomistics.vasp.procar import Procar
from pyiron_atomistics.vasp.structure import read_atoms, vasp_sorter
from pyiron_atomistics.vasp.vasprun import Vasprun as Vr
from pyiron_atomistics.vasp.vasprun import VasprunError, VasprunWarning
from pyiron_atomistics.vasp.volumetric_data import (
    VaspVolumetricData,
    volumetric_data_dict_to_hdf,
)


class Output:
    """
    Handles the output from a VASP simulation.

    Attributes:
        electronic_structure: Gives the electronic structure of the system
        electrostatic_potential: Gives the electrostatic/local potential of the system
        charge_density: Gives the charge density of the system
    """

    def __init__(self):
        self._structure = None
        self.outcar = Outcar()
        self.oszicar = Oszicar()
        self.generic_output = GenericOutput()
        self.dft_output = DFTOutput()
        self.description = (
            "This contains all the output static from this particular vasp run"
        )
        self.charge_density = VaspVolumetricData()
        self.electrostatic_potential = VaspVolumetricData()
        self.procar = Procar()
        self.electronic_structure = ElectronicStructure()
        self.vp_new = Vr()

    @property
    def structure(self):
        """
        Getter for the output structure
        """
        return self._structure

    @structure.setter
    def structure(self, atoms):
        """
        Setter for the output structure
        """
        self._structure = atoms

    def collect(self, directory=os.getcwd(), sorted_indices=None):
        """
        Collects output from the working directory

        Args:
            directory (str): Path to the directory
            sorted_indices (np.array/None):
        """
        if sorted_indices is None:
            sorted_indices = vasp_sorter(self.structure)
        files_present = os.listdir(directory)
        log_dict = dict()
        vasprun_working, outcar_working = False, False
        if not ("OUTCAR" in files_present or "vasprun.xml" in files_present):
            raise IOError("Either the OUTCAR or vasprun.xml files need to be present")
        if "OSZICAR" in files_present:
            self.oszicar.from_file(filename=posixpath.join(directory, "OSZICAR"))
        if "OUTCAR" in files_present:
            try:
                self.outcar.from_file(filename=posixpath.join(directory, "OUTCAR"))
                outcar_working = True
            except OutcarCollectError as e:
                state.logger.warning(f"OUTCAR present, but could not be parsed: {e}!")
                outcar_working = False
        if "vasprun.xml" in files_present:
            try:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    self.vp_new.from_file(
                        filename=posixpath.join(directory, "vasprun.xml")
                    )
                    if any([isinstance(warn.category, VasprunWarning) for warn in w]):
                        state.logger.warning(
                            "vasprun.xml parsed but with some inconsistencies. "
                            "Check vasp output to be sure"
                        )
                        warnings.warn(
                            "vasprun.xml parsed but with some inconsistencies. "
                            "Check vasp output to be sure",
                            VasprunWarning,
                        )
            except VasprunError:
                state.logger.warning(
                    "Unable to parse the vasprun.xml file. Will attempt to get data from OUTCAR"
                )
            else:
                # If parsing the vasprun file does not throw an error, then set to True
                vasprun_working = True
        if outcar_working:
            log_dict["temperature"] = self.outcar.parse_dict["temperatures"]
            log_dict["stresses"] = self.outcar.parse_dict["stresses"]
            log_dict["pressures"] = self.outcar.parse_dict["pressures"]
            log_dict["elastic_constants"] = self.outcar.parse_dict["elastic_constants"]
            self.generic_output.dft_log_dict["n_elect"] = self.outcar.parse_dict[
                "n_elect"
            ]
            if len(self.outcar.parse_dict["magnetization"]) > 0:
                magnetization = np.array(
                    self.outcar.parse_dict["magnetization"], dtype=object
                )
                final_magmoms = np.array(
                    self.outcar.parse_dict["final_magmoms"], dtype=object
                )
                # magnetization[sorted_indices] = magnetization.copy()
                if len(final_magmoms) != 0:
                    if len(final_magmoms.shape) == 3:
                        final_magmoms[:, sorted_indices, :] = final_magmoms.copy()
                    else:
                        final_magmoms[:, sorted_indices] = final_magmoms.copy()
                self.generic_output.dft_log_dict["magnetization"] = (
                    magnetization.tolist()
                )
                self.generic_output.dft_log_dict["final_magmoms"] = (
                    final_magmoms.tolist()
                )
            self.generic_output.dft_log_dict["e_fermi_list"] = self.outcar.parse_dict[
                "e_fermi_list"
            ]
            self.generic_output.dft_log_dict["vbm_list"] = self.outcar.parse_dict[
                "vbm_list"
            ]
            self.generic_output.dft_log_dict["cbm_list"] = self.outcar.parse_dict[
                "cbm_list"
            ]
            self.generic_output.dft_log_dict["ediel_sol"] = self.outcar.parse_dict[
                "ediel_sol"
            ]

        if vasprun_working:
            log_dict["forces"] = self.vp_new.vasprun_dict["forces"]
            log_dict["cells"] = self.vp_new.vasprun_dict["cells"]
            log_dict["volume"] = np.linalg.det(self.vp_new.vasprun_dict["cells"])
            # The vasprun parser also returns the energies printed again after the final SCF cycle under the key
            # "total_energies", but due to a bug in the VASP output, the energies reported there are wrong in Vasp 5.*;
            # instead use the last energy from the scf cycle energies
            # BUG link: https://ww.vasp.at/forum/viewtopic.php?p=19242
            try:
                # bug report is not specific to which Vasp5 versions are affected; be safe and workaround for all of
                # them
                is_vasp5 = self.vp_new.vasprun_dict["generator"]["version"].startswith(
                    "5."
                )
            except KeyError:  # in case the parser didn't read the version info
                is_vasp5 = True
            if is_vasp5:
                log_dict["energy_pot"] = np.array(
                    [e[-1] for e in self.vp_new.vasprun_dict["scf_fr_energies"]]
                )
            else:
                # total energies refers here to the total energy of the electronic system, not the total system of
                # electrons plus (potentially) moving ions; hence this is the energy_pot
                log_dict["energy_pot"] = self.vp_new.vasprun_dict["total_fr_energies"]
            if "kinetic_energies" in self.vp_new.vasprun_dict.keys():
                log_dict["energy_tot"] = (
                    log_dict["energy_pot"]
                    + self.vp_new.vasprun_dict["kinetic_energies"]
                )
            else:
                log_dict["energy_tot"] = log_dict["energy_pot"]
            log_dict["steps"] = np.arange(len(log_dict["energy_tot"]))
            log_dict["positions"] = self.vp_new.vasprun_dict["positions"]
            log_dict["forces"][:, sorted_indices] = log_dict["forces"].copy()
            log_dict["positions"][:, sorted_indices] = log_dict["positions"].copy()
            log_dict["positions"] = np.einsum(
                "nij,njk->nik", log_dict["positions"], log_dict["cells"]
            )
            # log_dict["scf_energies"] = self.vp_new.vasprun_dict["scf_energies"]
            # log_dict["scf_dipole_moments"] = self.vp_new.vasprun_dict["scf_dipole_moments"]
            self.electronic_structure = self.vp_new.get_electronic_structure()
            if self.electronic_structure.grand_dos_matrix is not None:
                self.electronic_structure.grand_dos_matrix[
                    :, :, :, sorted_indices, :
                ] = self.electronic_structure.grand_dos_matrix[:, :, :, :, :].copy()
            if self.electronic_structure.resolved_densities is not None:
                self.electronic_structure.resolved_densities[
                    :, sorted_indices, :, :
                ] = self.electronic_structure.resolved_densities[:, :, :, :].copy()
            self.structure.positions = log_dict["positions"][-1]
            self.structure.set_cell(log_dict["cells"][-1])
            self.generic_output.dft_log_dict["potentiostat_output"] = (
                self.vp_new.get_potentiostat_output()
            )
            valence_charges_orig = self.vp_new.get_valence_electrons_per_atom()
            valence_charges = valence_charges_orig.copy()
            valence_charges[sorted_indices] = valence_charges_orig
            self.generic_output.dft_log_dict["valence_charges"] = valence_charges

        elif outcar_working:
            # log_dict = self.outcar.parse_dict.copy()
            if len(self.outcar.parse_dict["energies"]) == 0:
                raise VaspCollectError("Error in parsing OUTCAR")
            log_dict["energy_tot"] = self.outcar.parse_dict["energies"]
            log_dict["temperature"] = self.outcar.parse_dict["temperatures"]
            log_dict["stresses"] = self.outcar.parse_dict["stresses"]
            log_dict["pressures"] = self.outcar.parse_dict["pressures"]
            log_dict["forces"] = self.outcar.parse_dict["forces"]
            log_dict["positions"] = self.outcar.parse_dict["positions"]
            log_dict["forces"][:, sorted_indices] = log_dict["forces"].copy()
            log_dict["positions"][:, sorted_indices] = log_dict["positions"].copy()
            if len(log_dict["positions"].shape) != 3:
                raise VaspCollectError("Improper OUTCAR parsing")
            elif log_dict["positions"].shape[1] != len(sorted_indices):
                raise VaspCollectError("Improper OUTCAR parsing")
            if len(log_dict["forces"].shape) != 3:
                raise VaspCollectError("Improper OUTCAR parsing")
            elif log_dict["forces"].shape[1] != len(sorted_indices):
                raise VaspCollectError("Improper OUTCAR parsing")
            log_dict["time"] = self.outcar.parse_dict["time"]
            log_dict["steps"] = self.outcar.parse_dict["steps"]
            log_dict["cells"] = self.outcar.parse_dict["cells"]
            log_dict["volume"] = np.array(
                [np.linalg.det(cell) for cell in self.outcar.parse_dict["cells"]]
            )
            self.generic_output.dft_log_dict["scf_energy_free"] = (
                self.outcar.parse_dict["scf_energies"]
            )
            self.generic_output.dft_log_dict["scf_dipole_mom"] = self.outcar.parse_dict[
                "scf_dipole_moments"
            ]
            self.generic_output.dft_log_dict["n_elect"] = self.outcar.parse_dict[
                "n_elect"
            ]
            self.generic_output.dft_log_dict["energy_int"] = self.outcar.parse_dict[
                "energies_int"
            ]
            self.generic_output.dft_log_dict["energy_free"] = self.outcar.parse_dict[
                "energies"
            ]
            self.generic_output.dft_log_dict["energy_zero"] = self.outcar.parse_dict[
                "energies_zero"
            ]
            self.generic_output.dft_log_dict["energy_int"] = self.outcar.parse_dict[
                "energies_int"
            ]
            if "PROCAR" in files_present:
                try:
                    self.electronic_structure = self.procar.from_file(
                        filename=posixpath.join(directory, "PROCAR")
                    )
                    #  Even the atom resolved values have to be sorted from the vasp atoms order to the Atoms order
                    self.electronic_structure.grand_dos_matrix[
                        :, :, :, sorted_indices, :
                    ] = self.electronic_structure.grand_dos_matrix[:, :, :, :, :].copy()
                    try:
                        self.electronic_structure.efermi = self.outcar.parse_dict[
                            "fermi_level"
                        ]
                    except KeyError:
                        self.electronic_structure.efermi = self.vp_new.vasprun_dict[
                            "efermi"
                        ]
                except ValueError:
                    pass
        # important that we "reverse sort" the atoms in the vasp format into the atoms in the atoms class
        self.generic_output.log_dict = log_dict
        if vasprun_working:
            # self.dft_output.log_dict["parameters"] = self.vp_new.vasprun_dict["parameters"]
            self.generic_output.dft_log_dict["scf_dipole_mom"] = (
                self.vp_new.vasprun_dict["scf_dipole_moments"]
            )
            if len(self.generic_output.dft_log_dict["scf_dipole_mom"][0]) > 0:
                total_dipole_moments = np.array(
                    [
                        dip[-1]
                        for dip in self.generic_output.dft_log_dict["scf_dipole_mom"]
                    ]
                )
                self.generic_output.dft_log_dict["dipole_mom"] = total_dipole_moments
            self.generic_output.dft_log_dict["scf_energy_int"] = (
                self.vp_new.vasprun_dict["scf_energies"]
            )
            self.generic_output.dft_log_dict["scf_energy_free"] = (
                self.vp_new.vasprun_dict["scf_fr_energies"]
            )
            self.generic_output.dft_log_dict["scf_energy_zero"] = (
                self.vp_new.vasprun_dict["scf_0_energies"]
            )
            self.generic_output.dft_log_dict["energy_int"] = np.array(
                [
                    e_int[-1]
                    for e_int in self.generic_output.dft_log_dict["scf_energy_int"]
                ]
            )
            self.generic_output.dft_log_dict["energy_free"] = np.array(
                [
                    e_free[-1]
                    for e_free in self.generic_output.dft_log_dict["scf_energy_free"]
                ]
            )
            # Overwrite energy_free with much better precision from the OSZICAR file
            if "energy_pot" in self.oszicar.parse_dict.keys():
                if np.array_equal(
                    self.generic_output.dft_log_dict["energy_free"],
                    np.round(self.oszicar.parse_dict["energy_pot"], 8),
                ):
                    self.generic_output.dft_log_dict["energy_free"] = (
                        self.oszicar.parse_dict["energy_pot"]
                    )
            self.generic_output.dft_log_dict["energy_zero"] = np.array(
                [
                    e_zero[-1]
                    for e_zero in self.generic_output.dft_log_dict["scf_energy_zero"]
                ]
            )
            self.generic_output.dft_log_dict["n_elect"] = float(
                self.vp_new.vasprun_dict["parameters"]["electronic"]["NELECT"]
            )
            if "kinetic_energies" in self.vp_new.vasprun_dict.keys():
                # scf_energy_kin is for backwards compatibility
                self.generic_output.dft_log_dict["scf_energy_kin"] = (
                    self.vp_new.vasprun_dict["kinetic_energies"]
                )
                self.generic_output.dft_log_dict["energy_kin"] = (
                    self.vp_new.vasprun_dict["kinetic_energies"]
                )

        if (
            "LOCPOT" in files_present
            and os.stat(posixpath.join(directory, "LOCPOT")).st_size != 0
        ):
            self.electrostatic_potential.from_file(
                filename=posixpath.join(directory, "LOCPOT"), normalize=False
            )
        if (
            "CHGCAR" in files_present
            and os.stat(posixpath.join(directory, "CHGCAR")).st_size != 0
        ):
            self.charge_density.from_file(
                filename=posixpath.join(directory, "CHGCAR"), normalize=True
            )
        self.generic_output.bands = self.electronic_structure

    def to_dict(self):
        hdf5_output = {
            "description": self.description,
            "generic": self.generic_output.to_dict(),
        }

        if self._structure is not None:
            hdf5_output["structure"] = self.structure.to_dict()

        if self.electrostatic_potential.total_data is not None:
            hdf5_output["electrostatic_potential"] = (
                self.electrostatic_potential.to_dict()
            )

        if self.charge_density.total_data is not None:
            hdf5_output["charge_density"] = self.charge_density.to_dict()

        if len(self.electronic_structure.kpoint_list) > 0:
            hdf5_output["electronic_structure"] = self.electronic_structure.to_dict()

        if len(self.outcar.parse_dict.keys()) > 0:
            hdf5_output["outcar"] = self.outcar.to_dict_minimal()
        return hdf5_output

    def to_hdf(self, hdf):
        """
        Save the object in a HDF5 file

        Args:
            hdf (pyiron_base.generic.hdfio.ProjectHDFio): HDF path to which the object is to be saved

        """
        output_dict_to_hdf(data_dict=self.to_dict(), hdf=hdf, group_name="output")

    def from_hdf(self, hdf):
        """
        Reads the attributes and reconstructs the object from a hdf file
        Args:
            hdf: The hdf5 instance
        """
        with hdf.open("output") as hdf5_output:
            # self.description = hdf5_output["description"]
            if self.structure is None:
                self.structure = Atoms()
            self.structure.from_hdf(hdf5_output)
            self.generic_output.from_hdf(hdf5_output)
            try:
                if "electrostatic_potential" in hdf5_output.list_groups():
                    self.electrostatic_potential.from_hdf(
                        hdf5_output, group_name="electrostatic_potential"
                    )
                if "charge_density" in hdf5_output.list_groups():
                    self.charge_density.from_hdf(
                        hdf5_output, group_name="charge_density"
                    )
                if "electronic_structure" in hdf5_output.list_groups():
                    self.electronic_structure.from_hdf(hdf=hdf5_output)
                if "outcar" in hdf5_output.list_groups():
                    self.outcar.from_hdf(hdf=hdf5_output, group_name="outcar")
            except (TypeError, IOError, ValueError):
                state.logger.warning("Routine from_hdf() not completely successful")


class GenericOutput:
    """

    This class stores the generic output like different structures, energies and forces from a simulation in a highly
    generic format. Usually the user does not have to access this class.

    Attributes:
        log_dict (dict): A dictionary of all tags and values of generic data (positions, forces, etc)
    """

    def __init__(self):
        self.log_dict = dict()
        self.dft_log_dict = dict()
        self.description = "generic_output contains generic output static"
        self._bands = ElectronicStructure()

    @property
    def bands(self):
        return self._bands

    @bands.setter
    def bands(self, val):
        self._bands = val

    def to_hdf(self, hdf):
        """
        Save the object in a HDF5 file

        Args:
            hdf (pyiron_base.generic.hdfio.ProjectHDFio): HDF path to which the object is to be saved

        """
        generic_output_dict_to_hdf(
            data_dict=self.to_dict(), hdf=hdf, group_name="generic"
        )

    def to_dict(self):
        hdf_go, hdf_dft = {}, {}
        for key, val in self.log_dict.items():
            hdf_go[key] = val
        for key, val in self.dft_log_dict.items():
            hdf_dft[key] = val
        hdf_go["dft"] = hdf_dft
        if self.bands.eigenvalue_matrix is not None:
            hdf_go["dft"]["bands"] = self.bands.to_dict()
        return hdf_go

    def from_hdf(self, hdf):
        """
        Reads the attributes and reconstructs the object from a hdf file
        Args:
            hdf: The hdf5 instance
        """
        with hdf.open("generic") as hdf_go:
            for node in hdf_go.list_nodes():
                if node == "description":
                    # self.description = hdf_go[node]
                    pass
                else:
                    self.log_dict[node] = hdf_go[node]
            if "dft" in hdf_go.list_groups():
                with hdf_go.open("dft") as hdf_dft:
                    for node in hdf_dft.list_nodes():
                        self.dft_log_dict[node] = hdf_dft[node]
                    if "bands" in hdf_dft.list_groups():
                        self.bands.from_hdf(hdf_dft, "bands")


class DFTOutput:
    """
    This class stores the DFT specific output

    Attributes:
        log_dict (dict): A dictionary of all tags and values of DFT data
    """

    def __init__(self):
        self.log_dict = dict()
        self.description = "contains DFT specific output"

    def to_hdf(self, hdf):
        """
        Save the object in a HDF5 file

        Args:
            hdf (pyiron_base.generic.hdfio.ProjectHDFio): HDF path to which the object is to be saved

        """
        with hdf.open("dft") as hdf_dft:
            # hdf_go["description"] = self.description
            for key, val in self.log_dict.items():
                hdf_dft[key] = val

    def from_hdf(self, hdf):
        """
        Reads the attributes and reconstructs the object from a hdf file
        Args:
            hdf: The hdf5 instance
        """
        with hdf.open("dft") as hdf_dft:
            for node in hdf_dft.list_nodes():
                if node == "description":
                    # self.description = hdf_go[node]
                    pass
                else:
                    self.log_dict[node] = hdf_dft[node]


class VaspCollectError(ValueError):
    pass


def generic_output_dict_to_hdf(data_dict, hdf, group_name="generic"):
    with hdf.open(group_name) as hdf_go:
        for k, v in data_dict.items():
            if k not in ["dft"]:
                hdf_go[k] = v

        with hdf_go.open("dft") as hdf_dft:
            for k, v in data_dict["dft"].items():
                if k not in ["bands"]:
                    hdf_dft[k] = v

            if "bands" in data_dict["dft"].keys():
                electronic_structure_dict_to_hdf(
                    data_dict=data_dict["dft"]["bands"],
                    hdf=hdf_dft,
                    group_name="bands",
                )


def output_dict_to_hdf(data_dict, hdf, group_name="output"):
    with hdf.open(group_name) as hdf5_output:
        for k, v in data_dict.items():
            if k not in [
                "generic",
                "structure",
                "electrostatic_potential",
                "charge_density",
                "electronic_structure",
                "outcar",
            ]:
                hdf5_output[k] = v

        if "generic" in data_dict.keys():
            generic_output_dict_to_hdf(
                data_dict=data_dict["generic"],
                hdf=hdf5_output,
                group_name="generic",
            )

        if "structure" in data_dict.keys():
            structure_dict_to_hdf(
                data_dict=data_dict["structure"],
                hdf=hdf5_output,
                group_name="structure",
            )

        if "electrostatic_potential" in data_dict.keys():
            volumetric_data_dict_to_hdf(
                data_dict=data_dict["electrostatic_potential"],
                hdf=hdf5_output,
                group_name="electrostatic_potential",
            )

        if "charge_density" in data_dict.keys():
            volumetric_data_dict_to_hdf(
                data_dict=data_dict["charge_density"],
                hdf=hdf5_output,
                group_name="charge_density",
            )

        if "electronic_structure" in data_dict.keys():
            electronic_structure_dict_to_hdf(
                data_dict=data_dict["electronic_structure"],
                hdf=hdf5_output,
                group_name="electronic_structure",
            )

        dict_group_to_hdf(data_dict=data_dict, hdf=hdf5_output, group="outcar")


def parse_vasp_output(
    working_directory: str, structure: Atoms = None, sorted_indices: list = None
) -> dict:
    """
    Parse the VASP output in the working_directory and return it as hierachical dictionary.

    Args:
        working_directory (str): directory of the VASP calculation
        structure (Atoms): atomistic structure as optional input for matching the output to the input of the calculation
        sorted_indices (list): list of indices used to sort the atomistic structure

    Returns:
        dict: hierarchical output dictionary
    """
    output_parser = Output()
    if structure is None or len(structure) == 0:
        try:
            structure = get_final_structure_from_file(
                working_directory=working_directory, filename="CONTCAR"
            )
        except IOError:
            structure = get_final_structure_from_file(
                working_directory=working_directory, filename="POSCAR"
            )
    if sorted_indices is None:
        sorted_indices = np.array(range(len(structure)))
    output_parser.structure = structure.copy()
    try:
        output_parser.collect(
            directory=working_directory, sorted_indices=sorted_indices
        )
    except VaspCollectError:
        raise
    # Try getting high precision positions from CONTCAR
    try:
        output_parser.structure = get_final_structure_from_file(
            working_directory=working_directory,
            filename="CONTCAR",
            structure=structure,
            sorted_indices=sorted_indices,
        )
    except (IOError, ValueError, FileNotFoundError):
        pass

    # Bader analysis
    if os.path.isfile(os.path.join(working_directory, "AECCAR0")) and os.path.isfile(
        os.path.join(working_directory, "AECCAR2")
    ):
        bader = Bader(working_directory=working_directory, structure=structure)
        try:
            charges_orig, volumes_orig = bader.compute_bader_charges()
        except ValueError:
            warnings.warn("Invoking Bader charge analysis failed")
        else:
            charges, volumes = charges_orig.copy(), volumes_orig.copy()
            charges[sorted_indices] = charges_orig
            volumes[sorted_indices] = volumes_orig
            if "valence_charges" in output_parser.generic_output.dft_log_dict.keys():
                valence_charges = output_parser.generic_output.dft_log_dict[
                    "valence_charges"
                ]
                # Positive values indicate electron depletion
                output_parser.generic_output.dft_log_dict["bader_charges"] = (
                    valence_charges - charges
                )
            output_parser.generic_output.dft_log_dict["bader_volumes"] = volumes
    return output_parser.to_dict()


def get_final_structure_from_file(
    working_directory, filename="CONTCAR", structure=None, sorted_indices=None
):
    """
    Get the final structure of the simulation usually from the CONTCAR file

    Args:
        filename (str): Path to the CONTCAR file in VASP

    Returns:
        pyiron.atomistics.structure.atoms.Atoms: The final structure
    """
    filename = posixpath.join(working_directory, filename)
    if structure is not None and sorted_indices is None:
        sorted_indices = vasp_sorter(structure)
    if structure is None:
        try:
            output_structure = read_atoms(filename=filename)
            input_structure = output_structure.copy()
        except (IndexError, ValueError, IOError):
            raise IOError("Unable to read output structure")
    else:
        input_structure = structure.copy()
        try:
            output_structure = read_atoms(
                filename=filename,
                species_list=input_structure.get_parent_symbols(),
            )
            input_structure.cell = output_structure.cell.copy()
            input_structure.positions[sorted_indices] = output_structure.positions
        except (IndexError, ValueError, IOError):
            raise IOError("Unable to read output structure")
    return input_structure
