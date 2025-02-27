from __future__ import print_function

from pyiron_vasp.vasp.volumetric_data import volumetric_data_dict_to_hdf

from pyiron_atomistics.atomistics.structure.atoms import (
    dict_group_to_hdf,
    structure_dict_to_hdf,
)
from pyiron_atomistics.dft.waves.electronic import (
    ElectronicStructure,
    electronic_structure_dict_to_hdf,
)


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
                        self.bands.from_hdf(hdf=hdf_dft, group_name="bands")


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
