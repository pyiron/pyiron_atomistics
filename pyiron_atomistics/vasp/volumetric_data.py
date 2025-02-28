# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from pyiron_vasp.vasp.volumetric_data import (
    VaspVolumetricData as _VaspVolumetricData,
)

from pyiron_atomistics.atomistics.structure.atoms import ase_to_pyiron

__author__ = "Sudarsan Surendralal"
__copyright__ = (
    "Copyright 2021, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "1.0"
__maintainer__ = "Sudarsan Surendralal"
__email__ = "surendralal@mpie.de"
__status__ = "production"
__date__ = "Sep 1, 2017"


class VaspVolumetricData(_VaspVolumetricData):
    def to_hdf(self, hdf, group_name="volumetric_data"):
        """
        Writes the data as a group to a HDF5 file

        Args:
            hdf (pyiron_base.generic.hdfio.ProjectHDFio): The HDF file/path to write the data to
            group_name (str): The name of the group under which the data must be stored as

        """
        volumetric_data_dict_to_hdf(
            data_dict=self.to_dict(),
            hdf=hdf,
            group_name=group_name,
        )

    def from_hdf(self, hdf, group_name="volumetric_data"):
        """
        Recreating the VolumetricData instance by reading data from the HDF5 files

        Args:
            hdf (pyiron_base.generic.hdfio.ProjectHDFio): The HDF file/path to write the data to
            group_name (str): The name of the group under which the data must be stored as

        Returns:
            pyiron.atomistics.volumetric.generic.VolumetricData: The VolumetricData instance

        """
        with hdf.open(group_name) as hdf_vd:
            self._total_data = hdf_vd["total"]
            if "diff" in hdf_vd.list_nodes():
                self._diff_data = hdf_vd["diff"]

    def _read_vol_data_old(self, filename, normalize=True):
        """
        Convenience method to parse a generic volumetric static file in the vasp like format.
        Used by subclasses for parsing the file. This routine is adapted from the pymatgen vasp VolumetricData
        class with very minor modifications. The new parser is faster

        http://pymatgen.org/_modules/pymatgen/io/vasp/outputs.html#VolumetricData.

        Args:
            filename (str): Path of file to parse
            normalize (boolean): Flag to normalize by the volume of the cell

        """
        atoms, total_data_list = super()._read_vol_data_old(
            filename=filename, normalize=normalize
        )
        if atoms is not None:
            atoms = ase_to_pyiron(atoms)
        return atoms, total_data_list

    def _read_vol_data(self, filename, normalize=True):
        """
        Parses the VASP volumetric type files (CHGCAR, LOCPOT, PARCHG etc). Rather than looping over individual values,
        this function utilizes numpy indexing resulting in a parsing efficiency of at least 10%.

        Args:
            filename (str): File to be parsed
            normalize (bool): Normalize the data with respect to the volume (Recommended for CHGCAR files)

        Returns:
            pyiron.atomistics.structure.atoms.Atoms: The structure of the volumetric snapshot
            list: A list of the volumetric data (length >1 for CHGCAR files with spin)

        """
        atoms, total_data_list = super()._read_vol_data(
            filename=filename, normalize=normalize
        )
        if atoms is not None:
            atoms = ase_to_pyiron(atoms)
        return atoms, total_data_list


def volumetric_data_dict_to_hdf(data_dict, hdf, group_name="volumetric_data"):
    with hdf.open(group_name) as hdf_vd:
        for k, v in data_dict.items():
            hdf_vd[k] = v
