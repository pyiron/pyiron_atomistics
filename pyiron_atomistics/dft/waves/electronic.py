# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from __future__ import print_function

import numpy as np
from pyiron_vasp.dft.waves.electronic import ElectronicStructure as _ElectronicStructure

from pyiron_atomistics.atomistics.structure.atoms import (
    Atoms,
    dict_group_to_hdf,
    structure_dict_to_hdf,
)

__author__ = "Sudarsan Surendralal"
__copyright__ = (
    "Copyright 2021, Max-Planck-Institut für Eisenforschung GmbH "
    "- Computational Materials Design (CM) Department"
)
__version__ = "1.0"
__maintainer__ = "Sudarsan Surendralal"
__email__ = "surendralal@mpie.de"
__status__ = "development"
__date__ = "Sep 1, 2017"


def electronic_structure_dict_to_hdf(data_dict, hdf, group_name):
    with hdf.open(group_name) as h_es:
        for k, v in data_dict.items():
            if k not in ["structure", "dos"]:
                h_es[k] = v

        if "structure" in data_dict.keys():
            structure_dict_to_hdf(data_dict=data_dict["structure"], hdf=h_es)

        dict_group_to_hdf(data_dict=data_dict, hdf=h_es, group="dos")


class ElectronicStructure(_ElectronicStructure):
    def to_hdf(self, hdf, group_name="electronic_structure"):
        """
        Store the object to hdf5 file
        Args:
            hdf: Path to the hdf5 file/group in the file
            group_name: Name of the group under which the attributes are o be stored
        """
        electronic_structure_dict_to_hdf(
            data_dict=self.to_dict(), hdf=hdf, group_name=group_name
        )

    def from_hdf(self, hdf, group_name="electronic_structure"):
        """
        Retrieve the object from the hdf5 file
        Args:
            hdf: Path to the hdf5 file/group in the file
            group_name: Name of the group under which the attributes are stored
        """
        if "dos" not in hdf[group_name].list_groups():
            self.from_hdf_old(hdf=hdf, group_name=group_name)
        else:
            with hdf.open(group_name) as h_es:
                if "TYPE" not in h_es.list_nodes():
                    h_es["TYPE"] = str(type(self))
                nodes = h_es.list_nodes()
                if self.structure is not None:
                    self.structure.to_hdf(h_es)
                self.kpoint_list = h_es["k_points"]
                self.kpoint_weights = h_es["k_weights"]
                if len(h_es["eig_matrix"].shape) == 2:
                    self.eigenvalue_matrix = np.array([h_es["eig_matrix"]])
                    self.occupancy_matrix = np.array([h_es["occ_matrix"]])
                else:
                    self._eigenvalue_matrix = h_es["eig_matrix"]
                    self._occupancy_matrix = h_es["occ_matrix"]
                self.n_spins = len(self._eigenvalue_matrix)
                if "efermi" in nodes:
                    self.efermi = h_es["efermi"]
                with h_es.open("dos") as h_dos:
                    nodes = h_dos.list_nodes()
                    self.dos_energies = h_dos["energies"]
                    self.dos_densities = h_dos["tot_densities"]
                    self.dos_idensities = h_dos["int_densities"]
                    if "grand_dos_matrix" in nodes:
                        self.grand_dos_matrix = h_dos["grand_dos_matrix"]
                    if "resolved_densities" in nodes:
                        self.resolved_densities = h_dos["resolved_densities"]
                self._output_dict = h_es.copy()
            self.generate_from_matrices()

    def from_hdf_old(self, hdf, group_name="electronic_structure"):
        """
        Retrieve the object from the hdf5 file
        Args:
            hdf: Path to the hdf5 file/group in the file
            group_name: Name of the group under which the attributes are stored
        """
        with hdf.open(group_name) as h_es:
            if "structure" in h_es.list_nodes():
                self.structure = Atoms().from_hdf(h_es)
            nodes = h_es.list_nodes()
            self.kpoint_list = h_es["k_points"]
            self.kpoint_weights = h_es["k_point_weights"]
            self.eigenvalue_matrix = np.array([h_es["eigenvalue_matrix"]])
            self.occupancy_matrix = np.array([h_es["occupancy_matrix"]])
            try:
                self.dos_energies = h_es["dos_energies"]
                self.dos_densities = h_es["dos_densities"]
                self.dos_idensities = h_es["dos_idensities"]
            except ValueError:
                pass
            if "fermi_level" in nodes:
                self.efermi = h_es["fermi_level"]
            if "grand_dos_matrix" in nodes:
                self.grand_dos_matrix = h_es["grand_dos_matrix"]
            if "resolved_densities" in nodes:
                self.resolved_densities = h_es["resolved_densities"]
            self._output_dict = h_es.copy()
        self.generate_from_matrices()
