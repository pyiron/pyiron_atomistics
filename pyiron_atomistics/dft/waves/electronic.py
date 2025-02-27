# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from __future__ import print_function

import numpy as np
from pyiron_vasp.dft.waves.dos import Dos

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


def from_hdf(es, hdf, group_name="electronic_structure"):
    """
    Retrieve the object from the hdf5 file
    Args:
        hdf: Path to the hdf5 file/group in the file
        group_name: Name of the group under which the attributes are stored
    """
    if "dos" not in hdf[group_name].list_groups():
        from_hdf_old(es=es, hdf=hdf, group_name=group_name)
    else:
        with hdf.open(group_name) as h_es:
            if "TYPE" not in h_es.list_nodes():
                h_es["TYPE"] = str(type(es))
            nodes = h_es.list_nodes()
            if es.structure is not None:
                es.structure.to_hdf(h_es)
            es.kpoint_list = h_es["k_points"]
            es.kpoint_weights = h_es["k_weights"]
            if len(h_es["eig_matrix"].shape) == 2:
                es.eigenvalue_matrix = np.array([h_es["eig_matrix"]])
                es.occupancy_matrix = np.array([h_es["occ_matrix"]])
            else:
                es._eigenvalue_matrix = h_es["eig_matrix"]
                es._occupancy_matrix = h_es["occ_matrix"]
            es.n_spins = len(es._eigenvalue_matrix)
            if "efermi" in nodes:
                es.efermi = h_es["efermi"]
            with h_es.open("dos") as h_dos:
                nodes = h_dos.list_nodes()
                es.dos_energies = h_dos["energies"]
                es.dos_densities = h_dos["tot_densities"]
                es.dos_idensities = h_dos["int_densities"]
                if "grand_dos_matrix" in nodes:
                    es.grand_dos_matrix = h_dos["grand_dos_matrix"]
                if "resolved_densities" in nodes:
                    es.resolved_densities = h_dos["resolved_densities"]
            es._output_dict = h_es.copy()
        es.generate_from_matrices()


def from_hdf_old(es, hdf, group_name="electronic_structure"):
    """
    Retrieve the object from the hdf5 file
    Args:
        hdf: Path to the hdf5 file/group in the file
        group_name: Name of the group under which the attributes are stored
    """
    with hdf.open(group_name) as h_es:
        if "structure" in h_es.list_nodes():
            es.structure = Atoms().from_hdf(h_es)
        nodes = h_es.list_nodes()
        es.kpoint_list = h_es["k_points"]
        es.kpoint_weights = h_es["k_point_weights"]
        es.eigenvalue_matrix = np.array([h_es["eigenvalue_matrix"]])
        es.occupancy_matrix = np.array([h_es["occupancy_matrix"]])
        try:
            es.dos_energies = h_es["dos_energies"]
            es.dos_densities = h_es["dos_densities"]
            es.dos_idensities = h_es["dos_idensities"]
        except ValueError:
            pass
        if "fermi_level" in nodes:
            es.efermi = h_es["fermi_level"]
        if "grand_dos_matrix" in nodes:
            es.grand_dos_matrix = h_es["grand_dos_matrix"]
        if "resolved_densities" in nodes:
            es.resolved_densities = h_es["resolved_densities"]
        es._output_dict = h_es.copy()
    es.generate_from_matrices()