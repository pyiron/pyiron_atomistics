# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import numpy as np
import structuretoolkit as stk
from pyiron_base import state
import pyiron_atomistics.atomistics.structure.atoms
from pyiron_base import Deprecator

deprecate = Deprecator()

__author__ = "Sarath Menon, Jan Janssen"
__copyright__ = (
    "Copyright 2021, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "1.0"
__maintainer__ = "Sarath Menon"
__email__ = "sarath.menon@rub.de"
__status__ = "development"
__date__ = "Nov 6, 2019"


@deprecate(arguments={"clustering": "use n_clusters=None instead of clustering=False."})
def get_steinhardt_parameter_structure(
    structure,
    neighbor_method="cutoff",
    cutoff=0,
    n_clusters=2,
    q=None,
    averaged=False,
    clustering=None,
):
    """
    Calculate Steinhardts parameters

    Args:
        structure (Atoms): The structure to analyse.
        neighbor_method (str) : can be ['cutoff', 'voronoi']. (Default is 'cutoff'.)
        cutoff (float) : Can be 0 for adaptive cutoff or any other value. (Default is 0, adaptive.)
        n_clusters (int/None) : Number of clusters for K means clustering or None to not cluster. (Default is 2.)
        q (list) : Values can be integers from 2-12, the required q values to be calculated. (Default is None, which
            uses (4, 6).)
        averaged (bool) : If True, calculates the averaged versions of the parameter. (Default is False.)

    Returns:
        numpy.ndarray: (number of q's, number of atoms) shaped array of q parameters
        numpy.ndarray: If `clustering=True`, an additional per-atom array of cluster ids is also returned
    """
    if clustering == False:
        n_clusters = None
    state.publications.add(publication())
    return stk.analyse.get_steinhardt_parameters(
        structure=structure,
        neighbor_method=neighbor_method,
        cutoff=cutoff,
        n_clusters=n_clusters,
        q=q,
        averaged=averaged,
    )


def analyse_centro_symmetry(structure, num_neighbors=12):
    """
    Analyse centrosymmetry parameter

    Args:
        structure: Atoms object
        num_neighbors (int) : number of neighbors

    Returns:
        csm (list) : list of centrosymmetry parameter
    """
    state.publications.add(publication())
    return stk.analyse.get_centro_symmetry_descriptors(
        structure=structure, num_neighbors=num_neighbors
    )


def analyse_diamond_structure(structure, mode="total", ovito_compatibility=False):
    """
    Analyse diamond structure

    Args:
        structure: Atoms object
        mode ("total"/"numeric"/"str"): Controls the style and level
        of detail of the output.
            - total : return number of atoms belonging to each structure
            - numeric : return a per atom list of numbers- 0 for unknown,
                1 fcc, 2 hcp, 3 bcc and 4 icosa
            - str : return a per atom string of sructures
        ovito_compatibility(bool): use ovito compatiblity mode

    Returns:
        (depends on `mode`)
    """
    state.publications.add(publication())
    return stk.analyse.get_diamond_structure_descriptors(
        structure=structure, mode=mode, ovito_compatibility=ovito_compatibility
    )


def analyse_cna_adaptive(structure, mode="total", ovito_compatibility=False):
    """
    Use common neighbor analysis

    Args:
        structure (pyiron_atomistics.structure.atoms.Atoms): The structure to analyze.
        mode ("total"/"numeric"/"str"): Controls the style and level
            of detail of the output.
            - total : return number of atoms belonging to each structure
            - numeric : return a per atom list of numbers- 0 for unknown,
                1 fcc, 2 hcp, 3 bcc and 4 icosa
            - str : return a per atom string of sructures
        ovito_compatibility(bool): use ovito compatiblity mode

    Returns:
        (depends on `mode`)
    """
    state.publications.add(publication())
    return stk.analyse.get_adaptive_cna_descriptors(
        structure=structure, mode=mode, ovito_compatibility=ovito_compatibility
    )


def analyse_voronoi_volume(structure):
    """
    Calculate the Voronoi volume of atoms

    Args:
        structure : (pyiron_atomistics.structure.atoms.Atoms): The structure to analyze.
    """
    state.publications.add(publication())
    return stk.analyse.get_voronoi_volumes(structure=structure)


def pyiron_to_pyscal_system(structure):
    """
    Converts atoms to ase atoms and than to a pyscal system.
    Also adds the pyscal publication.

    Args:
        structure (pyiron atoms): Structure to convert.

    Returns:
        Pyscal system: See the pyscal documentation.
    """
    state.publications.add(publication())
    return stk.common.ase_to_pyscal(
        pyiron_atomistics.atomistics.structure.atoms.pyiron_to_ase(structure)
    )


def analyse_find_solids(
    structure,
    neighbor_method="cutoff",
    cutoff=0,
    bonds=0.5,
    threshold=0.5,
    avgthreshold=0.6,
    cluster=False,
    q=6,
    right=True,
    return_sys=False,
):
    """
    Get the number of solids or the corresponding pyscal system.
    Calls necessary pyscal methods as described in https://pyscal.org/en/latest/methods/03_solidliquid.html.

    Args:
        neighbor_method (str, optional): Method used to get neighborlist. See pyscal documentation. Defaults to "cutoff".
        cutoff (int, optional): Adaptive if 0. Defaults to 0.
        bonds (float, optional): Number or fraction of bonds to consider atom as solid. Defaults to 0.5.
        threshold (float, optional): See pyscal documentation. Defaults to 0.5.
        avgthreshold (float, optional): See pyscal documentation. Defaults to 0.6.
        cluster (bool, optional): See pyscal documentation. Defaults to False.
        q (int, optional): Steinhard parameter to calculate. Defaults to 6.
        right (bool, optional): See pyscal documentation. Defaults to True.
        return_sys (bool, optional): Whether to return number of solid atoms or pyscal system. Defaults to False.

    Returns:
        int: number of solids,
        pyscal system: pyscal system when return_sys=True
    """
    state.publications.add(publication())
    return stk.analyse.find_solids(
        structure=structure,
        neighbor_method=neighbor_method,
        cutoff=cutoff,
        bonds=bonds,
        threshold=threshold,
        avgthreshold=avgthreshold,
        cluster=cluster,
        q=q,
        right=right,
        return_sys=return_sys,
    )


def publication():
    return {
        "pyscal": {
            "Menon2019": {
                "doi": "10.21105/joss.01824",
                "url": "https://doi.org/10.21105/joss.01824",
                "year": "2019",
                "volume": "4",
                "number": "43",
                "pages": "1824",
                "author": ["Sarath Menon", "Grisell Diaz Leines", "Jutta Rogal"],
                "title": "pyscal: A python module for structural analysis of atomic environments",
                "journal": "Journal of Open Source Software",
            }
        }
    }
