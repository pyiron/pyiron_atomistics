# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import numpy as np
from pyiron_base import state
import pyiron_atomistics.atomistics.structure.atoms
import pyscal.core as pc
from sklearn import cluster
from pyiron_base.generic.util import Deprecator

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
    atoms,
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
        atoms (Atoms): The structure to analyse.
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
    sys = pyiron_to_pyscal_system(atoms)
    q = (4, 6) if q is None else q
    if clustering == False:
        n_clusters = None

    sys.find_neighbors(method=neighbor_method, cutoff=cutoff)

    sys.calculate_q(q, averaged=averaged)

    sysq = np.array(sys.get_qvals(q, averaged=averaged))

    if n_clusters is not None:
        cl = cluster.KMeans(n_clusters=n_clusters)

        ind = cl.fit(list(zip(*sysq))).labels_
        return sysq, ind
    else:
        return sysq


def analyse_centro_symmetry(atoms, num_neighbors=12):
    """
    Analyse centrosymmetry parameter

    Args:
        atoms: Atoms object
        num_neighbors (int) : number of neighbors

    Returns:
        csm (list) : list of centrosymmetry parameter
    """
    sys = pyiron_to_pyscal_system(atoms)
    return np.array(sys.calculate_centrosymmetry(nmax=num_neighbors))


def analyse_diamond_structure(atoms, mode="total", ovito_compatibility=False):
    """
    Analyse diamond structure

    Args:
        atoms: Atoms object
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
    sys = pyiron_to_pyscal_system(atoms)
    diamond_dict = sys.identify_diamond()

    ovito_identifiers = [
        "Cubic diamond",
        "Cubic diamond (1st neighbor)",
        "Cubic diamond (2nd neighbor)",
        "Hexagonal diamond",
        "Hexagonal diamond (1st neighbor)",
        "Hexagonal diamond (2nd neighbor)",
        "Other",
    ]
    pyscal_identifiers = [
        "others",
        "fcc",
        "hcp",
        "bcc",
        "ico",
        "cubic diamond",
        "cubic diamond 1NN",
        "cubic diamond 2NN",
        "hex diamond",
        "hex diamond 1NN",
        "hex diamond 2NN",
    ]
    convert_to_ovito = {
        0: 6,
        1: 6,
        2: 6,
        3: 6,
        4: 6,
        5: 0,
        6: 1,
        7: 2,
        8: 3,
        9: 4,
        10: 5,
    }

    if mode == "total":
        if not ovito_compatibility:
            return diamond_dict
        else:
            return {
                "IdentifyDiamond.counts.CUBIC_DIAMOND": diamond_dict["cubic diamond"],
                "IdentifyDiamond.counts.CUBIC_DIAMOND_FIRST_NEIGHBOR": diamond_dict[
                    "cubic diamond 1NN"
                ],
                "IdentifyDiamond.counts.CUBIC_DIAMOND_SECOND_NEIGHBOR": diamond_dict[
                    "cubic diamond 2NN"
                ],
                "IdentifyDiamond.counts.HEX_DIAMOND": diamond_dict["hex diamond"],
                "IdentifyDiamond.counts.HEX_DIAMOND_FIRST_NEIGHBOR": diamond_dict[
                    "hex diamond 1NN"
                ],
                "IdentifyDiamond.counts.HEX_DIAMOND_SECOND_NEIGHBOR": diamond_dict[
                    "hex diamond 2NN"
                ],
                "IdentifyDiamond.counts.OTHER": diamond_dict["others"]
                + diamond_dict["fcc"]
                + diamond_dict["hcp"]
                + diamond_dict["bcc"]
                + diamond_dict["ico"],
            }
    elif mode == "numeric":
        if not ovito_compatibility:
            return np.array([atom.structure for atom in sys.atoms])
        else:
            return np.array([convert_to_ovito[atom.structure] for atom in sys.atoms])
    elif mode == "str":
        if not ovito_compatibility:
            return np.array([pyscal_identifiers[atom.structure] for atom in sys.atoms])
        else:
            return np.array(
                [
                    ovito_identifiers[convert_to_ovito[atom.structure]]
                    for atom in sys.atoms
                ]
            )
    else:
        raise ValueError(
            "Only total, str and numeric mode is imported for analyse_diamond_structure()"
        )


def analyse_cna_adaptive(atoms, mode="total", ovito_compatibility=False):
    """
    Use common neighbor analysis

    Args:
        atoms (pyiron_atomistics.structure.atoms.Atoms): The structure to analyze.
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
    sys = pyiron_to_pyscal_system(atoms)
    if mode not in ["total", "numeric", "str"]:
        raise ValueError("Unsupported mode")

    pyscal_parameter = ["others", "fcc", "hcp", "bcc", "ico"]
    ovito_parameter = [
        "CommonNeighborAnalysis.counts.OTHER",
        "CommonNeighborAnalysis.counts.FCC",
        "CommonNeighborAnalysis.counts.HCP",
        "CommonNeighborAnalysis.counts.BCC",
        "CommonNeighborAnalysis.counts.ICO",
    ]

    cna = sys.calculate_cna()

    if mode == "total":
        if not ovito_compatibility:
            return cna
        else:
            return {o: cna[p] for o, p in zip(ovito_parameter, pyscal_parameter)}
    else:
        atoms = sys.atoms
        cnalist = np.array([atom.structure for atom in atoms])
        if mode == "numeric":
            return cnalist
        elif mode == "str":
            if not ovito_compatibility:
                dd = ["others", "fcc", "hcp", "bcc", "ico"]
                return np.array([dd[int(x)] for x in cnalist])
            else:
                dd = ["Other", "FCC", "HCP", "BCC", "ICO"]
                return np.array([dd[int(x)] for x in cnalist])
        else:
            raise ValueError(
                "Only total, str and numeric mode is imported for analyse_cna_adaptive()"
            )


def analyse_voronoi_volume(atoms):
    """
    Calculate the Voronoi volume of atoms

    Args:
        atoms : (pyiron_atomistics.structure.atoms.Atoms): The structure to analyze.
    """
    sys = pyiron_to_pyscal_system(atoms)
    sys.find_neighbors(method="voronoi")
    atoms = sys.atoms
    return np.array([atom.volume for atom in atoms])


def pyiron_to_pyscal_system(atoms):
    """
    Converts atoms to ase atoms and than to a pyscal system.
    Also adds the pyscal publication.

    Args:
        atoms (pyiron atoms): Structure to convert.

    Returns:
        Pyscal system: See the pyscal documentation.
    """
    state.publications.add(publication())
    sys = pc.System()
    sys.read_inputfile(
        pyiron_atomistics.atomistics.structure.atoms.pyiron_to_ase(atoms),
        format="ase",
    )
    return sys


def analyse_find_solids(
    atoms,
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
    sys = pyiron_to_pyscal_system(atoms)
    sys.find_neighbors(method=neighbor_method, cutoff=cutoff)
    sys.find_solids(
        bonds=bonds,
        threshold=threshold,
        avgthreshold=avgthreshold,
        q=q,
        cutoff=cutoff,
        cluster=cluster,
        right=right,
    )
    if return_sys:
        return sys
    atoms = sys.atoms
    solids = [atom for atom in atoms if atom.solid]
    return len(solids)


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
