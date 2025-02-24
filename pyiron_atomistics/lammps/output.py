from __future__ import annotations

from typing import TYPE_CHECKING, List, Union

import numpy as np

if TYPE_CHECKING:
    from pyiron_atomistics.atomistics.structure.atoms import Atoms


def remap_indices(
    lammps_indices: Union[np.ndarray, List],
    potential_elements: Union[np.ndarray, List],
    structure: Atoms,
) -> np.ndarray:
    """
    Give the Lammps-dumped indices, re-maps these back onto the structure's indices to preserve the species.

    The issue is that for an N-element potential, Lammps dumps the chemical index from 1 to N based on the order
    that these species are written in the Lammps input file. But the indices for a given structure are based on the
    order in which chemical species were added to that structure, and run from 0 up to the number of species
    currently in that structure. Therefore we need to be a little careful with mapping.

    Args:
        lammps_indices (numpy.ndarray/list): The Lammps-dumped integers.
        potential_elements (numpy.ndarray/list):
        structure (pyiron_atomistics.atomistics.structure.Atoms):

    Returns:
        numpy.ndarray: Those integers mapped onto the structure.
    """
    lammps_symbol_order = np.array(potential_elements)

    # If new Lammps indices are present for which we have no species, extend the species list
    unique_lammps_indices = np.unique(lammps_indices)
    if len(unique_lammps_indices) > len(np.unique(structure.indices)):
        unique_lammps_indices -= (
            1  # Convert from Lammps start counting at 1 to python start counting at 0
        )
        new_lammps_symbols = lammps_symbol_order[unique_lammps_indices]
        structure.set_species(
            [structure.convert_element(el) for el in new_lammps_symbols]
        )

    # Create a map between the lammps indices and structure indices to preserve species
    structure_symbol_order = np.array([el.Abbreviation for el in structure.species])
    map_ = np.array(
        [
            int(np.argwhere(lammps_symbol_order == symbol)[0]) + 1
            for symbol in structure_symbol_order
        ]
    )

    structure_indices = np.array(lammps_indices)
    for i_struct, i_lammps in enumerate(map_):
        np.place(structure_indices, lammps_indices == i_lammps, i_struct)
    # TODO: Vectorize this for-loop for computational efficiency

    return structure_indices
