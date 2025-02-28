# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

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

from pyiron_vasp.vasp.structure import (
    atoms_from_string as _atoms_from_string,
)
from pyiron_vasp.vasp.structure import (
    get_poscar_content as _get_poscar_content,
)
from pyiron_vasp.vasp.structure import (
    read_atoms as _read_atoms,
)
from pyiron_vasp.vasp.structure import (
    write_poscar as _write_poscar,
)

from pyiron_atomistics.atomistics.structure.atoms import ase_to_pyiron, pyiron_to_ase


def read_atoms(
    filename="CONTCAR",
    return_velocities=False,
    species_list=None,
    species_from_potcar=False,
):
    """
    Routine to read structural static from a POSCAR type file

    Args:
        filename (str): Input filename
        return_velocities (bool): True if the predictor corrector velocities are read (only from MD output)
        species_list (list/numpy.ndarray): A list of the species (if not present in the POSCAR file or a POTCAR in the
        same directory)
        species_from_potcar (bool): True if the species list should be read from the POTCAR file in the same directory

    Returns:
        pyiron.atomistics.structure.atoms.Atoms: The generated structure object

    """
    if return_velocities:
        atoms, velocities = _read_atoms(
            filename=filename,
            return_velocities=return_velocities,
            species_list=species_list,
            species_from_potcar=species_from_potcar,
        )
        return ase_to_pyiron(atoms), velocities
    else:
        atoms = _read_atoms(
            filename=filename,
            return_velocities=return_velocities,
            species_list=species_list,
            species_from_potcar=species_from_potcar,
        )
        return ase_to_pyiron(atoms)


def get_poscar_content(structure, write_species=True, cartesian=True):
    return _get_poscar_content(
        pyiron_to_ase(structure), write_species=write_species, cartesian=cartesian
    )


def atoms_from_string(string, read_velocities=False, species_list=None):
    """
    Routine to convert a string list read from a input/output structure file and convert into Atoms instance

    Args:
        string (list): A list of strings (lines) read from the POSCAR/CONTCAR/CHGCAR/LOCPOT file
        read_velocities (bool): True if the velocities from a CONTCAR file should be read (predictor corrector)
        species_list (list/numpy.ndarray): A list of species of the atoms

    Returns:
        pyiron.atomistics.structure.atoms.Atoms: The required structure object

    """
    if read_velocities:
        atoms, velocities = _atoms_from_string(
            string=string, read_velocities=read_velocities, species_list=species_list
        )
        return ase_to_pyiron(atoms), velocities
    else:
        return ase_to_pyiron(
            _atoms_from_string(
                string=string,
                read_velocities=read_velocities,
                species_list=species_list,
            )
        )


def write_poscar(structure, filename="POSCAR", write_species=True, cartesian=True):
    _write_poscar(
        structure=pyiron_to_ase(structure),
        filename=filename,
        write_species=write_species,
        cartesian=cartesian,
    )
