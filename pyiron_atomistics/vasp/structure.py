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

import os
import warnings
from collections import OrderedDict

import numpy as np
from pyiron_vasp.vasp.structure import _dict_to_atoms, get_species_list_from_potcar


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
    potcar_file = os.path.join(os.path.dirname(filename), "POTCAR")
    if (species_list is None) and species_from_potcar:
        species_list = get_species_list_from_potcar(potcar_file)
        if len(species_list) == 0:
            warnings.warn("Warning! Unable to read species information from POTCAR")
    file_string = list()
    with open(filename) as f:
        for line in f:
            line = line.strip()
            file_string.append(line)
    return atoms_from_string(
        file_string, read_velocities=return_velocities, species_list=species_list
    )


def get_poscar_content(structure, write_species=True, cartesian=True):
    endline = "\n"
    selec_dyn = False
    line_lst = [
        "Poscar file generated with pyiron" + endline,
        "1.0" + endline,
    ]
    for a_i in structure.get_cell():
        x, y, z = a_i
        line_lst.append("{0:.15f} {1:.15f} {2:.15f}".format(x, y, z) + endline)
    atom_numbers = structure.get_number_species_atoms()
    if write_species:
        line_lst.append(" ".join(atom_numbers.keys()) + endline)
    num_str = [str(val) for val in atom_numbers.values()]
    line_lst.append(" ".join(num_str))
    line_lst.append(endline)
    if "selective_dynamics" in structure.get_tags():
        selec_dyn = True
        cartesian = False
        line_lst.append("Selective dynamics" + endline)
    sorted_coords = list()
    selec_dyn_lst = list()
    for species in atom_numbers.keys():
        indices = structure.select_index(species)
        for i in indices:
            if cartesian:
                sorted_coords.append(structure.positions[i])
            else:
                sorted_coords.append(structure.get_scaled_positions()[i])
            if selec_dyn:
                selec_dyn_lst.append(structure.selective_dynamics[i])
    if cartesian:
        line_lst.append("Cartesian" + endline)
    else:
        line_lst.append("Direct" + endline)
    if selec_dyn:
        for i, vec in enumerate(sorted_coords):
            x, y, z = vec
            sd_string = " ".join(["T" if sd else "F" for sd in selec_dyn_lst[i]])
            line_lst.append(
                "{0:.15f} {1:.15f} {2:.15f}".format(x, y, z) + " " + sd_string + endline
            )
    else:
        for i, vec in enumerate(sorted_coords):
            x, y, z = vec
            line_lst.append("{0:.15f} {1:.15f} {2:.15f}".format(x, y, z) + endline)
    return line_lst


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
    string = [s.strip() for s in string]
    string_lower = [s.lower() for s in string]
    atoms_dict = dict()
    atoms_dict["first_line"] = string[0]
    # del string[0]
    atoms_dict["selective_dynamics"] = False
    atoms_dict["relative"] = False
    if "direct" in string_lower or "d" in string_lower:
        atoms_dict["relative"] = True
    atoms_dict["scaling_factor"] = float(string[1])
    unscaled_cell = list()
    for i in [2, 3, 4]:
        vec = list()
        for j in range(3):
            vec.append(float(string[i].split()[j]))
        unscaled_cell.append(vec)
    if atoms_dict["scaling_factor"] > 0.0:
        atoms_dict["cell"] = np.array(unscaled_cell) * atoms_dict["scaling_factor"]
    else:
        atoms_dict["cell"] = np.array(unscaled_cell) * (
            (-atoms_dict["scaling_factor"]) ** (1.0 / 3.0)
        )
    if "selective dynamics" in string_lower:
        atoms_dict["selective_dynamics"] = True
    no_of_species = len(string[5].split())
    species_dict = OrderedDict()
    position_index = 7
    if atoms_dict["selective_dynamics"]:
        position_index += 1
    for i in range(no_of_species):
        species_dict["species_" + str(i)] = dict()
        try:
            species_dict["species_" + str(i)]["count"] = int(string[5].split()[i])
        except ValueError:
            species_dict["species_" + str(i)]["species"] = string[5].split()[i]
            species_dict["species_" + str(i)]["count"] = int(string[6].split()[i])
    atoms_dict["species_dict"] = species_dict
    if "species" in atoms_dict["species_dict"]["species_0"].keys():
        position_index += 1
    positions = list()
    selective_dynamics = list()
    n_atoms = sum(
        [
            atoms_dict["species_dict"][key]["count"]
            for key in atoms_dict["species_dict"].keys()
        ]
    )
    try:
        for i in range(position_index, position_index + n_atoms):
            string_list = np.array(string[i].split())
            positions.append([float(val) for val in string_list[0:3]])
            if atoms_dict["selective_dynamics"]:
                selective_dynamics.append(["T" in val for val in string_list[3:6]])
    except (ValueError, IndexError):
        raise AssertionError(
            "The number of positions given does not match the number of atoms"
        )
    atoms_dict["positions"] = np.array(positions)
    if not atoms_dict["relative"]:
        if atoms_dict["scaling_factor"] > 0.0:
            atoms_dict["positions"] *= atoms_dict["scaling_factor"]
        else:
            atoms_dict["positions"] *= (-atoms_dict["scaling_factor"]) ** (1.0 / 3.0)
    velocities = list()
    try:
        atoms = _dict_to_atoms(atoms_dict, species_list=species_list)
    except ValueError:
        atoms = _dict_to_atoms(atoms_dict, read_from_first_line=True)
    if atoms_dict["selective_dynamics"]:
        selective_dynamics = np.array(selective_dynamics)
        unique_sel_dyn, inverse, counts = np.unique(
            selective_dynamics, axis=0, return_counts=True, return_inverse=True
        )
        count_index = np.argmax(counts)
        atoms.add_tag(selective_dynamics=unique_sel_dyn.tolist()[count_index])
        is_not_majority = np.arange(len(unique_sel_dyn), dtype=int) != count_index
        for i, val in enumerate(unique_sel_dyn):
            if is_not_majority[i]:
                for key in np.argwhere(inverse == i).flatten():
                    atoms.selective_dynamics[int(key)] = val.tolist()
    if read_velocities:
        velocity_index = position_index + n_atoms + 1
        for i in range(velocity_index, velocity_index + n_atoms):
            try:
                velocities.append([float(val) for val in string[i].split()[0:3]])
            except IndexError:
                break
        if not (len(velocities) == n_atoms):
            warnings.warn(
                "The velocities are either not available or they are incomplete/corrupted. Returning empty "
                "list instead",
                UserWarning,
            )
            return atoms, list()
        return atoms, velocities
    else:
        return atoms
