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

from ase.atoms import Atoms as ASEAtoms

from pyiron_vasp.vasp.structure import (
    read_atoms as _read_atoms,
    atoms_from_string as _atoms_from_string,
)

from pyiron_atomistics.atomistics.structure.atoms import ase_to_pyiron


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


def read_atoms(
    filename="CONTCAR",
    return_velocities=False,
    species_list=None,
    species_from_potcar=False,
):
    return ase_to_pyiron(
        ase_obj=_read_atoms(
            filename=filename,
            return_velocities=return_velocities,
            species_list=species_list,
            species_from_potcar=species_from_potcar,
        )
    )


def atoms_from_string(string, read_velocities=False, species_list=None):
    output = _atoms_from_string(string=string, read_velocities=read_velocities, species_list=species_list)
    if not read_velocities:
        return ase_to_pyiron(ase_obj=output)
    else:
        atoms, velocities = output
        return ase_to_pyiron(ase_obj=atoms), velocities
