# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from __future__ import print_function

from collections import OrderedDict

import numpy as np
from pyiron_lammps.structure import LammpsStructure as LammpsStructureASE
from pyiron_lammps.units import UnitConverter

__author__ = "Joerg Neugebauer, Sudarsan Surendralal, Yury Lysogorskiy, Jan Janssen, Markus Tautschnig"
__copyright__ = (
    "Copyright 2021, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "1.0"
__maintainer__ = "Sudarsan Surendralal"
__email__ = "surendralal@mpie.de"
__status__ = "production"
__date__ = "Sep 1, 2017"


class LammpsStructure(LammpsStructureASE):
    """

    Args:
        input_file_name:
    """

    def __init__(self, bond_dict=None, job=None):
        if job is not None:
            super().__init__(bond_dict=bond_dict, units=job.units)
        else:
            super().__init__(bond_dict=bond_dict, units=None)
        self._molecule_ids = []

    @property
    def structure(self):
        """

        Returns:

        """
        return self._structure

    @structure.setter
    def structure(self, structure):
        """

        Args:
            structure:

        Returns:

        """
        self._structure = structure
        if self.atom_type == "full":
            input_str = self.structure_full()
        elif self.atom_type == "bond":
            input_str = self.structure_bond()
        elif self.atom_type == "charge":
            input_str = self.structure_charge()
        else:  # self.atom_type == 'atomic'
            input_str = self.structure_atomic()

        self._string_input = input_str + self._get_velocities_input_string()

    @property
    def molecule_ids(self):
        return self._molecule_ids

    @molecule_ids.setter
    def molecule_ids(self, molecule_ids=None):
        if molecule_ids is None:
            if "molecule_ids" in self.structure.get_tags():
                self._molecule_ids = self.structure.molecule_ids
            else:
                self._molecule_ids = np.ones(len(self.structure), dtype=int)
        else:
            self._molecule_ids = molecule_ids

    def structure_bond(self):
        """

        Returns:

        """
        species_lammps_id_dict = self.get_lammps_id_dict(self.el_eam_lst)
        self.molecule_ids = None
        # analyze structure to get molecule_ids, bonds, angles etc
        coords = self.rotate_positions(self._structure)

        elements = self._structure.get_chemical_symbols()

        ## Standard atoms stuff
        atoms = "Atoms \n\n"
        # atom_style bond
        # format: atom-ID, molecule-ID, atom_type, x, y, z
        format_str = "{0:d} {1:d} {2:d} {3:f} {4:f} {5:f} "
        if self._structure.dimension == 3:
            for id_atom, (x, y, z) in enumerate(coords):
                id_mol = self.molecule_ids[id_atom]
                atoms += (
                    format_str.format(
                        id_atom + 1,
                        id_mol,
                        species_lammps_id_dict[elements[id_atom]],
                        x,
                        y,
                        z,
                    )
                    + "\n"
                )
        elif self._structure.dimension == 2:
            for id_atom, (x, y) in enumerate(coords):
                id_mol = self.molecule_ids[id_atom]
                atoms += (
                    format_str.format(
                        id_atom + 1,
                        id_mol,
                        species_lammps_id_dict[elements[id_atom]],
                        x,
                        y,
                        0.0,
                    )
                    + "\n"
                )
        else:
            raise ValueError("dimension 1 not yet implemented")

        ## Bond related.
        # This seems independent from the lammps atom type ids, because bonds only use atom ids
        el_list = self._structure.get_species_symbols()
        el_dict = OrderedDict()
        for object_id, el in enumerate(el_list):
            el_dict[el] = object_id

        n_s = len(el_list)
        bond_type = np.ones([n_s, n_s], dtype=int)
        count = 0
        for i in range(n_s):
            for j in range(i, n_s):
                count += 1
                bond_type[i, j] = count
                bond_type[j, i] = count

        if self.structure.bonds is None:
            if self.cutoff_radius is None:
                bonds_lst = self.structure.get_bonds(max_shells=1)
            else:
                bonds_lst = self.structure.get_bonds(radius=self.cutoff_radius)
            bonds = []

            for ia, i_bonds in enumerate(bonds_lst):
                el_i = el_dict[elements[ia]]
                for el_j, b_lst in i_bonds.items():
                    b_type = bond_type[el_i][el_dict[el_j]]
                    for i_shell, ib_shell_lst in enumerate(b_lst):
                        for ib in np.unique(ib_shell_lst):
                            if ia < ib:  # avoid double counting of bonds
                                bonds.append([ia + 1, ib + 1, b_type])

            self.structure.bonds = np.array(bonds)
        bonds = self.structure.bonds

        bonds_str = "Bonds \n\n"
        for i_bond, (i_a, i_b, b_type) in enumerate(bonds):
            bonds_str += (
                "{0:d} {1:d} {2:d} {3:d}".format(i_bond + 1, b_type, i_a, i_b) + "\n"
            )

        return (
            self.lammps_header(
                structure=self.structure,
                cell_dimensions=self.simulation_cell(),
                species_lammps_id_dict=species_lammps_id_dict,
                nbonds=len(bonds),
                nbond_types=np.max(np.array(bonds)[:, 2]),
            )
            + "\n"
            + atoms
            + "\n"
            + bonds_str
            + "\n"
        )

    def structure_full(self):
        """
        Write routine to create atom structure static file for atom_type='full' that can be loaded by LAMMPS

        Returns:

        """
        species_lammps_id_dict = self.get_lammps_id_dict(self.el_eam_lst)
        self.molecule_ids = None
        coords = self.rotate_positions(self._structure)

        # extract electric charges from potential file
        q_dict = {}
        for species in self.structure.species:
            species_name = species.Abbreviation
            q_dict[species_name] = self.potential.get_charge(species_name)

        bonds_lst, angles_lst = [], []
        bond_type_lst, angle_type_lst = [], []
        # Using a cutoff distance to draw the bonds instead of the number of neighbors
        # Only if any bonds are defined
        if len(self._bond_dict.keys()) > 0:
            cutoff_list = list()
            for val in self._bond_dict.values():
                cutoff_list.append(np.max(val["cutoff_list"]))
            max_cutoff = np.max(cutoff_list)
            # Calculate neighbors only once
            neighbors = self._structure.get_neighbors_by_distance(
                cutoff_radius=max_cutoff
            )

            # Draw bonds between atoms is defined in self._bond_dict
            # Go through all elements for which bonds are defined
            for element, val in self._bond_dict.items():
                el_1_list = self._structure.select_index(element)
                if el_1_list is not None:
                    if len(el_1_list) > 0:
                        for i, v in enumerate(val["element_list"]):
                            el_2_list = self._structure.select_index(v)
                            cutoff_dist = val["cutoff_list"][i]
                            for j, ind in enumerate(
                                np.array(neighbors.indices, dtype=object)[el_1_list]
                            ):
                                # Only chose those indices within the cutoff distance and which belong
                                # to the species defined in the element_list
                                # i is the index of each bond type, and j is the element index
                                id_el = el_1_list[j]
                                bool_1 = (
                                    np.array(neighbors.distances, dtype=object)[id_el]
                                    <= cutoff_dist
                                )
                                act_ind = ind[bool_1]
                                bool_2 = np.in1d(act_ind, el_2_list)
                                final_ind = act_ind[bool_2]
                                # Get the bond and angle type
                                bond_type = val["bond_type_list"][i]
                                angle_type = val["angle_type_list"][i]
                                # Draw only maximum allowed bonds
                                final_ind = final_ind[: val["max_bond_list"][i]]
                                for fi in final_ind:
                                    bonds_lst.append([id_el + 1, fi + 1])
                                    bond_type_lst.append(bond_type)
                                # Draw angles if at least 2 bonds are present and if an angle type is defined for this
                                # particular set of bonds
                                if (
                                    len(final_ind) >= 2
                                    and val["angle_type_list"][i] is not None
                                ):
                                    angles_lst.append(
                                        [final_ind[0] + 1, id_el + 1, final_ind[1] + 1]
                                    )
                                    angle_type_lst.append(angle_type)

        if len(bond_type_lst) == 0:
            num_bond_types = 0
        else:
            num_bond_types = int(np.max(bond_type_lst))
        if len(angle_type_lst) == 0:
            num_angle_types = 0
        else:
            num_angle_types = int(np.max(angle_type_lst))

        atoms = "Atoms \n\n"

        # format: atom-ID, molecule-ID, atom_type, q, x, y, z
        format_str = "{0:d} {1:d} {2:d} {3:f} {4:f} {5:f} {6:f}"
        el_lst = self.structure.get_chemical_symbols()
        for id_atom, (el, coord) in enumerate(zip(el_lst, coords)):
            atoms += (
                format_str.format(
                    id_atom + 1,
                    self.molecule_ids[id_atom],
                    species_lammps_id_dict[el],
                    q_dict[el],
                    coord[0],
                    coord[1],
                    coord[2],
                )
                + "\n"
            )

        if len(bonds_lst) > 0:
            bonds_str = "Bonds \n\n"
            for i_bond, id_vec in enumerate(bonds_lst):
                bonds_str += (
                    "{0:d} {1:d} {2:d} {3:d}".format(
                        i_bond + 1, bond_type_lst[i_bond], id_vec[0], id_vec[1]
                    )
                    + "\n"
                )
        else:
            bonds_str = "\n"

        if len(angles_lst) > 0:
            angles_str = "Angles \n\n"
            for i_angle, id_vec in enumerate(angles_lst):
                angles_str += (
                    "{0:d} {1:d} {2:d} {3:d} {4:d}".format(
                        i_angle + 1,
                        angle_type_lst[i_angle],
                        id_vec[0],
                        id_vec[1],
                        id_vec[2],
                    )
                    + "\n"
                )
        else:
            angles_str = "\n"
        return (
            self.lammps_header(
                structure=self.structure,
                cell_dimensions=self.simulation_cell(),
                species_lammps_id_dict=species_lammps_id_dict,
                nbonds=len(bonds_lst),
                nangles=len(angles_lst),
                nbond_types=num_bond_types,
                nangle_types=num_angle_types,
            )
            + " \n"
            + atoms
            + "\n"
            + bonds_str
            + "\n"
            + angles_str
            + "\n"
        )

    def structure_charge(self):
        """
        Create atom structure including the atom charges.

        By convention the LAMMPS atom type numbers are chose alphabetically for the chemical species.

        Returns: LAMMPS readable structure.

        """
        species_lammps_id_dict = self.get_lammps_id_dict(self.el_eam_lst)
        atoms = "Atoms\n\n"
        coords = self.rotate_positions(self._structure)
        el_charge_lst = self._structure.get_initial_charges()
        el_lst = self._structure.get_chemical_symbols()
        for id_atom, (el, coord) in enumerate(zip(el_lst, coords)):
            dim = self._structure.dimension
            c = np.zeros(3)
            c[:dim] = coord
            atoms += (
                "{0:d} {1:d} {2:f} {3:.15f} {4:.15f} {5:.15f}".format(
                    id_atom + 1,
                    species_lammps_id_dict[el],
                    el_charge_lst[id_atom],
                    c[0],
                    c[1],
                    c[2],
                )
                + "\n"
            )
        return (
            self.lammps_header(
                structure=self.structure,
                cell_dimensions=self.simulation_cell(),
                species_lammps_id_dict=species_lammps_id_dict,
            )
            + atoms
            + "\n"
        )
