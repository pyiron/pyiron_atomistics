from collections import OrderedDict

import numpy as np
from pyiron_base import state
from pyiron_vasp.vasp.vasprun import (
    Vasprun as _Vasprun,
    clean_character,
)

from pyiron_atomistics.atomistics.structure.atoms import Atoms
from pyiron_atomistics.atomistics.structure.periodic_table import PeriodicTable


class Vasprun(_Vasprun):
    def get_initial_structure(self):
        """
        Gets the initial structure from the simulation

        Returns:
            pyiron.atomistics.structure.atoms.Atoms: The initial structure

        """
        try:
            el_list = self.vasprun_dict["atominfo"]["species_list"]
            cell = self.vasprun_dict["init_structure"]["cell"]
            positions = self.vasprun_dict["init_structure"]["positions"]
            if len(positions[positions > 1.01]) > 0:
                basis = Atoms(el_list, positions=positions, cell=cell, pbc=True)
            else:
                basis = Atoms(el_list, scaled_positions=positions, cell=cell, pbc=True)
            if "selective_dynamics" in self.vasprun_dict["init_structure"].keys():
                basis.add_tag(selective_dynamics=[True, True, True])
                for i, val in enumerate(
                        self.vasprun_dict["init_structure"]["selective_dynamics"]
                ):
                    basis[i].selective_dynamics = val
            return basis
        except KeyError:
            state.logger.warning(
                "The initial structure could not be extracted from vasprun properly"
            )
            return

    def get_final_structure(self):
        """
        Gets the final structure from the simulation

        Returns:
            pyiron.atomistics.structure.atoms.Atoms: The final structure

        """
        try:
            basis = self.get_initial_structure()
            basis.set_cell(self.vasprun_dict["final_structure"]["cell"])
            positions = self.vasprun_dict["final_structure"]["positions"]
            if len(positions[positions > 1.01]) > 0:
                basis.positions = positions
            else:
                basis.set_scaled_positions(positions)
            return basis
        except (KeyError, AttributeError, ValueError):
            return

    def parse_atom_information_to_dict(self, node, d):
        """
        Parses atom information from a node to a dictionary

        Args:
            node (xml.etree.Element instance): The node to parse
            d (dict): The dictionary to which data is to be parsed
        """
        if not (node.tag == "atominfo"):
            raise AssertionError()
        species_dict = OrderedDict()
        for leaf in node:
            if leaf.tag == "atoms":
                d["n_atoms"] = self._parse_vector(leaf)[0]
            if leaf.tag == "types":
                d["n_species"] = self._parse_vector(leaf)[0]
            if leaf.tag == "array":
                if leaf.attrib["name"] == "atomtypes":
                    for item in leaf:
                        if item.tag == "set":
                            for sp in item:
                                elements = sp
                                if elements[1].text in species_dict.keys():
                                    pse = PeriodicTable()
                                    count = 1
                                    not_unique = True
                                    species_key = None
                                    while not_unique:
                                        species_key = "_".join(
                                            [elements[1].text, str(count)]
                                        )
                                        if species_key not in species_dict.keys():
                                            not_unique = False
                                        else:
                                            count += 1
                                    if species_key is not None:
                                        pse.add_element(
                                            clean_character(elements[1].text),
                                            species_key,
                                        )
                                        special_element = pse.element(species_key)
                                        species_dict[special_element] = dict()
                                        species_dict[special_element]["n_atoms"] = int(
                                            elements[0].text
                                        )
                                        species_dict[special_element]["valence"] = (
                                            float(elements[3].text)
                                        )
                                else:
                                    species_key = elements[1].text
                                    species_dict[species_key] = dict()
                                    species_dict[species_key]["n_atoms"] = int(
                                        elements[0].text
                                    )
                                    species_dict[species_key]["valence"] = float(
                                        elements[3].text
                                    )
        d["species_dict"] = species_dict
        species_list = list()
        for key, val in species_dict.items():
            for sp in np.tile([key], species_dict[key]["n_atoms"]):
                species_list.append(clean_character(sp))
        d["species_list"] = species_list