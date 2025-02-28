# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from __future__ import division, print_function

import ast
import importlib
import warnings
from collections import OrderedDict
from collections.abc import Sequence
from copy import copy

import numpy as np
import seekpath
from ase.atoms import Atom as ASEAtom
from ase.atoms import Atoms as ASEAtoms
from ase.constraints import FixCartesian
from ase.symbols import Symbols as ASESymbols
from pyiron_base import state
from pyiron_snippets.deprecate import deprecate
from structuretoolkit.analyse import (
    find_mic,
    get_distances_array,
    get_neighborhood,
    get_neighbors,
    get_symmetry,
)
from structuretoolkit.common import (
    ase_to_pymatgen,
    center_coordinates_in_unit_cell,
    pymatgen_to_ase,
)

from pyiron_atomistics.atomistics.structure.analyse import Analyse
from pyiron_atomistics.atomistics.structure.atom import (
    Atom,
)
from pyiron_atomistics.atomistics.structure.atom import (
    ase_to_pyiron as ase_to_pyiron_atom,
)
from pyiron_atomistics.atomistics.structure.periodic_table import (
    ChemicalElement,
    PeriodicTable,
    chemical_element_dict_to_hdf,
)
from pyiron_atomistics.atomistics.structure.pyscal import pyiron_to_pyscal_system

__author__ = "Joerg Neugebauer, Sudarsan Surendralal"
__copyright__ = (
    "Copyright 2021, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "1.0"
__maintainer__ = "Sudarsan Surendralal"
__email__ = "surendralal@mpie.de"
__status__ = "production"
__date__ = "Sep 1, 2017"


class Atoms(ASEAtoms):
    """
    The Atoms class represents all the information required to describe a structure at the atomic scale. This class is
    derived from the `ASE atoms class`_.

    Args:
        elements (list/numpy.ndarray): List of strings containing the elements or a list of
                            atomistics.structure.periodic_table.ChemicalElement instances
        numbers (list/numpy.ndarray): List of atomic numbers of elements
        symbols (list/numpy.ndarray): List of chemical symbols
        positions (list/numpy.ndarray): List of positions
        scaled_positions (list/numpy.ndarray): List of scaled positions (relative coordinates)
        pbc (list/numpy.ndarray/boolean): Tells if periodic boundary conditions should be applied on the three axes
        cell (list/numpy.ndarray instance): A 3x3 array representing the lattice vectors of the structure

    Note: Only one of elements/symbols or numbers should be assigned during initialization

    Attributes:

        indices (numpy.ndarray): A list of size N which gives the species index of the structure which has N atoms

    .. _ASE atoms class: https://wiki.fysik.dtu.dk/ase/ase/atoms.html

    """

    def __init__(
        self,
        symbols=None,
        positions=None,
        numbers=None,
        tags=None,
        momenta=None,
        masses=None,
        magmoms=None,
        charges=None,
        scaled_positions=None,
        cell=None,
        pbc=None,
        celldisp=None,
        constraint=None,
        calculator=None,
        info=None,
        indices=None,
        elements=None,
        dimension=None,
        species=None,
        **qwargs,
    ):
        if symbols is not None:
            if elements is None:
                elements = symbols
            else:
                raise ValueError("Only elements OR symbols should be given.")
        if (
            tags is not None
            or momenta is not None
            or masses is not None
            or charges is not None
            or celldisp is not None
            or constraint is not None
            or calculator is not None
            or info is not None
        ):
            state.logger.debug("Not supported parameter used!")

        self._is_scaled = False

        self._species = list()
        self._pse = PeriodicTable()

        el_index_lst = list()
        element_list = None
        if numbers is not None:  # for ASE compatibility
            if not (elements is None):
                raise AssertionError()
            elements = self.numbers_to_elements(numbers)
        if elements is not None:
            el_object_list = None
            if isinstance(elements, str):
                element_list = self.convert_formula(elements)
            elif isinstance(elements, (list, tuple, np.ndarray)):
                if not all([isinstance(el, elements[0].__class__) for el in elements]):
                    object_list = list()
                    for el in elements:
                        if isinstance(el, str):
                            object_list.append(self.convert_element(el))
                        if isinstance(el, ChemicalElement):
                            object_list.append(el)
                        if isinstance(el, Atom):
                            object_list.append(el.element)
                        if isinstance(el, (int, np.integer)):
                            # pse = PeriodicTable()
                            object_list.append(self._pse.element(el))
                        el_object_list = object_list

                if len(elements) == 0:
                    element_list = elements
                else:
                    if isinstance(elements[0], (list, tuple, np.ndarray)):
                        elements = np.array(elements).flatten()
                    if isinstance(elements[0], str):
                        element_list = elements
                    elif isinstance(elements[0], ChemicalElement):
                        el_object_list = elements
                    elif isinstance(elements[0], Atom):
                        el_object_list = [el.element for el in elements]
                        positions = [el.position for el in elements]
                    elif elements.dtype in [int, np.integer]:
                        el_object_list = self.numbers_to_elements(elements)
                    else:
                        raise ValueError(
                            "Unknown static type for element in list: "
                            + str(type(elements[0]))
                        )

            if el_object_list is None:
                el_object_list = [self.convert_element(el) for el in element_list]

            # Create a list from a set but always preserve order
            self.set_species(list(dict.fromkeys(el_object_list)))
            el_index_lst = [self._species_to_index_dict[el] for el in el_object_list]

        elif indices is not None:
            el_index_lst = indices
            if species is None:
                raise ValueError(
                    "species must be given if indices is given, but is None."
                )
            self.set_species(species)

        indices = np.array(el_index_lst, dtype=int)

        el_lst = [
            el.Abbreviation if el.Parent is None else el.Parent for el in self.species
        ]
        symbols = np.array([el_lst[el] for el in indices])
        super(Atoms, self).__init__(
            symbols=symbols,
            positions=positions,
            numbers=None,
            tags=tags,
            momenta=momenta,
            masses=masses,
            magmoms=magmoms,
            charges=charges,
            scaled_positions=scaled_positions,
            cell=cell,
            pbc=pbc,
            celldisp=celldisp,
            constraint=constraint,
            calculator=calculator,
            info=info,
        )

        self.set_array("indices", indices)

        self.bonds = None
        self.units = {"length": "A", "mass": "u"}
        self.set_initial_magnetic_moments(magmoms)
        self._high_symmetry_points = None
        self._high_symmetry_path = None
        self.dimension = dimension
        if len(self.positions) > 0:
            self.dimension = len(self.positions[0])
        else:
            self.dimension = 0
        self._analyse = Analyse(self)
        # Velocities were not handled at all during file writing
        self._velocities = None

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __getstate__(self):
        # Only necessary to support pickling in python <3.11
        # https://docs.python.org/release/3.11.2/library/pickle.html#object.__getstate__
        return self.__dict__

    @property
    def velocities(self):
        return self._velocities

    @velocities.setter
    def velocities(self, val):
        if self.positions.shape == val.shape:
            self._velocities = val
        else:
            raise ValueError(
                f"Shape of velocities and positions has to match. Velocities shape: {val.shape}, positions shape: {self.positions.shape}"
            )

    @property
    def spins(self):
        """
        Magnetic spins for each atom in the structure

        Returns:
            numpy.ndarray/list: The magnetic moments for reach atom as a single value or a vector (non-collinear spins)

        """
        if self.has("initial_magmoms"):
            return self.arrays["initial_magmoms"]
        else:
            raise AttributeError("'Atoms' object has no attribute 'spins'")

    @spins.setter
    def spins(self, val):
        self.set_array("initial_magmoms", None)
        if val is not None:
            self.set_array("initial_magmoms", np.asarray(val))

    @property
    def analyse(self):
        return self._analyse

    @property
    def species(self):
        """
        list: A list of atomistics.structure.periodic_table.ChemicalElement instances

        """
        return self._species

    def set_species(self, value):
        """
        Setting the species list

        Args:
            value (list): A list atomistics.structure.periodic_table.ChemicalElement instances

        """
        if value is not None:
            self._species = list(value)[:]

    @property
    def _store_elements(self) -> dict:
        return {el.Abbreviation: el for el in self.species}

    @property
    def _species_to_index_dict(self) -> dict:
        return {el: i for i, el in enumerate(self.species)}

    @property
    def symbols(self):
        """
        Get chemical symbols as a :class:`ase.symbols.Symbols` object.

        The object works like ``atoms.numbers`` except its values
        are strings.  It supports in-place editing.
        """
        sym_obj = Symbols(self.numbers)
        sym_obj.structure = self
        return sym_obj

    @symbols.setter
    def symbols(self, obj):
        new_symbols = Symbols.fromsymbols(obj)
        self.numbers[:] = new_symbols.numbers

    @property
    def elements(self):
        """
        numpy.ndarray: A size N list of atomistics.structure.periodic_table.ChemicalElement instances according
                       to the ordering of the atoms in the instance

        """
        return np.array([self.species[el] for el in self.indices])

    def get_high_symmetry_points(self):
        """
        dictionary of high-symmetry points defined for this specific structure.

        Returns:
            dict: high_symmetry_points
        """
        return self._high_symmetry_points

    def _set_high_symmetry_points(self, new_high_symmetry_points):
        """
        Sets new high symmetry points dictionary.

        Args:
            new_high_symmetry_points (dict): new high symmetry points
        """
        if not isinstance(new_high_symmetry_points, dict):
            raise ValueError("has to be dict!")
        self._high_symmetry_points = new_high_symmetry_points

    def add_high_symmetry_points(self, new_points):
        """
        Adds new points to the dict of existing high symmetry points.

        Args:
            new_points (dict): Points to add
        """
        if self.get_high_symmetry_points() is None:
            raise AssertionError(
                "Construct high symmetry points first. Use self.create_line_mode_structure()."
            )
        else:
            self._high_symmetry_points.update(new_points)

    def get_high_symmetry_path(self):
        """
        Path used for band structure calculations

        Returns:
            dict: dict of pathes with start and end points.

        """
        return self._high_symmetry_path

    def _set_high_symmetry_path(self, new_path):
        """
        Sets new list for the high symmetry path used for band structure calculations.

        Args:
            new_path (dict): dictionary of lists of tuples with start and end point.
                E.G. {"my_path": [('Gamma', 'X'), ('X', 'Y')]}
        """
        self._high_symmetry_path = new_path

    def add_high_symmetry_path(self, path):
        """
        Adds a new path to the dictionary of pathes for band structure calculations.

        Args:
            path (dict): dictionary of lists of tuples with start and end point.
                E.G. {"my_path": [('Gamma', 'X'), ('X', 'Y')]}
        """
        if self.get_high_symmetry_path() is None:
            raise AssertionError(
                "Construct high symmetry path first. Use self.create_line_mode_structure()."
            )

        for values_all in path.values():
            for values in values_all:
                if not len(values) == 2:
                    raise ValueError(
                        "'{}' is not a propper trace! It has to contain exactly 2 values! (start and end point)".format(
                            values
                        )
                    )
                for v in values:
                    if v not in self.get_high_symmetry_points().keys():
                        raise ValueError(
                            "'{}' is not a valid high symmetry point".format(v)
                        )

        self._high_symmetry_path.update(path)

    @deprecate(
        "Use Atoms.set_array() instead: e.g. Atoms.set_array('selective_dynamics', np.repeat([[True, True, False]], len(Atoms), axis=0))"
    )
    def add_tag(self, **qwargs):
        """
        Add tags to the atoms object.

        Examples:

            For selective dynamics::

            >>> self.add_tag(selective_dynamics=[False, False, False])

        """
        for tag, value in qwargs.items():
            self.set_array(tag, np.repeat([value], len(self), axis=0))

    # @staticmethod
    def numbers_to_elements(self, numbers):
        """
        Convert atomic numbers in element objects (needed for compatibility with ASE)

        Args:
            numbers (list): List of Element Numbers (as Integers; default in ASE)

        Returns:
            list: A list of elements as needed for pyiron

        """
        # pse = PeriodicTable()  # TODO; extend to internal PSE which can contain additional elements and tags
        atom_number_to_element = {}
        for i_el in set(numbers):
            i_el = int(i_el)
            atom_number_to_element[i_el] = self._pse.element(i_el)
        return [atom_number_to_element[i_el] for i_el in numbers]

    def copy(self):
        """
        Returns a copy of the instance

        Returns:
            pyiron.atomistics.structure.atoms.Atoms: A copy of the instance

        """
        return self.__copy__()

    def to_dict(self):
        hdf_structure = {
            "TYPE": str(type(self)),
            "units": self.units,
            "dimension": self.dimension,
            "positions": self.positions,
            "info": self.info,
        }
        for el in self.species:
            if isinstance(el.tags, dict):
                for k, v in el.to_dict().items():
                    hdf_structure["new_species/" + el.Abbreviation + "/" + k] = v
        hdf_structure["species"] = [el.Abbreviation for el in self.species]
        hdf_structure["indices"] = self.indices

        for tag, value in self.arrays.items():
            if tag in ["positions", "numbers", "indices"]:
                continue
            hdf_structure["tags/" + tag] = value.tolist()

        if self.cell is not None:
            # Convert ASE cell object to numpy array before storing
            hdf_structure["cell/cell"] = np.array(self.cell)
            hdf_structure["cell/pbc"] = self.pbc

        if self.has("initial_magmoms"):
            hdf_structure["spins"] = self.spins
        # potentials with explicit bonds (TIP3P, harmonic, etc.)
        if self.bonds is not None:
            hdf_structure["explicit_bonds"] = self.bonds

        if self._high_symmetry_points is not None:
            hdf_structure["high_symmetry_points"] = self._high_symmetry_points

        if self._high_symmetry_path is not None:
            hdf_structure["high_symmetry_path"] = self._high_symmetry_path

        if self.calc is not None:
            calc_dict = self.calc.todict()
            calc_dict["label"] = self.calc.label
            calc_dict["class"] = (
                self.calc.__class__.__module__ + "." + self.calc.__class__.__name__
            )
            hdf_structure["calculator"] = calc_dict
        return hdf_structure

    def from_dict(self, obj_dict):
        if "new_species" in obj_dict.keys():
            self._pse.from_dict(obj_dict=obj_dict["new_species"])

        el_object_list = [
            self.convert_element(el, self._pse) for el in obj_dict["species"]
        ]
        self.arrays["indices"] = obj_dict["indices"]

        self.set_species(el_object_list)
        self.bonds = None

        tr_dict = {1: True, 0: False}
        self.dimension = obj_dict["dimension"]
        self.units = obj_dict["units"]

        if "cell" in obj_dict.keys():
            self.cell = obj_dict["cell"]["cell"]
            self.pbc = obj_dict["cell"]["pbc"]

        # Backward compatibility
        position_tag = "positions"
        if position_tag not in obj_dict.keys():
            position_tag = "coordinates"
        self.arrays["positions"] = obj_dict[position_tag]
        if "is_absolute" in obj_dict.keys() and not tr_dict[obj_dict["is_absolute"]]:
            self.set_scaled_positions(self.arrays["positions"])

        self.arrays["numbers"] = self.get_atomic_numbers()

        if "explicit_bonds" in obj_dict.keys():
            # print "bonds: "
            self.bonds = obj_dict["explicit_bonds"]
        if "spins" in obj_dict.keys():
            self.spins = obj_dict["spins"]
        if "tags" in obj_dict.keys():
            tags_dict = obj_dict["tags"]
            for tag, tag_item in tags_dict.items():
                if tag in ["initial_magmoms"]:
                    continue
                # tr_dict = {'0': False, '1': True}
                if isinstance(tag_item, (list, np.ndarray)):
                    my_list = tag_item
                else:  # legacy of SparseList
                    raise NotImplementedError()
                self.set_array(tag, np.asarray(my_list))

        if "bonds" in obj_dict.keys():
            self.bonds = obj_dict["explicit_bonds"]

        self._high_symmetry_points = None
        if "high_symmetry_points" in obj_dict.keys():
            self._high_symmetry_points = obj_dict["high_symmetry_points"]

        self._high_symmetry_path = None
        if "high_symmetry_path" in obj_dict.keys():
            self._high_symmetry_path = obj_dict["high_symmetry_path"]
        if "info" in obj_dict.keys():
            self.info = obj_dict["info"]
        if "calculator" in obj_dict.keys():
            calc_dict = obj_dict["calculator"]
            class_path = calc_dict.pop("class")
            calc_module = importlib.import_module(".".join(class_path.split(".")[:-1]))
            calc_class = getattr(calc_module, class_path.split(".")[-1])
            self.calc = calc_class(**calc_dict)
        return self

    def to_hdf(self, hdf, group_name="structure"):
        """
        Save the object in a HDF5 file

        Args:
            hdf (pyiron_base.generic.hdfio.FileHDFio): HDF path to which the object is to be saved
            group_name (str):
                Group name with which the object should be stored. This same name should be used to retrieve the object

        """
        structure_dict_to_hdf(data_dict=self.to_dict(), hdf=hdf, group_name=group_name)

    def from_hdf(self, hdf, group_name="structure"):
        """
        Retrieve the object from a HDF5 file

        Args:
            hdf (pyiron_base.generic.hdfio.FileHDFio): HDF path to which the object is to be saved
            group_name (str): Group name from which the Atoms object is retreived.

        Returns:
            pyiron_atomistics.structure.atoms.Atoms: The retrieved atoms class

        """
        return self.from_dict(
            obj_dict=hdf.open(group_name).read_dict_from_hdf(recursive=True)
        )

    def select_index(self, el):
        """
        Returns the indices of a given element in the structure

        Args:
            el (str/atomistics.structures.periodic_table.ChemicalElement/list): Element for which the indices should
                                                                                  be returned
        Returns:
            numpy.ndarray: An array of indices of the atoms of the given element

        """
        if isinstance(el, str):
            return np.where(self.get_chemical_symbols() == el)[0]
        elif isinstance(el, ChemicalElement):
            return np.where([e == el for e in self.get_chemical_elements()])[0]
        if isinstance(el, (list, np.ndarray)):
            if isinstance(el[0], str):
                return np.where(np.isin(self.get_chemical_symbols(), el))[0]
            elif isinstance(el[0], ChemicalElement):
                return np.where([e in el for e in self.get_chemical_elements()])[0]

    def select_parent_index(self, el):
        """
        Returns the indices of a given element in the structure ignoring user defined elements

        Args:
            el (str/atomistics.structures.periodic_table.ChemicalElement): Element for which the indices should
                                                                                  be returned
        Returns:
            numpy.ndarray: An array of indices of the atoms of the given element

        """
        parent_basis = self.get_parent_basis()
        return parent_basis.select_index(el)

    def get_tags(self):
        """
        Returns the keys of the stored tags of the structure

        Returns:
            dict_keys: Keys of the stored tags

        """
        return self.arrays.keys()

    def convert_element(self, el, pse=None):
        """
        Convert a string or an atom instance into a ChemicalElement instance

        Args:
            el (str/atomistics.structure.atom.Atom): String or atom instance from which the element should
                                                            be generated
            pse (atomistics.structure.periodictable.PeriodicTable): PeriodicTable instance from which the element
                                                                           is generated (optional)

        Returns:

            atomistics.structure.periodictable.ChemicalElement: The required chemical element

        """
        if el in list(self._store_elements.keys()):
            return self._store_elements[el]

        if isinstance(el, str):  # as symbol
            element = Atom(el, pse=pse).element
        elif isinstance(el, Atom):
            element = el.element
            el = el.element.Abbreviation
        elif isinstance(el, ChemicalElement):
            element = el
            el = el.Abbreviation
        else:
            raise ValueError("Unknown static type to specify a element")

        if hasattr(self, "species"):
            if element not in self.species:
                self._species.append(element)
                self.set_species(self._species)
        return element

    def get_chemical_formula(self):
        """
        Returns the chemical formula of structure

        Returns:
            str: The chemical formula as a string

        """
        species = self.get_number_species_atoms()
        formula = ""
        for string_sym, num in species.items():
            if num == 1:
                formula += str(string_sym)
            else:
                formula += str(string_sym) + str(num)
        return formula

    def get_chemical_indices(self):
        """
        Returns the list of chemical indices as ordered in self.species

        Returns:
            numpy.ndarray: A list of chemical indices

        """
        return self.indices.copy()

    def get_atomic_numbers(self):
        """
        Returns the atomic numbers of all the atoms in the structure

        Returns:
            numpy.ndarray: A list of atomic numbers

        """
        el_lst = [el.AtomicNumber for el in self.species]
        return np.array([el_lst[el] for el in self.indices])

    def get_chemical_symbols(self):
        """
        Returns the chemical symbols for all the atoms in the structure

        Returns:
            numpy.ndarray: A list of chemical symbols

        """
        el_lst = [el.Abbreviation for el in self.species]
        return np.array([el_lst[el] for el in self.indices])

    def get_parent_symbols(self):
        """
        Returns the chemical symbols for all the atoms in the structure even for user defined elements

        Returns:
            numpy.ndarray: A list of chemical symbols

        """
        sp_parent_list = list()
        for sp in self.species:
            if isinstance(sp.Parent, (float, type(None))):
                sp_parent_list.append(sp.Abbreviation)
            else:
                sp_parent_list.append(sp.Parent)
        return np.array([sp_parent_list[i] for i in self.indices])

    def get_parent_basis(self):
        """
        Returns the basis with all user defined/special elements as the it's parent

        Returns:
            pyiron.atomistics.structure.atoms.Atoms: Structure without any user defined elements

        """
        parent_basis = copy(self)
        new_species = np.array(parent_basis.species)
        for i, sp in enumerate(new_species):
            if not isinstance(sp.Parent, (float, type(None))):
                pse = PeriodicTable()
                new_species[i] = pse.element(sp.Parent)
        sym_list = [el.Abbreviation for el in new_species]
        if len(sym_list) != len(np.unique(sym_list)):
            uni, ind, inv_ind = np.unique(
                sym_list, return_index=True, return_inverse=True
            )
            new_species = new_species[ind].copy()
            parent_basis.set_species(list(new_species))
            indices_copy = parent_basis.indices.copy()
            for i, ind_ind in enumerate(inv_ind):
                indices_copy[parent_basis.indices == i] = ind_ind
            parent_basis.set_array("indices", indices_copy)
            return parent_basis
        parent_basis.set_species(list(new_species))
        return parent_basis

    def get_chemical_elements(self):
        """
        Returns the list of chemical element instances

        Returns:
            numpy.ndarray: A list of chemical element instances

        """
        return self.elements

    def get_number_species_atoms(self):
        """
        Returns a dictionary with the species in the structure and the corresponding count in the structure

        Returns:
            collections.OrderedDict: An ordered dictionary with the species and the corresponding count

        """
        count = OrderedDict()
        # print "sorted: ", sorted(set(self.elements))
        for el in sorted(set(self.get_chemical_symbols())):
            count[el] = 0

        for el in self.get_chemical_symbols():
            count[el] += 1
        return count

    def get_species_symbols(self):
        """
        Returns the symbols of the present species

        Returns:
            numpy.ndarray: List of the symbols of the species

        """
        return np.array(sorted([el.Abbreviation for el in self.species]))

    def get_species_objects(self):
        """


        Returns:

        """
        el_set = self.species
        el_sym_lst = {el.Abbreviation: i for i, el in enumerate(el_set)}
        el_sorted = self.get_species_symbols()
        return [el_set[el_sym_lst[el]] for el in el_sorted]

    def get_number_of_species(self):
        """

        Returns:

        """
        return len(self.species)

    def get_number_of_degrees_of_freedom(self):
        """

        Returns:

        """
        return len(self) * self.dimension

    def get_center_of_mass(self):
        """
        Returns:
            com (float): center of mass in A
        """
        masses = self.get_masses()
        return np.einsum("i,ij->j", masses, self.positions) / np.sum(masses)

    def get_masses(self):
        """
        Gets the atomic masses of all atoms in the structure

        Returns:
            numpy.ndarray: Array of masses

        """
        el_lst = [el.AtomicMass for el in self.species]
        return np.array([el_lst[el] for el in self.indices])

    def get_masses_dof(self):
        """

        Returns:

        """
        dim = self.dimension
        return np.repeat(self.get_masses(), dim)

    def get_volume(self, per_atom=False):
        """

        Args:
            per_atom (bool): True if volume per atom is to be returned

        Returns:
            volume (float): Volume in A**3

        """
        if per_atom:
            return np.abs(np.linalg.det(self.cell)) / len(self)
        else:
            return np.abs(np.linalg.det(self.cell))

    def get_density(self):
        """
        Returns the density in g/cm^3

        Returns:
            float: Density of the structure

        """
        # conv_factor = Ang3_to_cm3/scipi.constants.Avogadro
        # with Ang3_to_cm3 = 1e24
        conv_factor = 1.660539040427164
        return conv_factor * np.sum(self.get_masses()) / self.get_volume()

    def get_number_of_atoms(self):
        """

        Returns:

        """
        # assert(len(self) == np.sum(self.get_number_species_atoms().values()))
        return len(self)

    @deprecate
    def set_absolute(self):
        if self._is_scaled:
            self._is_scaled = False

    @deprecate
    def set_relative(self):
        if not self._is_scaled:
            self._is_scaled = True

    def get_wrapped_coordinates(self, positions, epsilon=1.0e-8):
        """
        Return coordinates in wrapped in the periodic cell

        Args:
            positions (list/numpy.ndarray): Positions
            epsilon (float): displacement to add to avoid wrapping of atoms at borders

        Returns:

            numpy.ndarray: Wrapped positions

        """
        scaled_positions = np.einsum(
            "ji,nj->ni", np.linalg.inv(self.cell), np.asarray(positions).reshape(-1, 3)
        )
        if any(self.pbc):
            scaled_positions[:, self.pbc] -= np.floor(
                scaled_positions[:, self.pbc] + epsilon
            )
        new_positions = np.einsum("ji,nj->ni", self.cell, scaled_positions)
        return new_positions.reshape(np.asarray(positions).shape)

    def center_coordinates_in_unit_cell(self, origin=0, eps=1e-4):
        """
        Wrap atomic coordinates within the supercell.

        Modifies object in place and returns itself.

        Args:
            origin (float):  0 to confine between 0 and 1, -0.5 to confine between -0.5 and 0.5
            eps (float): Tolerance to detect atoms at cell edges

        Returns:
            :class:`pyiron_atomistics.atomistics.structure.atoms.Atoms`: reference to this structure
        """
        return center_coordinates_in_unit_cell(structure=self, origin=origin, eps=eps)

    def create_line_mode_structure(
        self,
        with_time_reversal=True,
        recipe="hpkot",
        threshold=1e-07,
        symprec=1e-05,
        angle_tolerance=-1.0,
    ):
        """
        Uses 'seekpath' to create a new structure with high symmetry points and path for band structure calculations.

        Args:
            with_time_reversal (bool): if False, and the group has no inversion symmetry,
                additional lines are returned as described in the HPKOT paper.
            recipe (str): choose the reference publication that defines the special points and paths.
                Currently, only 'hpkot' is implemented.
            threshold (float): the threshold to use to verify if we are in and edge case
                (e.g., a tetragonal cell, but a==c). For instance, in the tI lattice, if abs(a-c) < threshold,
                a EdgeCaseWarning is issued. Note that depending on the bravais lattice,
                the meaning of the threshold is different (angle, length, …)
            symprec (float): the symmetry precision used internally by SPGLIB
            angle_tolerance (float): the angle_tolerance used internally by SPGLIB

        Returns:
            pyiron.atomistics.structure.atoms.Atoms: new structure
        """
        input_structure = (self.cell, self.get_scaled_positions(), self.indices)
        sp_dict = seekpath.get_path(
            structure=input_structure,
            with_time_reversal=with_time_reversal,
            recipe=recipe,
            threshold=threshold,
            symprec=symprec,
            angle_tolerance=angle_tolerance,
        )

        original_element_list = [el.Abbreviation for el in self.species]
        element_list = [original_element_list[l] for l in sp_dict["primitive_types"]]
        positions = sp_dict["primitive_positions"]
        pbc = self.pbc
        cell = sp_dict["primitive_lattice"]

        struc_new = Atoms(
            elements=element_list, scaled_positions=positions, pbc=pbc, cell=cell
        )

        struc_new._set_high_symmetry_points(sp_dict["point_coords"])
        struc_new._set_high_symmetry_path({"full": sp_dict["path"]})

        return struc_new

    @deprecate("Use Atoms.repeat")
    def set_repeat(self, vec):
        self *= vec

    def repeat_points(self, points, rep, centered=False):
        """
        Return points with repetition given according to periodic boundary conditions

        Args:
            points (np.ndarray/list): xyz vector or list/array of xyz vectors
            rep (int/list/np.ndarray): Repetition in each direction.
                                       If int is given, the same value is used for
                                       every direction
            centered (bool): Whether the original points should be in the center of
                             repeated points.

        Returns:
            (np.ndarray) repeated points
        """
        n = np.array([rep]).flatten()
        if len(n) == 1:
            n = np.tile(n, 3)
        if len(n) != 3:
            raise ValueError("rep must be an integer or a list of 3 integers")
        vector = np.array(points)
        if vector.shape[-1] != 3:
            raise ValueError(
                "points must be an xyz vector or a list/array of xyz vectors"
            )
        if centered and np.mod(n, 2).sum() != 3:
            warnings.warn("When centered, only odd number of repetition should be used")
        v = vector.reshape(-1, 3)
        n_lst = []
        for nn in n:
            if centered:
                n_lst.append(np.arange(nn) - int(nn / 2))
            else:
                n_lst.append(np.arange(nn))
        meshgrid = np.meshgrid(n_lst[0], n_lst[1], n_lst[2])
        v_repeated = np.einsum(
            "ni,ij->nj", np.stack(meshgrid, axis=-1).reshape(-1, 3), self.cell
        )
        v_repeated = v_repeated[:, np.newaxis, :] + v[np.newaxis, :, :]
        return v_repeated.reshape((-1,) + vector.shape)

    def reset_absolute(self, is_absolute):
        raise NotImplementedError("This function was removed!")

    @deprecate(
        "Use Atoms.analyse.pyscal_cna_adaptive() with ovito_compatibility=True instead"
    )
    def analyse_ovito_cna_adaptive(self, mode="total"):
        return self._analyse.pyscal_cna_adaptive(mode=mode, ovito_compatibility=True)

    analyse_ovito_cna_adaptive.__doc__ = Analyse.pyscal_cna_adaptive.__doc__

    @deprecate("Use Atoms.analyse.pyscal_centro_symmetry() instead")
    def analyse_ovito_centro_symmetry(self, num_neighbors=12):
        return self._analyse.pyscal_centro_symmetry(num_neighbors=num_neighbors)

    analyse_ovito_centro_symmetry.__doc__ = Analyse.pyscal_centro_symmetry.__doc__

    @deprecate("Use Atoms.analyse.pyscal_voronoi_volume() instead")
    def analyse_ovito_voronoi_volume(self):
        return self._analyse.pyscal_voronoi_volume()

    analyse_ovito_voronoi_volume.__doc__ = Analyse.pyscal_voronoi_volume.__doc__

    @deprecate("Use Atoms.analyse.pyscal_steinhardt_parameter() instead")
    def analyse_pyscal_steinhardt_parameter(
        self,
        neighbor_method="cutoff",
        cutoff=0,
        n_clusters=2,
        q=(4, 6),
        averaged=False,
        clustering=True,
    ):
        return self._analyse.pyscal_steinhardt_parameter(
            neighbor_method=neighbor_method,
            cutoff=cutoff,
            n_clusters=n_clusters,
            q=q,
            averaged=averaged,
            clustering=clustering,
        )

    analyse_pyscal_steinhardt_parameter.__doc__ = (
        Analyse.pyscal_steinhardt_parameter.__doc__
    )

    @deprecate("Use Atoms.analyse.pyscal_cna_adaptive() instead")
    def analyse_pyscal_cna_adaptive(self, mode="total", ovito_compatibility=False):
        return self._analyse.pyscal_cna_adaptive(
            mode=mode, ovito_compatibility=ovito_compatibility
        )

    analyse_pyscal_cna_adaptive.__doc__ = Analyse.pyscal_cna_adaptive.__doc__

    @deprecate("Use Atoms.analyse.pyscal_centro_symmetry() instead")
    def analyse_pyscal_centro_symmetry(self, num_neighbors=12):
        return self._analyse.pyscal_centro_symmetry(num_neighbors=num_neighbors)

    analyse_pyscal_centro_symmetry.__doc__ = Analyse.pyscal_centro_symmetry.__doc__

    @deprecate("Use Atoms.analyse.pyscal_diamond_structure() instead")
    def analyse_pyscal_diamond_structure(self, mode="total", ovito_compatibility=False):
        return self._analyse.pyscal_diamond_structure(
            mode=mode, ovito_compatibility=ovito_compatibility
        )

    analyse_pyscal_diamond_structure.__doc__ = Analyse.pyscal_diamond_structure.__doc__

    @deprecate("Use Atoms.analyse.pyscal_voronoi_volume() instead")
    def analyse_pyscal_voronoi_volume(self):
        return self._analyse.pyscal_voronoi_volume()

    analyse_pyscal_voronoi_volume.__doc__ = Analyse.pyscal_voronoi_volume.__doc__

    @deprecate("Use get_symmetry()['equivalent_atoms'] instead")
    def analyse_phonopy_equivalent_atoms(self):
        from pyiron_atomistics.atomistics.structure.phonopy import (
            analyse_phonopy_equivalent_atoms,
        )

        return analyse_phonopy_equivalent_atoms(atoms=self)

    def plot3d(
        self,
        mode="NGLview",
        show_cell=True,
        show_axes=True,
        camera="orthographic",
        spacefill=True,
        particle_size=1.0,
        select_atoms=None,
        background="white",
        color_scheme=None,
        colors=None,
        scalar_field=None,
        scalar_start=None,
        scalar_end=None,
        scalar_cmap=None,
        vector_field=None,
        vector_color=None,
        magnetic_moments=False,
        view_plane=np.array([0, 0, 1]),
        distance_from_camera=1.0,
        opacity=1.0,
        height=None,
    ):
        """
        Plot3d relies on NGLView or plotly to visualize atomic structures. Here, we construct a string in the "protein database"

        The final widget is returned. If it is assigned to a variable, the visualization is suppressed until that
        variable is evaluated, and in the meantime more NGL operations can be applied to it to modify the visualization.

        Args:
            mode (str): `NGLView`, `plotly` or `ase`
            show_cell (bool): Whether or not to show the frame. (Default is True.)
            show_axes (bool): Whether or not to show xyz axes. (Default is True.)
            camera (str): 'perspective' or 'orthographic'. (Default is 'perspective'.)
            spacefill (bool): Whether to use a space-filling or ball-and-stick representation. (Default is True, use
                space-filling atoms.)
            particle_size (float): Size of the particles. (Default is 1.)
            select_atoms (numpy.ndarray): Indices of atoms to show, either as integers or a boolean array mask.
                (Default is None, show all atoms.)
            background (str): Background color. (Default is 'white'.)
            color_scheme (str): NGLView color scheme to use. (Default is None, color by element.)
            colors (numpy.ndarray): A per-atom array of HTML color names or hex color codes to use for atomic colors.
                (Default is None, use coloring scheme.)
            scalar_field (numpy.ndarray): Color each atom according to the array value (Default is None, use coloring
                scheme.)
            scalar_start (float): The scalar value to be mapped onto the low end of the color map (lower values are
                clipped). (Default is None, use the minimum value in `scalar_field`.)
            scalar_end (float): The scalar value to be mapped onto the high end of the color map (higher values are
                clipped). (Default is None, use the maximum value in `scalar_field`.)
            scalar_cmap (matplotlib.cm): The colormap to use. (Default is None, giving a blue-red divergent map.)
            vector_field (numpy.ndarray): Add vectors (3 values) originating at each atom. (Default is None, no
                vectors.)
            vector_color (numpy.ndarray): Colors for the vectors (only available with vector_field). (Default is None,
                vectors are colored by their direction.)
            magnetic_moments (bool): Plot magnetic moments as 'scalar_field' or 'vector_field'.
            view_plane (numpy.ndarray): A Nx3-array (N = 1,2,3); the first 3d-component of the array specifies
                which plane of the system to view (for example, [1, 0, 0], [1, 1, 0] or the [1, 1, 1] planes), the
                second 3d-component (if specified, otherwise [1, 0, 0]) gives the horizontal direction, and the third
                component (if specified) is the vertical component, which is ignored and calculated internally. The
                orthonormality of the orientation is internally ensured, and therefore is not required in the function
                call. (Default is np.array([0, 0, 1]), which is view normal to the x-y plane.)
            distance_from_camera (float): Distance of the camera from the structure. Higher = farther away.
                (Default is 14, which also seems to be the NGLView default value.)
            height (int/float/None): height of the plot area in pixel (only
                available in plotly) Default: 600

            Possible NGLView color schemes:
              " ", "picking", "random", "uniform", "atomindex", "residueindex",
              "chainindex", "modelindex", "sstruc", "element", "resname", "bfactor",
              "hydrophobicity", "value", "volume", "occupancy"

        Returns:
            (nglview.NGLWidget): The NGLView widget itself, which can be operated on further or viewed as-is.

        Warnings:
            * Many features only work with space-filling atoms (e.g. coloring by a scalar field).
            * The colour interpretation of some hex codes is weird, e.g. 'green'.
        """
        from structuretoolkit.visualize import plot3d

        return plot3d(
            structure=pyiron_to_ase(self),
            mode=mode,
            show_cell=show_cell,
            show_axes=show_axes,
            camera=camera,
            spacefill=spacefill,
            particle_size=particle_size,
            select_atoms=select_atoms,
            background=background,
            color_scheme=color_scheme,
            colors=colors,
            scalar_field=scalar_field,
            scalar_start=scalar_start,
            scalar_end=scalar_end,
            scalar_cmap=scalar_cmap,
            vector_field=vector_field,
            vector_color=vector_color,
            magnetic_moments=magnetic_moments,
            view_plane=view_plane,
            distance_from_camera=distance_from_camera,
            opacity=opacity,
            height=height,
        )

    def pos_xyz(self):
        """

        Returns:

        """
        x = self.positions[:, 0]
        y = self.positions[:, 1]
        z = self.positions[:, 2]
        return x, y, z

    def scaled_pos_xyz(self):
        """

        Returns:

        """
        xyz = self.get_scaled_positions(wrap=False)
        return xyz[:, 0], xyz[:, 1], xyz[:, 2]

    def get_vertical_length(self, norm_order=2):
        """
        Return the length of the cell in each direction projected on the vector vertical to the
        plane.

        Example:

        For a cell `[[1, 1, 0], [0, 1, 0], [0, 0, 1]]`, this function returns
        `[1., 0.70710678, 1.]` because the first cell vector is projected on the vector vertical
        to the yz-plane (as well as the y component on the xz-plane).

        Args:
            norm_order (int): Norm order (cf. numpy.linalg.norm)
        """
        return np.linalg.det(self.cell) / np.linalg.norm(
            np.cross(np.roll(self.cell, -1, axis=0), np.roll(self.cell, 1, axis=0)),
            axis=-1,
            ord=norm_order,
        )

    def get_extended_positions(
        self, width, return_indices=False, norm_order=2, positions=None
    ):
        """
        Get all atoms in the boundary around the supercell which have a distance
        to the supercell boundary of less than dist

        Args:
            width (float): width of the buffer layer on every periodic box side within which all
                atoms across periodic boundaries are chosen.
            return_indices (bool): Whether or not return the original indices of the appended
                atoms.
            norm_order (float): Order of Lp-norm.
            positions (numpy.ndarray): Positions for which the extended positions are returned.
                If None, the atom positions of the structure are used.

        Returns:
            numpy.ndarray: Positions of all atoms in the extended box, indices of atoms in
                their original option (if return_indices=True)

        """
        if width < 0:
            raise ValueError("Invalid width")
        if positions is None:
            positions = self.positions
        if width == 0:
            if return_indices:
                return positions, np.arange(len(positions))
            return positions
        width /= self.get_vertical_length(norm_order=norm_order)
        rep = 2 * np.ceil(width).astype(int) * self.pbc + 1
        rep = [np.arange(r) - int(r / 2) for r in rep]
        meshgrid = np.meshgrid(rep[0], rep[1], rep[2])
        meshgrid = np.stack(meshgrid, axis=-1).reshape(-1, 3)
        v_repeated = np.einsum("ni,ij->nj", meshgrid, self.cell)
        v_repeated = v_repeated[:, np.newaxis, :] + positions[np.newaxis, :, :]
        v_repeated = v_repeated.reshape(-1, 3)
        indices = np.tile(np.arange(len(positions)), len(meshgrid))
        dist = v_repeated - np.sum(self.cell * 0.5, axis=0)
        dist = (
            np.absolute(np.einsum("ni,ij->nj", dist + 1e-8, np.linalg.inv(self.cell)))
            - 0.5
        )
        check_dist = np.all(dist - width < 0, axis=-1)
        indices = indices[check_dist] % len(positions)
        v_repeated = v_repeated[check_dist]
        if return_indices:
            return v_repeated, indices
        return v_repeated

    @deprecate("Use get_neighbors and call numbers_of_neighbors")
    def get_numbers_of_neighbors_in_sphere(
        self,
        cutoff_radius=10,
        num_neighbors=None,
        id_list=None,
        width_buffer=1.2,
    ):
        """
        Function to compute the maximum number of neighbors in a sphere around each atom.
        Args:
            cutoff_radius (float): Upper bound of the distance to which the search must be done
            num_neighbors (int/None): maximum number of neighbors found
            id_list (list): list of atoms the neighbors are to be looked for
            width_buffer (float): width of the layer to be added to account for pbc.

        Returns:
            (np.ndarray) : for each atom the number of neighbors found in the sphere of radius
                           cutoff_radius (<= num_neighbors if specified)
        """
        return self.get_neighbors(
            cutoff_radius=cutoff_radius,
            num_neighbors=num_neighbors,
            id_list=id_list,
            width_buffer=width_buffer,
        ).numbers_of_neighbors

    @deprecate(allow_ragged="use `mode='ragged'` instead.")
    def get_neighbors(
        self,
        num_neighbors=12,
        tolerance=2,
        id_list=None,
        cutoff_radius=np.inf,
        width_buffer=1.2,
        allow_ragged=None,
        mode="filled",
        norm_order=2,
    ):
        """

        Args:
            num_neighbors (int): number of neighbors
            tolerance (int): tolerance (round decimal points) used for computing neighbor shells
            id_list (list): list of atoms the neighbors are to be looked for
            cutoff_radius (float): Upper bound of the distance to which the search must be done
            width_buffer (float): width of the layer to be added to account for pbc.
            allow_ragged (bool): (Deprecated; use mode) Whether to allow ragged list of arrays or
                rectangular numpy.ndarray filled with np.inf for values outside cutoff_radius
            mode (str): Representation of per-atom quantities (distances etc.). Choose from
                'filled', 'ragged' and 'flattened'.
            norm_order (int): Norm to use for the neighborhood search and shell recognition. The
                definition follows the conventional Lp norm (cf.
                https://en.wikipedia.org/wiki/Lp_space). This is an feature and for anything
                other than norm_order=2, there is no guarantee that this works flawlessly.

        Returns:

            structuretoolkit.analyse.neighbors.Neighbors: Neighbors instances with the neighbor
                indices, distances and vectors

        """
        return get_neighbors(
            structure=self,
            num_neighbors=num_neighbors,
            tolerance=tolerance,
            id_list=id_list,
            cutoff_radius=cutoff_radius,
            width_buffer=width_buffer,
            mode=mode,
            norm_order=norm_order,
        )

    @deprecate(allow_ragged="use `mode='ragged'` instead.")
    @deprecate("Use get_neighbors", version="1.0.0")
    def get_neighbors_by_distance(
        self,
        cutoff_radius=5,
        num_neighbors=None,
        tolerance=2,
        id_list=None,
        width_buffer=1.2,
        allow_ragged=None,
        mode="ragged",
        norm_order=2,
    ):
        return self.get_neighbors(
            cutoff_radius=cutoff_radius,
            num_neighbors=num_neighbors,
            tolerance=tolerance,
            id_list=id_list,
            width_buffer=width_buffer,
            allow_ragged=allow_ragged,
            mode=mode,
            norm_order=norm_order,
        )

    get_neighbors_by_distance.__doc__ = get_neighbors.__doc__

    def get_neighborhood(
        self,
        positions,
        num_neighbors=12,
        cutoff_radius=np.inf,
        width_buffer=1.2,
        mode="filled",
        norm_order=2,
    ):
        """

        Args:
            position: Position in a box whose neighborhood information is analysed
            num_neighbors (int): Number of nearest neighbors
            cutoff_radius (float): Upper bound of the distance to which the search is to be done
            width_buffer (float): Width of the layer to be added to account for pbc.
            mode (str): Representation of per-atom quantities (distances etc.). Choose from
                'filled', 'ragged' and 'flattened'.
            norm_order (int): Norm to use for the neighborhood search and shell recognition. The
                definition follows the conventional Lp norm (cf.
                https://en.wikipedia.org/wiki/Lp_space). This is an feature and for anything
                other than norm_order=2, there is no guarantee that this works flawlessly.

        Returns:

            pyiron.atomistics.structure.atoms.Tree: Neighbors instances with the neighbor
                indices, distances and vectors

        """
        return get_neighborhood(
            structure=self,
            positions=positions,
            num_neighbors=num_neighbors,
            cutoff_radius=cutoff_radius,
            width_buffer=width_buffer,
            mode=mode,
            norm_order=norm_order,
        )

    @deprecate(
        "Use neigh.find_neighbors_by_vector() instead (after calling neigh = structure.get_neighbors())",
        version="1.0.0",
    )
    def find_neighbors_by_vector(
        self, vector, return_deviation=False, num_neighbors=96
    ):
        """
        Args:
            vector (list/np.ndarray): vector by which positions are translated (and neighbors are searched)
            return_deviation (bool): whether to return distance between the expect positions and real positions

        Returns:
            np.ndarray: list of id's for the specified translation

        Example:
            a_0 = 2.832
            structure = pr.create_structure('Fe', 'bcc', a_0)
            id_list = structure.find_neighbors_by_vector([0, 0, a_0])
            # In this example, you get a list of neighbor atom id's at z+=a_0 for each atom.
            # This is particularly powerful for SSA when the magnetic structure has to be translated
            # in each direction.
        """
        neighbors = self.get_neighbors(num_neighbors=num_neighbors)
        return neighbors.find_neighbors_by_vector(
            vector=vector, return_deviation=return_deviation
        )

    @deprecate(
        "Use neigh.get_shell_matrix() instead (after calling neigh = structure.get_neighbors())",
        version="1.0.0",
    )
    def get_shell_matrix(
        self,
        id_list=None,
        chemical_pair=None,
        num_neighbors=100,
        tolerance=2,
        cluster_by_distances=False,
        cluster_by_vecs=False,
    ):
        """
        Shell matrices for pairwise interaction. Note: The matrices are always symmetric, meaning if you
        use them as bilinear operators, you have to divide the results by 2.

        Args:
            chemical_pair (list): pair of chemical symbols (e.g. ['Fe', 'Ni'])

        Returns:
            list of sparse matrices for different shells


        Example:
            from pyiron_atomistics import Project
            structure = Project('.').create_structure('Fe', 'bcc', 2.83).repeat(2)
            J = -0.1 # Ising parameter
            magmoms = 2*np.random.random((len(structure)), 3)-1 # Random magnetic moments between -1 and 1
            neigh = structure.get_neighbors(num_neighbors=8) # Iron first shell
            shell_matrices = neigh.get_shell_matrix()
            print('Energy =', 0.5*J*magmoms.dot(shell_matrices[0].dot(matmoms)))
        """
        neigh_list = self.get_neighbors(
            num_neighbors=num_neighbors, id_list=id_list, tolerance=tolerance
        )
        return neigh_list.get_shell_matrix(
            chemical_pair=chemical_pair,
            cluster_by_distances=cluster_by_distances,
            cluster_by_vecs=cluster_by_vecs,
        )

    def occupy_lattice(self, **qwargs):
        """
        Replaces specified indices with a given species
        """
        new_species = list(np.array(self.species).copy())
        new_indices = np.array(self.indices.copy())
        for key, i_list in qwargs.items():
            el = self._pse.element(key)
            if el.Abbreviation not in [spec.Abbreviation for spec in new_species]:
                new_species.append(el)
                new_indices[i_list] = len(new_species) - 1
            else:
                index = np.argwhere(np.array(new_species) == el).flatten()
                new_indices[i_list] = index
        delete_species_indices = list()
        retain_species_indices = list()
        for i, el in enumerate(new_species):
            if len(np.argwhere(new_indices == i).flatten()) == 0:
                delete_species_indices.append(i)
            else:
                retain_species_indices.append(i)
        for i in delete_species_indices:
            new_indices[new_indices >= i] += -1
        new_species = np.array(new_species)[retain_species_indices]
        self.set_species(new_species)
        self.set_array("indices", new_indices)

    @deprecate(
        "Use neigh.cluster_analysis() instead (after calling neigh = structure.get_neighbors())",
        version="1.0.0",
    )
    def cluster_analysis(
        self, id_list, neighbors=None, radius=None, return_cluster_sizes=False
    ):
        """

        Args:
            id_list:
            neighbors:
            radius:
            return_cluster_sizes:

        Returns:

        """
        if neighbors is None:
            if radius is None:
                neigh = self.get_neighbors(num_neighbors=100)
                indices = np.unique(
                    neigh.shells[0][neigh.shells[0] <= 2], return_index=True
                )[1]
                radius = neigh.distances[0][indices]
                radius = np.mean(radius)
                # print "radius: ", radius
            neighbors = self.get_neighbors_by_distance(cutoff_radius=radius)
        return neighbors.cluster_analysis(
            id_list=id_list, return_cluster_sizes=return_cluster_sizes
        )

    # TODO: combine with corresponding routine in plot3d
    @deprecate(
        "Use neigh.get_bonds() instead (after calling neigh = structure.get_neighbors())",
        version="1.0.0",
    )
    def get_bonds(self, radius=np.inf, max_shells=None, prec=0.1, num_neighbors=20):
        """

        Args:
            radius:
            max_shells:
            prec: minimum distance between any two clusters (if smaller considered to be single cluster)
            num_neighbors:

        Returns:

        """
        neighbors = self.get_neighbors_by_distance(
            cutoff_radius=radius, num_neighbors=num_neighbors
        )
        return neighbors.get_bonds(radius=radius, max_shells=max_shells, prec=prec)

    def get_symmetry(
        self, use_magmoms=False, use_elements=True, symprec=1e-5, angle_tolerance=-1.0
    ):
        """

        Args:
            use_magmoms (bool): Whether to consider magnetic moments (cf.
            get_initial_magnetic_moments())
            use_elements (bool): If False, chemical elements will be ignored
            symprec (float): Symmetry search precision
            angle_tolerance (float): Angle search tolerance

        Returns:
            symmetry (:class:`pyiron.atomistics.structure.symmetry.Symmetry`): Symmetry class


        """
        return get_symmetry(
            structure=self,
            use_magmoms=use_magmoms,
            use_elements=use_elements,
            symprec=symprec,
            angle_tolerance=angle_tolerance,
        )

    @deprecate("Use structure.get_symmetry().symmetrize_vectors()")
    def symmetrize_vectors(
        self,
        vectors,
        use_magmoms=False,
        use_elements=True,
        symprec=1e-5,
        angle_tolerance=-1.0,
    ):
        """
        Symmetrization of natom x 3 vectors according to box symmetries

        Args:
            vectors (ndarray/list): natom x 3 array to symmetrize
            use_magmoms (bool): cf. get_symmetry
            use_elements (bool): cf. get_symmetry
            symprec (float): cf. get_symmetry
            angle_tolerance (float): cf. get_symmetry

        Returns:
            (np.ndarray) symmetrized vectors
        """
        return self.get_symmetry(
            use_magmoms=use_magmoms,
            use_elements=use_elements,
            symprec=symprec,
            angle_tolerance=angle_tolerance,
        ).symmetrize_vectors(vectors=vectors)

    @deprecate("Use structure.get_symmetry().get_arg_equivalent_sites() instead")
    def group_points_by_symmetry(
        self,
        points,
        use_magmoms=False,
        use_elements=True,
        symprec=1e-5,
        angle_tolerance=-1.0,
    ):
        """
        This function classifies the points into groups according to the box symmetry given by
        spglib.

        Args:
            points: (np.array/list) nx3 array which contains positions
            use_magmoms (bool): Whether to consider magnetic moments (cf.
            get_initial_magnetic_moments())
            use_elements (bool): If False, chemical elements will be ignored
            symprec (float): Symmetry search precision
            angle_tolerance (float): Angle search tolerance

        Returns: list of arrays containing geometrically equivalent positions

        It is possible that the original points are not found in the returned list, as the
        positions outsie the box will be projected back to the box.
        """
        return self.get_symmetry(
            use_magmoms=use_magmoms,
            use_elements=use_elements,
            symprec=symprec,
            angle_tolerance=angle_tolerance,
        ).get_arg_equivalent_sites(points)

    @deprecate("Use structure.get_symmetry().get_arg_equivalent_sites() instead")
    def get_equivalent_points(
        self,
        points,
        use_magmoms=False,
        use_elements=True,
        symprec=1e-5,
        angle_tolerance=-1.0,
    ):
        """

        Args:
            points (list/ndarray): 3d vector
            use_magmoms (bool): cf. get_symmetry()
            use_elements (bool): cf. get_symmetry()
            symprec (float): cf. get_symmetry()
            angle_tolerance (float): cf. get_symmetry()

        Returns:
            (ndarray): array of equivalent points with respect to box symmetries
        """
        return self.get_symmetry(
            use_magmoms=use_magmoms,
            use_elements=use_elements,
            symprec=symprec,
            angle_tolerance=angle_tolerance,
        ).get_arg_equivalent_sites(points)

    @deprecate("Use structure.get_symmetry().info instead")
    def get_symmetry_dataset(self, symprec=1e-5, angle_tolerance=-1.0):
        """

        Args:
            symprec:
            angle_tolerance:

        Returns:

        https://atztogo.github.io/spglib/python-spglib.html
        """
        return self.get_symmetry(symprec=symprec, angle_tolerance=angle_tolerance).info

    @deprecate("Use structure.get_symmetry().spacegroup instead")
    def get_spacegroup(self, symprec=1e-5, angle_tolerance=-1.0):
        """

        Args:
            symprec:
            angle_tolerance:

        Returns:

        https://atztogo.github.io/spglib/python-spglib.html
        """
        return self.get_symmetry(
            symprec=symprec, angle_tolerance=angle_tolerance
        ).spacegroup

    @deprecate("Use structure.get_symmetry().refine_cell() instead")
    def refine_cell(self, symprec=1e-5, angle_tolerance=-1.0):
        """

        Args:
            symprec:
            angle_tolerance:

        Returns:

        https://atztogo.github.io/spglib/python-spglib.html
        """
        return self.get_symmetry(
            symprec=symprec, angle_tolerance=angle_tolerance
        ).refine_cell()

    @deprecate(
        "Use structure.get_symmetry().get_primitive_cell(standardize=False) instead"
    )
    def get_primitive_cell(self, symprec=1e-5, angle_tolerance=-1.0):
        """

        Args:
            symprec:
            angle_tolerance:

        Returns:

        """
        return self.get_symmetry(
            symprec=symprec, angle_tolerance=angle_tolerance
        ).get_primitive_cell(standardize=False)

    @deprecate("Use structure.get_symmetry().get_ir_reciprocal_mesh() instead")
    def get_ir_reciprocal_mesh(
        self,
        mesh,
        is_shift=np.zeros(3, dtype="intc"),
        is_time_reversal=True,
        symprec=1e-5,
    ):
        """

        Args:
            mesh:
            is_shift:
            is_time_reversal:
            symprec:

        Returns:

        """
        return self.get_symmetry(symprec=symprec).get_ir_reciprocal_mesh(
            mesh=mesh,
            is_shift=is_shift,
            is_time_reversal=is_time_reversal,
        )

    def get_majority_species(self, return_count=False):
        """
        This function returns the majority species and their number in the box

        Returns:
            number of atoms of the majority species, chemical symbol and chemical index

        """
        el_dict = self.get_number_species_atoms()
        el_num = list(el_dict.values())
        el_name = list(el_dict.keys())
        if np.sum(np.array(el_num) == np.max(el_num)) > 1:
            warnings.warn("There are more than one majority species")
        symbol_to_index = dict(
            zip(self.get_chemical_symbols(), self.get_chemical_indices())
        )
        max_index = np.argmax(el_num)
        return {
            "symbol": el_name[max_index],
            "count": int(np.max(el_num)),
            "index": symbol_to_index[el_name[max_index]],
        }

    def close(self):
        # TODO: implement
        pass

    @deprecate("Use Atoms.analyse.pyscal_voronoi_volume() instead")
    def get_voronoi_volume(self):
        return self._analyse.pyscal_voronoi_volume()

    get_voronoi_volume.__doc__ = Analyse.pyscal_voronoi_volume.__doc__

    def is_skewed(self, tolerance=1.0e-8):
        """
        Check whether the simulation box is skewed/sheared. The algorithm compares the box volume
        and the product of the box length in each direction. If these numbers do not match, the box
        is considered to be skewed and the function returns True

        Args:
            tolerance (float): Relative tolerance above which the structure is considered as skewed

        Returns:
            (bool): Whether the box is skewed or not.
        """
        volume = self.get_volume()
        prod = np.linalg.norm(self.cell, axis=-1).prod()
        if volume > 0:
            if abs(volume - prod) / volume < tolerance:
                return False
        return True

    def find_mic(self, v, vectors=True):
        """
        Find vectors following minimum image convention (mic). In principle this
        function does the same as ase.geometry.find_mic

        Args:
            v (list/numpy.ndarray): 3d vector or a list/array of 3d vectors
            vectors (bool): Whether to return vectors (distances are returned if False)

        Returns: numpy.ndarray of the same shape as input with mic
        """
        return find_mic(structure=self, v=v, vectors=vectors)

    def get_distance(self, a0, a1, mic=True, vector=False):
        """
        Return distance between two atoms.
        Use mic=True to use the Minimum Image Convention.
        vector=True gives the distance vector (from a0 to a1).

        Args:
            a0 (int/numpy.ndarray/list): position or atom ID
            a1 (int/numpy.ndarray/list): position or atom ID
            mic (bool): minimum image convention (True if periodic boundary conditions should be considered)
            vector (bool): True, if instead of distnce the vector connecting the two positions should be returned

        Returns:
            float: distance or vectors in length unit
        """
        from ase.geometry import find_mic

        positions = self.positions
        if isinstance(a0, list) or isinstance(a0, np.ndarray):
            if not (len(a0) == 3):
                raise AssertionError()
            a0 = np.array(a0)
        else:
            a0 = positions[a0]
        if isinstance(a1, list) or isinstance(a1, np.ndarray):
            if not (len(a1) == 3):
                raise AssertionError()
            a1 = np.array(a1)
        else:
            a1 = positions[a1]
        distance = np.array([a1 - a0])
        if mic:
            distance, d_len = find_mic(distance, self.cell, self.pbc)
        else:
            d_len = np.array([np.sqrt((distance**2).sum())])
        if vector:
            return distance[0]

        return d_len[0]

    def get_distances_array(self, p1=None, p2=None, mic=True, vectors=False):
        """
        Return distance matrix of every position in p1 with every position in
        p2. If p2 is not set, it is assumed that distances between all
        positions in p1 are desired. p2 will be set to p1 in this case. If both
        p1 and p2 are not set, the distances between all atoms in the box are
        returned.

        Args:
            p1 (numpy.ndarray/list): Nx3 array of positions
            p2 (numpy.ndarray/list): Nx3 array of positions
            mic (bool): minimum image convention
            vectors (bool): return vectors instead of distances
        Returns:
            numpy.ndarray: NxN if vector=False and NxNx3 if vector=True

        """
        return get_distances_array(
            structure=self, p1=p1, p2=p2, mic=mic, vectors=vectors
        )

    def append(self, atom):
        if isinstance(atom, ASEAtom):
            super(Atoms, self).append(atom=atom)
        else:
            new_atoms = atom.copy()
            if new_atoms.pbc.all() and np.isclose(new_atoms.get_volume(), 0):
                new_atoms.cell = self.cell
                new_atoms.pbc = self.pbc
            self += new_atoms

    def extend(self, other):
        """
        Extend atoms object by appending atoms from *other*. (Extending the ASE function)

        Args:
            other (pyiron_atomistics.atomistics.structure.atoms.Atoms/ase.atoms.Atoms): Structure to append

        Returns:
            pyiron.atomistics.structure.atoms.Atoms: The extended structure

        """
        if isinstance(other, Atom):
            other = self.__class__([other])
        elif isinstance(other, ASEAtom):
            other = self.__class__([ase_to_pyiron_atom(other)])
        if not isinstance(other, Atoms) and isinstance(other, ASEAtoms):
            warnings.warn(
                "Converting ase structure to pyiron before appending the structure"
            )
            other = ase_to_pyiron(other)

        d = self._store_elements.copy()
        d.update(other._store_elements.copy())
        chem, new_indices = np.unique(
            self.get_chemical_symbols().tolist()
            + other.get_chemical_symbols().tolist(),
            return_inverse=True,
        )
        new_species = [d[c] for c in chem]

        super(Atoms, self).extend(other=other)
        if isinstance(other, Atoms):
            if not np.allclose(self.cell, other.cell):
                warnings.warn(
                    "You are adding structures with different cell shapes. "
                    "Taking the cell and pbc of the first structure:{}".format(
                        self.cell
                    )
                )
            if not np.array_equal(self.pbc, other.pbc):
                warnings.warn(
                    "You are adding structures with different periodic boundary conditions. "
                    "Taking the cell and pbc of the first structure:{}".format(
                        self.cell
                    )
                )
            self.set_array("indices", new_indices)
            self.set_species(new_species)
            if not len(set(self.indices)) == len(self.species):
                raise ValueError("Adding the atom instances went wrong!")
        return self

    __iadd__ = extend

    def __copy__(self):
        """
        Copies the atoms object

        Returns:
            atoms_new: A copy of the object

        """
        # Using ASE copy
        atoms_new = super(Atoms, self).copy()
        ase_keys = list(ASEAtoms().__dict__.keys())
        ase_keys.append("_pse")
        # Only copy the non ASE keys
        for key, val in self.__dict__.items():
            if key not in ase_keys:
                atoms_new.__dict__[key] = copy(val)
        atoms_new._analyse = Analyse(atoms_new)
        return atoms_new

    def __delitem__(self, key):
        super().__delitem__(np.array([key]).flatten())
        unique_ind, new_ind = np.unique(self.indices, return_inverse=True)
        self.set_array("indices", new_ind)
        self.set_species(np.array(self.species)[unique_ind])

    def __eq__(self, other):
        return super(Atoms, self).__eq__(other) and np.array_equal(
            self.get_chemical_symbols(), other.get_chemical_symbols()
        )

    def __ne__(self, other):
        return not self == other

    def __getitem__(self, item):
        new_dict = dict()
        if isinstance(item, int):
            for key, value in self.arrays.items():
                if key in ["positions", "numbers", "indices"]:
                    continue
                if item < len(value):
                    if value[item] is not None:
                        new_dict[key] = value[item]
            element = self.species[self.indices[item]]
            index = item
            position = self.positions[item]
            return Atom(
                element=element,
                position=position,
                pse=self._pse,
                index=index,
                atoms=self,
                **new_dict,
            )

        new_array = super(Atoms, self).__getitem__(item)
        new_array.dimension = self.dimension
        if isinstance(item, tuple):
            item = list(item)
        new_species_indices, new_indices = np.unique(
            self.indices[item], return_inverse=True
        )
        new_species = [self.species[ind] for ind in new_species_indices]
        new_array.set_species(new_species)
        new_array.arrays["indices"] = new_indices
        if isinstance(new_array, Atom):
            natoms = len(self)
            if item < -natoms or item >= natoms:
                raise IndexError("Index out of range.")
            new_array.index = item
        return new_array

    def __getattr__(self, item):
        try:
            return self.arrays[item]
        except KeyError:
            return object.__getattribute__(self, item)

    def __dir__(self):
        new_dir = super().__dir__()
        for key in self.arrays.keys():
            new_dir.append(key)
        return new_dir

    # def __len__(self):
    #     return len(self.indices)

    def __repr__(self):
        if len(self) == 0:
            return "[]"
        out_str = ""
        for el, pos in zip(self.get_chemical_symbols(), self.positions):
            out_str += el + ": " + str(pos) + "\n"
        if len(self.get_tags()) > 0:
            tags = self.get_tags()
            out_str += "tags: \n"  # + ", ".join(tags) + "\n"
            for tag in tags:
                if tag in ["positions", "numbers"]:
                    continue
                out_str += "    " + str(tag) + ": " + self.arrays[tag].__str__() + "\n"
        if self.cell is not None:
            out_str += "pbc: " + str(self.pbc) + "\n"
            out_str += "cell: \n"
            out_str += str(self.cell) + "\n"
        return out_str

    def __str__(self):
        return self.get_chemical_formula()

    def __setitem__(self, key, value):
        if isinstance(key, (int, np.integer)):
            old_el = self.species[self.indices[key]]
            if isinstance(value, str):
                el = PeriodicTable().element(value)
            elif isinstance(value, ChemicalElement):
                el = value
            else:
                raise TypeError("value should either be a string or a ChemicalElement.")
            if el != old_el:
                new_species = np.array(self.species).copy()
                if len(self.select_index(old_el)) == 1:
                    if el.Abbreviation not in [
                        spec.Abbreviation for spec in new_species
                    ]:
                        new_species[self.indices[key]] = el
                        self.set_species(list(new_species))
                    else:
                        el_list = np.array([sp.Abbreviation for sp in new_species])
                        ind = np.argwhere(el_list == el.Abbreviation).flatten()[-1]
                        remove_index = self.indices[key]
                        new_species = list(new_species)
                        del new_species[remove_index]
                        self.indices[key] = ind
                        self.indices[self.indices > remove_index] -= 1
                        self.set_species(new_species)
                else:
                    if el.Abbreviation not in [
                        spec.Abbreviation for spec in new_species
                    ]:
                        new_species = list(new_species)
                        new_species.append(el)
                        self.set_species(new_species)
                        self.indices[key] = len(new_species) - 1
                    else:
                        el_list = np.array([sp.Abbreviation for sp in new_species])
                        ind = np.argwhere(el_list == el.Abbreviation).flatten()[-1]
                        self.indices[key] = ind
        elif isinstance(key, slice) or isinstance(key, (list, tuple, np.ndarray)):
            if not isinstance(key, slice):
                if hasattr(key, "__len__"):
                    if len(key) == 0:
                        return
            else:
                # Generating the correct numpy array from a slice input
                if key.start is None:
                    start_val = 0
                elif key.start < 0:
                    start_val = key.start + len(self)
                else:
                    start_val = key.start
                if key.stop is None:
                    stop_val = len(self)
                elif key.stop < 0:
                    stop_val = key.stop + len(self)
                else:
                    stop_val = key.stop
                if key.step is None:
                    step_val = 1
                else:
                    step_val = key.step
                key = np.arange(start_val, stop_val, step_val)
            if isinstance(value, (str, int, np.integer)):
                el = PeriodicTable().element(value)
            elif isinstance(value, ChemicalElement):
                el = value
            else:
                raise ValueError(
                    "The value assigned should be a string, integer or a ChemicalElement instance"
                )
            replace_list = list()
            new_species = list(np.array(self.species).copy())

            for sp in self.species:
                replace_list.append(
                    np.array_equal(
                        np.sort(self.select_index(sp)),
                        np.sort(np.intersect1d(self.select_index(sp), key)),
                    )
                )
            if el.Abbreviation not in [spec.Abbreviation for spec in new_species]:
                if not any(replace_list):
                    new_species.append(el)
                    self.set_species(new_species)
                    self.indices[key] = len(new_species) - 1
                else:
                    replace_ind = np.where(replace_list)[0][0]
                    new_species[replace_ind] = el
                    if len(np.where(replace_list)[0]) > 1:
                        for ind in replace_list[1:]:
                            del new_species[ind]
                    self.set_species(new_species)
                    self.indices[key] = replace_ind
            else:
                el_list = np.array([sp.Abbreviation for sp in new_species])
                ind = np.argwhere(el_list == el.Abbreviation).flatten()[-1]
                if not any(replace_list):
                    self.set_species(new_species)
                    self.indices[key] = ind
                else:
                    self.indices[key] = ind
                    delete_indices = list()
                    new_indices = self.indices.copy()
                    for i, rep in enumerate(replace_list):
                        if i != ind and rep:
                            delete_indices.append(i)
                            # del new_species[i]
                            new_indices[new_indices >= i] -= 1
                    self.set_array("indices", new_indices.copy())
                    new_species = np.array(new_species)[
                        np.setdiff1d(np.arange(len(new_species)), delete_indices)
                    ].tolist()
                    self.set_species(new_species)
        else:
            raise NotImplementedError()
        # For ASE compatibility
        self.numbers = self.get_atomic_numbers()

    @staticmethod
    def convert_formula(elements):
        """
        Convert a given chemical formula into a list of elements

        Args:
            elements (str): A string of the required chamical formula (eg. H2O)

        Returns:
            list: A list of elements corresponding to the formula

        """
        el_list = []
        num_list = ""
        for i, char in enumerate(elements):
            is_last = i == len(elements) - 1
            if len(num_list) > 0:
                if (not char.isdigit()) or is_last:
                    el_fac = ast.literal_eval(num_list) * el_list[-1]
                    for el in el_fac[1:]:
                        el_list.append(el)
                    num_list = ""

            if char.isupper():
                el_list.append(char)
            elif char.islower():
                el_list[-1] += char
            elif char.isdigit():
                num_list += char

            if len(num_list) > 0:
                # print "num_list: ", el_list, num_list, el_list[-1], (not char.isdigit()) or is_last
                if (not char.isdigit()) or is_last:
                    el_fac = ast.literal_eval(num_list) * [el_list[-1]]
                    # print "el_fac: ", el_fac
                    for el in el_fac[1:]:
                        el_list.append(el)
                    num_list = ""

        return el_list

    def get_constraint(self):
        if "selective_dynamics" in self.arrays.keys():
            from ase.constraints import FixAtoms

            return FixAtoms(
                indices=[
                    atom_ind
                    for atom_ind in range(len(self))
                    if not any(self.selective_dynamics[atom_ind])
                ]
            )
        else:
            return None

    def set_constraint(self, constraint=None):
        super(Atoms, self).set_constraint(constraint)
        if constraint is not None:
            if constraint.todict()["name"] != "FixAtoms":
                raise ValueError(
                    "Only FixAtoms is supported as ASE compatible constraint."
                )
            if "selective_dynamics" not in self.arrays.keys():
                self.add_tag(selective_dynamics=None)
            for atom_ind in range(len(self)):
                if atom_ind in constraint.index:
                    self.selective_dynamics[atom_ind] = [False, False, False]
                else:
                    self.selective_dynamics[atom_ind] = [True, True, True]

    def apply_strain(self, epsilon, return_box=False, mode="linear"):
        """
        Apply a given strain on the structure. It applies the matrix `F` in the manner:

        ```
            new_cell = F @ current_cell
        ```

        Args:
            epsilon (float/list/ndarray): epsilon matrix. If a single number is set, the same
                strain is applied in each direction. If a 3-dim vector is set, it will be
                multiplied by a unit matrix.
            return_box (bool): whether to return a box. If set to True, only the returned box will
                have the desired strain and the original box will stay unchanged.
            mode (str): `linear` or `lagrangian`. If `linear`, `F` is equal to the epsilon - 1.
                If `lagrangian`, epsilon is given by `(F^T * F - 1) / 2`. It raises an error if
                the strain is not symmetric (if the shear components are given).
        """
        epsilon = np.array([epsilon]).flatten()
        if len(epsilon) == 3 or len(epsilon) == 1:
            epsilon = epsilon * np.eye(3)
        epsilon = epsilon.reshape(3, 3)
        if epsilon.min() < -1.0:
            raise ValueError("Strain value too negative")
        if return_box:
            structure_copy = self.copy()
        else:
            structure_copy = self
        cell = structure_copy.cell.copy()
        if mode == "linear":
            F = epsilon + np.eye(3)
        elif mode == "lagrangian":
            if not np.allclose(epsilon, epsilon.T):
                raise ValueError("Strain must be symmetric if `mode = 'lagrangian'`")
            E, V = np.linalg.eigh(2 * epsilon + np.eye(3))
            F = np.einsum("ik,k,jk->ij", V, np.sqrt(E), V)
        else:
            raise ValueError("mode must be `linear` or `lagrangian`")
        cell = np.matmul(F, cell)
        structure_copy.set_cell(cell, scale_atoms=True)
        if return_box:
            return structure_copy

    def get_spherical_coordinates(self, x=None):
        """
        Args:
            x (list/ndarray): coordinates to transform. If not set, the positions
                              in structure will be returned.

        Returns:
            array in spherical coordinates
        """
        if x is None:
            x = self.positions.copy()
        x = np.array(x).reshape(-1, 3)
        r = np.linalg.norm(x, axis=-1)
        phi = np.arctan2(x[:, 2], x[:, 1])
        theta = np.arctan2(np.linalg.norm(x[:, :2], axis=-1), x[:, 2])
        return np.stack((r, theta, phi), axis=-1)

    def get_initial_magnetic_moments(self):
        """
        Get array of initial magnetic moments.

        Returns:
            numpy.array()
        """
        try:
            return self.arrays["spin"]
        except KeyError:
            spin_lst = [
                element.tags["spin"] if "spin" in element.tags.keys() else None
                for element in self.get_chemical_elements()
            ]
            if any(spin_lst):
                if (
                    (
                        isinstance(spin_lst, str)
                        or (
                            isinstance(spin_lst, (list, np.ndarray))
                            and isinstance(spin_lst[0], str)
                        )
                    )
                    and list(set(spin_lst))[0] is not None
                    and "[" in list(set(spin_lst))[0]
                ):
                    return np.array(
                        [
                            (
                                [
                                    float(spin_dir)
                                    for spin_dir in spin.replace("[", "")
                                    .replace("]", "")
                                    .replace(",", "")
                                    .split()
                                ]
                                if spin
                                else [0.0, 0.0, 0.0]
                            )
                            for spin in spin_lst
                        ]
                    )
                elif isinstance(spin_lst, (list, np.ndarray)):
                    return np.array([float(s) if s else 0.0 for s in spin_lst])
                else:
                    return np.array([float(spin) if spin else 0.0 for spin in spin_lst])
            else:
                return np.zeros(len(self))

    def set_initial_magnetic_moments(self, magmoms=None):
        """
        Set array of initial magnetic moments.

        Args:
            magmoms (None/numpy.ndarray/list/dict/float): Default value is
                None (non magnetic calc). List, dict or single value assigning
                magnetic moments to the structure object.

        Non-collinear calculations may be specified through using a dict/list
        (see last example)

        If you want to make it non-magnetic, set `None`
        >>> structure.set_initial_magnetic_moments(None)

        Example I input: np.ndarray / List
        Assigns site moments via corresponding list of same length as number
        of sites in structure
        >>> from pyiron_atomistics import Project
        >>> structure = Project('.').create.structure.bulk('Ni', cubic=True)
        >>> structure[-1] = 'Fe'
        >>> spin_list = [1, 2, 3, 4]
        >>> structure.set_initial_magnetic_moments(spin_list)
        >>> structure.get_initial_magnetic_moments()
        array([1, 2, 3, 4])

        Example II input: dict
        Assigns species-specific magnetic moments
        >>> from pyiron_atomistics import Project
        >>> structure = Project('.').create.structure.bulk('Ni', cubic=True)
        >>> structure[-1] = 'Fe'
        >>> spin_dict = {'Fe': 1, 'Ni': 2}
        >>> structure.set_initial_magnetic_moments(spin_dict)
        >>> structure.get_initial_magnetic_moments()
        array([2, 2, 2, 1])

        Example III input: float
        Assigns the same magnetic moment to all sites in the structure
        >>> from pyiron_atomistics import Project
        >>> structure = Project('.').create.structure.bulk('Ni', cubic=True)
        >>> structure[-1] = 'Fe'
        >>> structure.set_initial_magnetic_moments(1)
        >>> print(structure.get_initial_magnetic_moments())
        array([1, 1, 1, 1])

        Example IV input: dict/list for non-collinear magmoms.
        Assigns non-collinear magnetic moments to the sites in structure
        >>> from pyiron_atomistics import Project
        >>> structure = Project('.').create.structure.bulk('Ni', cubic=True)
        >>> structure[-1] = 'Fe'

        Option 1: List input sets vectors for each individual site
        >>> non_coll_magmom_vect = [[1, 2, 3]
                                    [2, 3, 4],
                                    [3, 4, 5],
                                    [4, 5, 6]]
        >>> structure.set_initial_magnetic_moments(non_coll_magmom_vect)
        >>> print(structure.get_initial_magnetic_moments())
        array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]])

        Option 2: Dict input sets magmom vectors for individual species:
        >>> print(structure.get_initial_magnetic_moments())
        >>> non_coll_spin_dict = {'Fe': [2, 3, 4], 'Ni': [1, 2, 3]}
        >>> structure.set_initial_magnetic_moments(non_coll_spin_dict)
        >>> print(structure.get_initial_magnetic_moments())
        array([[1, 2, 3], [1, 2, 3], [1, 2, 3], [2, 3, 4]])
        """
        # pyiron part
        if magmoms is not None:
            if isinstance(magmoms, dict):
                if set(self.get_species_symbols()) != set(magmoms.keys()):
                    raise ValueError(
                        "Elements in structure {} not found in dict {}".format(
                            set(self.get_chemical_symbols()), set(magmoms.keys())
                        )
                    )
                magmoms = [magmoms[c] for c in self.get_chemical_symbols()]
            elif not isinstance(magmoms, (np.ndarray, Sequence)):
                magmoms = len(self) * [magmoms]
            if len(magmoms) != len(self):
                raise ValueError("magmoms can be collinear or non-collinear.")
            self.set_array("spin", None)
            self.set_array("spin", np.array(magmoms))
        self.spins = magmoms  # For self.array['initial_magmoms']

    def rotate(
        self, a=0.0, v=None, center=(0, 0, 0), rotate_cell=False, index_list=None
    ):
        """
        Rotate atoms based on a vector and an angle, or two vectors. This function is completely adopted from ASE code
        (https://wiki.fysik.dtu.dk/ase/_modules/ase/atoms.html#Atoms.rotate)

        Args:

            a (float/list) in degrees = None:
                Angle that the atoms is rotated around the vecor 'v'. If an angle
                is not specified, the length of 'v' is used as the angle
                (default). The angle can also be a vector and then 'v' is rotated
                into 'a'.

            v (list/numpy.ndarray/string):
                Vector to rotate the atoms around. Vectors can be given as
                strings: 'x', '-x', 'y', ... .

            center (tuple/list/numpy.ndarray/str): The center is kept fixed under the rotation. Use 'COM' to fix
                                                the center of mass, 'COP' to fix the center of positions or
                                                'COU' to fix the center of cell.

            rotate_cell = False:
                If true the cell is also rotated.

            index_list (list/numpy.ndarray):
                Indices of atoms to be rotated

        Examples:

        Rotate 90 degrees around the z-axis, so that the x-axis is
        rotated into the y-axis:

        >>> atoms = Atoms()
        >>> atoms.rotate(90, 'z')
        >>> atoms.rotate(90, (0, 0, 1))
        >>> atoms.rotate(-90, '-z')
        >>> atoms.rotate('x', 'y')
        >>> atoms.rotate((1, 0, 0), (0, 1, 0))
        """
        if index_list is None:
            super(Atoms, self).rotate(a=a, v=v, center=center, rotate_cell=rotate_cell)
        else:
            dummy_basis = copy(self)
            dummy_basis.rotate(a=a, v=v, center=center, rotate_cell=rotate_cell)
            self.positions[index_list] = dummy_basis.positions[index_list]

    def to_ase(self):
        return pyiron_to_ase(self)

    def to_pymatgen(self):
        return pyiron_to_pymatgen(self)

    def to_ovito(self):
        return pyiron_to_ovito(self)

    def to_pyscal_system(self):
        return pyiron_to_pyscal_system(self)


class _CrystalStructure(Atoms):
    """
    only for historical reasons

    Args:
        element:
        BravaisLattice:
        BravaisBasis:
        LatticeConstants:
        Dimension:
        relCoords:
        PSE:
        **kwargs:
    """

    def __init__(
        self,
        element="Fe",
        bravais_lattice="cubic",
        bravais_basis="primitive",
        lattice_constants=None,  # depending on symmetry length and angles
        dimension=3,
        rel_coords=True,
        pse=None,
        **kwargs,
    ):
        # print "basis0"
        # allow also for scalar input for LatticeConstants (for a cubic system)
        if lattice_constants is None:
            lattice_constants = [1.0]
        try:
            test = lattice_constants[0]
        except (TypeError, IndexError):
            lattice_constants = [lattice_constants]
        self.bravais_lattice = bravais_lattice
        self.bravais_basis = bravais_basis
        self.lattice_constants = lattice_constants
        self.dimension = dimension
        self.relCoords = rel_coords
        self.element = element

        self.__updateCrystal__(pse)

        self.crystalParamsDict = {
            "BravaisLattice": self.bravais_lattice,
            "BravaisBasis": self.bravais_basis,
            "LatticeConstants": self.lattice_constants,
        }

        self.crystal_lattice_dict = {
            3: {
                "cubic": ["fcc", "bcc", "primitive"],
                "hexagonal": ["primitive", "hcp"],
                "monoclinic": ["primitive", "base-centered"],
                "triclinic": ["primitive"],
                "orthorombic": [
                    "primitive",
                    "body-centered",
                    "base-centered",
                    "face-centered",
                ],
                "tetragonal": ["primitive", "body-centered"],
                "rhombohedral": ["primitive"],
            },
            2: {
                "oblique": ["primitive"],
                "rectangular": ["primitive", "centered"],
                "hexagonal": ["primitive"],
                "square": ["primitive"],
            },
            1: {"line": ["primitive"]},
        }

        # init structure for lattice parameters alat, blat, clat, alpha, beta, gamma
        self.crystalLatticeParams = {
            3: {
                "cubic": [1.0],
                "hexagonal": [1.0, 2.0],
                "monoclinic": [1.0, 1.0, 1.0, 90.0],
                "triclinic": [1.0, 2.0, 3.0, 90.0, 90.0, 90.0],
                "orthorombic": [1.0, 1.0, 1.0],
                "tetragonal": [1.0, 2.0],
                "rhombohedral": [1.0, 90.0, 90.0, 90.0],
            },
            2: {
                "oblique": [1.0, 1.0, 90.0],
                "rectangular": [1.0, 1.0],
                "hexagonal": [1.0],
                "square": [1.0],
            },
            1: {"line": [1.0]},
        }

        # print "basis"
        super(_CrystalStructure, self).__init__(
            elements=self.ElementList,
            scaled_positions=self.coordinates,
            cell=self.amat,  # tag = "Crystal",
            pbc=[True, True, True][0 : self.dimension],
        )

    # ## private member functions
    def __updateCrystal__(self, pse=None):
        """

        Args:
            pse:

        Returns:

        """
        self.__updateAmat__()
        self.__updateCoordinates__()
        self.__updateElementList__(pse)

    def __updateAmat__(self):  # TODO: avoid multi-call of this function
        """

        Returns:

        """
        # print "lat constants (__updateAmat__):", self.LatticeConstants
        a_lat = self.lattice_constants[0]

        if self.dimension == 3:
            alpha = None
            beta = None
            gamma = None
            b_lat, c_lat = None, None
            if self.bravais_lattice == "cubic":
                b_lat = c_lat = a_lat
                alpha = beta = gamma = 90 / 180.0 * np.pi  # 90 degrees
            elif self.bravais_lattice == "tetragonal":
                b_lat = a_lat
                c_lat = self.lattice_constants[1]
                alpha = beta = gamma = 0.5 * np.pi  # 90 degrees
            elif self.bravais_lattice == "triclinic":
                b_lat = self.lattice_constants[1]
                c_lat = self.lattice_constants[2]
                alpha = self.lattice_constants[3] / 180.0 * np.pi
                beta = self.lattice_constants[4] / 180.0 * np.pi
                gamma = self.lattice_constants[5] / 180.0 * np.pi
            elif self.bravais_lattice == "hexagonal":
                b_lat = a_lat
                c_lat = self.lattice_constants[1]
                alpha = 60.0 / 180.0 * np.pi  # 60 degrees
                beta = gamma = 0.5 * np.pi  # 90 degrees
            elif self.bravais_lattice == "orthorombic":
                b_lat = self.lattice_constants[1]
                c_lat = self.lattice_constants[2]
                alpha = beta = gamma = 0.5 * np.pi  # 90 degrees
            elif self.bravais_lattice == "rhombohedral":
                b_lat = a_lat
                c_lat = a_lat
                alpha = self.lattice_constants[1] / 180.0 * np.pi
                beta = self.lattice_constants[2] / 180.0 * np.pi
                gamma = self.lattice_constants[3] / 180.0 * np.pi
            elif self.bravais_lattice == "monoclinic":
                b_lat = self.lattice_constants[1]
                c_lat = self.lattice_constants[2]
                alpha = 0.5 * np.pi
                beta = self.lattice_constants[3] / 180.0 * np.pi
                gamma = 0.5 * np.pi

            b1 = np.cos(alpha)
            b2 = np.sin(alpha)
            c1 = np.cos(beta)
            c2 = (np.cos(gamma) - np.cos(beta) * np.cos(alpha)) / np.sin(alpha)
            self.amat = np.array(
                [
                    [a_lat, 0.0, 0.0],
                    [b_lat * b1, b_lat * b2, 0.0],
                    [c_lat * c1, c_lat * c2, c_lat * np.sqrt(1 - c2 * c2 - c1 * c1)],
                ]
            )
        elif self.dimension == 2:  # TODO not finished yet
            self.amat = a_lat * np.array([[1.0, 0.0], [0.0, 1.0]])
            if self.bravais_lattice == "rectangular":
                b_lat = self.lattice_constants[1]
                self.amat = np.array([[a_lat, 0.0], [0.0, b_lat]])
        elif self.dimension == 1:
            self.amat = a_lat * np.array([[1.0]])
        else:
            raise ValueError("Bravais lattice not defined!")

    def __updateElementList__(self, pse=None):
        """

        Args:
            pse:

        Returns:

        """
        self.ElementList = len(self.coordinates) * [self.element]

    def __updateCoordinates__(self):
        """

        Returns:

        """
        # if relative coordinates
        basis = None
        if self.dimension == 3:
            if self.bravais_basis == "fcc" or self.bravais_basis == "face-centered":
                basis = np.array(
                    [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]]
                )
            elif self.bravais_basis == "body-centered" or self.bravais_basis == "bcc":
                basis = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
            elif self.bravais_basis == "base-centered":
                basis = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]])
            elif self.bravais_basis == "hcp":
                # basis = r([[0.0,-1./np.sqrt(3.),np.sqrt(8./3.)]])
                # a = self.LatticeConstants[0]
                # c = self.LatticeConstants[1]
                basis = np.array([[0.0, 0.0, 0.0], [1.0 / 3.0, 1.0 / 3.0, 1.0 / 2.0]])
                # basis = np.dot(basis,np.linalg.inv(self.amat))
            elif self.bravais_basis == "primitive":
                basis = np.array([[0.0, 0.0, 0.0]])
            else:
                raise ValueError(
                    "Only fcc, bcc, hcp, base-centered, body-centered and primitive cells are supported for 3D."
                )
        elif self.dimension == 2:
            if self.bravais_basis == "primitive":
                basis = np.array([[0.0, 0.0]])
            elif self.bravais_basis == "centered":
                basis = np.array([[0.0, 0.0], [0.5, 0.5]])
            else:
                raise ValueError(
                    "Only centered and primitive cells are supported for 2D."
                )
        elif self.dimension == 1:
            if self.bravais_basis == "primitive":
                basis = np.array([[0.0]])
            else:
                raise ValueError("Only primitive cells are supported for 1D.")
        self.coordinates = basis

    # ########################### get commmands ########################
    def get_lattice_types(self):
        """

        Returns:

        """
        self.crystal_lattice_dict[self.dimension].keys().sort()
        return self.crystal_lattice_dict[self.dimension].keys()

    def get_dimension_of_lattice_parameters(self):
        """

        Returns:

        """
        # print "getDimensionOfLatticeParameters"
        counter = 0
        for k in self.get_needed_lattice_parameters():
            if k:
                counter += 1
        return counter

    def get_needed_lattice_parameters(self):
        """

        Returns:

        """
        # print "call: getNeededLatticeParams"
        needed_params = [True, False, False, False, False, False]
        if self.dimension == 3:
            if self.bravais_lattice == "cubic":
                needed_params = [
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                ]  # stands for alat, blat, clat, alpha, beta, gamma
            elif self.bravais_lattice == "triclinic":
                needed_params = [True, True, True, True, True, True]
            elif self.bravais_lattice == "monoclinic":
                needed_params = [True, True, True, True, False, False]
            elif self.bravais_lattice == "orthorombic":
                needed_params = [True, True, True, False, False, False]
            elif self.bravais_lattice == "tetragonal":
                needed_params = [True, False, True, False, False, False]
            elif self.bravais_lattice == "rhombohedral":
                needed_params = [True, False, False, True, True, True]
            elif self.bravais_lattice == "hexagonal":
                needed_params = [True, False, True, False, False, False]
        elif self.dimension == 2:
            if self.bravais_lattice == "oblique":
                needed_params = [True, True, False, True, False, False]
            elif self.bravais_lattice == "rectangular":
                needed_params = [True, True, False, False, False, False]
            elif self.bravais_lattice == "hexagonal":
                needed_params = [True, False, False, False, False, False]
            elif self.bravais_lattice == "square":
                needed_params = [True, False, False, False, False, False]
            else:  # TODO: need to be improved
                needed_params = [True, False, False, False, False, False]
        elif self.dimension == 1:
            if self.bravais_lattice == "line":
                needed_params = [True, False, False, False, False, False]
            else:  # TODO: improval needed
                needed_params = [True, False, False, False, False, False]
        else:
            raise ValueError("inconsistency in lattice structures")

        return needed_params

    def get_basis_types(self):
        """

        Returns:

        """
        self.crystal_lattice_dict[self.dimension].get(self.bravais_lattice).sort()
        return self.crystal_lattice_dict[self.dimension].get(self.bravais_lattice)

    def get_initial_lattice_constants(self):
        """

        Returns:

        """
        self.crystalLatticeParams[self.dimension].get(self.bravais_lattice).sort()
        return (
            self.crystalLatticeParams[self.dimension].get(self.bravais_lattice).sort()
        )

    # def getDimension(self):
    #     return self.dimension

    # def getCoordinates(self):
    #     return self.coordinates

    # def getCell(self):
    #     return self.amat

    def get_atom_structure(self, rel=True):
        """

        Args:
            rel:

        Returns:

        """
        #        print self.relCoords, self.amat
        return Atoms(
            elementList=self.ElementList,
            coordinates=self.coordinates,
            amat=self.amat,
            tag="Crystal",
            rel=rel,  # self.relCoords, #rel, # true or false # coordinates are given in relative lattice units
            pbc=[True, True, True][0 : self.dimension],
            Crystal=self.crystalParamsDict,
        )

    # #################### set commands #########################
    def set_lattice_constants(self, lattice_constants=None):
        """

        Args:
            lattice_constants:

        Returns:

        """
        if lattice_constants is None:
            lattice_constants = [1.0]
        for k in lattice_constants:
            if k <= 0:
                raise ValueError("negative lattice parameter(s)")
        self.lattice_constants = lattice_constants
        self.__updateCrystal__()

    def set_element(self, element="Fe"):
        """

        Args:
            element:

        Returns:

        """
        self.element = element
        self.__updateCrystal__()

    def set_dimension(self, dim=3):
        """

        Args:
            dim:

        Returns:

        """
        self.dimension = dim
        length = self.get_dimension_of_lattice_parameters()
        if dim == 3:  # # initial 3d structure
            self.lattice_constants = length * [1.0]
            self.bravais_lattice = "cubic"
            self.bravais_basis = "primitive"
        elif dim == 2:  # # initial 2d structure
            self.lattice_constants = length * [1.0]
            self.bravais_lattice = "square"
            self.bravais_basis = "primitive"
        elif dim == 1:  # # initial 1d structure
            self.lattice_constants = length * [1.0]
            self.bravais_lattice = "line"
            self.bravais_basis = "primitive"
        self.__updateCrystal__()

    def set_lattice_type(self, name_lattice="cubic"):
        """

        Args:
            name_lattice:

        Returns:

        """
        # catch input error
        # print "lattice type =", name_lattice
        if name_lattice not in self.get_lattice_types():
            raise ValueError("is not item of ")
        else:
            self.bravais_lattice = name_lattice
            self.set_lattice_constants(
                self.get_dimension_of_lattice_parameters() * [1.0]
            )
            self.set_basis_type(
                name_basis=self.crystal_lattice_dict[self.dimension].get(name_lattice)[
                    0
                ]
            )  # initial basis type

        self.__updateCrystal__()

    def set_basis_type(self, name_basis="primitive"):
        """

        Args:
            name_basis:

        Returns:

        """
        if name_basis not in self.get_basis_types():
            raise ValueError("is not item of")
        else:
            self.bravais_basis = name_basis
        self.__updateCrystal__()

    def atoms(self):
        """

        Returns:

        """
        return Atoms(
            elements=self.ElementList,
            scaled_positions=self.coordinates,
            cell=self.amat,
            pbc=[True, True, True][0 : self.dimension],
        )


class CrystalStructure(object):
    def __new__(cls, *args, **kwargs):
        basis = _CrystalStructure(*args, **kwargs).atoms()
        return basis


def ase_to_pyiron(ase_obj):
    """
    Convert an ase.atoms.Atoms structure object to its equivalent pyiron structure

    Args:
        ase_obj(ase.atoms.Atoms): The ase atoms instance to convert

    Returns:
        pyiron.atomistics.structure.atoms.Atoms: The equivalent pyiron structure

    """
    element_list = ase_obj.get_chemical_symbols()
    cell = ase_obj.cell
    positions = ase_obj.get_positions()
    pbc = ase_obj.get_pbc()
    spins = ase_obj.get_initial_magnetic_moments()
    if all(spins == np.array(None)) or sum(np.abs(spins)) == 0.0:
        pyiron_atoms = Atoms(
            elements=element_list, positions=positions, pbc=pbc, cell=cell
        )
    else:
        if any(spins == np.array(None)):
            spins[spins == np.array(None)] = 0.0
        pyiron_atoms = Atoms(
            elements=element_list,
            positions=positions,
            pbc=pbc,
            cell=cell,
            magmoms=spins,
        )
    if hasattr(ase_obj, "constraints") and len(ase_obj.constraints) != 0:
        for constraint in ase_obj.constraints:
            constraint_dict = constraint.todict()
            if constraint_dict["name"] == "FixAtoms":
                if "selective_dynamics" not in pyiron_atoms.arrays.keys():
                    pyiron_atoms.add_tag(selective_dynamics=[True, True, True])
                pyiron_atoms.selective_dynamics[
                    constraint_dict["kwargs"]["indices"]
                ] = [False, False, False]
            elif constraint_dict["name"] == "FixScaled":
                if "selective_dynamics" not in pyiron_atoms.arrays.keys():
                    pyiron_atoms.add_tag(selective_dynamics=[True, True, True])
                pyiron_atoms.selective_dynamics[constraint_dict["kwargs"]["a"]] = (
                    np.invert(constraint_dict["kwargs"]["mask"])
                )
            elif constraint_dict["name"] == "FixCartesian":
                if "selective_dynamics" not in pyiron_atoms.arrays.keys():
                    pyiron_atoms.add_tag(selective_dynamics=[True, True, True])
                pyiron_atoms.selective_dynamics[constraint_dict["kwargs"]["a"]] = (
                    np.invert(constraint_dict["kwargs"]["mask"])
                )
            else:
                warnings.warn("Unsupported ASE constraint: " + constraint_dict["name"])
    return pyiron_atoms


def pyiron_to_ase(pyiron_obj):
    element_list = pyiron_obj.get_parent_symbols()
    cell = pyiron_obj.cell
    positions = pyiron_obj.positions
    pbc = pyiron_obj.get_pbc()
    spins = pyiron_obj.get_initial_magnetic_moments()
    if all(spins == np.array(None)) or sum(np.abs(spins)) == 0.0:
        atoms = ASEAtoms(symbols=element_list, positions=positions, pbc=pbc, cell=cell)
    else:
        if any(spins == np.array(None)):
            spins[spins == np.array(None)] = 0.0
        atoms = ASEAtoms(
            symbols=element_list, positions=positions, pbc=pbc, cell=cell, magmoms=spins
        )
    if "selective_dynamics" in pyiron_obj.get_tags():
        constraints_dict = {
            label: []
            for label in ["TTT", "TTF", "FTT", "TFT", "TFF", "FFT", "FTF", "FFF"]
        }
        for i, val in enumerate(pyiron_obj.selective_dynamics):
            if val[0] and val[1] and val[2]:
                constraints_dict["TTT"].append(i)
            elif val[0] and val[1] and not val[2]:
                constraints_dict["TTF"].append(i)
            elif not val[0] and val[1] and val[2]:
                constraints_dict["FTT"].append(i)
            elif val[0] and not val[1] and val[2]:
                constraints_dict["TFT"].append(i)
            elif val[0] and not val[1] and not val[2]:
                constraints_dict["TFF"].append(i)
            elif not val[0] and val[1] and not val[2]:
                constraints_dict["FTF"].append(i)
            elif not val[0] and not val[1] and val[2]:
                constraints_dict["FFT"].append(i)
            elif not val[0] and not val[1] and not val[2]:
                constraints_dict["FFF"].append(i)
            else:
                raise ValueError("Selective Dynamics Error: " + str(val))

        constraints_lst = []
        for k, v in constraints_dict.items():
            if len(v) > 0:
                if k == "TTT":
                    constraints_lst.append(
                        FixCartesian(a=v, mask=(False, False, False))
                    )
                elif k == "TTF":
                    constraints_lst.append(FixCartesian(a=v, mask=(False, False, True)))
                elif k == "FTT":
                    constraints_lst.append(FixCartesian(a=v, mask=(True, False, False)))
                elif k == "TFT":
                    constraints_lst.append(FixCartesian(a=v, mask=(False, True, False)))
                elif k == "TFF":
                    constraints_lst.append(FixCartesian(a=v, mask=(False, True, True)))
                elif k == "FTF":
                    constraints_lst.append(FixCartesian(a=v, mask=(True, False, True)))
                elif k == "FFT":
                    constraints_lst.append(FixCartesian(a=v, mask=(True, True, False)))
                elif k == "FFF":
                    constraints_lst.append(FixCartesian(a=v, mask=(True, True, True)))
                else:
                    raise ValueError(
                        "Selective Dynamics Error: " + str(k) + ": " + str(v)
                    )
        atoms.set_constraint(constraints_lst)
    return atoms


def _check_if_simple_atoms(atoms):
    """
    Raise a warning if the ASE atoms object includes properties which can not be converted to pymatgen atoms.

    Args:
        atoms: ASE atoms object
    """
    dict_keys = [
        k
        for k in atoms.__dict__.keys()
        if k
        not in ["_celldisp", "arrays", "_cell", "_pbc", "_constraints", "info", "_calc"]
    ]
    array_keys = [
        k for k in atoms.__dict__["arrays"].keys() if k not in ["numbers", "positions"]
    ]
    if not len(dict_keys) == 0:
        warnings.warn("Found unknown keys: " + str(dict_keys))
    if not np.all(atoms.__dict__["_celldisp"] == np.array([[0.0], [0.0], [0.0]])):
        warnings.warn("Found cell displacement: " + str(atoms.__dict__["_celldisp"]))
    if not atoms.__dict__["_calc"] is None:
        warnings.warn("Found calculator: " + str(atoms.__dict__["_calc"]))
    if not atoms.__dict__["_constraints"] == []:
        warnings.warn("Found constraint: " + str(atoms.__dict__["_constraints"]))
    if not np.all(atoms.__dict__["_pbc"]):
        warnings.warn("Cell is not periodic: " + str(atoms.__dict__["_pbc"]))
    if not len(array_keys) == 0:
        warnings.warn("Found unknown flags: " + str(array_keys))
    if not atoms.__dict__["info"] == dict():
        warnings.warn("Info is not empty: " + str(atoms.__dict__["info"]))


def pymatgen_to_pyiron(structure):
    """
        Convert pymatgen Structure object to pyiron atoms object (pymatgen->ASE->pyiron)

    Args:
        pymatgen_obj: pymatgen Structure object

    Returns:
        pyiron atoms object
    """
    # This workaround is necessary because ASE refuses to accept limited degrees of freedom in their atoms object
    # e.g. only accepts [T T T] or [F F F] but rejects [T, T, F] etc.
    # Have to check for the property explicitly otherwise it just straight crashes
    # Let's just implement this workaround if any selective dynamics are present
    if "selective_dynamics" in structure.site_properties.keys():
        sel_dyn_list = structure.site_properties["selective_dynamics"]
        struct = structure.copy()
        struct.remove_site_property("selective_dynamics")
        pyiron_atoms = ase_to_pyiron(pymatgen_to_ase(structure=struct))
        pyiron_atoms.add_tag(selective_dynamics=[True, True, True])
        for i, _ in enumerate(pyiron_atoms):
            pyiron_atoms.selective_dynamics[i] = sel_dyn_list[i]
    else:
        pyiron_atoms = ase_to_pyiron(pymatgen_to_ase(structure=structure))
    return pyiron_atoms


def pyiron_to_pymatgen(pyiron_obj):
    """
    Convert pyiron atoms object to pymatgen Structure object

    Args:
        pyiron_obj: pyiron atoms object

    Returns:
        pymatgen Structure object
    """
    pyiron_obj_conv = pyiron_obj.copy()  # necessary to avoid changing original object
    # This workaround is necessary because ASE refuses to accept limited degrees of freedom in their atoms object
    # e.g. only accepts [T T T] or [F F F] but rejects [T, T, F] etc.
    # Let's just implement this workaround if any selective dynamics are present
    if hasattr(pyiron_obj, "selective_dynamics"):
        sel_dyn_list = pyiron_obj.selective_dynamics
        pyiron_obj_conv.selective_dynamics = [True, True, True]
        ase_obj = pyiron_to_ase(pyiron_obj_conv)
        pymatgen_obj_conv = ase_to_pymatgen(structure=ase_obj)
        new_site_properties = pymatgen_obj_conv.site_properties
        new_site_properties["selective_dynamics"] = sel_dyn_list
        pymatgen_obj = pymatgen_obj_conv.copy(site_properties=new_site_properties)
    else:
        ase_obj = pyiron_to_ase(pyiron_obj_conv)
        _check_if_simple_atoms(atoms=ase_obj)
        pymatgen_obj = ase_to_pymatgen(structure=ase_obj)
    return pymatgen_obj


def ovito_to_pyiron(ovito_obj):
    """

    Args:
        ovito_obj:

    Returns:

    """
    try:
        from ovito.data import ase_to_pyiron

        return ase_to_pyiron(ovito_obj.to_ase_atoms())
    except ImportError:
        raise ValueError("ovito package not yet installed")


def pyiron_to_ovito(atoms):
    """

    Args:
        atoms:

    Returns:

    """
    try:
        from ovito.data import DataCollection

        return DataCollection.create_from_ase_atoms(atoms)
    except ImportError:
        raise ValueError("ovito package not yet installed")


def string2symbols(s):
    """
    Convert string to list of chemical symbols.

    Args:
        s:

    Returns:

    """
    i = None
    n = len(s)
    if n == 0:
        return []
    c = s[0]
    if c.isdigit():
        i = 1
        while i < n and s[i].isdigit():
            i += 1
        return int(s[:i]) * string2symbols(s[i:])
    if c == "(":
        p = 0
        for i, c in enumerate(s):
            if c == "(":
                p += 1
            elif c == ")":
                p -= 1
                if p == 0:
                    break
        j = i + 1
        while j < n and s[j].isdigit():
            j += 1
        if j > i + 1:
            m = int(s[i + 1 : j])
        else:
            m = 1
        return m * string2symbols(s[1:i]) + string2symbols(s[j:])

    if c.isupper():
        i = 1
        if 1 < n and s[1].islower():
            i += 1
        j = i
        while j < n and s[j].isdigit():
            j += 1
        if j > i:
            m = int(s[i:j])
        else:
            m = 1
        return m * [s[:i]] + string2symbols(s[j:])
    else:
        raise ValueError


def symbols2numbers(symbols):
    """

    Args:
        symbols (list, str):

    Returns:

    """
    pse = PeriodicTable()
    df = pse.dataframe.T
    if isinstance(symbols, str):
        symbols = string2symbols(symbols)
    numbers = list()
    for sym in symbols:
        if isinstance(sym, str):
            numbers.append(df[sym]["AtomicNumber"])
        else:
            numbers.append(sym)
    return numbers


def string2vector(v):
    """

    Args:
        v:

    Returns:

    """
    if isinstance(v, str):
        if v[0] == "-":
            return -string2vector(v[1:])
        w = np.zeros(3)
        w["xyz".index(v)] = 1.0
        return w
    return np.array(v, float)


def default(data, dflt):
    """
    Helper function for setting default values.

    Args:
        data:
        dflt:

    Returns:

    """
    if data is None:
        return None
    elif isinstance(data, (list, tuple)):
        newdata = []
        allnone = True
        for x in data:
            if x is None:
                newdata.append(dflt)
            else:
                newdata.append(x)
                allnone = False
        if allnone:
            return None
        return newdata
    else:
        return data


class Symbols(ASESymbols):
    """
    Derived from the ase symbols class which has the following docs:

    Args:
        numbers list/numpy.ndarray): List of atomic numbers
    """

    def __init__(self, numbers):
        self.__doc__ = self.__doc__ + "\n" + super().__doc__
        super().__init__(numbers)
        self._structure = None

    @property
    def structure(self):
        """
        The structure to which the symbol is assigned to

        Returns:
            pyiron_atomistics.atomistics.structure.atoms.Atoms: The required structure
        """
        return self._structure

    @structure.setter
    def structure(self, val):
        self._structure = val

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if self._structure is not None:
            index_array = np.argwhere(
                self.numbers != self._structure.get_atomic_numbers()
            ).flatten()
            replace_elements = self.structure.numbers_to_elements(
                self.numbers[index_array]
            )
            for i, el in enumerate(replace_elements):
                self._structure[index_array[i]] = el


def structure_dict_to_hdf(data_dict, hdf, group_name="structure"):
    with hdf.open(group_name) as hdf_structure:
        for k, v in data_dict.items():
            if k not in ["new_species", "cell", "tags"]:
                hdf_structure[k] = v

        if "new_species" in data_dict.keys():
            for el, el_dict in data_dict["new_species"].items():
                chemical_element_dict_to_hdf(
                    data_dict=el_dict, hdf=hdf_structure, group_name="new_species/" + el
                )

        dict_group_to_hdf(data_dict=data_dict, hdf=hdf_structure, group="tags")
        dict_group_to_hdf(data_dict=data_dict, hdf=hdf_structure, group="cell")


def dict_group_to_hdf(data_dict, hdf, group):
    if group in data_dict.keys():
        with hdf.open(group) as hdf_tags:
            for k, v in data_dict[group].items():
                hdf_tags[k] = v
