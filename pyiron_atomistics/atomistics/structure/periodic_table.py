# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from __future__ import print_function, unicode_literals

import io
import pkgutil
from functools import lru_cache

import numpy as np
import pandas

__author__ = "Joerg Neugebauer, Sudarsan Surendralal, Martin Boeckmann"
__copyright__ = (
    "Copyright 2021, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "1.0"
__maintainer__ = "Sudarsan Surendralal"
__email__ = "surendralal@mpie.de"
__status__ = "production"
__date__ = "Sep 1, 2017"

pandas.options.mode.chained_assignment = None


MENDELEEV_PROPERTY_LIST = [
    "abundance_crust",
    "abundance_sea",
    "atomic_number",
    "atomic_radius",
    "atomic_radius_rahm",
    "atomic_volume",
    "atomic_weight",
    "atomic_weight_uncertainty",
    "block",
    "boiling_point",
    "c6",
    "c6_gb",
    "cas",
    "covalent_radius",
    "covalent_radius_bragg",
    "covalent_radius_cordero",
    "covalent_radius_pyykko",
    "covalent_radius_pyykko_double",
    "covalent_radius_pyykko_triple",
    "cpk_color",
    "density",
    "description",
    "dipole_polarizability",
    "dipole_polarizability_unc",
    "discoverers",
    "discovery_location",
    "discovery_year",
    "ec",
    "econf",
    "electron_affinity",
    "electronegativity",
    "electronegativity_allen",
    "electronegativity_allred_rochow",
    "electronegativity_cottrell_sutton",
    "electronegativity_ghosh",
    "electronegativity_gordy",
    "electronegativity_li_xue",
    "electronegativity_martynov_batsanov",
    "electronegativity_mulliken",
    "electronegativity_nagle",
    "electronegativity_pauling",
    "electronegativity_sanderson",
    "electronegativity_scales",
    "electrons",
    "electrophilicity",
    "en_allen",
    "en_ghosh",
    "en_miedema",
    "en_pauling",
    "evaporation_heat",
    "fusion_heat",
    "gas_basicity",
    "geochemical_class",
    "glawe_number",
    "goldschmidt_class",
    "group",
    "group_id",
    "hardness",
    "heat_of_formation",
    "inchi",
    "init_on_load",
    "ionenergies",
    "ionic_radii",
    "is_monoisotopic",
    "is_radioactive",
    "isotopes",
    "jmol_color",
    "lattice_constant",
    "lattice_structure",
    "mass",
    "mass_number",
    "mass_str",
    "melting_point",
    "mendeleev_number",
    "metadata",
    "metallic_radius",
    "metallic_radius_c12",
    "miedema_electron_density",
    "miedema_molar_volume",
    "molar_heat_capacity",
    "molcas_gv_color",
    "name",
    "name_origin",
    "neutrons",
    "nist_webbook_url",
    "nvalence",
    "oxidation_states",
    "oxides",
    "oxistates",
    "period",
    "pettifor_number",
    "phase_transitions",
    "proton_affinity",
    "protons",
    "registry",
    "scattering_factors",
    "sconst",
    "screening_constants",
    "series",
    "softness",
    "sources",
    "specific_heat",
    "specific_heat_capacity",
    "symbol",
    "thermal_conductivity",
    "uses",
    "vdw_radius",
    "vdw_radius_alvarez",
    "vdw_radius_batsanov",
    "vdw_radius_bondi",
    "vdw_radius_dreiding",
    "vdw_radius_mm3",
    "vdw_radius_rt",
    "vdw_radius_truhlar",
    "vdw_radius_uff",
    "zeff",
]


@lru_cache(maxsize=118)
def element(*args):
    import mendeleev

    return mendeleev.element(*args)


class ChemicalElement(object):
    """
    An Object which contains the element specific parameters
    """

    def __init__(self, sub):
        """
        Constructor: assign PSE dictionary to object
        """
        self._dataset = None
        self.sub = sub
        self._element_str = None
        stringtypes = str
        if isinstance(self.sub, stringtypes):
            self._element_str = self.sub
        elif "Parent" in self.sub.index and isinstance(self.sub.Parent, stringtypes):
            self._element_str = self.sub.Parent
        elif len(self.sub) > 0:
            self._element_str = self.sub.Abbreviation

        self._mendeleev_translation_dict = {
            "AtomicNumber": "atomic_number",
            "AtomicRadius": "covalent_radius_cordero",
            "AtomicMass": "mass",
            "Color": "cpk_color",
            "CovalentRadius": "covalent_radius",
            "CrystalStructure": "lattice_structure",
            "Density": "density",
            "DiscoveryYear": "discovery_year",
            "ElectronAffinity": "electron_affinity",
            "Electronegativity": "electronegativity",
            "Group": "group_id",
            "Name": "name",
            "Period": "period",
            "StandardName": "name",
            "VanDerWaalsRadius": "vdw_radius",
            "MeltingPoint": "melting_point",
        }
        self.el = None

    def __getattr__(self, item):
        if item in ["__array_struct__", "__array_interface__", "__array__"]:
            raise AttributeError
        return self[item]

    def __getitem__(self, item):
        if item in self._mendeleev_translation_dict.keys():
            item = self._mendeleev_translation_dict[item]
        if item in MENDELEEV_PROPERTY_LIST:
            return getattr(element(self._element_str), item)
        if item in self.sub.index:
            return self.sub[item]

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __getstate__(self):
        # Only necessary to support pickling in python <3.11
        # https://docs.python.org/release/3.11.2/library/pickle.html#object.__getstate__
        return self.__dict__

    def __eq__(self, other):
        if self is other:
            return True
        elif isinstance(other, self.__class__):
            conditions = list()
            conditions.append(self.sub.to_dict() == other.sub.to_dict())
            return all(conditions)
        elif isinstance(other, (np.ndarray, list)):
            conditions = list()
            for sp in other:
                conditions.append(self.sub.to_dict() == sp.sub.to_dict())
            return any(conditions)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __gt__(self, other):
        if self != other:
            if self["AtomicNumber"] != other["AtomicNumber"]:
                return self["AtomicNumber"] > other["AtomicNumber"]
            else:
                return self["Abbreviation"] > other["Abbreviation"]
        else:
            return False

    def __ge__(self, other):
        if self != other:
            return self > other
        else:
            return True

    def __hash__(self):
        return hash(repr(self))

    @property
    def tags(self):
        if "tags" not in self.sub.keys() or self.sub["tags"] is None:
            return dict()
        return self.sub["tags"]

    def __dir__(self):
        return list(self.sub.index) + super(ChemicalElement, self).__dir__()

    def __str__(self):
        return str([self._dataset, self.sub])

    def add_tags(self, tag_dic):
        """
        Add tags to an existing element inside its specific panda series without overwriting the old tags

        Args:
            tag_dic (dict): dictionary containing e.g. key = "spin" value = "up",
                            more than one tag can be added at once

        """
        (self.sub["tags"]).update(tag_dic)

    def to_dict(self):
        hdf_el = {}
        # TODO: save all parameters that are different from the parent (e.g. modified mass)
        if self.Parent is not None:
            self._dataset = {"Parameter": ["Parent"], "Value": [self.Parent]}
            hdf_el["elementData"] = self._dataset
        # "Dictionary of element tag static"
        hdf_el.update({"tagData/" + key: self.tags[key] for key in self.tags.keys()})
        return hdf_el

    def from_dict(self, obj_dict):
        pse = PeriodicTable()
        elname = self.sub.name
        if "elementData" in obj_dict.keys():
            element_data = obj_dict["elementData"]
            for key, val in zip(element_data["Parameter"], element_data["Value"]):
                if key in "Parent":
                    self.sub = pse.dataframe.loc[val]
                    self.sub["Parent"] = val
                    self._element_str = val
                else:
                    self.sub["Parent"] = None
                    self._element_str = elname
                self.sub.name = elname
        if "tagData" in obj_dict.keys():
            self.sub["tags"] = obj_dict["tagData"]

    def to_hdf(self, hdf):
        """
        saves the element with his parameters into his hdf5 job file
        Args:
            hdf (Hdfio): Hdfio object which will be used
        """
        chemical_element_dict_to_hdf(
            data_dict=self.to_dict(), hdf=hdf, group_name=self.Abbreviation
        )

    def from_hdf(self, hdf):
        """
        loads an element with his parameters from the hdf5 job file and store it into its specific pandas series
        Args:
            hdf (Hdfio): Hdfio object which will be used to read a hdf5 file
        """
        elname = self.sub.name
        with hdf.open(elname) as hdf_el:
            self.from_dict(obj_dict=hdf_el.read_dict_from_hdf(recursive=True))


class PeriodicTable:
    """
    An Object which stores an elementary table which can be modified for the current session
    """

    def __init__(self, file_name=None):  # PSE_dat_file = None):
        """

        Args:
            file_name (str): Possibility to choose an source hdf5 file
        """
        self.dataframe = self._get_periodic_table_df(file_name)
        if "Abbreviation" not in self.dataframe.columns.values:
            self.dataframe["Abbreviation"] = None
        if not all(self.dataframe["Abbreviation"].values):
            for item in self.dataframe.index.values:
                if self.dataframe["Abbreviation"][item] is None:
                    self.dataframe["Abbreviation"][item] = item
        self._parent_element = None
        self.el = None

    def __getattr__(self, item):
        return self[item]

    def __getitem__(self, item):
        if item in self.dataframe.columns.values:
            return self.dataframe[item]
        if item in self.dataframe.index.values:
            return self.dataframe.loc[item]

    def __setstate__(self, state):
        """
        Used by (cloud)pickle; force the state update to avoid recursion pickling Atoms
        """
        self.__dict__.update(state)

    def __getstate__(self):
        # Only necessary to support pickling in python <3.11
        # https://docs.python.org/release/3.11.2/library/pickle.html#object.__getstate__
        return self.__dict__

    def from_dict(self, obj_dict):
        for el, el_dict in obj_dict.items():
            sub = pandas.Series(dtype=object)
            new_element = ChemicalElement(sub)
            new_element.sub.name = el
            new_element.from_dict(obj_dict=el_dict)
            new_element.sub["Abbreviation"] = el

            if "sub_tags" in new_element.tags:
                if not new_element.tags["sub_tags"]:
                    del new_element.tags["sub_tags"]

            if new_element.Parent is None:
                if not (el in self.dataframe.index.values):
                    raise AssertionError()
                if len(new_element.sub["tags"]) > 0:
                    raise ValueError("Element cannot get tag-assignment twice")
                if "tags" not in self.dataframe.keys():
                    self.dataframe["tags"] = None
                self.dataframe["tags"][el] = new_element.tags
            else:
                self.dataframe = pandas.concat(
                    [self.dataframe, new_element.sub.to_frame().T]
                )
                self.dataframe["tags"] = self.dataframe["tags"].apply(
                    lambda x: None if pandas.isnull(x) else x
                )
                self.dataframe["Parent"] = self.dataframe["Parent"].apply(
                    lambda x: None if pandas.isnull(x) else x
                )

    def from_hdf(self, hdf):
        """
        loads an element with his parameters from the hdf5 job file by creating an Object of the ChemicalElement type.
        The new element will be stored in the current periodic table.
        Changes in the tags will also be modified inside the periodic table.

        Args:
            hdf (Hdfio): Hdfio object which will be used to read the data from a hdf5 file

        Returns:

        """
        self.from_dict(obj_dict=hdf.read_dict_from_hdf(recursive=True))

    def element(self, arg, **qwargs):
        """
        The method searches through the periodic table. If the table contains the element,
        it will return an Object of the type ChemicalElement containing all parameters from the periodic table.
        The option **qwargs allows a direct modification of the tag-values during the creation process
        Args:
            arg (str, ChemicalElement): sort of element
            **qwargs: e.g. a dictionary of tags

        Returns element (ChemicalElement): a element with all its properties (Abbreviation, AtomicMass, Weight, ...)

        """
        stringtypes = str
        if isinstance(arg, stringtypes):
            if arg in self.dataframe.index.values:
                self.el = arg
            else:
                raise KeyError(arg)
        elif isinstance(arg, int):
            if arg in list(self.dataframe["AtomicNumber"]):
                index = list(self.dataframe["AtomicNumber"]).index(arg)
                self.el = self.dataframe.iloc[index].name
        else:
            raise ValueError("type not defined: " + str(type(arg)))
        if len(qwargs.values()) > 0:
            if "tags" not in self.dataframe.columns.values:
                self.dataframe["tags"] = None
            self.dataframe["tags"][self.el] = qwargs
        element = self.dataframe.loc[self.el]
        # element['CovalentRadius'] /= 100
        return ChemicalElement(element)

    def is_element(self, symbol):
        """
        Compares the Symbol with the Abbreviations of elements inside the periodic table
        Args:
            symbol (str): name of element, str

        Returns boolean: true for the same element, false otherwise

        """
        return symbol in self.dataframe["Abbreviation"]

    def atomic_number_to_abbreviation(self, atom_no):
        """

        Args:
            atom_no:

        Returns:

        """
        if not isinstance(atom_no, int):
            raise ValueError("type not defined: " + str(type(atom_no)))

        return self.Abbreviation[
            np.nonzero(self.AtomicNumber.to_numpy() == atom_no)[0][0]
        ]

    def add_element(
        self, parent_element, new_element, use_parent_potential=False, **qwargs
    ):
        """
        Add "additional" chemical elements to the Periodic Table. These can be used to distinguish between the various
        potentials which may exist for a given species or to introduce artificial elements such as pseudohydrogen. For
        this case set use_parent_potential = False and add in the directory containing the potential files a new file
        which is derived from the name new element.

        This function may be also used to provide additional information for the identical chemical element, e.g., to
        define a Fe_up and Fe_down to perform the correct symmetry search as well as initialization.

        Args:
            parent_element (str): name of parent element
            new_element (str): name of new element
            use_parent_potential: True: use the potential from the parent species
            **qwargs: define tags and their values, e.g. spin = "up", relax = [True, True, True]

        Returns: new element (ChemicalElement)

        """

        pandas.options.mode.chained_assignment = None
        parent_element_data_series = self.dataframe.loc[parent_element]
        parent_element_data_series["Abbreviation"] = new_element
        parent_element_data_series["Parent"] = parent_element
        parent_element_data_series.name = new_element
        if new_element not in self.dataframe.T.columns:
            self.dataframe = pandas.concat(
                [self.dataframe, parent_element_data_series.to_frame().T],
            )
        else:
            self.dataframe.loc[new_element] = parent_element_data_series
        if use_parent_potential:
            self._parent_element = parent_element
        return self.element(new_element, **qwargs)

    @staticmethod
    @lru_cache(maxsize=1)
    def _get_periodic_table_df(file_name):
        """

        Args:
            file_name:

        Returns:

        """
        if not file_name:
            return pandas.read_csv(
                io.BytesIO(
                    pkgutil.get_data("pyiron_atomistics", "data/periodic_table.csv")
                ),
                index_col=0,
            )
        else:
            if file_name.endswith(".h5"):
                return pandas.read_hdf(file_name, mode="r")
            elif file_name.endswith(".csv"):
                return pandas.read_csv(file_name, index_col=0)
            raise TypeError(
                "PeriodicTable file format not recognised: "
                + file_name
                + " supported file formats are csv, h5."
            )


def chemical_element_dict_to_hdf(data_dict, hdf, group_name):
    with hdf.open(group_name) as hdf_el:
        if "elementData" in data_dict.keys():
            hdf_el["elementData"] = data_dict["elementData"]
        with hdf_el.open("tagData") as hdf_tag:
            if "tagData" in data_dict.keys():
                for k, v in data_dict["tagData"].items():
                    hdf_tag[k] = v
