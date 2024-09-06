# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

"""
An abstract Potential class to provide an easy access for the available potentials. Currently implemented for the
OpenKim https://openkim.org database.
"""

import os
from abc import ABC, abstractmethod

import pandas
from pyiron_base import state
from pyiron_snippets.resources import ResourceNotFound, ResourceResolver

__author__ = "Martin Boeckmann, Jan Janssen"
__copyright__ = (
    "Copyright 2021, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "1.0"
__maintainer__ = "Jan Janssen"
__email__ = "janssen@mpie.de"
__status__ = "development"
__date__ = "Sep 1, 2017"


class PotentialAbstract(ABC):
    """
    The PotentialAbstract class loads a list of available potentials and sorts them. Afterwards the potentials can be
    accessed through:
        PotentialAbstract.<Element>.<Element> or PotentialAbstract.find_potentials_set({<Element>, <Element>}

    Args:
        potential_df:
        default_df:
        selected_atoms:
    """

    @property
    @abstractmethod
    def resource_plugin_name(self) -> str:
        """Return the name of the folder of this plugin/code in the pyiron resources.

        One of lammps/vasp/sphinx, to be overriden in the specific sub classes."""
        pass

    def __init__(self, potential_df, default_df=None, selected_atoms=None):
        self._potential_df = potential_df
        self._default_df = default_df
        if selected_atoms is not None:
            self._selected_atoms = selected_atoms
        else:
            self._selected_atoms = []

    def find(self, element):
        """
        Find the potentials

        Args:
            element (set, str): element or set of elements for which you want the possible LAMMPS potentials

        Returns:
            list: of possible potentials for the element or the combination of elements

        """
        if isinstance(element, set):
            element = element
        elif isinstance(element, list):
            element = set(element)
        elif isinstance(element, str):
            element = set([element])
        else:
            raise TypeError("Only, str, list and set supported!")
        return self._potential_df[
            [
                True if set(element).issubset(species) else False
                for species in self._potential_df["Species"].values
            ]
        ]

    def find_by_name(self, potential_name):
        mask = self._potential_df["Name"] == potential_name
        if not mask.any():
            raise ValueError(
                "Potential '{}' not found in database.".format(potential_name)
            )
        return self._potential_df[mask]

    def list(self):
        """
        List the available potentials

        Returns:
            list: of possible potentials for the element or the combination of elements
        """
        return self._potential_df

    def __getattr__(self, item):
        return self[item]

    def __getitem__(self, item):
        potential_df = self.find(element=item)
        selected_atoms = self._selected_atoms + [item]
        return PotentialAbstract(
            potential_df=potential_df,
            default_df=self._default_df,
            selected_atoms=selected_atoms,
        )

    def __str__(self):
        return str(self.list())

    @classmethod
    def _get_resolver(cls):
        """Return a ResourceResolver that can be searched for potential files or potential dataframes.

        This exists primarily so that the lammps and sphinx sub classes can overload it to add their conda package
        specific resource paths.

        Returns:
            :class:`.ResourceResolver`
        """
        return ResourceResolver(
            state.settings.resource_paths,
            cls.resource_plugin_name,
            "potentials",
        )

    @classmethod
    def _get_potential_df(cls, file_name_lst):
        """

        Args:
            file_name_lst (set):

        Returns:
            pandas.DataFrame:
        """
        env = os.environ

        def read_csv(path):
            return pandas.read_csv(
                path,
                index_col=0,
                converters={
                    "Species": lambda x: x.replace("'", "").strip("[]").split(", "),
                    "Config": lambda x: x.replace("'", "")
                    .replace("\\n", "\n")
                    .strip("[]")
                    .split(", "),
                    "Filename": lambda x: x.replace("'", "").strip("[]").split(", "),
                },
            )

        files = cls._get_resolver().list(file_name_lst)
        if len(files) > 0:
            return pandas.concat(map(read_csv, files), ignore_index=True)
        else:
            raise ValueError(
                f"Was not able to locate the potential files in {cls._get_resolver()}!"
            )

    @classmethod
    def _get_potential_default_df(
        cls,
        file_name_lst={"potentials_vasp_pbe_default.csv"},
    ):
        """

        Args:
            file_name_lst (set):

        Returns:
            pandas.DataFrame:
        """
        try:
            return pandas.read_csv(
                cls._get_resolver().first(file_name_lst), index_col=0
            )
        except ResourceNotFound:
            raise ValueError("Was not able to locate the potential files.") from None

    @classmethod
    def find_potential_file(cls, path):
        res = cls._get_resolver()
        try:
            return res.first(path)
        except ResourceNotFound:
            raise ValueError(f"Could not find file '{path}' in {res}!") from None
