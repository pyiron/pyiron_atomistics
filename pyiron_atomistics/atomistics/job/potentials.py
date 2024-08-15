# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

"""
An abstract Potential class to provide an easy access for the available potentials. Currently implemented for the
OpenKim https://openkim.org database.
"""

import os

import pandas
from pyiron_base import state
from pyiron_snippets.resources import ResourceResolver, ResourceNotFound

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


class PotentialAbstract(object):
    """
    The PotentialAbstract class loads a list of available potentials and sorts them. Afterwards the potentials can be
    accessed through:
        PotentialAbstract.<Element>.<Element> or PotentialAbstract.find_potentials_set({<Element>, <Element>}

    Args:
        potential_df:
        default_df:
        selected_atoms:
    """

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
    def _get_resolver(cls, plugin_name):
        """Return a ResourceResolver that can be searched for potential files or potential dataframes.

        This exists primarily so that the lammps and sphinx sub classes can overload it to add their conda package
        specific resource paths.

        Args:
            plugin_name (str): one of "lammps", "vasp", "sphinx"; i.e. the name of the resource folder to search
        Returns:
            :class:`.ResourceResolver`
        """
        return ResourceResolver(
                state.settings.resource_paths,
                plugin_name, "potentials",
        )

    @classmethod
    def _get_potential_df(cls, plugin_name, file_name_lst):
        """

        Args:
            plugin_name (str):
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
                        "Species": lambda x: x.replace("'", "")
                        .strip("[]")
                        .split(", "),
                        "Config": lambda x: x.replace("'", "")
                        .replace("\\n", "\n")
                        .strip("[]")
                        .split(", "),
                        "Filename": lambda x: x.replace("'", "")
                        .strip("[]")
                        .split(", "),
                    },
            )
        files = cls._get_resolver(plugin_name).chain(
            # support iprpy-data package; data paths in the iprpy are of a different form than in
            # pyiron resources, so we cannot add it as an additional path to the resolver above.
            # Instead make a new resolver and chain it after the first one.
            # TODO: this is a fix specific for lammps potentials; it could be moved to the lammps
            # subclass
            ResourceResolver(
                [env[var] for var in ("CONDA_PREFIX", "CONDA_DIR") if var in env],
                "share", "iprpy",
            ),
        ).search(file_name_lst)
        return pandas.concat(map(read_csv, files), ignore_index=True)

    @staticmethod
    def _get_potential_default_df(
        plugin_name,
        file_name_lst={"potentials_vasp_pbe_default.csv"},
    ):
        """

        Args:
            plugin_name (str):
            file_name_lst (set):

        Returns:
            pandas.DataFrame:
        """
        try:
            file = ResourceResolver(
                    state.settings.resource_paths,
                    plugin_name, "potentials",
            ).first(file_name_lst)
            return pandas.read_csv(file, index_col=0)
        except ResourceNotFound:
            raise ValueError("Was not able to locate the potential files.") from None

def find_potential_file_base(path, resource_path_lst, rel_path):
    try:
        return ResourceResolver(
                resource_path_lst,
                rel_path,
        ).first(path)
    except ResourceNotFound:
        raise ValueError(
            "Either the filename or the functional has to be defined.",
            path,
            resource_path_lst,
        ) from None
