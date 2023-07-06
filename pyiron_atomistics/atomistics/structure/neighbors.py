# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import numpy as np
from pyiron_base import DataContainer, FlattenedStorage
from pyiron_base import deprecate

__author__ = "Joerg Neugebauer, Sam Waseda"
__copyright__ = (
    "Copyright 2021, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "1.0"
__maintainer__ = "Sam Waseda"
__email__ = "waseda@mpie.de"
__status__ = "production"
__date__ = "Sep 1, 2017"


class NeighborsTrajectory(DataContainer):
    """
    This class generates the neighbors for a given atomistic trajectory. The resulting indices, distances, and vectors
    are stored as numpy arrays.
    """

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls, *args, **kwargs)
        object.__setattr__(instance, "_has_structure", None)
        return instance

    def __init__(
        self,
        init=None,
        has_structure=None,
        num_neighbors=12,
        table_name="neighbors_traj",
        store=None,
        **kwargs,
    ):
        """

        Args:
            has_structure (:class:`.HasStructure`): object containing the structures to compute the neighbors on
            num_neighbors (int): The cutoff for the number of neighbors
            table_name (str): Table name for the base `DataContainer` (stores this object as a group in a HDF5 file with
                              this name)
            store (FlattenedStorage): internal storage that should be used to store the neighborhood information,
                                      creates a new one if not provided; if provided and not empty it must be compatible
                                      with the lengths of the structures in `has_structure`, but this is *not* checked
            **kwargs (dict): Additional arguments to be passed to the `get_neighbors()` routine
                             (eg. cutoff_radius, norm_order , etc.)
        """
        super().__init__(init=init, table_name=table_name)
        self._flat_store = store if store is not None else FlattenedStorage()
        self._flat_store.add_array(
            "indices", dtype=np.int64, shape=(num_neighbors,), per="element", fill=-1
        )
        self._flat_store.add_array(
            "distances", dtype=np.float64, shape=(num_neighbors,), per="element"
        )
        self._flat_store.add_array(
            "vecs", dtype=np.float64, shape=(num_neighbors, 3), per="element"
        )
        self._flat_store.add_array(
            "shells", dtype=np.int64, shape=(num_neighbors,), per="element"
        )
        self._num_neighbors = num_neighbors
        self._get_neighbors_kwargs = kwargs
        self.has_structure = has_structure

    @property
    def has_structure(self):
        return self._has_structure

    @has_structure.setter
    def has_structure(self, value):
        if value is not None:
            self._has_structure = value
            self._compute_neighbors()

    @property
    def indices(self):
        """
        Neighbour indices (excluding itself) of each atom computed using the get_neighbors_traj() method

        If the structures have different number of atoms, the array will have -1 on indices that are invalid.

        Returns:
            numpy.ndarray: An int array of dimension N_steps / stride x N_atoms x N_neighbors
        """
        return self._flat_store.get_array_filled("indices")

    @property
    def distances(self):
        """
        Neighbour distances (excluding itself) of each atom computed using the get_neighbors_traj() method

        If the structures have different number of atoms, the array will have NaN on indices that are invalid.

        Returns:
            numpy.ndarray: A float array of dimension N_steps / stride x N_atoms x N_neighbors
        """
        return self._flat_store.get_array_filled("distances")

    @property
    def vecs(self):
        """
        Neighbour vectors (excluding itself) of each atom computed using the get_neighbors_traj() method

        If the structures have different number of atoms, the array will have NaN on indices that are invalid.

        Returns:
            numpy.ndarray: A float array of dimension N_steps / stride x N_atoms x N_neighbors x 3
        """
        return self._flat_store.get_array_filled("vecs")

    @property
    def shells(self):
        """
        Neighbor shell indices (excluding itself) of each atom computed using the get_neighbors_traj() method.

        For trajectories with non constant amount of particles this array may contain -1 for invalid values, i.e.

        Returns:
            ndarray: An int array of dimension N_steps / stride x N_atoms x N_neighbors x 3
        """
        return self._flat_store.get_array_filled("shells")

    @property
    def num_neighbors(self):
        """
        The maximum number of neighbors to be computed

        Returns:

            int: The max number of neighbors
        """
        return self._num_neighbors

    def _compute_neighbors(self):
        for i, struct in enumerate(self._has_structure.iter_structures()):
            if (
                i < len(self._flat_store)
                and (self._flat_store["indices", i] != -1).all()
            ):
                # store already has valid entries for this structure, so skip it
                continue
            # Change the `allow_ragged` based on the changes in get_neighbors()
            neigh = struct.get_neighbors(
                num_neighbors=self._num_neighbors,
                allow_ragged=False,
                **self._get_neighbors_kwargs,
            )
            if i >= len(self._flat_store):
                self._flat_store.add_chunk(
                    len(struct),
                    indices=neigh.indices,
                    distances=neigh.distances,
                    vecs=neigh.vecs,
                    shells=neigh.shells,
                )
            else:
                self._flat_store.set_array("indices", i, neigh.indices)
                self._flat_store.set_array("distances", i, neigh.distances)
                self._flat_store.set_array("vecs", i, neigh.vecs)
                self._flat_store.set_array("shells", i, neigh.shells)
        return (
            self._flat_store.get_array_filled("indices"),
            self._flat_store.get_array_filled("distances"),
            self._flat_store.get_array_filled("vecs"),
        )

    @deprecate(
        "This has no effect, neighbors are automatically called on instantiation."
    )
    def compute_neighbors(self):
        """
        Compute the neighbors across the trajectory
        """
        pass
