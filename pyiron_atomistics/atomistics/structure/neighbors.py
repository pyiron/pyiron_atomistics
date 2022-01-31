# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.sparse import coo_matrix
from scipy.special import gamma
from pyiron_base import DataContainer, FlattenedStorage
from pyiron_atomistics.atomistics.structure.analyse import get_average_of_unique_labels
from scipy.spatial.transform import Rotation
from scipy.special import sph_harm
import warnings
from pyiron_base import deprecate
import itertools

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


class Tree:
    """
    Class to get tree structure for the neighborhood information.

    Main attributes (do not modify them):

    - distances (numpy.ndarray/list): Distances to the neighbors of given positions
    - indices (numpy.ndarray/list): Indices of the neighbors of given positions
    - vecs (numpy.ndarray/list): Vectors to the neighbors of given positions

    Auxiliary attributes:

    - wrap_positions (bool): Whether to wrap back the positions entered by user in get_neighborhood
        etc. Since the information outside the original box is limited to a few layer,
        wrap_positions=False might miss some points without issuing an error.

    Change representation mode via :attribute:`.Neighbors.mode`  (cf. its DocString)

    Furthermore, you can re-employ the original tree structure to get neighborhood information via
    get_neighborhood.

    """

    def __init__(self, ref_structure):
        """
        Args:
            ref_structure (pyiron_atomistics.atomistics.structure.atoms.Atoms): Reference
                structure.
        """
        self._distances = None
        self._vectors = None
        self._indices = None
        self._mode = {"filled": True, "ragged": False, "flattened": False}
        self._extended_positions = None
        self._positions = None
        self._wrapped_indices = None
        self._extended_indices = None
        self._ref_structure = ref_structure.copy()
        self.wrap_positions = False
        self._tree = None
        self.num_neighbors = None
        self.cutoff_radius = np.inf
        self._norm_order = 2

    @property
    def mode(self):
        """
        Change the mode of representing attributes (vecs, distances, indices, shells). The shapes
        of `filled` and `ragged` differ only if `cutoff_radius` is specified.

        - 'filled': Fill variables for the missing entries are filled as follows: `np.inf` in
            `distances`, `numpy.array([np.inf, np.inf, np.inf])` in `vecs`, `n_atoms+1` (or a
            larger value) in `indices` and -1 in `shells`.

        - 'ragged': Create lists of different lengths.

        - 'flattened': Return flattened arrays for distances, vecs and shells. The indices
            corresponding to the row numbers in 'filled' and 'ragged' are in `atom_numbers`

        The variables are stored in the `filled` mode.
        """
        for k, v in self._mode.items():
            if v:
                return k

    def _set_mode(self, new_mode):
        if new_mode not in self._mode.keys():
            raise KeyError(
                f"{new_mode} not found. Available modes: {', '.join(self._mode.keys())}"
            )
        self._mode = {key: False for key in self._mode.keys()}
        self._mode[new_mode] = True

    def __repr__(self):
        """
        Returns: __repr__
        """
        to_return = (
            "Main attributes:\n"
            + "- distances : Distances to the neighbors of given positions\n"
            + "- indices : Indices of the neighbors of given positions\n"
            + "- vecs : Vectors to the neighbors of given positions\n"
        )
        return to_return

    def copy(self):
        new_neigh = Tree(self._ref_structure)
        new_neigh._distances = self._distances.copy()
        new_neigh._indices = self._indices.copy()
        new_neigh._extended_positions = self._extended_positions
        new_neigh._wrapped_indices = self._wrapped_indices
        new_neigh._extended_indices = self._extended_indices
        new_neigh.wrap_positions = self.wrap_positions
        new_neigh._tree = self._tree
        new_neigh.num_neighbors = self.num_neighbors
        new_neigh.cutoff_radius = self.cutoff_radius
        new_neigh._norm_order = self._norm_order
        new_neigh._positions = self._positions.copy()
        return new_neigh

    def _reshape(self, value, key=None, ref_vector=None):
        if value is None:
            raise ValueError("Neighbors not initialized yet")
        if key is None:
            key = self.mode
        if key == "filled":
            return value
        elif key == "ragged":
            return self._contract(value, ref_vector=ref_vector)
        elif key == "flattened":
            return value[self._distances < np.inf]

    @property
    def distances(self):
        """Distances to neighboring atoms."""
        return self._reshape(self._distances)

    @property
    def _vecs(self):
        if self._vectors is None:
            self._vectors = self._get_vectors(
                positions=self._positions,
                distances=self.filled.distances,
                indices=self._extended_indices,
            )
        return self._vectors

    @property
    def vecs(self):
        """Vectors to neighboring atoms."""
        return self._reshape(self._vecs)

    @property
    def indices(self):
        """Indices of neighboring atoms."""
        return self._reshape(self._indices)

    @property
    def atom_numbers(self):
        """Indices of atoms."""
        n = np.zeros_like(self.filled.indices)
        n.T[:, :] = np.arange(len(n))
        return self._reshape(n)

    @property
    def norm_order(self):
        """
        Norm to use for the neighborhood search and shell recognition. The definition follows the
        conventional Lp norm (cf. https://en.wikipedia.org/wiki/Lp_space). This is still an
        experimental feature and for anything other than norm_order=2, there is no guarantee that
        this works flawlessly.
        """
        return self._norm_order

    @norm_order.setter
    def norm_order(self, value):
        raise ValueError(
            "norm_order cannot be changed after initialization. Re-initialize the Neighbor class"
            + " with the correct norm_order value"
        )

    def _get_max_length(self, ref_vector=None):
        if ref_vector is None:
            ref_vector = self.filled.distances
        if (
            ref_vector is None
            or len(ref_vector) == 0
            or not hasattr(ref_vector[0], "__len__")
        ):
            return None
        return max(len(dd[dd < np.inf]) for dd in ref_vector)

    def _contract(self, value, ref_vector=None):
        if self._get_max_length(ref_vector=ref_vector) is None:
            return value
        return [
            vv[: np.sum(dist < np.inf)]
            for vv, dist in zip(value, self.filled.distances)
        ]

    @property
    @deprecate("Use mode", version="1.0.0")
    def allow_ragged(self):
        """
        Whether to allow ragged list of distancs/vectors/indices or fill empty slots with numpy.inf
        to get rectangular arrays
        """
        return self._mode["ragged"]

    @allow_ragged.setter
    @deprecate("Use `mode='ragged'`", version="1.0.0")
    def allow_ragged(self, new_bool):
        if not isinstance(new_bool, bool):
            raise ValueError("allow_ragged must be a boolean")
        self._set_mode(self._allow_ragged_to_mode(new_bool))

    def _allow_ragged_to_mode(self, new_bool):
        if new_bool is None:
            return self.mode
        elif new_bool:
            return "ragged"
        return "filled"

    def _get_extended_positions(self):
        if self._extended_positions is None:
            return self._ref_structure.positions
        return self._extended_positions

    def _get_wrapped_indices(self):
        if self._wrapped_indices is None:
            return np.arange(len(self._ref_structure.positions))
        return self._wrapped_indices

    def _get_wrapped_positions(self, positions, distance_buffer=1.0e-12):
        if not self.wrap_positions:
            return np.asarray(positions)
        x = np.array(positions).copy()
        cell = self._ref_structure.cell
        x_scale = np.dot(x, np.linalg.inv(cell)) + distance_buffer
        x[..., self._ref_structure.pbc] -= np.dot(np.floor(x_scale), cell)[
            ..., self._ref_structure.pbc
        ]
        return x

    def _get_distances_and_indices(
        self,
        positions,
        num_neighbors=None,
        cutoff_radius=np.inf,
        width_buffer=1.2,
    ):
        num_neighbors = self._estimate_num_neighbors(
            num_neighbors=num_neighbors,
            cutoff_radius=cutoff_radius,
            width_buffer=width_buffer,
        )
        if (
            len(self._get_extended_positions()) < num_neighbors
            and cutoff_radius == np.inf
        ):
            raise ValueError(
                "num_neighbors too large - make width_buffer larger and/or make "
                + "num_neighbors smaller"
            )
        positions = self._get_wrapped_positions(positions)
        distances, indices = self._tree.query(
            positions,
            k=num_neighbors,
            distance_upper_bound=cutoff_radius,
            p=self.norm_order,
        )
        shape = positions.shape[:-1] + (num_neighbors,)
        distances = np.array([distances]).reshape(shape)
        indices = np.array([indices]).reshape(shape)
        if cutoff_radius < np.inf and np.any(distances.T[-1] < np.inf):
            warnings.warn(
                "Number of neighbors found within the cutoff_radius is equal to (estimated) "
                + "num_neighbors. Increase num_neighbors (or set it to None) or "
                + "width_buffer to find all neighbors within cutoff_radius."
            )
        self._extended_indices = indices.copy()
        indices[distances < np.inf] = self._get_wrapped_indices()[
            indices[distances < np.inf]
        ]
        indices[distances == np.inf] = np.iinfo(np.int32).max
        return (
            self._reshape(distances, ref_vector=distances),
            self._reshape(indices, ref_vector=distances),
        )

    @property
    def numbers_of_neighbors(self):
        """
        Get number of neighbors for each atom. Same number is returned if `cutoff_radius` was not
        given in the initialization.
        """
        return np.sum(self.filled.distances < np.inf, axis=-1)

    def _get_vectors(
        self,
        positions,
        num_neighbors=None,
        cutoff_radius=np.inf,
        distances=None,
        indices=None,
        width_buffer=1.2,
    ):
        if distances is None or indices is None:
            distances, indices = self._get_distances_and_indices(
                positions=positions,
                num_neighbors=num_neighbors,
                cutoff_radius=cutoff_radius,
                width_buffer=width_buffer,
            )
        vectors = np.zeros(distances.shape + (3,))
        vectors -= self._get_wrapped_positions(positions).reshape(
            distances.shape[:-1] + (-1, 3)
        )
        vectors[distances < np.inf] += self._get_extended_positions()[
            self._extended_indices[distances < np.inf]
        ]
        vectors[distances == np.inf] = np.array(3 * [np.inf])
        return vectors

    def _estimate_num_neighbors(
        self, num_neighbors=None, cutoff_radius=np.inf, width_buffer=1.2
    ):
        """

        Args:
            num_neighbors (int): number of neighbors
            width_buffer (float): width of the layer to be added to account for pbc.
            cutoff_radius (float): self-explanatory

        Returns:
            Number of atoms required for a given cutoff

        """
        if (
            num_neighbors is None
            and cutoff_radius == np.inf
            and self.num_neighbors is None
        ):
            raise ValueError("Specify num_neighbors or cutoff_radius")
        elif num_neighbors is None and self.num_neighbors is None:
            volume = self._ref_structure.get_volume(per_atom=True)
            width_buffer = 1 + width_buffer
            width_buffer *= get_volume_of_n_sphere_in_p_norm(3, self.norm_order)
            num_neighbors = max(14, int(width_buffer * cutoff_radius**3 / volume))
        elif num_neighbors is None:
            num_neighbors = self.num_neighbors
        if self.num_neighbors is None:
            self.num_neighbors = num_neighbors
            self.cutoff_radius = cutoff_radius
        if num_neighbors > self.num_neighbors:
            warnings.warn(
                "Taking a larger search area after initialization has the risk of "
                + "missing neighborhood atoms"
            )
        return num_neighbors

    def _estimate_width(
        self, num_neighbors=None, cutoff_radius=np.inf, width_buffer=1.2
    ):
        """

        Args:
            num_neighbors (int): number of neighbors
            width_buffer (float): width of the layer to be added to account for pbc.
            cutoff_radius (float): cutoff radius

        Returns:
            Width of layer required for the given number of atoms

        """
        if num_neighbors is None and cutoff_radius == np.inf:
            raise ValueError("Define either num_neighbors or cutoff_radius")
        if all(self._ref_structure.pbc == False):
            return 0
        elif cutoff_radius != np.inf:
            return cutoff_radius
        prefactor = get_volume_of_n_sphere_in_p_norm(3, self.norm_order)
        width = np.prod(
            np.linalg.norm(self._ref_structure.cell, axis=-1, ord=self.norm_order)
        )
        width *= prefactor * np.max([num_neighbors, 8]) / len(self._ref_structure)
        cutoff_radius = width_buffer * width ** (1 / 3)
        return cutoff_radius

    def get_neighborhood(
        self,
        positions,
        num_neighbors=None,
        cutoff_radius=np.inf,
        width_buffer=1.2,
    ):
        """
        Get neighborhood information of `positions`. What it returns is in principle the same as
        `get_neighborhood` in `Atoms`. The only one difference is the reuse of the same Tree
        structure, which makes the algorithm more efficient, but will fail if the reference
        structure changed in the meantime.

        Args:
            position: Position in a box whose neighborhood information is analysed
            num_neighbors (int): Number of nearest neighbors
            cutoff_radius (float): Upper bound of the distance to which the search is to be done
            width_buffer (float): Width of the layer to be added to account for pbc.

        Returns:

            pyiron.atomistics.structure.atoms.Tree: Neighbors instances with the neighbor indices,
                distances and vectors

        """
        new_neigh = self.copy()
        return new_neigh._get_neighborhood(
            positions=positions,
            num_neighbors=num_neighbors,
            cutoff_radius=cutoff_radius,
            exclude_self=False,
            width_buffer=width_buffer,
        )

    def _get_neighborhood(
        self,
        positions,
        num_neighbors=12,
        cutoff_radius=np.inf,
        exclude_self=False,
        width_buffer=1.2,
    ):
        start_column = 0
        if exclude_self:
            start_column = 1
            if num_neighbors is not None:
                num_neighbors += 1
        distances, indices = self._get_distances_and_indices(
            positions,
            num_neighbors=num_neighbors,
            cutoff_radius=cutoff_radius,
            width_buffer=width_buffer,
        )
        if num_neighbors is not None:
            self.num_neighbors -= 1
        max_column = np.sum(distances < np.inf, axis=-1).max()
        self._distances = distances[..., start_column:max_column]
        self._indices = indices[..., start_column:max_column]
        self._extended_indices = self._extended_indices[..., start_column:max_column]
        self._positions = positions
        return self

    def _check_width(self, width, pbc):
        if any(pbc) and np.prod(self.filled.distances.shape) > 0:
            if (
                np.linalg.norm(
                    self.flattened.vecs[..., pbc], axis=-1, ord=self.norm_order
                ).max()
                > width
            ):
                return True
        return False

    def get_spherical_harmonics(self, l, m, cutoff_radius=np.inf, rotation=None):
        """
        Args:
            l (int/numpy.array): Degree of the harmonic (int); must have ``l >= 0``.
            m (int/numpy.array): Order of the harmonic (int); must have ``|m| <= l``.
            cutoff_radius (float): maximum neighbor distance to include (default: inf, i.e. all
            atoms included in the neighbor search).
            rotation ( (3,3) numpy.array/list): Rotation to make sure phi does not become nan

        Returns:
            ( (natoms,) numpy.array) spherical harmonic values

        Spherical harmonics defined as follows

        Y^m_l(\theta,\phi) = \sqrt{\frac{2l+1}{4\pi} \frac{(l-m)!}{(l+m)!}}
        e^{i m \theta} P^m_l(\cos(\phi))

        The angles are calculated based on `self.vecs`, where the azimuthal angle is defined on the
        xy-plane and the polar angle is along the z-axis.

        See more on: scipy.special.sph_harm

        """
        vecs = self.filled.vecs
        if rotation is not None:
            vecs = np.einsum("ij,nmj->nmi", rotation, vecs)
        within_cutoff = self.filled.distances < cutoff_radius
        if np.any(np.all(~within_cutoff, axis=-1)):
            raise ValueError("cutoff_radius too small - some atoms have no neighbors")
        phi = np.zeros_like(self.filled.distances)
        theta = np.zeros_like(self.filled.distances)
        theta[within_cutoff] = np.arctan2(
            vecs[within_cutoff, 1], vecs[within_cutoff, 0]
        )
        phi[within_cutoff] = np.arctan2(
            np.linalg.norm(vecs[within_cutoff, :2], axis=-1), vecs[within_cutoff, 2]
        )
        return np.sum(sph_harm(m, l, theta, phi) * within_cutoff, axis=-1) / np.sum(
            within_cutoff, axis=-1
        )

    def get_steinhardt_parameter(self, l, cutoff_radius=np.inf):
        """
        Args:
            l (int/numpy.array): Order of Steinhardt parameter
            cutoff_radius (float): maximum neighbor distance to include (default: inf, i.e. all
            atoms included in the neighbor search).

        Returns:
            ( (natoms,) numpy.array) Steinhardt parameter values

        See more on https://pyscal.org/part3/steinhardt.html

        Note: This function does not have an internal algorithm to calculate a suitable cutoff
        radius. For automated uses, see Atoms.analyse.pyscal_steinhardt_parameter()
        """
        random_rotation = Rotation.from_mrp(np.random.random(3)).as_matrix()
        return np.sqrt(
            4
            * np.pi
            / (2 * l + 1)
            * np.sum(
                [
                    np.absolute(
                        self.get_spherical_harmonics(
                            l=l,
                            m=m,
                            cutoff_radius=cutoff_radius,
                            rotation=random_rotation,
                        )
                    )
                    ** 2
                    for m in np.arange(-l, l + 1)
                ],
                axis=0,
            )
        )

    @staticmethod
    def _get_all_possible_pairs(l):
        if l % 2 != 0:
            raise ValueError("Pairs cannot be formed for an uneven number of groups.")
        all_arr = np.array(list(itertools.permutations(np.arange(l), l)))
        all_arr = all_arr.reshape(len(all_arr), -1, 2)
        all_arr.sort(axis=-1)
        all_arr = all_arr[
            np.unique(all_arr.reshape(-1, l), axis=0, return_index=True)[1]
        ]
        indices = np.indices(all_arr.shape)
        all_arr = all_arr[
            indices[0], all_arr[:, :, 0].argsort(axis=-1)[:, :, np.newaxis], indices[2]
        ]
        return all_arr[np.unique(all_arr.reshape(-1, l), axis=0, return_index=True)[1]]

    @property
    def centrosymmetry(self):
        """
        Calculate centrosymmetry parameter for the given environment.

        cf. https://doi.org/10.1103/PhysRevB.58.11085

        NB: Currently very memory intensive for a large number of neighbors (works maybe up to 10)
        """
        all_arr = self._get_all_possible_pairs(self.distances.shape[-1])
        indices = np.indices((len(self.vecs),) + all_arr.shape[:-1])
        v = self.vecs[indices[0], all_arr[np.newaxis, :, :, 0]]
        v += self.vecs[indices[0], all_arr[np.newaxis, :, :, 1]]
        return np.sum(v**2, axis=-1).sum(axis=-1).min(axis=-1)

    def __getattr__(self, name):
        """Attributes for the mode. Same as setting `neigh.mode`."""
        if name not in ["filled", "ragged", "flattened"]:
            raise AttributeError(
                self.__class__.__name__ + " object has no attribute " + name
            )
        return Mode(name, self)

    def __dir__(self):
        """Attributes for the mode."""
        return ["filled", "ragged", "flattened"] + super().__dir__()


class Mode:
    """Helper class for mode

    Attributes: `distances`, `vecs`, `indices`, `shells`, `atom_numbers` and maybe more
    """

    def __init__(self, mode, ref_neigh):
        """
        Args:
            mode (str): Mode (`filled`, `ragged` or `flattened`)
            ref_neigh (Neighbors): Reference neighbor class
        """
        self.mode = mode
        self.ref_neigh = ref_neigh

    def __getattr__(self, name):
        """Return values according to their filling mode."""
        if "_" + name in self.ref_neigh.__dir__():
            name = "_" + name
        return self.ref_neigh._reshape(
            self.ref_neigh.__getattribute__(name), key=self.mode
        )

    def __dir__(self):
        """Show value names which are available for different filling modes."""
        return list(
            set(
                ["distances", "vecs", "indices", "shells", "atom_numbers"]
            ).intersection(self.ref_neigh.__dir__())
        )


class Neighbors(Tree):
    def __init__(self, ref_structure, tolerance=2):
        super().__init__(ref_structure=ref_structure)
        self._tolerance = tolerance
        self._cluster_vecs = None
        self._cluster_dist = None

    def __repr__(self):
        """
        Returns: __repr__
        """
        to_return = super().__repr__()
        return to_return.replace("given positions", "each atom")

    @property
    def chemical_symbols(self):
        """
        Returns chemical symbols of the neighboring atoms.

        Undefined neighbors (i.e. if the neighbor distance is beyond the cutoff radius) are
        considered as vacancies and are marked by 'v'
        """
        chemical_symbols = np.tile(["v"], self.filled.indices.shape).astype("<U2")
        cond = self.filled.indices < len(self._ref_structure)
        chemical_symbols[cond] = self._ref_structure.get_chemical_symbols()[
            self.filled.indices[cond]
        ]
        return chemical_symbols

    @property
    def shells(self):
        """
        Returns the cell numbers of each atom according to the distances
        """
        return self.get_local_shells(mode=self.mode)

    def get_local_shells(
        self,
        mode=None,
        tolerance=None,
        cluster_by_distances=False,
        cluster_by_vecs=False,
    ):
        """
        Set shell indices based on distances available to each atom. Clustering methods can be used
        at the same time, which will be useful at finite temperature results, but depending on how
        dispersed the atoms are, the algorithm could take some time. If the clustering method(-s)
        have already been launched before this function, it will use the results already available
        and does not execute the clustering method(-s) again.

        Args:
            mode (str): Representation of the variable. Choose from 'filled', 'ragged' and
                'flattened'.
            tolerance (int): decimals in np.round for rounding up distances
            cluster_by_distances (bool): If True, `cluster_by_distances` is called first and the distances obtained
                from the clustered distances are used to calculate the shells. If cluster_by_vecs is True at the same
                time, `cluster_by_distances` will use the clustered vectors for its clustering algorithm. For more,
                see the DocString of `cluster_by_distances`. (default: False)
            cluster_by_vecs (bool): If True, `cluster_by_vectors` is called first and the distances obtained from
                the clustered vectors are used to calculate the shells. (default: False)

        Returns:
            shells (numpy.ndarray): shell indices
        """
        if tolerance is None:
            tolerance = self._tolerance
        if mode is None:
            mode = self.mode
        if cluster_by_distances:
            if self._cluster_dist is None:
                self.cluster_by_distances(use_vecs=cluster_by_vecs)
            shells = np.array(
                [
                    np.unique(np.round(dist, decimals=tolerance), return_inverse=True)[
                        1
                    ]
                    + 1
                    for dist in self._cluster_dist.cluster_centers_[
                        self._cluster_dist.labels_
                    ]
                ]
            )
            shells[self._cluster_dist.labels_ < 0] = -1
            shells = shells.reshape(self.filled.indices.shape)
        elif cluster_by_vecs:
            if self._cluster_vecs is None:
                self.cluster_by_vecs()
            shells = np.array(
                [
                    np.unique(np.round(dist, decimals=tolerance), return_inverse=True)[
                        1
                    ]
                    + 1
                    for dist in np.linalg.norm(
                        self._cluster_vecs.cluster_centers_[self._cluster_vecs.labels_],
                        axis=-1,
                        ord=self.norm_order,
                    )
                ]
            )
            shells[self._cluster_vecs.labels_ < 0] = -1
            shells = shells.reshape(self.filled.indices.shape)
        else:
            distances = self.filled.distances.copy()
            distances[distances == np.inf] = np.max(distances[distances < np.inf]) + 1
            shells = np.array(
                [
                    np.unique(np.round(dist, decimals=tolerance), return_inverse=True)[
                        1
                    ]
                    + 1
                    for dist in distances
                ]
            )
            shells[self.filled.distances == np.inf] = -1
        return self._reshape(shells, key=mode)

    def get_global_shells(
        self,
        mode=None,
        tolerance=None,
        cluster_by_distances=False,
        cluster_by_vecs=False,
    ):
        """
        Set shell indices based on all distances available in the system instead of
        setting them according to the local distances (in contrast to shells defined
        as an attribute in this class). Clustering methods can be used at the same time,
        which will be useful at finite temperature results, but depending on how dispersed
        the atoms are, the algorithm could take some time. If the clustering method(-s)
        have already been launched before this function, it will use the results already
        available and does not execute the clustering method(-s) again.

        Args:
            mode (str): Representation of the variable. Choose from 'filled', 'ragged' and
                'flattened'.
            tolerance (int): decimals in np.round for rounding up distances (default: 2)
            cluster_by_distances (bool): If True, `cluster_by_distances` is called first and the distances obtained
                from the clustered distances are used to calculate the shells. If cluster_by_vecs is True at the same
                time, `cluster_by_distances` will use the clustered vectors for its clustering algorithm. For more,
                see the DocString of `cluster_by_distances`. (default: False)
            cluster_by_vecs (bool): If True, `cluster_by_vectors` is called first and the distances obtained from
                the clustered vectors are used to calculate the shells. (default: False)

        Returns:
            shells (numpy.ndarray): shell indices (cf. shells)
        """
        if tolerance is None:
            tolerance = self._tolerance
        if mode is None:
            mode = self.mode
        distances = self.filled.distances
        if cluster_by_distances:
            if self._cluster_dist is None:
                self.cluster_by_distances(use_vecs=cluster_by_vecs)
            distances = self._cluster_dist.cluster_centers_[
                self._cluster_dist.labels_
            ].reshape(self.filled.distances.shape)
            distances[self._cluster_dist.labels_ < 0] = np.inf
        elif cluster_by_vecs:
            if self._cluster_vecs is None:
                self.cluster_by_vecs()
            distances = np.linalg.norm(
                self._cluster_vecs.cluster_centers_[self._cluster_vecs.labels_],
                axis=-1,
                ord=self.norm_order,
            ).reshape(self.filled.distances.shape)
            distances[self._cluster_vecs.labels_ < 0] = np.inf
        dist_lst = np.unique(np.round(a=distances, decimals=tolerance))
        shells = -np.ones_like(self.filled.indices).astype(int)
        shells[distances < np.inf] = (
            np.absolute(
                distances[distances < np.inf, np.newaxis]
                - dist_lst[np.newaxis, dist_lst < np.inf]
            ).argmin(axis=-1)
            + 1
        )
        return self._reshape(shells, key=mode)

    def get_shell_matrix(
        self, chemical_pair=None, cluster_by_distances=False, cluster_by_vecs=False
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

        pairs = np.stack(
            (
                self.filled.indices,
                np.ones_like(self.filled.indices)
                * np.arange(len(self.filled.indices))[:, np.newaxis],
                self.get_global_shells(
                    cluster_by_distances=cluster_by_distances,
                    cluster_by_vecs=cluster_by_vecs,
                )
                - 1,
            ),
            axis=-1,
        ).reshape(-1, 3)
        shell_max = np.max(pairs[:, -1]) + 1
        if chemical_pair is not None:
            c = self._ref_structure.get_chemical_symbols()
            pairs = pairs[
                np.all(
                    np.sort(c[pairs[:, :2]], axis=-1) == np.sort(chemical_pair), axis=-1
                )
            ]
        shell_matrix = []
        for ind in np.arange(shell_max):
            indices = pairs[ind == pairs[:, -1]]
            if len(indices) > 0:
                ind_tmp = np.unique(indices[:, :-1], axis=0, return_counts=True)
                shell_matrix.append(
                    coo_matrix(
                        (ind_tmp[1], (ind_tmp[0][:, 0], ind_tmp[0][:, 1])),
                        shape=(len(self._ref_structure), len(self._ref_structure)),
                    )
                )
            else:
                shell_matrix.append(
                    coo_matrix((len(self._ref_structure), len(self._ref_structure)))
                )
        return shell_matrix

    def find_neighbors_by_vector(self, vector, return_deviation=False):
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

        z = np.zeros(len(self._ref_structure) * 3).reshape(-1, 3)
        v = np.append(z[:, np.newaxis, :], self.filled.vecs, axis=1)
        dist = np.linalg.norm(v - np.array(vector), axis=-1, ord=self.norm_order)
        indices = np.append(
            np.arange(len(self._ref_structure))[:, np.newaxis],
            self.filled.indices,
            axis=1,
        )
        if return_deviation:
            return indices[np.arange(len(dist)), np.argmin(dist, axis=-1)], np.min(
                dist, axis=-1
            )
        return indices[np.arange(len(dist)), np.argmin(dist, axis=-1)]

    def cluster_by_vecs(
        self,
        distance_threshold=None,
        n_clusters=None,
        linkage="complete",
        affinity="euclidean",
    ):
        """
        Method to group vectors which have similar values. This method should be used as a part of
        neigh.get_global_shells(cluster_by_vecs=True) or neigh.get_local_shells(cluster_by_vecs=True).
        However, in order to specify certain arguments (such as n_jobs or max_iter), it might help to
        have run this function before calling parent functions, as the data obtained with this function
        will be stored in the variable `_cluster_vecs`

        Args:
            distance_threshold (float/None): The linkage distance threshold above which, clusters
                will not be merged. (cf. sklearn.cluster.AgglomerativeClustering)
            n_clusters (int/None): The number of clusters to find.
                (cf. sklearn.cluster.AgglomerativeClustering)
            linkage (str): Which linkage criterion to use. The linkage criterion determines which
                distance to use between sets of observation. The algorithm will merge the pairs of
                cluster that minimize this criterion. (cf. sklearn.cluster.AgglomerativeClustering)
            affinity (str/callable): Metric used to compute the linkage. Can be `euclidean`, `l1`,
                `l2`, `manhattan`, `cosine`, or `precomputed`. If linkage is `ward`, only
                `euclidean` is accepted.

        """
        if distance_threshold is None and n_clusters is None:
            distance_threshold = np.min(self.filled.distances)
        dr = self.flattened.vecs
        self._cluster_vecs = AgglomerativeClustering(
            distance_threshold=distance_threshold,
            n_clusters=n_clusters,
            linkage=linkage,
            affinity=affinity,
        ).fit(dr)
        self._cluster_vecs.cluster_centers_ = get_average_of_unique_labels(
            self._cluster_vecs.labels_, dr
        )
        new_labels = -np.ones_like(self.filled.indices).astype(int)
        new_labels[self.filled.distances < np.inf] = self._cluster_vecs.labels_
        self._cluster_vecs.labels_ = new_labels

    def cluster_by_distances(
        self,
        distance_threshold=None,
        n_clusters=None,
        linkage="complete",
        affinity="euclidean",
        use_vecs=False,
    ):
        """
        Method to group vectors which have similar lengths. This method should be used as a part of
        neigh.get_global_shells(cluster_by_vecs=True) or
        neigh.get_local_shells(cluster_by_distances=True).  However, in order to specify certain
        arguments (such as n_jobs or max_iter), it might help to have run this function before
        calling parent functions, as the data obtained with this function will be stored in the
        variable `_cluster_distances`

        Args:
            distance_threshold (float/None): The linkage distance threshold above which, clusters
                will not be merged. (cf. sklearn.cluster.AgglomerativeClustering)
            n_clusters (int/None): The number of clusters to find.
                (cf. sklearn.cluster.AgglomerativeClustering)
            linkage (str): Which linkage criterion to use. The linkage criterion determines which
                distance to use between sets of observation. The algorithm will merge the pairs of
                cluster that minimize this criterion. (cf. sklearn.cluster.AgglomerativeClustering)
            affinity (str/callable): Metric used to compute the linkage. Can be `euclidean`, `l1`,
                `l2`, `manhattan`, `cosine`, or `precomputed`. If linkage is `ward`, only
                `euclidean` is accepted.
            use_vecs (bool): Whether to form clusters for vecs beforehand. If true, the distances
                obtained from the clustered vectors is used for the distance clustering. Otherwise
                neigh.distances is used.
        """
        if distance_threshold is None:
            distance_threshold = 0.1 * np.min(self.flattened.distances)
        dr = self.flattened.distances
        if use_vecs:
            if self._cluster_vecs is None:
                self.cluster_by_vecs()
            labels_to_consider = self._cluster_vecs.labels_[
                self._cluster_vecs.labels_ >= 0
            ]
            dr = np.linalg.norm(
                self._cluster_vecs.cluster_centers_[labels_to_consider],
                axis=-1,
                ord=self.norm_order,
            )
        self._cluster_dist = AgglomerativeClustering(
            distance_threshold=distance_threshold,
            n_clusters=n_clusters,
            linkage=linkage,
            affinity=affinity,
        ).fit(dr.reshape(-1, 1))
        self._cluster_dist.cluster_centers_ = get_average_of_unique_labels(
            self._cluster_dist.labels_, dr
        )
        new_labels = -np.ones_like(self.filled.indices).astype(int)
        new_labels[self.filled.distances < np.inf] = self._cluster_dist.labels_
        self._cluster_dist.labels_ = new_labels

    def reset_clusters(self, vecs=True, distances=True):
        """
        Method to reset clusters.

        Args:
            vecs (bool): Reset `_cluster_vecs` (cf. `cluster_by_vecs`)
            distances (bool): Reset `_cluster_distances` (cf. `cluster_by_distances`)
        """
        if vecs:
            self._cluster_vecs = None
        if distances:
            self._cluster_distances = None

    def cluster_analysis(self, id_list, return_cluster_sizes=False):
        """

        Args:
            id_list:
            return_cluster_sizes:

        Returns:

        """
        self._cluster = [0] * len(self._ref_structure)
        c_count = 1
        # element_list = self.get_atomic_numbers()
        for ia in id_list:
            # el0 = element_list[ia]
            nbrs = self.ragged.indices[ia]
            # print ("nbrs: ", ia, nbrs)
            if self._cluster[ia] == 0:
                self._cluster[ia] = c_count
                self.__probe_cluster(c_count, nbrs, id_list)
                c_count += 1

        cluster = np.array(self._cluster)
        cluster_dict = {
            i_c: np.where(cluster == i_c)[0].tolist() for i_c in range(1, c_count)
        }
        if return_cluster_sizes:
            sizes = [self._cluster.count(i_c + 1) for i_c in range(c_count - 1)]
            return cluster_dict, sizes

        return cluster_dict  # sizes

    def __probe_cluster(self, c_count, neighbors, id_list):
        """

        Args:
            c_count:
            neighbors:
            id_list:

        Returns:

        """
        for nbr_id in neighbors:
            if self._cluster[nbr_id] == 0:
                if nbr_id in id_list:  # TODO: check also for ordered structures
                    self._cluster[nbr_id] = c_count
                    nbrs = self.ragged.indices[nbr_id]
                    self.__probe_cluster(c_count, nbrs, id_list)

    # TODO: combine with corresponding routine in plot3d
    def get_bonds(self, radius=np.inf, max_shells=None, prec=0.1):
        """

        Args:
            radius:
            max_shells:
            prec: minimum distance between any two clusters (if smaller considered to be single cluster)

        Returns:

        """

        def get_cluster(dist_vec, ind_vec, prec=prec):
            ind_where = np.where(np.diff(dist_vec) > prec)[0] + 1
            ind_vec_cl = [np.sort(group) for group in np.split(ind_vec, ind_where)]
            return ind_vec_cl

        dist = self.filled.distances
        ind = self.ragged.indices
        el_list = self._ref_structure.get_chemical_symbols()

        ind_shell = []
        for d, i in zip(dist, ind):
            id_list = get_cluster(d[d < radius], i[d < radius])
            # print ("id: ", d[d<radius], id_list, dist_lst)
            ia_shells_dict = {}
            for i_shell_list in id_list:
                ia_shell_dict = {}
                for i_s in i_shell_list:
                    el = el_list[i_s]
                    if el not in ia_shell_dict:
                        ia_shell_dict[el] = []
                    ia_shell_dict[el].append(i_s)
                for el, ia_lst in ia_shell_dict.items():
                    if el not in ia_shells_dict:
                        ia_shells_dict[el] = []
                    if max_shells is not None:
                        if len(ia_shells_dict[el]) + 1 > max_shells:
                            continue
                    ia_shells_dict[el].append(ia_lst)
            ind_shell.append(ia_shells_dict)
        return ind_shell


Neighbors.__doc__ = Tree.__doc__


def get_volume_of_n_sphere_in_p_norm(n=3, p=2):
    """
    Volume of an n-sphere in p-norm. For more info:

    https://en.wikipedia.org/wiki/Volume_of_an_n-ball#Balls_in_Lp_norms
    """
    return (2 * gamma(1 + 1 / p)) ** n / gamma(1 + n / p)


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
