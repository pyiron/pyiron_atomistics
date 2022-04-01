# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from scipy.sparse import coo_matrix
from scipy.spatial import Voronoi, Delaunay
from pyiron_atomistics.atomistics.structure.pyscal import (
    get_steinhardt_parameter_structure,
    analyse_cna_adaptive,
    analyse_centro_symmetry,
    analyse_diamond_structure,
    analyse_voronoi_volume,
    analyse_find_solids,
)
from pyiron_atomistics.atomistics.structure.strain import Strain
from pyiron_base.generic.util import Deprecator
from scipy.spatial import ConvexHull

deprecate = Deprecator()

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


def get_mean_positions(positions, cell, pbc, labels):
    """
    This function calculates the average position(-s) across periodic boundary conditions according
    to the labels

    Args:
        positions (numpy.ndarray (n, 3)): Coordinates to be averaged
        cell (numpy.ndarray (3, 3)): Cell dimensions
        pbc (numpy.ndarray (3,)): Periodic boundary conditions (in boolean)
        labels (numpy.ndarray (n,)): labels according to which the atoms are grouped

    Returns:
        (numpy.ndarray): mean positions
    """
    # Translate labels to integer enumeration (0, 1, 2, ... etc.) and get their counts
    _, labels, counts = np.unique(labels, return_inverse=True, return_counts=True)
    # Get reference point for each unique label
    mean_positions = positions[np.unique(labels, return_index=True)[1]]
    # Get displacement vectors from reference points to all other points for the same labels
    all_positions = positions - mean_positions[labels]
    # Account for pbc
    all_positions = np.einsum("ji,nj->ni", np.linalg.inv(cell), all_positions)
    all_positions[:, pbc] -= np.rint(all_positions)[:, pbc]
    all_positions = np.einsum("ji,nj->ni", cell, all_positions)
    # Add average displacement vector of each label to the reference point
    np.add.at(mean_positions, labels, (all_positions.T / counts[labels]).T)
    return mean_positions


def get_average_of_unique_labels(labels, values):
    """

    This function returns the average values of those elements, which share the same labels

    Example:

    >>> labels = [0, 1, 0, 2]
    >>> values = [0, 1, 2, 3]
    >>> print(get_average_of_unique_labels(labels, values))
    array([1, 1, 3])

    """
    labels = np.unique(labels, return_inverse=True)[1]
    unique_labels = np.unique(labels)
    mat = coo_matrix((np.ones_like(labels), (labels, np.arange(len(labels)))))
    mean_values = np.asarray(
        mat.dot(np.asarray(values).reshape(len(labels), -1)) / mat.sum(axis=1)
    )
    if np.prod(mean_values.shape).astype(int) == len(unique_labels):
        return mean_values.flatten()
    return mean_values


class Interstitials:
    """
    Class to search for interstitial sites

    This class internally does the following steps:

        1. Initialize grid points (or Voronoi vertices) which are considered as
            interstitial site candidates.
        2. Eliminate points within a distance from the nearest neighboring atoms as
            given by `min_distance`
        3. Initialize neighbor environment using `get_neighbors`
        4. Shift interstitial candidates to the nearest symmetric points with respect to the
            neighboring atom sites/vertices.
        5. Kick out points with large neighbor distance variances; this eliminates "irregular"
            shaped interstitials
        6. Cluster interstitial candidates to avoid point overlapping.

    The interstitial sites can be obtained via `positions`

    In complex structures (i.e. grain boundary, dislocation etc.), the default parameters
    should be chosen properly. In order to see other quantities, which potentially
    characterize interstitial sites, see the following class methods:

        - `get_variances()`
        - `get_distances()`
        - `get_steinhardt_parameters()`
        - `get_volumes()`
        - `get_areas()`
    """

    def __init__(
        self,
        structure,
        num_neighbors,
        n_gridpoints_per_angstrom=5,
        min_distance=1,
        use_voronoi=False,
        variance_buffer=0.01,
        n_iterations=2,
        eps=0.1,
    ):
        """

        Args:
            num_neighbors (int): Number of neighbors/vertices to consider for the interstitial
                sites. By definition, tetrahedral sites should have 4 vertices and octahedral
                sites 6.
            n_gridpoints_per_angstrom (int): Number of grid points per angstrom for the
                initialization of the interstitial candidates. The finer the mesh (i.e. the larger
                the value), the likelier it is to find the correct sites but then also it becomes
                computationally more expensive. Ignored if `use_voronoi` is set to `True`
            min_distance (float): Minimum distance from the nearest neighboring atoms to the
                positions for them to be considered as interstitial site candidates. Set
                `min_distance` to 0 if no point should be removed.
            use_voronoi (bool): Use Voronoi vertices for the initial interstitial candidate
                positions instead of grid points.
            variance_buffer (bool): Maximum permitted variance value (in distance unit) of the
                neighbor distance values with respect to the minimum value found for each point.
                It should be close to 0 for perfect crystals and slightly higher values for
                structures containing defects. Set `variance_buffer` to `numpy.inf` if no selection
                by variance value should take place.
            n_iterations (int): Number of iterations for the shifting of the candidate positions
                to the nearest symmetric positions with respect to `num_neighbors`. In most of the
                cases, 1 is enough. In some rare cases (notably tetrahedral sites in bcc), it
                should be at least 2. It is unlikely that it has to be larger than 2. Set
                `n_iterations` to 0 if no shifting should take place.
            eps (float): Distance below which two interstitial candidate sites to be considered as
                one site after the symmetrization of the points. Set `eps` to 0 if clustering should
                not be done.
        """
        self._hull = None
        self._neigh = None
        self._positions = None
        self.num_neighbors = num_neighbors
        self.structure = structure
        self._initialize(
            n_gridpoints_per_angstrom=n_gridpoints_per_angstrom,
            min_distance=min_distance,
            use_voronoi=use_voronoi,
            variance_buffer=variance_buffer,
            n_iterations=n_iterations,
            eps=eps,
        )

    def _initialize(
        self,
        n_gridpoints_per_angstrom=5,
        min_distance=1,
        use_voronoi=False,
        variance_buffer=0.01,
        n_iterations=2,
        eps=0.1,
    ):
        if use_voronoi:
            self.positions = self.structure.analyse.get_voronoi_vertices()
        else:
            self.positions = self._create_gridpoints(
                n_gridpoints_per_angstrom=n_gridpoints_per_angstrom
            )
        self._remove_too_close(min_distance=min_distance)
        for _ in range(n_iterations):
            self._set_interstitials_to_high_symmetry_points()
        self._kick_out_points(variance_buffer=variance_buffer)
        self._cluster_points(eps=eps)

    @property
    def num_neighbors(self):
        """
        Number of atoms (vertices) to consider for each interstitial atom. By definition,
        tetrahedral sites should have 4 and octahedral sites 6.
        """
        return self._num_neighbors

    @num_neighbors.setter
    def num_neighbors(self, n):
        self.reset()
        self._num_neighbors = n

    def reset(self):
        self._hull = None
        self._neigh = None

    @property
    def neigh(self):
        """
        Neighborhood information of each interstitial candidate and their surrounding atoms. E.g.
        `class.neigh.distances[0][0]` gives the distance from the first interstitial candidate to
        its nearest neighboring atoms. The functionalities of `neigh` follow those of
        `pyiron_atomistics.structure.atoms.neighbors`.
        """
        if self._neigh is None:
            self._neigh = self.structure.get_neighborhood(
                self.positions, num_neighbors=self.num_neighbors
            )
        return self._neigh

    @property
    def positions(self):
        """
        Positions of the interstitial candidates (and not those of the atoms).

        IMPORTANT: Do not set positions via numpy setter, i.e.

        BAD:
        ```
        >>> Interstitials.neigh.positions[0][0] = x
        ```

        GOOD:
        ```
        >>> positions = Interstitials.neigh.positions
        >>> positions[0][0] = x
        >>> Interstitialsneigh.positions = positions
        ```

        This is because in the first case related properties (most importantly the neighborhood
        information) is not updated, which might lead to inconsistencies.
        """
        return self._positions

    @positions.setter
    def positions(self, x):
        self.reset()
        self._positions = x

    @property
    def hull(self):
        """
        Convex hull of each atom. It is mainly used for the volume and area calculation of each
        interstitial candidate. For more info, see `get_volumes` and `get_areas`.
        """
        if self._hull is None:
            self._hull = [ConvexHull(v) for v in self.neigh.vecs]
        return self._hull

    def _create_gridpoints(self, n_gridpoints_per_angstrom=5):
        cell = self.structure.get_vertical_length()
        n_points = (n_gridpoints_per_angstrom * cell).astype(int)
        positions = np.meshgrid(
            *[np.linspace(0, 1, n_points[i], endpoint=False) for i in range(3)]
        )
        positions = np.stack(positions, axis=-1).reshape(-1, 3)
        return np.einsum("ji,nj->ni", self.structure.cell, positions)

    def _remove_too_close(self, min_distance=1):
        neigh = self.structure.get_neighborhood(self.positions, num_neighbors=1)
        self.positions = self.positions[neigh.distances.flatten() > min_distance]

    def _set_interstitials_to_high_symmetry_points(self):
        self.positions = self.positions + np.mean(self.neigh.vecs, axis=-2)
        self.positions = self.structure.get_wrapped_coordinates(self.positions)

    def _kick_out_points(self, variance_buffer=0.01):
        variance = self.get_variances()
        min_var = variance.min()
        self.positions = self.positions[variance < min_var + variance_buffer]

    def _cluster_points(self, eps=0.1):
        if eps == 0:
            return
        extended_positions, indices = self.structure.get_extended_positions(
            eps, return_indices=True, positions=self.positions
        )
        labels = DBSCAN(eps=eps, min_samples=1).fit_predict(extended_positions)
        coo = coo_matrix((labels, (np.arange(len(labels)), indices)))
        labels = coo.max(axis=0).toarray().flatten()
        self.positions = get_mean_positions(
            self.positions, self.structure.cell, self.structure.pbc, labels
        )

    def get_variances(self):
        """
        Get variance of neighboring distances. Since interstitial sites are mostly in symmetric
        sites, the variance values tend to be small. In the case of fcc, both tetrahedral and
        octahedral sites as well as tetrahedral sites in bcc should have the value of 0.

        Returns:
            (numpy.array (n,)) Variance values
        """
        return np.std(self.neigh.distances, axis=-1)

    def get_distances(self, function_to_apply=np.min):
        """
        Get per-position return values of a given function for the neighbors.

        Args:
            function_to_apply (function): Function to apply to the distance array. Default is
                numpy.minimum

        Returns:
            (numpy.array (n,)) Function values on the distance array
        """
        return function_to_apply(self.neigh.distances, axis=-1)

    def get_steinhardt_parameters(self, l):
        """
        Args:
            l (int/numpy.array): Order of Steinhardt parameter

        Returns:
            (numpy.array (n,)) Steinhardt parameter values
        """
        return self.neigh.get_steinhardt_parameter(l=l)

    def get_volumes(self):
        """
        Returns:
            (numpy.array (n,)): Convex hull volume of each site.
        """
        return np.array([h.volume for h in self.hull])

    def get_areas(self):
        """
        Returns:
            (numpy.array (n,)): Convex hull area of each site.
        """
        return np.array([h.area for h in self.hull])


class Analyse:
    """Class to analyse atom structure."""

    def __init__(self, structure):
        """
        Args:
            structure (:class:`pyiron.atomistics.structure.atoms.Atoms`): reference Atom structure.
        """
        self._structure = structure

    def get_interstitials(
        self,
        num_neighbors,
        n_gridpoints_per_angstrom=5,
        min_distance=1,
        use_voronoi=False,
        variance_buffer=0.01,
        n_iterations=2,
        eps=0.1,
    ):
        return Interstitials(
            structure=self._structure,
            num_neighbors=num_neighbors,
            n_gridpoints_per_angstrom=n_gridpoints_per_angstrom,
            min_distance=min_distance,
            use_voronoi=use_voronoi,
            variance_buffer=variance_buffer,
            n_iterations=n_iterations,
            eps=eps,
        )

    get_interstitials.__doc__ = (
        Interstitials.__doc__.replace("Class", "Function")
        + Interstitials.__init__.__doc__
    )

    def get_layers(
        self,
        distance_threshold=0.01,
        id_list=None,
        wrap_atoms=True,
        planes=None,
        cluster_method=None,
    ):
        """
        Get an array of layer numbers.

        Args:
            distance_threshold (float): Distance below which two points are
                considered to belong to the same layer. For detailed
                description: sklearn.cluster.AgglomerativeClustering
            id_list (list/numpy.ndarray): List of atoms for which the layers
                should be considered.
            wrap_atoms (bool): Whether to consider periodic boundary conditions according to the box definition or not.
                If set to `False`, atoms lying on opposite box boundaries are considered to belong to different layers,
                regardless of whether the box itself has the periodic boundary condition in this direction or not.
                If `planes` is not `None` and `wrap_atoms` is `True`, this tag has the same effect as calling
                `get_layers()` after calling `center_coordinates_in_unit_cell()`
            planes (list/numpy.ndarray): Planes along which the layers are calculated. Planes are
                given in vectors, i.e. [1, 0, 0] gives the layers along the x-axis. Default planes
                are orthogonal unit vectors: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]. If you have a
                tilted box and want to calculate the layers along the directions of the cell
                vectors, use `planes=np.linalg.inv(structure.cell).T`. Whatever values are
                inserted, they are internally normalized, so whether [1, 0, 0] is entered or
                [2, 0, 0], the results will be the same.
            cluster_method (scikit-learn cluster algorithm): if given overrides the clustering method used, must be an
                instance of a cluster algorithm from scikit-learn (or compatible interface)

        Returns: Array of layer numbers (same shape as structure.positions)

        Example I - how to get the number of layers in each direction:

        >>> structure = Project('.').create_structure('Fe', 'bcc', 2.83).repeat(5)
        >>> print('Numbers of layers:', np.max(structure.analyse.get_layers(), axis=0)+1)

        Example II - get layers of only one species:

        >>> print('Iron layers:', structure.analyse.get_layers(
        ...       id_list=structure.select_index('Fe')))

        The clustering algorithm can be changed with the cluster_method argument

        >>> from sklearn.cluster import DBSCAN
        >>> layers = structure.analyse.get_layers(cluster_method=DBSCAN())
        """
        if distance_threshold <= 0:
            raise ValueError("distance_threshold must be a positive float")
        if id_list is not None and len(id_list) == 0:
            raise ValueError("id_list must contain at least one id")
        if wrap_atoms and planes is None:
            positions, indices = self._structure.get_extended_positions(
                width=distance_threshold, return_indices=True
            )
            if id_list is not None:
                id_list = np.arange(len(self._structure))[np.array(id_list)]
                id_list = np.any(
                    id_list[:, np.newaxis] == indices[np.newaxis, :], axis=0
                )
                positions = positions[id_list]
                indices = indices[id_list]
        else:
            positions = self._structure.positions
            if id_list is not None:
                positions = positions[id_list]
            if wrap_atoms:
                positions = self._structure.get_wrapped_coordinates(positions)
        if planes is not None:
            mat = np.asarray(planes).reshape(-1, 3)
            positions = np.einsum(
                "ij,i,nj->ni", mat, 1 / np.linalg.norm(mat, axis=-1), positions
            )
        if cluster_method is None:
            cluster_method = AgglomerativeClustering(
                linkage="complete",
                n_clusters=None,
                distance_threshold=distance_threshold,
            )
        layers = []
        for ii, x in enumerate(positions.T):
            cluster = cluster_method.fit(x.reshape(-1, 1))
            first_occurrences = np.unique(cluster.labels_, return_index=True)[1]
            permutation = x[first_occurrences].argsort().argsort()
            labels = permutation[cluster.labels_]
            if wrap_atoms and planes is None and self._structure.pbc[ii]:
                mean_positions = get_average_of_unique_labels(labels, positions)
                scaled_positions = np.einsum(
                    "ji,nj->ni", np.linalg.inv(self._structure.cell), mean_positions
                )
                unique_inside_box = np.all(
                    np.absolute(scaled_positions - 0.5 + 1.0e-8) < 0.5, axis=-1
                )
                arr_inside_box = np.any(
                    labels[:, None] == np.unique(labels)[unique_inside_box][None, :],
                    axis=-1,
                )
                first_occurences = np.unique(
                    indices[arr_inside_box], return_index=True
                )[1]
                labels = labels[arr_inside_box]
                labels -= np.min(labels)
                labels = labels[first_occurences]
            layers.append(labels)
        if planes is not None and len(np.asarray(planes).shape) == 1:
            return np.asarray(layers).flatten()
        return np.vstack(layers).T

    @deprecate(
        arguments={"clustering": "use n_clusters=None instead of clustering=False."}
    )
    def pyscal_steinhardt_parameter(
        self,
        neighbor_method="cutoff",
        cutoff=0,
        n_clusters=2,
        q=None,
        averaged=False,
        clustering=None,
    ):
        """
        Calculate Steinhardts parameters

        Args:
            neighbor_method (str) : can be ['cutoff', 'voronoi']. (Default is 'cutoff'.)
            cutoff (float) : Can be 0 for adaptive cutoff or any other value. (Default is 0, adaptive.)
            n_clusters (int/None) : Number of clusters for K means clustering or None to not cluster. (Default is 2.)
            q (list) : Values can be integers from 2-12, the required q values to be calculated. (Default is None, which
                uses (4, 6).)
            averaged (bool) : If True, calculates the averaged versions of the parameter. (Default is False.)

        Returns:
            numpy.ndarray: (number of q's, number of atoms) shaped array of q parameters
            numpy.ndarray: If `clustering=True`, an additional per-atom array of cluster ids is also returned
        """
        return get_steinhardt_parameter_structure(
            self._structure,
            neighbor_method=neighbor_method,
            cutoff=cutoff,
            n_clusters=n_clusters,
            q=q,
            averaged=averaged,
            clustering=clustering,
        )

    def pyscal_cna_adaptive(self, mode="total", ovito_compatibility=False):
        """
        Use common neighbor analysis

        Args:
            mode ("total"/"numeric"/"str"): Controls the style and level
                of detail of the output.
                - total : return number of atoms belonging to each structure
                - numeric : return a per atom list of numbers- 0 for unknown,
                    1 fcc, 2 hcp, 3 bcc and 4 icosa
                - str : return a per atom string of structures
            ovito_compatibility(bool): use ovito compatibility mode

        Returns:
            (depends on `mode`)
        """
        return analyse_cna_adaptive(
            atoms=self._structure, mode=mode, ovito_compatibility=ovito_compatibility
        )

    def pyscal_centro_symmetry(self, num_neighbors=12):
        """
        Analyse centrosymmetry parameter

        Args:
            num_neighbors (int) : number of neighbors

        Returns:
            list: list of centrosymmetry parameter
        """
        return analyse_centro_symmetry(
            atoms=self._structure, num_neighbors=num_neighbors
        )

    def pyscal_diamond_structure(self, mode="total", ovito_compatibility=False):
        """
        Analyse diamond structure

        Args:
            mode ("total"/"numeric"/"str"): Controls the style and level
            of detail of the output.
                - total : return number of atoms belonging to each structure
                - numeric : return a per atom list of numbers- 0 for unknown,
                    1 fcc, 2 hcp, 3 bcc and 4 icosa
                - str : return a per atom string of structures
            ovito_compatibility(bool): use ovito compatibility mode

        Returns:
            (depends on `mode`)
        """
        return analyse_diamond_structure(
            atoms=self._structure, mode=mode, ovito_compatibility=ovito_compatibility
        )

    def pyscal_voronoi_volume(self):
        """Calculate the Voronoi volume of atoms"""
        return analyse_voronoi_volume(atoms=self._structure)

    def pyscal_find_solids(
        self,
        neighbor_method="cutoff",
        cutoff=0,
        bonds=0.5,
        threshold=0.5,
        avgthreshold=0.6,
        cluster=False,
        q=6,
        right=True,
        return_sys=False,
    ):
        """
        Get the number of solids or the corresponding pyscal system.
        Calls necessary pyscal methods as described in https://pyscal.org/en/latest/methods/03_solidliquid.html.

        Args:
            neighbor_method (str, optional): Method used to get neighborlist. See pyscal documentation. Defaults to "cutoff".
            cutoff (int, optional): Adaptive if 0. Defaults to 0.
            bonds (float, optional): Number or fraction of bonds to consider atom as solid. Defaults to 0.5.
            threshold (float, optional): See pyscal documentation. Defaults to 0.5.
            avgthreshold (float, optional): See pyscal documentation. Defaults to 0.6.
            cluster (bool, optional): See pyscal documentation. Defaults to False.
            q (int, optional): Steinhard parameter to calculate. Defaults to 6.
            right (bool, optional): See pyscal documentation. Defaults to True.
            return_sys (bool, optional): Whether to return number of solid atoms or pyscal system. Defaults to False.

        Returns:
            int: number of solids,
            pyscal system: pyscal system when return_sys=True
        """
        return analyse_find_solids(
            atoms=self._structure,
            neighbor_method=neighbor_method,
            cutoff=cutoff,
            bonds=bonds,
            threshold=threshold,
            avgthreshold=avgthreshold,
            cluster=cluster,
            q=q,
            right=right,
            return_sys=return_sys,
        )

    def get_voronoi_vertices(
        self, epsilon=2.5e-4, distance_threshold=0, width_buffer=10
    ):
        """
        Get voronoi vertices of the box.

        Args:
            epsilon (float): displacement to add to avoid wrapping of atoms at borders
            distance_threshold (float): distance below which two vertices are considered as one.
                Agglomerative clustering algorithm (sklearn) is employed. Final positions are given
                as the average positions of clusters.
            width_buffer (float): width of the layer to be added to account for pbc.

        Returns:
            numpy.ndarray: 3d-array of vertices

        This function detect octahedral and tetrahedral sites in fcc; in bcc it detects tetrahedral
        sites. In defects (e.g. vacancy, dislocation, grain boundary etc.), it gives a list of
        positions interstitial atoms might want to occupy. In order for this to be more successful,
        it might make sense to look at the distance between the voronoi vertices and their nearest
        neighboring atoms via:

        >>> voronoi_vertices = structure_of_your_choice.analyse.get_voronoi_vertices()
        >>> neigh = structure_of_your_choice.get_neighborhood(voronoi_vertices)
        >>> print(neigh.distances.min(axis=-1))

        """
        voro = Voronoi(self._structure.get_extended_positions(width_buffer) + epsilon)
        xx = voro.vertices
        if distance_threshold > 0:
            cluster = AgglomerativeClustering(
                linkage="single", distance_threshold=distance_threshold, n_clusters=None
            )
            cluster.fit(xx)
            xx = get_average_of_unique_labels(cluster.labels_, xx)
        xx = xx[
            np.linalg.norm(
                xx - self._structure.get_wrapped_coordinates(xx, epsilon=0), axis=-1
            )
            < epsilon
        ]
        return xx - epsilon

    def get_strain(self, ref_structure, num_neighbors=None, only_bulk_type=False):
        """
        Calculate local strain of each atom following the Lagrangian strain tensor:

        strain = (F^T x F - 1)/2

        where F is the atomic deformation gradient.

        Args:
            ref_structure (pyiron_atomistics.atomistics.structure.Atoms): Reference bulk structure
                (against which the strain is calculated)
            num_neighbors (int): Number of neighbors to take into account to calculate the local
                frame. If not specified, it is estimated based on cna analysis (only available if
                the bulk structure is bcc, fcc or hcp).
            only_bulk_type (bool): Whether to calculate the strain of all atoms or only for those
                which cna considers has the same crystal structure as the bulk. Those which have
                a different crystal structure will get 0 strain.

        Returns:
            ((n_atoms, 3, 3)-array): Strain tensors

        Example:

        >>> from pyiron_atomistics import Project
        >>> pr = Project('.')
        >>> bulk = pr.create.structure.bulk('Fe', cubic=True)
        >>> structure = bulk.apply_strain(np.random.random((3,3))*0.1, return_box=True)
        >>> structure.analyse.get_strain(bulk)

        .. attention:: Differs from :meth:`.Atoms.apply_strain`!
            This strain is not the same as the strain applied in `Atoms.apply_strain`, which
            multiplies the strain tensor (plus identity matrix) with the basis vectors, while
            here it follows the definition given by the Lagrangian strain tensor. For small
            strain values they give similar results (i.e. when strain**2 can be neglected).

        """
        return Strain(
            structure=self._structure,
            ref_structure=ref_structure,
            num_neighbors=num_neighbors,
            only_bulk_type=only_bulk_type,
        ).strain

    def _get_neighbors(
        self,
        position_interpreter,
        data_field: str,
        width_buffer: float = 10,
    ) -> np.ndarray:
        positions, indices = self._structure.get_extended_positions(
            width_buffer, return_indices=True
        )
        interpretation = position_interpreter(positions)
        data = getattr(interpretation, data_field)
        x = positions[data]
        return indices[
            data[
                np.isclose(self._structure.get_wrapped_coordinates(x), x)
                .all(axis=-1)
                .any(axis=-1)
            ]
        ]

    def get_voronoi_neighbors(self, width_buffer: float = 10) -> np.ndarray:
        """
        Get pairs of atom indices sharing the same Voronoi vertices/areas.

        Args:
            width_buffer (float): Width of the layer to be added to account for pbc.

        Returns:
            pairs (ndarray): Pair indices
        """
        return self._get_neighbors(Voronoi, "ridge_points", width_buffer=width_buffer)

    def get_delaunay_neighbors(self, width_buffer: float = 10.0) -> np.ndarray:
        """
        Get indices of atoms sharing the same Delaunay tetrahedrons (commonly known as Delaunay
        triangles), i.e. indices of neighboring atoms, which form a tetrahedron, in which no other
        atom exists.

        Args:
            width_buffer (float): Width of the layer to be added to account for pbc.

        Returns:
            pairs (ndarray): Delaunay neighbor indices
        """
        return self._get_neighbors(Delaunay, "simplices", width_buffer=width_buffer)

    def cluster_positions(
        self, positions=None, eps=1, buffer_width=None, return_labels=False
    ):
        """
        Cluster positions according to the distances. Clustering algorithm uses DBSCAN:

        https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html

        Example I:

        ```
        analyse = Analyze(some_pyiron_structure)
        positions = analyse.cluster_points(eps=2)
        ```

        This example should return the atom positions, if no two atoms lie within a distance of 2.
        If there are at least two atoms which lie within a distance of 2, their entries will be
        replaced by their mean position.

        Example II:

        ```
        analyse = Analyze(some_pyiron_structure)
        print(analyse.cluster_positions([3*[0.], 3*[1.]], eps=3))
        ```

        This returns `[0.5, 0.5, 0.5]` (if the cell is large enough)

        Args:
            positions (numpy.ndarray): Positions to consider. Default: atom positions
            eps (float): The maximum distance between two samples for one to be considered as in
                the neighborhood of the other.
            buffer_width (float): Buffer width to consider across the periodic boundary
                conditions. If too small, it is possible that atoms that are meant to belong
                together across PBC are missed. Default: Same as eps
            return_labels (bool): Whether to return the labels given according to the grouping
                together with the mean positions

        Returns:
            positions (numpy.ndarray): Mean positions
            label (numpy.ndarray): Labels of the positions (returned when `return_labels = True`)
        """
        positions = (
            self._structure.positions if positions is None else np.array(positions)
        )
        if buffer_width is None:
            buffer_width = eps
        extended_positions, indices = self._structure.get_extended_positions(
            buffer_width, return_indices=True, positions=positions
        )
        labels = DBSCAN(eps=eps, min_samples=1).fit_predict(extended_positions)
        coo = coo_matrix((labels, (np.arange(len(labels)), indices)))
        labels = coo.max(axis=0).toarray().flatten()
        # make labels look nicer
        labels = np.unique(labels, return_inverse=True)[1]
        mean_positions = get_mean_positions(
            positions, self._structure.cell, self._structure.pbc, labels
        )
        if return_labels:
            return mean_positions, labels
        return mean_positions
