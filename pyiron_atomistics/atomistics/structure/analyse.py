# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import numpy as np
from pyiron_atomistics.atomistics.structure.pyscal import (
    get_steinhardt_parameter_structure,
    analyse_cna_adaptive,
    analyse_centro_symmetry,
    analyse_diamond_structure,
    analyse_voronoi_volume,
    analyse_find_solids,
)
from structuretoolkit.analyse import (
    get_strain,
    get_interstitials,
    get_layers,
    get_voronoi_vertices,
    get_voronoi_neighbors,
    get_delaunay_neighbors,
    get_cluster_positions,
)
from pyiron_base import Deprecator

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
        return get_interstitials(
            structure=self._structure,
            num_neighbors=num_neighbors,
            n_gridpoints_per_angstrom=n_gridpoints_per_angstrom,
            min_distance=min_distance,
            use_voronoi=use_voronoi,
            variance_buffer=variance_buffer,
            n_iterations=n_iterations,
            eps=eps,
        )

    get_interstitials.__doc__ = get_interstitials.__doc__

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
        return get_layers(
            structure=self._structure,
            distance_threshold=distance_threshold,
            id_list=id_list,
            wrap_atoms=wrap_atoms,
            planes=planes,
            cluster_method=cluster_method,
        )

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
            structure=self._structure,
            neighbor_method=neighbor_method,
            cutoff=cutoff,
            n_clusters=n_clusters,
            q=q,
            averaged=averaged,
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
            structure=self._structure,
            mode=mode,
            ovito_compatibility=ovito_compatibility,
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
            structure=self._structure, num_neighbors=num_neighbors
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
            structure=self._structure,
            mode=mode,
            ovito_compatibility=ovito_compatibility,
        )

    def pyscal_voronoi_volume(self):
        """Calculate the Voronoi volume of atoms"""
        return analyse_voronoi_volume(structure=self._structure)

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
            structure=self._structure,
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
        return get_voronoi_vertices(
            structure=self._structure,
            epsilon=epsilon,
            distance_threshold=distance_threshold,
            width_buffer=width_buffer,
        )

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
        return get_strain(
            structure=self._structure,
            ref_structure=ref_structure,
            num_neighbors=num_neighbors,
            only_bulk_type=only_bulk_type,
        )

    def get_voronoi_neighbors(self, width_buffer: float = 10) -> np.ndarray:
        """
        Get pairs of atom indices sharing the same Voronoi vertices/areas.

        Args:
            width_buffer (float): Width of the layer to be added to account for pbc.

        Returns:
            pairs (ndarray): Pair indices
        """
        return get_voronoi_neighbors(
            structure=self._structure, width_buffer=width_buffer
        )

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
        return get_delaunay_neighbors(
            structure=self._structure, width_buffer=width_buffer
        )

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
        return get_cluster_positions(
            structure=self._structure,
            positions=positions,
            eps=eps,
            buffer_width=buffer_width,
            return_labels=return_labels,
        )
