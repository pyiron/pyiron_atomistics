# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import numpy as np
from pyiron_base import Settings
from scipy.spatial import cKDTree

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

s = Settings()


class Symmetry:
    """Class to analyse atom structure.  """
    def __init__(
        self, structure, use_magmoms=False, use_elements=True, symprec=1e-5, angle_tolerance=-1.0
    ):
        """
        Args:
            structure (:class:`pyiron.atomistics.structure.atoms.Atoms`): reference Atom structure.
        """
        self._structure = structure
        self.use_magmoms = use_magmoms
        self.use_elements = use_elements
        self.symprec = symprec
        self.angle_tolerance = angle_tolerance

    def generate_equivalent_points(
        self,
        points,
        use_magmoms=False,
        use_elements=True,
        symprec=1e-5,
        angle_tolerance=-1.0,
        return_unique=True,
        epsilon=1.0e-8
    ):
        """

        Args:
            points (list/ndarray): 3d vector
            use_magmoms (bool): cf. get_symmetry()
            use_elements (bool): cf. get_symmetry()
            symprec (float): cf. get_symmetry()
            angle_tolerance (float): cf. get_symmetry()
            return_unique (bool): Return only points which appear once.
            epsilon (float): displacement to add to avoid wrapping of atoms at borders

        Returns:
            (ndarray): array of equivalent points with respect to box symmetries
        """
        symmetry_operations = self._structure.get_symmetry(
            use_magmoms=use_magmoms,
            use_elements=use_elements,
            symprec=symprec,
            angle_tolerance=angle_tolerance
        )
        R = symmetry_operations['rotations']
        t = symmetry_operations['translations']
        x = np.einsum('jk,nj->nk', np.linalg.inv(self._structure.cell), np.atleast_2d(points))
        x = np.einsum('nxy,my->mnx', R, x)+t
        if any(self._structure.pbc):
            x[:,:,self._structure.pbc] -= np.floor(x[:,:,self._structure.pbc]+epsilon)
        if not return_unique:
            return np.einsum(
                'ji,mnj->mni', self._structure.cell, x
            ).reshape((len(R),)+np.asarray(points).shape)
        x = x.reshape(-1, 3)
        _, indices = np.unique(
            np.round(x, decimals=int(-np.log10(symprec))), return_index=True, axis=0
        )
        return np.einsum('ji,mj->mi', self._structure.cell, x[indices])

    def get_arg_equivalent_sites(
        self,
        points,
        use_magmoms=False,
        use_elements=True,
        symprec=1e-5,
        angle_tolerance=-1.0,
        epsilon=1.0e-8
    ):
        """
        Group points according to the box symmetries

        Args:
            points (list/ndarray): 3d vector
            use_magmoms (bool): cf. get_symmetry()
            use_elements (bool): cf. get_symmetry()
            symprec (float): cf. get_symmetry()
            angle_tolerance (float): cf. get_symmetry()
            epsilon (float): displacement to add to avoid wrapping of atoms at borders

        Returns:
            (ndarray): array of ID's according to their groups
        """
        all_points = self.get_equivalent_points(
            points=points,
            use_magmoms=use_magmoms,
            use_elements=use_elements,
            symprec=symprec,
            angle_tolerance=angle_tolerance,
            epsilon=epsilon,
            return_unique=False
        )
        decimals = int(-np.log10(symprec))
        _, inverse = np.unique(
            np.round(all_points.reshape(-1, 3), decimals=decimals), axis=0, return_inverse=True
        )
        inverse = inverse.reshape(all_points.shape[:-1][::-1])
        indices = np.min(inverse, axis=1)
        return np.unique(indices, return_inverse=True)[1]

    def _get_symmetry(
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


        """
        lattice = np.array(self.get_cell().T, dtype="double", order="C")
        positions = np.array(
            self.get_scaled_positions(wrap=False), dtype="double", order="C"
        )
        if use_elements:
            numbers = np.array(self.get_atomic_numbers(), dtype="intc")
        else:
            numbers = np.ones_like(self.get_atomic_numbers(), dtype="intc")
        if use_magmoms:
            magmoms = self.get_initial_magnetic_moments()
            return spglib.get_symmetry(
                cell=(lattice, positions, numbers, magmoms),
                symprec=symprec,
                angle_tolerance=angle_tolerance,
            )
        else:
            return spglib.get_symmetry(
                cell=(lattice, positions, numbers),
                symprec=symprec,
                angle_tolerance=angle_tolerance,
            )

    def symmetrize_vectors(
        self,
        vectors,
        use_magmoms=False,
        use_elements=True,
        symprec=1e-5,
        angle_tolerance=-1.0
    ):
        """
        Symmetrization of natom x 3 vectors according to box symmetries

        Args:
            vectors (ndarray/list): natom x 3 array to symmetrize
            force_update (bool): whether to update the symmetry info
            use_magmoms (bool): cf. get_symmetry
            use_elements (bool): cf. get_symmetry
            symprec (float): cf. get_symmetry
            angle_tolerance (float): cf. get_symmetry

        Returns:
            (np.ndarray) symmetrized vectors
        """
        vectors = np.array(vectors).reshape(-1, 3)
        if vectors.shape != self.positions.shape:
            raise ValueError('Vector must be a natom x 3 array: {} != {}'.format(
                vectors.shape, self.positions.shape
            ))
        symmetry = self.get_symmetry(
            use_magmoms=use_magmoms,
            use_elements=use_elements,
            symprec=symprec,
            angle_tolerance=angle_tolerance
        )
        scaled_positions = self.get_scaled_positions(wrap=False)
        tree = cKDTree(scaled_positions)
        positions = np.einsum(
            'nij,kj->nki',
            symmetry['rotations'],
            scaled_positions
        )+symmetry['translations'][:,None,:]
        positions -= np.floor(positions+symprec)
        indices = tree.query(positions)[1].argsort(axis=-1)
        return np.einsum(
            'ijk,ink->nj',
            symmetry['rotations'],
            vectors[indices]
        )/len(symmetry['rotations'])

    def get_symmetry(
        self, use_magmoms=None, use_elements=None, symprec=None, angle_tolerance=None
    ):
        """

        Args:
            use_magmoms (bool): Whether to consider magnetic moments (cf.
            get_initial_magnetic_moments())
            use_elements (bool): If False, chemical elements will be ignored
            symprec (float): Symmetry search precision
            angle_tolerance (float): Angle search tolerance

        Returns:


        """
        lattice = np.array(self.get_cell().T, dtype="double", order="C")
        positions = np.array(
            self.get_scaled_positions(wrap=False), dtype="double", order="C"
        )
        if use_elements:
            numbers = np.array(self.get_atomic_numbers(), dtype="intc")
        else:
            numbers = np.ones_like(self.get_atomic_numbers(), dtype="intc")
        if use_magmoms:
            magmoms = self.get_initial_magnetic_moments()
            return spglib.get_symmetry(
                cell=(lattice, positions, numbers, magmoms),
                symprec=symprec,
                angle_tolerance=angle_tolerance,
            )
        else:
            return spglib.get_symmetry(
                cell=(lattice, positions, numbers),
                symprec=symprec,
                angle_tolerance=angle_tolerance,
            )

