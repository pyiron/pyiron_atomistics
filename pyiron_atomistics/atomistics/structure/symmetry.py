# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import numpy as np
from pyiron_base import Settings
from scipy.spatial import cKDTree
import spglib

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


class Symmetry(dict):
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
        for k,v in self._get_symmetry(
            use_magmoms=use_magmoms,
            use_elements=use_elements,
            symprec=symprec,
            angle_tolerance=angle_tolerance
        ).items():
            self[k] = v

    def generate_equivalent_points(
        self,
        points,
        return_unique=True,
        epsilon=1.0e-8
    ):
        """

        Args:
            points (list/ndarray): 3d vector
            return_unique (bool): Return only points which appear once.
            epsilon (float): displacement to add to avoid wrapping of atoms at borders

        Returns:
            (ndarray): array of equivalent points with respect to box symmetries, with a shape of
                (n_symmetry, original_shape) if return_unique=False, otherwise (n, 3), where n is
                the number of inequivalent vectors.
        """
        R = self['rotations']
        t = self['translations']
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
        epsilon=1.0e-8
    ):
        """
        Group points according to the box symmetries

        Args:
            points (list/ndarray): 3d vector
            epsilon (float): displacement to add to avoid wrapping of atoms at borders

        Returns:
            (ndarray): array of ID's according to their groups
        """
        all_points = self.generate_equivalent_points(
            points=points,
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

    def symmetrize_vectors(
        self,
        vectors,
    ):
        """
        Symmetrization of natom x 3 vectors according to box symmetries

        Args:
            vectors (ndarray/list): natom x 3 array to symmetrize

        Returns:
            (np.ndarray) symmetrized vectors
        """
        vectors = np.array(vectors).reshape(-1, 3)
        if vectors.shape != self.positions.shape:
            raise ValueError('Vector must be a natom x 3 array: {} != {}'.format(
                vectors.shape, self.positions.shape
            ))
        scaled_positions = self.get_scaled_positions(wrap=False)
        tree = cKDTree(scaled_positions)
        positions = np.einsum(
            'nij,kj->nki',
            self['rotations'],
            scaled_positions
        )+self['translations'][:,None,:]
        positions -= np.floor(positions+symprec)
        indices = tree.query(positions)[1].argsort(axis=-1)
        return np.einsum(
            'ijk,ink->nj',
            self['rotations'],
            vectors[indices]
        )/len(self['rotations'])

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
        lattice = np.array(self._structure.get_cell().T, dtype="double", order="C")
        positions = np.array(
            self._structure.get_scaled_positions(wrap=False), dtype="double", order="C"
        )
        if use_elements:
            numbers = np.array(self._structure.get_atomic_numbers(), dtype="intc")
        else:
            numbers = np.ones_like(self._structure.get_atomic_numbers(), dtype="intc")
        if use_magmoms:
            magmoms = self._structure.get_initial_magnetic_moments()
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

