# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import numpy as np
from pyiron_base import Settings
from scipy.spatial import cKDTree
import spglib
import ast

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

    """

    Return a class for operations related to box symmetries. Main attributes:

    - rotations: List of rotational matrices
    - translations: List of translational vectors

    All other functionalities depend on these two attributes.

    """

    def __init__(
        self, structure, use_magmoms=False, use_elements=True, symprec=1e-5, angle_tolerance=-1.0
    ):
        """
        Args:
            structure (:class:`pyiron.atomistics.structure.atoms.Atoms`): reference Atom structure.
            use_magmoms (bool): Whether to consider magnetic moments (cf.
            get_initial_magnetic_moments())
            use_elements (bool): If False, chemical elements will be ignored
            symprec (float): Symmetry search precision
            angle_tolerance (float): Angle search tolerance
        """
        self._structure = structure
        self._use_magmoms = use_magmoms
        self._use_elements = use_elements
        self._symprec = symprec
        self._angle_tolerance = angle_tolerance
        for k,v in self._get_symmetry(
            symprec=symprec,
            angle_tolerance=angle_tolerance
        ).items():
            self[k] = v

    @property
    def arg_equivalent_atoms(self):
        return self['equivalent_atoms']

    @property
    def rotations(self):
        """
        All rotational matrices. Two points x and y are equivalent with respect to the box
        box symmetry, if there is a rotational matrix R and a translational vector t which
        satisfy:

            x = R@y + t
        """
        return self['rotations']

    @property
    def translations(self):
        """
        All translational vectors. Two points x and y are equivalent with respect to the box
        box symmetry, if there is a rotational matrix R and a translational vector t which
        satisfy:

            x = R@y + t
        """
        return self['translations']

    def generate_equivalent_points(
        self,
        points,
        return_unique=True,
        decimals=5,
        epsilon=1.0e-8,
    ):
        """

        Args:
            points (list/ndarray): 3d vector
            return_unique (bool): Return only points which appear once.
            decimals (int): Number of decimal places to round to for the uniqueness of positions
                (Not relevant if return_unique=False)
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
            ).reshape((len(R),)+np.shape(points))
        x = x.reshape(-1, 3)
        _, indices = np.unique(
            np.round(x, decimals=decimals), return_index=True, axis=0
        )
        return np.einsum('ji,mj->mi', self._structure.cell, x[indices])

    def get_arg_equivalent_sites(
        self,
        points,
        decimals=5,
        epsilon=1.0e-8
    ):
        """
        Group points according to the box symmetries

        Args:
            points (list/ndarray): 3d vector
            decimals (int): Number of decimal places to round to for the uniqueness of positions
                (Not relevant if return_unique=False)
            epsilon (float): displacement to add to avoid wrapping of atoms at borders

        Returns:
            (ndarray): array of ID's according to their groups
        """
        if len(np.shape(points))!=2:
            raise ValueError('points must be a (n, 3)-array')
        all_points = self.generate_equivalent_points(
            points=points,
            epsilon=epsilon,
            return_unique=False
        )
        _, inverse = np.unique(
            np.round(all_points.reshape(-1, 3), decimals=decimals), axis=0, return_inverse=True
        )
        inverse = inverse.reshape(all_points.shape[:-1][::-1])
        indices = np.min(inverse, axis=1)
        return np.unique(indices, return_inverse=True)[1]

    def symmetrize_vectors(
        self,
        vectors,
        epsilon=1.0e-8,
    ):
        """
        Symmetrization of natom x 3 vectors according to box symmetries

        Args:
            vectors (ndarray/list): natom x 3 array to symmetrize
            epsilon (float): displacement to add to avoid wrapping of atoms at borders

        Returns:
            (np.ndarray) symmetrized vectors
        """
        vectors = np.array(vectors).reshape(-1, 3)
        if vectors.shape != self._structure.positions.shape:
            raise ValueError('Vector must be a natom x 3 array: {} != {}'.format(
                vectors.shape, self.positions.shape
            ))
        scaled_positions = self._structure.get_scaled_positions(wrap=False)
        tree = cKDTree(scaled_positions)
        positions = np.einsum(
            'nij,kj->nki',
            self['rotations'],
            scaled_positions
        )+self['translations'][:,None,:]
        positions -= np.floor(positions+epsilon)
        indices = tree.query(positions)[1].argsort(axis=-1)
        return np.einsum(
            'ijk,ink->nj',
            self['rotations'],
            vectors[indices]
        )/len(self['rotations'])

    def _get_spglib_cell(self, use_elements=None, use_magmoms=None):
        lattice = np.array(self._structure.get_cell(), dtype="double", order="C")
        positions = np.array(
            self._structure.get_scaled_positions(wrap=False), dtype="double", order="C"
        )
        if use_elements is None:
            use_elements = self._use_elements
        if use_magmoms is None:
            use_magmoms = self._use_magmoms
        if use_elements:
            numbers = np.array(self._structure.get_atomic_numbers(), dtype="intc")
        else:
            numbers = np.ones_like(self._structure.get_atomic_numbers(), dtype="intc")
        if use_magmoms:
            return lattice, positions, numbers, self._structure.get_initial_magnetic_moments()
        return lattice, positions, numbers

    def _get_symmetry(
        self, symprec=1e-5, angle_tolerance=-1.0
    ):
        """

        Args:
            symprec (float): Symmetry search precision
            angle_tolerance (float): Angle search tolerance

        Returns:


        """
        return spglib.get_symmetry(
            cell=self._get_spglib_cell(),
            symprec=symprec,
            angle_tolerance=angle_tolerance,
        )

    @property
    def info(self):
        """
        Get symmetry info

        https://atztogo.github.io/spglib/python-spglib.html
        """
        return spglib.get_symmetry_dataset(
            cell=self._get_spglib_cell(use_magmoms=False),
            symprec=self._symprec,
            angle_tolerance=self._angle_tolerance,
        )

    @property
    def spacegroup(self):
        """

        Args:
            symprec:
            angle_tolerance:

        Returns:

        https://atztogo.github.io/spglib/python-spglib.html
        """
        space_group = spglib.get_spacegroup(
            cell=self._get_spglib_cell(use_magmoms=False),
            symprec=self._symprec,
            angle_tolerance=self._angle_tolerance,
        ).split()
        if len(space_group) == 1:
            return {"Number": ast.literal_eval(space_group[0])}
        return {
            "InternationalTableSymbol": space_group[0],
            "Number": ast.literal_eval(space_group[1]),
        }

    def refine_cell(self):
        cell, coords, _ = spglib.refine_cell(
            cell=self._get_spglib_cell(use_magmoms=False),
            symprec=self._symprec,
            angle_tolerance=self._angle_tolerance,
        )
        new_structure = self._structure.copy()
        new_structure.positions = coords
        new_structure.cell = cell
        return new_structure

    @property
    def primitive_cell(self):
        el_dict = {}
        for el in set(self._structure.get_chemical_elements()):
            el_dict[el.AtomicNumber] = el
        cell, coords, atomic_numbers = spglib.find_primitive(
            cell=self._get_spglib_cell(use_magmoms=False),
            symprec=self._symprec,
            angle_tolerance=self._angle_tolerance,
        )
        # print atomic_numbers, type(atomic_numbers)
        el_lst = [el_dict[i_a] for i_a in atomic_numbers]

        # convert lattice vectors to standard (experimental feature!) TODO:
        red_structure = self._structure.copy()
        red_structure = red_structure[:len(coords)]
        red_structure.set_scaled_positions(coords)
        red_structure.set_species(el_lst)
        red_structure.cell = cell
        space_group = red_structure.get_symmetry(symprec=self._symprec).spacegroup["Number"]
        # print "space group: ", space_group
        if space_group == 225:  # fcc
            alat = np.max(cell[0])
            amat_fcc = alat * np.array([[1, 0, 1], [1, 1, 0], [0, 1, 1]])
            red_structure.cell = amat_fcc
        return red_structure

    def get_ir_reciprocal_mesh(
        self,
        mesh,
        is_shift=np.zeros(3, dtype="intc"),
        is_time_reversal=True,
    ):
        return spglib.get_ir_reciprocal_mesh(
            mesh=mesh,
            cell=self._get_spglib_cell(),
            is_shift=is_shift,
            is_time_reversal=is_time_reversal,
            symprec=self._symprec,
        )

