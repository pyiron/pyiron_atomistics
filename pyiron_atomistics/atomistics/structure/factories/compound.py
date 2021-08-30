# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from pyiron_atomistics.atomistics.structure.factories.ase import AseFactory
import numpy as np

__author__ = "Liam Huber"
__copyright__ = (
    "Copyright 2021, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "1.0.1"
__maintainer__ = "Liam Huber"
__email__ = "huber@mpie.de"
__status__ = "production"
__date__ = "Jun 28, 2021"

_ase = AseFactory()


def _bcc_lattice_constant_from_nn_distance(element):
    """
    Build a BCC lattice constant by making the BCC have the same nearest neighbour distance as the regular cell.

    Works because the NN distance doesn't even care what crystal structure the regular unit cell is.
    """
    return AseFactory().bulk(name=element).get_neighbors(num_neighbors=1).distances[0, 0] * (2 / np.sqrt(3))


class CompoundFactory:
    """A collection of routines for constructing Laves phases and other more complex structures."""

    @staticmethod
    def B2(element_a, element_b, a=None):
        """
        Builds a cubic $AB$ B2 structure of interpenetrating simple cubic lattices.

        Args:
            element_a (str): The chemical symbol for the A element.
            element_b (str): The chemical symbol for the B element.
            a (float): The lattice constant. (Default is None, which uses the default for element A.)

        Returns:
            (Atoms): The B2 unit cell.
        """
        a = _bcc_lattice_constant_from_nn_distance(element_a) if a is None else a

        return _ase.crystal((element_a, element_b), [(0, 0, 0), (1/2, 1/2, 1/2)], spacegroup=221, cell=(a, a, a))

    @staticmethod
    def C14():
        raise NotImplementedError

    @staticmethod
    def C15(element_a, element_b, a=None):
        """
        Builds a cubic $A B_2$ C15 Laves phase cell.

        Example use:

        >>> structure = CompoundFactory().C15('Al', 'Ca')
        >>> structure.repeat(2).plot3d(view_plane=([1, 1, 0], [0, 0, -1]))
        NGLWidget()

        Args:
            element_a (str): The chemical symbol for the A element.
            element_b (str): The chemical symbol for the B element.
            a (float): The lattice constant. (Default is None, which uses the default nearest-neighbour distance for
                the A-type element.)

        Returns:
            (Atoms): The C15 unit cell.
        """
        a = 2 * _bcc_lattice_constant_from_nn_distance(element_a) if a is None else a

        return _ase.crystal((element_a, element_b), [(0, 0, 0), (1/8, 5/8, 1/8)], spacegroup=227, cell=(a, a, a))

    @staticmethod
    def C36():
        raise NotImplementedError

    @staticmethod
    def D03(element_a, element_b, a=None):
        """
        Builds a cubic $A B_3$ D03 cubic cell.

        Args:
            element_a (str): The chemical symbol for the A element.
            element_b (str): The chemical symbol for the B element.
            a (float): The lattice constant. (Default is None, which uses the default nearest-neighbour distance for
                the A-type element.)

        Returns:
            (Atoms): The D03 unit cell.
        """
        a = 2 * _bcc_lattice_constant_from_nn_distance(element_a) if a is None else a

        return _ase.crystal(
            (element_b, element_a, element_b),
            [(0, 0, 0), (1 / 2, 1 / 2, 1 / 2), (1 / 4, 1 / 4, 1 / 4)],
            spacegroup=225,
            cell=(a, a, a)
        )
