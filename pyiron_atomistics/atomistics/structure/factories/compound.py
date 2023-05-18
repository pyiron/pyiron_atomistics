# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from structuretoolkit.build import B2, C14, C15, C36, D03
from pyiron_atomistics.atomistics.structure.atoms import ase_to_pyiron

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
        return ase_to_pyiron(B2(element_a=element_a, element_b=element_b, a=a))

    @staticmethod
    def C14(element_a, element_b, a=None, c_over_a=1.626, x1=0.1697, z1=0.5629):
        """
        Builds a hexagonal $A B_2$ C14 Laves phase cell.

        Default fractional coordinates are chosen to reproduce CaMg2 Laves phase from the Springer Materials Database

        https://materials.springer.com/isp/crystallographic/docs/sd_1822295

        .. attention:: Change in Stochiometry possible!

            If any of the fractional coordinates fall onto their high symmetry values the atoms may be placed on another
            Wyckoff position, with less sites and therefor the cell composition may change.

            For `x1` avoid 0, 1/3, 2/3, for `z1` avoid 1/4, 3/4.

        Args:
            element_a (str, ase.Atom): specificies A
            element_b (str, ase.Atom): specificies B
            a (float): length of a & b cell vectors
            c_over_a (float): c/a ratio
            x1 (float): fractional x coordinate of B atoms on Wyckoff 6h
            z1 (float): fractional z coordinate of A atoms on Wyckoff 4f
        """
        return ase_to_pyiron(
            C14(
                element_a=element_a,
                element_b=element_b,
                a=a,
                c_over_a=c_over_a,
                x1=x1,
                z1=z1,
            )
        )

    @staticmethod
    def C15(element_a, element_b, a=None):
        """
        Builds a cubic $A B_2$ C15 Laves phase cell.

        Example use:

        >>> structure = CompoundFactory().C15('Al', 'Ca')
        >>> structure.repeat(2).plot3d(view_plane=([1, 1, 0], [0, 0, -1])) # doctest: +SKIP
        NGLWidget()

        Args:
            element_a (str): The chemical symbol for the A element.
            element_b (str): The chemical symbol for the B element.
            a (float): The lattice constant. (Default is None, which uses the default nearest-neighbour distance for
                the A-type element.)

        Returns:
            (Atoms): The C15 unit cell.
        """
        return ase_to_pyiron(C15(element_a=element_a, element_b=element_b, a=a))

    @staticmethod
    def C36(
        element_a,
        element_b,
        a=None,
        c_over_a=3.252,
        x1=0.16429,
        z1=0.09400,
        z2=0.65583,
        z3=0.12514,
    ):
        """
        Create hexagonal $A B_2$ C36 Laves phase.

        Fractional coordinates are chosen to reproduce MgNi2 Laves phase from the Springer Materials Database

        https://materials.springer.com/isp/crystallographic/docs/sd_0260824

        .. attention:: Change in Stochiometry possible!

            If any of the fractional coordinates fall onto their high symmetry values the atoms may be placed on another
            Wyckoff position, with less sites and therefor the cell composition may change.

            For `x1` avoid 0, 1/3, 2/3, for `z1` avoid 0, 1/4, 1/2, 3/4, for `z1`/`z2` avoid 1/4, 3/4.

        Args:
            element_a (str, ase.Atom): specificies A
            element_b (str, ase.Atom): specificies B
            a (float): length of a & b cell vectors
            c_over_a (float): c/a ratio
            x1 (float): fractional x coordinate of B atoms on Wyckoff 6h
            z1 (float): fractional z coordinate of A atoms on Wyckoff 4e
            z2 (float): fractional z coordinate of A atoms on Wyckoff 4f
            z3 (float): fractional z coordinate of B atoms on Wyckoff 4f
        """
        return ase_to_pyiron(
            C36(
                element_a=element_a,
                element_b=element_b,
                a=a,
                c_over_a=c_over_a,
                x1=x1,
                z1=z1,
                z2=z2,
                z3=z3,
            )
        )

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
        return ase_to_pyiron(D03(element_a=element_a, element_b=element_b, a=a))
