# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from structuretoolkit.build import grainboundary, get_grainboundary_info
from pyiron_atomistics.atomistics.structure.atoms import ase_to_pyiron

__author__ = "Ujjal Saikia"
__copyright__ = (
    "Copyright 2021, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "1.0"
__maintainer__ = "Liam Huber"
__email__ = "huber@mpie.de"
__status__ = "production"
__date__ = "Feb 26, 2021"


class AimsgbFactory:
    @staticmethod
    def info(axis, max_sigma):
        """
        Provides a list of possible GB structures for a given rotational axis and upto the given maximum sigma value.

        Args:
            axis : Rotational axis for the GB you want to construct (for example, axis=[1,0,0])
            max_sigma (int) : The maximum value of sigma upto which you want to consider for your
            GB (for example, sigma=5)

        Returns:
            A list of possible GB structures in the format:

            {sigma value: {'theta': [theta value],
                'plane': the GB planes")
                'rot_matrix': array([the rotational matrix]),
                'csl': [array([the csl matrix])]}}

        To construct the grain boundary select a GB plane and sigma value from the list and pass it to the
        GBBuilder.gb_build() function along with the rotational axis and initial bulk structure.
        """
        return get_grainboundary_info(axis=axis, max_sigma=max_sigma)

    @staticmethod
    def build(
        axis,
        sigma,
        plane,
        initial_struct,
        to_primitive=False,
        delete_layer="0b0t0b0t",
        add_if_dist=0.0,
        uc_a=1,
        uc_b=1,
    ):
        """
        Generate a grain boundary structure based on the aimsgb.GrainBoundary module.

        Args:
            axis : Rotational axis for the GB you want to construct (for example, axis=[1,0,0])
            sigma (int) : The sigma value of the GB you want to construct (for example, sigma=5)
            plane: The grain boundary plane of the GB you want to construct (for example, plane=[2,1,0])
            initial_struct : Initial bulk structure from which you want to construct the GB (a pyiron
                            structure object).
            delete_layer : To delete layers of the GB. For example, delete_layer='1b0t1b0t'. The first
                           4 characters is for first grain and the other 4 is for second grain. b means
                           bottom layer and t means top layer. Integer represents the number of layers
                           to be deleted. The first t and second b from the left hand side represents
                           the layers at the GB interface. Default value is delete_layer='0b0t0b0t', which
                           means no deletion of layers.
            add_if_dist : If you want to add extra interface distance, you can specify add_if_dist.
                           Default value is add_if_dist=0.0
            to_primitive : To generate primitive or non-primitive GB structure. Default value is
                            to_primitive=False
            uc_a (int): Number of unit cell of grain A. Default to 1.
            uc_b (int): Number of unit cell of grain B. Default to 1.

        Returns:
            :class:`.Atoms`: final grain boundary structure
        """
        return ase_to_pyiron(
            grainboundary(
                axis=axis,
                sigma=sigma,
                plane=plane,
                initial_struct=initial_struct,
                to_primitive=to_primitive,
                delete_layer=delete_layer,
                add_if_dist=add_if_dist,
                uc_a=uc_a,
                uc_b=uc_b,
            )
        )
