# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import numpy as np
from structuretoolkit.visualize import plot3d

__author__ = "Joerg Neugebauer, Sudarsan Surendralal"
__copyright__ = (
    "Copyright 2021, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "1.0"
__maintainer__ = "Sudarsan Surendralal"
__email__ = "surendralal@mpie.de"
__status__ = "production"
__date__ = "Sep 1, 2017"


class Visualize:
    def __init__(self, atoms):
        self._ref_atoms = atoms

    def plot3d(
        self,
        mode="NGLview",
        show_cell=True,
        show_axes=True,
        camera="orthographic",
        spacefill=True,
        particle_size=1.0,
        select_atoms=None,
        background="white",
        color_scheme=None,
        colors=None,
        scalar_field=None,
        scalar_start=None,
        scalar_end=None,
        scalar_cmap=None,
        vector_field=None,
        vector_color=None,
        magnetic_moments=False,
        view_plane=np.array([0, 0, 1]),
        distance_from_camera=1.0,
        opacity=1.0,
    ):
        """
        Plot3d relies on NGLView or plotly to visualize atomic structures. Here, we construct a string in the "protein database"

        The final widget is returned. If it is assigned to a variable, the visualization is suppressed until that
        variable is evaluated, and in the meantime more NGL operations can be applied to it to modify the visualization.

        Args:
            mode (str): `NGLView`, `plotly` or `ase`
            show_cell (bool): Whether or not to show the frame. (Default is True.)
            show_axes (bool): Whether or not to show xyz axes. (Default is True.)
            camera (str): 'perspective' or 'orthographic'. (Default is 'perspective'.)
            spacefill (bool): Whether to use a space-filling or ball-and-stick representation. (Default is True, use
                space-filling atoms.)
            particle_size (float): Size of the particles. (Default is 1.)
            select_atoms (numpy.ndarray): Indices of atoms to show, either as integers or a boolean array mask.
                (Default is None, show all atoms.)
            background (str): Background color. (Default is 'white'.)
            color_scheme (str): NGLView color scheme to use. (Default is None, color by element.)
            colors (numpy.ndarray): A per-atom array of HTML color names or hex color codes to use for atomic colors.
                (Default is None, use coloring scheme.)
            scalar_field (numpy.ndarray): Color each atom according to the array value (Default is None, use coloring
                scheme.)
            scalar_start (float): The scalar value to be mapped onto the low end of the color map (lower values are
                clipped). (Default is None, use the minimum value in `scalar_field`.)
            scalar_end (float): The scalar value to be mapped onto the high end of the color map (higher values are
                clipped). (Default is None, use the maximum value in `scalar_field`.)
            scalar_cmap (matplotlib.cm): The colormap to use. (Default is None, giving a blue-red divergent map.)
            vector_field (numpy.ndarray): Add vectors (3 values) originating at each atom. (Default is None, no
                vectors.)
            vector_color (numpy.ndarray): Colors for the vectors (only available with vector_field). (Default is None,
                vectors are colored by their direction.)
            magnetic_moments (bool): Plot magnetic moments as 'scalar_field' or 'vector_field'.
            view_plane (numpy.ndarray): A Nx3-array (N = 1,2,3); the first 3d-component of the array specifies
                which plane of the system to view (for example, [1, 0, 0], [1, 1, 0] or the [1, 1, 1] planes), the
                second 3d-component (if specified, otherwise [1, 0, 0]) gives the horizontal direction, and the third
                component (if specified) is the vertical component, which is ignored and calculated internally. The
                orthonormality of the orientation is internally ensured, and therefore is not required in the function
                call. (Default is np.array([0, 0, 1]), which is view normal to the x-y plane.)
            distance_from_camera (float): Distance of the camera from the structure. Higher = farther away.
                (Default is 14, which also seems to be the NGLView default value.)

            Possible NGLView color schemes:
              " ", "picking", "random", "uniform", "atomindex", "residueindex",
              "chainindex", "modelindex", "sstruc", "element", "resname", "bfactor",
              "hydrophobicity", "value", "volume", "occupancy"

        Returns:
            (nglview.NGLWidget): The NGLView widget itself, which can be operated on further or viewed as-is.

        Warnings:
            * Many features only work with space-filling atoms (e.g. coloring by a scalar field).
            * The colour interpretation of some hex codes is weird, e.g. 'green'.
        """
        return plot3d(
            structure=self._ref_atoms,
            mode=mode,
            show_cell=show_cell,
            show_axes=show_axes,
            camera=camera,
            spacefill=spacefill,
            particle_size=particle_size,
            select_atoms=select_atoms,
            background=background,
            color_scheme=color_scheme,
            colors=colors,
            scalar_field=scalar_field,
            scalar_start=scalar_start,
            scalar_end=scalar_end,
            scalar_cmap=scalar_cmap,
            vector_field=vector_field,
            vector_color=vector_color,
            magnetic_moments=magnetic_moments,
            view_plane=view_plane,
            distance_from_camera=distance_from_camera,
            opacity=opacity,
        )
