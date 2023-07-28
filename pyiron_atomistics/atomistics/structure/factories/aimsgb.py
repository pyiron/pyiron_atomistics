# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from pyiron.atomistics.structure.atoms  import pyiron_to_pymatgen, pymatgen_to_pyiron
from pymatgen.io.ase import AseAtomsAdaptor
from aimsgb import GBInformation, GrainBoundary, Grain


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
    def info(axis,
             max_sigma=15,
             specific=False,
             table_view=False
            ):
        """
        A dictionary with information including possible sigma, CSL matrix, GB plane,
        rotation angle and rotation matrix. 
        Args:
            axis ([u, v, w]): Rotation axis.
            max_sigma (int): The largest sigma value. Dafult value is 15.
            specific (bool): Whether collecting information for a specific sigma.
                Dafult value is False.
            table_view (bool): Display output in a tabuler format. Default value is True.

        Returns:
            A aimsgb.GBInformation object or display a table depending on 
            the value of table_view parameter.

        To construct the grain boundary select a GB plane and sigma value from 
        the table and pass it to the GBBuilder.build() function along with 
        the rotational axis and initial bulk structure.
        """
        if table_view:
            return print(GBInformation(axis, max_sigma, specific).__str__())
        else:
            return GBInformation(axis, max_sigma, specific)

    @staticmethod
    def build(axis,
              sigma,
              plane,
              initial_struct,
              uc_a=1,
              uc_b=1,
              vacuum=0.0,
              gap=0.0,
              delete_layer='0b0t0b0t',
              tol=0.25,
              to_primitive=False
              ):
        """
        A grain boundary (GB) object. The initial structure can be cubic or non-cubic
        crystal. If non-cubic, the crystal will be transferred to conventional cell.
        The generated GB could be either tilted or twisted based on the given GB
        plane. If the GB plane is parallel to the rotation axis, the generated GB
        will be a twisted one. Otherwise, tilted. Build grain boundary based on 
        rotation axis, sigma, GB plane, grain size, initial_struct and vacuum thickness.
        Build an interface structure by stacking two grains along a given direction.
        The grain_b a- and b-vectors will be forced to be the grain_a's a- and b-vectors.

        Args:
            axis ([u, v, w]): Rotation axis.
            sigma (int): The area ratio between the unit cell of CSL and the 
                given crystal lattice.
            plane ([h, k, l]): Miller index of GB plane. If the GB plane is parallel
                to the rotation axis, the generated GB will be a twist GB. If they
                are perpendicular, the generated GB will be a tilt GB.
            initial_struct (Grain): Initial input structure. Must be an object of 
                pyiron_atomistics.atomistics.structure.atoms.Atoms
            uc_a (int): Number of unit cell of grain A. Default to 1.
            uc_b (int): Number of unit cell of grain B. Default to 1.
            vacuum (float): Vacuum space between the surface of the grains in Angstroms. Default to 0.0
            gap (float): Gap between the GB interface of the grains in Angstroms. Default to 0.0
            delete_layer (str): Delete top and bottom layers of the first and the second grains.
                8 characters in total. The first 4 characters is for the first grain and
                the other 4 is for the second grain. "b" means bottom layer and "t" means
                top layer. Integer represents the number of layers to be deleted.
                Default to "0b0t0b0t", which means no deletion of layers. The
                direction of top and bottom layers is based on the given direction.
            tol (float): Tolerance factor in Angstrom to determnine if sites are 
                in the same layer. Default to 0.25.
            to_primitive (bool): Whether to get primitive structure of GB. Default to False.
            

        Returns:
            :class:`.Atoms`: final grain boundary structure
        """
        basis_pmg = AseAtomsAdaptor.get_structure(initial_struct)
        
        basis_pmg = Grain(lattice=basis_pmg.lattice,
                          species=basis_pmg.species,
                          coords=basis_pmg.frac_coords)
        
        gb = GrainBoundary(axis=axis,
                           sigma=sigma,
                           plane=plane,
                           initial_struct=basis_pmg,
                           uc_a=uc_a,
                           uc_b=uc_b)
        
        structure = Grain.stack_grains(grain_a=gb.grain_a,
                                       grain_b=gb.grain_b,
                                       vacuum=vacuum,
                                       gap=gap,
                                       direction=gb.direction,
                                       delete_layer=delete_layer,
                                       tol=tol,
                                       to_primitive=to_primitive)
        
        return pymatgen_to_pyiron(structure)

