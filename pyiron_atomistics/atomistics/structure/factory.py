# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from ase.build import (
        add_adsorbate,
        add_vacuum,
        bcc100,
        bcc110,
        bcc111,
        diamond100,
        diamond111,
        fcc100,
        fcc110,
        fcc111,
        fcc211,
        hcp0001,
        hcp10m10,
        mx2,
        hcp0001_root,
        fcc111_root,
        bcc111_root,
        root_surface,
        root_surface_analysis,
        surface as ase_surf
)
import numpy as np
from pyiron_atomistics.atomistics.structure.factories.ase import AseFactory
from pyiron_atomistics.atomistics.structure.factories.aimsgb import AimsgbFactory
from pyiron_atomistics.atomistics.structure.pyironase import publication as publication_ase
from pyiron_atomistics.atomistics.structure.atoms import CrystalStructure, Atoms, \
    ase_to_pyiron, pymatgen_to_pyiron, ovito_to_pyiron
from pyiron_atomistics.atomistics.structure.periodic_table import PeriodicTable
from pyiron_base import Settings, PyironFactory, deprecate
import types
from functools import wraps

__author__ = "Sudarsan Surendralal"
__copyright__ = (
    "Copyright 2021, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "1.0"
__maintainer__ = "Sudarsan Surendralal"
__email__ = "surendralal@mpie.de"
__status__ = "production"
__date__ = "May 1, 2020"

s = Settings()


class StructureFactory(PyironFactory):
    def __init__(self):
        self._ase = AseFactory()
        self._aimsgb = AimsgbFactory()

    @property
    def ase(self):
        return self._ase

    @property
    def aimsgb(self):
        return self._aimsgb

    def cut(self, *args, **kwargs):
        return self.ase.cut(*args, **kwargs)
    cut.__doc__ = AseFactory.cut.__doc__

    def stack(self, *args, **kwargs):
        return self.ase.stack(*args, **kwargs)
    stack.__doc__ = AseFactory.stack.__doc__

    def read(self, *args, **kwargs):
        return self.ase.read(*args, **kwargs)
    read.__doc__ = AseFactory.read.__doc__

    @deprecate(message="Please use .read or .ase.read", version="0.2.2")
    def ase_read(self, *args, **kwargs):
        return self.ase.read(*args, **kwargs)
    ase_read.__doc__ = AseFactory.read.__doc__

    @deprecate(message="Please use .bulk or .ase.bulk", version="0.2.2")
    def ase_bulk(self, *args, **kwargs):
        return self.ase.bulk(*args, **kwargs)
    ase_bulk.__doc__ = AseFactory.bulk.__doc__

    def bulk(
            self,
            name,
            crystalstructure=None,
            a=None,
            c=None,
            covera=None,
            u=None,
            orthorhombic=False,
            cubic=False,
    ):
        """
        Creating bulk systems (using ASE bulk module). Crystal structure and lattice constant(s) will be guessed if not
        provided.

        name (str): Chemical symbol or symbols as in 'MgO' or 'NaCl'.
        crystalstructure (str): Must be one of sc, fcc, bcc, hcp, diamond, zincblende,
                                rocksalt, cesiumchloride, fluorite or wurtzite.
        a (float): Lattice constant.
        c (float): Lattice constant.
        c_over_a (float): c/a ratio used for hcp.  Default is ideal ratio: sqrt(8/3).
        u (float): Internal coordinate for Wurtzite structure.
        orthorhombic (bool): Construct orthorhombic unit cell instead of primitive cell which is the default.
        cubic (bool): Construct cubic unit cell if possible.

        Returns:
            pyiron.atomistics.structure.atoms.Atoms: Required bulk structure
        """
        return self.ase.bulk(
            name=name,
            crystalstructure=crystalstructure,
            a=a,
            c=c,
            covera=covera,
            u=u,
            orthorhombic=orthorhombic,
            cubic=cubic,
        )

    @staticmethod
    def surface(
        element, surface_type, size=(1, 1, 1), vacuum=1.0, center=False, pbc=True, **kwargs
    ):
        """
        Generate a surface based on the ase.build.surface module.

        Args:
            element (str): Element name
            surface_type (str): The string specifying the surface type generators available through ase (fcc111,
            hcp0001 etc.)
            size (tuple): Size of the surface
            vacuum (float): Length of vacuum layer added to the surface along the z direction
            center (bool): Tells if the surface layers have to be at the center or at one end along the z-direction
            pbc (list/numpy.ndarray): List of booleans specifying the periodic boundary conditions along all three
                                      directions.
            **kwargs: Additional, arguments you would normally pass to the structure generator like 'a', 'b',
            'orthogonal' etc.

        Returns:
            pyiron_atomistics.atomistics.structure.atoms.Atoms instance: Required surface

        """
        # https://gitlab.com/ase/ase/blob/master/ase/lattice/surface.py
        if pbc is None:
            pbc = True
        s.publication_add(publication_ase())
        for surface_class in [
            add_adsorbate,
            add_vacuum,
            bcc100,
            bcc110,
            bcc111,
            diamond100,
            diamond111,
            fcc100,
            fcc110,
            fcc111,
            fcc211,
            hcp0001,
            hcp10m10,
            mx2,
            hcp0001_root,
            fcc111_root,
            bcc111_root,
            root_surface,
            root_surface_analysis,
            ase_surf,
        ]:
            if surface_type == surface_class.__name__:
                surface_type = surface_class
                break
        if isinstance(surface_type, types.FunctionType):
            if center:
                surface = surface_type(
                    symbol=element, size=size, vacuum=vacuum, **kwargs
                )
            else:
                surface = surface_type(symbol=element, size=size, **kwargs)
                z_max = np.max(surface.positions[:, 2])
                surface.cell[2, 2] = z_max + vacuum
            surface.pbc = pbc
            return ase_to_pyiron(surface)
        else:
            raise ValueError(f"Surface type {surface_type} not recognized.")

    @staticmethod
    def surface_hkl(lattice, hkl, layers, vacuum=1.0, center=False, pbc=True):
        """
        Use ase.build.surface to build a surface with surface normal (hkl).

        Args:
            lattice (pyiron_atomistics.atomistics.structure.atoms.Atoms/str): bulk Atoms
                instance or str, e.g. "Fe", from which to build the surface
            hkl (list): miller indices of surface to be created
            layers (int): # of atomic layers in the surface
            vacuum (float): vacuum spacing
            center (bool): shift all positions to center the surface
                in the cell

        Returns:
            pyiron_atomistics.atomistics.structure.atoms.Atoms instance: Required surface
        """
        # https://gitlab.com/ase/ase/blob/master/ase/lattice/surface.py
        s.publication_add(publication_ase())

        surface = ase_surf(lattice, hkl, layers)
        z_max = np.max(surface.positions[:, 2])
        surface.cell[2, 2] = z_max + vacuum
        if center:
            surface.positions += 0.5 * surface.cell[2] - [0, 0, z_max/2]
        surface.pbc = pbc
        return ase_to_pyiron(surface)

    @staticmethod
    def crystal(element, bravais_basis, lattice_constant):
        """
        Create a crystal structure using pyiron's native crystal structure generator

        Args:
            element (str): Element name
            bravais_basis (str): Basis type
            lattice_constant (float/list): Lattice constants

        Returns:
            pyiron.atomistics.structure.atoms.Atoms: The required crystal structure
        """
        return CrystalStructure(
            element=element,
            bravais_basis=bravais_basis,
            lattice_constants=[lattice_constant],
        )

    @staticmethod
    def atoms(
            symbols=None,
            positions=None,
            numbers=None,
            tags=None,
            momenta=None,
            masses=None,
            magmoms=None,
            charges=None,
            scaled_positions=None,
            cell=None,
            pbc=None,
            celldisp=None,
            constraint=None,
            calculator=None,
            info=None,
            indices=None,
            elements=None,
            dimension=None,
            species=None,
            **qwargs
    ):
        """
        Creates a atomistics.structure.atoms.Atoms instance.

        Args:
            elements (list/numpy.ndarray): List of strings containing the elements or a list of
                                atomistics.structure.periodic_table.ChemicalElement instances
            numbers (list/numpy.ndarray): List of atomic numbers of elements
            symbols (list/numpy.ndarray): List of chemical symbols
            positions (list/numpy.ndarray): List of positions
            scaled_positions (list/numpy.ndarray): List of scaled positions (relative coordinates)
            pbc (boolean): Tells if periodic boundary conditions should be applied
            cell (list/numpy.ndarray): A 3x3 array representing the lattice vectors of the structure
            momenta (list/numpy.ndarray): List of momentum values
            tags (list/numpy.ndarray): A list of tags
            masses (list/numpy.ndarray): A list of masses
            magmoms (list/numpy.ndarray): A list of magnetic moments
            charges (list/numpy.ndarray): A list of point charges
            celldisp:
            constraint (list/numpy.ndarray): A list of constraints
            calculator: ASE calculator
            info (list/str): ASE compatibility
            indices (list/numpy.ndarray): The list of species indices
            dimension (int): Dimension of the structure
            species (list): List of species

        Returns:
            pyiron.atomistics.structure.atoms.Atoms: The required structure instance
        """
        if pbc is None:
            pbc = True
        return Atoms(
            symbols=symbols,
            positions=positions,
            numbers=numbers,
            tags=tags,
            momenta=momenta,
            masses=masses,
            magmoms=magmoms,
            charges=charges,
            scaled_positions=scaled_positions,
            cell=cell,
            pbc=pbc,
            celldisp=celldisp,
            constraint=constraint,
            calculator=calculator,
            info=info,
            indices=indices,
            elements=elements,
            dimension=dimension,
            species=species,
            **qwargs
        )

    @staticmethod
    def element(parent_element, new_element_name=None, spin=None, potential_file=None):
        """

        Args:
            parent_element (str, int): The parent element eq. "N", "O", "Mg" etc.
            new_element_name (str): The name of the new parent element (can be arbitrary)
            spin (float): Value of the magnetic moment (with sign)
            potential_file (str): Location of the new potential file if necessary

        Returns:
            atomistics.structure.periodic_table.ChemicalElement instance
        """
        periodic_table = PeriodicTable()
        if new_element_name is None:
            if spin is not None:
                new_element_name = (
                        parent_element + "_spin_" + str(spin).replace(".", "_")
                )
            else:
                new_element_name = parent_element + "_1"
        if potential_file is not None:
            if spin is not None:
                periodic_table.add_element(
                    parent_element=parent_element,
                    new_element=new_element_name,
                    spin=str(spin),
                    pseudo_potcar_file=potential_file,
                )
            else:
                periodic_table.add_element(
                    parent_element=parent_element,
                    new_element=new_element_name,
                    pseudo_potcar_file=potential_file,
                )
        elif spin is not None:
            periodic_table.add_element(
                parent_element=parent_element,
                new_element=new_element_name,
                spin=str(spin),
            )
        else:
            periodic_table.add_element(
                parent_element=parent_element, new_element=new_element_name
            )
        return periodic_table.element(new_element_name)

    @deprecate(message="Use .aimsgb.info", version="0.2.2")
    def aimsgb_info(self, axis, max_sigma):
        return self.aimsgb.info(axis=axis, max_sigma=max_sigma)
    aimsgb_info.__doc__ = AimsgbFactory.info.__doc__

    @deprecate(message="Use .aimsgb.build", version="0.2.2")
    def aimsgb_build(
            self,
            axis,
            sigma,
            plane,
            initial_struct,
            to_primitive=False,
            delete_layer='0b0t0b0t',
            add_if_dist=0.0
    ):
        return self.aimsgb.build(
            axis=axis,
            sigma=sigma,
            plane=plane,
            initial_struct=initial_struct,
            to_primitive=to_primitive,
            delete_layer=delete_layer,
            add_if_dist=add_if_dist
        )
    aimsgb_build.__doc__ = AimsgbFactory.build.__doc__

    @staticmethod
    @wraps(ase_to_pyiron)
    def from_ase(ase_atoms):
        return ase_to_pyiron(ase_atoms)

    @staticmethod
    @wraps(pymatgen_to_pyiron)
    def from_pymatgen(pymatgen_obj):
        return pymatgen_to_pyiron(pymatgen_obj)

    @staticmethod
    @wraps(ovito_to_pyiron)
    def from_ovito(ovito_obj):
        return ovito_to_pyiron(ovito_obj)
