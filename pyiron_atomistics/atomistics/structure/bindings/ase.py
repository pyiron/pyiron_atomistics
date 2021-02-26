# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from functools import wraps
from ase.build import cut as ase_cut, stack as ase_stack, bulk as ase_bulk
from ase.io import read as ase_read
from ase.spacegroup import crystal as ase_crystal
from pyiron_atomistics import ase_to_pyiron
from pyiron_atomistics.atomistics.structure.factory import s
from pyiron_atomistics.atomistics.structure.pyironase import publication as publication_ase
from pyiron_base import Settings

__author__ = "Ali Zendegani"
__copyright__ = (
    "Copyright 2021, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "1.0"
__maintainer__ = "Liam Huber"
__email__ = "huber@mpie.de"
__status__ = "production"
__date__ = "Feb 26, 2021"

s = Settings()


class AseFactory:
    @wraps(ase_bulk)
    def bulk(self, *args, **kwargs):
        s.publication_add(publication_ase())
        return ase_bulk(*args, **kwargs)

    @wraps(ase_cut)
    def cut(self, *args, **kwargs):
        """
        Returns an ASE's cut result, wrapped as a `pyiron_atomistics.atomstic.structure.atoms.Atoms` object.

        ase.build.cut docstring:

        """
        s.publication_add(publication_ase())
        return ase_cut(*args, **kwargs)

    @wraps(ase_stack)
    def stack(self, *args, **kwargs):
        """
        Returns an ASE's stack result, wrapped as a `pyiron_atomistics.atomstic.structure.atoms.Atoms` object.

        ase.build.stack docstring:

        """
        s.publication_add(publication_ase())
        return ase_stack(*args, **kwargs)

    @wraps(ase_crystal)
    def crystal(self, *args, **kwargs):
        """
        Returns an ASE's crystal result, wrapped as a `pyiron_atomistics.atomstic.structure.atoms.Atoms` object.

        ase.spacegroup.crystal docstring:

        """
        s.publication_add(publication_ase())
        return ase_to_pyiron(ase_crystal(*args, **kwargs))

    @wraps(ase_read)
    def read(self, *args, **kwargs):
        """
        Returns a ASE's read result, wrapped as a `pyiron_atomistics.atomstic.structure.atoms.Atoms` object.

        ase.io.read docstring:
        """
        return ase_to_pyiron(ase_read(*args, **kwargs))
