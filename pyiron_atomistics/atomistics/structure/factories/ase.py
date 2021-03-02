# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from functools import wraps
from inspect import getmodule
from ase.build import cut as ase_cut, stack as ase_stack, bulk as ase_bulk
from ase.io import read as ase_read
from ase.spacegroup import crystal as ase_crystal
from pyiron_atomistics.atomistics.structure.atoms import ase_to_pyiron
from pyiron_atomistics.atomistics.structure.pyironase import publication as publication_ase
from pyiron_base import Settings

__author__ = "Ali Zendegani"
__copyright__ = (
    "Copyright 2021, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "1.0.1"
__maintainer__ = "Liam Huber"
__email__ = "huber@mpie.de"
__status__ = "production"
__date__ = "Feb 26, 2021"

s = Settings()


def _ase_wrapped_doc(func):
    chain = getmodule(func).__name__
    name = chain.split('.')[-1]
    return f"""
    Returns an ASE's {name} result, wrapped as a `pyiron_atomistics.atomstic.structure.atoms.Atoms` object.

    {chain} docstring:

    """


class AseFactory:
    @wraps(ase_bulk)
    def bulk(self, *args, **kwargs):
        s.publication_add(publication_ase())
        return ase_to_pyiron(ase_bulk(*args, **kwargs))
    bulk.__doc__ = _ase_wrapped_doc(ase_bulk) + bulk.__doc__

    @wraps(ase_cut)
    def cut(self, *args, **kwargs):
        s.publication_add(publication_ase())
        return ase_cut(*args, **kwargs)
    cut.__doc__ = _ase_wrapped_doc(ase_cut) + cut.__doc__

    @wraps(ase_stack)
    def stack(self, *args, **kwargs):
        s.publication_add(publication_ase())
        return ase_stack(*args, **kwargs)
    stack.__doc__ = _ase_wrapped_doc(ase_stack) + stack.__doc__

    @wraps(ase_crystal)
    def crystal(self, *args, **kwargs):
        s.publication_add(publication_ase())
        return ase_to_pyiron(ase_crystal(*args, **kwargs))
    crystal.__doc__ = _ase_wrapped_doc(ase_crystal) + crystal.__doc__

    @wraps(ase_read)
    def read(self, *args, **kwargs):
        s.publication_add(publication_ase())
        return ase_to_pyiron(ase_read(*args, **kwargs))
    read.__doc__ = _ase_wrapped_doc(ase_read) + read.__doc__
