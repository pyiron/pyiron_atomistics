# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from functools import wraps
from inspect import getmodule
from ase.build import cut as ase_cut, stack as ase_stack, bulk as ase_bulk
from ase.io import read as ase_read
from ase.spacegroup import crystal as ase_crystal
from pyiron_atomistics.atomistics.structure.atoms import ase_to_pyiron
from pyiron_atomistics.atomistics.structure.pyironase import (
    publication as publication_ase,
)
from pyiron_base import state

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


def _ase_header(ase_func):
    chain = getmodule(ase_func).__name__
    name = chain.split(".")[-1]
    return f"""
    Returns an ASE's {name} result as a `pyiron_atomistics.atomstic.structure.atoms.Atoms`.

    {chain} docstring:

    """


def _ase_wraps(ase_func):
    def decorator(func):
        @wraps(ase_func)
        def wrapper(*args, **kwargs):
            state.publications.add(publication_ase())
            return func(*args, **kwargs)

        wrapper.__doc__ = _ase_header(ase_func) + wrapper.__doc__
        return wrapper

    return decorator


class AseFactory:
    @_ase_wraps(ase_bulk)
    def bulk(self, *args, **kwargs):
        return ase_to_pyiron(ase_bulk(*args, **kwargs))

    @_ase_wraps(ase_cut)
    def cut(self, *args, **kwargs):
        return ase_cut(*args, **kwargs)

    @_ase_wraps(ase_stack)
    def stack(self, *args, **kwargs):
        return ase_stack(*args, **kwargs)

    @_ase_wraps(ase_crystal)
    def crystal(self, *args, **kwargs):
        return ase_to_pyiron(ase_crystal(*args, **kwargs))

    @_ase_wraps(ase_read)
    def read(self, *args, **kwargs):
        return ase_to_pyiron(ase_read(*args, **kwargs))
