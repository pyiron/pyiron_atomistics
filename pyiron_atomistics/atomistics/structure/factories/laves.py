# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from pyiron_atomistics.atomistics.structure.factories.ase import AseFactory

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


class LavesFactory:
    """A collection of routines for constructing Laves phase structures."""
    def __init__(self):
        self._bulk = AseFactory().bulk

    def C14(self):
        raise NotImplementedError

    def C15(self):
        raise NotImplementedError

    def C36(self):
        raise NotImplementedError
