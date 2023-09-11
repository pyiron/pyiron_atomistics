# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from atomistics.shared.thermo.thermo import ThermoBulk as AtomisticsThermoBulk

__author__ = "Joerg Neugebauer, Jan Janssen"
__copyright__ = (
    "Copyright 2021, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "1.0"
__maintainer__ = "Jan Janssen"
__email__ = "janssen@mpie.de"
__status__ = "development"
__date__ = "Sep 1, 2017"


class ThermoBulk(AtomisticsThermoBulk):
    """
    Class should provide all tools to compute bulk thermodynamic quantities. Central quantity is the Free Energy F(V,T).
    ToDo: Make it a (light weight) pyiron object (introduce a new tool rather than job object).

    Args:
        project:
        name:

    """

    def __init__(self, project=None, name=None):
        # only for compatibility with pyiron objects
        self._project = project
        self._name = name

        super().__init__()
