# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from __future__ import print_function

import numpy as np
from pyiron_vasp.dft.waves.dos import Dos

from pyiron_atomistics.atomistics.structure.atoms import (
    Atoms,
    dict_group_to_hdf,
    structure_dict_to_hdf,
)

__author__ = "Sudarsan Surendralal"
__copyright__ = (
    "Copyright 2021, Max-Planck-Institut für Eisenforschung GmbH "
    "- Computational Materials Design (CM) Department"
)
__version__ = "1.0"
__maintainer__ = "Sudarsan Surendralal"
__email__ = "surendralal@mpie.de"
__status__ = "development"
__date__ = "Sep 1, 2017"


def electronic_structure_dict_to_hdf(data_dict, hdf, group_name):
    with hdf.open(group_name) as h_es:
        for k, v in data_dict.items():
            if k not in ["structure", "dos"]:
                h_es[k] = v

        if "structure" in data_dict.keys():
            structure_dict_to_hdf(data_dict=data_dict["structure"], hdf=h_es)

        dict_group_to_hdf(data_dict=data_dict, hdf=h_es, group="dos")
