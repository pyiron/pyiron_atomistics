# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from pyiron_atomistics.calphy.base import CalphyBase

__author__ = "Sarath Menon"
__copyright__ = (
    "Copyright 2022, Max-Planck-Institut für Eisenforschung GmbH - "
    "- Computational Materials Design (CM) Department"
)
__version__ = "1.0"
__maintainer__ = "Sarath Menon"
__email__ = "s.menon@mpie.de"
__status__ = "production"
__date__ = "Feb 14, 2022"


class Calphy(CalphyBase):
    """
    Class to setup and run and analyze Calphy simulations

    Args:
        project (pyiron_atomistics.project.Project instance):  Specifies the project path among other attributes
        job_name (str): Name of the job

    Attributes:
        input (calphy.Input instance): Instance which handles the input
    """

    def __init__(self, project, job_name):
        super(Calphy, self).__init__(project, job_name)
        self.__name__ = "Calphy"
        #self._executable_activate(enforce=True)