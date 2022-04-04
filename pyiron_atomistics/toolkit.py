# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.
"""
A toolkit for managing extensions to the project from atomistics.
"""

from pyiron_base import Toolkit, Project, JobFactoryCore, JOB_CLASS_DICT
from pyiron_atomistics.atomistics.structure.factory import StructureFactory

__author__ = "Liam Huber"
__copyright__ = (
    "Copyright 2021, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "1.0"
__maintainer__ = "Liam Huber"
__email__ = "huber@mpie.de"
__status__ = "production"
__date__ = "Sep 7, 2021"


class JobFactory(JobFactoryCore):
    @property
    def _job_class_dict(self) -> dict:
        return JOB_CLASS_DICT


class AtomisticsTools(Toolkit):
    def __init__(self, project: Project):
        super().__init__(project)
        self._structure = StructureFactory()
        self._job = JobFactory(project)

    @property
    def structure(self) -> StructureFactory:
        return self._structure

    @property
    def job(self) -> JobFactory:
        return self._job
