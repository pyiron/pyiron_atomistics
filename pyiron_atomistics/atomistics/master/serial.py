# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from __future__ import print_function
from collections import OrderedDict
from pyiron_base import SerialMasterBase
from pyiron_atomistics.atomistics.job.atomistic import AtomisticGenericJob

__author__ = "Jan Janssen"
__copyright__ = (
    "Copyright 2021, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "1.0"
__maintainer__ = "Jan Janssen"
__email__ = "janssen@mpie.de"
__status__ = "production"
__date__ = "Sep 1, 2017"


class GenericOutput(OrderedDict):
    def __init__(self):
        super(GenericOutput, self).__init__()


class SerialMaster(SerialMasterBase, AtomisticGenericJob):
    """
    Atomistic serial master job, runs jobs in serial like :class:`pyiron_base.SerialMasterBase`, but for atomistic jobs.

    Args:
        project (ProjectHDFio): ProjectHDFio instance which points to the HDF5 file the job is stored in
        job_name (str): name of the job, which has to be unique within the project

    Attributes:

        .. attribute:: structure
                allows to set the structure on the start job

    Methods:

        .. method:: get_structure
            dispatches to :meth:`.AtomisticGenericJob._get_structure` of the final job, i.e. allows you to access the
            structures of the final job.
    """

    @property
    def structure(self):
        if self.ref_job is not None:
            return self.ref_job.structure
        else:
            return None

    @structure.setter
    def structure(self, basis):
        if self.ref_job is not None:
            self.ref_job.structure = basis
        else:
            raise ValueError(
                "A structure can only be set after a start job has been assigned."
            )

    def _get_structure(self, frame=-1, wrap_atoms=True):
        return self.project.load(self.child_ids[-1]).get_structure(
            frame=frame, wrap_atoms=wrap_atoms
        )

    def _number_of_structures(self):
        if len(self.child_ids) > 0:
            return self.project.load(self.child_ids[-1])._number_of_structures()
        else:
            return 0
