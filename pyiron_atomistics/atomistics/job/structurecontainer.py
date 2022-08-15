# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

__author__ = "Yury Lysogorskiy, Jan Janssen, Marvin Poul"
__copyright__ = (
    "Copyright 2021, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "0.1"
__maintainer__ = "Marvin Poul"
__email__ = "poul@mpie.de"
__status__ = "development"
__date__ = "Aug 12, 2020"

from pyiron_base import DataContainer, GenericJob, deprecate
from pyiron_atomistics.atomistics.job.atomistic import AtomisticGenericJob
from pyiron_atomistics.atomistics.structure.atoms import Atoms
from pyiron_atomistics.atomistics.structure.has_structure import HasStructure
from pyiron_atomistics.atomistics.structure.structurestorage import StructureStorage


class StructureContainer(GenericJob, HasStructure):
    """
    Container to save a list of structures in HDF5 together with tags.

    Add new structures with :meth:`.append`, they are added to
    :attr:`.structure_lst`.  The HDF5 is written when :meth:`.run` is called.
    """

    def __init__(self, project, job_name):
        super().__init__(project, job_name)
        self.__version__ = "0.2.0"
        self.__hdf_version__ = "0.3.0"
        self._structure_lst = DataContainer(table_name="structures")
        self._container = StructureStorage()
        self.server.run_mode.interactive = True

    @property
    def structure_lst(self):
        """
        :class:`.DataContainer`: list of :class:`~.Atoms`
        """
        if len(self._structure_lst) != len(self._container):
            self._structure_lst = DataContainer(list(self._container.iter_structures()))
        return self._structure_lst

    @staticmethod
    def _to_structure(structure_or_job):
        """
        Return structure from structure or atomic job.

        Args:
            structure_or_job (:class:`~.AtomisticGenericJob`, :class:`~.Atoms`):
                if :class:`~.AtomisticGenericJob` try to get most recent structure,
        copy it and set the job_id in :attr:`~.Atoms.info`

        Returns:
            :class:`~.Atoms`: structure from the job or given structure

        Raises:
            ValueError: if given :class:`~.AtomisticGenericJob` has no structure set
            TypeError: if structure_or_job is of invalid type
        """
        if isinstance(structure_or_job, AtomisticGenericJob):
            if structure_or_job.structure:
                s = structure_or_job.get_structure(-1).copy()
                s.info["jobid"] = structure_or_job.job_id
                return s
            else:
                raise ValueError("The job does not contain any structure to import.")
        elif isinstance(structure_or_job, Atoms):
            return structure_or_job
        else:
            raise TypeError(
                f"structure_or_job must be of type {Atoms} or {AtomisticGenericJob}, not {type(structure_or_job)}"
            )

    def append(self, structure_or_job):
        """
        Add new structure to structure list.

        The added structure will available in :attr:`~.structure_lst`.  If the
        structure is added via a job, retrieve the latest structure and add its
        id to :attr:`pyiron_atomistics.atomistics.generic.Atoms.info`.

        Args:
            structure_or_job (:class:`~.AtomisticGenericJob`/:class:`~.Atoms`):
                if :class:`~.AtomisticGenericJob` add from
                :meth:`~.AtomisticGenericJob.get_structure`,
                otherwise add just the given :class:`~.Atoms`

        Returns:
            dict: item added to :attr:`~.structure_lst`
        """
        struct = self._to_structure(structure_or_job)
        self._container.add_structure(struct)
        return struct

    def add_structure(self, structure: Atoms, identifier: str = None, **kwargs):
        """
        Add a new structure.

        Args:
            structure (:class:`~.Atoms`): structure to add
            identifier (str, optional): optional identifier for the structure
            **kwargs: passed through to the underlying :meth:`.StructureStorage.add_structure`
        """
        self._container.add_structure(structure, identifier=identifier, **kwargs)

    def run_static(self):
        self.status.finished = True

    def run_if_interactive(self):
        self.to_hdf()
        self.status.finished = True

    def write_input(self):
        pass

    def collect_output(self):
        pass

    @property
    @deprecate("use get_structure()")
    def structure(self):
        return self._get_structure(frame=0)

    @structure.setter
    @deprecate("use append()")
    def structure(self, struct):
        self.append(struct)

    def _number_of_structures(self):
        return len(self._container)

    def _get_structure(self, frame=-1, wrap_atoms=True):
        return self._container._get_structure(frame=frame, wrap_atoms=wrap_atoms)

    def to_hdf(self, hdf=None, group_name=None):
        super().to_hdf(hdf=hdf, group_name=group_name)
        self._container.to_hdf(hdf=self.project_hdf5, group_name="structures")

    def from_hdf(self, hdf=None, group_name=None):
        # keep hdf structure for version peeking in separate variable, so that
        # the inherited from_hdf() can properly deal with it
        h5 = hdf or self.project_hdf5
        if group_name:
            h5 = h5[group_name]
        if "HDF_VERSION" in h5.list_nodes():
            hdf_version = h5["HDF_VERSION"]
        else:
            # old versions didn't use to set a HDF version
            hdf_version = "0.1.0"
        if hdf_version == "0.1.0":
            super().from_hdf(hdf=hdf, group_name=group_name)
            with self.project_hdf5.open("input") as hdf5_input:
                self.append(Atoms().from_hdf(hdf5_input))
        elif hdf_version == "0.2.0":
            GenericJob.from_hdf(self, hdf=hdf, group_name=group_name)

            hdf = self.project_hdf5["structures"]
            for group in sorted(hdf.list_groups()):
                self.append(Atoms().from_hdf(hdf=hdf, group_name=group))
        else:
            super().from_hdf(hdf=hdf, group_name=group_name)
            self._container.from_hdf(hdf=self.project_hdf5, group_name="structures")

    @property
    def plot(self):
        """
        Accessor for :class:`~.StructurePlots` instance using these structures.
        """
        return self._container.plot
