# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.
"""
A toolkit for managing extensions to the project from atomistics.
"""

from pyiron_base import Toolkit, Project, JobFactoryCore
from pyiron_atomistics.atomistics.structure.factory import StructureFactory

from pyiron_atomistics.interactive.activation_relaxation_technique import ART
from pyiron_atomistics.testing.randomatomistic import AtomisticExampleJob
from pyiron_atomistics.dft.master.convergence_encut_parallel import ConvEncutParallel
from pyiron_atomistics.dft.master.convergence_encut_serial import ConvEncutSerial
from pyiron_atomistics.atomistics.master.convergence_volume import ConvergenceVolume
from pyiron_atomistics.dft.master.convergence_kpoint_parallel import ConvKpointParallel
from pyiron_atomistics.atomistics.master.elastic import ElasticTensor
from pyiron_atomistics.testing.randomatomistic import ExampleJob

# from pyiron_atomistics.gaussian.gaussian import Gaussian
from pyiron_atomistics.gpaw.gpaw import Gpaw
from pyiron_atomistics.thermodynamics.hessian import HessianJob
from pyiron_atomistics.lammps.lammps import Lammps
from pyiron_atomistics.atomistics.master.parallel import MapMaster
from pyiron_atomistics.atomistics.master.murnaghan import Murnaghan
from pyiron_atomistics.dft.master.murnaghan_dft import MurnaghanDFT
from pyiron_atomistics.atomistics.master.phonopy import PhonopyJob
from pyiron_atomistics.atomistics.master.quasi import QuasiHarmonicJob
from pyiron_atomistics.interactive.scipy_minimizer import ScipyMinimizer
from pyiron_atomistics.atomistics.master.serial import SerialMaster
from pyiron_atomistics.sphinx.sphinx import Sphinx
from pyiron_atomistics.atomistics.job.structurecontainer import StructureContainer
from pyiron_atomistics.atomistics.master.structure import StructureListMaster
from pyiron_atomistics.atomistics.job.sqs import SQSJob
from pyiron_atomistics.atomistics.master.sqsmaster import SQSMaster
from pyiron_atomistics.thermodynamics.sxphonons import (
    SxPhonons,
    SxDynMat,
    SxUniqDispl,
    SxHarmPotTst,
)
from pyiron_atomistics.interactive.sxextoptint import SxExtOptInteractive
from pyiron_atomistics.table.datamining import TableJob
from pyiron_atomistics.vasp.vasp import Vasp
from pyiron_atomistics.vasp.metadyn import VaspMetadyn
from pyiron_atomistics.vasp.vaspsol import VaspSol

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
        return {
            "ART": ART,
            "AtomisticExampleJob": AtomisticExampleJob,
            "ConvEncutParallel": ConvEncutParallel,
            "ConvEncutSerial": ConvEncutSerial,
            "ConvergenceVolume": ConvergenceVolume,
            "ConvKpointParallel": ConvKpointParallel,
            "ElasticTensor": ElasticTensor,
            "ExampleJob": ExampleJob,
            # "Gaussian": Gaussian,
            "Gpaw": Gpaw,
            "HessianJob": HessianJob,
            "Lammps": Lammps,
            "MapMaster": MapMaster,
            "Murnaghan": Murnaghan,
            "MurnaghanDFT": MurnaghanDFT,
            "PhonopyJob": PhonopyJob,
            "QuasiHarmonicJob": QuasiHarmonicJob,
            "ScipyMinimizer": ScipyMinimizer,
            "SerialMaster": SerialMaster,
            "Sphinx": Sphinx,
            "StructureContainer": StructureContainer,
            "StructureListMaster": StructureListMaster,
            "SQSJob": SQSJob,
            "SQSMaster": SQSMaster,
            "SxDynMat": SxDynMat,
            "SxExtOptInteractive": SxExtOptInteractive,
            "SxHarmPotTst": SxHarmPotTst,
            "SxPhonons": SxPhonons,
            "SxUniqDispl": SxUniqDispl,
            "TableJob": TableJob,
            "Vasp": Vasp,
            "VaspMetadyn": VaspMetadyn,
            "VaspSol": VaspSol,
        }


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
