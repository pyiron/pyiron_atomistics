__version__ = "0.1"
__all__ = []

from pyiron_atomistics.project import Project
from pyiron_atomistics.toolkit import AtomisticsTools
from pyiron_atomistics.atomistics.structure.atoms import (
    ase_to_pyiron,
    pyiron_to_ase,
    Atoms,
)
from pyiron_base import Notebook, install_dialog, JOB_CLASS_DICT

from pyiron_base import Project as ProjectBase

ProjectBase.register_tools("atomistics", AtomisticsTools)

# To maintain backwards compatibility until we deprecate the old structure creation functions:
from pyiron_atomistics.atomistics.structure.factory import (
    StructureFactory as _StructureFactory,
)

create_surface = _StructureFactory.surface
create_ase_bulk = _StructureFactory().ase.bulk
create_structure = _StructureFactory.crystal

# Make classes available for new pyiron version
JOB_CLASS_DICT["ART"] = "pyiron_atomistics.interactive.activation_relaxation_technique"
JOB_CLASS_DICT["AtomisticExampleJob"] = "pyiron_atomistics.testing.randomatomistic"
JOB_CLASS_DICT["Calphy"] = "pyiron_atomistics.calphy.job"
JOB_CLASS_DICT[
    "ConvEncutParallel"
] = "pyiron_atomistics.dft.master.convergence_encut_parallel"
JOB_CLASS_DICT[
    "ConvEncutSerial"
] = "pyiron_atomistics.dft.master.convergence_encut_serial"
JOB_CLASS_DICT[
    "ConvergenceVolume"
] = "pyiron_atomistics.atomistics.master.convergence_volume"
JOB_CLASS_DICT[
    "ConvKpointParallel"
] = "pyiron_atomistics.dft.master.convergence_kpoint_parallel"
JOB_CLASS_DICT["ElasticTensor"] = "pyiron_atomistics.atomistics.master.elastic"
JOB_CLASS_DICT["ExampleJob"] = "pyiron_atomistics.testing.randomatomistic"
JOB_CLASS_DICT["Gaussian"] = "pyiron_atomistics.gaussian.gaussian"
JOB_CLASS_DICT["Gpaw"] = "pyiron_atomistics.gpaw.gpaw"
JOB_CLASS_DICT["HessianJob"] = "pyiron_atomistics.thermodynamics.hessian"
JOB_CLASS_DICT["Lammps"] = "pyiron_atomistics.lammps.lammps"
JOB_CLASS_DICT["MapMaster"] = "pyiron_atomistics.atomistics.master.parallel"
JOB_CLASS_DICT["Murnaghan"] = "pyiron_atomistics.atomistics.master.murnaghan"
JOB_CLASS_DICT["MurnaghanDFT"] = "pyiron_atomistics.dft.master.murnaghan_dft"
JOB_CLASS_DICT["PhonopyJob"] = "pyiron_atomistics.atomistics.master.phonopy"
JOB_CLASS_DICT["QuasiHarmonicJob"] = "pyiron_atomistics.atomistics.master.quasi"
JOB_CLASS_DICT["QuasiNewton"] = "pyiron_atomistics.interactive.quasi_newton"
JOB_CLASS_DICT["ScipyMinimizer"] = "pyiron_atomistics.interactive.scipy_minimizer"
JOB_CLASS_DICT["SerialMaster"] = "pyiron_atomistics.atomistics.master.serial"
JOB_CLASS_DICT["Sphinx"] = "pyiron_atomistics.sphinx.sphinx"
JOB_CLASS_DICT[
    "StructureContainer"
] = "pyiron_atomistics.atomistics.job.structurecontainer"
JOB_CLASS_DICT["StructureListMaster"] = "pyiron_atomistics.atomistics.master.structure"
JOB_CLASS_DICT["SQSJob"] = "pyiron_atomistics.atomistics.job.sqs"
JOB_CLASS_DICT["SQSMaster"] = "pyiron_atomistics.atomistics.master.sqsmaster"
JOB_CLASS_DICT["SxDynMat"] = "pyiron_atomistics.thermodynamics.sxphonons"
JOB_CLASS_DICT["SxExtOptInteractive"] = "pyiron_atomistics.interactive.sxextoptint"
JOB_CLASS_DICT["SxHarmPotTst"] = "pyiron_atomistics.thermodynamics.sxphonons"
JOB_CLASS_DICT["SxPhonons"] = "pyiron_atomistics.thermodynamics.sxphonons"
JOB_CLASS_DICT["SxUniqDispl"] = "pyiron_atomistics.thermodynamics.sxphonons"
JOB_CLASS_DICT["TableJob"] = "pyiron_atomistics.table.datamining"
JOB_CLASS_DICT["Vasp"] = "pyiron_atomistics.vasp.vasp"
JOB_CLASS_DICT["VaspMetadyn"] = "pyiron_atomistics.vasp.metadyn"
JOB_CLASS_DICT["VaspSol"] = "pyiron_atomistics.vasp.vaspsol"

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions


def install():
    install_dialog()
