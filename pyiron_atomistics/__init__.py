__version__ = "0.1"
__all__ = []

from pyiron_atomistics.project import Project
from pyiron_atomistics.toolkit import AtomisticsTools
from pyiron_atomistics.atomistics.structure.atoms import (
    ase_to_pyiron,
    pyiron_to_ase,
    Atoms,
)
from pyiron_base import Notebook, install_dialog, JobType

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
JobType.register("pyiron_atomistics.interactive.activation_relaxation_technique", "ART")
JobType.register("pyiron_atomistics.testing.randomatomistic", "AtomisticExampleJob")
JobType.register("pyiron_atomistics.calphy.job", "Calphy")
JobType.register(
    "pyiron_atomistics.dft.master.convergence_encut_parallel", "ConvEncutParallel"
)
JobType.register(
    "pyiron_atomistics.dft.master.convergence_encut_serial", "ConvEncutSerial"
)
JobType.register(
    "pyiron_atomistics.atomistics.master.convergence_volume", "ConvergenceVolume"
)
JobType.register(
    "pyiron_atomistics.dft.master.convergence_kpoint_parallel", "ConvKpointParallel"
)
JobType.register("pyiron_atomistics.atomistics.master.elastic", "ElasticTensor")
JobType.register("pyiron_atomistics.testing.randomatomistic", "ExampleJob")
JobType.register("pyiron_atomistics.gaussian.gaussian", "Gaussian")
JobType.register("pyiron_atomistics.gpaw.gpaw", "Gpaw")
JobType.register("pyiron_atomistics.thermodynamics.hessian", "HessianJob")
JobType.register("pyiron_atomistics.lammps.lammps", "Lammps")
JobType.register("pyiron_atomistics.atomistics.master.parallel", "MapMaster")
JobType.register("pyiron_atomistics.atomistics.master.murnaghan", "Murnaghan")
JobType.register("pyiron_atomistics.dft.master.murnaghan_dft", "MurnaghanDFT")
JobType.register("pyiron_atomistics.atomistics.master.phonopy", "PhonopyJob")
JobType.register("pyiron_atomistics.atomistics.master.quasi", "QuasiHarmonicJob")
JobType.register("pyiron_atomistics.interactive.quasi_newton", "QuasiNewton")
JobType.register("pyiron_atomistics.interactive.scipy_minimizer", "ScipyMinimizer")
JobType.register("pyiron_atomistics.atomistics.master.serial", "SerialMaster")
JobType.register("pyiron_atomistics.sphinx.sphinx", "Sphinx")
JobType.register(
    "pyiron_atomistics.atomistics.job.structurecontainer", "StructureContainer"
)
JobType.register("pyiron_atomistics.atomistics.master.structure", "StructureListMaster")
JobType.register("pyiron_atomistics.atomistics.job.sqs", "SQSJob")
JobType.register("pyiron_atomistics.atomistics.master.sqsmaster", "SQSMaster")
JobType.register("pyiron_atomistics.thermodynamics.sxphonons", "SxDynMat")
JobType.register("pyiron_atomistics.interactive.sxextoptint", "SxExtOptInteractive")
JobType.register("pyiron_atomistics.thermodynamics.sxphonons", "SxHarmPotTst")
JobType.register("pyiron_atomistics.thermodynamics.sxphonons", "SxPhonons")
JobType.register("pyiron_atomistics.thermodynamics.sxphonons", "SxUniqDispl")
JobType.register("pyiron_atomistics.table.datamining", "TableJob")
JobType.register("pyiron_atomistics.vasp.vasp", "Vasp")
JobType.register("pyiron_atomistics.vasp.metadyn", "VaspMetadyn")
JobType.register("pyiron_atomistics.vasp.vaspsol", "VaspSol")

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions


def install():
    install_dialog()
