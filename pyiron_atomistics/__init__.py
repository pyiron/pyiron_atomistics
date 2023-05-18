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
JOB_CLASS_DICT.update(
    {
        "ART": "pyiron_atomistics.interactive.activation_relaxation_technique",
        "AtomisticExampleJob": "pyiron_atomistics.testing.randomatomistic",
        "Calphy": "pyiron_atomistics.calphy.job",
        "ConvEncutParallel": "pyiron_atomistics.dft.master.convergence_encut_parallel",
        "ConvEncutSerial": "pyiron_atomistics.dft.master.convergence_encut_serial",
        "ConvergenceVolume": "pyiron_atomistics.atomistics.master.convergence_volume",
        "ConvKpointParallel": "pyiron_atomistics.dft.master.convergence_kpoint_parallel",
        "ElasticTensor": "pyiron_atomistics.atomistics.master.elastic",
        "ExampleJob": "pyiron_atomistics.testing.randomatomistic",
        "Gpaw": "pyiron_atomistics.gpaw.gpaw",
        "HessianJob": "pyiron_atomistics.thermodynamics.hessian",
        "Lammps": "pyiron_atomistics.lammps.lammps",
        "MapMaster": "pyiron_atomistics.atomistics.master.parallel",
        "Murnaghan": "pyiron_atomistics.atomistics.master.murnaghan",
        "MurnaghanDFT": "pyiron_atomistics.dft.master.murnaghan_dft",
        "PhonopyJob": "pyiron_atomistics.atomistics.master.phonopy",
        "QuasiHarmonicJob": "pyiron_atomistics.atomistics.master.quasi",
        "QuasiNewton": "pyiron_atomistics.interactive.quasi_newton",
        "ScipyMinimizer": "pyiron_atomistics.interactive.scipy_minimizer",
        "SerialMaster": "pyiron_atomistics.atomistics.master.serial",
        "Sphinx": "pyiron_atomistics.sphinx.sphinx",
        "StructureContainer": "pyiron_atomistics.atomistics.job.structurecontainer",
        "StructureListMaster": "pyiron_atomistics.atomistics.master.structure",
        "SQSJob": "pyiron_atomistics.atomistics.job.sqs",
        "SQSMaster": "pyiron_atomistics.atomistics.master.sqsmaster",
        "SxDynMat": "pyiron_atomistics.thermodynamics.sxphonons",
        "SxExtOptInteractive": "pyiron_atomistics.interactive.sxextoptint",
        "SxHarmPotTst": "pyiron_atomistics.thermodynamics.sxphonons",
        "SxPhonons": "pyiron_atomistics.thermodynamics.sxphonons",
        "SxUniqDispl": "pyiron_atomistics.thermodynamics.sxphonons",
        "TableJob": "pyiron_atomistics.table.datamining",
        "Vasp": "pyiron_atomistics.vasp.vasp",
        "VaspMetadyn": "pyiron_atomistics.vasp.metadyn",
        "VaspSol": "pyiron_atomistics.vasp.vaspsol",
    }
)

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions


def install():
    install_dialog()
