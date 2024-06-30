# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from pyiron_base import Project, ProjectHDFio

from pyiron_atomistics.lammps.interactive import LammpsInteractive
from pyiron_atomistics.atomistics.structure.atoms import ase_to_pyiron

__author__ = "Joerg Neugebauer, Sudarsan Surendralal, Jan Janssen"
__copyright__ = (
    "Copyright 2021, Max-Planck-Institut für Eisenforschung GmbH - "
    "- Computational Materials Design (CM) Department"
)
__version__ = "1.0"
__maintainer__ = "Sudarsan Surendralal"
__email__ = "surendralal@mpie.de"
__status__ = "production"
__date__ = "Sep 1, 2017"


class Lammps(LammpsInteractive):
    """
    Class to setup and run and analyze LAMMPS simulations.

    Example:

    >>> job = pr.create.job.Lammps(job_name='lmp_example')
    >>> job.structure = pr.create.structure.bulk('Fe', cubic=True)
    >>> job.run()

    How to set potential: Look up potentials via `job.view_potentials()` (detailed data frame)
    or via `job.list_potentials()` (potential names). Assign the potential e.g. via:

    >>> job.potential = job.list_potentials()[0]

    Lammps has 3 modes: `static`, `md` and `minimize`. Set a mode e.g. via:

    >>> job.calc_minimize()

    Args:
        project (pyiron_atomistics.project.Project instance):  Specifies the project path among other attributes
        job_name (str): Name of the job

    Attributes:
        input (lammps.Input instance): Instance which handles the input
    """

    def __init__(self, project, job_name):
        super(Lammps, self).__init__(project, job_name)

        self._executable_activate(enforce=True)


def lammps_function(
    working_directory,
    structure,
    potential,
    calc_mode="static",
    calc_kwargs={},
    cutoff_radius=None,
    units="metal",
    bonds_kwargs={},
    enable_h5md=False,
):
    job = Lammps(
        project=ProjectHDFio(
            project=Project(working_directory),
            file_name="lmp_funct_job",
            h5_path=None,
            mode=None,
        ),
        job_name="lmp_funct_job",
    )
    job.structure = ase_to_pyiron(structure)
    job.potential = potential
    job.cutoff_radius = cutoff_radius
    job.units = units
    if calc_mode == "static":
        job.calc_static()
    elif calc_mode == "md":
        job.calc_md(**calc_kwargs)
    elif calc_mode == "minimize":
        job.calc_minimize(**calc_kwargs)
    elif calc_mode == "vcsgc":
        job.calc_vcsgc(**calc_kwargs)
    else:
        raise ValueError()
    if enable_h5md:
        job.enable_h5md()
    if len(bonds_kwargs) > 0:
        job.define_bonds(**bonds_kwargs)

    calculate_kwargs = job.calculate_kwargs
    calculate_kwargs["working_directory"] = working_directory
    return job.get_calculate_function(), calculate_kwargs
