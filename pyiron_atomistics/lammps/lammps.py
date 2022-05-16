# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from pyiron_atomistics.lammps.interactive import LammpsInteractive

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
