# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.
import os
from typing import Optional

from ase.atoms import Atoms
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
    working_directory: str,
    structure: Atoms,
    potential: str,
    calc_mode: str = "static",
    calc_kwargs: dict = {},
    cutoff_radius: Optional[float] = None,
    units: str = "metal",
    bonds_kwargs: dict = {},
    server_kwargs: dict = {},
    enable_h5md: bool = False,
    write_restart_file: bool = False,
    read_restart_file: bool = False,
    restart_file: str = "restart.out",
    executable_version: Optional[str] = None,
    executable_path: Optional[str] = None,
):
    """

    Args:
        working_directory (str):
        structure (Atoms):
        potential (str):
        calc_mode (str):
        calc_kwargs (dict):
        cutoff_radius (float):
        units (str):
        bonds_kwargs (dict):
        server_kwargs (dict):
        enable_h5md (bool):
        write_restart_file (bool):
        read_restart_file (bool):
        restart_file (str):
        executable_version (str):
        executable_path (str):

    Returns:
        str, dict, bool: Tuple consisting of the shell output (str), the parsed output (dict) and a boolean flag if
                         the execution raised an accepted error.
    """
    os.makedirs(working_directory, exist_ok=True)
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
    server_dict = job.server.to_dict()
    server_dict.update(server_kwargs)
    job.server.from_dict(server_dict=server_dict)
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
    if executable_path is not None:
        job.executable = executable_path
    if executable_version is not None:
        job.version = executable_version
    if enable_h5md:
        job.enable_h5md()
    if write_restart_file:
        job.write_restart_file(filename=restart_file)
    if read_restart_file:
        job.read_restart_file(filename=os.path.basename(restart_file))
        job.restart_file_list.append(restart_file)
    if len(bonds_kwargs) > 0:
        job.define_bonds(**bonds_kwargs)

    calculate_kwargs = job.calculate_kwargs
    calculate_kwargs["working_directory"] = working_directory
    return job.get_calculate_function()(**calculate_kwargs)
