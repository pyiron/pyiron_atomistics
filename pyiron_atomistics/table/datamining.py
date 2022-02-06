# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import os
from pyiron_base import TableJob as BaseTableJob, PyironTable
from pyiron_atomistics.table.funct import (
    get_incar,
    get_sigma,
    get_total_number_of_atoms,
    get_elements,
    get_convergence_check,
    get_number_of_species,
    get_number_of_ionic_steps,
    get_ismear,
    get_encut,
    get_n_kpts,
    get_n_equ_kpts,
    get_number_of_final_electronic_steps,
    get_majority_species,
    get_job_name,
    get_energy_tot,
    get_energy_pot,
    get_energy_free,
    get_energy_int,
    get_energy_tot_per_atom,
    get_energy_pot_per_atom,
    get_energy_free_per_atom,
    get_energy_int_per_atom,
    get_e_conv_level,
    get_f_states,
    get_e_band,
    get_majority_crystal_structure,
    get_equilibrium_parameters,
    get_structure,
    get_forces,
    get_magnetic_structure,
    get_average_waves,
    get_plane_waves,
    get_ekin_error,
    get_volume,
    get_volume_per_atom,
)


__author__ = "Uday Gajera, Jan Janssen, Joerg Neugebauer"
__copyright__ = (
    "Copyright 2021, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "0.0.1"
__maintainer__ = "Jan Janssen"
__email__ = "janssen@mpie.de"
__status__ = "development"
__date__ = "Sep 1, 2018"


class TableJob(BaseTableJob):
    """
    Create pyiron table to facilitate data analysis.

    Example (this example requires a Murnaghan job to have run beforehand):

    >>> from pyiron_atomistics import Project
    >>> pr = Project('my_project')
    >>> def db_filter_function(job_table):
    >>>     # Returns a pandas Series of boolean values (True for entries that have status finished
    >>>     # and hamilton type Murnaghan.)
    >>>     return (job_table.status == "finished") & (job_table.hamilton == "Murnaghan")
    >>> def get_lattice_parameter(job):
    >>>     return job["output/equilibrium_volume"] ** (1/3)
    >>> def get_bm(job):
    >>>     return job["output/equilibrium_bulk_modulus"]
    >>> table = pr.create_table("table")
    >>> # assigning a database filter function
    >>> table.db_filter_function = db_filter_function
    >>> # Adding the functions using the labels you like
    >>> table.add["a_eq"] = get_lattice_parameter
    >>> table.add["bulk_modulus"] = get_bm
    >>> table.run()

    Data obtained can be analysed via `table.get_dataframe()`, which returns a pandas dataframe.

    More can be found on this page: https://github.com/pyiron/pyiron_atomistics/blob/master/notebooks/data_mining.ipynb
    """

    def __init__(self, project, job_name):
        super(TableJob, self).__init__(project, job_name)
        self._system_function_lst += [
            get_incar,
            get_sigma,
            get_total_number_of_atoms,
            get_elements,
            get_convergence_check,
            get_number_of_species,
            get_number_of_ionic_steps,
            get_ismear,
            get_encut,
            get_n_kpts,
            get_n_equ_kpts,
            get_number_of_final_electronic_steps,
            get_majority_species,
            get_job_name,
            get_energy_tot,
            get_energy_pot,
            get_energy_free,
            get_energy_int,
            get_energy_tot_per_atom,
            get_energy_pot_per_atom,
            get_energy_free_per_atom,
            get_energy_int_per_atom,
            get_e_conv_level,
            get_f_states,
            get_e_band,
            get_majority_crystal_structure,
            get_equilibrium_parameters,
            get_structure,
            get_forces,
            get_magnetic_structure,
            get_average_waves,
            get_plane_waves,
            get_ekin_error,
            get_volume,
            get_volume_per_atom,
        ]
        self._pyiron_table = PyironTable(
            project=None,
            system_function_lst=self._system_function_lst,
            csv_file_name=os.path.join(self.working_directory, "pyirontable.csv"),
        )
