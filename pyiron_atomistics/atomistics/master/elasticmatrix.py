# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "GPLv3", see the LICENSE file.

from collections import OrderedDict

import numpy as np
import spglib
import scipy.constants
from atomistics.workflows.elastic.helper import (
    generate_structures_helper,
    analyse_structures_helper,
)
from atomistics.workflows.elastic.elastic_moduli import ElasticProperties
from pyiron_atomistics.atomistics.master.parallel import AtomisticParallelMaster
from pyiron_base import JobGenerator

__author__ = "Yury Lysogorskiy"
__copyright__ = "Copyright 2020, Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department"
__version__ = "1.0"
__maintainer__ = "Jan Janssen"
__email__ = "janssen@mpie.de"
__status__ = "development"
__date__ = "Sep 1, 2017"


class ElasticMatrixCalculator(object):
    def __init__(
        self, basis_ref, num_of_point=5, eps_range=0.005, sqrt_eta=True, fit_order=2
    ):
        self.basis_ref = basis_ref.copy()
        self.num_of_point = num_of_point
        self.eps_range = eps_range
        self.sqrt_eta = sqrt_eta
        self.fit_order = fit_order
        self._data = OrderedDict()
        self._structure_dict = OrderedDict()
        self.SGN = None
        self.v0 = None
        self.LC = None
        self.Lag_strain_list = []
        self.epss = np.array([])
        self.zero_strain_job_name = "s_e_0"

    def generate_structures(self):
        """

        Returns:

        """
        self._data, self._structure_dict = generate_structures_helper(
            structure=self.basis_ref.copy(),
            eps_range=self.eps_range,
            num_of_point=self.num_of_point,
            zero_strain_job_name=self.zero_strain_job_name,
            sqrt_eta=self.sqrt_eta,
        )

        # Copy self._data to properties for backwards compatibility
        self.Lag_strain_list = self._data["Lag_strain_list"]
        self.epss = self._data["epss"]
        self.v0 = self._data["v0"]
        self.LC = self._data["LC"]
        self.SGN = self._data["SGN"]
        return self._structure_dict

    def analyse_structures(self, output_dict):
        """

        Returns:

        """
        (
            elastic_matrix,
            self._data["A2"],
            self._data["strain_energy"],
            self._data["e0"],
        ) = analyse_structures_helper(
            output_dict=output_dict,
            Lag_strain_list=self._data["Lag_strain_list"],
            epss=self._data["epss"],
            v0=self._data["v0"],
            LC=self._data["LC"],
            fit_order=self.fit_order,
            zero_strain_job_name=self.zero_strain_job_name,
        )
        elastic = ElasticProperties(elastic_matrix=elastic_matrix)
        self._data.update(
            {
                "C": elastic.elastic_matrix(),
                "S": elastic.elastic_matrix_inverse(),
                "BV": elastic.bulkmodul_voigt(),
                "BR": elastic.bulkmodul_reuss(),
                "BH": elastic.bulkmodul_hill(),
                "GV": elastic.shearmodul_voigt(),
                "GR": elastic.shearmodul_reuss(),
                "GH": elastic.shearmodul_hill(),
                "EV": elastic.youngsmodul_voigt(),
                "ER": elastic.youngsmodul_reuss(),
                "EH": elastic.youngsmodul_hill(),
                "nuV": elastic.poissonsratio_voigt(),
                "nuR": elastic.poissonsratio_reuss(),
                "nuH": elastic.poissonsratio_hill(),
                "AVR": elastic.AVR(),
                "C_eigval": elastic.elastic_matrix_eigval(),
            }
        )

    @staticmethod
    def subjob_name(i, eps):
        """

        Args:
            i:
            eps:

        Returns:

        """
        return ("s_%s_e_%.5f" % (i, eps)).replace(".", "_").replace("-", "m")


class ElasticJobGenerator(JobGenerator):
    @property
    def parameter_list(self):
        """

        Returns:
            (list)
        """
        return [
            [job_name, basis] for job_name, basis in self._job.structure_dict.items()
        ]

    @staticmethod
    def job_name(parameter):
        return str(parameter[0])

    def modify_job(self, job, parameter):
        job.structure = parameter[1]
        return job


class ElasticMatrixJob(AtomisticParallelMaster):
    def __init__(self, project, job_name="elasticmatrix"):
        super(ElasticMatrixJob, self).__init__(project, job_name)
        self.__name__ = "ElasticMatrixJob"
        self.__version__ = "0.0.1"
        self.input["num_of_points"] = (
            5,
            "number of sample point per deformation directions",
        )
        self.input["fit_order"] = (2, "order of the fit polynom")
        self.input["eps_range"] = (0.005, "strain variation")
        self.input["relax_atoms"] = (True, "relax atoms in deformed structure")
        self.input["sqrt_eta"] = (
            True,
            "calculate self-consistently sqrt of stress matrix eta",
        )
        self._data = OrderedDict()
        self.structure_dict = OrderedDict()
        self.property_calculator = None
        self.hdf_storage_group = "elasticmatrix"
        self._job_generator = ElasticJobGenerator(master=self)

    def create_calculator(self):
        if self.property_calculator is None:
            self.property_calculator = ElasticMatrixCalculator(
                basis_ref=self.ref_job.structure.copy(),
                num_of_point=int(self.input["num_of_points"]),
                eps_range=self.input["eps_range"],
                sqrt_eta=self.input["sqrt_eta"],
                fit_order=int(self.input["fit_order"]),
            )
            self.structure_dict = self.property_calculator.generate_structures()
            self._data.update(self.property_calculator._data)

    def run_static(self):
        self.create_calculator()
        if self.input["relax_atoms"]:
            self.ref_job.calc_minimize(pressure=None)
        else:
            self.ref_job.calc_static()
        super(ElasticMatrixJob, self).run_static()

    def run_if_interactive(self):
        self.create_calculator()
        if self.input["relax_atoms"]:
            self.ref_job.calc_minimize(pressure=None)
        else:
            self.ref_job.calc_static()
        super(ElasticMatrixJob, self).run_if_interactive()

    def run_if_refresh(self):
        self.create_calculator()
        super(ElasticMatrixJob, self).run_if_refresh()

    def collect_output(self):
        if not self._data:
            self.from_hdf()
        self.create_calculator()

        energies = {}
        self._data["id"] = []
        if self.server.run_mode.interactive:
            child_id = self.child_ids[0]
            self._data["id"].append(child_id)
            child_job = self.project_hdf5.inspect(child_id)
            energies = {
                job_name: energy
                for job_name, energy in zip(
                    self.structure_dict.keys(), child_job["output/generic/energy_tot"]
                )
            }
        else:
            for job_id in self.child_ids:
                ham = self.project_hdf5.inspect(job_id)
                en = ham["output/generic/energy_tot"][-1]
                energies[ham.job_name] = en
                self._data["id"].append(ham.job_id)

        self.property_calculator.analyse_structures(energies)
        self._data.update(self.property_calculator._data)
        self.to_hdf()

    def from_hdf(self, hdf=None, group_name=None):
        """
        Restore object from hdf5 format
        :param hdf: Optional hdf5 file, otherwise self._hdf5 is used.
        :param group_name: Optional hdf5 group in the hdf5 file.
        """
        super(ElasticMatrixJob, self).from_hdf(hdf=hdf, group_name=group_name)
        try:
            with self.project_hdf5.open("output") as hdf5_out:
                self._data.update(hdf5_out[self.hdf_storage_group])
        except Exception as e:
            print(e)

    def to_hdf(self, hdf=None, group_name=None):
        super(ElasticMatrixJob, self).to_hdf(hdf=hdf, group_name=group_name)
        with self.project_hdf5.open("output") as hdf5_out:
            hdf5_out[self.hdf_storage_group] = self._data
