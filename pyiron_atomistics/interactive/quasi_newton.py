# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import numpy as np
import os
import time
import warnings
from pyiron_base import Settings, GenericParameters, Executable, deprecate
from pyiron_atomistics.atomistics.job.interactivewrapper import (
    InteractiveWrapper,
    ReferenceJobOutput,
)
from pyiron_atomistics.atomistics.job.interactive import InteractiveInterface

__author__ = "Jan Janssen, Osamu Waseda"
__copyright__ = (
    "Copyright 2021, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "1.0"
__maintainer__ = "Jan Janssen"
__email__ = "janssen@mpie.de"
__status__ = "development"
__date__ = "Sep 1, 2018"


s = Settings()

class QuasiNewtonInteractive:
    def __init__(
        self,
        structure,
        initial_hessian=10,
        max_step_length=1.0e-1,
        ssa=False,
    ):
        self.hessian = initial_hessian*np.eye(np.prod(structure.positions.shape))
        self.ref_structure = structure
        self.max_step_length = max_step_length
        self.reverse_regularization_eigenvalues = True
        self.max_number_of_ridge_iterations = 1000
        self.gradient = np.zeros_like(structure.positions)

    def reverse_hessian(self, diffusion_vector, diffusion_id=None):
        if diffusion_id is not None:
            v = np.zeros_like(self.structure.positions)
            v[diffusion_id] = diffusion_vector
        else:
            v = diffusion_vector
        if v.shape != self.ref_structure.positions.shape:
            raise AssertionError('diffusion_vector must have the same shape as structure positions')
        v = v.flatten()
        v = np.einsum('i,j->ij', v, v)/np.linalg.norm(v)**2
        self.hessian -= (np.absolute(self.hessian).max()+1)*v

    def set_forces(forces):
        qn.update_hessian(-forces)

    def get_positions(self):
        ridge_parameter = 0
        ridge_matrix = 0
        for _ in range(self.max_number_of_ridge_iterations):
            H = self.hessian+ridge_matrix*ridge_parameter
            self.dx = -np.einsum(
                'ij,j->i', np.linalg.inv(H), self.gradient.flatten()
            ).reshape(-1, 3)
            if np.linalg.norm(self.dx, axis=-1).max()<self.max_step_length:
                break
            else:
                if ridge_parameter==0:
                    if self.reverse_regularization_eigenvalues:
                        E, D = np.linalg.eigh(self.hessian)
                        L = np.sign(E)
                        ridge_matrix = np.einsum('ij,j,lj->il', D, L, D)
                    else:
                        ridge_matrix = np.eye(len(self.hessian))
                ridge_parameter += 1.
        self.ref_structure.positions += self.dx
        return self.ref_structure.positions

    def update_hessian(self, gradient, threshold=1e-4):
        if np.linalg.norm(self.gradient)>0:
            dg = self.get_dg(gradient).flatten()
            H_tmp = dg-np.einsum('ij,j->i', self.hessian, self.dx.flatten())
            numerator = np.dot(H_tmp, self.dx.flatten())
            if np.absolute(numerator) > threshold:
                dH = np.outer(H_tmp, H_tmp)/numerator
                self.hessian = dH+self.hessian
        self.gradient = gradient

    def get_dg(self, gradient):
        return gradient-self.gradient


class QuasiNewton(InteractiveWrapper):
    def __init__(self, project, job_name):
        super().__init__(project, job_name)
        self.__name__ = "QuasiNewton"
        self.__version__ = (
            None
        )  # Reset the version number to the executable is set automatically
        self.input = Input()
        self.output = QuasiNewtonOutput(job=self)
        self._interactive_number_of_steps = 0

    def set_input_to_read_only(self):
        """
        This function enforces read-only mode for the input classes, but it has to be implement in
        the individual classes.
        """
        super().set_input_to_read_only()
        self.input.read_only = True

    def write_input(self):
        pass

    def run_static(self):
        """
        The run if modal function is called by run to execute the simulation, while waiting for the
        output. For this we use subprocess.check_output()
        """
        self._interactive_interface = QuasiNewtonInteractive(
            structure=self.ref_job.structure,
            max_step_length=float(self.input["max_step_length"]),
            initial_hessian=float(self.input["initial_hessian"]),
            ssa=self.input['ssa'],
        )
        self.status.running = True
        self._logger.info("job status: %s", self.status)
        new_positions = self.ref_job.structure.positions
        self.ref_job_initialize()
        while 1:
            str_temp = self.ref_job.structure
            str_temp.positions = new_positions
            self.ref_job.structure = str_temp
            if self.ref_job.server.run_mode.interactive:
                self._logger.debug("QuasiNewton: step start!")
                self.ref_job.run()
                self._logger.debug("QuasiNewton: step finished!")
            else:
                self.ref_job.run(delete_existing_job=True)
            if self.ref_job.output.force_max < self.input['ionic_force_tolerance']:
                break
            self._interactive_interface.set_forces(forces=self.get_forces())
            new_positions = self._interactive_interface.get_positions()
            self._interactive_number_of_steps += 1
            if self._interactive_number_of_steps > self.input["ionic_steps"]:
                break
        self.status.collect = True
        if self.ref_job.server.run_mode.interactive:
            self.ref_job.interactive_close()
        self._interactive_interface.interactive_close()
        self.run()

    def get_forces(self):
        ff = np.array(self.ref_job.output.forces[-1])
        if hasattr(self.ref_job.structure, "selective_dynamics"):
            ff[np.array(self.ref_job.structure.selective_dynamics) == False] = 0
            return ff
        return ff


class Input(GenericParameters):
    """
    class to control the generic input for a Sphinx calculation.

    Args:
        input_file_name (str): name of the input file
        table_name (str): name of the GenericParameters table
    """

    def __init__(self, input_file_name=None, table_name="input"):
        super(Input, self).__init__(
            input_file_name=input_file_name,
            table_name=table_name,
            comment_char="//",
            separator_char="=",
            end_value_char=";",
        )

    def load_default(self):
        """
        Loads the default file content
        """
        file_content = (
            "ionic_steps = 1000 // maximum number of ionic steps\n"
            "ionic_energy_tolerance = 1.0e-3\n"
            "ionic_force_tolerance = 1.0e-2\n"
            "max_step_length = 1.0e-1 // maximum displacement at each step\n"
            "ssa = False // ignore different magnetic moment values when internal symmetries are considered\n"
            "initial_hessian = 10.0 // Initial diagonal Hessian values\n"
        )
        self.load_string(file_content)


class QuasiNewtonOutput(ReferenceJobOutput):
    def __init__(self, job):
        super().__init__(job=job)
