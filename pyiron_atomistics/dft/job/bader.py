# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import numpy as np
import os
import subprocess

__author__ = "Sudarsan Surendralal"
__copyright__ = (
    "Copyright 2021, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "1.0"
__maintainer__ = "Sudarsan Surendralal"
__email__ = "surendralal@mpie.de"
__status__ = "production"
__date__ = "Sep 1, 2017"


class Bader:

    def __init__(self, job):
        self.job = job
        self._working_directory = job.working_directory
        self._structure = job.structure

    def create_cube_files(self):
        cd_val, cd_total = self.job.get_valence_and_total_charge_density()
        cd_val.write_cube_file(filename=os.path.join(self._working_directory, "valence_charge.CUBE"))
        cd_total.write_cube_file(filename=os.path.join(self._working_directory, "total_charge.CUBE"))

    def call_bader(self, extra_arguments=None):
        self.create_cube_files()
        # Go to the working directory to do this
        cmd_1 = "cd {}".format(self._working_directory)
        cmd_2 = list()
        cmd_2.append("bader")
        cmd_2.append("{0}/valence_charge.CUBE -ref {0}/total_charge.CUBE".format(self._working_directory))
        if extra_arguments is not None:
            cmd_2.append(extra_arguments)
        cmd_2 = " ".join(cmd_2)
        subprocess.call(";".join([cmd_1, cmd_2]), shell=True)
        self.remove_cube_files()
        return self.parse_charge_vol()

    def remove_cube_files(self):
        os.remove(os.path.join(self._working_directory, "valence_charge.CUBE"))
        os.remove(os.path.join(self._working_directory, "total_charge.CUBE"))

    def parse_charge_vol(self):
        filename = os.path.join(self._working_directory, "ACF.dat")
        with open(filename) as f:
            lines = f.readlines()
            charges = np.genfromtxt(lines[2:], max_rows=len(self._structure))[:, 4]
            volumes = np.genfromtxt(lines[2:], max_rows=len(self._structure))[:, 6]
        return charges, volumes
