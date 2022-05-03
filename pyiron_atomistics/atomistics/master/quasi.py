# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import numpy as np
from pyiron_atomistics.atomistics.master.murnaghan import MurnaghanJobGenerator
from pyiron_atomistics.atomistics.master.parallel import AtomisticParallelMaster
import matplotlib

__author__ = "Jan Janssen"
__copyright__ = (
    "Copyright 2021, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "0.0.1"
__maintainer__ = "Jan Janssen"
__email__ = "janssen@mpie.de"
__status__ = "development"
__date__ = "Oct 29, 2020"


def calc_v0_from_fit_funct(fit_funct, x, save_range=0.0, return_ind=False):
    fit_funct_der = fit_funct.deriv().r
    fit_funct_der_r = fit_funct_der[fit_funct_der.imag == 0].real
    fit_funct_der_val = fit_funct.deriv(2)(fit_funct_der_r)
    select = (
        (fit_funct_der_val > 0)
        & (fit_funct_der_r > np.min(x) * (1 - save_range))
        & (fit_funct_der_r < np.max(x) * (1 + save_range))
    )
    v0_lst = fit_funct_der_r[select]
    if len(v0_lst) == 1:
        if return_ind:
            return v0_lst[0], (np.abs(x - v0_lst[0])).argmin()
        else:
            return v0_lst[0]
    else:
        select = fit_funct_der_val > 0
        v0_lst = fit_funct_der_r[select]
        if len(v0_lst) == 1:
            if return_ind:
                return v0_lst[0], (np.abs(x - v0_lst[0])).argmin()
            else:
                return v0_lst[0]
        else:
            if return_ind:
                return None, None
            else:
                return None


class QuasiHarmonicJob(AtomisticParallelMaster):
    """
    Obtain finite temperature properties in the framework of quasi harmonic approximation. For the
    theoretical understanding take a look at the Wikipedia page:
    https://en.wikipedia.org/wiki/Quasi-harmonic_approximation

    Example:

    >>> pr = Project('my_project')
    >>> lmp = pr.create_job('Lammps', 'lmp')
    >>> lmp.structure = structure_of_your_choice
    >>> phono = lmp.create_job('PhonopyJob', 'phono')
    >>> qha = phono.create_job('QuasiHarmonicJob', 'qha')
    >>> qha.run()

    The final results can be obtained through `qha.optimise_volume()`.

    The temperature range defined in the input can be modified afterwards. For this, follow these
    lines:

    >>> qha.input['temperature_end'] = temperature_end
    >>> qha.input['temperature_steps'] = temperature_steps
    >>> qha.input['temperature_start'] = temperature_start
    >>> qha.collect_output()
    """

    def __init__(self, project, job_name="murnaghan"):
        """

        Args:
            project:
            job_name:
        """
        super(QuasiHarmonicJob, self).__init__(project, job_name)

        self.__version__ = "0.0.1"

        # define default input
        self.input["num_points"] = (11, "number of sample points")
        self.input["vol_range"] = (
            0.1,
            "relative volume variation around volume defined by ref_ham",
        )
        self.input["temperature_start"] = 0
        self.input["temperature_end"] = 500
        self.input["temperature_steps"] = 10
        self.input["polynomial_degree"] = 3
        self._job_generator = MurnaghanJobGenerator(self)

    def collect_output(self):
        free_energy_lst, entropy_lst, cv_lst, volume_lst = [], [], [], []
        for job_id in self.child_ids:
            job = self.project_hdf5.load(job_id)
            thermal_properties = job.get_thermal_properties(
                temperatures=np.linspace(
                    self.input["temperature_start"],
                    self.input["temperature_end"],
                    int(self.input["temperature_steps"]),
                )
            )
            free_energy_lst.append(thermal_properties.free_energies)
            entropy_lst.append(thermal_properties.entropy)
            cv_lst.append(thermal_properties.cv)
            volume_lst.append(job.structure.get_volume())

        arg_lst = np.argsort(volume_lst)

        self._output["free_energy"] = np.array(free_energy_lst)[arg_lst]
        self._output["entropy"] = np.array(entropy_lst)[arg_lst]
        self._output["cv"] = np.array(cv_lst)[arg_lst]

        temperature_mesh, volume_mesh = np.meshgrid(
            np.linspace(
                self.input["temperature_start"],
                self.input["temperature_end"],
                int(self.input["temperature_steps"]),
            ),
            np.array(volume_lst)[arg_lst],
        )

        self._output["volumes"] = volume_mesh
        self._output["temperatures"] = temperature_mesh
        with self.project_hdf5.open("output") as hdf5_out:
            for key, val in self._output.items():
                hdf5_out[key] = val

    def optimise_volume(self, bulk_eng):
        """
        Get finite temperature properties.

        Args:
            bulk_eng (numpy.ndarray): array of bulk energies corresponding to the box sizes given
                in the quasi harmonic calculations. For the sake of compatibility, it is strongly
                recommended to use the pyiron Murnaghan class (and make sure that you use the
                same values for `num_points` and `vol_range`).

        Returns:
            volume, free energy, entropy, heat capacity

        The corresponding temperature values can be obtained from `job['output/temperatures'][0]`
        """
        v0_lst, free_eng_lst, entropy_lst, cv_lst = [], [], [], []
        for i, [t, free_energy, cv, entropy, v] in enumerate(
            zip(
                self["output/temperatures"].T,
                self["output/free_energy"].T,
                self["output/cv"].T,
                self["output/entropy"].T,
                self["output/volumes"].T,
            )
        ):
            fit = np.poly1d(
                np.polyfit(
                    v, free_energy + bulk_eng, int(self.input["polynomial_degree"])
                )
            )
            v0, ind = calc_v0_from_fit_funct(
                fit_funct=fit, x=v, save_range=0.0, return_ind=True
            )

            v0_lst.append(v0)
            free_eng_lst.append(fit([v0]))
            entropy_lst.append(entropy[ind])
            cv_lst.append(cv[ind])
        return v0_lst, free_eng_lst, entropy_lst, cv_lst

    def plot_free_energy_volume_temperature(
        self,
        murnaghan_job=None,
        temperature_start=None,
        temperature_end=None,
        temperature_steps=None,
        color_map="coolwarm",
        axis=None,
        *args,
        **kwargs
    ):
        """
        Plot volume vs free energy curves for defined temperatures. If no Murnaghan job is assigned, plots free energy without total electronic energy at T=0.

        Args:
            murnaghan_job: job_name or job_id of the Murnaghan job. if None, total electronic energy at T=0 is set to zero.
            temperature_start: if None, value will be used from job.input['temperature_start']
            temperature_end: if None, value will be used from job.input['temperature_end']
            temperature_steps: if None, value will be used from job.input['temperature_steps']
            color_map: colormaps options accessible via matplotlib.cm.get_cmap. Default is 'coolwarm'.
            axis (matplotlib axis, optional): plot to this axis, if not given a new one is created.
            *args: passed through to matplotlib.pyplot.plot when plotting free energies.
            **kwargs: passed through to matplotlib.pyplot.plot when plotting free energies.

        Returns:
            matplib axis: the axis the figure has been drawn to, if axis is given the same object is returned.
        """
        energy_zero = 0
        if murnaghan_job != None:
            job_murn = self.project.load(murnaghan_job)
            energy_zero = job_murn["output/energy"]

        if not self.status.finished:
            raise ValueError(
                "QHA Job must be successfully run, before calling this method."
            )

        if not job_murn.status.finished:
            raise ValueError(
                "Murnaghan Job must be successfully run, before calling this method."
            )

        if axis is None:
            _, axis = matplotlib.pyplot.subplots(1, 1)

        cmap = matplotlib.cm.get_cmap(color_map)

        if temperature_start != None:
            self.input["temperature_start"] = temperature_start
        if temperature_end != None:
            self.input["temperature_end"] = temperature_end
        if temperature_steps != None:
            self.input["temperature_steps"] = temperature_steps
        self.collect_output()

        for i, [t, free_energy, v] in enumerate(
            zip(
                self["output/temperatures"].T,
                self["output/free_energy"].T,
                self["output/volumes"].T,
            )
        ):
            color = cmap(i / len(self["output/temperatures"].T))
            axis.plot(v, free_energy + energy_zero, color=color)

        axis.set_xlabel("Volume")
        axis.set_ylabel("Free Energy")

        temperatures = self["output/temperatures"]
        normalize = matplotlib.colors.Normalize(
            vmin=temperatures.min(), vmax=temperatures.max()
        )
        scalarmappaple = matplotlib.cm.ScalarMappable(norm=normalize, cmap=cmap)
        scalarmappaple.set_array(temperatures)
        cbar = matplotlib.pyplot.colorbar(scalarmappaple)
        cbar.set_label("Temperature")
        return axis
