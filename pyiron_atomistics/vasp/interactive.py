# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import os
from subprocess import PIPE, Popen

import numpy as np
from pyiron_vasp.vasp.parser.outcar import Outcar
from pyiron_vasp.vasp.structure import vasp_sorter

from pyiron_atomistics.atomistics.job.interactive import GenericInteractive
from pyiron_atomistics.vasp.base import VaspBase
from pyiron_atomistics.vasp.output import Output

# as of pyiron_atomistics <= 0.5.4 this module defined subclasses that are now removed; the base classes are still
# imported here in case HDF5 files in the wild refer to them.  The imports can be removed on the next big version bump.

__author__ = "Osamu Waseda, Jan Janssen"
__copyright__ = (
    "Copyright 2021, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "1.0"
__maintainer__ = "Jan Janssen"
__email__ = "janssen@mpie.de"
__status__ = "production"
__date__ = "Sep 1, 2018"


class VaspInteractive(VaspBase, GenericInteractive):
    def __init__(self, project, job_name):
        super(VaspInteractive, self).__init__(project, job_name)
        self._interactive_write_input_files = True
        self._interactive_vasprun = None
        self.interactive_flush_frequency = 1

    @property
    def structure(self):
        return GenericInteractive.structure.fget(self)

    @structure.setter
    def structure(self, structure):
        GenericInteractive.structure.fset(self, structure)
        self._reinit_potential_setter(structure=structure)

    @property
    def interactive_enforce_structure_reset(self):
        return self._interactive_enforce_structure_reset

    @interactive_enforce_structure_reset.setter
    def interactive_enforce_structure_reset(self, reset):
        raise NotImplementedError(
            "interactive_enforce_structure_reset() is not implemented!"
        )

    def _get_structure(self, frame=-1, wrap_atoms=True):
        if (
            self.server.run_mode.interactive
            or self.server.run_mode.interactive_non_modal
        ):
            return super(GenericInteractive, self)._get_structure(
                frame=frame, wrap_atoms=wrap_atoms
            )
        else:
            return super(VaspBase, self)._get_structure(
                frame=frame, wrap_atoms=wrap_atoms
            )

    def interactive_close(self):
        if self.interactive_is_activated():
            with open(os.path.join(self.working_directory, "STOPCAR"), "w") as stopcar:
                stopcar.write("LABORT = .TRUE.")  # stopcar.write('LSTOP = .TRUE.')
            try:
                self.run_if_interactive_non_modal()
                self.run_if_interactive_non_modal()
                for atom in self.current_structure.get_scaled_positions():
                    text = " ".join(map("{:19.16f}".format, atom))
                    self._interactive_library.stdin.write(text + "\n")
            except BrokenPipeError:
                self._logger.warning(
                    "VASP calculation exited before interactive_close() - already converged?"
                )
            for key in self.interactive_cache.keys():
                if isinstance(self.interactive_cache[key], list):
                    self.interactive_cache[key] = self.interactive_cache[key]
            super(VaspInteractive, self).interactive_close()
            self.status.collect = True
            self._output_parser = Output()
            if self["vasprun.xml"] is not None:
                self.run()

    def interactive_energy_tot_getter(self):
        return self.interactive_energy_pot_getter()

    def interactive_energy_pot_getter(self):
        if self._interactive_vasprun is not None:
            file_name = os.path.join(self.working_directory, "OUTCAR")
            return self._interactive_vasprun.get_energy_sigma_0(filename=file_name)[-1]
        else:
            return None

    def validate_ready_to_run(self):
        if (
            self.server.run_mode.interactive
            or self.server.run_mode.interactive_non_modal
        ) and "EDIFFG" in self.input.incar._dataset["Parameter"]:
            raise ValueError(
                "If EDIFFG is defined VASP interrupts the interactive run_mode."
            )
        super(VaspInteractive, self).validate_ready_to_run()

    def interactive_forces_getter(self):
        if self._interactive_vasprun is not None:
            file_name = os.path.join(self.working_directory, "OUTCAR")
            forces = self._interactive_vasprun.get_forces(filename=file_name)[-1]
            forces[vasp_sorter(self.structure)] = forces.copy()
            return forces
        else:
            return None

    def interactive_initialize_interface(self):
        if self.executable.executable_path == "":
            self.status.aborted = True
            raise ValueError("No executable set!")
        if self.executable.mpi:
            self._interactive_library = Popen(
                [self.executable.executable_path, str(self.server.cores)],
                stdout=PIPE,
                stdin=PIPE,
                stderr=PIPE,
                cwd=self.working_directory,
                universal_newlines=True,
            )
        else:
            self._interactive_library = Popen(
                self.executable.executable_path,
                stdout=PIPE,
                stdin=PIPE,
                stderr=PIPE,
                cwd=self.working_directory,
                universal_newlines=True,
            )

    def calc_minimize(
        self,
        electronic_steps=400,
        ionic_steps=100,
        max_iter=None,
        pressure=None,
        algorithm=None,
        retain_charge_density=False,
        retain_electrostatic_potential=False,
        ionic_energy_tolerance=None,
        ionic_force_tolerance=None,
        volume_only=False,
        cell_only=False,
    ):
        """
        Function to setup the hamiltonian to perform ionic relaxations using DFT. The ISIF tag has to be supplied
        separately.

        Args:
            electronic_steps (int): Maximum number of electronic steps
            ionic_steps (int): Maximum number of ionic
            max_iter (int): Maximum number of iterations
            pressure (float): External pressure to be applied
            algorithm (str): Type of VASP algorithm to be used "Fast"/"Accurate"
            retain_charge_density (bool): True if the charge density should be written
            retain_electrostatic_potential (boolean): True if the electrostatic potential should be written
            ionic_energy_tolerance (float): Ionic energy convergence criteria (eV)
            ionic_force_tolerance (float): Ionic forces convergence criteria (overwrites ionic energy) (ev/A)
            volume_only (bool): Option to relax only the volume (keeping the relative coordinates fixed
            cell_only (bool): Option to relax only the cell parameters (keeping the relative coordinates fixed)
        """
        if (
            self.server.run_mode.interactive
            or self.server.run_mode.interactive_non_modal
        ):
            raise NotImplementedError(
                "calc_minimize() is not implemented for the interactive mode use calc_static()!"
            )
        else:
            super(VaspInteractive, self).calc_minimize(
                electronic_steps=electronic_steps,
                ionic_steps=ionic_steps,
                max_iter=max_iter,
                pressure=pressure,
                algorithm=algorithm,
                retain_charge_density=retain_charge_density,
                retain_electrostatic_potential=retain_electrostatic_potential,
                ionic_energy_tolerance=ionic_energy_tolerance,
                ionic_force_tolerance=ionic_force_tolerance,
                volume_only=volume_only,
                cell_only=cell_only,
            )

    def calc_md(
        self,
        temperature=None,
        n_ionic_steps=1000,
        n_print=1,
        time_step=1.0,
        retain_charge_density=False,
        retain_electrostatic_potential=False,
        **kwargs,
    ):
        """
        Sets appropriate tags for molecular dynamics in VASP

        Args:
            temperature (int/float/list): Temperature/ range of temperatures in Kelvin
            n_ionic_steps (int): Maximum number of ionic steps
            n_print (int): Prints outputs every n_print steps
            time_step (float): time step (fs)
            retain_charge_density (bool): True id the charge density should be written
            retain_electrostatic_potential (bool): True if the electrostatic potential should be written
        """
        if (
            self.server.run_mode.interactive
            or self.server.run_mode.interactive_non_modal
        ):
            raise NotImplementedError(
                "calc_md() is not implemented for the interactive mode use calc_static()!"
            )
        else:
            super(VaspInteractive, self).calc_md(
                temperature=temperature,
                n_ionic_steps=n_ionic_steps,
                time_step=time_step,
                n_print=n_print,
                retain_charge_density=retain_charge_density,
                retain_electrostatic_potential=retain_electrostatic_potential,
                **kwargs,
            )

    def run_if_interactive_non_modal(self):
        initial_run = self.interactive_is_activated()
        super(VaspInteractive, self).run_if_interactive()
        if initial_run:
            atom_numbers = self.current_structure.get_number_species_atoms()
            for species in atom_numbers.keys():
                indices = self.current_structure.select_index(species)
                for i in indices:
                    text = " ".join(
                        map(
                            "{:19.16f}".format,
                            self.current_structure.get_scaled_positions()[i],
                        )
                    )
                    self._logger.debug("Vasp library: " + text)
                    self._interactive_library.stdin.write(text + "\n")
            self._interactive_library.stdin.flush()
        self._interactive_fetch_completed = False

    def run_if_interactive(self):
        self.run_if_interactive_non_modal()
        self._interactive_check_output()
        self._interactive_vasprun = Outcar()
        self.interactive_collect()

    def interactive_fetch(self):
        if (
            self._interactive_fetch_completed
            and self.server.run_mode.interactive_non_modal
        ):
            print("First run and then fetch")
        else:
            self._interactive_check_output()
            self._interactive_vasprun = Outcar()
            super(VaspInteractive, self).interactive_collect()
            self._logger.debug("interactive run - done")

    def interactive_positions_setter(self, positions):
        pass

    def _check_incar_parameter(self, parameter, value):
        if parameter not in self.input.incar._dataset["Parameter"]:
            self.input.incar[parameter] = value
        if parameter == "NSW":
            self._logger.debug("setting NSW to {}".format(value))
            self.input.incar["NSW"] = value

    def _interactive_check_output(self):
        while self._interactive_library.poll() is None:
            text = self._interactive_library.stdout.readline()
            if "POSITIONS: reading from stdin" in text:
                return

    def _run_if_new(self, debug=False):
        if (
            self.server.run_mode.interactive
            or self.server.run_mode.interactive_non_modal
        ):
            self.interactive_prepare()
        super(VaspInteractive, self)._run_if_new(debug=debug)

    def interactive_prepare(self):
        """
        Modifies/adds tags in the INCAR file that make it possible to run VASP interactively
        """
        self._check_incar_parameter(parameter="INTERACTIVE", value=True)
        self._check_incar_parameter(parameter="IBRION", value=-1)
        self._check_incar_parameter(parameter="POTIM", value=0.0)
        self._check_incar_parameter(parameter="NSW", value=np.iinfo(np.int32).max - 1)
        self._check_incar_parameter(parameter="ISYM", value=0)

    def convergence_check(self):
        """
        We don't care about convergence for interactive jobs! Always returns True

        Returns:
            bool: Always True

        """
        if self.server.run_mode.interactive:
            return True
        else:
            return super().convergence_check()
