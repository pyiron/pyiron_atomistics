# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from ctypes import c_double, c_int
import importlib
import numpy as np
import os
import pandas as pd
import warnings
from scipy import constants

from pyiron_atomistics.lammps.base import LammpsBase, _check_ortho_prism
from pyiron_atomistics.lammps.structure import UnfoldingPrism
from pyiron_atomistics.lammps.control import LammpsControl
from pyiron_atomistics.atomistics.job.interactive import GenericInteractive
from pyiron_atomistics.lammps.wrapper import (
    interactive_lib_command,
    interactive_positions_setter,
    interactive_positions_getter,
    interactive_cells_getter,
    interactive_cells_setter,
    interactive_volume_getter,
    interactive_forces_getter,
    interactive_structure_setter,
    interactive_energy_pot_getter,
    interactive_energy_tot_getter,
    interactive_steps_getter,
    interactive_temperatures_getter,
    interactive_indices_getter,
    interactive_pressures_getter
)


try:  # mpi4py is only supported on Linux and Mac Os X
    from pylammpsmpi import LammpsLibrary
except ImportError:
    pass
from pyiron_atomistics.lammps.units import UnitConverter

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


class LammpsInteractive(LammpsBase, GenericInteractive):
    def __init__(self, project, job_name):
        super(LammpsInteractive, self).__init__(project, job_name)
        self._check_opened = False
        self._interactive_run_command = None
        self._interactive_grand_canonical = True
        self._interactive_water_bonds = False
        self._interactive_mpi_communicator = None
        self._user_fix_external = None
        self._log_file = None
        if "stress" in self.interactive_output_functions.keys():
            del self.interactive_output_functions["stress"]

    @property
    def structure(self):
        return GenericInteractive.structure.fget(self)

    @structure.setter
    def structure(self, structure):
        self._prism = UnfoldingPrism(structure.cell)
        GenericInteractive.structure.fset(self, structure)

    @property
    def interactive_water_bonds(self):
        return self._interactive_water_bonds

    @interactive_water_bonds.setter
    def interactive_water_bonds(self, reset):
        if not isinstance(reset, bool):
            raise AssertionError()
        self._interactive_water_bonds = reset

    @property
    def interactive_mpi_communicator(self):
        return self._interactive_mpi_communicator

    @interactive_mpi_communicator.setter
    def interactive_mpi_communicator(self, comm):
        self._interactive_mpi_communicator = comm

    def _interactive_lib_command(self, command):
        interactive_lib_command(lmp=self._interactive_library, logger=self._logger, command=command)

    def interactive_positions_getter(self):
        uc = UnitConverter(units=self.units)
        positions = interactive_positions_getter(
            lmp=self._interactive_library,
            number_of_atoms=len(self.structure),
            prism=self._prism
        )
        positions = uc.convert_array_to_pyiron_units(positions, label="positions")
        return positions.tolist()

    def interactive_positions_setter(self, positions):
        interactive_positions_setter(
            lmp=self._interactive_library,
            logger=self._logger,
            positions=positions,
            prism=self._prism,
            cores=self.server.cores,
            interactive=self.server.run_mode.interactive
        )

    def interactive_cells_getter(self):
        uc = UnitConverter(units=self.units)
        cc = interactive_cells_getter(lmp=self._interactive_library)
        return uc.convert_array_to_pyiron_units(
            self._prism.unfold_cell(cc), label="cells"
        )

    def interactive_cells_setter(self, cell):
        self._prism = interactive_cells_setter(
            lmp=self._interactive_library,
            logger=self._logger,
            cell=cell,
            structure_current=self._structure_current,
            structure_previous=self._structure_previous
        )

    def interactive_volume_getter(self):
        uc = UnitConverter(units=self.units)
        return uc.convert_array_to_pyiron_units(
            interactive_volume_getter(lmp=self._interactive_library),
            label="volume"
        )

    def interactive_forces_getter(self):
        uc = UnitConverter(units=self.units)
        ff = interactive_forces_getter(
            lmp=self._interactive_library,
            prism=self._prism,
            number_of_atoms=len(self.structure)
        )
        ff = uc.convert_array_to_pyiron_units(ff, label="forces")
        return ff.tolist()

    def interactive_execute(self):
        interactive_lib_command(
            lmp=self._interactive_library,
            logger=self._logger,
            command=self._interactive_run_command
        )

    def _interactive_lammps_input(self):
        del self.input.control["dump___1"]
        del self.input.control["dump_modify___1"]
        for key, value in zip(
            self.input.control.dataset["Parameter"], self.input.control.dataset["Value"]
        ):
            if key in [
                "read_data",
                "units",
                "dimension",
                "boundary",
                "atom_style",
                "atom_modify",
                "include",
                "run",
                "minimize",
            ]:
                continue
            else:
                self._interactive_lib_command(
                    " ".join(key.split(self.input.control.multi_word_separator))
                    + " "
                    + str(value)
                )

    def _interactive_set_potential(self):
        potential_lst = []
        if self.input.potential.files is not None:
            for potential in self.input.potential.files:
                if not os.path.exists(potential):
                    raise ValueError("Potential not found: ", potential)
                potential_lst.append([potential.split("/")[-1], potential])

        style_full = self.input.control["atom_style"] == "full"
        for line in self.input.potential.get_string_lst():
            for potential in potential_lst:
                if " " + potential[0] in line:
                    line = line.replace(" " + potential[0], " " + potential[1])
            # Don't write the kspace_style or pair style commands if the atom style is "full"
            if not (style_full and ("kspace" in line or "pair" in line)):
                self._interactive_lib_command(line.split("\n")[0])
            if len(potential_lst) == 0:
                self._interactive_lib_command(line.split("\n")[0])
        if style_full and self._interactive_water_bonds:
            # Currently supports only water molecules. Please feel free to expand this
            self._interactive_water_setter()

    def _executable_activate_mpi(self):
        if (
            self.server.run_mode.interactive
            or self.server.run_mode.interactive_non_modal
        ):
            pass
        else:
            super(LammpsInteractive, self)._executable_activate_mpi()

    def _reset_interactive_run_command(self):
        df = pd.DataFrame(self.input.control.dataset)
        self._interactive_run_command = " ".join(df.T[df.index[-1]].values)

    def interactive_initialize_interface(self):
        self._create_working_directory()
        if self.server.run_mode.interactive and self.server.cores == 1:
            lammps = getattr(importlib.import_module("lammps"), "lammps")
            if self._log_file is None:
                self._log_file = os.path.join(self.working_directory, "log.lammps")
            self._interactive_library = lammps(
                cmdargs=["-screen", "none", "-log", self._log_file],
                comm=self._interactive_mpi_communicator,
            )
        else:
            self._interactive_library = LammpsLibrary(
                cores=self.server.cores, working_directory=self.working_directory
            )
        if not all(self.structure.pbc):
            self.input.control["boundary"] = " ".join(
                ["p" if coord else "f" for coord in self.structure.pbc]
            )
        self._reset_interactive_run_command()
        self.interactive_structure_setter(self.structure)

    def set_fix_external(
        self, function, n_call=1, n_apply=1, overload_internal_fix_external=False
    ):
        """
        **********************
        *** Expert feature ***
        **********************

        Set fix_external function that modifies forces.

        Args:
            function (function): User-defined function that returns forces (see below)
            n_call (int): Make `fix_external` every `n_call` steps (default: 1)
            n_apply (int): Apply `fix_external` forces every `n_apply` steps (default: 1)
            overload_internal_fix_external (bool): Whether to overload internal fix_external
                (see below).

        `function` must have the following form:

        ```
        def function(positions, ntimestep, nlocal):
            your_evaluation
            return forces
        ```
        where `positions` is the positions of all atoms on the local processor, `ntimestep` is the
        current timestep and `nlocal` is the number of atoms on the current processor. `forces`
        must be of the shape `(n_atoms, 3)`. The total translational force will be eliminated
        inside pyiron. The fix then adds these forces to each atom in the box, once every
        `n_apply` steps, similar to the way the fix addforce command works. Note that if
        `n_call` > `n_apply`, the force values produced by one callback will persist, and be used
        multiple times to update atom forces.

        Example: Add random forces

        ```
        from pyiron import Project

        def random_forces(x, *args):
            return 0.1 * np.random.randn(*x.shape)

        pr = Project('RANDOM')
        lmp = pr.create.job.Lammps('random_forces')
        lmp.structure = your_structure
        lmp.potential = your_potential
        lmp.interactive_open()
        lmp.set_fix_external(random_forces)
        lmp.calc_md()
        lmp.run()
        lmp.interactive_close()
        ```

        *** Super-expert feature: `overload_internal_fix_external = True` ***

        If `overload_internal_fix_external` is set to `True`, then `function` must have the
        following form:

        ```
        def function(ptr, timestep, nlocal, ids, x, fexternal):
            your_evaluation
        ```

        with the following arguemnts:

        - `ptr`: pointer provided by and simply passed back to external driver
        - `timestep`: current LAMMPS timestep
        - `nlocal`: # of atoms on this processor
        - `ids`: list of atom IDs on this processor
        - `x`: coordinates of atoms on this processor
        - `fexternal`: forces to add to atoms on this processor

        Overloading the internal fix_external function will have the advantage that the code will
        not need to do expensive copying, BUT it is extremely error-prone. Make sure that the
        code works without overloading the internal fix_external function first.

        Note: Do NOT overwrite `fexternal`, because it points to the internal memory of LAMMPS and
        therefore overwriting it will erase its functionality. E.g. DO `fexternal.fill(0)` and NOT
        `fexternal = np.zeros_like(x)`.
        """
        if not self.server.run_mode.interactive:
            raise AssertionError("Callback works only in interactive mode")
        self._user_fix_external = _FixExternal(function)
        self.input.control[
            "fix___fix_external"
        ] = "all external pf/callback {} {}".format(n_call, n_apply)
        if overload_internal_fix_external:
            self._user_fix_external.fix_external = function

    def calc_minimize(
        self,
        ionic_energy_tolerance=0.0,
        ionic_force_tolerance=1.0e-4,
        e_tol=None,
        f_tol=None,
        max_iter=100000,
        pressure=None,
        n_print=100,
        style="cg",
    ):
        # Docstring set programmatically -- Please ensure that changes to signature or defaults stay consistent!
        if e_tol is not None:
            warnings.warn(
                "e_tol is deprecated as of vers. 0.3.0. It is not guaranteed to be in service in vers. 0.4.0. Use ionic_energy_tolerance instead."
            )
            ionic_energy_tolerance = e_tol
            e_tol = None
        if f_tol is not None:
            warnings.warn(
                "f_tol is deprecated as of vers. 0.3.0. It is not guaranteed to be in service in vers. 0.4.0. Use ionic_force_tolerance instead."
            )
            ionic_force_tolerance = f_tol
            f_tol = None
        if self.server.run_mode.interactive_non_modal:
            warnings.warn(
                "calc_minimize() is not implemented for the non modal interactive mode use calc_static()!"
            )
        super(LammpsInteractive, self).calc_minimize(
            ionic_energy_tolerance=ionic_energy_tolerance,
            ionic_force_tolerance=ionic_force_tolerance,
            max_iter=max_iter,
            pressure=pressure,
            n_print=n_print,
            style=style,
        )
        if self.interactive_is_activated() and (
            self.server.run_mode.interactive
            or self.server.run_mode.interactive_non_modal
        ):
            self.interactive_structure_setter(self.structure)

    calc_minimize.__doc__ = LammpsControl.calc_minimize.__doc__

    def calc_md(
        self,
        temperature=None,
        pressure=None,
        n_ionic_steps=1000,
        time_step=1.0,
        n_print=100,
        temperature_damping_timescale=100.0,
        pressure_damping_timescale=1000.0,
        seed=None,
        tloop=None,
        initial_temperature=None,
        langevin=False,
        delta_temp=None,
        delta_press=None,
    ):
        super(LammpsInteractive, self).calc_md(
            temperature=temperature,
            pressure=pressure,
            n_ionic_steps=n_ionic_steps,
            time_step=time_step,
            n_print=n_print,
            temperature_damping_timescale=temperature_damping_timescale,
            pressure_damping_timescale=pressure_damping_timescale,
            seed=seed,
            tloop=tloop,
            initial_temperature=initial_temperature,
            langevin=langevin,
            delta_temp=delta_temp,
            delta_press=delta_press,
        )
        if self.interactive_is_activated() and (
            self.server.run_mode.interactive
            or self.server.run_mode.interactive_non_modal
        ):
            self.interactive_structure_setter(self.structure)

    def run_if_interactive(self):
        if self._generic_input["calc_mode"] in ["md", "vcsgc"]:
            self.input.control["run"] = self._generic_input["n_print"]
            super(LammpsInteractive, self).run_if_interactive()
            self._reset_interactive_run_command()
            if self._user_fix_external is not None:
                self._interactive_library.set_fix_external_callback(
                    "fix_external", self._user_fix_external.fix_external
                )
            counter = 0
            iteration_max = int(
                self._generic_input["n_ionic_steps"] / self._generic_input["n_print"]
            )
            while counter < iteration_max:
                self.interactive_execute()
                self.interactive_collect()
                counter += 1

        else:
            super(LammpsInteractive, self).run_if_interactive()
            self.interactive_execute()
            self.interactive_collect()

    def validate_ready_to_run(self):
        if (
            self.server.run_mode.interactive
            and self._generic_input["calc_mode"] in ["md", "vcsgc"]
            and "fix___langevin" in self.input.control.keys()
        ):
            warnings.warn(
                "Langevin thermostatted MD in interactive mode only gives correct physics in the limit that the "
                "n_print variable goes to infinity. A more in-depth discussion can be found "
                "[here](https://github.com/pyiron/pyiron/issues/1173).",
                stacklevel=2,
            )
        super().validate_ready_to_run()

    def run_if_interactive_non_modal(self):
        if not self._interactive_fetch_completed:
            print("Warning: interactive_fetch being effectuated")
            self.interactive_fetch()
        super(LammpsInteractive, self).run_if_interactive()
        self.interactive_execute()
        self._interactive_fetch_completed = False

    def interactive_fetch(self):
        if (
            self._interactive_fetch_completed
            and self.server.run_mode.interactive_non_modal
        ):
            print("First run and then fetch")
        else:
            self.interactive_collect()
            self._logger.debug("interactive run - done")

    def interactive_structure_setter(self, structure):
        self._prism, control_dict = interactive_structure_setter(
            lmp=self._interactive_library,
            logger=self._logger,
            structure_current=structure,
            structure_previous=self.structure,
            units=self.input.control["units"],
            dimension=self.input.control["dimension"],
            boundary=self.input.control["boundary"],
            atom_style=self.input.control["atom_style"],
            el_eam_lst=self.input.potential.get_element_lst(),
            calc_md=self._generic_input["calc_mode"] == "md",
            interactive=self.server.run_mode.interactive,
            cores=self.server.cores
        )
        for key, value in control_dict.items():
            self.input.control[key.replace(" ", "___")] = value
        self._interactive_lammps_input()
        self._interactive_set_potential()

    def _interactive_water_setter(self):
        """
        This function writes the bonds for water molecules present in the structure. It is assumed that only intact
        water molecules are present and the H atoms are within 1.3 $\AA$ of each O atom. Once the neighbor list is
        generated, the bonds and angles are created. This function needs to be generalized/extended to account for
        dissociated water. This function can also be used as an example to create bonds between other molecules.
        """
        neighbors = self.structure.get_neighbors(cutoff_radius=1.3)
        o_indices = self.structure.select_index("O")
        h_indices = self.structure.select_index("H")
        h1_indices = np.intersect1d(
            np.vstack(neighbors.indices[o_indices])[:, 0], h_indices
        )
        h2_indices = np.intersect1d(
            np.vstack(neighbors.indices[o_indices])[:, 1], h_indices
        )
        o_ind_str = (
            np.array2string(o_indices + 1).replace("[", "").replace("]", "").strip()
        )
        h1_ind_str = (
            np.array2string(h1_indices + 1).replace("[", "").replace("]", "").strip()
        )
        h2_ind_str = (
            np.array2string(h2_indices + 1).replace("[", "").replace("]", "").strip()
        )
        group_o = "group Oatoms id {}".format(o_ind_str).replace("  ", " ")
        group_h1 = "group H1atoms id {}".format(h1_ind_str).replace("  ", " ")
        group_h2 = "group H2atoms id {}".format(h2_ind_str).replace("  ", " ")
        self._interactive_lib_command(group_o)
        self._interactive_lib_command(group_h1)
        self._interactive_lib_command(group_h2)
        # A dummy pair style that does not have any Coulombic interactions needs to be initialized to create the bonds
        self._interactive_lib_command("kspace_style none")
        self._interactive_lib_command("pair_style lj/cut 2.5")
        self._interactive_lib_command("pair_coeff * * 0.0 0.0")
        self._interactive_lib_command("create_bonds many Oatoms H1atoms 1 0.7 1.4")
        self._interactive_lib_command("create_bonds many Oatoms H2atoms 1 0.7 1.4")
        for i, o_ind in enumerate(o_indices):
            self._interactive_lib_command(
                "create_bonds single/angle 1 {} {} {}".format(
                    int(h1_indices[i]) + 1, int(o_ind) + 1, int(h2_indices[i]) + 1
                )
            )
        # Now the actual pair styles are written
        self._interactive_lib_command(
            "pair_style " + self.input.potential["pair_style"]
        )
        values = np.array(self.input.potential._dataset["Value"])
        pair_val = values[
            ["pair_coeff" in val for val in self.input.potential._dataset["Parameter"]]
        ]
        for val in pair_val:
            self._interactive_lib_command("pair_coeff " + val)
        self._interactive_lib_command(
            "kspace_style " + self.input.potential["kspace_style"]
        )

    def from_hdf(self, hdf=None, group_name=None):
        """
        Recreates instance from the hdf5 file

        Args:
            hdf (str): Path to the hdf5 file
            group_name (str): Name of the group which contains the object
        """
        super(LammpsInteractive, self).from_hdf(hdf=hdf, group_name=group_name)
        self.species_from_hdf()

    def collect_output(self):
        if (
            self.server.run_mode.interactive
            or self.server.run_mode.interactive_non_modal
        ):
            pass
        else:
            super(LammpsInteractive, self).collect_output()

    def update_potential(self):
        self._interactive_lib_command(self.potential.Config[0][0])
        self._interactive_lib_command(self.potential.Config[0][1])

    def interactive_indices_getter(self):
        uc = UnitConverter(units=self.units)
        lammps_indices = interactive_indices_getter(lmp=self._interactive_library)
        indices = uc.convert_array_to_pyiron_units(
            self.remap_indices(lammps_indices), label="indices"
        )
        return indices.tolist()

    def interactive_indices_setter(self, indices):
        el_struct_lst = self._structure_current.get_species_symbols()
        el_obj_lst = self._structure_current.get_species_objects()
        el_eam_lst = self.input.potential.get_element_lst()
        el_dict = {}
        for id_eam, el_eam in enumerate(el_eam_lst):
            if el_eam in el_struct_lst:
                id_el = list(el_struct_lst).index(el_eam)
                el = el_obj_lst[id_el]
                el_dict[el] = id_eam + 1
        elem_all = np.array(
            [el_dict[self._structure_current.species[el]] for el in indices]
        )
        if self.server.run_mode.interactive and self.server.cores == 1:
            self._interactive_library.scatter_atoms(
                "type", 0, 1, (len(elem_all) * c_int)(*elem_all)
            )
        else:
            self._interactive_library.scatter_atoms("type", elem_all)

    def interactive_energy_pot_getter(self):
        uc = UnitConverter(units=self.units)
        return uc.convert_array_to_pyiron_units(
            interactive_energy_pot_getter(lmp=self._interactive_library),
            label="energy_pot"
        )

    def interactive_energy_tot_getter(self):
        uc = UnitConverter(units=self.units)
        return uc.convert_array_to_pyiron_units(
            interactive_energy_tot_getter(lmp=self._interactive_library),
            label="energy_tot"
        )

    def interactive_steps_getter(self):
        uc = UnitConverter(units=self.units)
        return uc.convert_array_to_pyiron_units(
            interactive_steps_getter(lmp=self._interactive_library),
            label="steps"
        )

    def interactive_temperatures_getter(self):
        uc = UnitConverter(units=self.units)
        return uc.convert_array_to_pyiron_units(
            interactive_temperatures_getter(lmp=self._interactive_library), \
            label="temperature"
        )

    def interactive_stress_getter(self):
        """
        This gives back an Nx3x3 array of stress/atom defined in http://lammps.sandia.gov/doc/compute_stress_atom.html
        Keep in mind that it is stress*volume in eV. Further discussion can be found on the website above.

        Returns:
            numpy.array: Nx3x3 np array of stress/atom
        """
        if not "stress" in self.interactive_cache.keys():
            self._interactive_lib_command("compute st all stress/atom NULL")
            self._interactive_lib_command("run 0")
            self.interactive_cache["stress"] = []
        id_lst = self._interactive_library.extract_atom("id", 0)
        id_lst = np.array([id_lst[i] for i in range(len(self.structure))]) - 1
        id_lst = np.arange(len(id_lst))[np.argsort(id_lst)]
        ind = np.array([0, 3, 4, 3, 1, 5, 4, 5, 2])
        ss = self._interactive_library.extract_compute("st", 1, 2)
        ss = np.array(
            [ss[i][j] for i in range(len(self.structure)) for j in range(6)]
        ).reshape(-1, 6)[id_lst]
        ss = (
            ss[:, ind].reshape(len(self.structure), 3, 3)
            / constants.eV
            * constants.bar
            * constants.angstrom**3
        )
        if _check_ortho_prism(prism=self._prism):
            ss = np.einsum("ij,njk->nik", self._prism.R, ss)
            ss = np.einsum("nij,kj->nik", ss, self._prism.R)
        return ss

    def interactive_pressures_getter(self):
        uc = UnitConverter(units=self.units)
        pp = interactive_pressures_getter(lmp=self._interactive_library, prism=self._prism)
        return uc.convert_array_to_pyiron_units(pp, label="pressure")

    def interactive_close(self):
        if self.interactive_is_activated():
            self._interactive_library.close()
            super(LammpsInteractive, self).interactive_close()
            with self.project_hdf5.open("output") as h5:
                if "interactive" in h5.list_groups():
                    for key in h5["interactive"].list_nodes():
                        h5["generic/" + key] = h5["interactive/" + key]


class _FixExternal:
    """
    Helper class to exploit `fix external`, which is one of the features of LAMMPS to modify
    force of each atom. More info can be found on https://docs.lammps.org/fix_external.html

    Inside pyiron, `fix_external()` is passed to LAMMPS, which is to be called during run. The
    function arguments are given by LAMMPS -> DO NOT modify them.
    """

    def __init__(self, function):
        self.function = function

    def fix_external(self, ptr, ntimestep, nlocal, tag, x, fext):
        """
        Args:
            ptr: pointer provided by and simply passed back to external driver
            timestep (int): current LAMMPS timestep
            nlocal (numpy.ndarray): # of atoms on this processor
            ids (numpy.ndarray): list of atom IDs on this processor
            x (numpy.ndarray): coordinates of atoms on this processor
            fexternal (numpy.ndarray): forces to add to atoms on this processor

        `fexternal` points to the forces to be added in LAMMPS, i.e. its content can be modified
        but not overwritten.
        """
        tags = tag.flatten().argsort()
        fext.fill(0)
        fext[tags] += self.function(x[tags], ntimestep, nlocal)
        fext -= np.mean(fext, axis=0)
