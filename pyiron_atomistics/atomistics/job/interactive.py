# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from collections import defaultdict

import numpy as np
from pyiron_base import InteractiveBase, state

from pyiron_atomistics.atomistics.job.atomistic import (
    AtomisticGenericJob,
    GenericOutput,
)
from pyiron_atomistics.atomistics.structure.periodic_table import PeriodicTable

__author__ = "Osamu Waseda, Jan Janssen"
__copyright__ = (
    "Copyright 2021, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "1.0"
__maintainer__ = "Jan Janssen"
__email__ = "janssen@mpie.de"
__status__ = "development"
__date__ = "Sep 1, 2017"


class GenericInteractive(AtomisticGenericJob, InteractiveBase):
    def __init__(self, project, job_name):
        super(GenericInteractive, self).__init__(project, job_name)
        self.output = GenericInteractiveOutput(job=self)
        self._structure_previous = None
        self._structure_current = None
        self._interactive_enforce_structure_reset = False
        self._interactive_grand_canonical = False
        self._interactive_fetch_completed = True
        self._interactive_species_lst = np.array([])
        self._periodic_table = PeriodicTable()
        self.interactive_input_functions = {
            "index": self.interactive_index_organizer,
            "cell": self.interactive_cell_organizer,
            "positions": self.interactive_positions_organizer,
            "magnetic_moments": self.interactive_magmom_organizer,
        }
        self.interactive_output_functions = {
            "cells": self.interactive_cells_getter,
            "energy_pot": self.interactive_energy_pot_getter,
            "energy_tot": self.interactive_energy_tot_getter,
            "forces": self.interactive_forces_getter,
            "positions": self.interactive_positions_getter,
            "pressures": self.interactive_pressures_getter,
            "stress": self.interactive_stress_getter,
            "steps": self.interactive_steps_getter,
            "temperature": self.interactive_temperatures_getter,
            "indices": self.interactive_indices_getter,
            "computation_time": self.interactive_computation_time_getter,
            "unwrapped_positions": self.interactive_unwrapped_positions_getter,
            "atom_spin_constraints": self.interactive_atom_spin_constraints_getter,
            "atom_spins": self.interactive_atom_spins_getter,
            "magnetic_forces": self.interactive_magnetic_forces_getter,
            "volume": self.interactive_volume_getter,
        }
        self.interactive_cache = defaultdict(list)

    @property
    def interactive_enforce_structure_reset(self):
        return self._interactive_enforce_structure_reset

    @interactive_enforce_structure_reset.setter
    def interactive_enforce_structure_reset(self, reset):
        if not isinstance(reset, bool):
            raise AssertionError()
        self._interactive_enforce_structure_reset = reset

    @property
    def initial_structure(self):
        return AtomisticGenericJob.structure.fget(self)

    @property
    def current_structure(self):
        return self.structure

    @current_structure.setter
    def current_structure(self, structure):
        self.structure = structure

    @property
    def structure(self):
        if self._structure_current is not None:
            return self._structure_current
        elif (
            self.server.run_mode.interactive
            or self.server.run_mode.interactive_non_modal
        ):
            self._structure_current = AtomisticGenericJob.structure.fget(self)
            return self._structure_current
        else:
            return AtomisticGenericJob.structure.fget(self)

    @structure.setter
    def structure(self, structure):
        if (
            self.server.run_mode.interactive
            or self.server.run_mode.interactive_non_modal
        ):
            # only overwrite the initial structure if it is not set already.
            if AtomisticGenericJob.structure.fget(self) is None:
                AtomisticGenericJob.structure.fset(self, structure.copy())
            self._structure_current = structure
        else:
            AtomisticGenericJob.structure.fset(self, structure)

    def species_from_hdf(self):
        if (
            "output" in self.project_hdf5.list_groups()
            and "interactive" in self.project_hdf5["output"].list_groups()
            and "species" in self.project_hdf5["output/interactive"].list_nodes()
        ):
            with self.project_hdf5.open("output/interactive") as hdf:
                self._interactive_species_lst = np.array(hdf["species"])

    def run_if_interactive(self):
        self.status.running = True
        if self.structure is None:
            raise ValueError("Input structure not set. Use method set_structure()")
        if not self.interactive_is_activated():
            self.interactive_initialize_interface()
        if self._structure_previous is None:
            self._structure_previous = self.structure.copy()
        if self._structure_current is not None:
            if (
                len(self._structure_current) != len(self._structure_previous)
                and not self._interactive_grand_canonical
            ):
                raise ValueError(
                    "The number of atoms changed, this is currently not supported!"
                )
            if not self._interactive_enforce_structure_reset:
                functions_to_execute = list(self.interactive_input_functions.values())
                for v in functions_to_execute:
                    v()
            else:
                self._logger.debug("Generic library: structure changed!")
                self.interactive_structure_setter(self._structure_current)
        self._update_previous_structure()

    def interactive_index_organizer(self):
        index_merge_lst = self._interactive_species_lst.tolist() + list(
            self._structure_current.get_species_symbols()
        )
        el_lst = sorted(set(index_merge_lst), key=index_merge_lst.index)
        current_structure_index = [
            el_lst.index(el) for el in self._structure_current.get_chemical_symbols()
        ]
        previous_structure_index = [
            el_lst.index(el) for el in self._structure_previous.get_chemical_symbols()
        ]
        if not np.array_equal(
            np.array(current_structure_index),
            np.array(previous_structure_index),
        ):
            self._logger.debug("Generic library: indices changed!")
            self.interactive_indices_setter(self._structure_current.indices)

    def interactive_cell_organizer(self):
        if self._generic_input["calc_mode"] != "static" or not np.allclose(
            self._structure_current.cell,
            self._structure_previous.cell,
            rtol=1e-15,
            atol=1e-15,
        ):
            self._logger.debug("Generic library: cell changed!")
            try:
                self.interactive_cells_setter(self._structure_current.cell)
            except NotImplementedError:
                del self.interactive_input_functions["cell"]

    def interactive_positions_organizer(self):
        if (
            self._generic_input["calc_mode"] != "static"
            or len(self._structure_current.positions)
            != len(self._structure_previous.positions)
            or not np.allclose(
                self._structure_current.positions,
                self._structure_previous.positions,
                rtol=1e-15,
                atol=1e-15,
            )
        ):
            self._logger.debug("Generic library: positions changed!")
            self.interactive_positions_setter(self._structure_current.positions)

    def interactive_magmom_organizer(self):
        if all(
            mm is None for mm in self._structure_current.get_initial_magnetic_moments()
        ):
            del self.interactive_input_functions["magnetic_moments"]
        elif (
            None in self._structure_previous.get_initial_magnetic_moments()
            or len(self._structure_current.get_initial_magnetic_moments())
            != len(self._structure_previous.get_initial_magnetic_moments())
            or not np.allclose(
                self._structure_current.get_initial_magnetic_moments(),
                self._structure_previous.get_initial_magnetic_moments(),
            )
        ):
            self._logger.debug("Generic library: magnetic moments changed!")
            try:
                self.interactive_spin_constraints_setter(
                    self._structure_current.get_initial_magnetic_moments()
                )
            except NotImplementedError:
                del self.interactive_input_functions["magnetic_moments"]

    def interactive_cells_getter(self):
        return self.initial_structure.cell

    def interactive_collect(self):
        del_key_lst = []
        for k, v in self.interactive_output_functions.items():
            try:
                value = v()
                if value is not None:
                    self.interactive_cache[k].append(value)
                else:
                    del_key_lst.append(k)
            except NotImplementedError:
                del_key_lst.append(k)
        for k in del_key_lst:
            del self.interactive_output_functions[k]
        if (
            len(list(self.interactive_cache.keys())) > 0
            and len(self.interactive_cache[list(self.interactive_cache.keys())[0]])
            % self._interactive_flush_frequency
            == 0
        ):
            self.interactive_flush(path="interactive")
        if self.server.run_mode.interactive_non_modal:
            self._interactive_fetch_completed = True

    def interactive_flush(self, path="interactive", include_last_step=False):
        """

        Args:
            path:
            include_last_step:

        Returns:

        """
        with self.project_hdf5.open("output") as hdf_output:
            with hdf_output.open(path) as hdf:
                hdf["species"] = self._interactive_species_lst.tolist()
        super(GenericInteractive, self).interactive_flush(
            path=path, include_last_step=include_last_step
        )

    def interactive_indices_getter(self):
        species_symbols = np.array(
            [e.Abbreviation for e in self.current_structure.species]
        )
        self._interactive_species_lst = self._extend_species_elements(
            struct_species=species_symbols, species_array=self._interactive_species_lst
        )
        index_merge_lst = self._interactive_species_lst.tolist() + list(
            self._structure_current.get_species_symbols()
        )
        el_lst = sorted(set(index_merge_lst), key=index_merge_lst.index)
        current_structure_index = np.array(
            [el_lst.index(el) for el in self._structure_current.get_chemical_symbols()]
        )
        return current_structure_index

    def interactive_positions_getter(self):
        return self.current_structure.positions

    def interactive_steps_getter(self):
        return len(self.interactive_cache[list(self.interactive_cache.keys())[0]])

    def interactive_time_getter(self):
        return self.interactive_steps_getter()

    def interactive_volume_getter(self):
        return self.initial_structure.get_volume()

    def _update_previous_structure(self):
        """Update the previous structure to the last step configuration."""
        self._structure_previous = self.structure.copy()

    @staticmethod
    def _extend_species_elements(struct_species, species_array):
        if not all(np.isin(struct_species, species_array)):
            new_elements_index = np.invert(np.isin(struct_species, species_array))
            species_array = np.append(species_array, struct_species[new_elements_index])
        return species_array

    # Functions which have to be implemented by the fin
    def interactive_cells_setter(self, cell):
        raise NotImplementedError("interactive_cells_getter() is not implemented!")

    def interactive_energy_pot_getter(self):
        raise NotImplementedError("interactive_energy_pot_getter() is not implemented!")

    def interactive_energy_tot_getter(self):
        raise NotImplementedError("interactive_energy_tot_getter() is not implemented!")

    def interactive_forces_getter(self):
        raise NotImplementedError("interactive_forces_getter() is not implemented!")

    def interactive_indices_setter(self, indices):
        raise NotImplementedError("interactive_indices_setter() is not implemented!")

    def interactive_spins_getter(self):
        raise NotImplementedError("interactive_spins_getter() is not implemented!")

    def interactive_spin_constraints_getter(self):
        raise NotImplementedError(
            "interactive_spin_constraints_getter() is not implemented!"
        )

    def interactive_atom_spin_constraints_getter(self):
        raise NotImplementedError(
            "interactive_atom_spin_constraints_getter() is not implemented!"
        )

    def interactive_atom_spins_getter(self):
        raise NotImplementedError("interactive_atom_spins_getter() is not implemented!")

    def interactive_magnetic_forces_getter(self):
        raise NotImplementedError(
            "interactive_magnetic_forces_getter() is not implemented!"
        )

    def interactive_spin_constraints_setter(self, spins):
        raise NotImplementedError(
            "iinteractive_spin_constraints_setter() is not implemented!"
        )

    def interactive_initialize_interface(self):
        raise NotImplementedError(
            "interactive_initialize_interface() is not implemented!"
        )

    def interactive_positions_setter(self, positions):
        raise NotImplementedError("interactive_positions_setter() is not implemented!")

    def interactive_pressures_getter(self):
        raise NotImplementedError("interactive_pressures_getter() is not implemented!")

    def interactive_stress_getter(self):
        raise NotImplementedError("interactive_stress_getter() is not implemented!")

    def interactive_structure_setter(self, structure):
        raise NotImplementedError("interactive_structure_setter() is not implemented!")

    def interactive_computation_time_getter(self):
        raise NotImplementedError(
            "interactive_computation_time_getter() is not implemented!"
        )

    def interactive_temperatures_getter(self):
        raise NotImplementedError(
            "interactive_temperatures_getter() is not implemented!"
        )

    def interactive_unwrapped_positions_getter(self):
        raise NotImplementedError(
            "interactive_unwrapped_positions_getter() is not implemented!"
        )


class GenericInteractiveOutput(GenericOutput):
    def __init__(self, job):
        super(GenericInteractiveOutput, self).__init__(job=job)

    def _key_from_cache(self, key):
        """
        Get all entries from the interactive cache for a specific key.

        Args:
            key (str): name of the key

        Returns:
            list: list of values stored in the interactive cache
        """
        if (
            key in self._job.interactive_cache.keys()
            and self._job.interactive_is_activated()
            and len(self._job.interactive_cache[key]) != 0
        ):
            return self._job.interactive_cache[key]
        else:
            return []

    def _lst_from_cache(self, key):
        """

        Args:
            key (str): name of the key

        Returns:
            list: list of arrays stored in the interactive cache
        """
        lst = self._key_from_cache(key)
        if len(lst) != 0 and isinstance(lst[-1], list):
            return [np.array(out) for out in lst]
        else:
            return lst

    def _key_from_hdf(self, key):
        """
        Get all entries from the HDF5 file for a specific key - stored under 'output/interactive/<key>'.  If not found
        there the key is looked up in the regular HDF storage location 'output/<key>' via the property on our super
        class :class:`.GenericOutput`.

        Args:
            key (str): name of the key

        Returns:

        """
        try:
            fetched = self._job.project_hdf5["output/interactive/" + key]
        except ValueError:
            fetched = None
        if fetched is None or len(fetched) == 0:
            fetched = getattr(super(), key)
        return fetched

    def _key_from_property(self, key, prop):
        """
        Fetch values for the given and property from interactive HDF5.

        Values are first looked up in the interactive cache, then in the interactive group in the HDF5 file
        ('output/interactive/<key>') and if not found there via the given prop function.

        Args:
            key (str): name of the key
            prop (function):

        Returns:

        """
        return_lst = self._key_from_cache(key)
        hdf5_output = self._key_from_hdf(key)
        if hdf5_output is not None:
            return np.concatenate([hdf5_output, *return_lst])
        else:
            prop_result = prop(self)
            if prop_result is not None:
                return np.concatenate([prop_result, *return_lst])
            else:
                return np.asarray(return_lst)

    def _lst_from_property(self, key):
        """
        Fetch latest values for the given property.

        Values are first looked up in the interactive cache, then in the interactive group in the HDF5 file
        ('output/interactive/<key>') and if not found there via approriate getter from our super class (GenericOutput).
        That means that key needs to be a property defined there.

        Args:
            key (str): output key

        Returns:
            :class:`numpy.ndarray`: collected values from all previous steps
        """
        cached = np.array(self._lst_from_cache(key))
        fetched = self._key_from_hdf(key)
        if fetched is None or len(fetched) == 0:
            return cached
        elif len(cached) == 0:
            return fetched
        else:
            return np.concatenate((cached, fetched))

    @property
    def indices(self):
        return self._lst_from_property("indices")

    @property
    def cells(self):
        return self._key_from_property(key="cells", prop=GenericOutput.cells.fget)

    @property
    def energy_pot(self):
        return self._key_from_property(
            key="energy_pot", prop=GenericOutput.energy_pot.fget
        )

    @property
    def energy_tot(self):
        return self._key_from_property(
            key="energy_tot", prop=GenericOutput.energy_tot.fget
        )

    @property
    def forces(self):
        return self._lst_from_property("forces")

    @property
    def positions(self):
        return self._lst_from_property("positions")

    @property
    def pressures(self):
        return self._key_from_property(
            key="pressures", prop=GenericOutput.pressures.fget
        )

    @property
    def steps(self):
        return self._key_from_property(key="steps", prop=GenericOutput.steps.fget)

    @property
    def temperature(self):
        return self._key_from_property(
            key="temperature", prop=GenericOutput.temperature.fget
        )

    @property
    def time(self):
        return self._key_from_property(
            key="computation_time", prop=GenericOutput.computation_time.fget
        )

    @property
    def unwrapped_positions(self):
        return self._lst_from_property("unwrapped_positions")

    @property
    def volume(self):
        return self._key_from_property(key="volume", prop=GenericOutput.volume.fget)

    def __dir__(self):
        return list(
            set(
                list(self._job.interactive_cache.keys())
                + super(GenericOutput).__dir__()
            )
        )


class InteractiveInterface(object):
    def __init__(self):
        self._logger = state.logger

    def get_cell(self):
        raise NotImplementedError

    def set_cell(self, cell):
        raise NotImplementedError

    def get_temperature(self):
        raise NotImplementedError

    def set_temperature(self, temperature):
        raise NotImplementedError

    def get_positions(self):
        raise NotImplementedError

    def set_positions(self, positions):
        raise NotImplementedError

    def get_forces(self):
        raise NotImplementedError

    def set_forces(self, forces):
        raise NotImplementedError

    def get_energy_tot(self):
        raise NotImplementedError

    def set_energy_tot(self, energy_tot):
        raise NotImplementedError

    def get_energy_pot(self):
        raise NotImplementedError

    def set_energy_pot(self, energy_pot):
        raise NotImplementedError

    def get_pressure(self):
        raise NotImplementedError

    def set_pressure(self, pressure):
        raise NotImplementedError

    def run_interactive(self):
        raise NotImplementedError
