# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import numpy as np
from pyiron_atomistics.atomistics.job.interactive import GenericInteractive
from pyiron_atomistics.atomistics.structure.atoms import Atoms

try:
    from ase.cell import Cell
except ImportError:
    Cell = None

__author__ = "Jan Janssen"
__copyright__ = (
    "Copyright 2021, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "1.0"
__maintainer__ = "Jan Janssen"
__email__ = "janssen@mpie.de"
__status__ = "development"
__date__ = "Sep 1, 2018"


class AseJob(GenericInteractive):
    def __init__(self, project, job_name):
        super(AseJob, self).__init__(project, job_name)

        self.__version__ = (
            None  # Reset the version number to the executable is set automatically
        )

    def to_hdf(self, hdf=None, group_name=None):
        super(AseJob, self).to_hdf(hdf=hdf, group_name=group_name)
        self._structure_to_hdf()

    def from_hdf(self, hdf=None, group_name=None):
        super(AseJob, self).from_hdf(hdf=hdf, group_name=group_name)
        self._structure_from_hdf()

    def run_static(self):
        pre_run_mode = self.server.run_mode
        self.server.run_mode.interactive = True
        self.run_if_interactive()
        self.interactive_close()
        self.server.run_mode = pre_run_mode

    def run_if_interactive(self):
        self._ensure_structure_calc_is_set()
        super(AseJob, self).run_if_interactive()
        self.interactive_collect()

    def _ensure_structure_calc_is_set(self):
        if self.structure.calc is None:
            self.set_calculator()

    def set_calculator(self):
        raise NotImplementedError(
            "The set_calculator function is not implemented for this code. Either set "
            "an ase calculator to the structure attribute, or subclass this job and "
            "define set_calculator."
        )

    def interactive_structure_setter(self, structure):
        self.structure.calc.calculate(structure)

    def interactive_positions_setter(self, positions):
        self.structure.positions = positions

    def interactive_initialize_interface(self):
        self.status.running = True
        self._ensure_structure_calc_is_set()
        self.structure.calc.set_label(self.working_directory + "/")
        self._interactive_library = True

    def interactive_close(self):
        if self.interactive_is_activated():
            super(AseJob, self).interactive_close()
            with self.project_hdf5.open("output") as h5:
                if "interactive" in h5.list_groups():
                    for key in h5["interactive"].list_nodes():
                        h5["generic/" + key] = h5["interactive/" + key]

    def interactive_forces_getter(self):
        return self.structure.get_forces()

    def interactive_pressures_getter(self):
        return -self.structure.get_stress(voigt=False)

    def interactive_energy_pot_getter(self):
        return self.structure.get_potential_energy()

    def interactive_energy_tot_getter(self):
        return self.structure.get_potential_energy()

    def interactive_indices_getter(self):
        element_lst = sorted(list(set(self.structure.get_chemical_symbols())))
        return np.array(
            [element_lst.index(el) for el in self.structure.get_chemical_symbols()]
        )

    def interactive_positions_getter(self):
        return self.structure.positions.copy()

    def interactive_steps_getter(self):
        return len(self.interactive_cache[list(self.interactive_cache.keys())[0]])

    def interactive_time_getter(self):
        return self.interactive_steps_getter()

    def interactive_volume_getter(self):
        return self.structure.get_volume()

    def interactive_cells_getter(self):
        return self.structure.cell.copy()

    def write_input(self):
        pass

    def collect_output(self):
        pass

    def run_if_scheduler(self):
        self._create_working_directory()
        super(AseJob, self).run_if_scheduler()

    def interactive_index_organizer(self):
        index_merge_lst = self._interactive_species_lst.tolist() + list(
            np.unique(self._structure_current.get_chemical_symbols())
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

    def _get_structure(self, frame=-1, wrap_atoms=True):
        if (
            self.server.run_mode.interactive
            or self.server.run_mode.interactive_non_modal
        ):
            # Warning: We only copy symbols, positions and cell information - no tags.
            if self.output.indices is not None and len(self.output.indices) != 0:
                indices = self.output.indices[frame]
            else:
                return None
            if len(self._interactive_species_lst) == 0:
                el_lst = list(np.unique(self._structure_current.get_chemical_symbols()))
            else:
                el_lst = self._interactive_species_lst.tolist()
            if indices is not None:
                if wrap_atoms:
                    positions = self.output.positions[frame]
                else:
                    if len(self.output.unwrapped_positions) > max([frame, 0]):
                        positions = self.output.unwrapped_positions[frame]
                    else:
                        positions = (
                            self.output.positions[frame]
                            + self.output.total_displacements[frame]
                        )
                atoms = Atoms(
                    symbols=np.array([el_lst[el] for el in indices]),
                    positions=positions,
                    cell=self.output.cells[frame],
                    pbc=self.structure.pbc,
                )
                # Update indicies to match the indicies in the cache.
                atoms.set_array("indices", indices)
                return atoms
            else:
                return None
        else:
            if (
                self.get("output/generic/cells") is not None
                and len(self.get("output/generic/cells")) != 0
            ):
                return super()._get_structure(frame=frame, wrap_atoms=wrap_atoms)
            else:
                return None
