from pyiron_base import state
from pyiron_vasp.vasp.vasprun import Vasprun as _Vasprun

from pyiron_atomistics.atomistics.structure.atoms import Atoms


class Vasprun(_Vasprun):
    def get_initial_structure(self):
        """
        Gets the initial structure from the simulation

        Returns:
            pyiron.atomistics.structure.atoms.Atoms: The initial structure

        """
        try:
            el_list = self.vasprun_dict["atominfo"]["species_list"]
            cell = self.vasprun_dict["init_structure"]["cell"]
            positions = self.vasprun_dict["init_structure"]["positions"]
            if len(positions[positions > 1.01]) > 0:
                basis = Atoms(el_list, positions=positions, cell=cell, pbc=True)
            else:
                basis = Atoms(el_list, scaled_positions=positions, cell=cell, pbc=True)
            if "selective_dynamics" in self.vasprun_dict["init_structure"].keys():
                basis.add_tag(selective_dynamics=[True, True, True])
                for i, val in enumerate(
                    self.vasprun_dict["init_structure"]["selective_dynamics"]
                ):
                    basis[i].selective_dynamics = val
            return basis
        except KeyError:
            state.logger.warning(
                "The initial structure could not be extracted from vasprun properly"
            )
            return
