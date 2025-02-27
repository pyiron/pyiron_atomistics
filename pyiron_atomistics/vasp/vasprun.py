from pyiron_vasp.vasp.vasprun import Vasprun as _Vasprun

from pyiron_atomistics.atomistics.structure.atoms import ase_to_pyiron


class Vasprun(_Vasprun):
    def get_initial_structure(self):
        return ase_to_pyiron(super().get_initial_structure())
