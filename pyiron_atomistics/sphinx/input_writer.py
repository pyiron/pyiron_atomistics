from shutil import copyfile
import os
import posixpath
import numpy as np


class InputWriter(object):
    """
    The SPHInX Input writer is called to write the
    SPHInX specific input files.
    """

    def __init__(self):
        self.structure = None
        self._id_pyi_to_spx = []
        self._id_spx_to_pyi = []
        self.file_dict = {}

    @property
    def id_spx_to_pyi(self):
        if self.structure is None:
            return None
        if len(self._id_spx_to_pyi) == 0:
            self._initialize_order()
        return self._id_spx_to_pyi

    @property
    def id_pyi_to_spx(self):
        if self.structure is None:
            return None
        if len(self._id_pyi_to_spx) == 0:
            self._initialize_order()
        return self._id_pyi_to_spx

    def _initialize_order(self):
        for elm_species in self.structure.get_species_objects():
            self._id_pyi_to_spx.append(
                np.arange(len(self.structure))[
                    self.structure.get_chemical_symbols() == elm_species.Abbreviation
                ]
            )
        self._id_pyi_to_spx = np.array(
            [ooo for oo in self._id_pyi_to_spx for ooo in oo]
        )
        self._id_spx_to_pyi = np.array([0] * len(self._id_pyi_to_spx))
        for i, p in enumerate(self._id_pyi_to_spx):
            self._id_spx_to_pyi[p] = i

    def copy_potentials(self, origins, destinations):
        """
        Args:
            origins (list): list of paths to potentials
            destinations (list): list of paths to copy potentials to
        """
        for ori, des in zip(origins, destinations):
            copyfile(ori, des)

    def write_spin_constraints(self, file_name="spins.in", cwd=None, spins_list=None):
        """
        Write a text file containing a list of all spins named spins.in -
        which is used for the external control scripts.

        Args:
            file_name (str): name of the file to be written (optional)
            cwd (str): the current working directory (optinal)
            spins_list (list): the input to write, if no input is
                given the default input will be written. (optional)
        """
        if self.structure.has("initial_magmoms"):
            if any(
                [
                    True
                    if isinstance(spin, list) or isinstance(spin, np.ndarray)
                    else False
                    for spin in self.structure.get_initial_magnetic_moments()
                ]
            ):
                raise ValueError("SPHInX only supports collinear spins at the moment.")
            else:
                constraint = self.structure.spin_constraint[self.id_pyi_to_spx]
                if spins_list is None or len(spins_list) == 0:
                    spins_list = self.structure.get_initial_magnetic_moments()
                spins = spins_list[self.id_pyi_to_spx].astype(str)
                spins[~np.asarray(constraint)] = "X"
                spins_str = "\n".join(spins) + "\n"
                if cwd is not None:
                    file_name = posixpath.join(cwd, file_name)
                with open(file_name, "w") as f:
                    f.write(spins_str)
