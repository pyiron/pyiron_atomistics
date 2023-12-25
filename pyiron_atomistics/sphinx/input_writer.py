from shutil import copyfile
import os
import posixpath
import numpy as np


def copy_potentials(origins, destinations):
    """
    Args:
        origins (list): list of paths to potentials
        destinations (list): list of paths to copy potentials to
    """
    for ori, des in zip(origins, destinations):
        copyfile(ori, des)


def write_spin_constraints(
    self, file_name="spins.in", cwd=None, magmoms=None, constraints=None
):
    """
    Write a text file containing a list of all spins named spins.in -
    which is used for the external control scripts.

    Args:
        file_name (str): name of the file to be written (optional)
        cwd (str): the current working directory (optinal)
        spins_list (list): list of spins
    """
    for spin in magmoms:
        if isinstance(spin, list) or isinstance(spin, np.ndarray):
            raise ValueError("SPHInX only supports collinear spins at the moment.")
    spins = np.array(magmoms).astype(str)
    spins[~np.asarray(constraints)] = "X"
    spins_str = "\n".join(spins) + "\n"
    if cwd is not None:
        file_name = posixpath.join(cwd, file_name)
    with open(file_name, "w") as f:
        f.write(spins_str)
