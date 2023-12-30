from shutil import copyfile
import os
import posixpath
import numpy as np
from pyiron_base import DataContainer


class Group(DataContainer):
    """
    Dictionary-like object to store SPHInX inputs.

    Attributes (sub-groups, parameters, & flags) can be set
    and accessed via dot notation, or as standard dictionary
    key/values.

    `to_{job_type}` converts the Group to the format
    expected by the given DFT code in its input files.
    """

    def set_group(self, name, content=None):
        """
        Set a new group in SPHInX input.

        Args:
            name (str): name of the group
            content: content to append

        This creates an input group of the type `name { content }`.
        """
        if content is None:
            self.create_group(name)
        else:
            self[name] = content

    def set_flag(self, flag, val=True):
        """
        Set a new flag in SPHInX input.

        Args:
            flag (str): name of the flag
            val (bool): boolean value

        This creates an input flag of the type `name = val`.
        """
        self[flag] = val

    def set_parameter(self, parameter, val):
        """
        Set a new parameter in SPHInX input.

        Args:
            parameter (str): name of the flag
            val (float): parameter value

        This creates an input parameter of the type `parameter = val`.
        """
        self[parameter] = val

    def to_sphinx(self, content="__self__", indent=0):
        if content == "__self__":
            content = self

        def format_value(v):
            if isinstance(v, bool):
                if v:
                    return ";"
                else:
                    return " = false;"
            elif isinstance(v, Group):
                if len(v) == 0:
                    return " {}"
                else:
                    return " {\n" + self.to_sphinx(v, indent + 1) + indent * "\t" + "}"
            else:
                if isinstance(v, np.ndarray):
                    v = v.tolist()
                return " = {!s};".format(v)

        line = ""
        for k, v in content.items():
            if isinstance(v, Group) and len(v) > 0 and not v.has_keys():
                for vv in v.values():
                    line += indent * "\t" + str(k) + format_value(vv) + "\n"
            else:
                line += indent * "\t" + str(k) + format_value(v) + "\n"

        return line


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
