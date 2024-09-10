import posixpath
from shutil import copyfile

import numpy as np
import scipy.constants
from pyiron_base import DataContainer

BOHR_TO_ANGSTROM = (
    scipy.constants.physical_constants["Bohr radius"][0] / scipy.constants.angstrom
)


class Group(DataContainer):
    """
    Dictionary-like object to store SPHInX inputs.

    Attributes (sub-groups, parameters, & flags) can be set
    and accessed via dot notation, or as standard dictionary
    key/values.

    `to_{job_type}` converts the Group to the format
    expected by the given DFT code in its input files.
    """

    def to_sphinx(self, content="__self__", indent=0):
        if content == "__self__":
            content = self

        def format_value(v):
            if isinstance(v, bool):
                return f" = {v};".lower()
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


def get_structure_group(
    positions,
    cell,
    elements,
    movable=None,
    labels=None,
    use_symmetry=True,
    keep_angstrom=False,
):
    """
    create a SPHInX Group object based on structure

    Args:
        positions ((n, 3)-list/numpy.ndarray): xyz-coordinates of the atoms
        cell ((3, 3)-list/numpy.ndarray): Simulation box cdimensions
        elements ((n,)-list/numpy.ndarray): Chemical symbols
        movable (None/(n, 3)-list/nump.ndarray): Whether to fix the
            movement of the atoms in given directions
        labels (None/(n,)-list/numpy.ndarray): Extra labels to distinguish
            atoms for symmetries (mainly for magnetic moments)
        use_symmetry (bool): Whether or not consider internal symmetry
        keep_angstrom (bool): Store distances in Angstroms or Bohr

    Returns:
        (Group): structure group
    """
    positions = np.array(positions)
    cell = np.array(cell)
    if not keep_angstrom:
        cell /= BOHR_TO_ANGSTROM
        positions /= BOHR_TO_ANGSTROM
    structure_group = Group({"cell": np.array(cell)})
    if movable is not None:
        movable = np.array(movable)
    else:
        movable = np.full(shape=positions.shape, fill_value=True)
    if positions.shape != movable.shape:
        raise ValueError("positions.shape != movable.shape")
    if labels is not None:
        labels = np.array(labels)
    else:
        labels = np.full(shape=(len(positions),), fill_value=None)
    if (len(positions),) != labels.shape:
        raise ValueError("len(positions) != labels.shape")
    species = structure_group.create_group("species")
    for elm_species in np.unique(elements):
        species.append(Group({"element": '"' + str(elm_species) + '"'}))
        elm_list = elements == elm_species
        atom_group = species[-1].create_group("atom")
        for elm_pos, elm_magmom, selective in zip(
            positions[elm_list],
            labels[elm_list],
            movable[elm_list],
        ):
            atom_group.append(Group())
            if elm_magmom is not None:
                atom_group[-1]["label"] = '"spin_' + str(elm_magmom) + '"'
            atom_group[-1]["coords"] = np.array(elm_pos)
            if all(selective):
                atom_group[-1]["movable"] = True
            elif any(selective):
                for xx in np.array(["X", "Y", "Z"])[selective]:
                    atom_group[-1]["movable" + xx] = True
    if not use_symmetry:
        structure_group.symmetry = Group(
            {"operator": {"S": "[[1,0,0],[0,1,0],[0,0,1]]"}}
        )
    return structure_group


def copy_potentials(origins, destinations):
    """
    Args:
        origins (list): list of paths to potentials
        destinations (list): list of paths to copy potentials to
    """
    for ori, des in zip(origins, destinations):
        copyfile(ori, des)


def write_spin_constraints(
    file_name="spins.in", cwd=None, magmoms=None, constraints=None
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
