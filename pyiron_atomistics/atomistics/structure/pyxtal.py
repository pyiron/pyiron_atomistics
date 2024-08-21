import warnings
from typing import List, Tuple, Union

import ase.atoms
from structuretoolkit.build import pyxtal as _pyxtal

from pyiron_atomistics.atomistics.structure.atoms import Atoms, ase_to_pyiron
from pyiron_atomistics.atomistics.structure.structurestorage import StructureStorage

publication = {
    "pyxtal": {
        "title": "PyXtal: A Python library for crystal structure generation and symmetry analysis",
        "journal": "Computer Physics Communications",
        "volume": "261",
        "pages": "107810",
        "year": "2021",
        "issn": "0010-4655",
        "doi": "https://doi.org/10.1016/j.cpc.2020.107810",
        "url": "http://www.sciencedirect.com/science/article/pii/S0010465520304057",
        "author": "Scott Fredericks and Kevin Parrish and Dean Sayre and Qiang Zhu",
    }
}


def pyxtal(
    group: Union[int, List[int]],
    species: Tuple[str],
    num_ions: Tuple[int],
    dim=3,
    repeat=1,
    storage=None,
    allow_exceptions=True,
    **kwargs,
) -> Union[Atoms, StructureStorage]:
    """
    Generate random crystal structures with PyXtal.

    `group` must be between 1 and the largest possible value for the given dimensionality:
        dim=3 => 1 - 230 (space groups)
        dim=2 => 1 -  80 (layer groups)
        dim=1 => 1 -  75 (rod groups)
        dim=0 => 1 -  58 (point groups)

    When `group` is passed as a list of integers or `repeat>1`, generate multiple structures and return them in a :class:`.StructureStorage`.

    Args:
        group (list of int, or int): the symmetry group to generate or a list of them
        species (tuple of str): which species to include, defines the stoichiometry together with `num_ions`
        num_ions (tuple of int): how many of each species to include, defines the stoichiometry together with `species`
        dim (int): dimensionality of the symmetry group, 0 is point groups, 1 is rod groups, 2 is layer groups and 3 is space groups
        repeat (int): how many random structures to generate
        storage (:class:`.StructureStorage`, optional): when generating multiple structures, add them to this instead of creating a new storage
        allow_exceptions (bool): when generating multiple structures, silence errors when the requested stoichiometry and symmetry group are incompatible
        **kwargs: passed to `pyxtal.pyxtal` function verbatim

    Returns:
        :class:`~.Atoms`: the generated structure, if repeat==1 and only one symmetry group is requested
        :class:`.StructureStorage`: a storage of all generated structure, if repeat>1 or multiple symmetry groups are requested

    Raises:
        ValueError: if `species` and `num_ions` are not of the same length
        ValueError: if stoichiometry and symmetry group are incompatible and allow_exceptions==False or only one structure is requested
    """
    ret = _pyxtal(
        group=group,
        species=species,
        num_ions=num_ions,
        dim=dim,
        repeat=repeat,
        allow_exceptions=allow_exceptions,
        **kwargs,
    )
    if isinstance(ret, ase.atoms.Atoms):
        return ase_to_pyiron(ret)
    else:
        stoich = "".join(f"{s}{n}" for s, n in zip(species, num_ions))
        if storage is None:
            storage = StructureStorage()
        for struct in ret:
            storage.add_structure(
                struct["atoms"],
                identifier=f"{stoich}_{struct['symmetry']}_{struct['repeat']}",
                symmetry=struct["symmetry"],
                repeat=struct["repeat"],
            )
        return storage
