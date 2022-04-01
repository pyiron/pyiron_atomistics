# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

"""
Alternative structure container that stores them in flattened arrays.
"""

from typing import List

import numpy as np

from pyiron_base import FlattenedStorage
from pyiron_atomistics.atomistics.structure.atom import Atom
from pyiron_atomistics.atomistics.structure.atoms import Atoms
from pyiron_atomistics.atomistics.structure.has_structure import HasStructure


class StructureStorage(FlattenedStorage, HasStructure):
    """
    Class that can write and read lots of structures from and to hdf quickly.

    This is done by storing positions, cells, etc. into large arrays instead of writing every structure into a new
    group.  Structures are stored together with an identifier that should be unique.  The class can be initialized with
    the number of structures and the total number of atoms in all structures, but re-allocates memory as necessary when
    more (or larger) structures are added than initially anticipated.

    You can add structures and a human-readable name with :method:`.add_structure()`.

    >>> container = StructureStorage()
    >>> container.add_structure(Atoms(...), "fcc")
    >>> container.add_structure(Atoms(...), "hcp")
    >>> container.add_structure(Atoms(...), "bcc")

    Accessing stored structures works with :method:`.get_strucure()`.  You can either pass the identifier you passed
    when adding the structure or the numeric index

    >>> container.get_structure(frame=0) == container.get_structure(frame="fcc")
    True

    Custom arrays may also be defined on the container

    >>> container.add_array("energy", shape=(), dtype=np.float64, fill=-1, per="chunk")

    (chunk means structure in this case, see below and :class:`.FlattenedStorage`)

    You can then pass arrays of the corresponding shape to :method:`add_structure()`

    >>> container.add_structure(Atoms(...), "grain_boundary", energy=3.14)

    Saved arrays are accessed with :method:`.get_array()`

    >>> container.get_array("energy", 3)
    3.14
    >>> container.get_array("energy", 0)
    -1

    It is also possible to use the same names in :method:`.get_array()` as in :method:`.get_structure()`.

    >>> container.get_array("energy", 0) == container.get_array("energy", "fcc")
    True

    The length of the container is the number of structures inside it.

    >>> len(container)
    4

    Each structure corresponds to a chunk in :class:`.FlattenedStorage` and each atom to an element.  By default the
    following arrays are defined for each structure:
        - identifier    shape=(),    dtype=str,          per chunk; human readable name of the structure
        - cell          shape=(3,3), dtype=np.float64,   per chunk; cell shape
        - pbc           shape=(3,),  dtype=bool          per chunk; periodic boundary conditions
        - symbols:      shape=(),    dtype=str,          per element; chemical symbol
        - positions:    shape=(3,),  dtype=np.float64,   per element: atomic positions
    If a structure has spins/magnetic moments defined on its atoms these will be saved in a per atom array as well.  In
    that case, however all structures in the container must either have all collinear spins or all non-collinear spins.
    """

    def __init__(self, num_atoms=1, num_structures=1):
        """
        Create new structure container.

        Args:
            num_atoms (int): total number of atoms across all structures to pre-allocate
            num_structures (int): number of structures to pre-allocate
        """
        super().__init__(num_elements=num_atoms, num_chunks=num_structures)
        self._element_cache = None

    def _init_arrays(self):
        super()._init_arrays()
        # 2 character unicode array for chemical symbols
        self._per_element_arrays["symbols"] = np.full(
            self._num_elements_alloc, "XX", dtype=np.dtype("U2")
        )
        self._per_element_arrays["positions"] = np.empty((self._num_elements_alloc, 3))

        self._per_chunk_arrays["cell"] = np.empty((self._num_chunks_alloc, 3, 3))
        self._per_chunk_arrays["pbc"] = np.empty(
            (self._num_elements_alloc, 3), dtype=bool
        )

    @property
    def symbols(self):
        """:meta private:"""
        return self._per_element_arrays["symbols"]

    @property
    def positions(self):
        """:meta private:"""
        return self._per_element_arrays["positions"]

    @property
    def start_index(self):
        """:meta private:"""
        return self._per_chunk_arrays["start_index"]

    @property
    def length(self):
        """:meta private:"""
        return self._per_chunk_arrays["length"]

    @property
    def identifier(self):
        """:meta private:"""
        return self._per_chunk_arrays["identifier"]

    @property
    def cell(self):
        """:meta private:"""
        return self._per_chunk_arrays["cell"]

    @property
    def pbc(self):
        """:meta private:"""
        return self._per_chunk_arrays["pbc"]

    def set_array(self, name, frame, value):
        """
        Add array for given structure.

        Works for per chunk and per element arrays.

        Args:
            name (str): name of array to set
            frame (int, str): selects structure to set, as in :method:`.get_strucure()`
            value: value (for per chunk) or array of values (for per element); type and shape as per :meth:`.hasarray()`.

        Raises:
            `KeyError`: if array with name does not exists
        """
        super().set_array(name, frame, value)
        # invalidate cache when writing to symbols; could be smarter by checking if value is not in in cache but w/e
        if name == "symbols":
            self._element_cache = None

    def get_elements(self) -> List[str]:
        """
        Return a list of chemical elements present in the storage.

        Returns:
            :class:`list`: list of unique elements as strings of chemical symbols
        """
        if self._element_cache is None:
            self._element_cache = list(np.unique(self._per_element_arrays["symbols"]))
        return self._element_cache

    def add_structure(self, structure, identifier=None, **arrays):
        """
        Add a new structure to the container.

        Additional keyword arguments given specify additional arrays to store for the structure.  If an array with the
        given keyword name does not exist yet, it will be added to the container.

        >>> container = StructureStorage()
        >>> container.add_structure(Atoms(...), identifier="A", energy=3.14)
        >>> container.get_array("energy", 0)
        3.14

        If the first axis of the extra array matches the length of the given structure, it will be added as an per atom
        array, otherwise as an per structure array.

        >>> structure = Atoms(...)
        >>> container.add_structure(structure, identifier="B", forces=len(structure) * [[0,0,0]])
        >>> len(container.get_array("forces", 1)) == len(structure)
        True

        Reshaping the array to have the first axis be length 1 forces the array to be set as per structure array.  That
        axis will then be stripped.

        >>> container.add_structure(Atoms(...), identifier="C", pressure=np.eye(3)[np.newaxis, :, :])
        >>> container.get_array("pressure", 2).shape
        (3, 3)

        Args:
            structure (:class:`.Atoms`): structure to add
            identifier (str, optional): human-readable name for the structure, if None use current structre index as
                                        string
            **kwargs: additional arrays to store for structure
        """

        if structure.spins is not None:
            arrays["spins"] = structure.spins

        self.add_chunk(
            len(structure),
            identifier=identifier,
            symbols=np.array(structure.symbols),
            positions=structure.positions,
            cell=[structure.cell.array],
            pbc=[structure.pbc],
            **arrays,
        )

    def _translate_frame(self, frame):
        try:
            return self.find_chunk(frame)
        except KeyError:
            raise KeyError(f"No structure named {frame}.") from None

    def _get_structure(self, frame=-1, wrap_atoms=True):
        elements = self.get_elements()
        index_map = {e: i for i, e in enumerate(elements)}
        try:
            magmoms = self.get_array("spins", frame)
        except KeyError:
            # not all structures have spins saved on them
            magmoms = None
        symbols = self.get_array("symbols", frame)
        return Atoms(
            species=[Atom(e).element for e in elements],
            indices=[index_map[e] for e in symbols],
            positions=self.get_array("positions", frame),
            cell=self.get_array("cell", frame),
            pbc=self.get_array("pbc", frame),
            magmoms=magmoms,
        )

    def _number_of_structures(self):
        return len(self)

    def _get_hdf_group_name(self):
        return "structures"
