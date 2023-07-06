# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

"""
Alternative structure container that stores them in flattened arrays.
"""

from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pyiron_base import FlattenedStorage, ImportAlarm
from pyiron_atomistics.atomistics.structure.atom import Atom
from pyiron_atomistics.atomistics.structure.atoms import Atoms
from structuretoolkit.common.error import SymmetryError
from pyiron_atomistics.atomistics.structure.neighbors import NeighborsTrajectory
import pyiron_atomistics.atomistics.structure.has_structure as pa_has_structure

with ImportAlarm(
    "Some plotting functionality requires the seaborn library."
) as seaborn_alarm:
    import seaborn as sns


class StructureStorage(FlattenedStorage, pa_has_structure.HasStructure):
    """
    Class that can write and read lots of structures from and to hdf quickly.

    This is done by storing positions, cells, etc. into large arrays instead of writing every structure into a new
    group.  Structures are stored together with an identifier that should be unique.  The class can be initialized with
    the number of structures and the total number of atoms in all structures, but re-allocates memory as necessary when
    more (or larger) structures are added than initially anticipated.

    You can add structures and a human-readable name with :meth:`.add_structure()`.

    >>> container = StructureStorage()
    >>> container.add_structure(Atoms(...), "fcc")
    >>> container.add_structure(Atoms(...), "hcp")
    >>> container.add_structure(Atoms(...), "bcc")

    Accessing stored structures works with :meth:`.get_strucure()`.  You can either pass the identifier you passed
    when adding the structure or the numeric index

    >>> container.get_structure(frame=0) == container.get_structure(frame="fcc")
    True

    Custom arrays may also be defined on the container

    >>> container.add_array("energy", shape=(), dtype=np.float64, fill=-1, per="chunk")

    (chunk means structure in this case, see below and :class:`.FlattenedStorage`)

    You can then pass arrays of the corresponding shape to :meth:`add_structure()`

    >>> container.add_structure(Atoms(...), "grain_boundary", energy=3.14)

    Saved arrays are accessed with :meth:`.get_array()`

    >>> container.get_array("energy", 3)
    3.14
    >>> container.get_array("energy", 0)
    -1

    It is also possible to use the same names in :meth:`.get_array()` as in :meth:`.get_structure()`.

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
        self._plots = None

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
            frame (int, str): selects structure to set, as in :meth:`.get_strucure()`
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

        if structure.has("initial_magmoms"):
            arrays["spins"] = structure.spins
        if "selective_dynamics" in structure.get_tags():
            arrays["selective_dynamics"] = structure.selective_dynamics

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
        symbols = self.get_array("symbols", frame)
        elements = [e for e in self.get_elements() if e in symbols]
        index_map = {e: i for i, e in enumerate(elements)}
        try:
            magmoms = self.get_array("spins", frame)
        except KeyError:
            # not all structures have spins saved on them
            magmoms = None
        structure = Atoms(
            species=[Atom(e).element for e in elements],
            indices=[index_map[e] for e in symbols],
            positions=self.get_array("positions", frame),
            cell=self.get_array("cell", frame),
            pbc=self.get_array("pbc", frame),
            magmoms=magmoms,
        )
        if self.has_array("selective_dynamics"):
            structure.add_tag(selective_dynamics=[True, True, True])
            selective_dynamics = self.get_array("selective_dynamics", frame)
            for i, d in enumerate(selective_dynamics):
                structure.selective_dynamics[i] = d.tolist()
        return structure

    def _number_of_structures(self):
        return len(self)

    def _get_hdf_group_name(self):
        return "structures"

    @property
    def plot(self):
        """
        Accessor for :class:`.StructurePlots` instance using these structures.
        """
        if self._plots is None:
            self._plots = StructurePlots(self)
        return self._plots


class StructurePlots:
    """
    Simple interface to plot various properties of structures.
    """

    @seaborn_alarm
    def __init__(self, store: StructureStorage):
        self._store = store
        self._neigh = None

    def atoms(self):
        """
        Plot a histogram of the number of atoms in each structure.
        """
        length = self._store["length"]
        lo = length.min()
        hi = length.max()
        # make the bins fall in between whole numbers and include hi
        plt.hist(length, bins=np.arange(lo, hi + 2) - 0.5)
        plt.xlabel("#Atoms")
        plt.ylabel("Count")

    def cell(self, angle_in_degrees=True):
        """
        Plot histograms of cell parameters.

        Plotted are atomic volume, density, cell vector lengths and cell vector angles in separate subplots all on a
        log-scale.

        Args:
            angle_in_degrees (bool): whether unit for angles is degree or radians

        Returns:
            `DataFrame`: contains the plotted information in the columns:
                            - a: length of first vector
                            - b: length of second vector
                            - c: length of third vector
                            - alpha: angle between first and second vector
                            - beta: angle between second and third vector
                            - gamma: angle between third and first vector
                            - V: volume of the cell
                            - N: number of atoms in the cell
        """
        N = self._store.get_array("length")
        C = self._store.get_array("cell")

        def get_angle(cell, idx=0):
            return np.arccos(
                np.dot(cell[idx], cell[(idx + 1) % 3])
                / np.linalg.norm(cell[idx])
                / np.linalg.norm(cell[(idx + 1) % 3])
            )

        def extract(n, c):
            return {
                "a": np.linalg.norm(c[0]),
                "b": np.linalg.norm(c[1]),
                "c": np.linalg.norm(c[2]),
                "alpha": get_angle(c, 0),
                "beta": get_angle(c, 1),
                "gamma": get_angle(c, 2),
            }

        df = pd.DataFrame([extract(n, c) for n, c in zip(N, C)])
        df["V"] = np.linalg.det(C)
        df["N"] = N
        if angle_in_degrees:
            df["alpha"] = np.rad2deg(df["alpha"])
            df["beta"] = np.rad2deg(df["beta"])
            df["gamma"] = np.rad2deg(df["gamma"])

        plt.subplot(1, 4, 1)
        plt.title("Atomic Volume")
        plt.hist(df.V / df.N, bins=20, log=True)
        plt.xlabel(r"$V$ [$\AA^3$]")

        plt.subplot(1, 4, 2)
        plt.title("Density")
        plt.hist(df.N / df.V, bins=20, log=True)
        plt.xlabel(r"$\rho$ [$\AA^{-3}$]")

        plt.subplot(1, 4, 3)
        plt.title("Lattice Vector Lengths")
        plt.hist([df.a, df.b, df.c], log=True)
        plt.xlabel(r"$a,b,c$ [$\AA$]")

        plt.subplot(1, 4, 4)
        plt.title("Lattice Vector Angles")
        plt.hist([df.alpha, df.beta, df.gamma], log=True)
        if angle_in_degrees:
            label = r"$\alpha,\beta,\gamma$ [°]"
        else:
            label = r"$\alpha,\beta,\gamma$ [rad]"
        plt.xlabel(label)

        return df

    def _calc_spacegroups(self, symprec=1e-3):
        """
        Calculate space groups of all structures.

        Args:
            symprec (float): symmetry precision given to spglib

        Returns:
            DataFrame: contains columns 'crystal_system' (str) and 'space_group' (int) for each structure
        """

        def get_crystal_system(num):
            if num in range(1, 3):
                return "triclinic"
            elif num in range(3, 16):
                return "monoclinic"
            elif num in range(16, 75):
                return "orthorhombic"
            elif num in range(75, 143):
                return "tetragonal"
            elif num in range(143, 168):
                return "trigonal"
            elif num in range(168, 195):
                return "hexagonal"
            elif num in range(195, 231):
                return "cubic"

        def extract(s):
            try:
                spg = s.get_symmetry(symprec=symprec).spacegroup["Number"]
            except SymmetryError:
                spg = 1
            return {"space_group": spg, "crystal_system": get_crystal_system(spg)}

        return pd.DataFrame(map(extract, self._store.iter_structures()))

    def spacegroups(self, symprec=1e-3):
        """
        Plot histograms of space groups and crystal systems.

        Spacegroups and crystal systems are plotted in separate subplots.

        Args:
            symprec (float): precision of the symmetry search (passed to spglib)

        Returns:
            DataFrame: contains two columns "space_group", "crystal_system"
                       for each structure
        """

        df = self._calc_spacegroups(symprec=symprec)
        plt.subplot(1, 2, 1)
        plt.hist(df.space_group, bins=230)
        plt.xlabel("Space Group")

        plt.subplot(1, 2, 2)
        l, h = np.unique(df.crystal_system, return_counts=True)
        sort_key = {
            "triclinic": 1,
            "monoclinic": 3,
            "orthorhombic": 16,
            "tetragonal": 75,
            "trigonal": 143,
            "hexagonal": 168,
            "cubic": 195,
        }
        I = np.argsort([sort_key[ll] for ll in l])
        plt.bar(l[I], h[I])
        plt.xlabel("Crystal System")
        plt.xticks(rotation=35)
        return df

    def _calc_neighbors(self, num_neighbors):
        """
        Calculate the neighbor information with additional caching.

        If 'distances' and 'shells' are provided in the underlying store, they are returned directly without checking
        `num_neighbors`.

        If they are not provided there, they are calculated here and cached.
        Recalculation happens when a different `num_neighbors` is provided than in a previous call or the underlying
        store changes.

        If `num_neighbors` is `None` on the first call, the default is 36.

        Returns:
            dict: with keys 'distances' and 'shells' containing the respective flattened arrays from
            :meth:`.Atoms.get_neighbors`.
        """
        if self._store.has_array("distances") and self._store.has_array("shells"):
            return {
                "distances": self._store["distances"],
                "shells": self._store["shells"],
            }
        # check that _store and _neigh are still consistent
        if (
            self._neigh is None
            or len(self._store) != len(self._neigh)
            or (
                num_neighbors is None
                or self._neigh.has_array("distances")["shape"][0] != num_neighbors
            )
        ):
            if num_neighbors is None:
                num_neighbors = 36
            self._neigh = FlattenedStorage()
            neigh_traj = NeighborsTrajectory(
                has_structure=self._store,
                num_neighbors=num_neighbors,
                store=self._neigh,
            )
        return {
            "distances": self._neigh["distances"],
            "shells": self._neigh["shells"],
        }

    def coordination(self, num_shells=4, log=True, num_neighbors=None):
        """
        Plot histogram of coordination in neighbor shells.

        Computes one histogram of the number of neighbors in each neighbor shell up to `num_shells` and then plots them
        together.

        If the underlying :class:`.StructureStorage` has a 'shells' array defined it is used, if not it is calculated on
        the fly.

        Args:
            num_shells (int): maximum shell to plot
            num_neighbors (int): maximum number of neighbors to calculate, when 'shells' is not defined in storage,
                                 default is the value from the previous call or 36
            log (float): plot histogram values on a log scale
        """
        neigh = self._calc_neighbors(num_neighbors=num_neighbors)
        shells = neigh["shells"]

        shell_index = (
            shells[np.newaxis, :, :]
            == np.arange(1, num_shells + 1)[:, np.newaxis, np.newaxis]
        )
        neigh_count = shell_index.sum(axis=-1)
        ticks = np.arange(neigh_count.min(), neigh_count.max() + 1)
        plt.hist(
            neigh_count.T,
            bins=ticks - 0.5,
            log=True,
            label=[f"{i}." for i in range(1, num_shells + 1)],
        )
        plt.xticks(ticks)
        plt.xlabel("Number of Neighbors")
        plt.legend(title="Shell")
        plt.title("Neighbor Coordination in Shells")

    def distances(
        self,
        bins: int = 50,
        num_neighbors: int = None,
        normalize: bool = False,
    ):
        """
        Plot a histogram of the neighbor distances.

        Setting `normalize` plots the radial distribution function.

        Args:
            bins (int): number of bins
            num_neighbors (int): maximum number of neighbors to calculate, when 'shells' or 'distances' are not defined in storage
                                 default is the value from the previous call or 36
            normalize (bool): normalize the distribution by the surface area of
                              the radial bin, 4pi r^2
        """
        neigh = self._calc_neighbors(num_neighbors=num_neighbors)
        distances = neigh["distances"].flatten()

        if normalize:
            plt.hist(
                distances,
                bins=bins,
                weights=1 / (4 * np.pi * distances**2),
            )
            plt.ylabel("Neighbor density [$\mathrm{\AA}^{-2}$]")
        else:
            plt.hist(distances, bins=bins)
            plt.ylabel("Neighbor count")
        plt.xlabel(r"Distance [$\mathrm{\AA}$]")

    def shell_distances(self, num_shells=4, num_neighbors=None):
        """
        Plot a violin plot of the neighbor distances in shells up to `num_shells`.

        Args:
            num_shells (int): maximum shell to plot
            num_neighbors (int): maximum number of neighbors to calculate, when 'shells' or 'distances' are not defined in storage
                                 default is the value from the previous call or 36
        """
        neigh = self._calc_neighbors(num_neighbors=num_neighbors)
        shells = neigh["shells"]
        distances = neigh["distances"]

        R = distances.flatten()
        S = shells.ravel()
        d = pd.DataFrame(
            {"distance": R[S < num_shells + 1], "shells": S[S < num_shells + 1]}
        )
        sns.violinplot(y=d.shells, x=d.distance, scale="width", orient="h")
        plt.xlabel(r"Distance [$\AA$]")
        plt.ylabel("Shell")

    def concentration(self, elements: List[str] = None, **kwargs) -> pd.DataFrame:
        """
        Plot histograms of the concentrations in each structure.

        Args:
            elements (list of str): elements to plot the histograms for; default is for all elements in the container
            **kwargs: passed through to `seaborn.histplot`

        Returns:
            `pandas.DataFrame`: table of concentrations in each structure; column headers are the element names
        """
        if elements is not None:
            for elem in elements:
                if elem not in self._store.get_elements():
                    raise ValueError(f"Element {elem} not present in storage!")
        else:
            elements = self._store.get_elements()

        df = pd.DataFrame(
            [
                {elem: sum(elem == sym) / len(sym) for elem in elements}
                for sym in self._store.get_array_ragged("symbols")
            ]
        )

        sns.histplot(
            data=df.melt(var_name="element", value_name="concentration"),
            x="concentration",
            hue="element",
            multiple="dodge",
            **kwargs,
        )

        return df
