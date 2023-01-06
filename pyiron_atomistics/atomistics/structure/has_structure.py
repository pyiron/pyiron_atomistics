# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from abc import ABC, abstractmethod, abstractproperty
from typing import Callable
import numbers
from pyiron_base import deprecate, ImportAlarm
from pyiron_atomistics.atomistics.structure.atoms import Atoms

with ImportAlarm("Animation of atomic structures requires nglview") as nglview_alarm:
    import nglview

"""
Mixin for classes that have one or more structures attached to them.
"""

__author__ = "Marvin Poul"
__copyright__ = (
    "Copyright 2021, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "1.2"
__maintainer__ = "Marvin Poul"
__email__ = "poul@mpie.de"
__status__ = "production"
__date__ = "Apr 22, 2021"


class HasStructure(ABC):
    """
    Mixin for classes that have one or more structures attached to them.

    Necessary overrides are :meth:`._get_structure()` and
    :meth:`._number_of_structures()`.

    :meth:`.get_structure()` checks that iteration_step is valid; implementations of
    :meth:`._get_structure()` therefore don't have to check it.

    :attr:`.number_of_structures` may be zero, e.g. if there's no structure stored in the object yet or a
    job will compute this structure, but hasn't been run yet.

    Sub classes that wish to document special behavior of their implementation of :meth:`.get_structure` may do so by
    adding documention to it in the "Methods:" sub section of their class docstring.

    Sub classes may support custom data types as indices for `frame` in :meth:`.get_structure()` by overriding
    :meth:`._translate_frame()`.

    The example below shows how to implement this mixin and how to check whether an object derives from it

    >>> from pyiron_atomistics.atomistics.structure.atoms import Atoms
    >>> class Foo(HasStructure):
    ...     '''
    ...     Methods:
    ...         .. method:: get_structure
    ...             returns structure with single Fe atom at (0, 0, 0)
    ...     '''
    ...     def _get_structure(self, frame=-1, wrap_atoms=True):
    ...         return Atoms(symbols=['Fe'], positions=[[0,0,0]])
    ...     def _number_of_structures(self):
    ...         return 1

    >>> f = Foo()
    >>> for s in f.iter_structures():
    ...     print(s)
    Fe: [0. 0. 0.]
    pbc: [False False False]
    cell:
    Cell([0.0, 0.0, 0.0])
    <BLANKLINE>

    >>> isinstance(f, HasStructure)
    True


    .. document private functions
    .. automethod:: _get_structure
    .. automethod:: _number_of_structures
    .. automethod:: _translate_frame
    """

    @deprecate(iteration_step="use frame instead")
    def get_structure(self, frame=-1, wrap_atoms=True, iteration_step=None):
        """
        Retrieve structure from object.  The number of available structures depends on the job and what kind of
        calculation has been run on it, see :attr:`.number_of_structures`.

        Args:
            frame (int, object): index of the structure requested, if negative count from the back; if
            :meth:`_translate_frame()` is overridden, frame will pass through it
            iteration_step (int): deprecated alias for frame
            wrap_atoms (bool): True if the atoms are to be wrapped back into the unit cell

        Returns:
            :class:`pyiron_atomistics.atomistics.structure.atoms.Atoms`: the requested structure

        Raises:
            IndexError: if not -:attr:`.number_of_structures` <= iteration_step < :attr:`.number_of_structures`
        """
        if iteration_step is not None:
            frame = iteration_step
        if not isinstance(frame, numbers.Integral):
            try:
                frame = self._translate_frame(frame)
            except NotImplementedError:
                raise KeyError(
                    f"argument frame {frame} is not an integer and _translate_frame() not implemented!"
                ) from None
        num_structures = self.number_of_structures
        if frame < 0:
            frame += num_structures
        if not (0 <= frame < num_structures):
            raise IndexError(
                f"argument frame {frame} out of range [-{num_structures}, {num_structures})."
            )

        return self._get_structure(frame=frame, wrap_atoms=wrap_atoms)

    def _translate_frame(self, frame):
        """
        Translate frame to an integer for :meth:`_get_structure()`.

        Args:
            frame (object): any object to translate into an integer id

        Returns:
            int: valid integer to be passed to :meth:`._get_structure()`

        Raises:
            KeyError: if given frame does not exist in this object
        """
        raise NotImplementedError("No frame translation implemented!")

    @abstractmethod
    def _get_structure(self, frame=-1, wrap_atoms=True):
        pass

    @property
    def number_of_structures(self):
        """
        `int`: maximum `iteration_step` + 1 that can be passed to :meth:`.get_structure()`.
        """
        return self._number_of_structures()

    @abstractmethod
    def _number_of_structures(self):
        pass

    def iter_structures(self, wrap_atoms=True):
        """
        Iterate over all structures in this object.

        Args:
            wrap_atoms (bool): True if the atoms are to be wrapped back into the unit cell; passed to
                               :meth:`.get_structure()`

        Yields:
            :class:`pyiron_atomistics.atomistitcs.structure.atoms.Atoms`: every structure attached to the object
        """
        for i in range(self.number_of_structures):
            yield self._get_structure(frame=i, wrap_atoms=wrap_atoms)

    def transform_structures(self, modify) -> "TransformStructure":
        """
        Return a modified object by applying a function to each object lazily.

        Args:
            modify (function): applied to each structure, has to return the modified structure

        Returns:
            :class:`.TransformStructure`: a container with the modified structures
        """
        return TransformStructure(self, modify)

    def collect_structures(self, filter_function=None) -> "StructureStorage":
        """
        Collects a copy of all structures in a compact :class:`.StructureStorage`.

        This can be used to force lazily applied modifications with :meth:`.transform_structures` or simply to obtain a
        known object type from a generic :class:`.HasStructure` object.

        Args:
            filter_function (function): include structure only if this function returns True for it

        Returns:
            :class:`.StructureStorage`: a copy of all (filtered) structures
        """
        # breaks cyclical import
        # this is a bit annoying, but I want to give users an entry point to using StructureStorage without having to
        # import it
        from pyiron_atomistics.atomistics.structure.structurestorage import (
            StructureStorage,
        )

        store = StructureStorage()
        for structure in self.iter_structures():
            if filter_function is None or filter_function(structure):
                store.add_structure(structure)
        return store

    @nglview_alarm
    def animate_structures(
        self,
        spacefill: bool = True,
        show_cell: bool = True,
        center_of_mass: bool = False,
        particle_size: float = 0.5,
        camera: str = "orthographic",
    ):
        """
        Animate a series of atomic structures.

        Args:
            spacefill (bool): If True, then atoms are visualized in spacefill stype
            show_cell (bool): True if the cell boundaries of the structure is to be shown
            particle_size (float): Scaling factor for the spheres representing the atoms.
                                    (The radius is determined by the atomic number)
            center_of_mass (bool): False (default) if the specified positions are w.r.t. the origin
            camera (str): camera perspective, choose from "orthographic" or "perspective"

        Returns:
            animation: nglview IPython widget
        """

        if self._number_of_structures() <= 1:
            raise ValueError("job must have more than one structure to animate!")

        animation = nglview.show_asetraj(_TrajectoryAdapter(self))
        if spacefill:
            animation.add_spacefill(radius_type="vdw", scale=0.5, radius=particle_size)
            animation.remove_ball_and_stick()
        else:
            animation.add_ball_and_stick()
        if show_cell:
            animation.add_unitcell()
        animation.camera = camera
        return animation


class TransformStructure(HasStructure):
    """
    Modifies any HasStructure by applying a function to each structure lazily.
    """

    __slots__ = ("_source", "_modify")

    def __init__(self, source: HasStructure, modify: Callable[[Atoms], Atoms]):
        self._source = source
        self._modify = modify

    def _number_of_structures(self):
        return self._source._number_of_structures()

    def _translate_frame(self, frame):
        return self._source._translate_frame(frame)

    def _get_structure(self, frame=-1, wrap_atoms=True):
        return self._modify(self._source._get_structure(frame, wrap_atoms=wrap_atoms))


class _TrajectoryAdapter:
    """
    Class that translates between HasStructure and the ASE Trajectory interface.

    The ASE interface is needed e.g. by nglview to animate a series of structures, but the Trajectory interface uses
    methods that not all can implement (e.g. AtomisticGenericJob already overloads __getitem__).
    """

    __slots__ = "_underlying"

    def __init__(self, underlying: HasStructure):
        self._underlying = underlying

    def __getitem__(self, item):
        return self._underlying.get_structure(item)

    def __len__(self):
        return self._underlying.number_of_structures

    def __iter__(self):
        yield from self._underlying.iter_structures()
