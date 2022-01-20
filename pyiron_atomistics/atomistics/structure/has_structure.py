# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from abc import ABC, abstractmethod, abstractproperty
import numbers
from pyiron_base import deprecate

"""
Mixin for classes that have one or more structures attached to them.
"""

__author__ = "Marvin Poul"
__copyright__ = (
    "Copyright 2021, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "1.1"
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
