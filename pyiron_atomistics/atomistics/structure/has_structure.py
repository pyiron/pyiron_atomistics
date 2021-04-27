# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from abc import ABC, abstractmethod, abstractproperty

"""
Mixin for classes that have one or more structures attached to them.
"""

__author__ = "Marvin Poul"
__copyright__ = (
    "Copyright 2021, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "1.0"
__maintainer__ = "Marvin Poul"
__email__ = "poul@mpie.de"
__status__ = "production"
__date__ = "Apr 22, 2021"


class HasStructure(ABC):
    """
    Mixin for classes that have one or more structures attached to them.

    Necessary overrides are :abstractmethod:`._get_structure()` and
    :abstractmethod:`._number_of_structures()`.

    :method:`.get_structure()` checks that iteration_step is valid; implementations of
    :abstractmethod:`._get_structure()` therefore don't have to check it.

    :method:`.get_number_of_structures()` may return zero, e.g. if there's no structure stored in the object yet or a
    job will compute this structure, but hasn't been run yet.

    Sub classes that wish to document special behavior of their implementation of :method:`.get_structure` may do so by
    adding documention to it in the "Methods:" sub section of their class docstring.

    The example below shows how to implement this mixin and how to check whether an object derives from it

    >>> from pyiron_atomistics.atomistics.structure.atoms import Atoms
    >>> class Foo(HasStructure):
    ...     '''
    ...     Methods:
    ...         .. method:: get_structure
    ...             returns structure with single Fe atom at (0, 0, 0)
    ...     '''
    ...     def _get_structure(self, iteration_step=-1, wrap_atoms=True):
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
    """

    def get_structure(self, iteration_step=-1, wrap_atoms=True):
        """
        Gets the structure from a given iteration step of the simulation (MD/ionic relaxation). For static calculations
        there is only one ionic iteration step.

        Args:
            iteration_step (int): Step for which the structure is requested, if negative count from the back
            wrap_atoms (bool): True if the atoms are to be wrapped back into the unit cell

        Returns:
            :class:`pyiron_atomistics.atomistics.structure.atoms.Atoms`: the requested structure

        Raises:
            IndexError: if not -:property:`.number_of_structures` <= iteration_step < :property:`.number_of_structures`
        """
        num_structures = self.number_of_structures
        if iteration_step < 0:
            iteration_step += num_structures
        if not (0 <= iteration_step < num_structures):
            raise IndexError(f"iteration_step {iteration_step} out of range [-{num_structures}, {num_structures}).")

        return self._get_structure(iteration_step=iteration_step, wrap_atoms=wrap_atoms)

    @abstractmethod
    def _get_structure(self, iteration_step=-1, wrap_atoms=True):
        pass

    @property
    def number_of_structures(self):
        """
        `int`: maximum `iteration_step` + 1 that can be passed to :method:`.get_structure()`.
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
                               :method:`.get_structure()`

        Yields:
            :class:`pyiron_atomistics.atomistitcs.structure.atoms.Atoms`: every structure attached to the object
        """
        for i in range(self.number_of_structures):
            yield self._get_structure(iteration_step=i, wrap_atoms=wrap_atoms)
