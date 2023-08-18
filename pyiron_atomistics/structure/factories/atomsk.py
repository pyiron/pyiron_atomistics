# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import subprocess
import tempfile
import os.path
import shutil
import io

from pyiron_atomistics.atomistics.structure.atoms import ase_to_pyiron

from ase.io import read, write
import numpy as np

__author__ = "Marvin Poul"
__copyright__ = (
    "Copyright 2021, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "1.0"
__maintainer__ = "Marvin Poul"
__email__ = "poul@mpie.de"
__status__ = "production"
__date__ = "Jun 30, 2021"

_ATOMSK_EXISTS = shutil.which("atomsk") != None


class AtomskError(Exception):
    pass


class AtomskBuilder:
    """Class to build CLI arguments to Atomsk."""

    def __init__(self):
        self._options = []
        self._structure = None

    @classmethod
    def create(cls, lattice, a, *species, c=None, hkl=None):
        """
        Instiate new builder and add create mode.

        See https://atomsk.univ-lille.fr/doc/en/mode_create.html or :meth:`.AtomskFactory.create` for arguments.

        Returns:
            :class:`.AtomskBuilder`: self
        """

        self = cls()

        a_and_c = str(a) if c is None else f"{a} {c}"
        line = f"--create {lattice} {a_and_c} {' '.join(species)}"
        if hkl is not None:
            if np.asarray(hkl).shape not in ((3, 3), (3, 4)):
                raise ValueError(
                    f"hkl must have shape 3x3 or 3x4 if provided, not {hkl}!"
                )
            line += " orient " + "  ".join(
                "[" + "".join(map(str, a)) + "]" for a in hkl
            )
        # TODO: check len(species) etc. with the document list of supported phases
        self._options.append(line)
        return self

    @classmethod
    def modify(cls, structure):
        """ "
        Instiate new builder to modify and existing structure.

        See :meth:`.AtomskFactory.modify` for arguments.

        Returns:
            :class:`.AtomskBuilder`: self
        """
        self = cls()
        self._structure = structure
        self._options.append("input.exyz")
        return self

    def duplicate(self, nx, ny=None, nz=None):
        """
        See https://atomsk.univ-lille.fr/doc/en/option_duplicate.html

        Args:
            nx (int): replicas in x directions
            ny (int, optional): replicas in y directions, default to nx
            nz (int, optional): replicas in z directions, default to ny

        Returns:
            :class:`.AtomskBuilder`: self
        """
        if ny is None:
            ny = nx
        if nz is None:
            nz = ny
        self._options.append(f"-duplicate {nx} {ny} {nz}")
        return self

    def build(self):
        """
        Call Atomsk with the options accumulated so far.

        Returns:
            :class:`.Atoms`: new structure
        """
        self._options.append("- exyz")  # output to stdout as exyz format
        with tempfile.TemporaryDirectory() as temp_dir:
            if self._structure is not None:
                write(
                    os.path.join(temp_dir, "input.exyz"),
                    self._structure,
                    format="extxyz",
                )
            proc = subprocess.run(
                ["atomsk", *" ".join(self._options).split()],
                capture_output=True,
                cwd=temp_dir,
            )
            output = proc.stdout.decode("utf8")
            for l in output.split("\n"):
                if l.strip().startswith("X!X ERROR:"):
                    raise AtomskError(f"atomsk returned error: {output}")
            return ase_to_pyiron(read(io.StringIO(output), format="extxyz"))

    def __getattr__(self, name):
        # magic method to map method calls of the form self.foo_bar to options like -foo-bar; arguments converted str
        # and appended after option, keyword arguments are mapped to strings like 'key value'
        def meth(*args, **kwargs):
            args_str = " ".join(map(str, args))
            kwargs_str = " ".join(f"{k} {v}" for k, v in kwargs.items())
            self._options.append(f"-{name.replace('_', '-')} {args_str} {kwargs_str}")
            return self

        return meth


class AtomskFactory:
    """
    Wrapper around the atomsk CLI.

    Use :meth:`.create()` to create a new structure and :meth:`.modify()` to pass an existing structure to atomsk.
    Both of them return a :class:`.AtomskBuilder`, which has methods named like the flags of atomsk.  Calling them with
    the appropriate arguments adds the flags to the command line.  Once you added all flags, call
    :meth:`.AtomskBuilder.build()` to create the new structure.  All methods to add flags return the
    :class:`AtomskBuilder` instance they are called on to allow method chaining.

    >>> from pyiron_atomistics import Project
    >>> pr = Project('atomsk')
    >>> pr.create.structure.atomsk.create("fcc", 3.6, "Cu").duplicate(2, 1, 1).build()
    Cu: [0. 0. 0.]
    Cu: [1.8 1.8 0. ]
    Cu: [0.  1.8 1.8]
    Cu: [1.8 0.  1.8]
    Cu: [3.6 0.  0. ]
    Cu: [5.4 1.8 0. ]
    Cu: [3.6 1.8 1.8]
    Cu: [5.4 0.  1.8]
    pbc: [ True  True  True]
    cell:
    Cell([7.2, 3.6, 3.6])
    >>> s = pr.create.structure.atomsk.create("fcc", 3.6, "Cu").duplicate(2, 1, 1).build()
    >>> pr.create.structure.atomsk.modify(s).cell("add", 3, "x").build()
    Cu: [0. 0. 0.]
    Cu: [1.8 1.8 0. ]
    Cu: [0.  1.8 1.8]
    Cu: [1.8 0.  1.8]
    Cu: [3.6 0.  0. ]
    Cu: [5.4 1.8 0. ]
    Cu: [3.6 1.8 1.8]
    Cu: [5.4 0.  1.8]
    pbc: [ True  True  True]
    cell:
    Cell([10.2, 3.6, 3.6])

    Methods that you call on :class:`.AtomskBuilder` are automatically translated into options, translating '_' in the
    method name to '-' and appending all arguments as strings.  All atomsk options are therefore supported, but no error
    checking is performed whether the translated options exist or follow the syntax prescribed by atomsk, except for
    special cases defined on the class.
    """

    def create(self, lattice, a, *species, c=None, hkl=None):
        """
        Create a new structure with Atomsk.

        See https://atomsk.univ-lille.fr/doc/en/mode_create.html for supported lattices.

        Call :meth:`.AtomskBuilder.build()` on the returned object to actually create a structure.

        Args:
            lattice (str): lattice type to create
            a (float): first lattice parameter
            *species (list of str): chemical short symbols for the type of atoms to create, length depends on lattice
                                    type
            c (float, optional): third lattice parameter, only necessary for some lattice types
            hkl (array of int, (3,3) or (3,4)): three hkl vectors giving the crystallographic axes that should point along the x,
                                       y, z directions

        Returns:
            AtomskBuilder: builder instances
        """
        return AtomskBuilder.create(lattice, a, *species, c=c, hkl=hkl)

    def modify(self, structure):
        """
        Modify existing structure with Atomsk.

        Call :meth:`.AtomskBuilder.build()` on the returned object to actually create a structure.

        Args:
            structure (:class:`.Atoms`): input structure

        Returns:
            AtomskBuilder: builder instances
        """
        return AtomskBuilder.modify(structure)
