# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import os
import math
import warnings

import numpy as np
from pyiron_atomistics.vasp.structure import (
    atoms_from_string,
    get_species_list_from_potcar,
)
from pyiron_vasp.vasp.volumetric_data import VaspVolumetricData as _VaspVolumetricData

__author__ = "Sudarsan Surendralal"
__copyright__ = (
    "Copyright 2021, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "1.0"
__maintainer__ = "Sudarsan Surendralal"
__email__ = "surendralal@mpie.de"
__status__ = "production"
__date__ = "Sep 1, 2017"


class VaspVolumetricData(_VaspVolumetricData):
    @staticmethod
    def _read_vol_data_old(filename, normalize=True):
        """
        Convenience method to parse a generic volumetric static file in the vasp like format.
        Used by subclasses for parsing the file. This routine is adapted from the pymatgen vasp VolumetricData
        class with very minor modifications. The new parser is faster

        http://pymatgen.org/_modules/pymatgen/io/vasp/outputs.html#VolumetricData.

        Args:
            filename (str): Path of file to parse
            normalize (boolean): Flag to normalize by the volume of the cell

        """
        if os.stat(filename).st_size == 0:
            warnings.warn("File:" + filename + "seems to be corrupted/empty")
            return None, None
        poscar_read = False
        poscar_string = list()
        dataset = list()
        all_dataset = list()
        dim = None
        dimline = None
        read_dataset = False
        ngrid_pts = 0
        data_count = 0
        atoms = None
        volume = None
        with open(filename, "r") as f:
            for line in f:
                line = line.strip()
                if read_dataset:
                    toks = line.split()
                    for tok in toks:
                        if data_count < ngrid_pts:
                            # This complicated procedure is necessary because
                            # vasp outputs x as the fastest index, followed by y
                            # then z.
                            x = data_count % dim[0]
                            y = int(math.floor(data_count / dim[0])) % dim[1]
                            z = int(math.floor(data_count / dim[0] / dim[1]))
                            dataset[x, y, z] = float(tok)
                            data_count += 1
                    if data_count >= ngrid_pts:
                        read_dataset = False
                        data_count = 0
                        all_dataset.append(dataset)
                elif not poscar_read:
                    if line != "" or len(poscar_string) == 0:
                        poscar_string.append(line)
                    elif line == "":
                        try:
                            atoms = atoms_from_string(poscar_string)
                        except ValueError:
                            pot_str = filename.split("/")
                            pot_str[-1] = "POTCAR"
                            potcar_file = "/".join(pot_str)
                            species = get_species_list_from_potcar(potcar_file)
                            atoms = atoms_from_string(
                                poscar_string, species_list=species
                            )
                        volume = atoms.get_volume()
                        poscar_read = True
                elif not dim:
                    dim = [int(i) for i in line.split()]
                    ngrid_pts = dim[0] * dim[1] * dim[2]
                    dimline = line
                    read_dataset = True
                    dataset = np.zeros(dim)
                elif line == dimline:
                    read_dataset = True
                    dataset = np.zeros(dim)
            if not normalize:
                volume = 1.0
            if len(all_dataset) == 0:
                warnings.warn("File:" + filename + "seems to be corrupted/empty")
                return None, None
            if len(all_dataset) == 2:
                data = {
                    "total": all_dataset[0] / volume,
                    "diff": all_dataset[1] / volume,
                }
                return atoms, [data["total"], data["diff"]]
            else:
                data = {"total": all_dataset[0] / volume}
                return atoms, [data["total"]]

    def _read_vol_data(self, filename, normalize=True):
        """
        Parses the VASP volumetric type files (CHGCAR, LOCPOT, PARCHG etc). Rather than looping over individual values,
        this function utilizes numpy indexing resulting in a parsing efficiency of at least 10%.

        Args:
            filename (str): File to be parsed
            normalize (bool): Normalize the data with respect to the volume (Recommended for CHGCAR files)

        Returns:
            pyiron.atomistics.structure.atoms.Atoms: The structure of the volumetric snapshot
            list: A list of the volumetric data (length >1 for CHGCAR files with spin)

        """
        if not os.path.getsize(filename) > 0:
            warnings.warn("File:" + filename + "seems to be empty! ")
            return None, None
        with open(filename, "r") as f:
            struct_lines = list()
            get_grid = False
            n_x = 0
            n_y = 0
            n_z = 0
            n_grid = 0
            n_grid_str = None
            total_data_list = list()
            atoms = None
            for line in f:
                strip_line = line.strip()
                if not get_grid:
                    if strip_line == "":
                        get_grid = True
                    struct_lines.append(strip_line)
                elif n_grid_str is None:
                    n_x, n_y, n_z = [int(val) for val in strip_line.split()]
                    n_grid = n_x * n_y * n_z
                    n_grid_str = " ".join([str(val) for val in [n_x, n_y, n_z]])
                    load_txt = np.genfromtxt(f, max_rows=int(n_grid / 5))
                    load_txt = np.hstack(load_txt)
                    if n_grid % 5 != 0:
                        add_line = np.genfromtxt(f, max_rows=1)
                        load_txt = np.append(load_txt, np.hstack(add_line))
                    total_data = self._fastest_index_reshape(load_txt, [n_x, n_y, n_z])
                    try:
                        atoms = atoms_from_string(struct_lines)
                    except ValueError:
                        pot_str = filename.split("/")
                        pot_str[-1] = "POTCAR"
                        potcar_file = "/".join(pot_str)
                        species = get_species_list_from_potcar(potcar_file)
                        atoms = atoms_from_string(struct_lines, species_list=species)
                    if normalize:
                        total_data /= atoms.get_volume()
                    total_data_list.append(total_data)
                elif atoms is not None:
                    grid_str = n_grid_str.replace(" ", "")
                    if grid_str == strip_line.replace(" ", ""):
                        load_txt = np.genfromtxt(f, max_rows=int(n_grid / 5))
                        load_txt = np.hstack(load_txt)
                        if n_grid % 5 != 0:
                            add_line = np.genfromtxt(f, max_rows=1)
                            load_txt = np.hstack(
                                np.append(load_txt, np.hstack(add_line))
                            )
                        total_data = self._fastest_index_reshape(
                            load_txt, [n_x, n_y, n_z]
                        )
                        if normalize:
                            total_data /= atoms.get_volume()
                        total_data_list.append(total_data)
            if len(total_data_list) == 0:
                warnings.warn(
                    "File:"
                    + filename
                    + "seems to be corrupted/empty even after parsing!"
                )
                return None, None
            return atoms, total_data_list
