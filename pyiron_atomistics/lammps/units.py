# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import numpy as np
import scipy.constants as spc
import warnings

__author__ = "Joerg Neugebauer, Sudarsan Surendralal, Jan Janssen"
__copyright__ = (
    "Copyright 2021, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)


# Conversion factors for transfroming pyiron units to Lammps units (alphabetical)

AMU_TO_G = spc.atomic_mass * spc.kilo
AMU_TO_KG = spc.atomic_mass
ANG_PER_FS_TO_ANG_PER_PS = spc.pico / spc.femto
ANG_PER_FS_TO_BOHR_PER_FS = spc.angstrom / spc.physical_constants["Bohr radius"][0]
ANG_PER_FS_TO_CM_PER_S = (spc.angstrom / spc.femto) / spc.centi
ANG_PER_FS_TO_M_PER_S = spc.angstrom / spc.femto
ANG_TO_BOHR = spc.angstrom / spc.physical_constants["Bohr radius"][0]
ANG_TO_CM = spc.angstrom / spc.centi
ANG_TO_M = spc.angstrom
EL_TO_COUL = spc.elementary_charge
EV_PER_ANG_TO_DYNE = (spc.electron_volt / spc.angstrom) / spc.dyne
EV_PER_ANG_TO_HA_PER_BOHR = (
    spc.physical_constants["electron volt-hartree relationship"][0]
    * spc.physical_constants["Bohr radius"][0]
    / spc.angstrom
)
EV_PER_ANG_TO_KCAL_PER_MOL_ANG = spc.eV / (spc.kilo * spc.calorie / spc.N_A)
EV_PER_ANG_TO_N = spc.electron_volt / spc.angstrom
EV_TO_ERG = spc.electron_volt / spc.erg
EV_TO_HA = spc.physical_constants["electron volt-hartree relationship"][0]
EV_TO_J = spc.electron_volt
EV_TO_KCAL_PER_MOL = spc.eV / (spc.kilo * spc.calorie / spc.N_A)
FS_TO_PS = spc.femto / spc.pico
FS_TO_S = spc.femto
GPA_TO_ATM = spc.giga / spc.atm
GPA_TO_BAR = spc.giga / spc.bar
GPA_TO_BARYE = spc.giga / (spc.micro * spc.bar)  # "barye" = 1e-6 bar
GPA_TO_PA = spc.giga

# Conversions for most of the Lammps units to Pyiron units
# Lammps units source doc: https://lammps.sandia.gov/doc/units.html
# Pyrion units source doc: https://pyiron.readthedocs.io/en/latest/source/faq.html
# At time of writing, not all these conversion factors are used, but may be helpful later.

LAMMPS_UNIT_CONVERSIONS = {
    "metal": {
        "mass": 1.0,
        "distance": 1.0,
        "time": FS_TO_PS,
        "energy": 1.0,
        "velocity": ANG_PER_FS_TO_ANG_PER_PS,
        "force": 1.0,
        "temperature": 1.0,
        "pressure": GPA_TO_BAR,
        "charge": 1.0,
        "natoms": 1,
    },
    "si": {
        "mass": AMU_TO_KG,
        "distance": ANG_TO_M,
        "time": FS_TO_S,
        "energy": EV_TO_J,
        "velocity": ANG_PER_FS_TO_M_PER_S,
        "force": EV_PER_ANG_TO_N,
        "temperature": 1.0,
        "pressure": GPA_TO_PA,
        "charge": EL_TO_COUL,
        "natoms": 1,
    },
    "cgs": {
        "mass": AMU_TO_G,
        "distance": ANG_TO_CM,
        "time": FS_TO_S,
        "energy": EV_TO_ERG,
        "velocity": ANG_PER_FS_TO_CM_PER_S,
        "force": EV_PER_ANG_TO_DYNE,
        "temperature": 1.0,
        "pressure": GPA_TO_BARYE,
        "charge": 4.8032044e-10,  # In statCoulombs, but these are deprecated and thus not in scipt.constants
        "natoms": 1,
    },
    "real": {
        "mass": 1.0,
        "distance": 1.0,
        "time": 1.0,
        "energy": EV_TO_KCAL_PER_MOL,
        "velocity": 1.0,
        "force": EV_PER_ANG_TO_KCAL_PER_MOL_ANG,
        "temperature": 1.0,
        "pressure": GPA_TO_ATM,
        "charge": 1.0,
        "natoms": 1,
    },
    "electron": {
        "mass": 1.0,
        "distance": ANG_TO_BOHR,
        "time": 1.0,
        "energy": EV_TO_HA,
        "velocity": ANG_PER_FS_TO_BOHR_PER_FS,
        "force": EV_PER_ANG_TO_HA_PER_BOHR,
        "temperature": 1.0,
        "pressure": GPA_TO_PA,
        "charge": 1.0,
        "natoms": 1,
    },
}

# Define conversion for volumes based on distances
for values in LAMMPS_UNIT_CONVERSIONS.values():
    values["volume"] = values["distance"] ** 3
    values["dimensionless_integer_quantity"] = 1


# Hard coded list of all quantities we store in pyiron and the type of quantity it stores (Expand if necessary)
_conversion_dict = dict()
_conversion_dict["distance"] = [
    "positions",
    "cells",
    "unwrapped_positions",
    "mean_unwrapped_positions",
]
_conversion_dict["volume"] = ["volume", "volumes"]
_conversion_dict["pressure"] = ["pressure", "pressures", "mean_pressures"]
_conversion_dict["time"] = ["time"]
_conversion_dict["energy"] = [
    "energy_tot",
    "energy_pot",
    "energy_pot_per_atom",
    "energy_kin_per_atom",
    "mean_energy_pot",
]
_conversion_dict["temperature"] = ["temperature", "temperatures"]
_conversion_dict["velocity"] = ["velocity", "velocities", "mean_velocities"]
_conversion_dict["mass"] = ["mass"]
_conversion_dict["charge"] = ["charges", "charge"]
_conversion_dict["force"] = ["forces", "force", "mean_forces"]
_conversion_dict["dimensionless_integer_quantity"] = ["steps", "indices"]
_conversion_dict["natoms"] = ["natoms"]

# Reverse _conversion_dict
quantity_dict = dict()
for key, val in _conversion_dict.items():
    for v in val:
        quantity_dict[v] = key

# Data type dict to ensure that the units are converted to the proper units
dtype_dict = dict()
for key, val in _conversion_dict.items():
    for v in val:
        # Everything except dimensionless integer quantities are stored as floats
        if key in ["dimensionless_integer_quantity"]:
            dtype_dict[v] = int
        else:
            dtype_dict[v] = np.float64


class UnitConverter:
    """This is a class to aid conversion of physical quantities between LAMMPS and pyiron units."""

    def __init__(self, units):
        """
        Initialize the class by specifiying the type of lammps units used

        Args:
            units (str): The type of LAMMPS units used (eg. metal, real, cgs, lj, etc.)

        """
        self._units = units
        self._dict = LAMMPS_UNIT_CONVERSIONS[self._units]

    def __getitem__(self, quantity):
        """
        Get quantity from `_dict`

        Args:
            quantity (str): The physical quantity

        Returns:
            float: The conversion factor
        """
        return self._dict[quantity]

    def lammps_to_pyiron(self, quantity):
        """
        Get the conversion factor for a given physical quantity to be converted from LAMMPS to pyiron units

        Args:
            quantity (str): The physical quantity (must be a key in the dictionary `_conversion_dict`)

        Returns:

            float: The conversion factor

        """
        return 1.0 / self[quantity]

    def pyiron_to_lammps(self, quantity):
        """
        Get the conversion factor for a given physical quantity to be converted from pyiron to LAMMPS units

        Args:
            quantity (str): The physical quantity (must be a key in the dictionary `_conversion_dict`)

        Returns:

            float: The conversion factor

        """
        return self[quantity]

    def convert_array_to_pyiron_units(self, array, label):
        """
        Convert a labelled numpy array into pyiron units based on the physical quantity the label corresponds to

        Args:
            array (numpy.ndarray/list): The array to be converted
            label (str): The label of the quantity (must be a key in the dictionary `quantity_dict`)

        Returns:
            ndarray: The array after conversion

        """
        if label in quantity_dict.keys():
            return np.array(
                array * self.lammps_to_pyiron(quantity_dict[label]), dtype_dict[label]
            )
        else:
            warnings.warn(
                message="Warning: Couldn't determine the LAMMPS to pyiron unit conversion type of quantity "
                "{}. Returning un-normalized quantity".format(label)
            )
            return array
