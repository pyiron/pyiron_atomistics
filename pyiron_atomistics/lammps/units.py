# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

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
ANG_PER_FS_TO_BOHR_PER_FS = spc.angstrom / spc.physical_constants['Bohr radius'][0]
ANG_PER_FS_TO_CM_PER_S = (spc.angstrom / spc.femto) / spc.centi
ANG_PER_FS_TO_M_PER_S = spc.angstrom / spc.femto
ANG_TO_BOHR = spc.angstrom / spc.physical_constants['Bohr radius'][0]
ANG_TO_CM = spc.angstrom / spc.centi
ANG_TO_M = spc.angstrom
EL_TO_COUL = spc.elementary_charge
EV_PER_ANG_TO_DYNE = (spc.electron_volt / spc.angstrom) / spc.dyne
EV_PER_ANG_TO_HA_PER_BOHR = spc.physical_constants["electron volt-hartree relationship"][0] * \
                            spc.physical_constants['Bohr radius'][0] / spc.angstrom
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
        "mass": 1.,
        "distance": 1.,
        "time": FS_TO_PS,
        "energy": 1.,
        "velocity": ANG_PER_FS_TO_ANG_PER_PS,
        "force": 1.,
        "temperature": 1.,
        "pressure": GPA_TO_BAR,
        "charge": 1.
    },
    "si": {
        "mass": AMU_TO_KG,
        "distance": ANG_TO_M,
        "time": FS_TO_S,
        "energy": EV_TO_J,
        "velocity": ANG_PER_FS_TO_M_PER_S,
        "force": EV_PER_ANG_TO_N,
        "temperature": 1.,
        "pressure": GPA_TO_PA,
        "charge": EL_TO_COUL
    },
    "cgs": {
        "mass": AMU_TO_G,
        "distance": ANG_TO_CM,
        "time": FS_TO_S,
        "energy": EV_TO_ERG,
        "velocity": ANG_PER_FS_TO_CM_PER_S,
        "force": EV_PER_ANG_TO_DYNE,
        "temperature": 1.,
        "pressure": GPA_TO_BARYE,
        "charge": 4.8032044e-10  # In statCoulombs, but these are deprecated and thus not in scipt.constants
    },
    "real": {
        "mass": 1.,
        "distance": 1.,
        "time": 1.,
        "energy": EV_TO_KCAL_PER_MOL,
        "velocity": 1.,
        "force": EV_PER_ANG_TO_KCAL_PER_MOL_ANG,
        "temperature": 1.,
        "pressure": GPA_TO_ATM,
        "charge": 1.
    },
    "electron": {
        "mass": 1.,
        "distance": ANG_TO_BOHR,
        "time": 1.,
        "energy": EV_TO_HA,
        "velocity": ANG_PER_FS_TO_BOHR_PER_FS,
        "force": EV_PER_ANG_TO_HA_PER_BOHR,
        "temperature": 1.,
        "pressure": GPA_TO_PA,
        "charge": 1.
    },
}

# Hard coded list of all quantities we store in pyiron and the type of quantity it stores (Expand if necessary)
conversion_dict = dict()
conversion_dict["distance"] = ["positions", "cells", "unwrapped_positions"]
conversion_dict["pressure"] = ["pressure", "pressures", "mean_pressures"]
conversion_dict["volume"] = ["volume"]
conversion_dict["time"] = ["time"]
conversion_dict["energy"] = ["energy_tot", "energy_pot"]
conversion_dict["temperature"] = ["temperature", "temperatures"]
conversion_dict["velocity"] = ["velocity"]
conversion_dict["mass"] = ["mass"]
conversion_dict["charge"] = ["charges", "charge"]


# Reverse conversion_dict
quantity_dict = dict()
for key, val in conversion_dict.items():
    for v in val:
        quantity_dict[v] = key


class UnitConverter:

    def __init__(self, units):
        self._units = units
        self._dict = LAMMPS_UNIT_CONVERSIONS[self._units]

    def __getitem__(self, quantity):
        return self._dict[quantity]

    def lammps_to_pyiron(self, quantity):
        return 1. / self[quantity]

    def pyiron_to_lammps(self, quantity):
        return self[quantity]

    def convert_array_to_pyiron_units(self, array, label):
        if label in conversion_dict.keys():
            return array * self.lammps_to_pyiron(conversion_dict[label])
        else:
            warnings.warn(message="Warning: Couldn't determine the LAMMPS to pyiron unit conversion type of quantity "
                                  "{}. Returning un-normalized quantity".format(label))
            return array
