import os

from pyiron_atomistics.vasp.volumetric_data import VaspVolumetricData


def get_valence_and_total_charge_density(working_directory):
    """
    Gives the valence and total charge densities

    Returns:
        tuple: The required charge densities
    """
    cd_core = VaspVolumetricData()
    cd_total = VaspVolumetricData()
    cd_val = VaspVolumetricData()
    if os.path.isfile(working_directory + "/AECCAR0"):
        cd_core.from_file(working_directory + "/AECCAR0")
        cd_val.from_file(working_directory + "/AECCAR2")
        cd_val.atoms = cd_val.atoms
        cd_total.total_data = cd_core.total_data + cd_val.total_data
        cd_total.atoms = cd_val.atoms
    return cd_val, cd_total
