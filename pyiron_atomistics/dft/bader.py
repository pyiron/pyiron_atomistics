import os

from pyiron_vasp.dft.bader import Bader as _Bader

from pyiron_atomistics.vasp.volumetric_data import VaspVolumetricData


class Bader(_Bader):
    def _create_cube_files(self):
        """
        Create CUBE format files of the total and valce charges to be used by the Bader program
        """
        cd_val, cd_total = get_valence_and_total_charge_density(
            working_directory=self._working_directory
        )
        cd_val.write_cube_file(
            filename=os.path.join(self._working_directory, "valence_charge.CUBE")
        )
        cd_total.write_cube_file(
            filename=os.path.join(self._working_directory, "total_charge.CUBE")
        )


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
