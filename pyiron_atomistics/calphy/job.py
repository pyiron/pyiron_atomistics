import os
import numpy as np
from typing import Union, List, Tuple
import pandas as pd

from pyiron_base import DataContainer
from pyiron_atomistics.lammps.potential import LammpsPotential, LammpsPotentialFile
from pyiron_base import GenericJob, ImportAlarm
from pyiron_atomistics.lammps.structure import LammpsStructure, UnfoldingPrism

with ImportAlarm(
    "Calphy functionality requires the `calphy` module (and its dependencies) specified as extra"
    "requirements. Please install it and try again."
) as calphy_alarm:
    from calphy import Calculation, Solid, Liquid, Alchemy
    from calphy.routines import routine_fe, routine_ts, routine_alchemy, routine_pscale

__author__ = "Sarath Menon"
__copyright__ = (
    "Copyright 2021, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "1.0"
__maintainer__ = "Sarath Menon"
__email__ = "s.menon@mpie.de"
__status__ = "development"
__date__ = "April 1, 2022"

inputdict = {
    "mode": None,
    "pressure": None,
    "temperature": None,
    "reference_phase": None,
    "npt": None,
    "n_equilibration_steps": 15000,
    "n_switching_steps": 25000,
    "n_print_steps": 0,
    "n_iterations": 1,
    "md": {
        "timestep": 0.001,
        "n_small_steps": 10000,
        "n_every_steps": 10,
        "n_repeat_steps": 10,
        "n_cycles": 100,
        "thermostat_damping": 0.5,
        "barostat_damping": 0.1,
    },
    "tolerance": {
        "lattice_constant": 0.0002,
        "spring_constant": 0.01,
        "solid_fraction": 0.7,
        "liquid_fraction": 0.05,
        "pressure": 0.5,
    },
}


class Calphy(GenericJob):
    """
    Class to set up and run calphy jobs for calculation of free energies using LAMMPS.

    An input structure (:attr:`structure`) and interatomic potential (:attr:`potential`) are necessary input options. The additional input options such as the temperature and pressure are specified in the :meth:`.calc_free_energy` method. Depending on the input parameters, a corresponding calculation mode is selected. Further input options can be accessed through :attr:`input.md` and :attr:`input.tolerance`.

    An example which calculates the free energy of Cu using an interatomic potential:

    ```python
    job.structure = pr.create.structure.ase.bulk('Cu', cubic=True).repeat(5)
    job.potential = "2001--Mishin-Y--Cu-1--LAMMPS--ipr1"
    job.calc_free_energy(temperature=1100, pressure=0, reference_phase="solid")
    job.run()
    ```

    In order to calculate the free energy of the liquid phase, the `reference_phase` should be set to `liquid`.

    The different modes can be selected as follows:

    For free energy at a given temperature and pressure:

    ```python
    job.calc_free_energy(temperature=1100, pressure=0, reference_phase="solid")
    ```

    Alternatively, :func:`calc_mode_fe` can be used.

    To obtain the free energy between a given temperature range (temperature scaling):

    ```python
    job.calc_free_energy(temperature=[1100, 1400], pressure=0, reference_phase="solid")
    ```

    Alternatively, :func:`calc_mode_ts` can be used.

    For free energy between a given pressure range (pressure scaling)

    ```python
    job.calc_free_energy(temperature=1000, pressure=[0, 100000], reference_phase="solid")
    ```

    Alternatively, :func:`calc_mode_pscale` can be used.

    To obtain the free energy difference between two interatomic potentials (alchemy/upsampling)

    ```python
    job.potential = ["2001--Mishin-Y--Cu-1--LAMMPS--ipr1", "1986--Foiles-S-M--Cu--LAMMPS--ipr1"]
    job.calc_free_energy(temperature=1100, pressure=0, reference_phase="solid")
    job.run()
    ```

    Alternatively, :func:`calc_mode_alchemy` can be used.

    The way `pressure` is specified determines how the barostat affects the system. For isotropic pressure control:

    ```python
    job.calc_free_energy(temperature=[1100, 1400], pressure=0, reference_phase="solid")
    ```

    For anisotropic pressure control:

    ```python
    job.calc_free_energy(temperature=[1100, 1400], pressure=[0, 0, 0], reference_phase="solid")
    ```

    To constrain the lattice:

    ```python
    job.calc_free_energy(temperature=[1100, 1400], pressure=None, reference_phase="solid")
    ```

    In addition the boolean option :attr:`input.npt` can be used to determine the MD ensemble. If True, temperature integration and alchemy/upsampling are carried out in the NPT ensemble. If False, the NVT ensemble is employed.

    After the calculation is over, the various output options can be accessed through `job.output`.
    """

    def __init__(self, project, job_name):
        super().__init__(project, job_name)
        self.input = DataContainer(inputdict, table_name="inputdata")
        self._potential_initial = None
        self._potential_final = None
        self.input.potential_initial_name = None
        self.input.potential_final_name = None
        self.input.structure = None
        self.output = DataContainer(table_name="output")
        self._data = None
        self.input._pot_dict_initial = None
        self.input._pot_dict_final = None

    def set_potentials(self, potential_filenames: Union[list, str]):
        """
        Set the interatomic potential from a given name

        Args:
            potential_filenames (list, str): list of filenames

        Returns:
            None
        """
        if not isinstance(potential_filenames, list):
            potential_filenames = [potential_filenames]
        if len(potential_filenames) > 0:
            if isinstance(potential_filenames[0], pd.DataFrame):
                potential = potential_filenames[0]
                self.input._pot_dict_initial = potential  # .to_dict()
            else:
                potential = LammpsPotentialFile().find_by_name(potential_filenames[0])
                self.input.potential_initial_name = potential_filenames[0]
            self._potential_initial = LammpsPotential()
            self._potential_initial.df = potential

        if len(potential_filenames) > 1:
            if isinstance(potential_filenames[1], pd.DataFrame):
                potential = potential_filenames[1]
                self.input._pot_dict_final = potential  # .to_dict()
            else:
                potential = LammpsPotentialFile().find_by_name(potential_filenames[1])
                self.input.potential_final_name = potential_filenames[1]
            self._potential_final = LammpsPotential()
            self._potential_final.df = potential
        if len(potential_filenames) > 2:
            raise ValueError("Maximum two potentials can be provided")

    def get_potentials(self) -> List[str]:
        """
        Return the interatomic potentials

        Args:
            None

        Returns:
            list of str: list of interatomic potentials
        """
        if self._potential_final is None:
            return [self._potential_initial.df]
        else:
            return [self._potential_initial.df, self._potential_final.df]

    def copy_pot_files(self):
        """
        Copy potential files to the working directory

        Args:
            None

        Returns:
            None
        """
        if self._potential_initial is not None:
            self._potential_initial.copy_pot_files(self.working_directory)
        if self._potential_final is not None:
            self._potential_final.copy_pot_files(self.working_directory)

    def _prepare_pair_styles(self) -> Tuple[List, List]:
        """
        Prepare pair style and pair coeff

        Args:
            None

        Returns:
            list: pair style and pair coeff
        """
        pair_style = []
        pair_coeff = []
        if self._potential_initial is not None:
            pair_style.append(
                self._potential_initial.df["Config"]
                .to_list()[0][0]
                .strip()
                .split()[1:][0]
            )
            pair_coeff.append(
                " ".join(
                    self._potential_initial.df["Config"]
                    .to_list()[0][1]
                    .strip()
                    .split()[1:]
                )
            )
        if self._potential_final is not None:
            pair_style.append(
                self._potential_final.df["Config"]
                .to_list()[0][0]
                .strip()
                .split()[1:][0]
            )
            pair_coeff.append(
                " ".join(
                    self._potential_final.df["Config"]
                    .to_list()[0][1]
                    .strip()
                    .split()[1:]
                )
            )
        return pair_style, pair_coeff

    def _potential_from_hdf(self):
        """
        Recreate the potential from filename stored in hdf5

        Args:
            None

        Returns:
            None
        """
        filenames = []
        if self.input.potential_initial_name is not None:
            filenames.append(self.input.potential_initial_name)
        elif self.input._pot_dict_initial is not None:
            filenames.append(pd.DataFrame(data=self.input._pot_dict_initial))
        if self.input.potential_final_name is not None:
            filenames.append(self.input.potential_final_name)
        elif self.input._pot_dict_final is not None:
            filenames.append(pd.DataFrame(data=self.input._pot_dict_final))

        self.set_potentials(filenames)

    @property
    def potential(self):
        potentials = self.get_potentials()
        if len(potentials) == 1:
            return potentials[0]
        return potentials

    @potential.setter
    def potential(self, potential_filenames):
        self.set_potentials(potential_filenames)

    @property
    def structure(self):
        return self.input.structure

    @structure.setter
    def structure(self, val):
        self.input.structure = val

    def view_potentials(self) -> List:
        """
        View a list of available interatomic potentials

        Args:
            None

        Returns:
            list: list of available potentials
        """
        if not self.structure:
            raise ValueError("please assign a structure first")
        else:
            list_of_elements = set(self.structure.get_chemical_symbols())
        list_of_potentials = LammpsPotentialFile().find(list_of_elements)
        if list_of_potentials is not None:
            return list_of_potentials
        else:
            raise TypeError(
                "No potentials found for this kind of structure: ",
                str(list_of_elements),
            )

    def list_potentials(self):
        """
        List of interatomic potentials suitable for the current atomic structure.

        use self.potentials_view() to get more details.

        Args:
            None

        Returns:
            list: potential names
        """
        return list(self.view_potentials()["Name"].values)

    def structure_to_lammps(self):
        """
        Convert structure to LAMMPS structure

        Args:
            None

        Returns:
            list: pair style and pair coeff
        """
        prism = UnfoldingPrism(self.input.structure.cell)
        lammps_structure = self.input.structure.copy()
        lammps_structure.set_cell(prism.A)
        lammps_structure.positions = np.matmul(self.input.structure.positions, prism.R)
        return lammps_structure

    def write_structure(self, structure, file_name: str, working_directory: str):
        """
        Write structure to file

        Args:
            structure: input structure
            file_name (str): output file name
            working_directory (str): output working directory

        Returns:
            None
        """
        lmp_structure = LammpsStructure()
        lmp_structure.potential = self._potential_initial
        lmp_structure.el_eam_lst = set(structure.get_chemical_symbols())
        lmp_structure.structure = self.structure_to_lammps()
        lmp_structure.write_file(file_name=file_name, cwd=working_directory)

    def determine_mode(self):
        """
        Determine the calculation mode

        Args:
            None

        Returns:
            None
        """
        if len(self.get_potentials()) == 2:
            self.input.mode = "alchemy"
            self.input.reference_phase = "alchemy"
        elif isinstance(self.input.pressure, list):
            if len(self.input.pressure) == 2:
                self.input.mode = "pscale"
        elif isinstance(self.input.temperature, list):
            if len(self.input.temperature) == 2:
                self.input.mode = "ts"
        else:
            self.input.mode = "fe"
        # if mode was not set, raise Error
        if self.input.mode is None:
            raise RuntimeError("Could not determine the mode")

    def write_input(self):
        """
        Write input for calphy calculation

        Args:
            None

        Returns:
            None
        """
        # now prepare the calculation
        calc = Calculation()
        for key in inputdict.keys():
            if key not in ["md", "tolerance"]:
                setattr(calc, key, self.input[key])
        for key in inputdict["md"].keys():
            setattr(calc.md, key, self.input["md"][key])
        for key in inputdict["tolerance"].keys():
            setattr(calc.tolerance, key, self.input["tolerance"][key])

        file_name = "conf.data"
        self.write_structure(self.structure, file_name, self.working_directory)
        calc.lattice = os.path.join(self.working_directory, "conf.data")

        self.copy_pot_files()
        pair_style, pair_coeff = self._prepare_pair_styles()
        calc._fix_potential_path = False
        calc.pair_style = pair_style
        calc.pair_coeff = pair_coeff

        calc.element = list(np.unique(self.structure.get_chemical_symbols()))
        calc.mass = list(np.unique(self.structure.get_masses()))

        calc.queue.cores = self.server.cores
        self.calc = calc

    def calc_mode_fe(
        self,
        temperature: float = None,
        pressure: Union[list, float, None] = None,
        reference_phase: str = None,
        n_equilibration_steps: int = 15000,
        n_switching_steps: int = 25000,
        n_print_steps: int = 0,
        n_iterations: int = 1,
    ):
        """
        Calculate free energy at given conditions

        Args:
            None

        Returns:
            None
        """
        if temperature is None:
            raise ValueError("provide a temperature")
        if reference_phase is None:
            raise ValueError("provide a reference_phase")

        self.input.temperature = temperature
        self.input.pressure = pressure
        self.input.reference_phase = reference_phase
        self.input.n_equilibration_steps = n_equilibration_steps
        self.input.n_switching_steps = n_switching_steps
        self.input.n_print_steps = n_print_steps
        self.input.n_iterations = n_iterations
        self.input.mode = "fe"

    def calc_mode_ts(
        self,
        temperature: float = None,
        pressure: Union[list, float, None] = None,
        reference_phase: str = None,
        n_equilibration_steps: int = 15000,
        n_switching_steps: int = 25000,
        n_print_steps: int = 0,
        n_iterations: int = 1,
    ):
        """
        Calculate free energy between given temperatures

        Args:
            None

        Returns:
            None
        """
        if temperature is None:
            raise ValueError("provide a temperature")
        if reference_phase is None:
            raise ValueError("provide a reference_phase")

        self.input.temperature = temperature
        self.input.pressure = pressure
        self.input.reference_phase = reference_phase
        self.input.n_equilibration_steps = n_equilibration_steps
        self.input.n_switching_steps = n_switching_steps
        self.input.n_print_steps = n_print_steps
        self.input.n_iterations = n_iterations
        self.input.mode = "ts"

    def calc_mode_alchemy(
        self,
        temperature: float = None,
        pressure: Union[list, float, None] = None,
        reference_phase: str = None,
        n_equilibration_steps: int = 15000,
        n_switching_steps: int = 25000,
        n_print_steps: int = 0,
        n_iterations: int = 1,
    ):
        """
        Perform upsampling/alchemy between two interatomic potentials

        Args:
            None

        Returns:
            None
        """
        if temperature is None:
            raise ValueError("provide a temperature")
        self.input.temperature = temperature
        self.input.pressure = pressure
        self.input.reference_phase = reference_phase
        self.input.n_equilibration_steps = n_equilibration_steps
        self.input.n_switching_steps = n_switching_steps
        self.input.n_print_steps = n_print_steps
        self.input.n_iterations = n_iterations
        self.input.mode = "alchemy"

    def calc_mode_pscale(
        self,
        temperature: float = None,
        pressure: Union[list, float, None] = None,
        reference_phase: str = None,
        n_equilibration_steps: int = 15000,
        n_switching_steps: int = 25000,
        n_print_steps: int = 0,
        n_iterations: int = 1,
    ):
        """
        Calculate free energy between two given pressures

        Args:
            None

        Returns:
            None
        """
        if temperature is None:
            raise ValueError("provide a temperature")
        if reference_phase is None:
            raise ValueError("provide a reference_phase")

        self.input.temperature = temperature
        self.input.pressure = pressure
        self.input.reference_phase = reference_phase
        self.input.n_equilibration_steps = n_equilibration_steps
        self.input.n_switching_steps = n_switching_steps
        self.input.n_print_steps = n_print_steps
        self.input.n_iterations = n_iterations
        self.input.mode = "pscale"

    def calc_free_energy(
        self,
        temperature: float = None,
        pressure: Union[list, float, None] = None,
        reference_phase: str = None,
        n_equilibration_steps: int = 15000,
        n_switching_steps: int = 25000,
        n_print_steps: int = 0,
        n_iterations: int = 1,
    ):
        """
        Calculate free energy at given conditions

        Args:
            None

        Returns:
            None
        """
        if temperature is None:
            raise ValueError("provide a temperature")
        self.input.temperature = temperature
        self.input.pressure = pressure
        self.input.reference_phase = reference_phase
        self.input.n_equilibration_steps = n_equilibration_steps
        self.input.n_switching_steps = n_switching_steps
        self.input.n_print_steps = n_print_steps
        self.input.n_iterations = n_iterations
        self.determine_mode()
        if self.input.mode != "alchemy":
            if reference_phase is None:
                raise ValueError("provide a reference_phase")

    def run_static(self):
        self.status.running = True
        if self.input.reference_phase == "alchemy":
            job = Alchemy(calculation=self.calc, simfolder=self.working_directory)
        elif self.input.reference_phase == "solid":
            job = Solid(calculation=self.calc, simfolder=self.working_directory)
        elif self.input.reference_phase == "liquid":
            job = Liquid(calculation=self.calc, simfolder=self.working_directory)
        else:
            raise ValueError("Unknown reference state")

        if self.input.mode == "alchemy":
            routine_alchemy(job)
        elif self.input.mode == "fe":
            routine_fe(job)
        elif self.input.mode == "ts":
            routine_ts(job)
        elif self.input.mode == "pscale":
            routine_pscale(job)
        else:
            raise ValueError("Unknown mode")
        self._data = job.report
        # save conc for later use
        self.input.concentration = job.concentration
        del self._data["input"]
        self.status.collect = True
        self.run()

    def collect_general_output(self):
        """
        Collect the output from calphy

        Args:
            None

        Returns:
            None
        """
        if self._data is not None:
            if "spring_constant" in self._data["average"].keys():
                self.output.spring_constant = self._data["average"]["spring_constant"]
            self.output.energy_free = self._data["results"]["free_energy"]
            self.output.energy_free_error = self._data["results"]["error"]
            self.output.energy_free_reference = self._data["results"][
                "reference_system"
            ]
            self.output.energy_work = self._data["results"]["work"]
            self.output.temperature = self.input.temperature
            f_ediff, b_ediff, flambda, blambda = self.collect_ediff()
            self.output.forward_energy_diff = list(f_ediff)
            self.output.backward_energy_diff = list(b_ediff)
            self.output.forward_lambda = list(flambda)
            self.output.backward_lambda = list(blambda)
            if self.input.mode == "ts":
                datfile = os.path.join(self.working_directory, "temperature_sweep.dat")
                t, fe, ferr = np.loadtxt(datfile, unpack=True, usecols=(0, 1, 2))
                self.output.energy_free = np.array(fe)
                self.output.energy_free_error = np.array(ferr)
                self.output.temperature = np.array(t)

            elif self.input.mode == "pscale":
                datfile = os.path.join(self.working_directory, "pressure_sweep.dat")
                p, fe, ferr = np.loadtxt(datfile, unpack=True, usecols=(0, 1, 2))
                self.output.energy_free = np.array(fe)
                self.output.energy_free_error = np.array(ferr)
                self.output.pressure = np.array(p)

    def collect_ediff(self):
        """
        Calculate the energy difference between reference system and system of interest

        Args:
            None

        Returns:
            None
        """
        f_ediff = []
        b_ediff = []

        for i in range(1, self.input.n_iterations + 1):
            fwdfilename = os.path.join(self.working_directory, "forward_%d.dat" % i)
            bkdfilename = os.path.join(self.working_directory, "backward_%d.dat" % i)
            nelements = self.calc.n_elements

            if self.input.reference_phase == "solid":
                fdui = np.loadtxt(fwdfilename, unpack=True, comments="#", usecols=(0,))
                bdui = np.loadtxt(bkdfilename, unpack=True, comments="#", usecols=(0,))

                fdur = np.zeros(len(fdui))
                bdur = np.zeros(len(bdui))

                for i in range(nelements):
                    fdur += self.input.concentration[i] * np.loadtxt(
                        fwdfilename, unpack=True, comments="#", usecols=(i + 1,)
                    )
                    bdur += self.input.concentration[i] * np.loadtxt(
                        bkdfilename, unpack=True, comments="#", usecols=(i + 1,)
                    )

                flambda = np.loadtxt(
                    fwdfilename, unpack=True, comments="#", usecols=(nelements + 1,)
                )
                blambda = np.loadtxt(
                    bkdfilename, unpack=True, comments="#", usecols=(nelements + 1,)
                )
            else:
                fdui, fdur, flambda = np.loadtxt(
                    fwdfilename, unpack=True, comments="#", usecols=(0, 1, 2)
                )
                bdui, bdur, blambda = np.loadtxt(
                    bkdfilename, unpack=True, comments="#", usecols=(0, 1, 2)
                )

            f_ediff.append(fdui - fdur)
            b_ediff.append(bdui - bdur)

        return f_ediff, b_ediff, flambda, blambda

    def collect_output(self):
        self.collect_general_output()
        self.to_hdf()

    def to_hdf(self, hdf=None, group_name=None):
        super().to_hdf(hdf=hdf, group_name=group_name)
        self.input.to_hdf(self.project_hdf5)
        self.output.to_hdf(self.project_hdf5)

    def from_hdf(self, hdf=None, group_name=None):
        super().from_hdf(hdf=hdf, group_name=group_name)
        self.input.from_hdf(self.project_hdf5)
        self.output.from_hdf(self.project_hdf5)
        self._potential_from_hdf()

    @property
    def publication(self):
        return {
            "calphy": {
                "calphy": {
                    "title": "Automated free-energy calculation from atomistic simulations",
                    "journal": "Physical Review Materials",
                    "volume": "5",
                    "number": "10",
                    "pages": "103801",
                    "year": "2021",
                    "doi": "10.1103/PhysRevMaterials.5.103801",
                    "url": "https://journals.aps.org/prmaterials/abstract/10.1103/PhysRevMaterials.5.103801",
                    "author": [
                        "Menon, Sarath and Lysogorskiy, Yury and Rogal, Jutta and Drautz, Ralf"
                    ],
                }
            }
        }
