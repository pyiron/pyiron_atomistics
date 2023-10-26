import os
import numpy as np
from typing import Union, List, Tuple
import pandas as pd

from pyiron_base import DataContainer
from pyiron_atomistics.lammps.potential import LammpsPotential, LammpsPotentialFile
from pyiron_base import GenericJob, ImportAlarm
from pyiron_atomistics.lammps.structure import (
    LammpsStructure,
    UnfoldingPrism,
    structure_to_lammps,
)

from pyiron_atomistics.atomistics.structure.atoms import Atoms
from pyiron_atomistics.atomistics.structure.atoms import ase_to_pyiron
from pyiron_atomistics.atomistics.structure.has_structure import HasStructure

calphy_version = "1.0.0"

with ImportAlarm(
    "Calphy functionality requires the `calphy` module (and its dependencies) specified as extra"
    "requirements. Please install it and try again."
) as calphy_alarm:
    from calphy import Calculation, Solid, Liquid, Alchemy
    from calphy.routines import routine_fe, routine_ts, routine_alchemy, routine_pscale
    from calphy import __version__ as calphy_version
    from pyscal.trajectory import Trajectory as PyscalTrajectory

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


class Calphy(GenericJob, HasStructure):
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

    Specialised output depending on the selected mode is also available. For example the energy difference between the system of
    interest and the reference system for mode `fe` (free energy calculation), is available under `job.output.fe`. Similarly other output
    corresponding to modes such as temperature scaling and pressure scaling can be found under `job.output.ts` and `job.output.ps`.
    """

    def __init__(self, project, job_name):
        super().__init__(project, job_name)
        self.input = DataContainer(self._default_input, table_name="inputdata")
        self._potential_initial = None
        self._potential_final = None
        self.input.potential_initial_name = None
        self.input.potential_final_name = None
        self.input.structure = None
        self.output = DataContainer(table_name="output")
        self._data = None
        self.input._pot_dict_initial = None
        self.input._pot_dict_final = None
        self.__version__ = calphy_version

    @property
    def _default_input(self):
        return {
            "mode": None,
            "pressure": None,
            "temperature": None,
            "reference_phase": None,
            "npt": None,
            "n_equilibration_steps": 15000,
            "n_switching_steps": 25000,
            "n_print_steps": 1000,
            "n_iterations": 1,
            "spring_constants": None,
            "equilibration_control": None,
            "melting_cycle": True,
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
            "nose_hoover": {
                "thermostat_damping": 0.1,
                "barostat_damping": 0.1,
            },
            "berendsen": {
                "thermostat_damping": 100.0,
                "barostat_damping": 100.0,
            },
        }

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

    def _copy_pot_files(self):
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
                " ".join(
                    self._potential_initial.df["Config"]
                    .to_list()[0][0]
                    .strip()
                    .split()[1:]
                )
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
                " ".join(
                    self._potential_final.df["Config"]
                    .to_list()[0][0]
                    .strip()
                    .split()[1:]
                )
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

    def _get_element_list(self) -> List[str]:
        """
        Get elements as defined in pair style

        Args:
            None

        Returns:
            list: symbols of the elements
        """
        elements_from_pot = self._potential_initial.get_element_lst()
        elements_struct_lst = self.structure.get_species_symbols()

        elements = []
        for element_name in elements_from_pot:
            if element_name in elements_struct_lst:
                elements.append(element_name)

        return elements

    def _get_masses(self) -> List[float]:
        """
        Get masses as defined in pair style

        Args:
            None

        Returns:
            list: masses of the elements
        """
        elements_from_pot = self._potential_initial.get_element_lst()
        elements_object_lst = self.structure.get_species_objects()
        elements_struct_lst = self.structure.get_species_symbols()

        masses = []
        for element_name in elements_from_pot:
            if element_name in elements_struct_lst:
                index = list(elements_struct_lst).index(element_name)
                masses.append(elements_object_lst[index].AtomicMass)

        # this picks the actual masses, now we should pad with 1s to match length
        length_diff = len(elements_from_pot) - len(masses)
        return masses, length_diff

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
        lmp_structure.atom_type = "atomic"
        lmp_structure.el_eam_lst = self._potential_initial.get_element_lst()
        lmp_structure.structure = structure_to_lammps(structure)

        if not set(lmp_structure.structure.get_species_symbols()).issubset(
            set(lmp_structure.el_eam_lst)
        ):
            raise ValueError(
                "The selected potentials do not support the given combination of elements."
            )
        lmp_structure.write_file(file_name=file_name, cwd=working_directory)

    def _determine_mode(self):
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

    def _create_calc(self):
        """
        Create a calc object
        """
        calc = Calculation()
        for key in self._default_input.keys():
            if key not in ["md", "tolerance", "nose_hoover", "berendsen"]:
                setattr(calc, key, self.input[key])
        for key in self._default_input["md"].keys():
            setattr(calc.md, key, self.input["md"][key])
        for key in self._default_input["tolerance"].keys():
            setattr(calc.tolerance, key, self.input["tolerance"][key])
        for key in self._default_input["nose_hoover"].keys():
            setattr(calc.nose_hoover, key, self.input["nose_hoover"][key])
        for key in self._default_input["berendsen"].keys():
            setattr(calc.berendsen, key, self.input["berendsen"][key])

        calc.lattice = os.path.join(self.working_directory, "conf.data")

        pair_style, pair_coeff = self._prepare_pair_styles()
        calc._fix_potential_path = False
        calc.pair_style = pair_style
        calc.pair_coeff = pair_coeff

        calc.element = self._get_element_list()
        calc.mass, ghost_elements = self._get_masses()
        calc._ghost_element_count = ghost_elements

        calc.queue.cores = self.server.cores
        return calc

    def write_input(self):
        """
        Write input for calphy calculation

        Args:
            None

        Returns:
            None
        """
        file_name = "conf.data"
        self.write_structure(self.structure, file_name, self.working_directory)
        self._copy_pot_files()

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
        self.input.npt = pressure is not None
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
        self.input.npt = pressure is not None
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
        self.input.npt = pressure is not None
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
        self.input.npt = True
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
        self.input.npt = pressure is not None
        self.input.reference_phase = reference_phase
        self.input.n_equilibration_steps = n_equilibration_steps
        self.input.n_switching_steps = n_switching_steps
        self.input.n_print_steps = n_print_steps
        self.input.n_iterations = n_iterations
        self._determine_mode()
        if self.input.mode != "alchemy":
            if reference_phase is None:
                raise ValueError("provide a reference_phase")

    def run_static(self):
        calc = self._create_calc()
        self.status.running = True
        if self.input.reference_phase == "alchemy":
            job = Alchemy(calculation=calc, simfolder=self.working_directory)
        elif self.input.reference_phase == "solid":
            job = Solid(calculation=calc, simfolder=self.working_directory)
        elif self.input.reference_phase == "liquid":
            job = Liquid(calculation=calc, simfolder=self.working_directory)
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

    def _get_structure(self, frame=-1, wrap_atoms=False):
        """ """
        return [self.structure, self.output.structure_final][frame]

    def _number_of_structures(self):
        return 2

    def collect_general_output(self):
        """
        Collect the output from calphy

        Args:
            None

        Returns:
            None
        """
        if self._data is not None:
            # solid liquid specific outputs
            if "spring_constant" in self._data["average"].keys():
                self.output["spring_constant"] = self._data["average"][
                    "spring_constant"
                ]
            if "density" in self._data["average"].keys():
                self.output["atomic_density"] = self._data["average"]["density"]
            self.output["atomic_volume"] = self._data["average"]["vol_atom"]

            # main results from mode fe
            self.output["temperature"] = self.input.temperature
            self.output["pressure"] = self.input.pressure
            self.output["energy_free"] = self._data["results"]["free_energy"]
            self.output["energy_free_error"] = self._data["results"]["error"]
            self.output["energy_free_harmonic_reference"] = self._data["results"][
                "reference_system"
            ]
            self.output["energy_work"] = self._data["results"]["work"]
            self.output["energy_pressure"] = self._data["results"]["pv"]

            # collect ediffs and so on
            f_ediff, b_ediff, flambda, blambda = self._collect_ediff()
            self.output["fe/forward/energy_diff"] = list(f_ediff)
            self.output["fe/backward/energy_diff"] = list(b_ediff)
            self.output["fe/forward/lambda"] = list(flambda)
            self.output["fe/backward/lambda"] = list(blambda)

            # get final structure
            traj = PyscalTrajectory(
                os.path.join(self.working_directory, "conf.equilibration.dump")
            )
            aseobj = traj[0].to_ase(species=self._get_element_list())[0]
            pyiron_atoms = ase_to_pyiron(aseobj)
            self.output["structure_final"] = pyiron_atoms

            if self.input.mode == "ts":
                datfile = os.path.join(self.working_directory, "temperature_sweep.dat")
                t, fe, ferr = np.loadtxt(datfile, unpack=True, usecols=(0, 1, 2))

                # replace the quantities with updates ones
                self.output["energy_free"] = np.array(fe)
                self.output["energy_free_error"] = np.array(ferr)
                self.output["temperature"] = np.array(t)

                # collect diffs
                (
                    f_ediff,
                    b_ediff,
                    f_vol,
                    b_vol,
                    f_press,
                    b_press,
                    flambda,
                    blambda,
                ) = self._collect_thermo(mode="ts")
                self.output["ts/forward/energy_diff"] = list(f_ediff)
                self.output["ts/backward/energy_diff"] = list(b_ediff)
                self.output["ts/forward/lambda"] = list(flambda)
                self.output["ts/backward/lambda"] = list(blambda)
                self.output["ts/forward/volume"] = list(f_vol)
                self.output["ts/backward/volume"] = list(b_vol)
                self.output["ts/forward/pressure"] = list(f_press)
                self.output["ts/backward/pressure"] = list(b_press)

                # populate structures
                (
                    fwd_positions,
                    bkd_positions,
                    fwd_cells,
                    bkd_cells,
                ) = self._get_positions()
                self.output["ts/forward/positions"] = fwd_positions
                self.output["ts/backward/positions"] = bkd_positions
                self.output["ts/forward/cells"] = fwd_cells
                self.output["ts/backward/cells"] = bkd_cells

            elif self.input.mode == "pscale":
                datfile = os.path.join(self.working_directory, "pressure_sweep.dat")
                p, fe, ferr = np.loadtxt(datfile, unpack=True, usecols=(0, 1, 2))
                self.output["energy_free"] = np.array(fe)
                self.output["energy_free_error"] = np.array(ferr)
                self.output["pressure"] = np.array(p)

                (
                    f_ediff,
                    b_ediff,
                    f_vol,
                    b_vol,
                    f_press,
                    b_press,
                    flambda,
                    blambda,
                ) = self._collect_thermo(mode="pscale")
                self.output["ps/forward/energy_diff"] = list(f_ediff)
                self.output["ps/backward/energy_diff"] = list(b_ediff)
                self.output["ps/forward/lambda"] = list(flambda)
                self.output["ps/backward/lambda"] = list(blambda)
                self.output["ps/forward/volume"] = list(f_vol)
                self.output["ps/backward/volume"] = list(b_vol)
                self.output["ps/forward/pressure"] = list(f_press)
                self.output["ps/backward/pressure"] = list(b_press)

    def _collect_ediff(self):
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
            nelements = len(self._get_element_list())

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

    def _collect_thermo(self, mode="ts"):
        """
        Collect thermo quantities after ts run
        """
        f_ediff = []
        b_ediff = []
        f_vol = []
        b_vol = []
        f_press = []
        b_press = []

        for i in range(1, self.input.n_iterations + 1):
            fwdfilename = os.path.join(
                self.working_directory, f"{mode}.forward_{i}.dat"
            )
            bkdfilename = os.path.join(
                self.working_directory, f"{mode}.backward_{i}.dat"
            )

            fdx, fp, fvol, flambda = np.loadtxt(fwdfilename, unpack=True, comments="#")
            bdx, bp, bvol, blambda = np.loadtxt(bkdfilename, unpack=True, comments="#")

            fdx /= flambda
            bdx /= blambda

            f_ediff.append(fdx)
            b_ediff.append(bdx)

            f_vol.append(fvol)
            b_vol.append(bvol)

            f_press.append(fp)
            b_press.append(bp)

        return f_ediff, b_ediff, f_vol, b_vol, f_press, b_press, flambda, blambda

    def _get_positions(self):
        """
        Collect positions and cells
        """
        fwd_positions = []
        bkd_positions = []
        fwd_cells = []
        bkd_cells = []

        for i in range(1, self.input.n_iterations + 1):
            fwdfilename = os.path.join(
                self.working_directory, f"traj.ts.forward_{i}.dat"
            )
            bkdfilename = os.path.join(
                self.working_directory, f"traj.ts.backward_{i}.dat"
            )

            fp = []
            fc = []
            bp = []
            bc = []

            if os.path.exists(fwdfilename):
                traj = PyscalTrajectory(fwdfilename)
                for x in traj.nblocks:
                    aseobj = traj[x].to_ase(species=self._get_element_list())
                    fp.append(aseobj.positions)
                    fc.append(list(aseobj.cell))
            if os.path.exists(bkdfilename):
                traj = PyscalTrajectory(bkdfilename)
                for x in traj.nblocks:
                    aseobj = traj[x].to_ase(species=self._get_element_list())
                    bp.append(aseobj.positions)
                    bc.append(list(aseobj.cell))

            fwd_positions.append(fp)
            bkd_positions.append(bp)
            fwd_cells.append(fc)
            bkd_positions.append(bc)

        return fwd_positions, bkd_positions, fwd_cells, bkd_cells

    def collect_output(self):
        self.collect_general_output()
        self.to_hdf()

    def db_entry(self):
        """
        Generate the initial database entry
        Returns:
            (dict): db_dict
        """
        db_dict = super(Calphy, self).db_entry()
        if self.structure:
            if isinstance(self.structure, Atoms):
                parent_structure = self.structure.get_parent_basis()
            else:
                parent_structure = self.structure.copy()
            db_dict["ChemicalFormula"] = parent_structure.get_chemical_formula()

        return db_dict

    def to_hdf(self, hdf=None, group_name=None):
        super().to_hdf(hdf=hdf, group_name=group_name)
        self.input.to_hdf(self.project_hdf5)
        self.output.to_hdf(self.project_hdf5)

    def from_hdf(self, hdf=None, group_name=None):
        super().from_hdf(hdf=hdf, group_name=group_name)
        self.input.from_hdf(self.project_hdf5)
        self.output.from_hdf(self.project_hdf5)
        self._potential_from_hdf()
        # self.create_calc()

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
