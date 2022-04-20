from pyiron_base import DataContainer
from pyiron_base.interfaces.has_hdf import HasHDF
from pyiron_atomistics.lammps.potential import LammpsPotential
from pyiron_atomistics.lammps.potential import LammpsPotentialFile, PotentialAvailable
from pyiron_base import GenericJob
from pyiron_atomistics.lammps.structure import LammpsStructure, UnfoldingPrism

from calphy import Calculation, Solid, Liquid, Alchemy
from calphy.routines import routine_fe, routine_ts, routine_alchemy, routine_pscale

import copy
import os
import numpy as np

inputdict = {
    "mode": None,
    "pressure": None,
    "temperature": None,
    "temperature_high": None,
    "reference_phase": None,
    "npt": None,
    "n_equilibration_steps": 15000,
    "n_switching_steps": 25000,
    "n_print_steps": 0,
    "n_iterations": 1,
    "md": {'timestep': 0.001,
         'n_small_steps': 10000,
         'n_every_steps': 10,
         'n_repeat_steps': 10,
         'n_cycles': 100,
         'thermostat_damping': 0.5,
         'barostat_damping': 0.1},
    "tolerance": {'lattice_constant': 0.0002,
         'spring_constant': 0.01,
         'solid_fraction': 0.7,
         'liquid_fraction': 0.05,
         'pressure': 0.5}
}


class CalphyBase(GenericJob):
    def __init__(self, project, job_name):
        super(CalphyBase, self).__init__(project, job_name)
        self.__name__ = "CalphyJob"
        self.input = DataContainer(inputdict, table_name="inputdata")
        self._potential_initial = None
        self._potential_final = None
        self.input.potential_initial_name = None
        self.input.potential_final_name = None
        self.input.structure = None
        self.output = DataContainer(table_name="output")        
        self._data = None

    def set_potentials(self, potential_filenames):
        if not isinstance(potential_filenames, list):
            potential_filenames = [potential_filenames]            
        if len(potential_filenames) == 1:
            potential = LammpsPotentialFile().find_by_name(potential_filenames[0])
            self._potential_initial = LammpsPotential()
            self._potential_initial.df = potential
            self.input.potential_initial_name = potential_filenames[0]
        if len(potential_filenames) == 2:
            potential = LammpsPotentialFile().find_by_name(potential_filenames[1])
            self._potential_final = LammpsPotential()
            self._potential_final.df = potential
            self.input.potential_final_name = potential_filenames[0]
        if len(potential_filenames) > 2:
            raise ValueError("Maximum two potentials can be provided")
            
    def get_potentials(self):
        if self._potential_final is None:
            return [self._potential_initial.df]
        else:
            return [self._potential_initial.df, self._potential_final.df]
        
    def copy_pot_files(self):
        if self._potential_initial is not None:
            self._potential_initial.copy_pot_files(self.working_directory)
        if self._potential_final is not None:
            self._potential_final.copy_pot_files(self.working_directory)

    def prepare_pair_styles(self):
        pair_style = []
        pair_coeff = []
        if self._potential_initial is not None:
            pair_style.append(self._potential_initial.df['Config'].to_list()[0][0].strip().split()[1:][0])
            pair_coeff.append(" ".join(self._potential_initial.df['Config'].to_list()[0][1].strip().split()[1:]))
        if self._potential_final is not None:
            pair_style.append(self._potential_final.df['Config'].to_list()[0][0].strip().split()[1:][0])
            pair_coeff.append(" ".join(self._potential_final.df['Config'].to_list()[0][1].strip().split()[1:]))
        return pair_style, pair_coeff

    #we wrap some properties for easy access
    @property
    def potential(self):
        return self.get_potentials()
    
    @potential.setter
    def potential(self, potential_filenames):
        self.set_potentials(potential_filenames)
    
    @property
    def structure(self):
        return self.input.structure

    @structure.setter
    def structure(self, val):
        self.input.structure = val

    def view_potentials(self):
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
                str(list_of_elements),)

    def structure_to_lammps(self):
        prism = UnfoldingPrism(self.input.structure.cell)
        lammps_structure = self.input.structure.copy()
        lammps_structure.set_cell(prism.A)
        lammps_structure.positions = np.matmul(self.input.structure.positions, prism.R)
        return lammps_structure
    
    def write_structure(self, structure, file_name, working_directory):
        lmp_structure = LammpsStructure()
        lmp_structure.potential = self._potential_initial
        lmp_structure.el_eam_lst = set(structure.get_chemical_symbols())
        lmp_structure.structure = self.structure_to_lammps()
        lmp_structure.write_file(file_name=file_name, cwd=working_directory)

    def determine_mode(self):
        if len(self.potential) == 2:
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
        #if mode was not set, raise Error
        if self.input.mode is None:
            raise RuntimeError("Could not determine the mode")
    
    def write_input(self):
        #now prepare the calculation
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
        pair_style, pair_coeff = self.prepare_pair_styles()
        calc._fix_potential_path = False
        calc.pair_style = pair_style
        calc.pair_coeff = pair_coeff

        calc.element = list(np.unique(self.structure.get_chemical_symbols()))
        calc.mass = list(np.unique(self.structure.get_masses()))

        calc.queue.cores = self.server.cores
        self.calc = calc

    
    def calc_mode_fe(self, temperature=None, pressure=None,
        reference_phase=None, n_equilibration_steps=15000,
        n_switching_steps=25000, n_print_steps=0, 
        n_iterations=1):
        if temperature is None:
            raise ValueError("provide a temperature")
        self.input.temperature = temperature
        self.input.pressure = pressure
        self.input.reference_phase = reference_phase
        self.input.n_equilibration_steps = n_equilibration_steps
        self.input.n_switching_steps = n_switching_steps
        self.input.n_print_steps = n_print_steps
        self.input.n_iterations = n_iterations
        self.input.mode = "fe"

    def calc_mode_ts(self, temperature=None, pressure=None,
        reference_phase=None, n_equilibration_steps=15000,
        n_switching_steps=25000, n_print_steps=0, 
        n_iterations=1):
        if temperature is None:
            raise ValueError("provide a temperature")
        self.input.temperature = temperature
        self.input.pressure = pressure
        self.input.reference_phase = reference_phase
        self.input.n_equilibration_steps = n_equilibration_steps
        self.input.n_switching_steps = n_switching_steps
        self.input.n_print_steps = n_print_steps
        self.input.n_iterations = n_iterations
        self.input.mode = "ts"

    def calc_mode_alchemy(self, temperature=None, pressure=None,
        reference_phase=None, n_equilibration_steps=15000,
        n_switching_steps=25000, n_print_steps=0, 
        n_iterations=1):
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

    def calc_mode_pscale(self, temperature=None, pressure=None,
        reference_phase=None, n_equilibration_steps=15000,
        n_switching_steps=25000, n_print_steps=0, 
        n_iterations=1):
        if temperature is None:
            raise ValueError("provide a temperature")
        self.input.temperature = temperature
        self.input.pressure = pressure
        self.input.reference_phase = reference_phase
        self.input.n_equilibration_steps = n_equilibration_steps
        self.input.n_switching_steps = n_switching_steps
        self.input.n_print_steps = n_print_steps
        self.input.n_iterations = n_iterations
        self.input.mode = "pscale"

    def calc_free_energy(self, temperature=None, pressure=None,
        reference_phase=None, n_equilibration_steps=15000,
        n_switching_steps=25000, n_print_steps=0, 
        n_iterations=1):
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
    
    def run_static(self):
        self.status.running = True
        if self.input.reference_phase == "alchemy":
            job = Alchemy(calculation=self.calc,
                          simfolder=self.working_directory)
        elif self.input.reference_phase == "solid":
            job = Solid(calculation=self.calc,
                          simfolder=self.working_directory)
        elif self.input.reference_phase == "liquid":
            job = Liquid(calculation=self.calc,
                          simfolder=self.working_directory)
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
        #save conc for later use
        self.input.concentration = job.concentration
        del self._data['input']
        self.status.collect = True
        self.run()
    

    def collect_general_output(self):        
        if self._data is not None:
            if "spring_constant" in self._data["average"].keys():
                self.output.spring_constant = self._data["average"]["spring_constant"]
            self.output.energy_free = self._data['results']['free_energy']
            self.output.energy_free_error = self._data['results']['error']
            self.output.energy_free_reference = self._data['results']['reference_system']
            self.output.energy_work = self._data['results']['work']
            self.output.temperature = self.input.temperature
            f_ediff, b_ediff, flambda, blambda = self.collect_ediff()
            self.output.forward_energy_diff = list(f_ediff)
            self.output.backward_energy_diff = list(b_ediff)
            self.output.forward_lambda = list(flambda)
            self.output.backward_lambda = list(blambda)
            if self.input.mode == "ts":
                datfile = os.path.join(self.working_directory, "temperature_sweep.dat")
                t, fe, ferr = np.loadtxt(datfile, unpack=True, usecols=(0,1,2))                
                self.output.energy_free = np.array(fe)
                self.output.energy_free_error = np.array(ferr)
                self.output.temperature = np.array(t)

            elif self.input.mode == "pscale":
                datfile = os.path.join(self.working_directory, "pressure_sweep.dat")
                p, fe, ferr = np.loadtxt(datfile, unpack=True, usecols=(0,1,2))                
                self.output.energy_free = np.array(fe)
                self.output.energy_free_error = np.array(ferr)
                self.output.pressure = np.array(p)

    def collect_ediff(self):

        f_ediff = []
        b_ediff = []

        for i in range(1, self.input.n_iterations+1):            
            fwdfilename = os.path.join(self.working_directory, "forward_%d.dat"%i)
            bkdfilename = os.path.join(self.working_directory, "backward_%d.dat"%i)
            nelements = self.calc.n_elements

            if self.input.reference_phase == "solid":
                fdui = np.loadtxt(fwdfilename, unpack=True, comments="#", usecols=(0,))
                bdui = np.loadtxt(bkdfilename, unpack=True, comments="#", usecols=(0,))

                fdur = np.zeros(len(fdui))
                bdur = np.zeros(len(bdui))

                for i in range(nelements):
                    fdur += self.input.concentration[i]*np.loadtxt(fwdfilename, unpack=True, comments="#", usecols=(i+1,))
                    bdur += self.input.concentration[i]*np.loadtxt(bkdfilename, unpack=True, comments="#", usecols=(i+1,))

                flambda = np.loadtxt(fwdfilename, unpack=True, comments="#", usecols=(nelements+1,))
                blambda = np.loadtxt(bkdfilename, unpack=True, comments="#", usecols=(nelements+1,))
            else:
                fdui, fdur, flambda = np.loadtxt(fwdfilename, unpack=True, comments="#", usecols=(0,1,2))
                bdui, bdur, blambda = np.loadtxt(bkdfilename, unpack=True, comments="#", usecols=(0,1,2))

            f_ediff.append(fdui-fdur)
            b_ediff.append(bdui-bdur)

        return f_ediff, b_ediff, flambda, blambda



    def collect_output(self):
        self.collect_general_output()
        self.to_hdf()
    
    def to_hdf(self, hdf=None, group_name=None):
        super().to_hdf(hdf=hdf, group_name=group_name)
        self.input.to_hdf(self.project_hdf5)
        self.output.to_hdf(self.project_hdf5)

        #self.structure.to_hdf(hdf)
        #with self.project_hdf5.open("input") as hdf5_in:
        #with self.project_hdf5.open("output") as hdf5_out:

    def from_hdf(self, hdf=None, group_name=None):
        super().from_hdf(hdf=hdf, group_name=group_name)
        self.input.from_hdf(self.project_hdf5)
        self.output.from_hdf(self.project_hdf5)

        #self.structure.to_hdf(hdf)
        #with self.project_hdf5.open("input") as hdf5_in:
        #with self.project_hdf5.open("output") as hdf5_out: