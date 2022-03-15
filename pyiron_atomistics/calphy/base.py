from pyiron_base import DataContainer
from pyiron_atomistics.lammps.potential import LammpsPotential
from pyiron_atomistics.lammps.potential import LammpsPotentialFile, PotentialAvailable
from pyiron_base import GenericJob
from pyiron_atomistics.lammps.structure import LammpsStructure, UnfoldingPrism

from calphy.input import check_and_convert_to_list
from calphy.queuekernel import Solid, Liquid, Alchemy, routine_fe, routine_ts, routine_alchemy

import copy
import os
import numpy as np

class Potential:
    def __init__(self):
        pass
    
    def set_potentials(self, potential_filenames):
        if not isinstance(potential_filenames, list):
            potential_filenames = [potential_filenames]
        potential_db = LammpsPotentialFile()
        potentials = [potential_db.find_by_name(potential_filename) for potential_filename in potential_filenames]
        self.potentials = [LammpsPotential() for x in range(len(potential_filenames))]
        for x in range(len(potential_filenames)):
            self.potentials[x].df = potentials[x]
    
    def get_potentials(self):
        return [potential.df for potential in self.potentials]
    
    def to_hdf(self, hdf=None, group_name=None):
        with hdf.open("potentials") as mlevel:
            mlevel["count"] = len(self.potentials)
            for count, potential in enumerate(self.potentials):
                with mlevel.open("p%d"%count) as hdf5_out:
                    potential.to_hdf(hdf5_out)

    def from_hdf(self, hdf=None, group_name=None):
        plist = []
        with hdf.open("potentials") as mlevel:
            count = mlevel["count"]
            for x in range(count):
                with mlevel.open("p%d"%count) as hdf5_out:
                    potential = LammpsPotential()
                    potential.from_hdf(hdf5_out)
                    plist.append(potential)
        self.potentials = plist
    
    def copy_pot_files(self, working_directory):
        for potential in self.potentials:
            potential.copy_pot_files(working_directory)

    def prepare_pair_styles(self):
        pair_style = []
        pair_coeff = []
        
        for potential in self.potentials:
            pair_style.append(potential.df['Config'].to_list()[0][0].strip().split()[1:][0])
            pair_coeff.append(" ".join(potential.df['Config'].to_list()[0][1].strip().split()[1:]))
        return pair_style, pair_coeff
                
class Input(DataContainer):
    def __init__(self):
        super(Input, self).__init__(table_name="input")
        self.md = {
            #pair elements
            "pair_style": None, "pair_coeff": None,
            #time related properties
            "timestep": 0.001, "nsmall": 10000, "nevery": 10, "nrepeat": 10, "ncycles": 100,
            #ensemble properties
            "tdamp": 0.1, "pdamp": 0.1,
            #eqbr and switching time
            "te": 25000, "ts": 50000, "tguess": None,
            "dtemp": 200, "maxattempts": 5, "traj_interval": 0,
        }
        self.conv = {
            "alat_tol": 0.0002, "k_tol": 0.01,
            "solid_frac": 0.7, "liquid_frac": 0.05, "p_tol": 0.5,
        }
        #This is to satisfy calphy; Should be removed from the calphy side
        self.queue = {
            "scheduler": "local", "cores": 1, "jobname": "ti",
            "walltime": "23:50:00", "queuename": None, "memory": "3GB",
            "commands": None, "modules": None, "options": None
        }
        self.mode = None
        self._potential = Potential()
        self.tguess = None
        self.structure = None
        self.options = {}
        
        #Temperature
        self._temperature = [0]

        #Pressure
        self._pressure = 0
        self.iso = True
        self.fix_lattice = False
        self.npt = True

        self.reference_state = None
        self.n_cycles = 1
        self.n_switching_steps = 25000
        self.n_equilibration_steps = 10000

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        """
        float/list of 2 entries
        """
        if isinstance(value, list):
            if not len(value) == 2:
                raise ValueError("Temperature should be float of list of 2 floats")
        else:
            value = [value]
        self._temperature = value


    @property
    def pressure(self):
        return self._pressure

    @pressure.setter
    def pressure(self, value):
        """
        None: fix_lattice True, else fix_lattice False
        scalar: iso True
        vector: iso False
        npt: for the moment, True
        """
        if value is None:
            self.fix_lattice = True
        elif isinstance(value, list):
            if len(value) == 3:
                if (value[0]==value[1]==value[2]):
                    self._pressure = value[0]
                    self.iso = False
                else:
                    raise NotImplementedError()
            else:
                raise NotImplementedError()
        else:
            self._pressure = value
            self.iso = True
            self.fix_lattice = False
                
    
    @property
    def potential(self):
        return self._potential.get_potentials()
    
    @potential.setter
    def potential(self, potential_filenames):
        self._potential.set_potentials(potential_filenames)        
        #potential_db = LammpsPotentialFile()
        #potential = potential_db.find_by_name(potential_filename)
        #self._potential = LammpsPotential()
        #self._potential.df = potential
    
    def structure_to_lammps(self, structure):
        prism = UnfoldingPrism(structure.cell)
        lammps_structure = structure.copy()
        lammps_structure.set_cell(prism.A)
        lammps_structure.positions = np.matmul(structure.positions, prism.R)
        return lammps_structure
    
    def write_structure(self, structure, file_name, working_directory):
        lmp_structure = LammpsStructure()
        lmp_structure.potential = self.potential
        lmp_structure.el_eam_lst = set(structure.get_chemical_symbols())
        lmp_structure.structure = self.structure_to_lammps(structure)
        lmp_structure.write_file(file_name=file_name, cwd=working_directory)
    
    def determine_mode(self):
        if len(self.input.potential) == 2:
            self.mode = "alchemy"
            self.reference_state = "alchemy"
        elif len(self.temperature) == 1:
            self.mode = "fe"
        elif len(self.temperature) == 2:
            self.mode = "ts"
        #if mode was not set, raise Error
        if self.mode is None:
            raise RuntimeError("Could not determine the mode")
    
    def prepare_input(self, structure, working_directory, cores=1):
                
        #self.structure = structure
        file_name = "conf.data"
        self.write_structure(structure, file_name, working_directory)      
        self._potential.copy_pot_files(working_directory)
        
        #this still needs to be fixed
        self.options["element"] = list(np.unique(structure.get_chemical_symbols()))
        self.options["nelements"] = len(self.options["element"])
        self.options["mass"] = list(np.unique(structure.get_masses()))
        
        self.options["md"] = self.md
        self.options["md"]["te"] = self.n_equilibration_steps
        self.options["md"]["ts"] = self.n_switching_steps
        
        self.options["conv"] = self.conv
        self.options["queue"] = self.queue
        
        #add parallel support if needed
        self.options["queue"]["cores"] = cores
        
        #prepare potentials
        pair_style, pair_coeff = self._potential.prepare_pair_styles()
        self.options["md"]["pair_style"] = pair_style
        self.options["md"]["pair_coeff"] = pair_coeff
        self.options["calculations"] = []
        
        #this is also to satisfy calphys algorithm; it can take a number of calculations
        #at the same time. But this is likely what we do not want here.
        cdict = {}
        cdict["mode"] = self.mode
        
        if isinstance(self._temperature, list):
            cdict["temperature"] = self._temperature[0]
            cdict["temperature_stop"] = self._temperature[-1]
        else:
            cdict["temperature"] = self._temperature
            cdict["temperature_stop"] = self._temperature
        
        cdict["lattice"] = os.path.join(working_directory, "conf.data")
        cdict["state"] = self.reference_state
        cdict["nelements"] = self.options["nelements"]
        cdict["element"] = self.options["element"]
        cdict["lattice_constant"] = 0

        cdict["pressure"] = self._pressure
        cdict["iso"] = self.iso
        cdict["fix_lattice"] = self.fix_lattice
        cdict["npt"] = self.npt
        
        cdict["tguess"] = self.tguess
        cdict["repeat"] = [1, 1, 1]
        cdict["nsims"] = self.n_cycles
        cdict["thigh"] = 2.0*cdict["temperature_stop"]
        cdict["dtemp"] = 200
        cdict["maxattempts"] = 5
        self.options["calculations"].append(cdict)
    
class CalphyBase(GenericJob):
    def __init__(self, project, job_name):
        super(CalphyBase, self).__init__(project, job_name)
        self.__name__ = "CalphyJob"
        self.structure = False
        self.input = Input()
        self.output = DataContainer(table_name="output")        
        self._data = None
    
    #we wrap some properties for easy access
    @property
    def potential(self):
        return self.input.potential
    
    @potential.setter
    def potential(self, potential_filename):
        self.input.potential = potential_filename
    
    def view_potentials(self):
        if not self.structure:
            list_of_elements = set(self.input.element)
        else:    
            list_of_elements = set(self.structure.get_chemical_symbols())
        list_of_potentials = LammpsPotentialFile().find(list_of_elements)
        if list_of_potentials is not None:
            return list_of_potentials
        else:
            raise TypeError(
                "No potentials found for this kind of structure: ",
                str(list_of_elements),)
    
    def write_input(self):
        self.input.prepare_input(self.structure, self.working_directory, 
                                 cores = self.server.cores)
    
    def calc_mode_fe(self):
        self.input.mode = "fe"

    def calc_mode_ts(self):
        self.input.mode = "ts"

    def calc_mode_alchemy(self):
        self.input.mode = "alchemy"

    def calc_free_energy(self):
        self.input.determine_mode()
    
    def run_static(self):
        self.status.running = True
        if self.input.reference_state == "alchemy":
            job = Alchemy(options=self.input.options, 
                          kernel=0, 
                          simfolder=self.working_directory)
        elif self.input.reference_state == "solid":
            job = Solid(options=self.input.options, 
                         kernel=0, 
                         simfolder=self.working_directory)
        elif self.input.reference_state == "liquid":
            job = Liquid(options=self.input.options, 
                         kernel=0, 
                         simfolder=self.working_directory)
        else:
            raise ValueError("Unknown reference state")
        
        if self.input.mode == "alchemy":
            routine_alchemy(job)
        elif self.input.mode == "fe":
            routine_fe(job)        
        elif self.input.mode == "ts":
            routine_ts(job)
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
            self.output.spring_constant = self._data["average"]["spring_constant"]
            with self.project_hdf5.open("output") as hdf5_out:
                #hdf5_out["spring_constant"] = self._data["average"]["spring_constant"]
                hdf5_out["energy_free"] = self._data['results']['free_energy']
                hdf5_out["energy_free_error"] = self._data['results']['error']
                hdf5_out["energy_free_reference"] = self._data['results']['reference_system']
                hdf5_out["energy_work"] = self._data['results']['work']
                hdf5_out["temperature"] = self.input.temperature

                f_ediff, b_ediff, flambda, blambda = self.collect_ediff()

                hdf5_out["forward/energy_diff"] = f_ediff
                hdf5_out["backward/energy_diff"] = b_ediff
                hdf5_out["forward/lambda"] = flambda
                hdf5_out["backward/lambda"] = blambda

                if self.input.mode == "ts":
                        datfile = os.path.join(self.working_directory, "temperature_sweep.dat")
                        t, fe, ferr = np.loadtxt(datfile, unpack=True, usecols=(0,1,2))                
                        hdf5_out["energy_free"] = fe
                        hdf5_out["energy_free_error"] = ferr
                        hdf5_out["temperature"] = t

    def collect_ediff(self):

        f_ediff = []
        b_ediff = []

        for i in range(1, self.input.n_cycles+1):            
            fwdfilename = os.path.join(self.working_directory, "forward_%d.dat"%i)
            bkdfilename = os.path.join(self.working_directory, "backward_%d.dat"%i)
            nelements = self.input.options["nelements"]

            if self.input.reference_state == "solid":
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
