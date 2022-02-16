import copy
import os
import numpy as np

from pyiron_atomistics.lammps.potential import LammpsPotential
from pyiron_atomistics.lammps.structure import LammpsStructure, UnfoldingPrism
from pyiron_atomistics.lammps.potential import LammpsPotentialFile, PotentialAvailable
from pyiron_base import GenericJob

from calphy.input import check_and_convert_to_list
from calphy.queuekernel import Solid, Liquid, routine_fe, routine_ts

class CalphyBase(GenericJob):
    def __init__(self, project, job_name):
        super(CalphyBase, self).__init__(project, job_name)
        #now calphy input parameters need to be handled; for the moment minimal is better
        self.__name__ = "CalphyJob"
        #things that will be taken over by pyiron
        #element, mass, lattice, repeat, lattice-constant
        self.structure = False
        self.input = Input()
        self._data = None
        
    @property
    def potential(self):
        return self.input.potential.df
    
    @potential.setter
    def potential(self, potential_filename):
        potential_db = LammpsPotentialFile()
        potential = potential_db.find_by_name(potential_filename)
        self.input.potential.df = potential
        
    def view_potentials(self):
        if not self.structure:
            raise ValueError("No structure set.")
        list_of_elements = set(self.structure.get_chemical_symbols())
        list_of_potentials = LammpsPotentialFile().find(list_of_elements)
        if list_of_potentials is not None:
            return list_of_potentials
        else:
            raise TypeError(
                "No potentials found for this kind of structure: ",
                str(list_of_elements),
            )

    def write_input(self):
        self.input.prepare_input(self.structure, 
                                cores = self.server.cores)
        #write structure
        file_name = "conf.data"
        self.input.write_structure(file_name, self.working_directory)
        #write potentials
        self.input.potential.copy_pot_files(self.working_directory)
        #now reset the lattice in options
        self.input.options["calculations"][0]["lattice"] = os.path.join(self.working_directory,
                                                                       "conf.data")
    def run_static(self):
        #now we have to actually run the job;
        if self.input.mode == "fe":
            if self.input.options["calculations"][0]["state"] == "liquid":
                job = Liquid(options=self.input.options, 
                             kernel=0, 
                             simfolder=self.working_directory)
            else:
                job = Solid(options=self.input.options, 
                             kernel=0, 
                             simfolder=self.working_directory)
            routine_fe(job)
            #this could be changed to an ordered dict;
            #needs to be done on the pyscal side
            self._data = job.report
            del self._data['input']

        elif self.input.mode == "ts":
            if self.input.options["calculations"][0]["state"] == "liquid":
                job = Liquid(options=self.input.options, 
                             kernel=0, 
                             simfolder=self.working_directory)
            else:
                job = Solid(options=self.input.options, 
                             kernel=0, 
                             simfolder=self.working_directory)
            routine_ts(job)
            self._data = job.report
            del self._data['input']

        self.collect_output()

    def collect_output(self):
        self.to_hdf()
        self.collect_logs()
        
    def to_hdf(self, hdf=None, group_name=None):
        super().to_hdf(hdf=hdf, group_name=group_name)
        with self.project_hdf5.open("input") as h5in:
            self.input.to_hdf(h5in)
            self.structure.to_hdf(h5in)
        if self._data is not None:
            with self.project_hdf5.open("output") as hdf5_out:
                hdf5_out["average"] = self._data["average"]
                hdf5_out["fe_total"] = self._data['results']['free_energy']
                hdf5_out["fe_error"] = self._data['results']['error']
                hdf5_out["fe_reference"] = self._data['results']['reference_system']
                hdf5_out["fe_work"] = self._data['results']['work']
                hdf5_out["fe_pv"] = self._data['results']['pv']
                hdf5_out["temperature"] = self.input.temperature

                if self.input.mode == "ts":
                    datfile = os.path.join(self.working_directory, "temperature_sweep.dat")
                    t, fe, ferr = np.loadtxt(datfile, unpack=True, usecols=(0,1,2))
                    hdf5_out["fe_total"] = fe
                    hdf5_out["temperature"] = t
                    hdf5_out["fe_error"] = ferr

    def from_hdf(self, hdf=None, group_name=None):
        super().from_hdf(hdf=hdf, group_name=group_name)
        with self.project_hdf5.open("input") as h5in:
            self.input.from_hdf(h5in)
            self.structure.from_hdf(h5in)

    def collect_logs(self):
        logile = os.path.join(self.working_directory, "calphy.log")
        with open(logile, 'r') as fin:
            content = fin.read()
        with self.project_hdf5.open("output") as hdf5_out:
            hdf5_out["calphy_log"] = content
        logile = os.path.join(self.working_directory, "log.lammps")
        with open(logile, 'r') as fin:
            content = fin.read()
        with self.project_hdf5.open("output") as hdf5_out:
            hdf5_out["lammps_log"] = content

    def collect_files(self):
        datfile = os.path.join(self.working_directory, "temperature_sweep.dat")
        t, fe, ferr = np.loadtxt(datfile, unpack=True, usecols=(0,1,2))


class Input:
    def __init__(self):
        self.element = None
        self.mass = 1.00
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
        self.temperature = None
        self.pressure = None
        self.potential = LammpsPotential()
        self.state = None
        self.nsims = 1
        self._mode = "fe"
        self.iso = False
        self.fix_lattice = False
        self.npt = True
        self.tguess = None
        self.options = {}

    @property
    def mode(self):
        return self._mode
    
    @mode.setter
    def mode(self, value):
        if value in ["fe", "ts"]:
            self._mode = value
        else:
            raise ValueError("Currently only mode fe is supported")
    
    def structure_to_lammps(self):
        prism = UnfoldingPrism(self.structure.cell)
        lammps_structure = self.structure.copy()
        lammps_structure.set_cell(prism.A)
        lammps_structure.positions = np.matmul(self.structure.positions, prism.R)
        return lammps_structure
    
    def write_structure(self, file_name, working_directory):
        lmp_structure = LammpsStructure()
        lmp_structure.potential = self.potential
        lmp_structure.el_eam_lst = set(self.structure.get_chemical_symbols())
        lmp_structure.structure = self.structure_to_lammps()
        lmp_structure.write_file(file_name=file_name, cwd=working_directory)
        
    def prepare_input(self, structure, cores=1):
        self.structure = structure
        self.options["element"] = check_and_convert_to_list(self.element)
        self.options["nelements"] = len(self.options["element"])
        self.options["mass"] = check_and_convert_to_list(self.mass)
        self.options["md"] = self.md
        self.options["conv"] = self.conv
        self.options["queue"] = self.queue
        #add parallel support if needed
        self.options["queue"]["cores"] = cores
        self.options["md"]["pair_style"] = self.potential.df['Config'].to_list()[0][0].strip().split()[1:]
        self.options["md"]["pair_coeff"] = [" ".join(self.potential.df['Config'].to_list()[0][1].strip().split()[1:])]
        self.options["calculations"] = []
        #this is also to satisfy calphys algorithm; it can take a number of calculations
        #at the same time. But this is likely what we do not want here.
        cdict = {}
        cdict["mode"] = self.mode
        if isinstance(self.temperature, list):
            cdict["temperature"] = self.temperature[0]
            cdict["temperature_stop"] = self.temperature[-1]
        else:
            cdict["temperature"] = self.temperature
            cdict["temperature_stop"] = self.temperature
        cdict["lattice"] = None
        cdict["pressure"] = self.pressure
        cdict["state"] = self.state
        cdict["nelements"] = len(self.options["element"])
        cdict["element"] = self.options["element"]
        cdict["lattice_constant"] = 0
        cdict["iso"] = self.iso
        cdict["fix_lattice"] = self.fix_lattice
        cdict["npt"] = self.npt
        cdict["tguess"] = self.tguess
        cdict["repeat"] = [1, 1, 1]
        cdict["nsims"] = self.nsims
        cdict["thigh"] = 2.0*cdict["temperature_stop"]
        cdict["dtemp"] = 200
        cdict["maxattempts"] = 5
        self.options["calculations"].append(cdict)
    
    def to_hdf(self, hdf5):
        self.potential.to_hdf(hdf5)
        hdf5["input"] = self.options
    
    def from_hdf(self, hdf5):
        self.potential.from_hdf(hdf5)
        self.options = hdf5["input"]

