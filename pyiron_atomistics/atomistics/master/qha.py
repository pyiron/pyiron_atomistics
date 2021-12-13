import numpy as np
import pint
from sklearn.linear_model import LinearRegression
from pyiron_atomistics.atomistics.master.parallel import AtomisticParallelMaster
from pyiron_base import JobGenerator


unit = pint.UnitRegistry()


def get_helmholtz_free_energy(frequencies, temperature, E_0=0):
    hn = (frequencies * 1e12 * unit.hertz * unit.planck_constant).to('eV').magnitude
    hn = hn[hn > 0]
    kBT = (temperature * unit.kelvin * unit.boltzmann_constant).to('eV').magnitude
    kBT = np.atleast_1d(kBT)
    values = np.zeros_like(kBT)
    values[kBT > 0] = kBT[kBT > 0] * np.log(
        1 - np.exp(-np.einsum('i,...->...i', hn, 1 / kBT[kBT > 0]))
    ).sum(axis=-1)
    return 0.5 * np.sum(hn) + values + E_0


def get_potential_energy(frequencies, temperature, E_0=0):
    hn = (frequencies * 1e12 * unit.hertz * unit.planck_constant).to('eV').magnitude
    kBT = (temperature * unit.kelvin * unit.boltzmann_constant).to('eV').magnitude
    kBT = np.atleast_1d(kBT)
    values = np.zeros_like(kBT)
    values[kBT > 0] = np.sum(
        hn / (np.exp(-np.einsum('i,...->...i', hn, 1 / kBT[kBT > 0])) - 1), axis=-1
    )
    return 0.5 * np.sum(hn) + values + E_0


def generate_displacements(structure, symprec=1.0e-2):
    sym = structure.get_symmetry(symprec=symprec)
    _, comp = np.unique(sym.arg_equivalent_vectors, return_index=True)
    ind_x, ind_y = np.unravel_index(comp, structure.positions.shape)
    displacements = np.zeros((len(ind_x),) + structure.positions.shape)
    displacements[np.arange(len(ind_x)), ind_x, ind_y] = 1
    return displacements


class Hessian:
    def __init__(self, structure, dx=0.01, symprec=1.0e-2, include_zero_displacement=True):
        """

        Args:
            structure (pyiron_atomistics.structure.atoms.Atoms): Structure
            dx (float): Displacement (in distance unit)
            symprec (float): Symmetry search precision
            include_zero_displacement (bool): Whether to include zero displacement, which is not
                required to get the force constants, but increases the precision, especially if
                the initial positions are not the relaxed structure.
        """
        self.structure = structure.copy()
        self._symmetry = None
        self._permutations = None
        self.dx = dx
        self._displacements = []
        self._forces = None
        self._inequivalent_displacements = None
        self._symprec = symprec
        self._nu = None
        self._energy = None
        self._hessian = None
        self.include_zero_displacement = include_zero_displacement

    @property
    def symmetry(self):
        if self._symmetry is None:
            self._symmetry = self.structure.get_symmetry(symprec=self._symprec)
        return self._symmetry

    def reset(self):
        self._inequivalent_displacements = None
        self._nu = None
        self._hessian = None

    @property
    def energy(self):
        return self._energy

    @energy.setter
    def energy(self, energy):
        if energy is not None and len(energy) != len(self.displacements):
            raise AssertionError('Energy shape does not match existing displacement shape')
        self._energy = np.array(energy)
        self.reset()

    @property
    def forces(self):
        return self._forces

    @forces.setter
    def forces(self, forces):
        if np.array(forces).shape != self.displacements.shape:
            raise AssertionError('Force shape does not match existing displacement shape')
        self._forces = np.array(forces)
        self.reset()

    @property
    def all_displacements(self):
        return np.einsum(
            'nxy,lnmy->lnmx', self.symmetry.rotations,
            self.displacements[:, self.symmetry.permutations]
        ).reshape(-1, np.prod(self.structure.positions.shape))

    @property
    def inequivalent_indices(self):
        return np.unique(self.all_displacements, axis=0, return_index=True)[1]

    @property
    def inequivalent_displacements(self):
        if self._inequivalent_displacements is None:
            self._inequivalent_displacements = self.all_displacements[self.inequivalent_indices]
            self._inequivalent_displacements -= self.origin.flatten()
        return self._inequivalent_displacements

    @property
    def inequivalent_forces(self):
        forces = np.einsum(
            'nxy,lnmy->lnmx', self.symmetry.rotations, self.forces[:, self.symmetry.permutations]
        ).reshape(-1, np.prod(self.structure.positions.shape))
        return forces[self.inequivalent_indices]

    @property
    def _x_outer(self):
        return np.einsum('ik,ij->kj', *2 * [self.inequivalent_displacements])

    @property
    def hessian(self):
        if self._hessian is None:
            H = -np.einsum(
                'kj,in,ik->nj',
                np.linalg.inv(self._x_outer),
                self.inequivalent_forces,
                self.inequivalent_displacements,
                optimize=True
            )
            self._hessian = 0.5 * (H + H.T)
        return self._hessian

    @property
    def _mass_tensor(self):
        m = np.tile(self.structure.get_masses(), (3, 1)).T.flatten()
        return np.sqrt(m * m[:, np.newaxis])

    @property
    def vibrational_frequencies(self):
        if self._nu is None:
            H = self.hessian
            nu_square = (np.linalg.eigh(
                H / self._mass_tensor
            )[0] * unit.electron_volt / unit.angstrom**2 / unit.amu).to('THz**2').magnitude
            self._nu = np.sign(nu_square) * np.sqrt(np.absolute(nu_square)) / (2 * np.pi)
        return self._nu

    def get_free_energy(self, temperature):
        return get_helmholtz_free_energy(
            self.vibrational_frequencies[3:], temperature, self.min_energy
        )

    def get_potential_energy(self, temperature):
        return get_potential_energy(
            self.vibrational_frequencies[3:], temperature, self.min_energy
        )

    @property
    def minimum_displacements(self):
        return generate_displacements(structure=self.structure, symprec=self._symprec) * self.dx

    @property
    def displacements(self):
        if len(self._displacements) == 0:
            self.displacements = self.minimum_displacements
            if self.include_zero_displacement:
                self.displacements = np.concatenate(
                    ([np.zeros_like(self.structure.positions)], self.displacements), axis=0
                )
        return np.asarray(self._displacements)

    @displacements.setter
    def displacements(self, d):
        self._displacements = np.asarray(d).tolist()
        self._inequivalent_displacements = None

    @property
    def _fit(self):
        E = self.energy + 0.5 * np.einsum('nij,nij->n', self.forces, self.displacements)
        E = np.repeat(E, len(self.symmetry.rotations))[self.inequivalent_indices]
        reg = LinearRegression()
        reg.fit(self.inequivalent_forces, E)
        return reg

    @property
    def min_energy(self):
        return self._fit.intercept_

    @property
    def origin(self):
        if self.energy is None or self.forces is None:
            return np.zeros_like(self.structure.positions)
        return 2 * self.symmetry.symmetrize_vectors(self._fit.coef_.reshape(-1, 3))

    @property
    def volume(self):
        return self.structure.get_volume()


class QHAJobGenerator(JobGenerator):
    @property
    def parameter_list(self):
        """

        Returns:
            (list)
        """
        cell, positions = self._master.get_supercells_with_displacements()
        return [[c, p] for c, p in zip(cell, positions)]

    @staticmethod
    def modify_job(job, parameter):
        job.structure.set_cell(parameter[0], scale_atoms=True)
        job.structure.set_scaled_positions(parameter[1])
        return job


class QuasiHarmonicApproximation(AtomisticParallelMaster):
    """
    Launch Quasi Harmonic Approximation jobs to calculate free energies and derived thermodynamic
    quantities.


    Example:

    >>> qha = pr.create.job.QuasiHarmonicApproximation('qha')
    >>> qha.ref_job = your_job
    >>> qha.run()
    >>> qha.get_helmholtz_free_energy(temperature=300, strain=0)


    Important differences to the existing `QuasiHarmonicJob`:

    - It does not do an internal multiplication of the structure, i.e. the input structure
        directly defines the interaction range. For this reason, you might be interested to make
        sure that the structure is large enough.
    - The minimum energy of each of the strained structures is estimated using the formula:
        `U = U_0 + 0.5 * \sum_i forces_i * (dx_i - dX_i)`
        where `dx_i` are the internally generated displacement field, `U` is the potential energy,
        `forces_i` are the forces and `U_0` and `dX_i` are fitting parameters. The same formula
        also estimates the correct force-free positions and therefore the correct displacement
        field for the given strain.
    - If you don't need a volume or pressure dependence, you can set
        `self.input['num_points'] = 1`, in which case all values are returned for the given
        volume of the reference structure.
    """
    def __init__(self, project, job_name):
        """

        Args:
            project: project
            job_name: job name
        """
        super().__init__(project, job_name)
        self.__name__ = "QuasiHarmonicApproximation"
        self.__version__ = "0.0.1"
        self.input["displacement"] = (0.01, "Atom displacement magnitude")
        self.input['symprec'] = 1.0e-2
        self.input['include_zero_displacement'] = True
        self.input["num_points"] = (11, "number of sample points")
        self.input["vol_range"] = (
            0.1,
            "relative volume variation around volume defined by ref_job",
        )
        self._job_generator = QHAJobGenerator(self)
        self._hessian = None
        self._td = None

    @property
    def strain_lst(self):
        if self.input['num_points'] == 1:
            return [0]
        return np.linspace(-1, 1, self.input['num_points']) * self.input['vol_range'] / 2

    @property
    def hessian(self):
        if self._hessian is None:
            self._hessian = Hessian(self.structure)
        return self._hessian

    def get_supercells_with_displacements(self):
        dx = self.structure.positions + self.hessian.displacements
        sc_dx = np.einsum('ij,...i->...j', np.linalg.inv(self.structure.cell), dx)
        positions = np.concatenate((len(self.strain_lst)*[sc_dx]))
        cell = np.repeat(
            self.structure.cell * (1 + self.strain_lst[:, np.newaxis, np.newaxis]),
            len(self.hessian.displacements),
            axis=0
        )
        return cell, positions

    def to_hdf(self, hdf=None, group_name=None):
        """
        Store the QHA in an HDF5 file

        Args:
            hdf (ProjectHDFio): HDF5 group object - optional
            group_name (str): HDF5 subgroup name - optional
        """
        super().to_hdf(hdf=hdf, group_name=group_name)
        with self.project_hdf5.open("input") as hdf5_input:
            hdf5_input["displacements"] = self.hessian.displacements

    def from_hdf(self, hdf=None, group_name=None):
        """
        Restore the QHA from an HDF5 file

        Args:
            hdf (ProjectHDFio): HDF5 group object - optional
            group_name (str): HDF5 subgroup name - optional
        """
        super().from_hdf(hdf=hdf, group_name=group_name)
        with self.project_hdf5.open("input") as hdf5_input:
            if 'displacements' in hdf5_input.list_nodes():
                self.hessian.displacements = hdf5_input['displacements']

    def load_output(self):
        if self.ref_job.server.run_mode.interactive:
            inspect = self.project_hdf5.inspect(self.child_ids[0])
            forces_lst = inspect["output/generic/forces"]
            energy_lst = inspect['output/generic/energy_pot']
        else:
            pr_job = self.project_hdf5.project.open(self.job_name + "_hdf5")
            forces_lst, energy_lst = [], []
            for job_name in self._get_jobs_sorted():
                inspect = pr_job.inspect(job_name)
            forces_lst.append(inspect["output/generic/forces"])
            energy_lst.append(inspect['output/generic/energy_pot'])
        forces_lst = np.asarray(forces_lst).reshape(
            (self.input['num_points'], ) + self.hessian.displacements.shape
        )
        energy_lst = np.asarray(energy_lst).reshape(
            self.input['num_points'], len(self.hessian.displacements)
        )
        return forces_lst, energy_lst

    def collect_output(self):
        """

        Returns:

        """
        forces, energy = self.load_output()
        nu_lst, h_lst, e_lst = [], [], []
        for f, e in zip(forces, energy):
            self.hessian.forces = f
            self.hessian.energy = e
            h_lst.append(self.hessian.hessian)
            nu_lst.append(self.hessian.vibrational_frequencies)
            e_lst.append(self.hessian.min_energy)

        with self.project_hdf5.open("output") as hdf5_out:
            hdf5_out["force_constants"] = np.asarray(h_lst)
            hdf5_out["vibrational_frequencies"] = np.asarray(nu_lst)
            hdf5_out["box_energy"] = np.asarray(e_lst)

    @property
    def _thermo(self):
        if self._td is None:
            self._td = Thermodynamics(
                self.strain_lst,
                self['output/vibrational_frequencies'],
                self['output/box_energy'],
                self.hessian.volume
            )
        return self._td

    def get_helmholtz_free_energy(self, temperature, strain=None):
        return self._thermo.get_helmholtz_free_energy(temperature=temperature, strain=strain)

    def get_gibbs_free_energy(self, temperature, pressure):
        return self._thermo.get_gibbs_free_energy(temperature=temperature, pressure=pressure)

    def get_volume(self, temperature, pressure):
        return self.hessian.volume * (
            1 + self.get_strain(temperature=temperature, pressure=pressure)
        )

    def get_strain(self, temperature, pressure):
        return self._thermo.get_strain(temperature=temperature, pressure=pressure)

    def get_pressure(self, temperature, strain):
        return self._thermo.get_pressure(temperature=temperature, strain=strain)

    def validate_ready_to_run(self):
        if self.ref_job._generic_input["calc_mode"] != "static":
            raise ValueError("Phonopy reference jobs should be static calculations, but got {}".format(
                self.ref_job._generic_input["calc_mode"]
            ))
        super().validate_ready_to_run()


class Thermodynamics:
    def __init__(self, strain, nu, E, volume):
        """

        Args:
            strain ((n_snapshots, n_atoms, 3)-np.ndarray): List of strain values
            nu ((n_snapshots, 3 * n_atoms)-np.ndarray): Vibrational frequencies
            E ((n_snapshots,)-np.ndarray): Force free energy values
        """
        self.strain = strain
        self.nu = nu
        self.E = E
        self.volume = volume
        self._fit_mat = None
        self._temperature = None
        self._free_energy = None

    @property
    def coeff(self):
        return np.einsum('ni,nj->ji', self.fit_mat, self.free_energy)

    @property
    def free_energy(self):
        if self._free_energy is None:
            self._free_energy = np.array([
                get_helmholtz_free_energy(nu, self.temperature, E)
                for nu, E in zip(self.nu, self.E)
            ])
        return self._free_energy

    @staticmethod
    def _harmonize(T, V):
        if T is None or V is None:
            return T, V
        V = np.atleast_1d(V)
        T = np.atleast_1d(T) * np.ones_like(V)
        V = V * np.ones_like(T)
        return T, V

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, T):
        self._temperature = np.asarray([T]).flatten()
        self._free_energy = None

    @property
    def fit_mat(self):
        if self._fit_mat is None:
            s_ni = self.strain[:, np.newaxis]**np.arange(4)
            S_inv = np.linalg.inv(np.einsum('ni,nk->ik', *2 * [s_ni]))
            self._fit_mat = np.einsum('ik,nk->ni', S_inv, s_ni)
        return self._fit_mat

    def _get_min_strain(self, pressure):
        c = self.coeff.T
        return (-c[2] + np.sqrt(c[2]**2 - 3 * c[3] * (c[1] + pressure.flatten()))) / (3 * c[3])

    def get_helmholtz_free_energy(self, temperature, strain=None):
        temperature, strain = self._harmonize(temperature, strain)
        self.temperature = temperature
        if strain is None or len(self.strain) == 1:
            return self.free_energy.squeeze()
        return np.einsum(
            'ik,ik->i', self.coeff, strain.flatten()[..., np.newaxis]**np.arange(4)
        ).reshape(np.shape(temperature)).squeeze()

    def get_gibbs_free_energy(self, temperature, pressure):
        temperature, pressure = self._harmonize(temperature, pressure)
        pV = (
            (pressure * self.volume * 1e9 * unit.Pa * unit.angstrom**3).to('eV')
        ).magnitude
        self.temperature = temperature
        strain = self._get_min_strain(pV)
        return np.einsum(
            'ik,ik->i', self.coeff, strain.flatten()[:, np.newaxis]**np.arange(4)
        ).reshape(np.shape(temperature))

    def get_strain(self, temperature, pressure):
        temperature, pressure = self._harmonize(temperature, pressure)
        pV = (
            (pressure * self.volume * 1e9 * unit.Pa * unit.angstrom**3).to('eV')
        ).magnitude
        self.temperature = temperature
        return self._get_min_strain(pV).reshape(np.shape(temperature))

    def get_pressure(self, temperature, strain):
        temperature, strain = self._harmonize(temperature, strain)
        self.temperature = temperature
        pressure = -np.einsum(
            'ik,ik->i', (self.coeff * np.arange(4))[:, 1:],
            np.array([strain]).flatten()[:, np.newaxis]**np.arange(3)
        ).reshape(np.shape(strain)) / self.volume
        return (pressure / 1.0e9 * unit.eV / unit.angstrom**3).to(unit.Pa).magnitude
