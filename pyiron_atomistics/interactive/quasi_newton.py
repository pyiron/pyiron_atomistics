import numpy as np
from pyiron_base import DataContainer
import warnings
from scipy.spatial import cKDTree
import pint
from pyiron_atomistics.atomistics.job.interactivewrapper import (
    InteractiveWrapper,
    ReferenceJobOutput,
)


class QuasiNewtonInteractive:
    def __init__(
        self,
        structure,
        starting_h=10,
        diffusion_id=None,
        use_eigenvalues=True,
        diffusion_direction=None,
        symmetrize=True,
        max_displacement=0.1,
    ):
        self.use_eigenvalues = use_eigenvalues
        self._hessian = None
        self._eigenvalues = None
        self._eigenvectors = None
        self.g_old = None
        self.symmetry = None
        self.max_displacement = max_displacement
        self.regularization = None
        if symmetrize:
            self.symmetry = structure.get_symmetry()
        self._initialize_hessian(
            structure=structure,
            starting_h=starting_h,
            diffusion_id=diffusion_id,
            diffusion_direction=diffusion_direction
        )

    def _initialize_hessian(
        self, structure, starting_h=10, diffusion_id=None, diffusion_direction=None
    ):
        if np.prod(np.array(starting_h).shape) == np.prod(structure.positions.shape)**2:
            self.hessian = starting_h
        else:
            self.hessian = starting_h * np.eye(np.prod(structure.positions.shape))
        if diffusion_id is not None and diffusion_direction is not None:
            v = np.zeros_like(structure.positions)
            v[diffusion_id] = diffusion_direction
            v = v.flatten()
            self.hessian -= (starting_h + 1) * np.einsum('i,j->ij', v, v) / np.linalg.norm(v)**2
            self.use_eigenvalues = True
        elif diffusion_id is not None or diffusion_direction is not None:
            raise ValueError('diffusion id or diffusion direction not specified')

    def _set_regularization(self, g, max_cycle=20, max_value=20, tol=1.0e-8):
        self.regularization = 1
        for _ in range(max_cycle):
            H_inv = self.inv_hessian
            gH = H_inv.dot(g)
            dH_inv = self._get_d_inv_hessian(H_inv, gH, g)
            value = np.absolute(gH).max() - self.max_displacement
            if np.absolute(value) < tol:
                break
            self.regularization -= value / dH_inv
            if np.absolute(self.regularization) > max_value:
                self.regularization = max_value * np.sign(self.regularization)

    def _get_d_inv_hessian(self, H_inv, gH, g):
        s = np.sign(gH[np.absolute(gH).argmax()])
        if self.use_eigenvalues:
            return -np.einsum(
                ',k,k,k,jk,j->',
                np.exp(self.regularization),
                self.eigenvectors[np.absolute(gH).argmax()],
                self.eigenvalues,
                1 / (self.eigenvalues**2 + np.exp(self.regularization))**2,
                self.eigenvectors, g,
                optimize=True
            ) * s
        else:
            return -np.einsum(
                ',j,j->',
                np.exp(self.regularization),
                H_inv[np.absolute(gH).argmax()],
                gH, optimize=True
            ) * s

    @property
    def inv_hessian(self):
        if self.regularization is None:
            return np.linalg.inv(self.hessian)
        if self.use_eigenvalues:
            return np.einsum(
                'ik,k,jk->ij',
                self.eigenvectors,
                self.eigenvalues / (self.eigenvalues**2 + np.exp(self.regularization)),
                self.eigenvectors
            )
        else:
            return np.linalg.inv(self.hessian + np.eye(len(self.hessian)) * np.exp(self.regularization))

    @property
    def hessian(self):
        return self._hessian

    @hessian.setter
    def hessian(self, v):
        self._hessian = np.array(v)
        length = int(np.sqrt(np.prod(self._hessian.shape)))
        self._hessian = self._hessian.reshape(length, length)
        self._eigenvalues = None
        self._eigenvectors = None
        self.regularization = None

    def _calc_eig(self):
        self._eigenvalues, self._eigenvectors = np.linalg.eigh(self.hessian)

    @property
    def eigenvalues(self):
        if self._eigenvalues is None:
            self._calc_eig()
        return self._eigenvalues

    @property
    def eigenvectors(self):
        if self._eigenvectors is None:
            self._calc_eig()
        return self._eigenvectors

    def get_dx(self, g, threshold=1e-4, mode='PSB', update_hessian=True):
        if update_hessian:
            self.update_hessian(g, threshold=threshold, mode=mode)
        self.dx = -np.einsum('ij,j->i', self.inv_hessian, g.flatten()).reshape(-1, 3)
        if self.symmetry is not None:
            self.dx = self.symmetry.symmetrize_vectors(self.dx)
        if np.linalg.norm(
            self.dx, axis=-1
        ).max() > self.max_displacement and self.regularization is None:
            self._set_regularization(g=g.flatten())
            return self.get_dx(g=g, threshold=threshold, mode=mode, update_hessian=False)
        return self.dx

    def _get_SR(self, dx, dg, H_tmp, threshold=1e-4):
        denominator = np.dot(H_tmp, dx)
        if np.absolute(denominator) < threshold:
            denominator += threshold
        return np.outer(H_tmp, H_tmp) / denominator

    def _get_PSB(self, dx, dg, H_tmp):
        dxdx = np.einsum('i,i->', dx, dx)
        dH = np.einsum('i,j->ij', H_tmp, dx)
        dH = (dH + dH.T) / dxdx
        return dH - np.einsum('i,i,j,k->jk', dx, H_tmp, dx, dx, optimize='optimal') / dxdx**2

    def _get_BFGS(self, dx, dg, H_tmp):
        return np.outer(dg, dg) / dg.dot(dx) - np.outer(H_tmp, H_tmp) / dx.dot(H_tmp)

    def update_hessian(self, g, threshold=1e-4, mode='PSB'):
        if self.g_old is None:
            self.g_old = g
            return
        dg = self.get_dg(g).flatten()
        dx = self.dx.flatten()
        H_tmp = dg - np.einsum('ij,j->i', self.hessian, dx)
        if mode == 'SR':
            self.hessian = self._get_SR(dx, dg, H_tmp) + self.hessian
        elif mode == 'PSB':
            self.hessian = self._get_PSB(dx, dg, H_tmp) + self.hessian
        elif mode == 'BFGS':
            self.hessian = self._get_BFGS(dx, dg, H_tmp) + self.hessian
        else:
            raise ValueError(
                'Mode not recognized: {}. Choose from `SR`, `PSB` and `BFGS`'.format(mode)
            )
        self.g_old = g

    def get_dg(self, g):
        return g - self.g_old


def run_qn(
    job,
    mode='PSB',
    ionic_steps=100,
    ionic_force_tolerance=1.0e-2,
    ionic_energy_tolerance=0,
    starting_h=10,
    diffusion_id=None,
    use_eigenvalues=True,
    diffusion_direction=None,
    symmetrize=True,
    max_displacement=0.1,
    min_displacement=1.0e-8,
):
    qn = QuasiNewtonInteractive(
        structure=job.structure,
        starting_h=starting_h,
        diffusion_id=diffusion_id,
        use_eigenvalues=use_eigenvalues,
        diffusion_direction=diffusion_direction,
        max_displacement=max_displacement,
        symmetrize=symmetrize,
    )
    job.run()
    for _ in range(ionic_steps):
        f = job.output.forces[-1]
        if np.linalg.norm(f, axis=-1).max() < ionic_force_tolerance:
            break
        dx = qn.get_dx(-f, mode=mode)
        if np.linalg.norm(dx, axis=-1).max() < min_displacement:
            warnings.warn('line search alpha is zero')
            break
        job.structure.positions += dx
        job.structure.center_coordinates_in_unit_cell()
        job.run()
    return qn


class QuasiNewton(InteractiveWrapper):
    def __init__(self, project, job_name):
        super().__init__(project, job_name)
        self.__name__ = "QuasiNewton"
        self.__version__ = (
            None
        )  # Reset the version number to the executable is set automatically
        self.input = Input()
        self.output = Output(self)
        self._interactive_interface = None

    def _run(self):
        run_qn(
            job=self.ref_job,
            mode=self.input.mode,
            ionic_steps=self.input.ionic_steps,
            ionic_force_tolerance=self.input.ionic_force_tolerance,
            ionic_energy_tolerance=self.input.ionic_energy_tolerance,
            starting_h=self.input.starting_h,
            diffusion_id=self.input.diffusion_id,
            use_eigenvalues=self.input.use_eigenvalues,
            diffusion_direction=self.input.diffusion_direction,
            symmetrize=self.input.symmetrize
        )
        self.collect_output()

    def run_if_interactive(self):
        self._run()

    def run_static(self):
        self.status.running = True
        self.ref_job_initialize()
        self._run()
        if self.ref_job.server.run_mode.interactive:
            self.ref_job.interactive_close()
        self.status.collect = True
        self.run()

    def interactive_open(self):
        self.server.run_mode.interactive = True
        self.ref_job.interactive_open()

    def interactive_close(self):
        self.status.collect = True
        if self.ref_job.server.run_mode.interactive:
            self.ref_job.interactive_close()
        self.run()

    def write_input(self):
        pass

    def to_hdf(self, hdf=None, group_name=None):
        super().to_hdf(
            hdf=hdf,
            group_name=group_name
        )

    def from_hdf(self, hdf=None, group_name=None):
        super().from_hdf(
            hdf=hdf,
            group_name=group_name
        )

    def collect_output(self):
        self.output._index_lst.append(len(self.ref_job.output.energy_pot))


class Input(DataContainer):
    """
    Args:
        minimizer (str): minimizer to use (currently only 'CG' and 'BFGS' run
            reliably)
        ionic_steps (int): max number of steps
        ionic_force_tolerance (float): maximum force tolerance
    """

    def __init__(self, input_file_name=None, table_name="input"):
        self.mode = 'PSB'
        self.ionic_steps = 100
        self.ionic_force_tolerance = 1.0e-2
        self.ionic_energy_tolerance = 0
        self.starting_h = 10
        self.diffusion_id = None
        self.use_eigenvalues = True
        self.diffusion_direction = None
        self.symmetrize = True


class Output(ReferenceJobOutput):
    def __init__(self, job):
        super().__init__(job=job)
        self._index_lst = []

    @property
    def index_lst(self):
        return np.asarray(self._index_lst)


class Hessian:
    def __init__(self, structure, dx=0.01):
        self.structure = structure.copy()
        self._symmetry = None
        self._permutations = None
        self.dx = dx
        self._displacements = []
        self._all_displacements = None
        self._forces = None
        self._unique_atoms = None

    @property
    def unique_atoms(self):
        if self._unique_atoms is None:
            self._unique_atoms = np.unique(self.symmetry.arg_equivalent_atoms)
        return self._unique_atoms

    @property
    def symmetry(self):
        if self._symmetry is None:
            self._symmetry = self.structure.get_symmetry()
        return self._symmetry

    @property
    def permutations(self):
        if self._permutations is None:
            epsilon = 1.0e-8
            x_scale = self.structure.get_scaled_positions()
            x = np.einsum(
                'nxy,my->mnx', self.symmetry.rotations, x_scale
            ) + self.symmetry.translations
            if any(self.structure.pbc):
                x[:, :, self.structure.pbc] -= np.floor(x[:, :, self.structure.pbc] + epsilon)
            x = np.einsum('nmx->mnx', x)
            tree = cKDTree(x_scale)
            self._permutations = np.argsort(tree.query(x)[1], axis=-1)
        return self._permutations

    def _get_equivalent_vector(self, v, indices=None):
        result = np.einsum('nxy,nmy->nmx', self.symmetry.rotations, v[self.permutations])
        if indices is None:
            indices = np.sort(np.unique(result, return_index=True, axis=0)[1])
        return result[indices], indices

    @property
    def forces(self):
        return self._forces

    @forces.setter
    def forces(self, forces):
        if np.array(forces).shape != self.displacements.shape:
            raise AssertionError('Force shape does not match existing displacement shape')
        self._forces = np.array(forces)

    @property
    def all_displacements(self):
        if self._all_displacements is None:
            self._all_displacements = np.einsum(
                'nxy,lnmy->lnmx', self.symmetry.rotations, self.displacements[:, self.permutations]
            ).reshape(-1, np.prod(self.structure.positions.shape))
        return self._all_displacements

    @property
    def inequivalent_indices(self):
        return np.unique(self.all_displacements, axis=0, return_index=True)[1]

    @property
    def inequivalent_displacements(self):
        return self.all_displacements[self.inequivalent_indices]

    @property
    def inequivalent_forces(self):
        forces = np.einsum(
            'nxy,lnmy->lnmx', self.symmetry.rotations, self.forces[:, self.permutations]
        ).reshape(-1, np.prod(self.structure.positions.shape))
        return forces[self.inequivalent_indices]

    @property
    def _sum_displacements(self):
        if len(self._displacements) == 0:
            displacements = np.zeros((len(self.unique_atoms),) + self.structure.positions.shape)
            displacements[np.arange(len(self.unique_atoms)), self.unique_atoms, 0] = self.dx
            self.displacements = displacements
        return np.absolute(self.inequivalent_displacements).sum(axis=0).reshape(
            self.structure.positions.shape
        )

    @property
    def _next_displacement(self):
        ix = np.stack(np.where(self._sum_displacements == 0), axis=-1)
        if len(ix) == 0:
            return None
        ix = ix[np.unique(ix[:, 0], return_index=True)[1]]
        ix = ix[np.any(ix[:, 0, None] == self.unique_atoms, axis=1)]
        displacements = np.zeros((len(ix), ) + self.structure.positions.shape)
        displacements[np.arange(len(ix)), ix[:, 0], ix[:, 1]] = self.dx
        return displacements.tolist()

    def _generate_displacements(self):
        for _ in range(np.prod(self.structure.positions.shape)):
            displacement = self._next_displacement
            if displacement is None:
                break
            self._displacements.extend(displacement)
            self._all_displacements = None

    def get_hessian(self, forces=None):
        if forces is None and self.forces is None:
            raise AssertionError('Forces not set yet')
        if forces is not None:
            self.forces = forces
        X = np.einsum(
            'ik,ij->kj',
            self.inequivalent_displacements,
            self.inequivalent_displacements, optimize=True
        )
        return -np.einsum(
            'kj,in,ik->nj',
            np.linalg.inv(X),
            self.inequivalent_forces,
            self.inequivalent_displacements,
            optimize=True
        )

    @property
    def _to_THz(self):
        u = pint.UnitRegistry()
        return np.sqrt((1 * (u.electron_volt / u.angstrom**2 / u.amu)).to('THz**2').magnitude)

    @property
    def _mass_tensor(self):
        m = np.tile(self.structure.get_masses(), (3, 1)).T.flatten()
        return np.sqrt(m * m[:, np.newaxis])

    def get_vibrational_frequencies(self, forces=None):
        H = self.get_hessian(forces=forces)
        nu_square = np.linalg.eigh(H / self._mass_tensor)[0]
        return np.sign(nu_square) * np.sqrt(np.absolute(nu_square)) / (2 * np.pi) * self._to_THz

    @property
    def displacements(self):
        if len(self._displacements) == 0:
            self._generate_displacements()
        return np.asarray(self._displacements)

    @displacements.setter
    def displacements(self, d):
        self._displacements = np.asarray(d).tolist()
        self._all_displacements = None
