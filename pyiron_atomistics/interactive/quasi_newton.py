from scipy import sparse
import numpy as np
from pyiron_base import DataContainer
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
        regularization=1e-6,
    ):
        self.use_eigenvalues = use_eigenvalues
        self._hessian = None
        self._eigenvalues = None
        self._eigenvectors = None
        self.g_old = None
        self._initialize_hessian(
            structure=structure,
            starting_h=starting_h,
            diffusion_id=diffusion_id,
            diffusion_direction=diffusion_direction
        )
        if self.use_eigenvalues:
            self.regularization = regularization**2
        else:
            self.regularization = regularization
        if self.use_eigenvalues and self.regularization==0:
            raise ValueError('Regularization must be larger than 0 when eigenvalues are used')

    def _initialize_hessian(
        self, structure, starting_h=10, diffusion_id=None, diffusion_direction=None
    ):
        self.hessian = starting_h*np.eye(np.prod(structure.positions.shape))
        if diffusion_id is not None and diffusion_direction is not None:
            v = np.zeros_like(structure.positions)
            v[diffusion_id] = diffusion_direction
            v = v.flatten()
            self.hessian -= (starting_h+1)*np.einsum('i,j->ij', v, v)/np.linalg.norm(v)**2
            self.use_eigenvalues = True
        elif diffusion_id is not None or diffusion_direction is not None:
            raise ValueError('diffusion id or diffusion direction not specified')

    @property
    def inv_hessian(self):
        if self.regularization > 0:
            if self.use_eigenvalues:
                return np.einsum(
                    'ik,k,jk->ij',
                    self.eigenvectors,
                    self.eigenvalues/(self.eigenvalues**2+self.regularization),
                    self.eigenvectors
                )
            else:
                return np.linnalg.inv(self.hessian+np.eye(len(self.hessian))*self.regularization)
        return np.linnalg.inv(self.hessian)

    @property
    def hessian(self):
        return self._hessian

    @hessian.setter
    def hessian(self, v):
        self._eigenvalues = None
        self._eigenvectors = None
        self._hessian = v

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

    def get_dx(self, g, threshold=1e-4, mode='PSB'):
        self.update_hessian(g, threshold=threshold, mode=mode)
        self.dx = -np.einsum('ij,j->i', self.inv_hessian, g.flatten()).reshape(-1, 3)
        return self.dx
    
    def update_hessian(self, g, threshold=1e-4, mode='PSB'):
        if self.g_old is None:
            self.g_old = g
            return
        dg = self.get_dg(g).flatten()
        dx = self.dx.flatten()
        H_tmp = dg-np.einsum('ij,j->i', self.hessian, dx)
        if mode=='SR':
            denominator = np.dot(H_tmp, self.dx.flatten())
            if np.absolute(denominator) > threshold:
                dH = np.outer(H_tmp, H_tmp)/denominator
                self.hessian = dH+self.hessian
        elif mode=='PSB':
            dxdx = np.einsum('i,i->', dx, dx)
            dH = np.einsum('i,j->ij', H_tmp, dx)
            dH = (dH+dH.T)/dxdx
            dH -= np.einsum('i,i,j,k->jk', dx, H_tmp, dx, dx, optimize='optimal')/dxdx**2
            self.hessian = dH+self.hessian
        self.g_old = g            

    def get_dg(self, g):
        return g-self.g_old

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
    regularization=1e-6,
):
    qn = QuasiNewtonInteractive(
        structure=job.structure,
        starting_h=starting_h,
        diffusion_id=diffusion_id,
        use_eigenvalues=use_eigenvalues,
        diffusion_direction=diffusion_direction,
        regularization=regularization,
    )
    job.run()
    for _ in range(ionic_steps):
        f = job.output.forces[-1]
        if np.linalg.norm(f, axis=-1).max() < ionic_force_tolerance:
            break
        dx = qn.get_dx(-f, mode=mode)
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
            regularization=self.input.regularization
        )
        self.collect_output()

    def run_if_interactive(self):
        self._run()

    def run_if_static(self):
        self._run()
        self.interactive_close()

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
        self.regularization = 1e-6

class Output(ReferenceJobOutput):
    def __init__(self, job):
        super().__init__(job=job)
        self._index_lst = []

    @property
    def index_lst(self):
        return np.asarray(self._index_lst)
