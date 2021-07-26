from scipy import sparse
import numpy as np


class QuasiNewton:
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

    def _initialize_hessian(self, starting_h=10, diffusion_id=None, diffusion_direction=None):
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

    def get_dx(self, g):
        self.update_hessian(g)
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
