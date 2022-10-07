import numpy as np
from pyiron_base import GenericParameters
import warnings
from pyiron_atomistics.atomistics.job.interactivewrapper import (
    InteractiveWrapper,
    ReferenceJobOutput,
)


class QuasiNewtonInteractive:
    """
    Interactive class of Quasi Newton. This class can be used without a pyiron job definition.
    After the initialization, the displacement is obtained by calling `get_dx` successively.
    """

    def __init__(
        self,
        structure,
        starting_h=10,
        diffusion_id=None,
        diffusion_direction=None,
        use_eigenvalues=True,
        symmetrize=True,
        max_displacement=0.1,
    ):
        """
        Args:
            structure (pyiron_atomistics.atomistics.structure.atoms.Atoms): pyiron structure
            starting_H (float/ndarray): Starting Hessian value (diagonal value or total Hessian)
            diffusion_id (int/None): Atom id at saddle point. No need to define if the structure
                is close enough to the saddle point. This has to be defined together with
                `diffusion_direction`.
            use_eigenvalues (bool): Whether to use the eigenvalue softening or standard Tikhonov
                regularization to prevent unphysical displacement.
            symmetrize (bool): Whether to symmetrize forces following the box symmetries. DFT
                calculations might fail if set to `False`
            max_displacement (float): Maximum displacement allowed for an atom.
        """
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
            diffusion_direction=diffusion_direction,
        )

    def _initialize_hessian(
        self, structure, starting_h=10, diffusion_id=None, diffusion_direction=None
    ):
        if (
            np.prod(np.array(starting_h).shape)
            == np.prod(structure.positions.shape) ** 2
        ):
            self.hessian = starting_h
        else:
            self.hessian = starting_h * np.eye(np.prod(structure.positions.shape))
        if diffusion_id is not None and diffusion_direction is not None:
            v = np.zeros_like(structure.positions)
            v[diffusion_id] = diffusion_direction
            v = v.flatten()
            self.hessian -= (
                (starting_h + 1) * np.einsum("i,j->ij", v, v) / np.linalg.norm(v) ** 2
            )
            self.use_eigenvalues = True
        elif diffusion_id is not None or diffusion_direction is not None:
            raise ValueError("diffusion id or diffusion direction not specified")

    def _set_regularization(self, g, max_cycle=20, max_value=20, tol=1.0e-8):
        self.regularization = -2
        for _ in range(max_cycle):
            if np.absolute(self.inv_hessian.dot(g)).max() < self.max_displacement:
                break
            self.regularization += 1
            if np.absolute(self.regularization) > max_value:
                self.regularization = max_value

    @property
    def inv_hessian(self):
        if self.regularization is None:
            return np.linalg.inv(self.hessian)
        if self.use_eigenvalues:
            return np.einsum(
                "ik,k,jk->ij",
                self.eigenvectors,
                self.eigenvalues
                / (self.eigenvalues**2 + np.exp(self.regularization)),
                self.eigenvectors,
            )
        else:
            return np.linalg.inv(
                self.hessian + np.eye(len(self.hessian)) * np.exp(self.regularization)
            )

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

    def get_dx(self, g, threshold=1e-4, mode="PSB", update_hessian=True):
        if update_hessian:
            self.update_hessian(g, threshold=threshold, mode=mode)
        self.dx = -np.einsum("ij,j->i", self.inv_hessian, g.flatten()).reshape(-1, 3)
        if self.symmetry is not None:
            self.dx = self.symmetry.symmetrize_vectors(self.dx)
        if (
            np.linalg.norm(self.dx, axis=-1).max() > self.max_displacement
            and self.regularization is None
        ):
            self._set_regularization(g=g.flatten())
            return self.get_dx(
                g=g, threshold=threshold, mode=mode, update_hessian=False
            )
        return self.dx

    @staticmethod
    def _get_SR(dx, dg, H_tmp, threshold=1e-4):
        denominator = np.dot(H_tmp, dx)
        if np.absolute(denominator) < threshold:
            denominator += threshold
        return np.outer(H_tmp, H_tmp) / denominator

    @staticmethod
    def _get_PSB(dx, dg, H_tmp):
        dxdx = np.einsum("i,i->", dx, dx)
        dH = np.einsum("i,j->ij", H_tmp, dx)
        dH = (dH + dH.T) / dxdx
        return (
            dH
            - np.einsum("i,i,j,k->jk", dx, H_tmp, dx, dx, optimize="optimal")
            / dxdx**2
        )

    @staticmethod
    def _get_BFGS(dx, dg, H):
        Hx = H.dot(dx)
        return np.outer(dg, dg) / dg.dot(dx) - np.outer(Hx, Hx) / dx.dot(Hx)

    def update_hessian(self, g, threshold=1e-4, mode="PSB"):
        if self.g_old is None:
            self.g_old = g
            return
        dg = self.get_dg(g).flatten()
        dx = self.dx.flatten()
        H_tmp = dg - np.einsum("ij,j->i", self.hessian, dx)
        if mode == "SR":
            self.hessian = self._get_SR(dx, dg, H_tmp) + self.hessian
        elif mode == "PSB":
            self.hessian = self._get_PSB(dx, dg, H_tmp) + self.hessian
        elif mode == "BFGS":
            self.hessian = self._get_BFGS(dx, dg, self.hessian) + self.hessian
        else:
            raise ValueError(
                "Mode not recognized: {}. Choose from `SR`, `PSB` and `BFGS`".format(
                    mode
                )
            )
        self.g_old = g

    def get_dg(self, g):
        return g - self.g_old


def run_qn(
    job,
    mode="PSB",
    ionic_steps=100,
    ionic_force_tolerance=1.0e-2,
    ionic_energy_tolerance=0,
    starting_h=10,
    diffusion_id=None,
    diffusion_direction=None,
    use_eigenvalues=True,
    symmetrize=True,
    max_displacement=0.1,
    min_displacement=1.0e-8,
):
    """
    Args:
        job (pyiron): pyiron job
        mode (str): Hessian update scheme. `PSB`, `SR` and `BFGS` are currently available.
        ionic_steps (int): Maximum number of steps.
        ionic_force_tolerance (float): Maximum force of an atom tolerated for convergence
        ionic_energy_tolerance (float): Maximum energy difference for convergence
        starting_H (float/ndarray): Starting Hessian value (diagonal value or total Hessian)
        diffusion_id (int/None): Atom id at saddle point. No need to define if the structure
            is close enough to the saddle point. This has to be defined together with
            `diffusion_direction`.
        use_eigenvalues (bool): Whether to use the eigenvalue softening or standard Tikhonov
            regularization to prevent unphysical displacement.
        symmetrize (bool): Whether to symmetrize forces following the box symmetries. DFT
            calculations might fail if set to `False`
        max_displacement (float): Maximum displacement allowed for an atom.
        min_displacement (float): Minimum displacement for a system to rerun

    Returns:
        qn (QuasiNewtonInteractive): Quasi Newton class variable
    """
    qn = QuasiNewtonInteractive(
        structure=job.structure,
        starting_h=starting_h,
        diffusion_id=diffusion_id,
        diffusion_direction=diffusion_direction,
        use_eigenvalues=use_eigenvalues,
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
            warnings.warn("line search alpha is zero")
            break
        job.structure.positions += dx
        job.structure.center_coordinates_in_unit_cell()
        if job.server.run_mode.interactive:
            job.run()
        else:
            job.run(delete_existing_job=True)
    return qn


class QuasiNewton(InteractiveWrapper):

    """
    Structure optimization scheme via Quasi-Newton algorithm.

    Example:

    >>> from pyiron_atomistics import Project
    >>> spx = pr.create.job.Sphinx('spx')
    >>> spx.structure = pr.create.structure.bulk('Al')
    >>> spx.structure[0] = 'Ni'
    >>> spx.interactive_open()
    >>> qn = spx.create_job('QuasiNewton', 'qn')
    >>> qn.run()

    Currently, there are three Hessian update schemes available (cf. `qn.input.mode`):

    - `PSB`: Powell-Symmetric-Broyden
    - `SR`: Symmetric-Rank-One
    - `BFGS`: Broyden–Fletcher–Goldfarb–Shanno

    `PBS` and `SR` do not enforce positive definite Hessian matrix, meaning they can be used to
    obtain an energy barrier state. An energy barrier state calculation is automatically
    performed if the system is within a harmonic distance from the saddle point. If, however,
    the diffusion direction is already known, this information can be inserted in
    `qn.input.diffusion_direction` and the atom id in `qn.input.diffusion_id`.

    There are two types of regularization: Tikhonov regularization and eigenvalue softening
    (`qn.input.use_eivenvalues = True`: eigenvalue softening, `... = False`: Tihkonov
    regularization). In both cases, the regularization value is increased until the largest
    displacement is smaller than `qn.input.max_displacement`.

    Tikhonov regularization:

    `x = (H + L)^{-1} * f`

    where `x` is the displacement field, `H` is the Hessian matrix, `L` is the regularization
    matrix and `f` is the force field. The regularization values get an opposite sign for the
    directions along the negative eigenvalues, to make sure that the regularization indeed
    regularizes when there is a saddle point configuration.

    Eigenvalue softening:

    `x = M * (d / (d^2 + L)) * M^{-1} * f`

    where `M` is the eigenvector matrix, `d` are the eigenvalues and `L` is the regularization.
    """

    def __init__(self, project, job_name):
        super().__init__(project, job_name)

        self.__version__ = None
        self.input = Input()
        self.output = Output(self)
        self._interactive_interface = None
        self.qn = None

    __init__.__doc__ = InteractiveWrapper.__init__.__doc__

    def _run(self):
        self.qn = run_qn(
            job=self.ref_job,
            mode=self.input["mode"],
            ionic_steps=self.input["ionic_steps"],
            ionic_force_tolerance=self.input["ionic_force_tolerance"],
            ionic_energy_tolerance=self.input["ionic_energy_tolerance"],
            starting_h=self.input["starting_h"],
            diffusion_id=self.input["diffusion_id"],
            diffusion_direction=self.input["diffusion_direction"],
            use_eigenvalues=self.input["use_eigenvalues"],
            symmetrize=self.input["symmetrize"],
            max_displacement=self.input["max_displacement"],
        )
        self.collect_output()

    def run_static(self):
        self.status.running = True
        self.ref_job_initialize()
        self._run()
        if self.ref_job.server.run_mode.interactive:
            self.ref_job.interactive_close()
        self.status.collect = True
        self.run()

    run_static.__doc__ = InteractiveWrapper.run_static.__doc__

    def interactive_close(self):
        self.status.collect = True
        if self.ref_job.server.run_mode.interactive:
            self.ref_job.interactive_close()
        self.run()

    interactive_close.__doc__ = InteractiveWrapper.interactive_close.__doc__

    def write_input(self):
        pass

    def collect_output(self):
        self.output._index_lst.append(len(self.ref_job.output.energy_pot))
        if self.qn is not None:
            self.output.hessian = self.qn.hessian
        self.output.to_hdf(hdf=self.project_hdf5)

    collect_output.__doc__ = InteractiveWrapper.collect_output.__doc__


class Input(GenericParameters):
    """
    class to control the generic input for a Sphinx calculation.

    Args:
        input_file_name (str): name of the input file
        table_name (str): name of the GenericParameters table
    """

    def __init__(self, input_file_name=None, table_name="input"):
        super(Input, self).__init__(
            input_file_name=input_file_name,
            table_name=table_name,
            comment_char="//",
            separator_char="=",
            end_value_char=";",
        )

    __init__.__doc__ = GenericParameters.__init__.__doc__

    def load_default(self):
        file_content = (
            "mode = 'PSB'\n"
            "ionic_steps = 100\n"
            "ionic_force_tolerance = 1.0e-2\n"
            "ionic_energy_tolerance = 0\n"
            "starting_h = 10\n"
            "diffusion_id = None\n"
            "use_eigenvalues = True\n"
            "diffusion_direction = None\n"
            "symmetrize = True\n"
            "max_displacement = 0.1\n"
        )
        self.load_string(file_content)


class Output(ReferenceJobOutput):
    def __init__(self, job):
        super().__init__(job=job)
        self._index_lst = []
        self.hessian = None

    @property
    def index_lst(self):
        return np.asarray(self._index_lst)

    def to_hdf(self, hdf, group_name="output"):
        if self.hessian is not None:
            with hdf.open(group_name) as hdf_output:
                hdf_output["hessian"] = self.hessian
