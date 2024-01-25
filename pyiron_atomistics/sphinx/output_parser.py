from functools import cached_property
import numpy as np
import re
import scipy.constants
from pathlib import Path
import h5py
import warnings


BOHR_TO_ANGSTROM = (
    scipy.constants.physical_constants["Bohr radius"][0] / scipy.constants.angstrom
)
HARTREE_TO_EV = scipy.constants.physical_constants["Hartree energy in eV"][0]
HARTREE_OVER_BOHR_TO_EV_OVER_ANGSTROM = HARTREE_TO_EV / BOHR_TO_ANGSTROM


def splitter(arr, counter):
    if len(arr) == 0 or len(counter) == 0:
        return []
    arr_new = []
    spl_loc = list(np.where(np.array(counter) == min(counter))[0])
    spl_loc.append(None)
    for ii, ll in enumerate(spl_loc[:-1]):
        arr_new.append(np.array(arr[ll : spl_loc[ii + 1]]).tolist())
    return arr_new


def collect_energy_dat(file_name="energy.dat", cwd=None):
    """

    Args:
        file_name (str): file name
        cwd (str): directory path

    Returns:
        (dict): results

    """
    path = Path(file_name)
    if cwd is not None:
        path = Path(cwd) / path
    energies = np.loadtxt(str(path), ndmin=2)
    results = {"scf_computation_time": splitter(energies[:, 1], energies[:, 0])}
    results["scf_energy_int"] = splitter(energies[:, 2] * HARTREE_TO_EV, energies[:, 0])

    def en_split(e, counter=energies[:, 0]):
        return splitter(e * HARTREE_TO_EV, counter)

    if len(energies[0]) == 7:
        results["scf_energy_free"] = en_split(energies[:, 3])
        results["scf_energy_zero"] = en_split(energies[:, 4])
        results["scf_energy_band"] = en_split(energies[:, 5])
        results["scf_electronic_entropy"] = en_split(energies[:, 6])
    else:
        results["scf_energy_band"] = en_split(energies[:, 3])
    return results


def collect_residue_dat(file_name="residue.dat", cwd="."):
    """

    Args:
        file_name (str): file name
        cwd (str): directory path

    Returns:
        (dict): results

    """
    if cwd is None:
        cwd = "."
    residue = np.loadtxt(str(Path(cwd) / Path(file_name)), ndmin=2)
    if len(residue) == 0:
        return {}
    return {"scf_residue": splitter(residue[:, 1:].squeeze(), residue[:, 0])}


def _collect_eps_dat(file_name="eps.dat", cwd=None):
    """

    Args:
        file_name:
        cwd:

    Returns:

    """
    path = Path(file_name)
    if cwd is not None:
        path = Path(cwd) / path
    return np.loadtxt(str(path), ndmin=2)[..., 1:]


def collect_eps_dat(file_name=None, cwd=None, spins=True):
    if file_name is not None:
        values = [_collect_eps_dat(file_name=file_name, cwd=cwd)]
    elif spins:
        values = [_collect_eps_dat(file_name=f"eps.{i}.dat", cwd=cwd) for i in [0, 1]]
    else:
        values = [_collect_eps_dat(file_name="eps.dat", cwd=cwd)]
    values = np.stack(values, axis=0)
    return {"bands_eigen_values": values.reshape((-1,) + values.shape)}


def collect_energy_struct(file_name="energy-structOpt.dat", cwd=None):
    """

    Args:
        file_name (str): file name
        cwd (str): directory path

    Returns:
        (dict): results

    """
    path = Path(file_name)
    if cwd is not None:
        path = Path(cwd) / path
    return {
        "energy_free": np.loadtxt(str(path), ndmin=2).reshape(-1, 2)[:, 1]
        * HARTREE_TO_EV
    }


def check_permutation(index_permutation):
    if index_permutation is None:
        return
    indices, counter = np.unique(index_permutation, return_counts=True)
    if np.any(counter != 1):
        raise ValueError("multiple entries in the index_permutation")
    if np.any(np.diff(np.sort(indices)) != 1):
        raise ValueError("missing entries in the index_permutation")


def collect_spins_dat(file_name="spins.dat", cwd=None, index_permutation=None):
    """

    Args:
        file_name (str): file name
        cwd (str): directory path
        index_permutation (numpy.ndarray): Indices for the permutation

    Returns:
        (dict): results

    """
    check_permutation(index_permutation)
    path = Path(file_name)
    if cwd is not None:
        path = Path(cwd) / path
    spins = np.loadtxt(str(path), ndmin=2)
    if index_permutation is not None:
        s = np.array([ss[index_permutation] for ss in spins[:, 1:]])
    else:
        s = spins[:, 1:]
    return {"atom_scf_spins": splitter(s, spins[:, 0])}


def collect_relaxed_hist(file_name="relaxHist.sx", cwd=None, index_permutation=None):
    """

    Args:
        file_name (str): file name
        cwd (str): directory path
        index_permutation (numpy.ndarray): Indices for the permutation

    Returns:
        (dict): results

    # TODO: parse movable, elements, species etc.
    """
    check_permutation(index_permutation)
    path = Path(file_name)
    if cwd is not None:
        path = Path(cwd) / path
    with open(str(path), "r") as f:
        file_content = "".join(f.readlines())
    n_steps = max(len(re.findall("// --- step \d", file_content, re.MULTILINE)), 1)
    f_v = ",".join(3 * [r"\s*([\d.-]+)"])

    def get_value(term, f=file_content, n=n_steps, p=index_permutation):
        value = (
            np.array(re.findall(term, f, re.MULTILINE)).astype(float).reshape(n, -1, 3)
        )
        if p is not None:
            value = np.array([ff[p] for ff in value])
        return value

    cell = re.findall(
        r"cell = \[\[" + r"\],\n\s*\[".join(3 * [f_v]) + r"\]\];",
        file_content,
        re.MULTILINE,
    )
    cell = np.array(cell).astype(float).reshape(n_steps, 3, 3) * BOHR_TO_ANGSTROM
    return {
        "positions": get_value(r"atom {coords = \[" + f_v + r"\];") * BOHR_TO_ANGSTROM,
        "forces": get_value(r"force  = \[" + f_v + r"\]; }"),
        "cell": cell,
    }


class SphinxLogParser:
    def __init__(self, file_name="sphinx.log", cwd=None, index_permutation=None):
        """
        Args:
            file_name (str): file name
            cwd (str): directory path
            index_permutation (numpy.ndarray): Indices for the permutation

        """
        path = Path(file_name)
        if cwd is not None:
            path = Path(cwd) / path
        with open(str(path), "r") as sphinx_log_file:
            self.log_file = sphinx_log_file.read()
        self._scf_not_entered = False
        self._check_enter_scf()
        self._log_main = None
        self._n_atoms = None
        check_permutation(index_permutation)
        self._index_permutation = index_permutation
        self.generic_dict = {
            "volume": self.get_volume,
            "forces": self.get_forces,
            "job_finished": self.job_finished,
        }
        self.dft_dict = {
            "n_valence": self.get_n_valence,
            "bands_k_weights": self.get_bands_k_weights,
            "kpoints_cartesian": self.get_kpoints_cartesian,
            "bands_e_fermi": self.get_fermi,
            "bands_occ": self.get_occupancy,
            "bands_eigen_values": self.get_band_energy,
            "scf_convergence": self.get_convergence,
            "scf_energy_int": self.get_energy_int,
            "scf_energy_free": self.get_energy_free,
            "scf_magnetic_forces": self.get_magnetic_forces,
        }

    @property
    def index_permutation(self):
        return self._index_permutation

    @property
    def spin_enabled(self):
        return len(re.findall("The spin for the label", self.log_file)) > 0

    @property
    def log_main(self):
        if self._log_main is None:
            match = re.search("Enter Main Loop", self.log_file)
            self._log_main = match.end() + 1
        return self.log_file[self._log_main :]

    def job_finished(self):
        if (
            len(re.findall("Program exited normally.", self.log_file, re.MULTILINE))
            == 0
        ):
            warnings.warn("scf loops did not converge")
            return False
        return True

    def _check_enter_scf(self):
        if len(re.findall("Enter Main Loop", self.log_file, re.MULTILINE)) == 0:
            warnings.warn("Log file created but first scf loop not reached")
            self._scf_not_entered = True

    def get_n_valence(self):
        log = self.log_file.split("\n")
        return {
            log[ii - 1].split()[1]: int(ll.split("=")[-1])
            for ii, ll in enumerate(log)
            if ll.startswith("| Z=")
        }

    @property
    def _log_k_points(self):
        start_match = re.search(
            "-ik-     -x-      -y-       -z-    \|  -weight-    -nG-    -label-",
            self.log_file,
        )
        log_part = self.log_file[start_match.end() + 1 :]
        log_part = log_part[: re.search("^\n", log_part, re.MULTILINE).start()]
        return log_part.split("\n")[:-2]

    def get_bands_k_weights(self):
        return np.array([float(kk.split()[6]) for kk in self._log_k_points])

    @property
    def _rec_cell(self):
        log_extract = re.findall("b[1-3]:.*$", self.log_file, re.MULTILINE)
        return (
            np.array([ll.split()[1:4] for ll in log_extract]).astype(float)
            / BOHR_TO_ANGSTROM
        )[:3]

    def get_kpoints_cartesian(self):
        return np.einsum("ni,ij->nj", self.k_points, self._rec_cell)

    @property
    def k_points(self):
        return np.array(
            [[float(kk.split()[i]) for i in range(2, 5)] for kk in self._log_k_points]
        )

    def get_volume(self):
        volume = re.findall("Omega:.*$", self.log_file, re.MULTILINE)
        if len(volume) > 0:
            volume = float(volume[0].split()[1])
            volume *= BOHR_TO_ANGSTROM**3
        else:
            volume = 0
        return np.array(self.n_steps * [volume])

    @property
    def counter(self):
        return [
            int(re.sub("[^0-9]", "", line.split("=")[0]))
            for line in re.findall("F\(.*$", self.log_main, re.MULTILINE)
        ]

    def _get_energy(self, pattern):
        c, F = np.array(re.findall(pattern, self.log_main, re.MULTILINE)).T
        return splitter(F.astype(float) * HARTREE_TO_EV, c.astype(int))

    def get_energy_free(self):
        return self._get_energy(pattern=r"F\((\d+)\)=(-?\d+\.\d+)")

    def get_energy_int(self):
        return self._get_energy(pattern=r"eTot\((\d+)\)=(-?\d+\.\d+)")

    @property
    def n_atoms(self):
        if self._n_atoms is None:
            self._n_atoms = len(
                np.unique(re.findall("^Species.*\{", self.log_main, re.MULTILINE))
            )
        return self._n_atoms

    def get_forces(self):
        """
        Returns:
            (numpy.ndarray): Forces of the shape (n_steps, n_atoms, 3)
        """
        str_fl = "([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)"
        pattern = r"Atom: (\d+)\t{" + ",".join(3 * [str_fl]) + r"\}"
        arr = np.array(re.findall(pattern, self.log_file))
        if len(arr) == 0:
            return []
        forces = arr[:, 1:].astype(float).reshape(-1, self.n_atoms, 3)
        forces *= HARTREE_OVER_BOHR_TO_EV_OVER_ANGSTROM
        if self.index_permutation is not None:
            for ii, ff in enumerate(forces):
                forces[ii] = ff[self.index_permutation]
        return forces

    def get_magnetic_forces(self):
        """
        Returns:
            (numpy.ndarray): Magnetic forces of the shape (n_steps, n_atoms)
        """
        magnetic_forces = [
            HARTREE_TO_EV * float(line.split()[-1])
            for line in re.findall("^nu\(.*$", self.log_main, re.MULTILINE)
        ]
        if len(magnetic_forces) != 0:
            magnetic_forces = np.array(magnetic_forces).reshape(-1, self.n_atoms)
            if self.index_permutation is not None:
                for ii, mm in enumerate(magnetic_forces):
                    magnetic_forces[ii] = mm[self.index_permutation]
        return splitter(magnetic_forces, self.counter)

    @property
    def n_steps(self):
        return len(re.findall("\| SCF calculation", self.log_file, re.MULTILINE))

    def _parse_band(self, term):
        content = re.findall(term, self.log_main, re.MULTILINE)
        if len(content) == 0:
            return []
        arr = np.loadtxt(content, ndmin=2)
        shape = (-1, len(self.k_points), arr.shape[-1])
        if self.spin_enabled:
            shape = (-1, 2, len(self.k_points), shape[-1])
        return arr.reshape(shape)

    def get_band_energy(self):
        return self._parse_band("final eig \[eV\]:(.*)$")

    def get_occupancy(self):
        return self._parse_band("final focc:(.*)$")

    def get_convergence(self):
        conv_dict = {
            "WARNING: Maximum number of steps exceeded": False,
            "Convergence reached.": True,
        }
        key = "|".join(list(conv_dict.keys())).replace(".", "\.")
        items = re.findall(key, self.log_main, re.MULTILINE)
        convergence = [conv_dict[k] for k in items]
        diff = self.n_steps - len(convergence)
        for _ in range(diff):
            convergence.append(False)
        return convergence

    def get_fermi(self):
        pattern = r"Fermi energy:\s+(\d+\.\d+)\s+eV"
        return np.array(re.findall(pattern, self.log_main)).astype(float)

    @property
    def results(self):
        if self._scf_not_entered:
            return {}
        results = {"generic": {}, "dft": {}}
        for key, func in self.generic_dict.items():
            value = func()
            if key == "job_finished" or len(value) > 0:
                results["generic"][key] = value
        for key, func in self.dft_dict.items():
            value = func()
            if len(value) > 0:
                results["dft"][key] = value
        return results


class SphinxWavesReader:
    """Class to read SPHInX waves.sxb files (HDF5 format)

    Initialize with waves.sxb filename, or use load ()

    """

    def __init__(self, file_name="waves.sxb", cwd="."):
        """
        Args:
            file_name (str): file name
            cwd (str): directory path
        """
        if Path(file_name).is_absolute():
            self.wfile = h5py.File(Path(file_name))
        else:
            path = Path(cwd) / Path(file_name)
            self.wfile = h5py.File(path)

    @property
    def _n_gk(self):
        return self.wfile["nGk"][:]

    @cached_property
    def mesh(self):
        return self.wfile["meshDim"][:]

    @property
    def Nx(self):
        return self.mesh[0]

    @property
    def Ny(self):
        return self.mesh[1]

    @property
    def Nz(self):
        return self.mesh[2]

    @cached_property
    def n_states(self):
        return self.wfile["nPerK"][0]

    @cached_property
    def n_spin(self):
        return self.wfile["nSpin"].shape[0]

    @cached_property
    def k_weights(self):
        return self.wfile["kWeights"][:]

    @cached_property
    def k_vec(self):
        return self.wfile["kVec"][:]

    @cached_property
    def eps(self):
        """All eigenvalues (in Hartree) as (nk,n_states) block"""
        return (self.wfile["eps"][:].reshape(-1, self.n_spin, self.n_states)).T

    @cached_property
    def _fft_idx(self):
        fft_idx = []
        off = 0
        for ngk in self._n_gk:
            fft_idx.append(self.wfile["fftIdx"][off : off + ngk])
            off += ngk
        return fft_idx

    def get_psi_rec(self, i, ispin, ik, compact=False):
        """
        Loads a single wavefunction on full FFT mesh of shape mesh.

        params: i: state index (int)
               ispin: spin index (int)
               ik: k index (int)
               compact: (bool)
        returns:
            res: complex valued wavefunction indexed by (i,ispin,ik) loaded on to the FFT mesh.
            compact_wave: compact wavefunctions, without loading on FFT mesh.
        """
        # translate indices to pythonic style.
        i = np.arange(self.n_states)[i]
        ik = np.arange(self.nk)[ik]
        ispin = np.arange(self.n_spin)[ispin]

        off = self._n_gk[ik] * (i + ispin * self.n_states)
        psire = self.wfile[f"psi-{ik+1}.re"][off : off + self._n_gk[ik]]
        psiim = self.wfile[f"psi-{ik+1}.im"][off : off + self._n_gk[ik]]
        compact_wave = psire + 1j * psiim
        if compact:
            return compact_wave
        res = np.zeros(shape=self.mesh, dtype=np.complex128)
        res.flat[self._fft_idx[ik]] = compact_wave
        return res

    @property
    def nk(self):
        """Number of k-points"""
        return self.k_weights.shape[0]
