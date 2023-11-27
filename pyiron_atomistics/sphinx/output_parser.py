import numpy as np
import re

def splitter(arr, counter):
    if len(arr) == 0 or len(counter) == 0:
        return []
    arr_new = []
    spl_loc = list(np.where(np.array(counter) == min(counter))[0])
    spl_loc.append(None)
    for ii, ll in enumerate(spl_loc[:-1]):
        arr_new.append(np.array(arr[ll : spl_loc[ii + 1]]).tolist())
    return arr_new


class SphinxLogParser:
    def __init__(self, log_file):
        """
        Args:
            log_file (str): content of the log files

        Log file should contain the plain text of the log. You can get it for
        example via:

        >>> with open("sphinx.log", "r") as f:
        >>>     log_file = f.read()

        """
        self.log_file = log_file
        self._check_enter_scf()
        self._log_main = None
        self._counter = None
        self._n_atoms = None
        self._n_steps = None

    @property
    def spin_enabled(self):
        return len(re.findall("The spin for the label", self.log_file)) > 0

    @property
    def log_main(self):
        if self._log_main is None:
            match = re.search("Enter Main Loop", self.log_file)
            self._log_main = match.end() + 1
        return self.log_file[self._log_main :]

    @property
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
            raise AssertionError("Log file created but first scf loop not reached")

    def get_n_valence(self):
        log = self.log_file.split("\n")
        return {
            log[ii - 1].split()[1]: int(ll.split("=")[-1])
            for ii, ll in enumerate(log)
            if ll.startswith("| Z=")
        }

    @property
    def log_k_points(self):
        start_match = re.search(
            "-ik-     -x-      -y-       -z-    \|  -weight-    -nG-    -label-",
            self.log_file,
        )
        log_part = self.log_file[start_match.end() + 1 :]
        log_part = log_part[: re.search("^\n", log_part, re.MULTILINE).start()]
        return log_part.split("\n")[:-2]

    def get_bands_k_weights(self):
        return np.array([float(kk.split()[6]) for kk in self.log_k_points])

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
            [[float(kk.split()[i]) for i in range(2, 5)] for kk in self.log_k_points]
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
        if self._counter is None:
            self._counter = [
                int(re.sub("[^0-9]", "", line.split("=")[0]))
                for line in re.findall("F\(.*$", self.log_main, re.MULTILINE)
            ]
        return self._counter

    def get_energy_free(self):
        return splitter(
            [
                float(line.split("=")[1]) * HARTREE_TO_EV
                for line in re.findall("F\(.*$", self.log_main, re.MULTILINE)
            ],
            self.counter,
        )

    def get_energy_int(self):
        return splitter(
            [
                float(line.replace("=", " ").replace(",", " ").split()[1])
                * HARTREE_TO_EV
                for line in re.findall("^eTot\([0-9].*$", self.log_main, re.MULTILINE)
            ],
            self.counter,
        )

    @property
    def n_atoms(self):
        if self._n_atoms is None:
            self._n_atoms = len(
                np.unique(re.findall("^Species.*\{", self.log_main, re.MULTILINE))
            )
        return self._n_atoms

    def get_forces(self, spx_to_pyi=None):
        forces = [
            float(re.split("{|}", line)[1].split(",")[i])
            * HARTREE_OVER_BOHR_TO_EV_OVER_ANGSTROM
            for line in re.findall("^Species.*$", self.log_main, re.MULTILINE)
            for i in range(3)
        ]
        if len(forces) != 0:
            forces = np.array(forces).reshape(-1, self.n_atoms, 3)
            if spx_to_pyi is not None:
                for ii, ff in enumerate(forces):
                    forces[ii] = ff[spx_to_pyi]
        return forces

    def get_magnetic_forces(self, spx_to_pyi=None):
        magnetic_forces = [
            HARTREE_TO_EV * float(line.split()[-1])
            for line in re.findall("^nu\(.*$", self.log_main, re.MULTILINE)
        ]
        if len(magnetic_forces) != 0:
            magnetic_forces = np.array(magnetic_forces).reshape(-1, self.n_atoms)
            if spx_to_pyi is not None:
                for ii, mm in enumerate(magnetic_forces):
                    magnetic_forces[ii] = mm[spx_to_pyi]
        return splitter(magnetic_forces, self.counter)

    @property
    def n_steps(self):
        if self._n_steps is None:
            self._n_steps = len(
                re.findall("\| SCF calculation", self.log_file, re.MULTILINE)
            )
        return self._n_steps

    def _parse_band(self, term):
        arr = np.loadtxt(re.findall(term, self.log_main, re.MULTILINE))
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
        return np.array(
            [
                float(line.split()[2])
                for line in re.findall("Fermi energy:.*$", self.log_main, re.MULTILINE)
            ]
        )
