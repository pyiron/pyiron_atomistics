import numpy as np
from scipy.spatial.transform import Rotation


class Strain:

    """
    Calculate local strain of each atom following the Lagrangian strain tensor:

    >>> strain = (F.T*F - 1)/2

    where `F` is the atomic deformation gradient.

    Example:

    >>> from pyiron_atomistics import Project
    >>> pr = Project('.')
    >>> bulk = pr.create.structure.bulk('Fe', cubic=True)
    >>> structure = bulk.apply_strain(np.random.random((3,3))*0.1, return_box=True)
    >>> Strain(structure, bulk).strain

    """

    def __init__(
        self, structure, ref_structure, num_neighbors=None, only_bulk_type=False
    ):
        """

        Args:
            structure (pyiron_atomistics.atomistics.structure.Atoms): Structure to calculate the
                strain values.
            ref_structure (pyiron_atomistics.atomistics.structure.Atoms): Reference bulk structure
                (against which the strain is calculated)
            num_neighbors (int): Number of neighbors to take into account to calculate the local
                frame. If not specified, it is estimated based on cna analysis (only available if
                the bulk structure is bcc, fcc or hcp).
            only_bulk_type (bool): Whether to calculate the strain of all atoms or only for those
                which cna considers has the same crystal structure as the bulk. Those which have
                a different crystal structure will get 0 strain.
        """
        self.structure = structure
        self.ref_structure = ref_structure
        self._num_neighbors = num_neighbors
        self.only_bulk_type = only_bulk_type
        self._crystal_phase = None
        self._ref_coord = None
        self._coords = None
        self._rotations = None

    @property
    def num_neighbors(self):
        """Number of neighbors to consider the local frame. Should be the coordination number."""
        if self._num_neighbors is None:
            self._num_neighbors = self._get_number_of_neighbors(self.crystal_phase)
        return self._num_neighbors

    @property
    def crystal_phase(self):
        """Majority crystal phase calculated via common neighbor analysis."""
        if self._crystal_phase is None:
            self._crystal_phase = self._get_majority_phase(self.ref_structure)
        return self._crystal_phase

    @property
    def _nullify_non_bulk(self):
        return np.array(
            self.structure.analyse.pyscal_cna_adaptive(mode="str") != self.crystal_phase
        )

    def _get_perpendicular_unit_vectors(self, vec, vec_axis=None):
        if vec_axis is not None:
            vec_axis = self._get_safe_unit_vectors(vec_axis)
            vec = np.array(
                vec - np.einsum("...i,...i,...j->...j", vec, vec_axis, vec_axis)
            )
        return self._get_safe_unit_vectors(vec)

    @staticmethod
    def _get_safe_unit_vectors(vectors, minimum_value=1.0e-8):
        v = np.linalg.norm(vectors, axis=-1)
        v += (v < minimum_value) * minimum_value
        return np.einsum("...i,...->...i", vectors, 1 / v)

    def _get_angle(self, v, w):
        v = self._get_safe_unit_vectors(v)
        w = self._get_safe_unit_vectors(w)
        prod = np.sum(v * w, axis=-1)
        # Safety measure - in principle not required.
        if hasattr(prod, "__len__"):
            prod[np.absolute(prod) > 1] = np.sign(prod)[np.absolute(prod) > 1]
        return np.arccos(prod)

    def _get_rotation_from_vectors(self, vec_before, vec_after, vec_axis=None):
        v = self._get_perpendicular_unit_vectors(vec_before, vec_axis)
        w = self._get_perpendicular_unit_vectors(vec_after, vec_axis)
        if vec_axis is None:
            vec_axis = np.cross(v, w)
        vec_axis = self._get_safe_unit_vectors(vec_axis)
        sign = np.sign(np.sum(np.cross(v, w) * vec_axis, axis=-1))
        vec_axis = np.einsum(
            "...i,...->...i", vec_axis, np.tan(sign * self._get_angle(v, w) / 4)
        )
        return Rotation.from_mrp(vec_axis).as_matrix()

    @property
    def rotations(self):
        """Rotation for each atom to find the correct pairs of coordinates."""
        if self._rotations is None:
            v = self.coords.copy()[:, 0, :]
            w_first = self.ref_coord[
                np.linalg.norm(
                    self.ref_coord[None, :, :] - v[:, None, :], axis=-1
                ).argmin(axis=1)
            ].copy()
            first_rot = self._get_rotation_from_vectors(v, w_first)
            all_vecs = np.einsum("nij,nkj->nki", first_rot, self.coords)
            highest_angle_indices = np.absolute(
                np.sum(all_vecs * all_vecs[:, :1], axis=-1)
            ).argmin(axis=-1)
            v = all_vecs[np.arange(len(self.coords)), highest_angle_indices, :]
            dv = self.ref_coord[None, :, :] - v[:, None, :]
            dist = np.linalg.norm(dv, axis=-1) + np.absolute(
                np.sum(dv * all_vecs[:, :1], axis=-1)
            )
            w_second = self.ref_coord[dist.argmin(axis=1)].copy()
            second_rot = self._get_rotation_from_vectors(v, w_second, all_vecs[:, 0])
            self._rotations = np.einsum("nij,njk->nik", second_rot, first_rot)
        return self._rotations

    @staticmethod
    def _get_best_match_indices(coords, ref_coord):
        distances = np.linalg.norm(
            coords[:, :, None, :] - ref_coord[None, None, :, :], axis=-1
        )
        return np.argmin(distances, axis=-1)

    @staticmethod
    def _get_majority_phase(structure):
        cna = structure.analyse.pyscal_cna_adaptive()
        return np.asarray([k for k in cna.keys()])[np.argmax([v for v in cna.values()])]

    @staticmethod
    def _get_number_of_neighbors(crystal_phase):
        if crystal_phase == "bcc":
            return 8
        elif crystal_phase == "fcc" or crystal_phase == "hcp":
            return 12
        else:
            raise ValueError(f'Crystal structure "{crystal_phase}" not recognized')

    @property
    def ref_coord(self):
        """Reference local coordinates."""
        if self._ref_coord is None:
            self._ref_coord = self.ref_structure.get_neighbors(
                num_neighbors=self.num_neighbors
            ).vecs[0]
        return self._ref_coord

    @property
    def coords(self):
        """Local coordinates of each atom."""
        if self._coords is None:
            self._coords = self.structure.get_neighbors(
                num_neighbors=self.num_neighbors
            ).vecs
        return self._coords

    @property
    def _indices(self):
        all_vecs = np.einsum("nij,nkj->nki", self.rotations, self.coords)
        return self._get_best_match_indices(all_vecs, self.ref_coord)

    @property
    def strain(self):
        """Strain value of each atom"""
        Dinverse = np.einsum("ij,ik->jk", self.ref_coord, self.ref_coord)
        D = np.linalg.inv(Dinverse)
        J = np.einsum(
            "ij,nml,nlj,nmk->nik",
            D,
            self.ref_coord[self._indices],
            self.rotations,
            self.coords,
        )
        if self.only_bulk_type:
            J[self._nullify_non_bulk] = np.eye(3)
        return 0.5 * (np.einsum("nij,nkj->nik", J, J) - np.eye(3))
