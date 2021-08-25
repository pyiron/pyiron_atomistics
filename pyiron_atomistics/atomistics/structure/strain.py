import numpy as np
from scipy.spatial.transform import Rotation

class Strain:
    def __init__(self, structure, ref_structure, num_neighbors=None, only_bulk_type=False):
        self.structure = structure
        self.ref_structure = ref_structure
        self._num_neighbors = num_neighbors
        self.only_bulk_type = only_bulk_type
        self._crystal_phase = None
        self._ref_frame = None
        self._frames = None
        self._rotations = None

    @property
    def num_neighbors(self):
        if self._num_neighbors is None:
            self._num_neighbors = self._get_number_of_neighbors(self.crystal_phase)
        return self._num_neighbors

    @property
    def crystal_phase(self):
        if self._crystal_phase is None:
            self._crystal_phase = self._get_majority_phase(self.ref_structure)
        return self._crystal_phase

    @property
    def nullify_non_bulk(self):
        return np.array(
            self.structure.analyse.pyscal_cna_adaptive(mode='str')!=self.crystal_phase
        )

    def _get_perpendicular_unit_vectors(self, vec, vec_axis=None):
        if vec_axis is not None:
            vec_axis = self._get_safe_unit_vectors(vec_axis)
            vec -= np.einsum('...i,...i,...j->...j', vec, vec_axis, vec_axis)
        return self._get_safe_unit_vectors(vec)

    @staticmethod
    def _get_safe_unit_vectors(vectors, minimum_value=1.0e-8):
        v = np.linalg.norm(vectors, axis=-1)
        if len(v.shape) > 1:
            v[v<minimum_value] = minimum_value
        return np.einsum('...i,...->...i', vectors, 1/v)

    def _get_angle(self, v, w):
        v = self._get_safe_unit_vectors(v)
        w = self._get_safe_unit_vectors(w)
        prod = np.sum(v*w, axis=-1)
        prod[np.absolute(prod)>1] = np.sign(prod)[np.absolute(prod)>1]
        return np.arccos(prod)

    def _get_rotation_from_vectors(self, vec_before, vec_after, vec_axis=None):
        v = self._get_perpendicular_unit_vectors(vec_before, vec_axis)
        w = self._get_perpendicular_unit_vectors(vec_after, vec_axis)
        if vec_axis is None:
            vec_axis = self._get_safe_unit_vectors(np.cross(v, w))
        sign = np.sign(np.sum(np.cross(v, w)*vec_axis, axis=-1))
        vec_axis *= (sign*self._get_angle(v, w)/4)[:,None]
        return Rotation.from_mrp(vec_axis).as_matrix()

    @property
    def rotations(self):
        if self._rotations is None:
            v = self.frames.copy()[:,0,:]
            w_first = self.ref_frame[
                np.linalg.norm(self.ref_frame[None,:,:]-v[:,None,:], axis=-1).argmin(axis=1)
            ].copy()
            first_rot = self._get_rotation_from_vectors(v, w_first)
            all_vecs = np.einsum('nij,nkj->nki', first_rot, self.frames)
            highest_angle_indices = np.absolute(
                np.sum(all_vecs*all_vecs[:,:1], axis=-1)
            ).argmin(axis=-1)
            v = all_vecs[np.arange(len(self.frames)),highest_angle_indices,:]
            dv = self.ref_frame[None,:,:]-v[:,None,:]
            dist = np.linalg.norm(dv, axis=-1)
            dist += np.absolute(np.sum(dv*all_vecs[:,:1], axis=-1))
            w_second = self.ref_frame[dist.argmin(axis=1)].copy()
            second_rot = self._get_rotation_from_vectors(v, w_second, all_vecs[:,0])
            self._rotations = np.einsum('nij,njk->nik', second_rot, first_rot)
        return self._rotations

    @staticmethod
    def _get_best_match_indices(frames, ref_frame):
        distances = np.linalg.norm(frames[:,:,None,:]-ref_frame[None,None,:,:], axis=-1)
        return np.argmin(distances, axis=-1)

    @staticmethod
    def _get_majority_phase(structure):
        cna = structure.analyse.pyscal_cna_adaptive()
        return np.asarray([k for k in cna.keys()])[np.argmax([v for v in cna.values()])]

    @staticmethod
    def _get_number_of_neighbors(crystal_phase):
        if crystal_phase=='bcc':
            return 8
        elif crystal_phase=='fcc' or crystal_phase=='hcp':
            return 12
        else:
            raise ValueError('Crystal structure not recognized')

    @property
    def ref_frame(self):
        if self._ref_frame is None:
            self._ref_frame = self.ref_structure.get_neighbors(
                num_neighbors=self.num_neighbors
            ).vecs[0]
        return self._ref_frame

    @property
    def frames(self):
        if self._frames is None:
            self._frames = self.structure.get_neighbors(num_neighbors=self.num_neighbors).vecs
        return self._frames

    @property
    def indices(self):
        all_vecs = np.einsum('nij,nkj->nki', self.rotations, self.frames)
        return self._get_best_match_indices(all_vecs, self.ref_frame)

    @property
    def strain(self):
        D = np.einsum('ij,ik->jk', self.ref_frame, self.ref_frame)
        D = np.linalg.inv(D)
        J = np.einsum('nij,nik->njk', self.ref_frame[self.indices], self.frames)
        J = np.einsum('ij,njk->nik', D, J)
        if self.only_bulk_type:
            J[self.nullify_non_bulk] = np.eye(3)
        return 0.5*(np.einsum('nij,nkj->nik', J, J)-np.eye(3))


