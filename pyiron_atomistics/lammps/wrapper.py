from ctypes import c_double, c_int
import importlib
import numpy as np
import os
from scipy import constants
import warnings

from pyiron_atomistics.lammps.structure import UnfoldingPrism
from pyiron_atomistics.lammps.output import _check_ortho_prism

try:  # mpi4py is only supported on Linux and Mac Os X
    from pylammpsmpi import LammpsLibrary
except ImportError:
    pass


class PyironLammpsLibrary(object):
    def __init__(
        self,
        working_directory,
        cores=1,
        comm=None,
        logger=None,
        log_file=None,
        library=None,
    ):
        self._logger = logger
        self._prism = None
        self._structure = None
        self._cores = cores
        if library is not None:
            self._interactive_library = library
        elif self._cores == 1:
            lammps = getattr(importlib.import_module("lammps"), "lammps")
            if log_file is None:
                log_file = os.path.join(working_directory, "log.lammps")
            self._interactive_library = lammps(
                cmdargs=["-screen", "none", "-log", log_file],
                comm=comm,
            )
        else:
            self._interactive_library = LammpsLibrary(
                cores=self._cores, working_directory=working_directory
            )

    def interactive_lib_command(self, command):
        if self._logger is not None:
            self._logger.debug("Lammps library: " + command)
        self._interactive_library.command(command)

    def interactive_positions_getter(self):
        positions = np.reshape(
            np.array(self._interactive_library.gather_atoms("x", 1, 3)),
            (len(self._structure), 3),
        )
        if _check_ortho_prism(prism=self._prism):
            positions = np.matmul(positions, self._prism.R.T)
        return positions

    def interactive_positions_setter(self, positions):
        if _check_ortho_prism(prism=self._prism):
            positions = np.array(positions).reshape(-1, 3)
            positions = np.matmul(positions, self._prism.R)
        positions = np.array(positions).flatten()
        if self._cores == 1:
            self._interactive_library.scatter_atoms(
                "x", 1, 3, (len(positions) * c_double)(*positions)
            )
        else:
            self._interactive_library.scatter_atoms("x", positions)
        self.interactive_lib_command(command="change_box all remap")

    def interactive_cells_getter(self):
        cc = np.array(
            [
                [self._interactive_library.get_thermo("lx"), 0, 0],
                [
                    self._interactive_library.get_thermo("xy"),
                    self._interactive_library.get_thermo("ly"),
                    0,
                ],
                [
                    self._interactive_library.get_thermo("xz"),
                    self._interactive_library.get_thermo("yz"),
                    self._interactive_library.get_thermo("lz"),
                ],
            ]
        )
        return self._prism.unfold_cell(cc)

    def interactive_cells_setter(self, cell):
        self._prism = UnfoldingPrism(cell)
        lx, ly, lz, xy, xz, yz = self._prism.get_lammps_prism()
        if _check_ortho_prism(prism=self._prism):
            warnings.warn(
                "Warning: setting upper trangular matrix might slow down the calculation"
            )

        is_skewed = cell_is_skewed(cell=cell, tolerance=1.0e-8)
        was_skewed = cell_is_skewed(cell=self._structure.cell, tolerance=1.0e-8)

        if is_skewed:
            if not was_skewed:
                self.interactive_lib_command(command="change_box all triclinic")
            self.interactive_lib_command(
                command="change_box all x final 0 %f y final 0 %f z final 0 %f  xy final %f xz final %f yz final %f remap units box"
                % (lx, ly, lz, xy, xz, yz),
            )
        elif was_skewed:
            self.interactive_lib_command(
                command="change_box all x final 0 %f y final 0 %f z final 0 %f xy final %f xz final %f yz final %f remap units box"
                % (lx, ly, lz, 0.0, 0.0, 0.0),
            )
            self.interactive_lib_command(command="change_box all ortho")
        else:
            self.interactive_lib_command(
                command="change_box all x final 0 %f y final 0 %f z final 0 %f remap units box"
                % (lx, ly, lz),
            )

    def interactive_volume_getter(self):
        return self._interactive_library.get_thermo("vol")

    def interactive_forces_getter(self):
        ff = np.reshape(
            np.array(self._interactive_library.gather_atoms("f", 1, 3)),
            (len(self._structure), 3),
        )
        if _check_ortho_prism(prism=self._prism):
            ff = np.matmul(ff, self._prism.R.T)
        return ff

    def interactive_structure_setter(
        self,
        structure,
        units,
        dimension,
        boundary,
        atom_style,
        el_eam_lst,
        calc_md=True,
    ):
        if self._structure is not None:
            old_symbols = self._structure.get_species_symbols()
            new_symbols = structure.get_species_symbols()
            if any(old_symbols != new_symbols):
                raise ValueError(
                    f"structure has different chemical symbols than old one: {new_symbols} != {old_symbols}"
                )
        self.interactive_lib_command(command="clear")
        control_dict = self._set_selective_dynamics(
            structure=structure, calc_md=calc_md
        )
        self.interactive_lib_command(command="units " + units)
        self.interactive_lib_command(command="dimension " + str(dimension))
        self.interactive_lib_command(command="boundary " + boundary)
        self.interactive_lib_command(command="atom_style " + atom_style)

        self.interactive_lib_command(command="atom_modify map array")
        self._prism = UnfoldingPrism(structure.cell)
        if _check_ortho_prism(prism=self._prism):
            warnings.warn(
                "Warning: setting upper trangular matrix might slow down the calculation"
            )
        xhi, yhi, zhi, xy, xz, yz = self._prism.get_lammps_prism()
        if self._prism.is_skewed():
            self.interactive_lib_command(
                command="region 1 prism"
                + " 0.0 "
                + str(xhi)
                + " 0.0 "
                + str(yhi)
                + " 0.0 "
                + str(zhi)
                + " "
                + str(xy)
                + " "
                + str(xz)
                + " "
                + str(yz)
                + " units box",
            )
        else:
            self.interactive_lib_command(
                command="region 1 block"
                + " 0.0 "
                + str(xhi)
                + " 0.0 "
                + str(yhi)
                + " 0.0 "
                + str(zhi)
                + " units box",
            )
        el_struct_lst = structure.get_species_symbols()
        el_obj_lst = structure.get_species_objects()
        if atom_style == "full":
            self.interactive_lib_command(
                command="create_box "
                + str(len(el_eam_lst))
                + " 1 "
                + "bond/types 1 "
                + "angle/types 1 "
                + "extra/bond/per/atom 2 "
                + "extra/angle/per/atom 2 ",
            )
        else:
            self.interactive_lib_command(
                command="create_box " + str(len(el_eam_lst)) + " 1"
            )
        el_dict = {}
        for id_eam, el_eam in enumerate(el_eam_lst):
            if el_eam in el_struct_lst:
                id_el = list(el_struct_lst).index(el_eam)
                el = el_obj_lst[id_el]
                el_dict[el] = id_eam + 1
                self.interactive_lib_command(
                    command="mass {0:3d} {1:f}".format(id_eam + 1, el.AtomicMass),
                )
            else:
                self.interactive_lib_command(
                    command="mass {0:3d} {1:f}".format(id_eam + 1, 1.00),
                )
        positions = structure.positions.flatten()
        if _check_ortho_prism(prism=self._prism):
            positions = np.array(positions).reshape(-1, 3)
            positions = np.matmul(positions, self._prism.R)
        positions = positions.flatten()
        try:
            elem_all = np.array(
                [el_dict[el] for el in structure.get_chemical_elements()]
            )
        except KeyError:
            missing = set(structure.get_chemical_elements()).difference(el_dict.keys())
            missing = ", ".join([el.Abbreviation for el in missing])
            raise ValueError(
                f"Structure contains elements [{missing}], that are not present in the potential!"
            )
        if self._cores == 1:
            self._interactive_library.create_atoms(
                n=len(structure),
                id=None,
                type=(len(elem_all) * c_int)(*elem_all),
                x=(len(positions) * c_double)(*positions),
                v=None,
                image=None,
                shrinkexceed=False,
            )
        else:
            self._interactive_library.create_atoms(
                n=len(structure),
                id=None,
                type=elem_all,
                x=positions,
                v=None,
                image=None,
                shrinkexceed=False,
            )
        self.interactive_lib_command(command="change_box all remap")
        for key, value in control_dict.items():
            self.interactive_lib_command(command=key + " " + value)
        self._structure = structure

    @staticmethod
    def _set_selective_dynamics(structure, calc_md):
        control_dict = {}
        if "selective_dynamics" in structure._tag_list.keys():
            if structure.selective_dynamics._default is None:
                structure.selective_dynamics._default = [True, True, True]
            sel_dyn = np.logical_not(structure.selective_dynamics.list())
            # Enter loop only if constraints present
            if len(np.argwhere(np.any(sel_dyn, axis=1)).flatten()) != 0:
                all_indices = np.arange(len(structure), dtype=int)
                constraint_xyz = np.argwhere(np.all(sel_dyn, axis=1)).flatten()
                not_constrained_xyz = np.setdiff1d(all_indices, constraint_xyz)
                # LAMMPS starts counting from 1
                constraint_xyz += 1
                ind_x = np.argwhere(sel_dyn[not_constrained_xyz, 0]).flatten()
                ind_y = np.argwhere(sel_dyn[not_constrained_xyz, 1]).flatten()
                ind_z = np.argwhere(sel_dyn[not_constrained_xyz, 2]).flatten()
                constraint_xy = not_constrained_xyz[np.intersect1d(ind_x, ind_y)] + 1
                constraint_yz = not_constrained_xyz[np.intersect1d(ind_y, ind_z)] + 1
                constraint_zx = not_constrained_xyz[np.intersect1d(ind_z, ind_x)] + 1
                constraint_x = (
                    not_constrained_xyz[np.setdiff1d(np.setdiff1d(ind_x, ind_y), ind_z)]
                    + 1
                )
                constraint_y = (
                    not_constrained_xyz[np.setdiff1d(np.setdiff1d(ind_y, ind_z), ind_x)]
                    + 1
                )
                constraint_z = (
                    not_constrained_xyz[np.setdiff1d(np.setdiff1d(ind_z, ind_x), ind_y)]
                    + 1
                )
                control_dict = {}
                if len(constraint_xyz) > 0:
                    control_dict["group constraintxyz"] = "id " + " ".join(
                        [str(ind) for ind in constraint_xyz]
                    )
                    control_dict[
                        "fix constraintxyz"
                    ] = "constraintxyz setforce 0.0 0.0 0.0"
                    if calc_md:
                        control_dict["velocity constraintxyz"] = "set 0.0 0.0 0.0"
                if len(constraint_xy) > 0:
                    control_dict["group constraintxy"] = "id " + " ".join(
                        [str(ind) for ind in constraint_xy]
                    )
                    control_dict[
                        "fix constraintxy"
                    ] = "constraintxy setforce 0.0 0.0 NULL"
                    if calc_md:
                        control_dict["velocity constraintxy"] = "set 0.0 0.0 NULL"
                if len(constraint_yz) > 0:
                    control_dict["group constraintyz"] = "id " + " ".join(
                        [str(ind) for ind in constraint_yz]
                    )
                    control_dict[
                        "fix constraintyz"
                    ] = "constraintyz setforce NULL 0.0 0.0"
                    if calc_md:
                        control_dict["velocity constraintyz"] = "set NULL 0.0 0.0"
                if len(constraint_zx) > 0:
                    control_dict["group constraintxz"] = "id " + " ".join(
                        [str(ind) for ind in constraint_zx]
                    )
                    control_dict[
                        "fix constraintxz"
                    ] = "constraintxz setforce 0.0 NULL 0.0"
                    if calc_md:
                        control_dict["velocity constraintxz"] = "set 0.0 NULL 0.0"
                if len(constraint_x) > 0:
                    control_dict["group constraintx"] = "id " + " ".join(
                        [str(ind) for ind in constraint_x]
                    )
                    control_dict[
                        "fix constraintx"
                    ] = "constraintx setforce 0.0 NULL NULL"
                    if calc_md:
                        control_dict["velocity constraintx"] = "set 0.0 NULL NULL"
                if len(constraint_y) > 0:
                    control_dict["group constrainty"] = "id " + " ".join(
                        [str(ind) for ind in constraint_y]
                    )
                    control_dict[
                        "fix constrainty"
                    ] = "constrainty setforce NULL 0.0 NULL"
                    if calc_md:
                        control_dict["velocity constrainty"] = "set NULL 0.0 NULL"
                if len(constraint_z) > 0:
                    control_dict["group constraintz"] = "id " + " ".join(
                        [str(ind) for ind in constraint_z]
                    )
                    control_dict[
                        "fix constraintz"
                    ] = "constraintz setforce NULL NULL 0.0"
                    if calc_md:
                        control_dict["velocity constraintz"] = "set NULL NULL 0.0"
        return control_dict

    def interactive_indices_getter(self):
        return np.array(self._interactive_library.gather_atoms("type", 0, 1))

    def interactive_energy_pot_getter(self):
        return self._interactive_library.get_thermo("pe")

    def interactive_energy_tot_getter(self):
        return self._interactive_library.get_thermo("etotal")

    def interactive_steps_getter(self):
        return self._interactive_library.get_thermo("step")

    def interactive_temperatures_getter(self):
        return self._interactive_library.get_thermo("temp")

    def interactive_pressures_getter(self):
        pp = np.array(
            [
                [
                    self._interactive_library.get_thermo("pxx"),
                    self._interactive_library.get_thermo("pxy"),
                    self._interactive_library.get_thermo("pxz"),
                ],
                [
                    self._interactive_library.get_thermo("pxy"),
                    self._interactive_library.get_thermo("pyy"),
                    self._interactive_library.get_thermo("pyz"),
                ],
                [
                    self._interactive_library.get_thermo("pxz"),
                    self._interactive_library.get_thermo("pyz"),
                    self._interactive_library.get_thermo("pzz"),
                ],
            ]
        )
        if _check_ortho_prism(prism=self._prism):
            rotation_matrix = self._prism.R.T
            pp = rotation_matrix.T @ pp @ rotation_matrix
        return pp

    def interactive_indices_setter(self, indices, el_eam_lst):
        el_struct_lst = self._structure.get_species_symbols()
        el_obj_lst = self._structure.get_species_objects()
        el_dict = {}
        for id_eam, el_eam in enumerate(el_eam_lst):
            if el_eam in el_struct_lst:
                id_el = list(el_struct_lst).index(el_eam)
                el = el_obj_lst[id_el]
                el_dict[el] = id_eam + 1
        elem_all = np.array([el_dict[self._structure.species[el]] for el in indices])
        if self._cores == 1:
            self._interactive_library.scatter_atoms(
                "type", 0, 1, (len(elem_all) * c_int)(*elem_all)
            )
        else:
            self._interactive_library.scatter_atoms("type", elem_all)

    def interactive_stress_getter(self, enable_stress_computation=True):
        """
        This gives back an Nx3x3 array of stress/atom defined in http://lammps.sandia.gov/doc/compute_stress_atom.html
        Keep in mind that it is stress*volume in eV. Further discussion can be found on the website above.

        Returns:
            numpy.array: Nx3x3 np array of stress/atom
        """
        if enable_stress_computation:
            self.interactive_lib_command("compute st all stress/atom NULL")
            self.interactive_lib_command("run 0")
        id_lst = self._interactive_library.extract_atom("id", 0)
        id_lst = np.array([id_lst[i] for i in range(len(self._structure))]) - 1
        id_lst = np.arange(len(id_lst))[np.argsort(id_lst)]
        ind = np.array([0, 3, 4, 3, 1, 5, 4, 5, 2])
        ss = self._interactive_library.extract_compute("st", 1, 2)
        ss = np.array(
            [ss[i][j] for i in range(len(self._structure)) for j in range(6)]
        ).reshape(-1, 6)[id_lst]
        ss = (
            ss[:, ind].reshape(len(self._structure), 3, 3)
            / constants.eV
            * constants.bar
            * constants.angstrom**3
        )
        if _check_ortho_prism(prism=self._prism):
            ss = np.einsum("ij,njk->nik", self._prism.R, ss)
            ss = np.einsum("nij,kj->nik", ss, self._prism.R)
        return ss

    def close(self):
        if self._interactive_library is not None:
            self._interactive_library.close()

    def set_fix_external_callback(self, fix_id, callback, caller=None):
        self._interactive_library.set_fix_external_callback(
            fix_id=fix_id, callback=callback, caller=caller
        )


def cell_is_skewed(cell, tolerance=1.0e-8):
    """
    Check whether the simulation box is skewed/sheared. The algorithm compares the box volume
    and the product of the box length in each direction. If these numbers do not match, the box
    is considered to be skewed and the function returns True

    Args:
        tolerance (float): Relative tolerance above which the structure is considered as skewed

    Returns:
        (bool): Whether the box is skewed or not.
    """
    volume = np.abs(np.linalg.det(cell))
    prod = np.linalg.norm(cell, axis=-1).prod()
    if volume > 0:
        if abs(volume - prod) / volume < tolerance:
            return False
    return True
