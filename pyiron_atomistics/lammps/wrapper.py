from ctypes import c_double, c_int
import numpy as np
import warnings
from pyiron_atomistics.lammps.structure import UnfoldingPrism
from pyiron_atomistics.lammps.base import _check_ortho_prism


def interactive_positions_getter(lmp, number_of_atoms, prism):
    positions = np.reshape(
        np.array(lmp.gather_atoms("x", 1, 3)),
        (number_of_atoms, 3),
    )
    if _check_ortho_prism(prism=prism):
        positions = np.matmul(positions, prism.R.T)
    return positions


def interactive_positions_setter(lmp, logger, positions, prism, cores, interactive):
    if _check_ortho_prism(prism=prism):
        positions = np.array(positions).reshape(-1, 3)
        positions = np.matmul(positions, prism)
    positions = np.array(positions).flatten()
    if interactive and cores == 1:
        lmp.scatter_atoms("x", 1, 3, (len(positions) * c_double)(*positions))
    else:
        lmp.scatter_atoms("x", positions)
    interactive_lib_command(lmp=lmp, logger=logger, command="change_box all remap")


def interactive_lib_command(lmp, logger, command):
    logger.debug("Lammps library: " + command)
    lmp.command(command)


def interactive_cells_getter(lmp):
    return np.array(
        [
            [lmp.get_thermo("lx"), 0, 0],
            [
                lmp.get_thermo("xy"),
                lmp.get_thermo("ly"),
                0,
            ],
            [
                lmp.get_thermo("xz"),
                lmp.get_thermo("yz"),
                lmp.get_thermo("lz"),
            ],
        ]
    )


def interactive_cells_setter(lmp, logger, cell, structure_current, structure_previous):
    prism = UnfoldingPrism(cell)
    lx, ly, lz, xy, xz, yz = prism.get_lammps_prism()
    if _check_ortho_prism(prism=prism):
        warnings.warn(
            "Warning: setting upper trangular matrix might slow down the calculation"
        )

    is_skewed = structure_current.is_skewed(tolerance=1.0e-8)
    was_skewed = structure_previous.is_skewed(tolerance=1.0e-8)

    if is_skewed:
        if not was_skewed:
            interactive_lib_command(
                lmp=lmp, logger=logger, command="change_box all triclinic"
            )
        interactive_lib_command(
            lmp=lmp,
            logger=logger,
            command="change_box all x final 0 %f y final 0 %f z final 0 %f  xy final %f xz final %f yz final %f remap units box"
            % (lx, ly, lz, xy, xz, yz),
        )
    elif was_skewed:
        interactive_lib_command(
            lmp=lmp,
            logger=logger,
            command="change_box all x final 0 %f y final 0 %f z final 0 %f xy final %f xz final %f yz final %f remap units box"
            % (lx, ly, lz, 0.0, 0.0, 0.0),
        )
        interactive_lib_command(lmp=lmp, logger=logger, command="change_box all ortho")
    else:
        interactive_lib_command(
            lmp=lmp,
            logger=logger,
            command="change_box all x final 0 %f y final 0 %f z final 0 %f remap units box"
            % (lx, ly, lz),
        )
    return prism


def interactive_volume_getter(lmp):
    return lmp.get_thermo("vol")


def interactive_forces_getter(lmp, prism, number_of_atoms):
    ff = np.reshape(
        np.array(lmp.gather_atoms("f", 1, 3)),
        (number_of_atoms, 3),
    )
    if _check_ortho_prism(prism=prism):
        ff = np.matmul(ff, prism.R.T)
    return ff


def interactive_structure_setter(
    lmp,
    logger,
    structure_current,
    structure_previous,
    units,
    dimension,
    boundary,
    atom_style,
    el_eam_lst,
    calc_md,
    interactive,
    cores,
):
    old_symbols = structure_previous.get_species_symbols()
    new_symbols = structure_current.get_species_symbols()
    if any(old_symbols != new_symbols):
        raise ValueError(
            f"structure has different chemical symbols than old one: {new_symbols} != {old_symbols}"
        )
    interactive_lib_command(lmp=lmp, logger=logger, command="clear")
    control_dict = set_selective_dynamics(structure=structure_current, calc_md=calc_md)
    interactive_lib_command(lmp=lmp, logger=logger, command="units " + units)
    interactive_lib_command(
        lmp=lmp, logger=logger, command="dimension " + str(dimension)
    )
    interactive_lib_command(lmp=lmp, logger=logger, command="boundary " + boundary)
    interactive_lib_command(lmp=lmp, logger=logger, command="atom_style " + atom_style)

    interactive_lib_command(lmp=lmp, logger=logger, command="atom_modify map array")
    prism = UnfoldingPrism(structure_current.cell)
    if _check_ortho_prism(prism=prism):
        warnings.warn(
            "Warning: setting upper trangular matrix might slow down the calculation"
        )
    xhi, yhi, zhi, xy, xz, yz = prism.get_lammps_prism()
    if prism.is_skewed():
        interactive_lib_command(
            lmp=lmp,
            logger=logger,
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
        interactive_lib_command(
            lmp=lmp,
            logger=logger,
            command="region 1 block"
            + " 0.0 "
            + str(xhi)
            + " 0.0 "
            + str(yhi)
            + " 0.0 "
            + str(zhi)
            + " units box",
        )
    el_struct_lst = structure_current.get_species_symbols()
    el_obj_lst = structure_current.get_species_objects()
    if atom_style == "full":
        interactive_lib_command(
            lmp=lmp,
            logger=logger,
            command="create_box "
            + str(len(el_eam_lst))
            + " 1 "
            + "bond/types 1 "
            + "angle/types 1 "
            + "extra/bond/per/atom 2 "
            + "extra/angle/per/atom 2 ",
        )
    else:
        interactive_lib_command(
            lmp=lmp, logger=logger, command="create_box " + str(len(el_eam_lst)) + " 1"
        )
    el_dict = {}
    for id_eam, el_eam in enumerate(el_eam_lst):
        if el_eam in el_struct_lst:
            id_el = list(el_struct_lst).index(el_eam)
            el = el_obj_lst[id_el]
            el_dict[el] = id_eam + 1
            interactive_lib_command(
                lmp=lmp,
                logger=logger,
                command="mass {0:3d} {1:f}".format(id_eam + 1, el.AtomicMass),
            )
        else:
            interactive_lib_command(
                lmp=lmp,
                logger=logger,
                command="mass {0:3d} {1:f}".format(id_eam + 1, 1.00),
            )
    positions = structure_current.positions.flatten()
    if _check_ortho_prism(prism=prism):
        positions = np.array(positions).reshape(-1, 3)
        positions = np.matmul(positions, prism.R)
    positions = positions.flatten()
    try:
        elem_all = np.array(
            [el_dict[el] for el in structure_current.get_chemical_elements()]
        )
    except KeyError:
        missing = set(structure_current.get_chemical_elements()).difference(
            el_dict.keys()
        )
        missing = ", ".join([el.Abbreviation for el in missing])
        raise ValueError(
            f"Structure contains elements [{missing}], that are not present in the potential!"
        )
    if interactive and cores == 1:
        lmp.create_atoms(
            n=len(structure_current),
            id=None,
            type=(len(elem_all) * c_int)(*elem_all),
            x=(len(positions) * c_double)(*positions),
            v=None,
            image=None,
            shrinkexceed=False,
        )
    else:
        lmp.create_atoms(
            n=len(structure_current),
            id=None,
            type=elem_all,
            x=positions,
            v=None,
            image=None,
            shrinkexceed=False,
        )
    interactive_lib_command(lmp=lmp, logger=logger, command="change_box all remap")
    return prism, control_dict


def set_selective_dynamics(structure, calc_md):
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
                not_constrained_xyz[np.setdiff1d(np.setdiff1d(ind_x, ind_y), ind_z)] + 1
            )
            constraint_y = (
                not_constrained_xyz[np.setdiff1d(np.setdiff1d(ind_y, ind_z), ind_x)] + 1
            )
            constraint_z = (
                not_constrained_xyz[np.setdiff1d(np.setdiff1d(ind_z, ind_x), ind_y)] + 1
            )
            control_dict = {}
            if len(constraint_xyz) > 0:
                control_dict["group constraintxyz"] = "id " + " ".join(
                    [str(ind) for ind in constraint_xyz]
                )
                control_dict["fix constraintxyz"] = "constraintxyz setforce 0.0 0.0 0.0"
                if calc_md:
                    control_dict["velocity constraintxyz"] = "set 0.0 0.0 0.0"
            if len(constraint_xy) > 0:
                control_dict["group constraintxy"] = "id " + " ".join(
                    [str(ind) for ind in constraint_xy]
                )
                control_dict["fix constraintxy"] = "constraintxy setforce 0.0 0.0 NULL"
                if calc_md:
                    control_dict["velocity constraintxy"] = "set 0.0 0.0 NULL"
            if len(constraint_yz) > 0:
                control_dict["group constraintyz"] = "id " + " ".join(
                    [str(ind) for ind in constraint_yz]
                )
                control_dict["fix constraintyz"] = "constraintyz setforce NULL 0.0 0.0"
                if calc_md:
                    control_dict["velocity constraintyz"] = "set NULL 0.0 0.0"
            if len(constraint_zx) > 0:
                control_dict["group constraintxz"] = "id " + " ".join(
                    [str(ind) for ind in constraint_zx]
                )
                control_dict["fix constraintxz"] = "constraintxz setforce 0.0 NULL 0.0"
                if calc_md:
                    control_dict["velocity constraintxz"] = "set 0.0 NULL 0.0"
            if len(constraint_x) > 0:
                control_dict["group constraintx"] = "id " + " ".join(
                    [str(ind) for ind in constraint_x]
                )
                control_dict["fix constraintx"] = "constraintx setforce 0.0 NULL NULL"
                if calc_md:
                    control_dict["velocity constraintx"] = "set 0.0 NULL NULL"
            if len(constraint_y) > 0:
                control_dict["group constrainty"] = "id " + " ".join(
                    [str(ind) for ind in constraint_y]
                )
                control_dict["fix constrainty"] = "constrainty setforce NULL 0.0 NULL"
                if calc_md:
                    control_dict["velocity constrainty"] = "set NULL 0.0 NULL"
            if len(constraint_z) > 0:
                control_dict["group constraintz"] = "id " + " ".join(
                    [str(ind) for ind in constraint_z]
                )
                control_dict["fix constraintz"] = "constraintz setforce NULL NULL 0.0"
                if calc_md:
                    control_dict["velocity constraintz"] = "set NULL NULL 0.0"
    return control_dict


def interactive_indices_getter(lmp):
    return np.array(lmp.gather_atoms("type", 0, 1))


def interactive_energy_pot_getter(lmp):
    return lmp.get_thermo("pe")


def interactive_energy_tot_getter(lmp):
    return lmp.get_thermo("etotal")


def interactive_steps_getter(lmp):
    return lmp.get_thermo("step")


def interactive_temperatures_getter(lmp):
    return lmp.get_thermo("temp")


def interactive_pressures_getter(lmp, prism):
    pp = np.array(
        [
            [
                lmp.get_thermo("pxx"),
                lmp.get_thermo("pxy"),
                lmp.get_thermo("pxz"),
            ],
            [
                lmp.get_thermo("pxy"),
                lmp.get_thermo("pyy"),
                lmp.get_thermo("pyz"),
            ],
            [
                lmp.get_thermo("pxz"),
                lmp.get_thermo("pyz"),
                lmp.get_thermo("pzz"),
            ],
        ]
    )
    if _check_ortho_prism(prism=prism):
        rotation_matrix = prism.R.T
        pp = rotation_matrix.T @ pp @ rotation_matrix
    return pp
