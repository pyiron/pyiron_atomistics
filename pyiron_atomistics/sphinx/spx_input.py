import numpy as np
from typing import Optional


def to_camel_case(snake_str):
    return "".join(x.capitalize() for x in snake_str.lower().split("_"))


def to_lower_camel_case(snake_str):
    # We capitalize the first letter of each component except the first one
    # with the 'capitalize' method and join them together.
    camel_string = to_camel_case(snake_str)
    return snake_str[0].lower() + camel_string[1:]


def format_value(v, indent=0):
    if isinstance(v, bool):
        return f" = {v};".lower()
    elif isinstance(v, dict) or isinstance(v, list):
        if len(v) == 0:
            return " {}"
        else:
            return " {\n" + to_sphinx(v, indent + 1) + indent * "\t" + "}"
    else:
        if isinstance(v, np.ndarray):
            v = v.tolist()
        return " = {!s};".format(v)


def to_sphinx(obj, indent=0):
    line = ""
    for k, v in obj.items():
        current_line = indent * "\t" + k.split("___")[0]
        if isinstance(v, list):
            for vv in v:
                line += current_line + format_value(vv, indent) + "\n"
        else:
            line += current_line + format_value(v, indent) + "\n"
    return line


def fill_values(group=None, **kwargs):
    if group is None:
        group = {}
    for k, v in kwargs.items():
        if v is not None and v is not False:
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                for i, vv in enumerate(v):
                    group[f"{k}___{i}"] = vv
            else:
                group[k] = v
    return group


def get_structure_group(
    cell: np.ndarray,
    movable: Optional[bool] = None,
    movable_x: Optional[bool] = None,
    movable_y: Optional[bool] = None,
    movable_z: Optional[bool] = None,
    species: Optional[list] = None,
) -> dict:
    """
    Args:
        cell (np.ndarray): Cell matrix
        movable (bool): Allow atoms to move. Default: all atoms are movable,
            unless any movable tag is used for any species/atom.
        movable_x (bool): Allow atoms to move in the x direction.
            Default: movable, unless movableY or movableZ are used.
        movable_y (bool): Allow atoms to move in the y direction.
            Default: movable, unless movableX or movableZ are used.
        movable_z (bool): Allow atoms to move in the z direction.
            Default: movable, unless movableX or movableY are used.
        species (list): Species
    """
    return fill_values(
        cell=cell,
        movable=movable,
        movableX=movable_x,
        movableY=movable_y,
        movableZ=movable_z,
        species=species,
    )


def get_species_group(
    element: str,
    atom: Optional[dict] = None,
) -> dict:
    """
    Args:
        element (str): Element
        atom (list): Atom
    """
    return fill_values(element=element, atom=atom)


def get_atom_group(
    coords: np.ndarray,
    relative: Optional[bool] = None,
    movable_line: Optional[list] = None,
    label: Optional[str] = None,
) -> dict:
    """
    Args:
        coords (np.ndarray): The atomic coordinates as a 3-vector. Un- less the
            relative tag is employed, the coordinates are Cartesian (in bohr)
        relative (bool): The coordinates are given relative to the unit cell vectors.
        movable_line (list): The movement of the atom is restricted to a line.
            The value gives the direction of the line as a 3-vector.
        label (str): Assign a label (or rather a tag) to this atom. If labels
            are used, atoms with different labels are considered inequivalent.
            Think of spin congurations for a use-case.
    """
    return fill_values(coords=coords, relative=relative, movableLine=movable_line, label=label)


def get_symmetry_group(
    operator: dict
) -> dict:
    """
    Args:
        operator (np.ndarray): Symmetry operator
    """
    return fill_values(operator=operator)


def get_basis_group(
    e_cut: float,
    g_cut: Optional[float] = None,
    folding: Optional[int] = None,
    mesh: Optional[list] = None,
    mesh_accuracy: Optional[float] = None,
    save_memory: Optional[bool] = None,
    k_points: Optional[dict] = None,
) -> dict:
    """
    Args:
        e_cut (float): Energy cutoff
        g_cut (float): Gradient cutoff
        folding (int): Folding
        mesh (list): Mesh
        mesh_accuracy (float): Mesh accuracy
        save_memory (bool): Save memory
        k_point (dict): K point
    """
    return fill_values(
        eCut=e_cut,
        gCut=g_cut,
        folding=folding,
        mesh=mesh,
        meshAccuracy=mesh_accuracy,
        saveMemory=save_memory,
        kPoint=k_point,
    )


def get_k_point_group(
    coords: np.ndarray,
    relative: Optional[bool] = None,
    weight: Optional[float] = None,
) -> dict:
    """
    Args:
        coords (np.ndarray): The k-point coordinates as a 3-vector. Unless the
            relative tag is employed, the coordinates are Cartesian.
        relative (bool): The coordinates are given relative to the unit cell vectors.
        weight (float): The weight of the k-point in the sampling.
    """
    return fill_values(coords=coords, relative=relative, weight=weight)


def get_CCG_group(
    d_energy: Optional[float] = None,
    max_steps: Optional[int] = None,
    print_steps: Optional[int] = None,
    initial_diag: Optional[bool] = None,
    final_diag: Optional[bool] = None,
    kappa: Optional[float] = None,
    keep_occ_fixed: Optional[bool] = None,
    ekt: Optional[float] = None,
    dipole_correction: Optional[bool] = None,
    no_rho_storage: Optional[bool] = None,
    no_wave_storage: Optional[bool] = None,
) -> dict:
    """
    Args:
        d_energy (float): Energy convergence criterion
        max_steps (int): Maximum number of SCF steps
        print_steps (int): Print SCF steps
        initial_diag (bool): Initial diagonalization
        final_diag (bool): Final diagonalization
        kappa (float): Kappa parameter
        keep_occ_fixed (bool): Keep occupation fixed
        ekt (float): Temperature
        dipole_correction (bool): Dipole correction
        no_rho_storage (bool): Do not store density
        no_wave_storage (bool): Do not store wave functions
    """
    return fill_values(
        dEnergy=d_energy,
        maxSteps=max_steps,
        printSteps=print_steps,
        initialDiag=initial_diag,
        finalDiag=final_diag,
        kappa=kappa,
        keepOccFixed=keep_occ_fixed,
        ekt=ekt,
        dipoleCorrection=dipole_correction,
        noRhoStorage=no_rho_storage,
        noWaveStorage=no_wave_storage,
    )


def get_scf_CCG_group(
    d_rel_eps: Optional[float] = None,
    max_steps_CCG: Optional[int] = None,
    d_energy: Optional[float] = None,
) -> dict:
    """
    Args:
        d_rel_eps (float): Relative energy convergence criterion
        max_steps_CCG (int): Maximum number of CCG steps
        d_energy (float): Energy convergence criterion
    """
    return fill_values(
        dRelEps=d_rel_eps,
        maxStepsCCG=max_steps_CCG,
        dEnergy=d_energy,
    )


def get_scf_block_CCG_group(
    d_rel_eps: Optional[float] = None,
    max_steps_CCG: Optional[int] = None,
    block_size: Optional[int] = None,
    n_sloppy: Optional[int] = None,
    d_energy: Optional[float] = None,
    verbose: Optional[bool] = None,
    numerical_limit: Optional[bool] = None,
) -> dict:
    """
    Args:
        d_rel_eps (float): Relative energy convergence criterion
        max_steps_CCG (int): Maximum number of CCG steps
        block_size (int): Block size
        n_sloppy (int): Number of sloppy steps
        d_energy (float): Energy convergence criterion
        verbose (bool): Verbose output
        numerical_limit (bool): Numerical limit
    """
    return fill_values(
        dRelEps=d_rel_eps,
        maxStepsCCG=max_steps_CCG,
        blockSize=block_size,
        nSloppy=n_sloppy,
        dEnergy=d_energy,
        verbose=verbose,
        numericalLimit=numerical_limit,
    )


def get_preconditioner_group(
    type: Optional[str] = "KERKER",
    scaling: Optional[float] = None,
    spin_scaling: Optional[float] = None,
    kerker_camping: Optional[float] = None,
    dielec_constant: Optional[float] = None,
) -> dict:
    return fill_values(
        type=type,
        scaling=scaling,
        spinScaling=spin_scaling,
        kerkerCamping=kerker_camping,
        dielecConstant=dielec_constant,
    )


def get_scf_diag_group(
    d_energy: Optional[float] = None,
    max_steps: Optional[int] = None,
    max_residue: Optional[float] = None,
    print_steps: Optional[int] = None,
    mixing_method: Optional[str] = None,
    n_pulay_steps: Optional[int] = None,
    rho_mixing: Optional[float] = None,
    spin_mixing: Optional[float] = None,
    keep_rho_fixed: Optional[bool] = None,
    keep_occ_fixed: Optional[bool] = None,
    keep_spin_fixed: Optional[bool] = None,
    ekt: Optional[float] = None,
    dipole_correction: Optional[bool] = None,
    d_spin_moment: Optional[float] = None,
    no_rho_storage: Optional[bool] = None,
    no_wave_storage: Optional[bool] = None,
    CCG: Optional[dict] = None,
    block_CCG: Optional[dict] = None,
    preconditioner: Optional[dict] = None,
) -> dict:
    """
    Args:
        d_energy (float): Energy convergence criterion
        max_steps (int): Maximum number of SCF steps
        max_residue (float): Residue convergence criterion
        print_steps (int): Print SCF steps
        mixing_method (str): Mixing method
        n_pulay_steps (int): Number of Pulay steps
        rho_mixing (float): Density mixing parameter
        spin_mixing (float): Spin mixing parameter
        keep_rho_fixed (bool): Keep density fixed
        keep_occ_fixed (bool): Keep occupation fixed
        keep_spin_fixed (bool): Keep spin fixed
        ekt (float): Temperature
        dipole_correction (bool): Dipole correction
        d_spin_moment (float): Spin moment convergence criterion
        no_rho_storage (bool): Do not store density
        no_wave_storage (bool): Do not store wave functions
        CCG (dict): Conjugate gradient method
        block_CCG (dict): Block conjugate gradient method
        preconditioner (dict): Preconditioner
    """
    return fill_values(
        dEnergy=d_energy,
        maxSteps=max_steps,
        maxResidue=max_residue,
        printSteps=print_steps,
        mixingMethod=mixing_method,
        nPulaySteps=n_pulay_steps,
        rhoMixing=rho_mixing,
        spinMixing=spin_mixing,
        keepRhoFixed=keep_rho_fixed,
        keepOccFixed=keep_occ_fixed,
        keepSpinFixed=keep_spin_fixed,
        ekt=ekt,
        dipoleCorrection=dipole_correction,
        dSpinMoment=d_spin_moment,
        noRhoStorage=no_rho_storage,
        noWaveStorage=no_wave_storage,
        CCG=CCG,
        blockCCG=block_CCG,
        preconditioner=preconditioner,
    )


def get_born_oppenheimer_group(
    scf_diag: Optional[dict] = None,
) -> dict:
    """
    Args:
        scf_diag (dict): SCF diagonalization
    """
    return fill_values(scfDiag=scf_diag)


def get_QN_group(
    max_steps: Optional[int] = None,
    dX: Optional[float] = None,
    dF: Optional[float] = None,
    d_energy: Optional[float] = None,
    max_step_length: Optional[float] = None,
    hessian: Optional[str] = None,
    drift_filter: Optional[bool] = None,
    born_oppenheimer: Optional[dict] = None,
) -> dict:
    """
    Args:
        max_steps (int): Maximum number of steps
        dX (float): Position convergence criterion
        dF (float): Force convergence criterion
        d_energy (float): Energy convergence criterion
        max_step_length (float): Maximum step length
        hessian (str): Initialize Hessian from file
        drift_filter (bool): Drift filter
        born_oppenheimer (dict): Born-Oppenheimer
    """
    return fill_values(
        maxSteps=max_steps,
        dX=dX,
        dF=dF,
        dEnergy=d_energy,
        maxStepLength=max_step_length,
        hessian=hessian,
        driftFilter=drift_filter,
        bornOppenheimer=born_oppenheimer,
    )


def get_linQN_group(
    max_steps: Optional[int] = None,
    dX: Optional[float] = None,
    dF: Optional[float] = None,
    d_energy: Optional[float] = None,
    max_step_length: Optional[float] = None,
    n_projectors: Optional[int] = None,
    hessian: Optional[str] = None,
    drift_filter: Optional[bool] = None,
    born_oppenheimer: Optional[dict] = None,
) -> dict:
    """
    Args:
        max_steps (int): Maximum number of steps
        dX (float): Position convergence criterion
        dF (float): Force convergence criterion
        d_energy (float): Energy convergence criterion
        max_step_length (float): Maximum step length
        n_projectors (int): Number of projectors
        hessian (str): Initialize Hessian from file
        drift_filter (bool): Drift filter
        born_oppenheimer (dict): Born-Oppenheimer
    """
    return fill_values(
        maxSteps=max_steps,
        dX=dX,
        dF=dF,
        dEnergy=d_energy,
        maxStepLength=max_step_length,
        nProjectors=n_projectors,
        hessian=hessian,
        driftFilter=drift_filter,
        bornOppenheimer=born_oppenheimer,
    )


def get_ricQN_group(
    max_steps: Optional[int] = None,
    dX: Optional[float] = None,
    dF: Optional[float] = None,
    d_energy: Optional[float] = None,
    max_step_length: Optional[float] = None,
    n_projectors: Optional[int] = None,
    soft_mode_damping: Optional[float] = None,
    drift_filter: Optional[bool] = None,
    born_oppenheimer: Optional[dict] = None,
) -> dict:
    """
    Args:
        max_steps (int): Maximum number of steps
        dX (float): Position convergence criterion
        dF (float): Force convergence criterion
        d_energy (float): Energy convergence criterion
        max_step_length (float): Maximum step length
        n_projectors (int): Number of projectors
        soft_mode_damping (float): Soft mode damping
        drift_filter (bool): Drift filter
        born_oppenheimer (dict): Born-Oppenheimer
    """
    return fill_values(
        maxSteps=max_steps,
        dX=dX,
        dF=dF,
        dEnergy=d_energy,
        maxStepLength=max_step_length,
        nProjectors=n_projectors,
        softModeDamping=soft_mode_damping,
        driftFilter=drift_filter,
        bornOppenheimer=born_oppenheimer,
    )


def get_ric_group(
    max_dist: Optional[float] = None,
    typify_threshold: Optional[float] = None,
    rms_threshold: Optional[float] = None,
    plane_cut_limit: Optional[float] = None,
    with_angles: Optional[bool] = None,
    bvk_atoms: Optional[str] = None,
    born_oppenheimer: Optional[dict] = None,
) -> dict:
    """
    Args:
        max_dist (float): Maximum distance
        typify_threshold (float): Typify threshold
        rms_threshold (float): RMS threshold
        plane_cut_limit (float): Plane cut limit
        with_angles (bool): With angles
        bvk_atoms (str): (experimental) List of atom ids (starting from 1) for
            which born-von-Karman transversal force constants are added. The
            comma-separated list must be enclosed by square brackets []. This
            adds a bond-directional coordinate to each bond of the atoms in the
            list.
        born_oppenheimer (dict): Born-Oppenheimer
    """
    return fill_values(
        maxDist=max_dist,
        typifyThreshold=typify_threshold,
        rmsThreshold=rms_threshold,
        planeCutLimit=plane_cut_limit,
        withAngles=with_angles,
        bvkAtoms=bvk_atoms,
        bornOppenheimer=born_oppenheimer,
    )


def get_ricTS_group(
    max_steps: Optional[int] = None,
    dX: Optional[float] = None,
    dF: Optional[float] = None,
    d_energy: Optional[float] = None,
    n_projectors: Optional[int] = None,
    max_step_length: Optional[float] = None,
    trans_curvature: Optional[float] = None,
    any_stationary_point: Optional[bool] = None,
    max_dir_rot: Optional[float] = None,
    scheme: Optional[int] = None,
    drift_filter: Optional[bool] = None,
    born_oppenheimer: Optional[dict] = None,
) -> dict:
    """
    Args:
        max_steps (int): Maximum number of steps
        dX (float): Position convergence criterion
        dF (float): Force convergence criterion
        d_energy (float): Energy convergence criterion
        n_projectors (int): Number of projectors
        max_step_length (float): Maximum step length
        trans_curvature (float): Transversal curvature
        any_stationary_point (bool): Any stationary point
        max_dir_rot (float): Maximum direction rotation
        scheme (int): Scheme
        drift_filter (bool): Drift filter
        born_oppenheimer (dict): Born-Oppenheimer
    """
    return fill_values(
        maxSteps=max_steps,
        dX=dX,
        dF=dF,
        dEnergy=d_energy,
        nProjectors=n_projectors,
        maxStepLength=max_step_length,
        transCurvature=trans_curvature,
        anyStationaryPoint=any_stationary_point,
        maxDirRot=max_dir_rot,
        scheme=scheme,
        driftFilter=drift_filter,
        bornOppenheimer=born_oppenheimer,
    )
