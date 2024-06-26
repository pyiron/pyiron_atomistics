Simulates output directory of a VASP job that crashed early in its run.
In particular too early for the OUTCAR to contain the NIONS tag, which the
outcar parser implicitly relies on.
