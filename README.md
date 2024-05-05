# pyiron_atomistics

[![Build Status](https://github.com/pyiron/pyiron_atomistics/workflows/Python%20package/badge.svg)](https://github.com/pyiron//pyiron/actions)
![Anaconda](https://anaconda.org/conda-forge/pyiron_atomistics/badges/downloads.svg)
![Release](https://anaconda.org/conda-forge/pyiron_atomistics/badges/latest_release_date.svg)

pyiron - an integrated development environment (IDE) for computational materials science. It combines several tools in
a common platform:

* Atomic structure objects – compatible to the [Atomic Simulation Environment (ASE)](https://wiki.fysik.dtu.dk/ase/).
* Atomistic simulation codes – like [LAMMPS](http://lammps.sandia.gov) and [VASP](https://www.vasp.at).
* Feedback Loops – to construct dynamic simulation life cycles.
* Hierarchical data management – interfacing with storage resources like SQL and [HDF5](https://support.hdfgroup.org/HDF5/).
* Integrated visualization – based on [NGLview](https://github.com/arose/nglview).
* Interactive simulation protocols - based on [Jupyter notebooks](http://jupyter.org).
* Object-oriented job management – for scaling complex simulation protocols from single jobs to high-throughput simulations.

![Screenshot of pyiron_atomistics running inside jupyterlab.](https://raw.githubusercontent.com/pyiron/pyiron_atomistics/main/docs/images/screenshots.png)

pyiron (called pyron) is developed in the [Computational Materials Design department](https://www.mpie.de/CM) of
[Joerg Neugebauer](https://www.mpie.de/person/43010/2763386) at the [Max Planck Institut für Eisenforschung (Max Planck Institute for iron research)](https://www.mpie.de/2281/en).
While its original focus was to provide a framework to develop and run complex simulation protocols as needed for ab
initio thermodynamics it quickly evolved into a versatile tool to manage a wide variety of simulation tasks. In 2016 the
[Interdisciplinary Centre for Advanced Materials Simulation (ICAMS)](http://www.icams.de) joined the development of the
framework with a specific focus on high throughput applications. In 2018 pyiron was released as open-source project.

**pyiron_atomistics**: This is the documentation page for the basic infrastructure moduls of pyiron.  If you're new to
pyiron and want to get an overview head over to [pyiron](https://pyiron.readthedocs.io/en/latest/).  If you're looking
for the API docs of pyiron_base check [pyiron_base](https://pyiron_base.readthedocs.io/en/latest/).

## Explore pyiron_atomistics
We provide various options to install, explore and run pyiron_atomistics:

* **Workstation Installation (recommeded)**: for Windows, Linux or Mac OS X workstations (interface for local VASP 
  executable, support for the latest jupyterlab based GUI)
* **Mybinder.org (beta)**: test pyiron directly in your browser (no VASP license, no visualization, only temporary data
  storage)
* **Docker (for demonstration)**: requires Docker installation (no VASP license, only temporary data storage)

## Join the development
Please contact us if you are interested in using pyiron:

* to interface your simulation code or method
* implementing high-throughput approaches based on atomistic codes
* to learn more about method development and Big Data in material science.

Please also check out the pyiron_atomistics [contributing guidelines](https://github.com/pyiron/pyiron_atomistics/blob/main/CONTRIBUTING.rst).

## Citing
If you use pyiron in your research, please consider citing the following work:

```
@article{pyiron-paper,
    title = {pyiron: An integrated development environment for computational materials science},
    journal = {Computational Materials Science},
    volume = {163},
    pages = {24 - 36},
    year = {2019},
    issn = {0927-0256},
    doi = {https://doi.org/10.1016/j.commatsci.2018.07.043},
    url = {http://www.sciencedirect.com/science/article/pii/S0927025618304786},
    author = {Jan Janssen and Sudarsan Surendralal and Yury Lysogorskiy and Mira Todorova and Tilmann Hickel and Ralf Drautz and Jörg Neugebauer},
    keywords = {Modelling workflow, Integrated development environment, Complex simulation protocols},
}
```
