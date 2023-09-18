"""
Setuptools based setup module
"""
from setuptools import setup, find_packages

import versioneer

setup(
    name='pyiron-atomistics',
    version=versioneer.get_version(),
    description='pyiron - an integrated development environment (IDE) for computational materials science.',
    long_description='http://pyiron.org',

    url='https://github.com/pyiron/pyiron_atomistics',
    author='Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department',
    author_email='janssen@mpie.de',
    license='BSD',

    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: BSD License',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11'
    ],

    keywords='pyiron',
    packages=find_packages(exclude=[
        "*tests*", 
        "*docs*", 
        "*binder*", 
        "*.devcontainer*", 
        "*notebooks*",
        "*.ci_support*", 
        "*test_benchmarks*", 
        "*test_integration*", 
        "*.github*"
    ]),
    install_requires=[
        'ase==3.22.1',
        'atomistics==0.0.5',
        'defusedxml==0.7.1',
        'h5py==3.9.0',
        'matplotlib==3.7.2',
        'mendeleev==0.14.0',
        'mp-api==0.36.1',
        'numpy==1.24.3',
        'pandas==2.1.0',
        'phonopy==2.20.0',
        'pint==0.22',
        'pyiron_base==0.6.5',
        'scipy==1.11.2',
        'seekpath==2.1.0',
        'scikit-learn==1.3.0',
        'spglib==2.0.2',
        'structuretoolkit==0.0.11'
    ],
    cmdclass=versioneer.get_cmdclass(),

    )
