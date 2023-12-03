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
        'atomistics==0.1.12',
        'defusedxml==0.7.1',
        'h5py==3.10.0',
        'matplotlib==3.8.2',
        'mendeleev==0.14.0',
        'mp-api==0.39.0',
        'numpy==1.26.2',
        'pandas==2.1.3',
        'phonopy==2.20.0',
        'pint==0.22',
        'pyiron_base==0.6.11',
        'pylammpsmpi==0.2.9',
        'scipy==1.11.4',
        'seekpath==2.1.0',
        'scikit-learn==1.3.2',
        'spglib==2.1.0',
        'structuretoolkit==0.0.15'
    ],
    cmdclass=versioneer.get_cmdclass(),
    package_data={'': ['data/*.csv']},
    )
