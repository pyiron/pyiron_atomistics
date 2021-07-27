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

    classifiers=['Development Status :: 5 - Production/Stable',
                 'Topic :: Scientific/Engineering :: Physics',
                 'License :: OSI Approved :: BSD License',
                 'Intended Audience :: Science/Research',
                 'Operating System :: OS Independent',
                 'Programming Language :: Python :: 3',
                 'Programming Language :: Python :: 3.6',
                 'Programming Language :: Python :: 3.7',
                 'Programming Language :: Python :: 3.8',
                 'Programming Language :: Python :: 3.9'],

    keywords='pyiron',
    packages=find_packages(exclude=["*tests*", "*docs*", "*binder*", "*conda*", "*notebooks*", "*.ci_support*"]),
    install_requires=[
        'aimsgb==0.1.0',
        'ase==3.22.0',
        'defusedxml==0.7.1',
        'future==0.18.2',
        'h5py==3.3.0',
        'matplotlib==3.4.2',
        'mendeleev==0.7.0',
        'numpy==1.21.1',
        'pandas==1.3.0',
        'phonopy==2.11.0',
        'pyiron_base==0.2.20',
        'pymatgen==2022.0.11',
        'scipy==1.7.0',
        'seekpath==2.0.1',
        'six==1.16.0',
        'scikit-learn==0.24.2',
        'spglib==1.16.1',
        'tables==3.6.1'
    ],
    cmdclass=versioneer.get_cmdclass(),

    )
