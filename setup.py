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
                 'Programming Language :: Python :: 3.8',
                 'Programming Language :: Python :: 3.9',
                 'Programming Language :: Python :: 3.10'],

    keywords='pyiron',
    packages=find_packages(exclude=["*tests*", "*docs*", "*binder*", "*conda*", "*notebooks*", "*.ci_support*"]),
    install_requires=[
        'aimsgb==0.1.0',
        'ase==3.22.1',
        'defusedxml==0.7.1',
        'future==0.18.2',
        'h5py==3.6.0',
        'matplotlib==3.5.1',
        'mendeleev==0.9.0',
        'numpy==1.22.3',
        'pandas==1.4.2',
        'phonopy==2.14.0',
        'pint==0.19.2',
        'pyiron_base==0.5.11',
        'pymatgen==2022.4.19',
        'scipy==1.8.0',
        'seekpath==2.0.1',
        'scikit-learn==1.0.2',
        'spglib==1.16.3',
        'tables==3.7.0'
    ],
    cmdclass=versioneer.get_cmdclass(),

    )
