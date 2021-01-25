# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from __future__ import print_function
from ast import literal_eval
import numpy as np
import pandas as pd
import shutil
import os
from pyiron_base import Settings, GenericParameters
from pyiron_base import InputList
from pyiron_atomistics.atomistics.job.potentials import PotentialAbstract, find_potential_file_base

__author__ = "Joerg Neugebauer, Sudarsan Surendralal, Jan Janssen"
__copyright__ = (
    "Copyright 2020, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "1.0"
__maintainer__ = "Sudarsan Surendralal"
__email__ = "surendralal@mpie.de"
__status__ = "production"
__date__ = "Sep 1, 2017"

s = Settings()


class LammpsPotential(GenericParameters):

    """
    This module helps write commands which help in the control of parameters related to the potential used in LAMMPS
    simulations
    """

    def __init__(self, input_file_name=None):
        super(LammpsPotential, self).__init__(
            input_file_name=input_file_name,
            table_name="potential_inp",
            comment_char="#",
        )
        self._potential = None
        self._attributes = {}
        self._df = None
        self._file_eam = None
        self.input = InputList()

    @property
    def df(self):
        try:
            self._df["Filename"].values[0]
            self.parameters
            return self.input.pot
        except:
            self.parameters
            return self._df

    @df.setter
    def df(self, new_dataframe):
        self._df = new_dataframe
        # ToDo: In future lammps should also support more than one potential file - that is currently not implemented.
        try:
            self.load_string("".join(list(new_dataframe["Config"])[0]))
        except IndexError:
            raise ValueError(
                "Potential not found! "
                "Validate the potential name by self.potential in self.list_potentials()."
            )
    @property
    def parameters(self):
        if self._file_eam is None:
            file_name = self.files[0]
            with open(file_name, 'r') as ffile:
                self._file_eam = ffile.readlines()
            self._initialize_eam()
        else:
            pass

    def __setitem__(self, key, value):
        if hasattr(value, '__len__'):
            value = np.array(value).tolist()
        param = key.split('/')
        if len(param) == 1:
            super(LammpsPotential, self).__setitem__(key, value)
        else:
            elems = sorted(param[1].split('-'))
            if len(elems) == 1:
                key = str(param[0]) + '/' + str(elems[0]) + '-' + str(elems[0])
            else:
                key = str(param[0]) + '/' + str(elems[0]) + '-' + str(elems[1])
            super(LammpsPotential, self).__setitem__(key, value)

            # To do!
            # for kk in self.pair_coeff_parameters:
            #    if kk == param[0] or kk.endswith('_eam'):
            #        continue
            #    if self[key.replace(param[0], kk)] is None and self[key] is not None:
            #        self[key.replace(param[0], kk)] = self._pair_pot_keys(self._model_pair_pot[0])[0][kk]
            #    elif self[key.replace(param[0], kk)] is not None and self[key] is None:
            #        self[key.replace(param[0], kk)] = None

    def __getitem__(self, key):
        param = key.split('/')
        if len(param) != 1:
            elems = sorted(param[1].split('-'))
            if len(elems) == 1:
                key = str(param[0]) + '/' + '-' + str(elems[0]) + str(elems[0])
            else:
                key = str(param[0]) + '/' + str(elems[0]) + '-' + str(elems[1])
        value = super(LammpsPotential, self).__getitem__(key)
        if isinstance(value, list):
            return np.array(value, dtype=float)
        else:
            return value
        
    @property
    def _model_mb(self): # mb stands for 'many-body'
        #if self.style=='pairwise':
        #    return []
        #intersect = list(set(self._pot_type).intersection(self._mb_pot_keys().keys()))
        #if len(intersect)!=0:
        #    return intersect
        if 'fs' in self._df["Filename"].values[0][0][-3:]:
            return ['eam/fs']
        else:
            return ['eam/alloy']
    
    @property
    def elements(self):
        return self.get_element_lst()

    @property
    def combinations(self):
        try:
            return self._combinations
        except AttributeError:
            self._combinations = np.array(
                [[elem, self.elements[j]] for i, elem in enumerate(self.elements) for j in range(i + 1)])
            return self._combinations

    @property
    def _elements_in_eam(self):
        if self._keys_eam is None or len(self._keys_eam[0]) <= 1:
            raise ValueError("EAM parsing not performed")
        return [self[k] for k in self._keys_eam[0][1:]]

    @property
    def eam_element_comb(self):
        try:
            return self._eam_comb
        except AttributeError:
            eam_list_comb = []
            self._eam_comb = np.array(self.combinations).tolist()
            for i, k in enumerate(self.combinations):
                if list(set(self.elements) - set(self._elements_in_eam)) in k:
                    warnings.simplefilter(action='ignore', category=FutureWarning)
                    eam_list_comb.append(i)
            for i in sorted(eam_list_comb, reverse=True):
                del self._eam_comb[i]
            return self._eam_comb

    @property
    def _eam_ending(self):
        # only eam implemented yet
        # if self.style == 'hybrid':
        #    return '_eam'
        # else:
        return ''

    @staticmethod
    def _has_non_number(list_of_something):
        if not hasattr(list_of_something, '__len__'):
            list_of_something = [list_of_something]
        for elem in list_of_something:
            try:
                float(elem)
            except ValueError:
                return True
        return False
    
    def _initialize_eam(self):
        self.input.create_group('pot')
        self._keys_eam = [['No_Elements' + self._eam_ending],
                          ['Nrho' + self._eam_ending, 'drho' + self._eam_ending, 'Nr' + self._eam_ending,
                           'dr' + self._eam_ending, 'cutoff' + self._eam_ending]]        
        self._keys_eam[0].extend(['Element_eam_' + str(i + 1) for i in range(len(self._file_eam[3].split()) - 1)])        
        for i in range(2):
            for k, v in zip(self._keys_eam[i], self._file_eam[i + 3].split()):                
                self[k] = v                
                self.input.pot[k] = v
        for k in ['Nrho' + self._eam_ending, 'Nr' + self._eam_ending]:
            self[k] = int(self[k])
            self.input.pot[k] = int(self[k])
        for k in ['drho' + self._eam_ending, 'dr' + self._eam_ending, 'cutoff' + self._eam_ending]:            
            self[k] = float(self[k])
            self.input.pot[k] = float(self[k])
        self._elem_props=['atomic_number'+self._eam_ending, 'mass'+self._eam_ending, 'lattice_constant'+self._eam_ending, 'lattice_type'+self._eam_ending]
        [self.input.pot.create_group(str(prop)) for prop in self._elem_props]
        #self._elem_pr ps = ['atomic_number' + self._eam_ending, 'mass' + self._eam_ending]
        self['meaningless_item'] = None
        property_lines = list(filter(lambda a: self._has_non_number(a.split()), self._file_eam[5:]))
        if len(self._elements_in_eam) != len(property_lines):
            warnings.warn('EAM parsing might have failed; number of elements defined (' + str(
                len(self._elements_in_eam)) + ') != number of element property lines (' + str(
                len(property_lines)) + ')')        
        for i_elem, elem in enumerate(self._elements_in_eam):            
            for i_prop, prop in enumerate(self._elem_props):
                self[str(prop) + '/' + str(elem) + '-' + (elem)] = property_lines[i_elem].split()[i_prop]                
                #self.input.pot.create_group(str(prop))                
                self.input.pot[str(prop) + '/' + str(elem) + '-' + (elem)] = property_lines[i_elem].split()[i_prop]
        tab_lines = list(filter(lambda a: self._has_non_number(a.split()) == False, self._file_eam[5:]))
        tab_lines = ''.join(tab_lines).replace('\n', ' ').split()        
        start_line = 0
        self.input.pot.create_group('F')
        self.input.pot.create_group('rho')
        self.input.pot.create_group('phi')
        for elem in self._elements_in_eam:            
            self['F' + self._eam_ending + '/' + elem + '-' + elem] = [float(value) for value in tab_lines[                                                                                                start_line:start_line +
                                                                                                           self[                                                                                    'Nrho' + self._eam_ending]]]
            self.input.pot['F' + self._eam_ending + '/' + elem + '-' + elem] = [float(value) for value in tab_lines[
                                                                                                start_line:start_line +                                                                                                           self[
                                                                                                               'Nrho' + self._eam_ending]]]
            start_line += self['Nrho' + self._eam_ending]
            if self._model_mb == ['eam/fs'] and self['No_Elements']>1:
                rho_Nr = self['Nrho'+self._eam_ending]*2
            else:
                rho_Nr = self['Nrho' + self._eam_ending]
            self['rho' + self._eam_ending + '/' + elem + '-' + elem] = [float(value) for value in
                                                                        tab_lines[start_line:start_line + rho_Nr]]
            self.input.pot['rho' + self._eam_ending + '/' + elem + '-' + elem] = [float(value) for value in
                                                                        tab_lines[start_line:start_line + rho_Nr]]            
            start_line += rho_Nr
        for i in range(self['No_Elements' + self._eam_ending]):
            for j in range(i + 1):
                self['phi' + self._eam_ending + '/' + self._elements_in_eam[j] + '-' + self._elements_in_eam[i]] = [
                    float(value) for value in tab_lines[start_line:start_line + self['Nr' + self._eam_ending]]]
                self.input.pot['phi' + self._eam_ending + '/' + self._elements_in_eam[j] + '-' + self._elements_in_eam[i]] = [
                    float(value) for value in tab_lines[start_line:start_line + self['Nr' + self._eam_ending]]]
                start_line += self['Nr' + self._eam_ending]
       # print(self.input.pot.phi())
        if len(self._keys_eam[0]) != self['No_Elements' + self._eam_ending] + 1:
            print('WARNING: Number of elements is not consistent in EAM File')
        self._value_modified = {k: False for k in self.get_pandas()['Parameter']}
        self._eam_parsing_successful = True

    @property
    def comment_str(self):
        return ''.join([line for line in self._file_eam[0:3]])

    @property
    def eam_info_str(self):
        first_line = ' '.join([str(self.input.pot[k]) for k in self._keys_eam[0]]) + '\n'
        second_line = ' '.join([str(self.input.pot[k]) for k in self._keys_eam[1]]) + '\n'
        return first_line + second_line

    def element_prop(self, element_1, element_2):
        elem_prop = ([self.input.pot[str(prop) + '/' + str(element_1) + '-' + str(element_2)] for prop in self._elem_props])        
        prop = ' '.join([str(prop) for prop in elem_prop]) + '\n'
        return prop

    def F_rho_str(self, element_1, element_2):
        F = np.array(self.input.pot['F' + self._eam_ending + '/' + str(element_1) + '-' + str(element_2)]).tolist()
        rho = np.array(self.input.pot['rho' + self._eam_ending + '/' + str(element_1) + '-' + str(element_2)]).tolist()
        F_rho = '\n'.join([str(value) for value in F + rho]) + '\n'
        return F_rho

    def phi_str(self, element_1, element_2):
        phi = np.array(self.input.pot['phi' + self._eam_ending + '/' + str(element_2) + '-' + str(element_1)]).tolist()
        return '\n'.join([str(pp) for pp in phi]) + '\n'

    def _update_eam_file(self):
        # self.parameters
        self.return_file = self.comment_str + self.eam_info_str
        for elem in self._elements_in_eam:
            self.return_file += self.element_prop(elem, elem) + self.F_rho_str(elem, elem)
        for elem in self.eam_element_comb:
            self.return_file += self.phi_str(elem[0], elem[1])

    def remove_structure_block(self):
        self.remove_keys(["units"])
        self.remove_keys(["atom_style"])
        self.remove_keys(["dimension"])

    @property
    def files(self):
        if len(self._df["Filename"].values[0]) > 0 and self._df["Filename"].values[0] != ['']:
            absolute_file_paths = [
                files for files in list(self._df["Filename"])[0] if os.path.isabs(files)
            ]
            relative_file_paths = [
                files
                for files in list(self._df["Filename"])[0]
                if not os.path.isabs(files)
            ]
            env = os.environ
            resource_path_lst = s.resource_paths
            for conda_var in ["CONDA_PREFIX", "CONDA_DIR"]:
                if conda_var in env.keys():  # support iprpy-data package
                    resource_path_lst += [os.path.join(env[conda_var], "share", "iprpy")]
            for path in relative_file_paths:
                absolute_file_paths.append(find_potential_file_base(
                    path=path,
                    resource_path_lst=resource_path_lst,
                    rel_path=os.path.join("lammps", "potentials")
                ))
            if len(absolute_file_paths) != len(list(self._df["Filename"])[0]):
                raise ValueError("Was not able to locate the potentials.")
            else:
                return absolute_file_paths

    def copy_pot_files(self, working_directory):
        #if self.files is not None:
        #    _ = [shutil.copy(path_pot, working_directory) for path_pot in self.files]
        self._update_eam_file()
        self.file_name = working_directory + '/' + self.files[0].split('/')[-1]
        with open(self.file_name, 'w') as ff:
            for line in self.return_file:
                ff.write(line)
        _ = [self.file_name]
        pot_f = working_directory + '/potential.inp'
        with open(pot_f, 'w') as ff:
            for line in self._df["Config"].values[0]:
                ff.write(line)
        _ = [pot_f]
    def get_element_lst(self):
        return list(self._df["Species"])[0]

    def _find_line_by_prefix(self, prefix):
        """
        Find a line that starts with the given prefix.  Differences in white
        space are ignored.  Raises a ValueError if not line matches the prefix.

        Args:
            prefix (str): line prefix to search for

        Returns:
            list: words of the matching line

        Raises:
            ValueError: if not matching line was found
        """

        def isprefix(prefix, lst):
            if len(prefix) > len(lst): return False
            return all(n == l for n, l in zip(prefix, lst))

        # compare the line word by word to also match lines that differ only in
        # whitespace
        prefix = prefix.split()
        for parameter, value in zip(self._dataset["Parameter"],
                                    self._dataset["Value"]):
            words = (parameter + " " + value).strip().split()
            if isprefix(prefix, words):
                return words

        raise ValueError("No line with prefix \"{}\" found.".format(
                            " ".join(prefix)))

    def get_element_id(self, element_symbol):
        """
        Return numeric element id for element. If potential does not contain
        the element raise a :class:NameError.  Only makes sense for potentials
        with pair_style "full".

        Args:
            element_symbol (str): short symbol for element

        Returns:
            int: id matching the given symbol

        Raise:
            NameError: if potential does not contain this element
        """

        try:
            line = "group {} type".format(element_symbol)
            return int(self._find_line_by_prefix(line)[3])

        except ValueError:
            msg = "potential does not contain element {}".format(
                    element_symbol)
            raise NameError(msg) from None

    def get_charge(self, element_symbol):
        """
        Return charge for element. If potential does not specify a charge,
        raise a :class:NameError.  Only makes sense for potentials
        with pair_style "full".

        Args:
            element_symbol (str): short symbol for element

        Returns:
            float: charge speicified for the given element

        Raises:
            NameError: if potential does not specify charge for this element
        """

        try:
            line = "set group {} charge".format(element_symbol)
            return float(self._find_line_by_prefix(line)[4])

        except ValueError:
            msg = "potential does not specify charge for element {}".format(
                    element_symbol)
            raise NameError(msg) from None

    def to_hdf(self, hdf, group_name=None):
        if self._df is not None:
            with hdf.open("potential") as hdf_pot:
                hdf_pot["Config"] = self._df["Config"].values[0]
                hdf_pot["Filename"] = self._df["Filename"].values[0]
                hdf_pot["Name"] = self._df["Name"].values[0]
                hdf_pot["Model"] = self._df["Model"].values[0]  
                hdf_pot["Species"] = self._df["Species"].values[0]
                if "Citations" in self._df.columns.values:
                    hdf_pot["Citations"] = self._df["Citations"].values[0]
        super(LammpsPotential, self).to_hdf(hdf, group_name=group_name)

    def from_hdf(self, hdf, group_name=None):
        with hdf.open("potential") as hdf_pot:
            try:
                entry_dict = {
                    "Config": [hdf_pot["Config"]],
                    "Filename": [hdf_pot["Filename"]],
                    "Name": [hdf_pot["Name"]],
                    "Model": [hdf_pot["Model"]],
                    "Species": [hdf_pot["Species"]],
                }
                if "Citations" in hdf_pot.list_nodes():
                    entry_dict["Citations"] = [hdf_pot["Citations"]]
                self._df = pd.DataFrame(entry_dict)
            except ValueError:
                pass
        super(LammpsPotential, self).from_hdf(hdf, group_name=group_name)
class LammpsPotentialFile(PotentialAbstract):
    """
    The Potential class is derived from the PotentialAbstract class, but instead of loading the potentials from a list,
    the potentials are loaded from a file.

    Args:
        potential_df:
        default_df:
        selected_atoms:
    """

    def __init__(self, potential_df=None, default_df=None, selected_atoms=None):
        if potential_df is None:
            potential_df = self._get_potential_df(
                plugin_name="lammps",
                file_name_lst={"potentials_lammps.csv"},
                backward_compatibility_name="lammpspotentials",
            )
        super(LammpsPotentialFile, self).__init__(
            potential_df=potential_df,
            default_df=default_df,
            selected_atoms=selected_atoms,
        )

    def default(self):
        if self._default_df is not None:
            atoms_str = "_".join(sorted(self._selected_atoms))
            return self._default_df[
                (self._default_df["Name"] == self._default_df.loc[atoms_str].values[0])
            ]
        return None

    def find_default(self, element):
        """
        Find the potentials
        Args:
            element (set, str): element or set of elements for which you want the possible LAMMPS potentials
            path (bool): choose whether to return the full path to the potential or just the potential name
        Returns:
            list: of possible potentials for the element or the combination of elements
        """
        if isinstance(element, set):
            element = element
        elif isinstance(element, list):
            element = set(element)
        elif isinstance(element, str):
            element = set([element])
        else:
            raise TypeError("Only, str, list and set supported!")
        element_lst = list(element)
        if self._default_df is not None:
            merged_lst = list(set(self._selected_atoms + element_lst))
            atoms_str = "_".join(sorted(merged_lst))
            return self._default_df[
                (self._default_df["Name"] == self._default_df.loc[atoms_str].values[0])
            ]
        return None

    def __getitem__(self, item):
        potential_df = self.find(element=item)
        selected_atoms = self._selected_atoms + [item]
        return LammpsPotentialFile(
            potential_df=potential_df,
            default_df=self._default_df,
            selected_atoms=selected_atoms,
        )


class PotentialAvailable(object):
    def __init__(self, list_of_potentials):
        self._list_of_potentials = list_of_potentials

    def __getattr__(self, name):
        if name in self._list_of_potentials:
            return name
        else:
            raise AttributeError

    def __dir__(self):
        return self._list_of_potentials

    def __repr__(self):
        return str(dir(self))

