# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

__author__ = "Joerg Neugebauer, Sudarsan Surendralal, Jan Janssen"
__copyright__ = (
    "Copyright 2021, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "1.0"
__maintainer__ = "Sudarsan Surendralal"
__email__ = "surendralal@mpie.de"
__status__ = "production"
__date__ = "Sep 1, 2017"

import pandas as pd
from pyiron_atomistics.lammps.potential import LammpsPotentialFile
import numpy as np
import warnings


class LammpsPotential:
    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls)
        cls._df = None
        return obj

    def copy(self):
        new_pot = LammpsPotential()
        new_pot.set_df(self.get_df())
        return new_pot

    @staticmethod
    def _harmonize_args(args):
        if len(args) == 0:
            raise ValueError("Chemical elements not specified")
        if len(args) == 1:
            args *= 2
        return list(args)

    @property
    def model(self):
        return "_and_".join(set(self.df.model))

    @property
    def name(self):
        return "_and_".join(set(self.df.name))

    @property
    def species(self):
        species = set([ss for s in self.df.interacting_species for ss in s])
        preset = set(["___".join(s) for s in self.df.preset_species if len(s) > 0])
        if len(preset) == 0:
            return list(species)
        elif len(preset) > 1:
            raise NotImplementedError("Currently not possible to have multiple file-based potentials")
        preset = list(preset)[0].split('___')
        return preset + list(species - set(preset))

    @property
    def filename(self):
        return [f for f in set(self.df.filename) if len(f) > 0]

    @property
    def citations(self):
        return "".join(np.unique([c for c in self.df.citations if len(c) > 0]))

    @property
    def is_scaled(self):
        return "scale" in self.df

    @property
    def pair_style(self):
        if len(set(self.df.pair_style)) == 1:
            return "pair_style " + list(set(self.df.pair_style))[0]
        elif "scale" not in self.df:
            return "pair_style hybrid"
        elif all(self.df.scale == 1):
            return "pair_style hybrid/overlay " + " ".join(list(self.df.pair_style))
        return "pair_style hybrid/scaled " + " ".join([str(ss) for s in self.df[["scale", "pair_style"]].values for ss in s])

    @property
    def pair_coeff(self):
        def convert(c, species=self.species):
            s_dict = dict(
                zip(species, (np.arange(len(species)) + 1).astype(str))
            )
            s_dict.update({"*": "*"})
            return [s_dict[cc] for cc in c]
        if "hybrid" in self.pair_style:
            return [
                " ".join(
                    ["pair_coeff"] + convert(c[0]) + [c[1]] + [c[2]] + ["\n"]
                )
                for c in self.df[["interacting_species", "pair_style", "pair_coeff"]].values
            ]
        return [
            " ".join(
                ["pair_coeff"] + convert(c[0]) + [c[1]] + ["\n"]
            )
            for c in self.df[["interacting_species", "pair_coeff"]].values
        ]

    def __repr__(self):
        return self.df.__repr__()

    def _repr_html_(self):
        return self.df._repr_html_()

    def set_df(self, df):
        for key in ["pair_style", "interacting_species", "pair_coeff", "preset_species"]:
            if key not in df:
                raise ValueError(f"{key} missing")
        self._df = df

    @property
    def df(self):
        return self._df

    def get_df(self, default_scale=None):
        if default_scale is None or "scale" in self.df:
            return self.df.copy()
        df = self.df.copy()
        df["scale"] = 1
        return df

    def __mul__(self, scale_or_potential):
        if isinstance(scale_or_potential, LammpsPotential):
            if self.is_scaled or scale_or_potential.is_scaled:
                raise ValueError("You cannot mix hybrid types")
            new_pot = LammpsPotential()
            new_pot.set_df(pd.concat((self.get_df(), scale_or_potential.get_df()), ignore_index=True))
            return new_pot
        if self.is_scaled:
            raise NotImplementedError("Currently you cannot scale twice")
        new_pot = self.copy()
        new_pot.df['scale'] = scale_or_potential
        return new_pot

    __rmul__ = __mul__

    def __add__(self, potential):
        new_pot = LammpsPotential()
        new_pot.set_df(pd.concat((self.get_df(default_scale=1), potential.get_df(default_scale=1)), ignore_index=True))
        return new_pot

    def _initialize_df(
        self,
        pair_style,
        interacting_species,
        pair_coeff,
        preset_species=None,
        model=None,
        citations=None,
        filename=None,
        name=None,
        scale=None
    ):
        def check_none_n_length(variable, default, length=len(pair_coeff)):
            if variable is None:
                variable = default
            if len(variable) == 1 and len(variable) < length:
                variable = length * variable
            return variable
        arg_dict = {
            "pair_style": pair_style,
            "interacting_species": interacting_species,
            "pair_coeff": pair_coeff,
            "preset_species": check_none_n_length(preset_species, [[]]),
            "model": check_none_n_length(model, pair_style),
            "citations": check_none_n_length(citations, [[]]),
            "filename": check_none_n_length(filename, [""]),
            "name": check_none_n_length(name, pair_style)
        }
        if scale is not None:
            arg_dict["scale"] = scale
        self.set_df(pd.DataFrame(arg_dict))


class EAM(LammpsPotential):
    @staticmethod
    def _get_pair_style(config):
        if any(["hybrid" in c for c in config]):
            return [c.split()[3] for c in config if "pair_coeff" in c]
        for c in config:
            if "pair_style" in c:
                return [" ".join(c.replace('\n', '').split()[1:])] * sum(["pair_coeff" in c for c in config])
        raise ValueError(f"pair_style could not determined: {config}")

    @staticmethod
    def _get_pair_coeff(config):
        try:
            if any(["hybrid" in c for c in config]):
                return [" ".join(c.split()[4:]) for c in config if "pair_coeff" in c]
            return [" ".join(c.split()[3:]) for c in config if "pair_coeff" in c]
        except IndexError:
            raise AssertionError(f"{config} does not follow the format 'pair_coeff element_1 element_2 args'")

    @staticmethod
    def _get_interacting_species(config, species):
        def _convert(c, s):
            if c == "*":
                return c
            return s[int(c) - 1]
        return [[_convert(cc, species) for cc in c.split()[1:3]] for c in config if c.startswith('pair_coeff')]


    @staticmethod
    def _get_scale(config):
        for c in config:
            if not c.startswith("pair_style"):
                continue
            if "hybrid/overlay" in c:
                return 1
            elif "hybrid/scaled" in c:
                raise NotImplementedError("Too much work for something inexistent in pyiron database for now")
        return

    def __init__(self, *chemical_elements, name=None, pair_style=None):
        if name is not None:
            self._df_candidates = LammpsPotentialFile().find_by_name(name)
        else:
            self._df_candidates = LammpsPotentialFile().find(list(chemical_elements))

    def list_potentials(self):
        return self._df_candidates.Name

    def view_potentials(self):
        return self._df_candidates

    @property
    def df(self):
        if self._df is None:
            df = self._df_candidates.iloc[0]
            if len(self._df_candidates) > 1:
                warnings.warn(f"Potential not specified - chose {df.Name}")
            self._initialize_df(
                pair_style=self._get_pair_style(df.Config),
                interacting_species=self._get_interacting_species(df.Config, df.Species),
                pair_coeff=self._get_pair_coeff(df.Config),
                preset_species=[df.Species],
                model=df.Model,
                citations=df.Citations,
                filename=df.Filename,
                name=df.Name,
                scale=self._get_scale(df.Config)
            )
        return self._df


class Morse(LammpsPotential):
    def __init__(self, *chemical_elements, D_0, alpha, r_0, cutoff, pair_style="morse"):
        self._initialize_df(
            pair_style=[pair_style],
            interacting_species=[self._harmonize_args(chemical_elements)],
            pair_coeff=[" ".join([str(cc) for cc in [D_0, alpha, r_0, cutoff]])],
        )

class CustomPotential(LammpsPotential):
    def __init__(self, pair_style, *chemical_elements, **kwargs):
        self._initialize_df(
            pair_style=[pair_style],
            interacting_species=[self._harmonize_args(chemical_elements)],
            pair_coeff=[" ".join([str(cc) for cc in kwargs.values()])],
        )
