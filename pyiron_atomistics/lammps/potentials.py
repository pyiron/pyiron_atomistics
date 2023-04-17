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


class LammpsPotentials:
    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls)
        cls._df = None
        return obj

    def copy(self):
        new_pot = LammpsPotentials()
        new_pot.set_df(self.get_df())
        return new_pot

    @staticmethod
    def _harmonize_args(args) -> str:
        if len(args) == 0:
            raise ValueError("Chemical elements not specified")
        if len(args) == 1:
            args *= 2
        return list(args)

    @property
    def model(self): -> str
        """Model name (required in pyiron df)"""
        return "_and_".join(set(self.df.model))

    @property
    def name(self) -> str:
        """Potential name (required in pyiron df)"""
        return "_and_".join(set(self.df.name))

    @property
    def species(self):
        """Species defined in the potential"""
        species = set([ss for s in self.df.interacting_species for ss in s])
        preset = set(["___".join(s) for s in self.df.preset_species if len(s) > 0])
        if len(preset) == 0:
            return list(species)
        elif len(preset) > 1:
            raise NotImplementedError(
                "Currently not possible to have multiple file-based potentials"
            )
        preset = list(preset)[0].split("___")
        return [p for p in preset + list(species - set(preset)) if p != "*"]

    @property
    def filename(self) -> list:
        """LAMMPS potential files"""
        return [f for f in set(self.df.filename) if len(f) > 0]

    @property
    def citations(self) -> str:
        """Citations to be included"""
        return "".join(np.unique([c for c in self.df.citations if len(c) > 0]))

    @property
    def is_scaled(self) -> bool:
        """Scaling in pair_style hybrid/scaled and hybrid/overlay (whih is scale=1)"""
        return "scale" in self.df

    @property
    def pair_style(self) -> str:
        """LAMMPS pair_style"""
        if len(set(self.df.pair_style)) == 1:
            pair_style = "pair_style " + list(set(self.df.pair_style))[0]
            if np.max(self.df.cutoff) > 0:
                pair_style += f" {np.max(self.df.cutoff)}"
            return pair_style + "\n"
        elif "scale" not in self.df:
            pair_style = "pair_style hybrid"
        elif all(self.df.scale == 1):
            pair_style = "pair_style hybrid/overlay"
        else:
            pair_style = "pair_style hybrid/scaled"
        for ii, s in enumerate(self.df[["pair_style", "cutoff"]].values):
            if pair_style.startswith("pair_style hybrid/scaled"):
                pair_style += f" {self.df.iloc[ii].scale}"
            pair_style += f" {s[0]}"
            if s[1] > 0:
                pair_style += f" {s[1]}"
        return pair_style + "\n"

    @property
    def pair_coeff(self) -> list:
        """LAMMPS pair_coeff"""
        class PairCoeff:
            def __init__(
                self,
                is_hybrid,
                pair_style,
                interacting_species,
                pair_coeff,
                species,
                preset_species,
            ):
                self.is_hybrid = is_hybrid
                self._interacting_species = interacting_species
                self._pair_coeff = pair_coeff
                self._species = species
                self._preset_species = preset_species
                self._pair_style = pair_style

            @property
            def counter(self):
                """
                Enumeration of potentials if a potential is used multiple
                times in hybrid (which is a requirement from LAMMPS)
                """
                key, count = np.unique(self._pair_style, return_counts=True)
                counter = {kk: 1 for kk in key[count > 1]}
                results = []
                for coeff in self._pair_style:
                    if coeff in counter and self.is_hybrid:
                        results.append(str(counter[coeff]))
                        counter[coeff] += 1
                    else:
                        results.append("")
                return results

            @property
            def pair_style(self):
                """pair_style to be output only in hybrid"""
                if self.is_hybrid:
                    return self._pair_style
                else:
                    return len(self._pair_style) * [""]

            @property
            def results(self):
                """pair_coeff lines to be used in pyiron df"""
                return [
                    " ".join((" ".join(("pair_coeff",) + c)).split()) + "\n"
                    for c in zip(
                        self.interacting_species,
                        self.pair_style,
                        self.counter,
                        self.pair_coeff,
                    )
                ]

            @property
            def interacting_species(self) -> list:
                """
                Species in LAMMPS notation (i.e. in numbers instead of chemical
                symbols)
                """
                s_dict = dict(
                    zip(self._species, (np.arange(len(self._species)) + 1).astype(str))
                )
                s_dict.update({"*": "*"})
                return [
                    " ".join([s_dict[cc] for cc in c])
                    for c in self._interacting_species
                ]

            @property
            def pair_coeff(self) -> list:
                """
                Args for pair_coeff. Elements defined in EAM files are
                complemented with the ones defined in other potentials in the
                case of hybrid (filled with NULL)
                """
                if not self.is_hybrid:
                    return self._pair_coeff
                results = []
                for pc, ps in zip(self._pair_coeff, self._preset_species):
                    if len(ps) > 0 and "eam" in pc:
                        s = " ".join(ps + (len(self._species) - len(ps)) * ["NULL"])
                        pc = pc.replace(" ".join(ps), s)
                    results.append(pc)
                return results

        return PairCoeff(
            is_hybrid="hybrid" in self.pair_style,
            pair_style=self.df.pair_style,
            interacting_species=self.df.interacting_species,
            pair_coeff=self.df.pair_coeff,
            species=self.species,
            preset_species=self.df.preset_species,
        ).results

    @property
    def pyiron_df(self):
        """df used in pyiron potential"""
        return pd.DataFrame(
            {
                "Config": [[self.pair_style] + self.pair_coeff],
                "Filename": [self.filename],
                "Model": [self.model],
                "Name": [self.name],
                "Species": [self.species],
                "Citations": [self.citations],
            }
        )

    def __repr__(self):
        return self.df.__repr__()

    def _repr_html_(self):
        return self.df._repr_html_()

    def set_df(self, df):
        for key in [
            "pair_style",
            "interacting_species",
            "pair_coeff",
            "preset_species",
        ]:
            if key not in df:
                raise ValueError(f"{key} missing")
        self._df = df

    @property
    def df(self):
        """DataFrame containing all info for each pairwise interactions"""
        return self._df

    def get_df(self, default_scale=None):
        if default_scale is None or "scale" in self.df:
            return self.df.copy()
        df = self.df.copy()
        df["scale"] = 1
        return df

    def __mul__(self, scale_or_potential):
        if isinstance(scale_or_potential, LammpsPotentials):
            if self.is_scaled or scale_or_potential.is_scaled:
                raise ValueError("You cannot mix hybrid types")
            new_pot = LammpsPotentials()
            new_pot.set_df(
                pd.concat(
                    (self.get_df(), scale_or_potential.get_df()), ignore_index=True
                )
            )
            return new_pot
        if self.is_scaled:
            raise NotImplementedError("Currently you cannot scale twice")
        new_pot = self.copy()
        new_pot.df["scale"] = scale_or_potential
        return new_pot

    __rmul__ = __mul__

    def __add__(self, potential):
        new_pot = LammpsPotentials()
        new_pot.set_df(
            pd.concat(
                (self.get_df(default_scale=1), potential.get_df(default_scale=1)),
                ignore_index=True,
            )
        )
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
        scale=None,
        cutoff=None,
    ):
        def check_none_n_length(variable, default, length=len(pair_coeff)):
            if variable is None:
                variable = default
            if not isinstance(variable, list):
                return variable
            if len(variable) == 1 and len(variable) < length:
                variable = length * variable
            return variable

        arg_dict = {
            "pair_style": pair_style,
            "interacting_species": interacting_species,
            "pair_coeff": pair_coeff,
            "preset_species": check_none_n_length(preset_species, [[]]),
            "cutoff": check_none_n_length(cutoff, 0),
            "model": check_none_n_length(model, pair_style),
            "citations": check_none_n_length(citations, [[]]),
            "filename": check_none_n_length(filename, [""]),
            "name": check_none_n_length(name, pair_style),
        }
        if scale is not None:
            arg_dict["scale"] = scale
        self.set_df(pd.DataFrame(arg_dict))


class EAM(LammpsPotentials):
    @staticmethod
    def _get_pair_style(config):
        if any(["hybrid" in c for c in config]):
            return [c.split()[3] for c in config if "pair_coeff" in c]
        for c in config:
            if "pair_style" in c:
                return [" ".join(c.replace("\n", "").split()[1:])] * sum(
                    ["pair_coeff" in c for c in config]
                )
        raise ValueError(f"pair_style could not determined: {config}")

    @staticmethod
    def _get_pair_coeff(config):
        try:
            if any(["hybrid" in c for c in config]):
                return [" ".join(c.split()[4:]) for c in config if "pair_coeff" in c]
            return [" ".join(c.split()[3:]) for c in config if "pair_coeff" in c]
        except IndexError:
            raise AssertionError(
                f"{config} does not follow the format 'pair_coeff element_1 element_2 args'"
            )

    @staticmethod
    def _get_interacting_species(config, species):
        def _convert(c, s):
            if c == "*":
                return c
            return s[int(c) - 1]

        return [
            [_convert(cc, species) for cc in c.split()[1:3]]
            for c in config
            if c.startswith("pair_coeff")
        ]

    @staticmethod
    def _get_scale(config):
        for c in config:
            if not c.startswith("pair_style"):
                continue
            if "hybrid/overlay" in c:
                return 1
            elif "hybrid/scaled" in c:
                raise NotImplementedError(
                    "Too much work for something inexistent in pyiron database for now"
                )
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
                warnings.warn(
                    f"Potential not uniquely specified - use default {df.Name}"
                )
            self._initialize_df(
                pair_style=self._get_pair_style(df.Config),
                interacting_species=self._get_interacting_species(
                    df.Config, df.Species
                ),
                pair_coeff=self._get_pair_coeff(df.Config),
                preset_species=[df.Species],
                model=df.Model,
                citations=df.Citations,
                filename=df.Filename,
                name=df.Name,
                scale=self._get_scale(df.Config),
            )
        return self._df


class Morse(LammpsPotentials):
    def __init__(self, *chemical_elements, D_0, alpha, r_0, cutoff, pair_style="morse"):
        self._initialize_df(
            pair_style=[pair_style],
            interacting_species=[self._harmonize_args(chemical_elements)],
            pair_coeff=[" ".join([str(cc) for cc in [D_0, alpha, r_0, cutoff]])],
            cutoff=cutoff,
        )


class CustomPotential(LammpsPotentials):
    def __init__(self, pair_style, *chemical_elements, cutoff, **kwargs):
        self._initialize_df(
            pair_style=[pair_style],
            interacting_species=[self._harmonize_args(chemical_elements)],
            pair_coeff=[" ".join([str(cc) for cc in kwargs.values()]) + f" {cutoff}"],
            cutoff=cutoff,
        )
