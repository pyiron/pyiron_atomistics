# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import os
import unittest
from pyiron_atomistics.lammps.potential import LammpsPotentialFile
from pyiron_atomistics.vasp.potential import VaspPotential
from pyiron_base import state
from abc import ABC


class _PotentialTester(unittest.TestCase, ABC):
    """Overrides the settings so that the tests/static directory is used as the resources, then refreshes afterwards"""

    @classmethod
    def setUpClass(cls):
        state.update({'resource_paths': os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../static")})

    @classmethod
    def tearDownClass(cls) -> None:
        state.update()


class TestLammpsPotentialFile(_PotentialTester):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.kim = LammpsPotentialFile()

    def test_find(self):
        Fe_desired = ["Fe_C_Becquart_eam", "Fe_C_Hepburn_Ackland_eam"]
        Fe_available = list(self.kim.find("Fe")["Name"])
        for potl in Fe_desired:
            self.assertIn(
                potl, Fe_available,
                msg=f"Failed to find {potl}, which is expected since it is in the tests/static resources"
            )
        self.assertIn(
            "Al_Mg_Mendelev_eam", list(self.kim.find({"Al", "Mg"})["Name"]),
            msg="Failed to find Al_Mg_Mendelev_eam, which is expected since it is in the tests/static resources"
        )

    def test_pythonic_functions(self):
        self.assertEqual(
            list(self.kim.find("Fe")["Name"]), list(self.kim["Fe"].list()["Name"]),
            msg="List conversion method failed."
        )
        self.assertEqual(
            list(self.kim.find("Fe")["Name"]), list(self.kim.Fe.list()["Name"]),
            msg="Element symbol attribute does not find element."
        )
        self.assertEqual(
            list(self.kim.find({"Al", "Mg"})["Name"]),
            list(self.kim["Al"]["Mg"].list()["Name"]),
            msg="Double find not equivalent to nested dictionary access."
        )
        self.assertEqual(
            list(self.kim.find({"Al", "Mg"})["Name"]),
            list(self.kim.Al.Mg.list()["Name"]),
            msg="Nested attribute access failed"
        )
        self.assertEqual(
            list(self.kim.Mg.Al.list()["Name"]),
            list(self.kim.Al.Mg.list()["Name"]),
            msg="Elemental pairs should be commutative"
        )


class TestVaspPotential(_PotentialTester):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.vasp = VaspPotential()

    def test_find(self):
        self.assertEqual(
            list(self.vasp.pbe.find("Fe")["Name"]),
            [
                "Fe-gga-pbe",
                "Fe_GW-gga-pbe",
                "Fe_pv-gga-pbe",
                "Fe_sv-gga-pbe",
                "Fe_sv_GW-gga-pbe",
            ],
        )
        self.assertEqual(
            sorted(list(self.vasp.pbe.find({"Fe", "C"})["Name"])),
            [
                "C-gga-pbe",
                "C_GW-gga-pbe",
                "C_GW_new-gga-pbe",
                "C_h-gga-pbe",
                "C_s-gga-pbe",
                "Fe-gga-pbe",
                "Fe_GW-gga-pbe",
                "Fe_pv-gga-pbe",
                "Fe_sv-gga-pbe",
                "Fe_sv_GW-gga-pbe",
            ],
        )

    def test_pythonic_functions(self):
        self.assertEqual(
            list(self.vasp.pbe.Fe.list()["Name"]),
            [
                "Fe-gga-pbe",
                "Fe_GW-gga-pbe",
                "Fe_pv-gga-pbe",
                "Fe_sv-gga-pbe",
                "Fe_sv_GW-gga-pbe",
            ],
        )
        self.assertEqual(
            sorted(list(self.vasp.pbe.Fe.C.list()["Name"])),
            [
                "C-gga-pbe",
                "C_GW-gga-pbe",
                "C_GW_new-gga-pbe",
                "C_h-gga-pbe",
                "C_s-gga-pbe",
                "Fe-gga-pbe",
                "Fe_GW-gga-pbe",
                "Fe_pv-gga-pbe",
                "Fe_sv-gga-pbe",
                "Fe_sv_GW-gga-pbe",
            ],
        )


if __name__ == "__main__":
    unittest.main()
