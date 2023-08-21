# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from pyiron_atomistics._tests import TestWithProject
from pyiron_atomistics.atomistics.structure.factories.aimsgb import AimsgbFactory


class TestAimsgbFactory(TestWithProject):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.fcc_basis = cls.project.create.structure.bulk('Al', cubic=True)
        cls.factory = AimsgbFactory()

    def test_grain_thickness(self):
        axis = [0, 0, 1]
        sigma = 5
        plane = [1, 2, 0]
        gb1 = self.factory.build(axis, sigma, plane, self.fcc_basis)  # Default thicknesses expected to be 1
        uc_a, uc_b = 2, 3  # Make grains thicker
        gb2 = self.factory.build(axis, sigma, plane, self.fcc_basis, uc_a=uc_a, uc_b=uc_b)
        self.assertEqual(
            ((uc_a + uc_b)/2)*len(gb1), len(gb2),
            msg="Expected structure to be bigger in proportion to grain thickness"
        )
