# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from unittest import TestCase
from pyiron_atomistics.atomistics.structure.factories.laves import LavesFactory


class TestAimsgbFactory(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.laves = LavesFactory()

    def test_C14(self):
        struct = self.laves.C14()

    def test_C15(self):
        struct = self.laves.C14()

    def test_C36(self):
        struct = self.laves.C14()
