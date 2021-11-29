# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import unittest
import numpy as np
from pyiron_atomistics.atomistics.structure.neighbors import NeighborsTrajectory
from pyiron_atomistics.atomistics.structure.structurestorage import StructureStorage
from pyiron_atomistics.atomistics.structure.factory import StructureFactory

class TestNeighborsTrajectory(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.canonical = StructureStorage()
        cls.grandcanonical = StructureStorage()

        bulk = StructureFactory().bulk("Fe")
        for n in range(3):
            bulk.positions += np.random.normal(scale=.1, size=bulk.positions.shape)
            cls.canonical.add_structure(bulk)
            cls.grandcanonical.add_structure(bulk.repeat( (n+1, 1, 1) ))

    def test_canonical(self):
        traj = NeighborsTrajectory(has_structure=self.canonical)
        traj.compute_neighbors()
        self.assertEqual(traj.indices.shape, (3, 1, 12),
                         "indices from trajectory have wrong shape.")
        self.assertEqual(traj.distances.shape, (3, 1, 12),
                         "distances from trajectory have wrong shape.")
        self.assertEqual(traj.vecs.shape, (3, 1, 12, 3),
                         "vecs from trajectory have wrong shape.")
        for i in range(3):
            neigh = self.canonical.get_structure(i).get_neighbors()
            self.assertTrue(np.array_equal(traj.indices[i], neigh.indices),
                            "indices from trajectory and get_neighbors not equal")
            self.assertTrue(np.array_equal(traj.distances[i], neigh.distances),
                            "distances from trajectory and get_neighbors not equal")
            self.assertTrue(np.array_equal(traj.vecs[i], neigh.vecs),
                            "vecs from trajectory and get_neighbors not equal")

    def test_grandcanonical(self):
        traj = NeighborsTrajectory(has_structure=self.grandcanonical)
        traj.compute_neighbors()
        self.assertEqual(traj.indices.shape, (3, 3, 12),
                         "indices from trajectory have wrong shape.")
        self.assertEqual(traj.distances.shape, (3, 3, 12),
                         "distances from trajectory have wrong shape.")
        self.assertEqual(traj.vecs.shape, (3, 3, 12, 3),
                         "vecs from trajectory have wrong shape.")
        for i in range(3):
            neigh = self.grandcanonical.get_structure(i).get_neighbors()
            self.assertTrue(np.array_equal(traj.indices[i][:i+1], neigh.indices),
                            "indices from trajectory and get_neighbors not equal")
            self.assertTrue(np.array_equal(traj.distances[i][:i+1], neigh.distances),
                            "distances from trajectory and get_neighbors not equal")
            self.assertTrue(np.array_equal(traj.vecs[i][:i+1], neigh.vecs),
                            "vecs from trajectory and get_neighbors not equal")
