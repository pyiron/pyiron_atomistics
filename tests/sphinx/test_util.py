# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import os
import unittest
from tempfile import TemporaryDirectory
import warnings

from pyiron_base import Settings
from pyiron_atomistics.sphinx.util import sxversions


class TestSphinxUtil(unittest.TestCase):
    @unittest.skipIf('linux' not in sys.platform, "Running of the addon is only supported on linux")
    def test_sxversions(self):
        rp = Settings().resource_paths

        # count warnings upon executing sxversions(True)
        n_other_warnings = 0
        try:
            with warnings.catch_warnings(record=True) as w:
                sxversions(True)
                n_other_warnings = len(w)
        except:
            pass

        with TemporaryDirectory() as tempd:
            try:
                # add tempd to resource_paths
                rp.append(tempd)
                # create .json file
                sxdir = os.path.join(tempd, "sphinx")
                os.mkdir(sxdir)
                with open(os.path.join(sxdir, "sxversions.json"), "w") as jsonfile:
                    jsonfile.write(
                        '{ "sxv_json_tst" : "echo json", "sxv_json_tst2" : "echo json" }\n'
                    )
                # test that this is parsed correctly
                sxv = sxversions(True)
                self.assertIn("sxv_json_tst", sxv.keys())
                self.assertIn("sxv_json_tst2", sxv.keys())
                self.assertEqual(sxv["sxv_json_tst"], "echo json")

                # create json-writing script
                scriptname = os.path.join(sxdir, "sxversions.sh")
                with open(scriptname, "w") as jsonscript:
                    jsonscript.writelines(
                        ["#!/bin/sh\n", 'echo \'{ "sxv_json_tst" : "echo script" }\'\n']
                    )
                # make script executable
                os.chmod(scriptname, 0o700)
                # test that sxversions is unchanged without refresh
                sxv = sxversions()
                self.assertEqual(sxv["sxv_json_tst"], "echo json")
                # --- test that script contents are parsed correctly,
                #     and we get 1 more warning for overwriting
                with warnings.catch_warnings(record=True) as w:
                    sxv = sxversions(True)
                    self.assertEqual(len(w), n_other_warnings + 1)
                # test that script output overwrites .json contents,
                self.assertEqual(sxv["sxv_json_tst"], "echo script")

                # and that other .json stuff is still there
                self.assertIn("sxv_json_tst2", sxv.keys())
                self.assertEqual(sxv["sxv_json_tst2"], "echo json")

                # test that cleaning works as well
                os.remove(os.path.join(sxdir, "sxversions.json"))
                sxv = sxversions(True)
                self.assertEqual(sxv["sxv_json_tst"], "echo script")
                self.assertNotIn("sxv_json_tst2", sxv.keys())
            finally:
                # ensure that tempd is always removed from resource_paths
                rp.remove(tempd)
                sxversions(True)


if __name__ == "__main__":
    unittest.main()
