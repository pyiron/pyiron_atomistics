# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import unittest
from pyiron_atomistics.sphinx import input_functions as spx_input


class TestSphinx(unittest.TestCase):
    def test_loop(self):
        scf_ccg = [
            spx_input.get_scf_block_CCG_group(d_energy=1 / (1 + i))
            for i in range(2)
        ]
        born_oppenheimer = spx_input.get_scf_diag_group(
            d_energy=1e-6, block_CCG=scf_ccg
        )
        self.assertEqual(
            spx_input.to_sphinx(born_oppenheimer),
            "dEnergy = 1e-06;\nblockCCG {\n\tdEnergy = 1.0;\n}\nblockCCG {\n\tdEnergy = 0.5;\n}\n"
        )

    def test_multiple_keys(self):
        main = spx_input.get_main_group(
            scfDiag=spx_input.get_scf_diag_group(),
            QN=spx_input.get_QN_group(),
            linQN=spx_input.get_linQN_group(),
            QN_=spx_input.get_QN_group()
        )
        inp_all = spx_input.get_all_group(main=main)
        self.assertEqual(
            spx_input.to_sphinx(inp_all),
            "main {\n\tscfDiag {}\n\tQN {}\n\tlinQN {}\n\tQN {}\n}\n"
        )


if __name__ == "__main__":
    unittest.main()
