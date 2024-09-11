# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import unittest
from pyiron_atomistics.sphinx import input_functions as spx_input


class TestSphinx(unittest.TestCase):
    def test_main(self):

        scf_ccg = [
            spx_input.get_scf_block_CCG_group(d_energy=1 / (1 + i))
            for i in range(3)
        ]
        born_oppenheimer = spx_input.get_scf_diag_group(
            d_energy=1e-6, block_CCG=scf_ccg
        )
        main = spx_input.get_main_group(
            scfDiag=spx_input.get_scf_diag_group(max_steps=10),
            QN=spx_input.get_QN_group(
                max_steps=100, born_oppenheimer=born_oppenheimer
            ),
            linQN=spx_input.get_linQN_group(max_steps=10, dF=1.0e-4),
            QN_=spx_input.get_QN_group(dX=1.0e-2)
        )
        inp_all = spx_input.get_all_group(main=main)
        self.assertEqual(
            spx_input.to_sphinx(inp_all),
            "main {\n\tscfDiag {\n\t\tmaxSteps = 10;\n\t}\n\tQN {\n\t\tmaxSteps = 100;\n\t\tbornOppenheimer {\n\t\t\tdEnergy = 1e-06;\n\t\t\tblockCCG {\n\t\t\t\tdEnergy = 1.0;\n\t\t\t}\n\t\t\tblockCCG {\n\t\t\t\tdEnergy = 0.5;\n\t\t\t}\n\t\t\tblockCCG {\n\t\t\t\tdEnergy = 0.3333333333333333;\n\t\t\t}\n\t\t}\n\t}\n\tlinQN {\n\t\tmaxSteps = 10;\n\t\tdF = 0.0001;\n\t}\n\tQN {\n\t\tdX = 0.01;\n\t}\n}\n"
        )


if __name__ == "__main__":
    unittest.main()
