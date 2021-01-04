# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
from pyscf import lib
from pyscf.pbc import gto, scf
from pyscf.pbc.tools import k2gamma

cell = gto.Cell()
cell.a = '''
     1.755000    1.755000    -1.755000
     1.755000    -1.755000    1.755000
     -1.755000    1.755000    1.755000'''
cell.atom = '''Li      0.00000      0.00000      0.00000'''
#same type of basis for different elements
cell.basis = 'gth-szv'
cell.pseudo = {'Li': 'GTH-PBE-q3'}
cell.mesh = [20]*3
cell.verbose = 6
cell.output = '/dev/null'
cell.build()

kpts = cell.make_kpts([2,2,2])

mf = scf.KUKS(cell, kpts)
mf.xc = 'lda,vwn'
mf.kernel()

def tearDownModule():
    global cell, mf
    del cell, mf


class KnownValues(unittest.TestCase):
    def test_k2gamma(self):
        popa, popb = mf.mulliken_meta()[0]
        self.assertAlmostEqual(lib.finger(popa).sum(), 1.5403023058, 7)
        self.assertAlmostEqual(lib.finger(popb).sum(), 1.5403023058, 7)

        popa, popb = k2gamma.k2gamma(mf).mulliken_meta()[0]
        self.assertAlmostEqual(lib.finger(popa), 0.8007278745, 7)
        self.assertAlmostEqual(lib.finger(popb), 0.8007278745, 7)


if __name__ == '__main__':
    print("Full Tests for pbc.tools.k2gamma")
    unittest.main()
