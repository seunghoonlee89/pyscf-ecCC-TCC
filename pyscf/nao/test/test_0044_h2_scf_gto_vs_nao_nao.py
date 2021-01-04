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

from __future__ import print_function, division
import unittest, numpy as np
from pyscf import gto, scf
from pyscf.nao import nao, scf as scf_nao
#from pyscf.nao.hf import RHF

mol = gto.M( verbose = 1,
    atom = '''
        H     0    0        0
        H     0    0.757    0.587''', basis = 'cc-pvdz',)

class KnowValues(unittest.TestCase):
    
  def test_scf_gto_vs_nao(self):
    """ Test computation of overlaps between NAOs against overlaps computed between GTOs"""
    gto_hf = scf.RHF(mol)
    gto_hf.kernel()
    
    nao_hf = scf_nao(mf=gto_hf, gto=mol)
    nao_hf.dump_chkfile=False
    e_tot = nao_hf.kernel_scf()
    self.assertAlmostEqual(gto_hf.e_tot, e_tot, 4)
    for e1,e2 in zip(nao_hf.mo_energy[0,0],gto_hf.mo_energy): self.assertAlmostEqual(e1, e2, 3)
    for o1,o2 in zip(nao_hf.mo_occ[0,0],gto_hf.mo_occ): self.assertAlmostEqual(o1, o2)

if __name__ == "__main__": unittest.main()
