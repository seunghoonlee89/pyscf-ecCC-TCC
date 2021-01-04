#!/usr/bin/env python
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

from pyscf import scf
from pyscf import gto

mol = gto.Mole()
mol.verbose = 5
mol.output = None#"out_bz"

mol.atom.extend([
    ["C", (-0.65830719,  0.61123287, -0.00800148)],
    ["C", ( 0.73685281,  0.61123287, -0.00800148)],
    ["C", ( 1.43439081,  1.81898387, -0.00800148)],
    ["C", ( 0.73673681,  3.02749287, -0.00920048)],
    ["C", (-0.65808819,  3.02741487, -0.00967948)],
    ["C", (-1.35568919,  1.81920887, -0.00868348)],
    ["H", (-1.20806619, -0.34108413, -0.00755148)],
    ["H", ( 1.28636081, -0.34128013, -0.00668648)],
    ["H", ( 2.53407081,  1.81906387, -0.00736748)],
    ["H", ( 1.28693681,  3.97963587, -0.00925948)],
    ["H", (-1.20821019,  3.97969587, -0.01063248)],
    ["H", (-2.45529319,  1.81939187, -0.00886348)],])


mol.basis = {"H": 'ccpvdz',
             "C": 'ccpvdz',}

mol.build()

##############
# SCF result
import time
rhf = scf.RHF(mol)
print('E_RHF =', rhf.scf())
print(time.clock())

import os
import tempfile
import numpy
import h5py
from pyscf import ao2mo
f, eritmp = tempfile.mkstemp()
os.close(f)

nocc = mol.nelectron // 2
co = rhf.mo_coeff[:,:nocc]
cv = rhf.mo_coeff[:,nocc:]
ao2mo.outcore.general(mol, (co,cv,co,cv), eritmp, max_memory=100, dataname='mp2_bz')
f = h5py.File(eritmp, 'r')

print(time.clock())
eia = rhf.mo_energy[:nocc].reshape(nocc,1) - rhf.mo_energy[nocc:]
nvir = eia.shape[1]
emp2 = 0
for i in range(nocc):
    # use numpy.array to load a block into memory
    g_i = numpy.array(f['mp2_bz'][i*nvir:i*nvir+nvir])
    for j in range(nocc):
        for a in range(nvir):
            ja = j * nvir + a
            for b in range(nvir):
                jb = j * nvir + b
                emp2 += g_i[a,jb] * (g_i[a,jb]*2-g_i[b,ja]) \
                        / (eia[i,a]+eia[j,b])
print('E_MP2 =', emp2) # -0.80003653259
f.close()
os.remove(eritmp)

print(time.clock())
