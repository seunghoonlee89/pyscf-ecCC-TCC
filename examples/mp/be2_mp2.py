#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
A simple example to run MP2 calculation.
'''

import pyscf
from pyscf import gto, scf, lo, tools, symm

mol = gto.M(atom = 'Be     0.0000    0.000000    2.0000;\
                    Be     0.0000    0.000000    0.0000',
         basis = "sto3g",
         verbose=5, spin=0) # When orbitals are localized the symmetry goes down from D6h to Cs. (Cs since sigma and pi do not mix)

mf = mol.RHF().run()

mf.MP2().run()

mycc = mf.CCSD().run()
et = mycc.ccsd_t()
print('CCSD(T) correlation energy', mycc.e_corr + et)


