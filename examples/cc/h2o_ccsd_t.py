#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
A simple example to run CCSD(T) and UCCSD(T) calculation.
'''

import pyscf

mol = pyscf.M(
    atom = 'O       -0.26677564    -0.27872083     0.00000000;\
          H       -0.26677564     0.82127917     0.00000000;\
          H       -0.26677564    -0.64538753     1.03708994',
    basis = 'ccpvtz')

mf = mol.RHF().run()
mycc = mf.CCSD().run()
et = mycc.ccsd_t()
print('CCSD(T) correlation energy', mycc.e_corr + et)

