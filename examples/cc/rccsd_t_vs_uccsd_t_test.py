#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
A simple example to run CCSD(T) and UCCSD(T) calculation.
'''

import pyscf

mol = pyscf.M(
    atom = '''
            O     0.000000   0.000000    0.000000
            H     0.000000  -0.857000    0.587000
            H     0.000000   0.757000    0.687000
           ''',
    basis = 'ccpvdz')

mf = mol.RHF().run()
mycc = mf.CCSD()
mycc.kernel()
et = mycc.ccsd_t()
print('CCSD(T) correlation energy', mycc.e_corr + et)

mf = mol.UHF().run()
mycc = mf.CCSD()
mycc.kernel()
et = mycc.ccsd_t()
print('UCCSD(T) correlation energy', mycc.e_corr + et)



