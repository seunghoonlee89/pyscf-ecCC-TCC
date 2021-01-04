#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
A simple example to run CCSD(T) and UCCSD(T) calculation.
'''

import pyscf

mol = pyscf.M(
#      atom = '''
#             C     0.000000   1.396792    0.000000
#             C     0.000000  -1.396792    0.000000
#             C     1.209657   0.698396    0.000000
#             C    -1.209657  -0.698396    0.000000
#             C    -1.209657   0.698396    0.000000
#             C     1.209657  -0.698396    0.000000
#             H     0.000000   2.484212    0.000000
#             H     2.151390   1.242106    0.000000
#             H    -2.151390  -1.242106    0.000000
#             H    -2.151390   1.242106    0.000000
#             H     2.151390  -1.242106    0.000000
#             H     0.000000  -2.484212    0.000000
#            ''',
#    atom = '''
#            Be     0.000000   0.000000    2.000000
#            Be     0.000000   0.000000    0.000000
#           ''',
    atom = '''
            O     0.000000   0.000000    0.000000
            H     0.000000  -0.857000    0.587000
            H     0.000000   0.757000    0.687000
           ''',
    #verbose=7,
    #basis = '321g')
    #basis = 'sto-3g')
    basis = 'ccpvdz')

mf = mol.RHF().run()
mycc = mf.CCSD()
#mycc.frozen = 6
mycc.kernel()
et = mycc.ccsd_t()
print('CCSD(T) correlation energy', mycc.e_corr + et)

mf = mol.UHF().run()
mycc = mf.CCSD()
#mycc.frozen = 6
mycc.kernel()
et = mycc.ccsd_t()
print('UCCSD(T) correlation energy', mycc.e_corr + et)



