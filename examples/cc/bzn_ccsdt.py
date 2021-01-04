#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
A simple example to run CCSD(T) and UCCSD(T) calculation.
'''

import pyscf

mol = pyscf.M(
atom = 'C     0.0000    1.396792    0.0000;\
C     0.0000    -1.396792    0.0000;\
C     1.209657    0.698396    0.0000;\
C     -1.209657    -0.698396    0.0000;\
C    -1.209657    0.698396    0.0000;\
C     1.209657    -0.698396    0.0000;\
H     0.0000    2.484212    0.0000;\
H     2.151390    1.242106    0.0000;\
H     -2.151390    -1.242106    0.0000;\
H     -2.151390    1.242106    0.0000;\
H     2.151390    -1.242106    0.0000;\
H     0.0000    -2.484212    0.0000',
    basis = 'ccpvdz')

mf = mol.RHF().run()
mycc = mf.CCSD().run()
et = mycc.ccsd_t()
print('CCSD(T) correlation energy', mycc.e_corr + et)

