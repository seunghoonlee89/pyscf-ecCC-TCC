#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
CCSD frozen core
'''

from pyscf import gto, scf, cc
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

mf = scf.RHF(mol).run()

#
# Freeze the inner most two orbitals.
#
mycc = cc.CCSD(mf)
mycc.frozen = 6 
mycc.kernel()
print('CCSD correlation energy', mycc.e_corr)
et = mycc.ccsd_t()
print('CCSD(T) correlation energy', mycc.e_corr + et)


