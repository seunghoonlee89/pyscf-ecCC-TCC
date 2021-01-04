#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
A simple example to run MP2 calculation.
'''

import pyscf
from pyscf import gto, scf, lo, tools, symm

mol = gto.M(atom = 'C     0.0000    1.396792    0.0000;\
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
basis = "ccpvdz",
verbose=5, symmetry="Cs", spin=0) # When orbitals are localized the symmetry goes down from D6h to Cs. (Cs since sigma and pi do not mix)

mf = mol.RHF().run()

mp2 = mf.MP2()
#mp2 = mf.MP2(EN=True)

mp2.run()

