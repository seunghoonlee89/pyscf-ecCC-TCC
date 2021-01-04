#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
A simple example to run CCSD(T) and UCCSD(T) calculation.
'''

import pyscf

mol = pyscf.M(
    atom = 'H 0 0 0; F 0 0 1.1',
    basis = 'ccpvdz')

#mf = mol.RHF().run()
#mycc = mf.CCSD().run()
#et = mycc.ccsd_t()
#print('CCSD(T) correlation energy', mycc.e_corr + et)

mf = mol.UHF().run()
mycc = mf.CCSD().run()
et = mycc.ccsd_t()
print('UCCSD(T) correlation energy', mycc.e_corr + et)


t1a, t1b = mycc.t1
print(t1a.shape, t1b.shape)
t2aa, t2ab, t2bb = mycc.t2
print(t2aa.shape, t2ab.shape, t2bb.shape)
nmo = mol.nao
nelec= mol.nelectron
nocc = nelec // 2
nvir = nmo-nocc
print('nmo, nocc, nvir =', nmo, nocc, nvir)

for i in range(nocc):
    for a in range(nvir):
        print('t1a',i,a,t1a[i][a])

for i in range(nocc):
    for j in range(nocc):
        for a in range(nvir):
            for b in range(nvir):
                print('t2aa',i,j,a,b,t2aa[i][j][a][b])

for i in range(nocc):
    for j in range(nocc):
        for a in range(nvir):
            for b in range(nvir):
                print('t2ab',i,j,a,b,t2ab[i][j][a][b])

