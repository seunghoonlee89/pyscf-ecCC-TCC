#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
A simple example to run MP2 calculation.
'''

import pyscf
from pyscf import gto, scf, lo, tools, symm, ao2mo


mol = gto.Mole()
mol.atom = '''
Be     0.000000   0.000000    2.000000
Be     0.000000   0.000000    0.000000
'''
mol.basis = 'sto-3g'
mol.build()

mf = scf.RHF(mol).run()
#mf.MP2().run()

orb = mf.mo_coeff
print('# of AO = %d, # of MO = %d' % (orb.shape[0], orb.shape[1]))

nelec = mol.nelectron
nmo  = orb.shape[1]
nocc = nelec//2
nvir = nmo - nocc 
print('# of occ = %d, # of vir = %d'%(nocc, nvir))

#2e int
eri = ao2mo.kernel(mol, orb, aosym=1)
print('MO integrals (ij|kl) with no symmetry have shape %s' %
      str(eri.shape))
eri = eri.reshape(nmo,nmo,nmo,nmo)
print('MO integrals (ij|kl) with no symmetry have reshape %s' %
      str(eri.shape))

mo_e = mf.mo_energy
print('MO energy have shape %s' %
      str(mo_e.shape))

#emp2_corr = 0
#for i in range(nocc):
#    for j in range(nocc):
#        for a in range(nvir):
#            for b in range(nvir):
#                tmp = eri[i][a+nocc][j][b+nocc]
#                tmp2= eri[i][b+nocc][j][a+nocc]
#                Edno= mo_e[a+nocc] + mo_e[b+nocc] - mo_e[i] - mo_e[j]
#                emp2_corr -= tmp * (2*tmp - tmp2) / Edno

emp2EN_corr = 0
for i in range(nocc):
    for j in range(nocc):
        for a in range(nvir):
            for b in range(nvir):
                tmp = eri[i][a+nocc][j][b+nocc]
                tmp2= eri[i][b+nocc][j][a+nocc]
                # si = sa, sj = sb, si != sj, sa != sb
                Edno = mo_e[a+nocc] + mo_e[b+nocc] - mo_e[i] - mo_e[j]
                Edno+= eri[i][i][j][j] 
                Edno+= eri[a+nocc][a+nocc][b+nocc][b+nocc] 
                Edno-= eri[i][i][a+nocc][a+nocc] - eri[i][a+nocc][a+nocc][i]  
                Edno-= eri[i][i][b+nocc][b+nocc] 
                Edno-= eri[j][j][a+nocc][a+nocc]
                Edno-= eri[j][j][b+nocc][b+nocc] - eri[j][b+nocc][b+nocc][j]  
                emp2EN_corr -= tmp * tmp / Edno
                # si = sj = sa = sb
                if(i!=j): Edno+= - eri[i][j][j][i]
                if(a!=b): Edno+= - eri[a+nocc][b+nocc][b+nocc][a+nocc] 
                Edno-= - eri[i][b+nocc][b+nocc][i] 
                Edno-= - eri[j][a+nocc][a+nocc][j] 
                emp2EN_corr -= tmp * (tmp - tmp2) / Edno


#print ("EMP2_corr w/o EN = ",emp2_corr)
print ("EMP2_corr w   EN = ",emp2EN_corr)
#print ("EMP2 w/o EN = ",mf.e_tot+emp2_corr)
print ("EMP2 w   EN = ",mf.e_tot+emp2EN_corr)

#mycc = mf.CCSD().run()
#et = mycc.ccsd_t()
#print('CCSD(T) correlation energy', mycc.e_corr + et)


