#!/usr/bin/env python
#
# Author: Seunghoon Lee <seunghoonlee89@gmail.com>
#

'''
An example to run externally corrected (ec)-CCSD calculation using FCI.
'''

from pyscf import gto, scf
import numpy 

mol = gto.Mole()
#mol.atom = [
#    ['O', ( 0., 0.    , 0.   )],
#    ['H', ( 0., -0.857, 0.587)],
#    ['H', ( 0., 0.757 , 0.687)],]
#mol.basis = '321g'

mol.atom = [
    ['Be', ( 0., 0., 2.)],
    ['Be', ( 0., 0., 0.)],]
mol.basis = 'sto3g'
mol.build()
#mol.verbose = 4

##################################
# HF 
##################################
mf = scf.RHF(mol)
#mf.chkfile ='hf.chk'
mf.conv_tol = 1e-15
mf.kernel()

nmo = mol.nao
nocc = mol.nelectron // 2
nvir = nmo-nocc
#print('nmo, nocc, nvir =', nmo, nocc, nvir)

##################################
# FCI 
##################################
from pyscf import fci
fcisolver = fci.FCI(mf)
fcisolver.conv_tol = 1e-15
E_fci, fcivec = fcisolver.kernel()

from pyscf.cc.fci_index import fci_coeff 
coeff = fci_coeff(fcivec, nmo, nocc)

##################################
# externally corrected CCSD
##################################
from pyscf import cc 

#mycc = cc.CCSD(mf)
#mycc.conv_tol = 1e-15 
#mycc.max_cycle = 1000
#mycc.kernel()
#E_ccsd = mf.e_tot + mycc.e_corr

myeccc = cc.CCSD(mf, ecCCSD=True)
myeccc.conv_tol = 1e-15 
myeccc.verbose = 5 
myeccc.max_cycle = 1000
#myeccc.kernel(coeff, thresh=-1)
myeccc.kernel(coeff)
E_ec_ccsd_e0 = mf.e_tot + myeccc.e_corr

print ('----------------------------------------------')
print (' RHF       total energy = ',mf.e_tot)
#print (' CCSD      total energy = ',E_ccsd)
print (' ecCCSD 0  total energy = ',E_ec_ccsd_e0)
#print (' ecCCSD    total energy = ',E_ec_ccsd_eNone)
print (' FCI       total energy = ',E_fci)
print ('----------------------------------------------')


