#!/usr/bin/env python
#
# Author: Seunghoon Lee <seunghoonlee89@gmail.com>
#

'''
An example to run externally corrected (ec)-CCSD calculation using FCI.
'''

from pyscf import gto, scf
import numpy 
import pyscf 

mol = gto.Mole()
#mol.atom = [
#    ['O', ( 0.00000000, 0.0, -0.12427075)],
#    ['H', (-1.60402215, 0.0,  0.98640722)],
#    ['H', ( 1.60402215, 0.0,  0.98640722)],]      # R_OH = 2.0 R_e
#mol.basis = '3-21g'
#mol.basis = '6-31g'

#mol = pyscf.M(
#    atom = 'H 0 0 0; F 0 0 1.1',
#    basis = 'ccpvdz')

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
print('nmo, nocc, nvir =', nmo, nocc, nvir)

##################################
# SHCI 
##################################
# Gen FCIDUMP using stackblock 
from pyscf import mcscf
from pyscf import dmrgscf
mc = dmrgscf.DMRGSCF(mf, nmo, 2*nocc)
dmrgscf.dryrun(mc)

import os
import sys 
os.system('rm integrals_cache.dat')
os.system('rm wf_eps1*')
#os.system('cp config.json.h2o config.json')
os.system('cp config.json.be2 config.json')

eps = sys.argv[1]
eps1="{:e}".format(float(eps)*1000)
eps2="{:e}".format(float(eps)*100)
eps3="{:e}".format(float(eps)*10)
eps4="{:e}".format(float(eps))
eps1s="{:e}".format(float(eps)*5000)
eps2s="{:e}".format(float(eps)*500)
eps3s="{:e}".format(float(eps)*50)
eps4s="{:e}".format(float(eps)*5)

eps_vars = "[%s, %s, %s, %s]"%(eps1, eps2, eps3, eps4)
eps_vars_schedule = "[%s, %s, %s, %s]"%(eps1s, eps2s, eps3s, eps4s)

cmd1="find config.json -type f -exec sed -i 's/\[ 1e-5, 1e-7, 1e-8, 1e-9 \]/%s/g' {} \;"%(eps_vars)
cmd2="find config.json -type f -exec sed -i 's/\[ 5e-5, 5e-7, 5e-8, 5e-9 \]/%s/g' {} \;"%(eps_vars_schedule)
os.system(cmd1)
os.system(cmd2)

os.system('mpirun -n 5 shci > out')
os.system('get_CIcoef_SHCI.sh out > CIcoeff_shci.out')

from pyscf.cc.fci_index import fci_index
idx   = fci_index(nocc, nvir) 
idx.get_S()
idx.get_D()
idx.get_T()
idx.get_Q()

from pyscf.cc.shci import shci_coeff 
coeff_shci = shci_coeff("CIcoeff_shci.out", nocc, nvir, idx)
coeff_shci.get_All()
E_shci = coeff_shci.E_shci

#coeff_shci.interm_norm(Q=True)

##################################
# externally corrected CCSD
##################################
from pyscf import cc 

mycc = cc.CCSD(mf)
mycc.conv_tol = 1e-15 
mycc.max_cycle = 1000
mycc.kernel()
E_ccsd = mf.e_tot + mycc.e_corr

et = mycc.ccsd_t()
E_ccsd_t = E_ccsd + et  

myeccc = cc.CCSD(mf, ecCCSD=True)
#myeccc.conv_tol = 1e-5 
#myeccc.conv_tol_normt = 1e-3
myeccc.verbose = 5 
myeccc.max_cycle = 1000
#myeccc.iterative_dampling = 0.5
#myeccc.kernel(coeff, thresh=-1)
#myeccc.max_memory = 100000  # 100 g
myeccc.kernel(coeff_shci)
E_ec_ccsd_e0 = mf.e_tot + myeccc.e_corr

print ('----------------------------------------------')
print (' RHF       total energy = ',mf.e_tot)
print (' CCSD      total energy = ',E_ccsd)
print (' CCSD(T)   total energy = ',E_ccsd_t)
print (' ecCCSD    total energy = ',E_ec_ccsd_e0)
#print (' ecCCSD    total energy = ',E_ec_ccsd_eNone)
print (' SHCI      total energy = ',E_shci)
print ('----------------------------------------------')

