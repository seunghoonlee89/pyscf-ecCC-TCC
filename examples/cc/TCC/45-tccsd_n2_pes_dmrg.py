#!/usr/bin/env python
#
# Author: Seunghoon Lee <seunghoonlee89@gmail.com>
#

'''
N2 PES of TCCSD and TCCSD(T).
'''

import pyscf
from pyscf import mcscf, dmrgscf
from pyscf.dmrgscf import settings
settings.BLOCKEXE = '/home/slee89/opt/TCC/StackBlock_sampling_determinants/block.spin_adapted' 
settings.BLOCKSCRATCHDIR = './' 
settings.MPIPREFIX = ''
import os
#os.environ["OMP_NUM_THREADS"] = '28' 

def n2(atom):
    mol = pyscf.M(
        atom = atom,
        basis = 'ccpvdz')
    mf = mol.RHF().run()
    
    ##################################
    # variational DMRG wave function #
    ##################################
    no_cas = 6
    ne_cas = 6
    bond_dim = 500
    mc = mcscf.CASCI(mf, no_cas, ne_cas)
    mc.fcisolver = dmrgscf.DMRGCI(mol,maxM= bond_dim )
    mc.fcisolver.twopdm = False 
    mc.fcisolver.block_extra_keyword.append('num_thrds 28')
    mc.fcisolver.tol = 1e-9
    reorder = 'noreorder'
    #reorder = 'gaopt default'
    mc.fcisolver.block_extra_keyword.append(reorder)
    mc.canonicalization = False
    E_dmrg = mc.casci()[0]
    
    #############################################
    # extracting CI coeff by StackBlock & Block #
    #############################################
    from tccsd_dmrg_n2 import extracting_CI_coeff_from_MPS
    scal = 0.1
    sweeps    = mc.fcisolver.scheduleSweeps
    bond_dims = mc.fcisolver.scheduleMaxMs
    noises    = mc.fcisolver.scheduleNoises
    extracting_CI_coeff_from_MPS(ne_cas, bond_dim, scal, reorder, sweeps, bond_dims, noises)
   
    #############################
    # externally corrected CCSD #
    #############################
    from pyscf import cc 
    mytcc = cc.CCSD(mf, TCCSD=True)
    mytcc.verbose          = 5 
    mytcc.max_memory       = 10000  # 10 g
    mytcc.max_cycle        = 1000
    mytcc.conv_tol         = 1e-6
    mytcc.diis             = False     # deactivate diis 
    mytcc.level_shift      = 0.3
    #mytcc.iterative_damping= 0.8
    mytcc.kernel(mc, ext_source="DMRG")
    E_TCCSD = mytcc.e_tot
    
    #et = mytcc.ccsd_t()
    #E_TCCSD_t = E_TCCSD+et

    return E_TCCSD

if __name__ == '__main__':
    bohr2ang = 0.529177249
    bonds= [1.5, 1.7, 1.9, 2.0, 2.118, 2.25, 2.4, 2.7, 3.0, 3.6, 4.2, 4.8, 5.4, 6.0, 8.0, 10.0] 
    atom = ['''
             N     0.000000   0.000000    %10.6f 
             N     0.000000   0.000000    %10.6f 
            '''%(-b*bohr2ang/2.0, b*bohr2ang/2.0) for b in bonds]

    etcc = []
    for i in range(len(bonds)):
        etcc.append(n2(atom[i]))

    for i in range(len(bonds)):
        print(bonds[i], etcc[i])
 
