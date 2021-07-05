#!/usr/bin/env python
#
# Author: Seunghoon Lee <seunghoonlee89@gmail.com>
#

'''
A simple example to run TCCSD and TCCSD(T) calculation.
'''

import pyscf

mol = pyscf.M(
    atom = 'N 0 0 0; N 0 0 2.0',
    basis = 'ccpvdz')
mf = mol.RHF().run()

##################################
# variational DMRG wave function #
##################################
no_cas = 6
ne_cas = 6

bond_dim = 500
from pyscf import mcscf, dmrgscf
from pyscf.dmrgscf import settings
settings.BLOCKEXE = '/home/slee89/opt/TCC/StackBlock_sampling_determinants/block.spin_adapted' 
settings.BLOCKSCRATCHDIR = './' 
settings.MPIPREFIX = ''
#os.environ["OMP_NUM_THREADS"] = '1' 

mc = mcscf.CASCI(mf, no_cas, ne_cas)
mc.fcisolver = dmrgscf.DMRGCI(mol,maxM= bond_dim )
#mc.fcisolver.hf_occ = 'canonical'
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
#mytcc.verbose          = 5 
#mytcc.max_memory       = 1000  # 10 g
mytcc.max_cycle        = 1000
mytcc.conv_tol         = 1e-6
mytcc.diis             = False     # deactivate diis 
mytcc.level_shift      = 0.3
#mytcc.iterative_damping= 0.8
mytcc.kernel(mc, ext_source="DMRG")
E_TCCSD_HCI = mytcc.e_tot

et = mytcc.ccsd_t()
E_TCCSD_t_HCI = E_TCCSD_HCI+et 
