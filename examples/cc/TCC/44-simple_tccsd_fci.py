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

###################################
# variational CASCI wave function #
###################################
no_cas = 6
ne_cas = 6

from pyscf import mcscf, fci
mc = mcscf.CASCI(mf, no_cas, ne_cas)
mc.fcisolver      = fci.direct_spin0.FCISolver(mol) 
mc.fcisolver.spin = 0 
mc.casci()

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
mytcc.kernel(mc, ext_source="FCI")
E_TCCSD_HCI = mytcc.e_tot

et = mytcc.ccsd_t()
E_TCCSD_t_HCI = E_TCCSD_HCI+et 
