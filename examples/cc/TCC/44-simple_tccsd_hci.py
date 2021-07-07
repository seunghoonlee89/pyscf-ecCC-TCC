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

#################################
# variational HCI wave function #
#################################
no_cas = 6
ne_cas = 6

from pyscf import mcscf
from pyscf.cornell_shci import shci, settings
settings.SHCIEXE = '/home/slee89/opt/shci/shci'
# Arrow
mc = mcscf.CASCI(mf, no_cas, ne_cas)
mc.fcisolver = shci.SHCI(mf.mol, tol=1e-9) 
mc.fcisolver.config['var_only']     = True 
mc.fcisolver.config['s2']           = True
mc.fcisolver.config['eps_vars']     = [1e-5]
E_HCI = mc.kernel()[0]

#############################
# externally corrected CCSD #
#############################
from pyscf import cc 
mytcc = cc.CCSD(mf, TCCSD=True)
#mytcc.verbose          = 5 
mytcc.max_memory       = 10000  # 10 g
mytcc.max_cycle        = 1000
mytcc.conv_tol         = 1e-6
mytcc.diis             = False     # deactivate diis 
mytcc.level_shift      = 0.3
#mytcc.iterative_damping= 0.8
mytcc.kernel(mc, ext_source="SHCI")
E_TCCSD_HCI = mytcc.e_tot

et = mytcc.ccsd_t()
E_TCCSD_t_HCI = E_TCCSD_HCI+et 
