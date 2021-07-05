#!/usr/bin/env python
#
# Author: Seunghoon Lee <seunghoonlee89@gmail.com>
#

'''
A simple example to run TCCSD and TCCSD(T) calculation.
'''

import pyscf

def n2(atom):
    mol = pyscf.M(
        atom = atom,
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
    mytcc.verbose          = 5 
    mytcc.max_memory       = 10000  # 10 g
    mytcc.max_cycle        = 1000
    mytcc.conv_tol         = 1e-6
    mytcc.diis             = False     # deactivate diis 
    mytcc.level_shift      = 0.3
    #mytcc.iterative_damping= 0.8
    mytcc.kernel(mc, ext_source="SHCI")
    E_TCCSD_HCI = mytcc.e_tot
    
    #et = mytcc.ccsd_t()
    #E_TCCSD_t_HCI = E_TCCSD_HCI+et

    return E_TCCSD_HCI

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
 
