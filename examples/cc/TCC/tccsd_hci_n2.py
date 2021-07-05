#!/usr/bin/env python
#
# Author: Seunghoon Lee <seunghoonlee89@gmail.com>
#

'''
An example to run tailored CCSD calculation using HCI wave function.
'''

bohr2ang = 0.529177249

import os, sys, time
import struct
import tempfile
from subprocess import check_call, CalledProcessError
import numpy

from pyscf import gto, scf
from pyscf import mcscf, dmrgscf
#from pyscf.dmrgscf import settings
from pyscf.cornell_shci import settings
#from pyscf.shci import settings

import pyscf.tools
import pyscf.lib
from pyscf.lib import logger, chkfile
from pyscf.tools.fcidump import *

pauling050 = False
paulingnode= False
pauling050 = True 
#paulingnode= True 

if pauling050: 
    #settings.BLOCKSCRATCHDIR = './' 
    settings.SHCIRUNTIMEDIR = './' 
    settings.MPIPREFIX = 'mpirun -n 1 '
    os.environ["OMP_NUM_THREADS"] = '1' 
elif paulingnode: 
    #settings.BLOCKSCRATCHDIR = '%s'%os.environ['SCRATCHDIR']
    settings.SHCIRUNTIMEDIR = '%s'%os.environ['SCRATCHDIR']
    settings.MPIPREFIX = 'srun '

#settings.BLOCKEXE = '/home/slee89/opt/stackblocklatest/stackblock_stopt/TRIEdeterminant/block.spin_adapted' 
settings.SHCIEXE = '/home/slee89/opt/shci/shci'
#settings.SHCIEXE = '/home/slee89/opt/shci_Dice/Dice'

def run_TCCSD_t(bond, atom, basis, symmetry, norb_cas, nele_cas, norb_corr, nele_corr, frozen_hole, frozen_ptcl, eps):
    assert nele_corr >= nele_cas and norb_corr >= norb_cas 
    mol = gto.Mole()
    mol.build(
    verbose = 5,
    symmetry = symmetry,
    basis = basis,
    #output = 'bzn.%do.%de.out'%(norb_cas, nele_cas),
    atom = atom) 

    nmo       = mol.nao
    nelec     = mol.nelectron
    nocc      = nelec // 2
    nvir      = nmo-nocc
    print('nmo, nocc, nvir =', nmo, nocc, nvir)
    assert nmo  == frozen_hole + norb_corr + frozen_ptcl
    nocc_corr = nele_corr // 2
    nvir_corr = norb_corr - nocc_corr
    print('norb_corr, nocc_corr, nvir_corr =', norb_corr, nocc_corr, nvir_corr)
    assert nocc == frozen_hole + nocc_corr 
    assert nvir == frozen_ptcl + nvir_corr
    nocc_cas  = nele_cas // 2
    nvir_cas  = norb_cas  - nocc_cas 
    print('norb_cas , nocc_cas , nvir_cas  =', norb_cas , nocc_cas , nvir_cas)

    fhole = [ i for i in range(frozen_hole) ]
    fptcl = [ nmo - (i+1) for i in range(frozen_ptcl) ]
    frozen= fhole + fptcl
    if len(frozen) > 1: print ('frozen orbs =', frozen)
    
    ######
    # HF # 
    ######
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    #mf.chkfile = 'hf.chk'
    mf.level_shift = 0.4
    mf.kernel()
    E_HF = mf.e_tot
    #print('mo energy:',mf.mo_energy)

    # obtaining CAS MO
    mc = mcscf.CASSCF(mf, 6, 6)
    mc.fix_spin(shift=.2, ss=0)
    mc.chkfile= 'mc.chk'
    mc.kernel()

    # Dice
    import h5py
    with h5py.File('mc.chk','r') as f :
        mo = f['mcscf/mo_coeff'].value
 
    #################################
    # variational HCI wave function #
    #################################
    # Arrow
    mc = mcscf.CASCI(mf, norb_cas, nele_cas)
    from pyscf.cornell_shci import shci
    mc.fcisolver = shci.SHCI(mf.mol, tol=1e-9) 
    mc.mo_coeff = mo
    mc.fcisolver.config['get_1rdm_csv'] = False 
    mc.fcisolver.config['get_2rdm_csv'] = False 
    mc.fcisolver.config['var_only']     = True 
    mc.fcisolver.config['s2']           = True
    mc.fcisolver.config['eps_vars']     = [eps]
    E_HCI = mc.kernel()[0]
#    # Dice
#    mc = mcscf.CASCI(mf, norb_cas, nele_cas)
#    from pyscf.shci import shci     
#    mc.fcisolver = shci.SHCI(mf.mol, tol=tol) 
#    mc.fcisolver.stochastic    = False 
#    mc.fcisolver.nPTiter       = 0 
#    mc.fcisolver.sweep_iter    = [0]
#    mc.fcisolver.sweep_epsilon = [eps]
#    E_HCI = mc.kernel()[0]

    ifCC = False 
    if ifCC:
        ###################
        # CCSD (optional) #
        ###################
        from pyscf import cc 
        mycc = cc.CCSD(mf)
        mycc.conv_tol    = 1e-12
        mycc.max_cycle = 1000
        if len(frozen)>1: mycc.frozen = frozen 
        mycc.level_shift = 0.3
        mycc.kernel()
        E_CCSD = mf.e_tot + mycc.e_corr
        et = mycc.ccsd_t()
        E_CCSDt = E_CCSD + et  
        et = mycc.rccsd_t()
        E_RCCSDt = E_CCSD + et  
    else:
        E_CCSD = 0.0 
        E_CCSDt = 0.0 
        E_RCCSDt = 0.0 

    #############################
    # externally corrected CCSD #
    #############################
    from pyscf import cc 
    mytcc = cc.CCSD(mf, TCCSD=True, mo_coeff=mo)
    if len(frozen)>1: mytcc.frozen = frozen 
    mytcc.verbose          = 5 
    mytcc.max_memory       = 10000  # 10 g
    mytcc.max_cycle        = 100
    mytcc.conv_tol         = 1e-9
    mytcc.diis             = True
    mytcc.level_shift      = 0.3
    #mytcc.iterative_damping= 0.01
    mytcc.kernel(nocc_corr, nvir_corr, nocc_cas, nvir_cas)
    E_TCCSD_HCI = mytcc.e_tot

    et = mytcc.ccsd_t()
    E_TCCSD_t_HCI = E_TCCSD_HCI+et 
    os.system("cp CIcoeff_shci.out CIcoeff_shci.%f.out"%(bond))
    os.system("cp output.dat output.%f.dat"%(bond))

    print (bond,'----------------------------------------------')
    print (bond,' RHF       = ',E_HF)
    print (bond,' CCSD      = ',E_CCSD)
    print (bond,' CCSD(T)   = ',E_CCSDt)
    print (bond,' RCCSD(T)  = ',E_RCCSDt)
    print (bond,' HCI       = ',E_HCI)
    print (bond,' TCCSD    = ',E_TCCSD_HCI)
    print (bond,' TCCSD(T) = ',E_TCCSD_t_HCI)
    print (bond,'----------------------------------------------')

    return E_HF, E_HCI, E_CCSD, E_CCSDt, E_RCCSDt, E_TCCSD_HCI, E_TCCSD_t_HCI

if __name__ == '__main__':
#    #be2
#    basis = 'sto-3g'
#    symmetry = 'c1'
#    frozen_hole = 0    # # of frozen occupied orbitals
#    frozen_ptcl = 0    # # of frozen virtual  orbitals
#    norb_cas = 10      # # of orbitals  in CAS
#    nele_cas = 8       # # of electrons in CAS
#    norb_corr= 10      # # of orbitals  in active + inactive space 
#    nele_corr= 8       # # of electrons in active + inactive space
#    eps = 10
#    #eps = 1e-9 
#    atom = '''
#            Be     0.000000   0.000000    2.000000
#            Be     0.000000   0.000000    0.000000
#           '''
#    run_ecRCCSD_t(1.0, atom, basis, symmetry, norb_cas, nele_cas, norb_corr, nele_corr, frozen_hole, frozen_ptcl, eps)


#    #h2o
#    basis = '321g'
#    symmetry = 'c1'
#    frozen_hole = 0    # # of frozen occupied orbitals
#    frozen_ptcl = 0    # # of frozen virtual  orbitals
#    norb_cas = 12      # # of orbitals  in CAS
#    nele_cas = 10      # # of electrons in CAS
#    norb_corr= 13      # # of orbitals  in active + inactive space 
#    nele_corr= 10      # # of electrons in active + inactive space
#    eps = 1e-12
#    atom = '''
#            O     0.000000   0.000000    0.000000
#            H     0.000000  -0.857000    0.587000
#            H     0.000000   0.757000    0.687000
#           '''
#
#    run_tccSD_t(1.0, atom, basis, symmetry, norb_cas, nele_cas, norb_corr, nele_corr, frozen_hole, frozen_ptcl, eps)

    #n2
    basis = 'cc-pVTZ'
    symmetry = 'c1'
    frozen_hole = 0        # # of frozen occupied orbitals
    frozen_ptcl = 0        # # of frozen virtual  orbitals
    norb_cas = int(sys.argv[1]) # # of orbitals  in CAS
    nele_cas = int(sys.argv[2]) # # of electrons in CAS
    norb_corr= 60          # # of orbitals  in active + inactive space 
    nele_corr= 14          # # of electrons in active + inactive space
    eps = float(sys.argv[3])
    bonds= [1.5, 1.7, 1.9, 2.0, 2.118, 2.25, 2.4, 2.7, 3.0, 3.6, 4.2, 4.8, 5.4, 6.0] 
    #bonds= [6.0, 8.0, 10.0] 
    #bonds= [1.14] 
    atom = ['''
             N     0.000000   0.000000    %10.6f 
             N     0.000000   0.000000    %10.6f 
            '''%(-b*bohr2ang/2.0, b*bohr2ang/2.0) for b in bonds]

    eps_l = []
    Ehci_l = []
    ETCCSD_l = []
    ETCCSDt_l = []
    ECCSD_l = []
    ECCSDt_l = []
    ERCCSDt_l = []

    for i in range(len(bonds)):
        E_HF, E_HCI, E_CCSD, E_CCSDt, E_RCCSDt, E_TCCSD, E_TCCSDt = run_TCCSD_t(bonds[i], atom[i], basis, symmetry, norb_cas, nele_cas, norb_corr, nele_corr, frozen_hole, frozen_ptcl, eps)
        eps_l.append(eps)
        Ehci_l.append(E_HCI)
        ECCSD_l.append(E_CCSD)
        ECCSDt_l.append(E_CCSDt)
        ERCCSDt_l.append(E_RCCSDt)
        ETCCSD_l.append(E_TCCSD)
        ETCCSDt_l.append(E_TCCSDt)

    def print_summary_shci(eps_l, bondl_l, Ehci_l, ECCSD_l, ECCSDt_l, ERCCSDt_l, ETCCSD_l, ETCCSDt_l, num_iter):
        print("======================")
        print("Summary of calculation")
        print("======================")
        print("eps / bondleng / ECCSD / ECCSDt / ERCCSDt / Ehci / ETCCSD / ETCCSDt")
        for i in range(num_iter):
            print(eps_l[i], " ", bondl_l[i], " ", ECCSD_l[i], " ", ECCSDt_l[i], " ", ERCCSDt_l[i], " ", Ehci_l[i] , " ", ETCCSD_l[i], " ", ETCCSDt_l[i])

    print_summary_shci(eps_l, bonds, Ehci_l, ECCSD_l, ECCSDt_l, ERCCSDt_l, ETCCSD_l, ETCCSDt_l, len(bonds))

