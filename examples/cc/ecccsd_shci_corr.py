#!/usr/bin/env python
#
# Author: Seunghoon Lee <seunghoonlee89@gmail.com>
#

'''
An example to run externally corrected (ec)-CCSD calculation using HCI wave function.
'''

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
    settings.MPIPREFIX = 'mpirun -n 2 '
    os.environ["OMP_NUM_THREADS"] = '2' 
elif paulingnode: 
    #settings.BLOCKSCRATCHDIR = '%s'%os.environ['SCRATCHDIR']
    settings.SHCIRUNTIMEDIR = '%s'%os.environ['SCRATCHDIR']
    settings.MPIPREFIX = 'srun '

#settings.BLOCKEXE = '/home/slee89/opt/stackblocklatest/stackblock_stopt/TRIEdeterminant/block.spin_adapted' 
settings.SHCIEXE = '/home/slee89/opt/shci/shci'
#settings.SHCIEXE = '/home/slee89/opt/shci_Dice/Dice'

def run_ecCCSD_t(atom, basis, symmetry, norb_cas, nele_cas, norb_corr, nele_corr, frozen_hole, frozen_ptcl, eps):
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
    nocc_iact = nocc_corr - nocc_cas
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
    #mf.level_shift = 0.4
    mf.kernel()
    E_HF = mf.e_tot
    #print('mo energy:',mf.mo_energy)
 
    #################################
    # variational HCI wave function #
    #################################
    # Arrow
    mc = mcscf.CASCI(mf, norb_cas, nele_cas)
    from pyscf.cornell_shci import shci
    mc.fcisolver = shci.SHCI(mf.mol, tol=1e-12) 
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
        #mycc.conv_tol    = 1e-12
        mycc.conv_tol    = 1e-7
        mycc.max_cycle = 500
        if len(frozen)>1: mycc.frozen = frozen 
        mycc.level_shift = 0.3
        mycc.kernel()
        E_CCSD = mf.e_tot + mycc.e_corr
        et = mycc.ccsd_t()
        E_ccsd_t = E_CCSD + et  
    else:
        E_CCSD = 0.0

    #############################
    # externally corrected CCSD #
    #############################
    from pyscf import cc 
    myeccc = cc.CCSD(mf, ecCCSD=True)
    if len(frozen)>1: myeccc.frozen = frozen 
    myeccc.verbose          = 5 
    myeccc.max_memory       = 10000  # 10 g
    myeccc.max_cycle        = 500
    #myeccc.conv_tol         = 1e-12
    myeccc.conv_tol         = 1e-7
    #myeccc.diis             = True 
    #myeccc.level_shift      = 0.3
    #myeccc.iterative_damping= 0.01
    myeccc.kernel(nocc_corr, nvir_corr, nocc_cas, nvir_cas, nocc_iact)
    E_ecCCSD_HCI = myeccc.e_tot

    et = myeccc.ccsd_t()
    E_ecCCSD_t_HCI = E_ecCCSD_HCI+et 
    #E_ecCCSD_t_HCI = 0.0 

    return E_HF, E_HCI, E_CCSD, E_ecCCSD_HCI, E_ecCCSD_t_HCI

if __name__ == '__main__':
    #be2
    basis = 'sto-3g'
    symmetry = 'c1'
    frozen_hole = 0    # # of frozen occupied orbitals
    frozen_ptcl = 0    # # of frozen virtual  orbitals
    norb_cas = 10      # # of orbitals  in CAS
    nele_cas = 8       # # of electrons in CAS
    norb_corr= 10      # # of orbitals  in active + inactive space 
    nele_corr= 8       # # of electrons in active + inactive space
    eps = 1e-12
    #eps = 1 
    atom = '''
            Be     0.000000   0.000000    2.000000
            Be     0.000000   0.000000    0.000000
           '''
#    #h2o
#    basis = '321g'
#    symmetry = 'c1'
#    frozen_hole = 0    # # of frozen occupied orbitals
#    frozen_ptcl = 0    # # of frozen virtual  orbitals
#    norb_cas = 13      # # of orbitals  in CAS
#    nele_cas = 10      # # of electrons in CAS
#    norb_corr= 13      # # of orbitals  in active + inactive space 
#    nele_corr= 10      # # of electrons in active + inactive space
#    #eps = 1e-12
#    eps = 1
#    atom = '''
#            O     0.000000   0.000000    0.000000
#            H     0.000000  -0.857000    0.587000
#            H     0.000000   0.757000    0.687000
#           '''
#    #benzene
#    basis = 'cc-pvdz'
#    symmetry = 'd6h'
#    frozen_hole = 6     # use frozen core as Ne for each Cr 
#    frozen_ptcl = 0 
#    norb_corr = 108         # active space for CAS
#    nele_corr = 30          # electrons    for CAS
#    norb_cas  = 24          # active space for reference space of TCC
#    nele_cas  = 24          # electrons    for reference space of TCC
#    #eps = 1e-6
#    eps = 1 
#    atom = '''
#             C     0.000000   1.396792    0.000000
#             C     0.000000  -1.396792    0.000000
#             C     1.209657   0.698396    0.000000
#             C    -1.209657  -0.698396    0.000000
#             C    -1.209657   0.698396    0.000000
#             C     1.209657  -0.698396    0.000000
#             H     0.000000   2.484212    0.000000
#             H     2.151390   1.242106    0.000000
#             H    -2.151390  -1.242106    0.000000
#             H    -2.151390   1.242106    0.000000
#             H     2.151390  -1.242106    0.000000
#             H     0.000000  -2.484212    0.000000
#           '''

    t0 = time.time()
    E_HF, E_HCI, E_CCSD, E_ecCCSD, E_ecCCSD_t = run_ecCCSD_t(atom, basis, symmetry, norb_cas, nele_cas, norb_corr, nele_corr, frozen_hole, frozen_ptcl, eps)

    print ('----------------------------------------------')
    print (' RHF           = ',E_HF)
    print (' CCSD          = ',E_CCSD)
    print (' SHCI          = ',E_HCI)
    print (' ecCCSD        = ',E_ecCCSD)
    print (' ecCCSD(T)     = ',E_ecCCSD_t)
    print ('----------------------------------------------')

