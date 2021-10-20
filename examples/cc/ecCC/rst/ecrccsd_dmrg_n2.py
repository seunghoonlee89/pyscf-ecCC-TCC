#!/usr/bin/env python
#
# Author: Seunghoon Lee <seunghoonlee89@gmail.com>
#

'''
An example to run externally corrected (ec)-CCSD calculation using HCI wave function.
'''

bohr2ang = 0.529177210903
import os, sys, time
import struct
import tempfile
from subprocess import check_call, CalledProcessError
import numpy

from pyscf import gto, scf
from pyscf import mcscf, dmrgscf
from pyscf.dmrgscf import settings
#from pyscf.cornell_shci import settings
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
    settings.BLOCKSCRATCHDIR = './' 
    #settings.SHCIRUNTIMEDIR = './' 
    settings.MPIPREFIX = 'mpirun -n 1 '
    os.environ["OMP_NUM_THREADS"] = '1' 
elif paulingnode: 
    settings.BLOCKSCRATCHDIR = '%s'%os.environ['SCRATCHDIR']
    #settings.SHCIRUNTIMEDIR = '%s'%os.environ['SCRATCHDIR']
    settings.MPIPREFIX = 'srun '

settings.BLOCKEXE = '/home/slee89/opt/TCC/StackBlock_sampling_determinants/block.spin_adapted'
#settings.SHCIEXE = '/home/slee89/opt/shci/shci'
#settings.SHCIEXE = '/home/slee89/opt/shci_Dice/Dice'

def run_ecRCCSD_t(bond, atom, basis, symmetry, norb_cas, nele_cas, norb_corr, nele_corr, frozen_hole, frozen_ptcl, bond_dim, scal):
    assert nele_corr >= nele_cas and norb_corr >= norb_cas 
    mol = gto.Mole()
    mol.build(
    #verbose = 5,
    #symmetry = symmetry,
    basis = basis,
    output = 'n2.%f.out'%(bond),
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
    mf.conv_tol = 1e-9
#    mf.chkfile = 'hf.chk'
    mf.level_shift = 0.4
    mf.mo_coeff = scf.chkfile.load('hf.%f.chk'%(bond), 'scf/mo_coeff')
    mf.mo_occ = scf.chkfile.load('hf.%f.chk'%(bond), 'scf/mo_occ')
    E_HF = scf.chkfile.load('hf.%f.chk'%(bond), 'scf/e_tot')
 
    ##################################
    # variational DMRG wave function #
    ##################################
    # DMRG StackBlock
    mc = mcscf.CASCI(mf, norb_cas, nele_cas)
    mc.fcisolver = dmrgscf.DMRGCI(mol,maxM= bond_dim )
#    mc.fcisolver.scheduleSweeps = [0, 8]
#    mc.fcisolver.scheduleMaxMs  = [bond_dim,bond_dim] 
#    mc.fcisolver.scheduleTols   = [1e-5, 1e-15]
#    mc.fcisolver.scheduleNoises = [1e-4, 0.0]
#    mc.fcisolver.twodot_to_onedot = 10 
#    mc.fcisolver.maxIter = 20
    mc.fcisolver.hf_occ = 'canonical'
    mc.fcisolver.twopdm = False 
    mc.fcisolver.block_extra_keyword.append('num_thrds 28')
    mc.fcisolver.tol = 1e-14
    #mc.fcisolver.block_extra_keyword.append('fullrestart')
    mc.fcisolver.block_extra_keyword.append('mem 250 m')
    reorder = 'noreorder'
    #reorder = 'gaopt default'
    mc.fcisolver.block_extra_keyword.append(reorder)
    mc.canonicalization = False
    #E_dmrg = mc.casci()[0]

    #############################
    # externally corrected CCSD #
    #############################
    from pyscf import cc 
    myeccc = cc.CCSD(mf, ecCCSD=True)
    if len(frozen)>1: myeccc.frozen = frozen 
    myeccc.verbose          = 5 
    myeccc.max_memory       = 1000  # 10 g
    myeccc.max_cycle        = 5000
    myeccc.conv_tol         = 1e-6
    #myeccc.conv_tol_normt   = 1e-4
    myeccc.diis             = False 
    myeccc.level_shift      = 0.3
    myeccc.iterative_damping= 0.05
    # to restart
    os.system("cp t1.%f.npy t1.npy"%(bond))
    os.system("cp t2.%f.npy t2.npy"%(bond))
    os.system("cp t1_t3c.%f.npy t1_t3c.npy"%(bond))
    os.system("cp t2_t3t4c.%f.npy t2_t3t4c.npy"%(bond))
    myeccc.restart = 1  # load  

    myeccc.kernel(mc, ext_source="DMRG")
    E_ecCCSD_dmrg = myeccc.e_tot

#    #et = myeccc.ccsd_t(myeccc.coeff)
#    et = myeccc.ccsd_t()
#    E_ecCCSD_t_dmrg = E_ecCCSD_dmrg+et 
#
#    et = myeccc.rccsd_t()
#    E_ecRCCSD_t_dmrg = E_ecCCSD_dmrg+et 

    E_CCSD=.0
    E_CCSDt=.0
    E_RCCSDt=.0
    E_dmrg=.0
    E_ecCCSD_t_dmrg = 0.0 
    E_ecRCCSD_t_dmrg = 0.0
    print (bond,'----------------------------------------------')
    print (bond,' RHF       = ',E_HF)
    print (bond,' CCSD      = ',E_CCSD)
    print (bond,' CCSD(T)   = ',E_CCSDt)
    print (bond,' RCCSD(T)  = ',E_RCCSDt)
    print (bond,' DMRG      = ',E_dmrg)
    print (bond,' ecCCSD    = ',E_ecCCSD_dmrg)
    print (bond,' ecCCSD(T) = ',E_ecCCSD_t_dmrg)
    print (bond,' ecRCCSD(T)= ',E_ecRCCSD_t_dmrg)
    print (bond,'----------------------------------------------')

    return E_HF, E_dmrg, E_CCSD, E_CCSDt, E_RCCSDt, E_ecCCSD_dmrg, E_ecCCSD_t_dmrg, E_ecRCCSD_t_dmrg
    #return 0, 0, 0, 0, 0, 0

if __name__ == '__main__':
    #n2
    basis = 'cc-pVTZ'
    symmetry = 'c1'
    frozen_hole = 0        # # of frozen occupied orbitals
    frozen_ptcl = 0        # # of frozen virtual  orbitals
    norb_cas = 6 # # of orbitals  in CAS
    nele_cas = 6 # # of electrons in CAS
    norb_corr= 60          # # of orbitals  in active + inactive space 
    nele_corr= 14          # # of electrons in active + inactive space
    scal = 0.01
    bond_dim = 2000 
    #bonds= [1.5, 1.7, 1.9, 2.0, 2.118, 2.25, 2.4, 2.7, 3.0, 3.6, 4.2, 4.8, 5.4, 6.0] 
    #bonds= [1.0, 4.0] 
    bonds= [8.0] 
    #bonds= [2.25, 6.0] 
    atom = ['''
             N     0.000000   0.000000    %10.6f 
             N     0.000000   0.000000    %10.6f 
            '''%(-b*bohr2ang/2.0, b*bohr2ang/2.0) for b in bonds]

    bond_dim_l = []
    Edmrg_l = []
    EecCCSD_l = []
    EecCCSDt_l = []
    EecRCCSDt_l = []
    ECCSD_l = []
    ECCSDt_l = []
    ERCCSDt_l = []

    for i in range(len(bonds)):
        E_HF, E_dmrg, E_CCSD, E_CCSDt, E_RCCSDt, E_ecCCSD, E_ecCCSDt, E_ecRCCSDt = run_ecRCCSD_t(bonds[i], atom[i], basis, symmetry, norb_cas, nele_cas, norb_corr, nele_corr, frozen_hole, frozen_ptcl, bond_dim, scal)
        bond_dim_l.append(bond_dim)
        Edmrg_l.append(E_dmrg)
        ECCSD_l.append(E_CCSD)
        ECCSDt_l.append(E_CCSDt)
        ERCCSDt_l.append(E_RCCSDt)
        EecCCSD_l.append(E_ecCCSD)
        EecCCSDt_l.append(E_ecCCSDt)
        EecRCCSDt_l.append(E_ecRCCSDt)

    def print_summary_dmrg(bond_dim_l, bondl_l, Edmrg_l, ECCSD_l, ECCSDt_l, ERCCSDt_l, EecCCSD_l, EecCCSDt_l, EecRCCSDt_l, num_iter):
        print("======================")
        print("Summary of calculation")
        print("======================")
        print("bond_dim / bondleng / ECCSD / ECCSDt / ERCCSDt / Edmrg / EecCCSD / EecCCSDt / EecRCCSDt")
        for i in range(num_iter):
            print(bond_dim_l[i], " ", bondl_l[i], " ", ECCSD_l[i], " ", ECCSDt_l[i], " ", ERCCSDt_l[i], " ", Edmrg_l[i] , " ", EecCCSD_l[i], " ", EecCCSDt_l[i], " ", EecRCCSDt_l[i])

    print_summary_dmrg(bond_dim_l, bonds, Edmrg_l, ECCSD_l, ECCSDt_l, ERCCSDt_l, EecCCSD_l, EecCCSDt_l, EecRCCSDt_l, len(bonds))

