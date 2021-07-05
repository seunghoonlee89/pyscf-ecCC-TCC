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

settings.BLOCKEXE = '/home/slee89/opt/stackblocklatest/stackblock_stopt/block.spin_adapted' 
#settings.SHCIEXE = '/home/slee89/opt/shci/shci'
#settings.SHCIEXE = '/home/slee89/opt/shci_Dice/Dice'

import subprocess
from subprocess import Popen, PIPE
def grep_cut(word, file, cut_range):
    p1 = Popen(["grep", "-i", word, file], stdout=PIPE)
    p2 = Popen(["cut", "-c", cut_range], stdin=p1.stdout, stdout=PIPE)
    output, errors = p2.communicate()
    return output.decode('ascii').strip() 

def extracting_CI_coeff_from_MPS(nelectron, bond_dim, scal, reorder, sweeps, bond_dims, noises, symmetry="c1"):
    CITRIEEXE1 = '/home/slee89/opt/stackblocklatest/stackblock_stopt/TRIEdeterminant/CITRIE'
    CITRIEEXE2 = '/home/slee89/opt/Block_pt_stored/TRIEdeterminant/MPS2CI'

    def get_largest_discarded_w(bond_dim, sweeps, bond_dims, noises):
       numzero = 1e-18
       w = grep_cut('Largest Discarded Weight =', 'dmrg.out', '57-71').split('\n')
       w = [float(i) for i in w]
       maxDw = 0.0
       for i in range(len(bond_dims)):
           if bond_dims[i] == bond_dim and maxDw < w[i] and noises[i] < numzero: maxDw = w[i] 
       if maxDw < numzero: maxDw = numzero
       return maxDw

    def Write_input_for_MPS_to_CI(nelectron, symmetry, err, reorder):
        filename='extractCI.conf'
        with open(filename, 'w') as fout:
            fout.write('%s %d\n'%('nelec', nelectron))
            fout.write('%s\n'%('hf_occ canonical'))
            fout.write('%s\n'%('spin 0'))
            fout.write('%s\n'%('orbitals FCIDUMP'))
            fout.write('%s %s\n'%('symmetry', symmetry))
            fout.write('%s\n'%('irrep 1'))
            fout.write('\n')
            fout.write('%s\n'%('schedule'))
            fout.write('%s\n'%('0 3000 1e-5 5e-5'))
            fout.write('%s\n'%('2 3000 1e-8 0.0'))
            fout.write('%s\n'%('end'))
            fout.write('\n')
            fout.write('%s\n'%(reorder))
            fout.write('%s\n'%('warmup local_2site'))
            fout.write('%s\n'%('maxiter 6'))
            fout.write('%s\n'%('onedot'))
            fout.write('%s\n'%('sweep_tol 1e-07'))
            fout.write('%s %s\n'%('prefix',dmrgscf.settings.BLOCKSCRATCHDIR))
            fout.write('%s\n'%('fullrestart'))
            fout.write('%s\n'%('sto_pt_restart'))
            fout.write('%s %15.12f\n'%('pt_tol', err))
            fout.write('\n')
            fout.write('%s\n'%('nroots 1'))
            fout.write('%s\n'%('outputlevel 3'))

        filename2='extractCI_init.conf'
        with open(filename2, 'w') as fout:
            fout.write('%s %d\n'%('nelec', nelectron))
            fout.write('%s\n'%('hf_occ canonical'))
            fout.write('%s\n'%('spin 0'))
            fout.write('%s\n'%('orbitals FCIDUMP'))
            fout.write('%s %s\n'%('symmetry', symmetry))
            fout.write('%s\n'%('irrep 1'))
            fout.write('\n')
            fout.write('%s\n'%('schedule'))
            fout.write('%s\n'%('0 5000 1e-5 5e-5'))
            fout.write('%s\n'%('2 5000 1e-8 0.0'))
            fout.write('%s\n'%('end'))
            fout.write('\n')
            fout.write('%s\n'%(reorder))
            fout.write('%s\n'%('warmup local_2site'))
            fout.write('%s\n'%('maxiter 6'))
            fout.write('%s\n'%('onedot'))
            fout.write('%s\n'%('sweep_tol 1e-07'))
            fout.write('%s %s\n'%('prefix',dmrgscf.settings.BLOCKSCRATCHDIR))
            fout.write('%s\n'%('fullrestart'))
            fout.write('%s\n'%('sto_pt_nsamples 0'))
            fout.write('%s\n'%('sto_pt_Hsamples 1'))
            fout.write('%s %15.12f\n'%('pt_tol', err))
            fout.write('\n')
            fout.write('%s\n'%('nroots 1'))
            fout.write('%s\n'%('outputlevel 3'))
            fout.write('%s\n'%('mem 20 g'))
   
    discW = get_largest_discarded_w(bond_dim, sweeps, bond_dims, noises)
    err = scal * numpy.sqrt(discW)
    #err = discW * scal
    print('discW, err =',discW, err)

    #err = 1e-10
    Write_input_for_MPS_to_CI(nelectron, symmetry, err, reorder)
    
    #TODO: mc.CITRIE(cutoff)
    #      would make input of CITRIE and run CITRIE
    cmd3="%s extractCI_init.conf > extractCI_init.out"%(CITRIEEXE1)
    cmd4="%s extractCI.conf > extractCI.out"%(CITRIEEXE2)
    os.system(cmd3)
    os.system(cmd4)
    os.system('get_CIcoef_DMRG.sh extractCI.out > CIcoeff_dmrg.out')

def run_ecRCCSD_t(bond, atom, basis, symmetry, norb_cas, nele_cas, norb_corr, nele_corr, frozen_hole, frozen_ptcl, bond_dim, scal):
    assert nele_corr >= nele_cas and norb_corr >= norb_cas 
    mol = gto.Mole()
    mol.build(
    #verbose = 5,
    #symmetry = symmetry,
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
    mf.conv_tol = 1e-9
    mf.chkfile = 'hf.chk'
    mf.level_shift = 0.4
    mf.kernel()
    E_HF = mf.e_tot
    #print('mo energy:',mf.mo_energy)
 
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
    E_dmrg = mc.casci()[0]

    #############################################
    # extracting CI coeff by StackBlock & Block #
    #############################################
    sweeps    = mc.fcisolver.scheduleSweeps
    bond_dims = mc.fcisolver.scheduleMaxMs
    noises    = mc.fcisolver.scheduleNoises
    extracting_CI_coeff_from_MPS(nele_cas, bond_dim, scal, reorder, sweeps, bond_dims, noises)

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
    myeccc = cc.CCSD(mf, ecCCSD=True)
    if len(frozen)>1: myeccc.frozen = frozen 
    myeccc.verbose          = 5 
    myeccc.max_memory       = 1000  # 10 g
    myeccc.max_cycle        = 5000
    myeccc.conv_tol         = 1e-6
    #myeccc.conv_tol_normt   = 1e-4
    myeccc.diis             = False 
    myeccc.level_shift      = 0.3
    myeccc.iterative_damping= 0.01
    myeccc.kernel(nocc_corr, nvir_corr, nocc_cas, nvir_cas, ext_source="DMRG")
    E_ecCCSD_dmrg = myeccc.e_tot

    #et = myeccc.ccsd_t(myeccc.coeff)
    et = myeccc.ccsd_t()
    E_ecCCSD_t_dmrg = E_ecCCSD_dmrg+et 

    et = myeccc.rccsd_t()
    E_ecRCCSD_t_dmrg = E_ecCCSD_dmrg+et 
    os.system("cp CIcoeff_dmrg.out CIcoeff_dmrg.%f.out"%(bond))
    os.system("cp hf.chk hf.%f.chk"%(bond))
    os.system("cp dmrg.out dmrg.%f.out"%(bond))

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
    norb_cas = int(sys.argv[1]) # # of orbitals  in CAS
    nele_cas = int(sys.argv[2]) # # of electrons in CAS
    norb_corr= 60          # # of orbitals  in active + inactive space 
    nele_corr= 14          # # of electrons in active + inactive space
    scal = 0.01
    bond_dim = float(sys.argv[3])
    #bonds= [1.5, 1.7, 1.9, 2.0, 2.118, 2.25, 2.4, 2.7, 3.0, 3.6, 4.2, 4.8, 5.4, 6.0] 
    #bonds= [1.0, 4.0] 
    bonds= [5.4] 
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

