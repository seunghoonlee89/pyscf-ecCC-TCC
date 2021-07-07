#!/usr/bin/env python
#
# Author: Seunghoon Lee <seunghoonlee89@gmail.com>
#

'''
An example to run externally corrected (ec)-CCSD calculation using HCI wave function.
'''

bohr2ang = 0.529177249

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

mem = 120

ifstoPT = False 
nsamp_stopt = 10000

if pauling050: 
    settings.BLOCKSCRATCHDIR = './' 
    #settings.SHCIRUNTIMEDIR = './' 
    settings.MPIPREFIX = 'mpirun -n 1 '
    #os.environ["OMP_NUM_THREADS"] = '1' 
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

def calculate_fock(h1e, eris, casdm1):
    fock = h1e.copy()
    fock += numpy.einsum('ij,pqij->pq',casdm1, eris)
    fock -= 0.5*numpy.einsum('ij,piqj->pq',casdm1, eris)
    return fock

def energy_from_rdm(rdm1,rdm2,h1e,h2e):
    e = 0.0
    e += numpy.einsum('ij,ij',rdm1,h1e)
    e += 0.5*numpy.einsum('ijkl,ijkl',rdm2,h2e)
    return e

def zero_integrals(output, h1e, h2e, nmo, nelec, nuc=0, ms=0, orbsym=[],
                   tol=1e-15):
    with open(output, 'w') as fout:
        write_head(fout, nmo, nelec, ms, orbsym)
        h1e = h1e.reshape(nmo,nmo)
        for i in range(nmo):
            if abs(h2e[i,j,k,l]) > tol:
                fout.write(' %.16g %4d %4d %4d %4d\n' %(h2e[i,i,i,i],i+1,i+1,i+1,i+1))
            if abs(h1e[i,i]) > tol:
                fout.write(' %.16g %4d %4d  0  0\n' % (h1e[i,i], i+1, i+1))
        fout.write(' %.16g  0  0  0  0\n' % nuc)

def WriteZeroHamiltonian(mc, filename='H0',state=0,setzero=True):
    h1e, energy_core = mc.get_h1eff()
    mo_cas = mc.mo_coeff[:,mc.ncore:mc.ncore+mc.ncas]
    eri_cas = mc.get_h2eff(mo_cas)
    eri_cas = pyscf.ao2mo.restore(1, eri_cas, mc.ncas)
    integralFile = os.path.join(mc.fcisolver.runtimeDir, filename)

    if mc.fcisolver.groupname is not None and mc.fcisolver.orbsym:
        orbsym = dmrg_sym.convert_orbsym(mc.fcisolver.groupname, mc.fcisolver.orbsym)
    else:
        orbsym = []
#    orbsym = [1] * mc.ncas

    e_cas= mc.e_cas 
    casdm1, casdm2 = mc.fcisolver.make_rdm12(mc.ci, mc.ncas, mc.nelecas)
    E= 0.0

    with open(filename, 'w') as fout:
        write_head(fout, mc.ncas, mc.nelecas[0]+mc.nelecas[1], ms=abs(mc.nelecas[0]-mc.nelecas[1]), orbsym=orbsym)
        for i in range(h1e.shape[0]):
            for j in range(i):
                fout.write(' %.16g %4d %4d %4d %4d\n' %(eri_cas[i,i,j,j],i+1,i+1,j+1,j+1))
                E += 0.5*casdm2[i,i,j,j]*eri_cas[i,i,j,j]
                E += 0.5*casdm2[j,j,i,i]*eri_cas[j,j,i,i]
                fout.write(' %.16g %4d %4d %4d %4d\n' %(eri_cas[i,j,j,i],i+1,j+1,j+1,i+1))
                E += 0.5*casdm2[i,j,j,i]*eri_cas[i,j,j,i]
                E += 0.5*casdm2[j,i,i,j]*eri_cas[j,i,i,j]
            fout.write(' %.16g %4d %4d %4d %4d\n' %(eri_cas[i,i,i,i],i+1,i+1,i+1,i+1))
            E += 0.5*casdm2[i,i,i,i]*eri_cas[i,i,i,i]
        fout.write(' %.16g %4d %4d %4d %4d\n' %(eri_cas[i,i,i,i],i+1,i+1,i+1,i+1))
        for i in range(h1e.shape[0]):
            fout.write(' %.16g %4d %4d %4d %4d\n' %(h1e[i,i],i+1,i+1,0,0))
            E += casdm1[i,i]*h1e[i,i]
        fout.write(' %.16g  0  0  0  0\n' %(-0.5*(E+e_cas)))
    #zero_integrals(integralFile, h1eff, eri_cas, mc.ncas,
    #                                   mc.nelecas[0]+mc.nelecas[1], ms=abs(mc.nelecas[0]-mc.nelecas[1]),
    #                                   nuc=-e_cas, orbsym=orbsym)

def WritePerturbationIntegral(mc, filename='H1',state=0,setzero=True):
    h1eff, energy_core = mc.get_h1eff()
    mo_cas = mc.mo_coeff[:,mc.ncore:mc.ncore+mc.ncas]
    eri_cas = mc.get_h2eff(mo_cas)
    eri_cas = pyscf.ao2mo.restore(1, eri_cas, mc.ncas)
    integralFile = os.path.join(mc.fcisolver.runtimeDir, filename)
    if mc.fcisolver.groupname is not None and mc.fcisolver.orbsym:
        orbsym = dmrg_sym.convert_orbsym(mc.fcisolver.groupname, mc.fcisolver.orbsym)
    else:
        orbsym = []
#    orbsym = [1] * mc.ncas

    h1e = h1eff

    eri_cas = pyscf.ao2mo.restore(1, eri_cas, mc.ncas)
    e_cas= mc.e_cas 
    with open(filename, 'w') as fout:
        write_head(fout, mc.ncas, mc.nelecas[0]+mc.nelecas[1], ms=abs(mc.nelecas[0]-mc.nelecas[1]), orbsym=orbsym)
        for i in range(h1e.shape[0]):
          for j in range(h1e.shape[0]):
            for k in range(h1e.shape[0]):
              for l in range(h1e.shape[0]):
                if(abs(eri_cas[i,j,k,l]) > 1e-15):
                  fout.write(' %.16g %4d %4d %4d %4d\n' %(eri_cas[i,j,k,l],i+1,j+1,k+1,l+1))
        write_hcore(fout, h1e, mc.ncas, tol=1e-15)
        fout.write(' %.16g  0  0  0  0\n' %(-1.0*e_cas))

def compress(mc, state, perturbfile):
    WritePerturbationIntegral(mc, filename=perturbfile, state=state)

def perturbation(mc, state, zerofile, perturbfile):
    compress(mc, state, perturbfile)
    mc2 = mc
    WriteZeroHamiltonian(mc2, filename=zerofile, state=state)

def extracting_CI_coeff_from_MPS(nelectron, bond_dim, scal, reorder, sweeps, bond_dims, noises, symmetry="c1"):
    CITRIEEXE1 = '/home/slee89/opt/TCC/StackBlock_sampling_determinants/CITRIE'
    CITRIEEXE2 = '/home/slee89/opt/TCC/Block_sampling_determinants/MPS2CI_ecCC'

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
            fout.write('mem %d g\n'%(mem))
   
    discW = get_largest_discarded_w(bond_dim, sweeps, bond_dims, noises)
    err = scal * numpy.sqrt(discW)
    #err = discW * scal
    #print('discW, err =',discW, err)

    #err = 1e-10
    Write_input_for_MPS_to_CI(nelectron, symmetry, err, reorder)
    
    #TODO: mc.CITRIE(cutoff)
    #      would make input of CITRIE and run CITRIE
    cmd3="%s extractCI_init.conf > extractCI_init.out"%(CITRIEEXE1)
    cmd4="%s extractCI.conf > extractCI.out"%(CITRIEEXE2)
    os.system(cmd3)
    os.system(cmd4)
    os.system('get_CIcoef_DMRG.sh extractCI.out > CIcoeff_dmrg.out')

def run_TCCSD_t(bond, atom, basis, symmetry, norb_cas, nele_cas, norb_corr, nele_corr, frozen_hole, frozen_ptcl, bond_dim, scal):
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
    print('norb_cas , nocc_cas , nvir_cas  =', norb_cas , nocc_cas , nvir_cas)

    fhole = [ i for i in range(frozen_hole) ]
    fptcl = [ nmo - (i+1) for i in range(frozen_ptcl) ]
    frozen= fhole + fptcl
    if len(frozen) > 1: print ('frozen orbs =', frozen)

    rst = False 
    #rst = True    
    ######
    # HF # 
    ######
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-9
    if not rst: mf.chkfile = 'hf.chk'
    mf.level_shift = 0.4
    mf.kernel()
    E_HF = mf.e_tot
    #print('mo energy:',mf.mo_energy)
 
    t0 = time.time()

    if rst:
        mol,mf_r = scf.chkfile.load_scf('hf.%f.chk'%(bond))
        mf.mo_coeff = mf_r["mo_coeff"]

        E_dmrg = 0.0 

        from pyscf.cc.fci_index import fci_index_nomem
        idx = fci_index_nomem(nocc_cas, nvir_cas) 
        from pyscf.cc.dmrg import dmrg_coeff 
        coeff = dmrg_coeff("CIcoeff_dmrg.%f.out"%(bond), nocc_cas, nvir_cas, idx)
    else:
        ##################################
        # variational DMRG wave function #
        ##################################
        # DMRG StackBlock
        mc = mcscf.CASCI(mf, norb_cas, nele_cas)
        mc.fcisolver = dmrgscf.DMRGCI(mol,maxM= bond_dim )
#        mc.fcisolver.scheduleSweeps = [0, 10]
#        mc.fcisolver.scheduleMaxMs  = [bond_dim,bond_dim] 
#        mc.fcisolver.scheduleTols   = [1e-5, 1e-9]
#        mc.fcisolver.scheduleNoises = [1e-4, 0.0]
#        mc.fcisolver.twodot_to_onedot = 12 
#        mc.fcisolver.maxIter = 20 
        mc.fcisolver.hf_occ = 'canonical'
        if not ifstoPT: mc.fcisolver.twopdm = False 
        mc.fcisolver.block_extra_keyword.append('num_thrds 28')
        mc.fcisolver.tol = 1e-9
        #mc.fcisolver.block_extra_keyword.append('fullrestart')
        mc.fcisolver.block_extra_keyword.append('mem %d g'%(mem))
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

        os.system("cp hf.chk hf.%f.chk"%(bond))
        os.system("cp CIcoeff_dmrg.out CIcoeff_dmrg.%f.out"%(bond))
        os.system("cp dmrg.out dmrg.%f.out"%(bond))

    t1 = time.time()
    print("time for DMRG (s):", t1 - t0 )

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
    mytcc = cc.CCSD(mf, TCCSD=True)
    if len(frozen)>1: mytcc.frozen = frozen 
    #mytcc.verbose          = 5 
    mytcc.max_memory       = 10000  # 10 g
    mytcc.max_cycle        = 1000
    mytcc.conv_tol         = 1e-9
    mytcc.diis             = True
    mytcc.level_shift      = 0.3
    #mytcc.iterative_damping= 0.01
    if rst: mytcc.kernel(nocc_corr, nvir_corr, nocc_cas, nvir_cas, ext_source="DMRG", coeff=coeff)
    else: mytcc.kernel(nocc_corr, nvir_corr, nocc_cas, nvir_cas, ext_source="DMRG")
    E_TCCSD = mytcc.e_tot

    t2 = time.time()
    print("time for TCCSD correction (s):", t2 - t1 )

    et = mytcc.ccsd_t()
    E_TCCSD_t = E_TCCSD+et 

    t3 = time.time()
    print("time for TCCSD(T) correction (s):", t3 - t2 )

    os.system("cp hf.chk hf.%f.chk"%(bond))
    os.system("cp CIcoeff_dmrg.out CIcoeff_dmrg.%f.out"%(bond))
    os.system("cp output.dat output.%f.dat"%(bond))

    print (bond,'----------------------------------------------')
    print (bond,' RHF       = ',E_HF)
    print (bond,' CCSD      = ',E_CCSD)
    print (bond,' CCSD(T)   = ',E_CCSDt)
    print (bond,' RCCSD(T)  = ',E_RCCSDt)
    print (bond,' DMRG      = ',E_dmrg)
    print (bond,' TCCSD    = ',E_TCCSD)
    print (bond,' TCCSD(T) = ',E_TCCSD_t)
    print (bond,'----------------------------------------------')

    return E_HF, E_dmrg, E_CCSD, E_CCSDt, E_RCCSDt, E_TCCSD, E_TCCSD_t

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

    #h2o
    basis = 'cc-pVTZ'
    symmetry = 'c1'
    frozen_hole = 0        # # of frozen occupied orbitals
    frozen_ptcl = 0        # # of frozen virtual  orbitals
    norb_cas = int(sys.argv[1]) # # of orbitals  in CAS
    nele_cas = int(sys.argv[2]) # # of electrons in CAS
    norb_corr= 60          # # of orbitals  in active + inactive space 
    nele_corr= 14          # # of electrons in active + inactive space
    bond_dim = float(sys.argv[3])
    scal = 0.1

    bohr2ang = 0.529177249
    bonds= [1.5, 1.7, 1.9, 2.0, 2.118, 2.25, 2.4, 2.7, 3.0, 3.6, 4.2, 4.8, 5.4, 6.0, 8.0, 10.0] 
    #bonds= [5.4] 
    #bonds= [1.0, 4.0] 
    #bonds= [1.14] 
    atom = ['''
             N     0.000000   0.000000    %10.6f 
             N     0.000000   0.000000    %10.6f 
            '''%(-b*bohr2ang/2.0, b*bohr2ang/2.0) for b in bonds]

    bond_dim_l = []
    Edmrg_l = []
    ETCCSD_l = []
    ETCCSDt_l = []
    ECCSD_l = []
    ECCSDt_l = []
    ERCCSDt_l = []

    for i in range(len(bonds)):
        E_HF, E_dmrg, E_CCSD, E_CCSDt, E_RCCSDt, E_TCCSD, E_TCCSDt = run_TCCSD_t(bonds[i], atom[i], basis, symmetry, norb_cas, nele_cas, norb_corr, nele_corr, frozen_hole, frozen_ptcl, bond_dim, scal)
        bond_dim_l.append(bond_dim)
        Edmrg_l.append(E_dmrg)
        ECCSD_l.append(E_CCSD)
        ECCSDt_l.append(E_CCSDt)
        ERCCSDt_l.append(E_RCCSDt)
        ETCCSD_l.append(E_TCCSD)
        ETCCSDt_l.append(E_TCCSDt)

    def print_summary_dmrg(bond_dim_l, bondl_l, Edmrg_l, ECCSD_l, ECCSDt_l, ERCCSDt_l, ETCCSD_l, ETCCSDt_l, num_iter):
        print("======================")
        print("Summary of calculation")
        print("======================")
        print("bond_dim / bondleng / ECCSD / ECCSDt / ERCCSDt / Edmrg / ETCCSD / ETCCSDt")
        for i in range(num_iter):
            print(bond_dim_l[i], " ", bondl_l[i], " ", ECCSD_l[i], " ", ECCSDt_l[i], " ", ERCCSDt_l[i], " ", Edmrg_l[i] , " ", ETCCSD_l[i], " ", ETCCSDt_l[i])



    print_summary_dmrg(bond_dim_l, bonds, Edmrg_l, ECCSD_l, ECCSDt_l, ERCCSDt_l, ETCCSD_l, ETCCSDt_l, len(bonds))

