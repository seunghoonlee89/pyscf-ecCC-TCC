#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
clone of UCCSD(T) code for ecRCCSD(T)
'''

import time
import ctypes
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.cc import _ccsd

def kernel(mycc, eris, coeff, t1=None, t2=None, verbose=logger.NOTE, ecTCCSD=False):
    cpu1 = cpu0 = (time.clock(), time.time())
    log = logger.new_logger(mycc, verbose)
    if t1 is None: t1 = mycc.t1
    if t2 is None: t2 = mycc.t2

    if ecTCCSD: raise NotImplementedError
    else: mycc.coeff.exclude_t_ecCCSDt()    # off diag Pmat cas 
    #else: mycc.coeff.get_Pmat_ccsdt_cas()    # full Pmat cas 
    #else: mycc.coeff.get_Pmat_ccsdt()    # full Pmat 

    nocc_cas = int(mycc.coeff.nocc_cas)
    nvir_cas = int(mycc.coeff.nvir_cas)
    nocc_iact= int(mycc.coeff.nocc_iact)
    nocc2 = int(nocc_cas*(nocc_cas-1)/2)
    nocc3 = int(nocc_cas*(nocc_cas-1)*(nocc_cas-2)/6 )

    t1a = t1
    t1b = t1
    t2ab= t2
    t2aa= t2 - t2.transpose(0,1,3,2)
    t2bb= t2aa

    nocca, nvira = t1.shape 
    noccb, nvirb = t1.shape
    nmoa = nocca + nvira 
    nmob = noccb + nvirb

    if mycc.incore_complete:
        ftmp = None
    else:
        ftmp = lib.H5TmpFile()
    t1aT = t1a.T.copy()
    t1bT = t1aT
    t2aaT = t2aa.transpose(2,3,0,1).copy()
    t2bbT = t2aaT

    eris_vooo = numpy.asarray(eris.ovoo).transpose(1,3,0,2).conj().copy()

    eris_vvop = _sort_eri(mycc, eris, ftmp, log)
    cpu1 = log.timer_debug1('ecCCSD(T) sort_eri', *cpu1)

    # denominator
    denom = 1.0
    denom+= 2.0*numpy.einsum('ia,ia', t1, t1)
    tmpt  = 2.0*t2 - t2.transpose(0,1,3,2)
    tmpc  = t2 + numpy.einsum('ia,jb->ijab', t1, t1) 
    denom+= numpy.einsum('ijab,ijab', tmpt, tmpc)
    denom*= 2.0

    dtype = numpy.result_type(t1a.dtype, t2aa.dtype, eris_vooo.dtype)
    et_sum = numpy.zeros(1, dtype=dtype)
    dn_sum = numpy.zeros(1, dtype=dtype)
    mem_now = lib.current_memory()[0]
    max_memory = max(0, mycc.max_memory - mem_now)
    # aaa
    bufsize = max(8, int((max_memory*.5e6/8-nocca**3*3*lib.num_threads())*.4/(nocca*nmoa)))
    log.debug('max_memory %d MB (%d MB in use)', max_memory, mem_now)
    orbsym = numpy.zeros(nocca, dtype=int)
    contract = _gen_contract_aaa(t1aT, t2aaT, eris_vooo, eris.fock,
                                 eris.mo_energy, orbsym, mycc.coeff.Paaa,
                                 nocc_iact, nvir_cas, nocc3, log)
    with lib.call_in_background(contract, sync=not mycc.async_io) as ctr:
        for a0, a1 in reversed(list(lib.prange_tril(0, nvira, bufsize))):
            cache_row_a = numpy.asarray(eris_vvop[a0:a1,:a1], order='C')
            if a0 == 0:
                cache_col_a = cache_row_a
            else:
                cache_col_a = numpy.asarray(eris_vvop[:a0,a0:a1], order='C')
            ctr(et_sum, dn_sum, a0, a1, a0, a1, (cache_row_a,cache_col_a,
                                         cache_row_a,cache_col_a))

            for b0, b1 in lib.prange_tril(0, a0, bufsize/8):
                cache_row_b = numpy.asarray(eris_vvop[b0:b1,:b1], order='C')
                if b0 == 0:
                    cache_col_b = cache_row_b
                else:
                    cache_col_b = numpy.asarray(eris_vvop[:b0,b0:b1], order='C')
                ctr(et_sum, dn_sum, a0, a1, b0, b1, (cache_row_a,cache_col_a,
                                             cache_row_b,cache_col_b))
    cpu1 = log.timer_debug1('contract_aaa', *cpu1)

    et_aaa = et_sum[0]*0.5
    dn_aaa = dn_sum[0]*0.5

    print('ecCCSD(T) aaa numerator   =',et_sum[0]*0.5)
    print('ecCCSD(T) aaa denominator =',dn_sum[0]*0.5)

    # Cache t2abT in t2ab to reduce memory footprint
    assert(t2ab.flags.c_contiguous)
    t2abT = lib.transpose(t2ab.copy().reshape(nocca*noccb,nvira*nvirb), out=t2ab)
    t2abT = t2abT.reshape(nvira,nvirb,nocca,noccb)
    # baa
    bufsize = int(max(12, (max_memory*.5e6/8-noccb*nocca**2*5)*.7/(nocca*nmob)))
    ts = t1aT, t1bT, t2aaT, t2abT
    fock = eris.fock
    vooo = eris_vooo
    contract = _gen_contract_baa(ts, vooo, fock, eris.mo_energy, orbsym, 
                                 mycc.coeff.Pbaa, nocc_cas, nvir_cas, nocc_iact, nocc2, log)
    with lib.call_in_background(contract, sync=not mycc.async_io) as ctr:
        for a0, a1 in lib.prange(0, nvirb, int(bufsize/nvira+1)):
            cache_row_a = numpy.asarray(eris_vvop[a0:a1,:], order='C')
            cache_col_a = numpy.asarray(eris_vvop[:,a0:a1], order='C')
            for b0, b1 in lib.prange_tril(0, nvira, bufsize/6/2):
                cache_row_b = numpy.asarray(eris_vvop[b0:b1,:b1], order='C')
                cache_col_b = numpy.asarray(eris_vvop[:b0,b0:b1], order='C')
                ctr(et_sum, a0, a1, b0, b1, (cache_row_a,cache_col_a,
                                             cache_row_b,cache_col_b))
    cpu1 = log.timer_debug1('contract_baa', *cpu1)

    print('ecCCSD(T) baa contribution =',0.5*et_sum[0]-et_aaa)

#    t2baT = numpy.ndarray((nvirb,nvira,noccb,nocca), buffer=t2abT,
#                          dtype=t2abT.dtype)
#    t2baT[:] = t2abT.copy().transpose(1,0,3,2)
#    # abb
#    ts = t1bT, t1aT, t2bbT, t2baT
#    fock = (eris.fockb, eris.focka)
#    mo_energy = (eris.mo_energy[1], eris.mo_energy[0])
#    vooo = (eris_VOOO, eris_VoOo, eris_vOoO)
#    contract = _gen_contract_baa(ts, vooo, fock, mo_energy, orbsym, log)
#    for a0, a1 in lib.prange(0, nvira, int(bufsize/nvirb+1)):
#        with lib.call_in_background(contract, sync=not mycc.async_io) as ctr:
#            cache_row_a = numpy.asarray(eris_vVoP[a0:a1,:], order='C')
#            cache_col_a = numpy.asarray(eris_VvOp[:,a0:a1], order='C')
#            for b0, b1 in lib.prange_tril(0, nvirb, bufsize/6/2):
#                cache_row_b = numpy.asarray(eris_VVOP[b0:b1,:b1], order='C')
#                cache_col_b = numpy.asarray(eris_VVOP[:b0,b0:b1], order='C')
#                ctr(et_sum, a0, a1, b0, b1, (cache_row_a,cache_col_a,
#                                             cache_row_b,cache_col_b))
#    cpu1 = log.timer_debug1('contract_abb', *cpu1)
#
#    # Restore t2ab
#    lib.transpose(t2baT.transpose(1,0,3,2).copy().reshape(nvira*nvirb,nocca*noccb),
#                  out=t2ab)
    et_sum *= .5
    if abs(et_sum[0].imag) > 1e-4:
        logger.warn(mycc, 'Non-zero imaginary part of ecCCSD(T) energy was found %s',
                    et_sum[0])
    et = et_sum[0].real
    mem_now = lib.current_memory()[0]
    log.debug('max_memory %d MB (%d MB in use)', max_memory, mem_now)
    log.timer('ecCCSD(T)', *cpu0)
    log.note('ecCCSD(T) correction = %.15g', et)
    return et

def _gen_contract_aaa(t1T, t2T, vooo, fock, mo_energy, orbsym, paaa, nocc_iact, nvir_cas, nocc3, log):
    nvir, nocc = t1T.shape
    mo_energy = numpy.asarray(mo_energy, order='C')
    fvo = fock[nocc:,:nocc].copy()

    cpu2 = [time.clock(), time.time()]
    orbsym = numpy.hstack((numpy.sort(orbsym[:nocc]),numpy.sort(orbsym[nocc:])))
    o_ir_loc = numpy.append(0, numpy.cumsum(numpy.bincount(orbsym[:nocc], minlength=8)))
    v_ir_loc = numpy.append(0, numpy.cumsum(numpy.bincount(orbsym[nocc:], minlength=8)))
    o_sym = orbsym[:nocc]
    oo_sym = (o_sym[:,None] ^ o_sym).ravel()
    oo_ir_loc = numpy.append(0, numpy.cumsum(numpy.bincount(oo_sym, minlength=8)))
    nirrep = max(oo_sym) + 1

    orbsym   = orbsym.astype(numpy.int32)
    o_ir_loc = o_ir_loc.astype(numpy.int32)
    v_ir_loc = v_ir_loc.astype(numpy.int32)
    oo_ir_loc = oo_ir_loc.astype(numpy.int32)
    dtype = numpy.result_type(t2T.dtype, vooo.dtype, fock.dtype)
    if dtype == numpy.complex:
        #drv = _ccsd.libcc.CCuccsd_t_zaaa
        raise NotImplementedError
    else:
        drv = _ccsd.libcc.CCecrccsd_t_aaa
    def contract(et_sum, dn_sum, a0, a1, b0, b1, cache):
        cache_row_a, cache_col_a, cache_row_b, cache_col_b = cache
        drv(et_sum.ctypes.data_as(ctypes.c_void_p),
            dn_sum.ctypes.data_as(ctypes.c_void_p),
            mo_energy.ctypes.data_as(ctypes.c_void_p),
            t1T.ctypes.data_as(ctypes.c_void_p),
            t2T.ctypes.data_as(ctypes.c_void_p),
            vooo.ctypes.data_as(ctypes.c_void_p),
            fvo.ctypes.data_as(ctypes.c_void_p),
            paaa.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nocc), ctypes.c_int(nvir),
            ctypes.c_int(nocc_iact), ctypes.c_int(nvir_cas),
            ctypes.c_int(nocc3),
            ctypes.c_int(a0), ctypes.c_int(a1),
            ctypes.c_int(b0), ctypes.c_int(b1),
            ctypes.c_int(nirrep),
            o_ir_loc.ctypes.data_as(ctypes.c_void_p),
            v_ir_loc.ctypes.data_as(ctypes.c_void_p),
            oo_ir_loc.ctypes.data_as(ctypes.c_void_p),
            orbsym.ctypes.data_as(ctypes.c_void_p),
            cache_row_a.ctypes.data_as(ctypes.c_void_p),
            cache_col_a.ctypes.data_as(ctypes.c_void_p),
            cache_row_b.ctypes.data_as(ctypes.c_void_p),
            cache_col_b.ctypes.data_as(ctypes.c_void_p))
        cpu2[:] = log.timer_debug1('contract %d:%d,%d:%d'%(a0,a1,b0,b1), *cpu2)
    return contract

def _gen_contract_baa(ts, vooo, fock, mo_energy, orbsym,
                      pbaa, nocc_cas, nvir_cas, nocc_iact, nocc2, log):
    t1aT, t1bT, t2aaT, t2abT = ts
    focka = fock
    fockb = fock
    vOoO  = vooo
    VoOo  = vooo
    nvira, nocca = t1aT.shape
    nvirb, noccb = t1bT.shape
    mo_ea = numpy.asarray(mo_energy, order='C')
    mo_eb = mo_ea 
    fvo = focka[nocca:,:nocca].copy()
    fVO = fockb[noccb:,:noccb].copy()

    cpu2 = [time.clock(), time.time()]
    dtype = numpy.result_type(t2aaT.dtype, vooo.dtype)
    if dtype == numpy.complex:
        raise NotImplementedError
        #drv = _ccsd.libcc.CCuccsd_t_zbaa
    else:
        #drv = _ccsd.libcc.CCuccsd_t_baa
        drv = _ccsd.libcc.CCecccsd_t_baa
    def contract(et_sum, a0, a1, b0, b1, cache):
        cache_row_a, cache_col_a, cache_row_b, cache_col_b = cache
#        drv(et_sum.ctypes.data_as(ctypes.c_void_p),
#            mo_ea.ctypes.data_as(ctypes.c_void_p),
#            mo_eb.ctypes.data_as(ctypes.c_void_p),
#            t1aT.ctypes.data_as(ctypes.c_void_p),
#            t1bT.ctypes.data_as(ctypes.c_void_p),
#            t2aaT.ctypes.data_as(ctypes.c_void_p),
#            t2abT.ctypes.data_as(ctypes.c_void_p),
#            vooo.ctypes.data_as(ctypes.c_void_p),
#            vOoO.ctypes.data_as(ctypes.c_void_p),
#            VoOo.ctypes.data_as(ctypes.c_void_p),
#            fvo.ctypes.data_as(ctypes.c_void_p),
#            fVO.ctypes.data_as(ctypes.c_void_p),
#            ctypes.c_int(nocca), ctypes.c_int(noccb),
#            ctypes.c_int(nvira), ctypes.c_int(nvirb),
#            ctypes.c_int(a0), ctypes.c_int(a1),
#            ctypes.c_int(b0), ctypes.c_int(b1),
#            cache_row_a.ctypes.data_as(ctypes.c_void_p),
#            cache_col_a.ctypes.data_as(ctypes.c_void_p),
#            cache_row_b.ctypes.data_as(ctypes.c_void_p),
#            cache_col_b.ctypes.data_as(ctypes.c_void_p))
        drv(et_sum.ctypes.data_as(ctypes.c_void_p),
            mo_ea.ctypes.data_as(ctypes.c_void_p),
            mo_eb.ctypes.data_as(ctypes.c_void_p),
            t1aT.ctypes.data_as(ctypes.c_void_p),
            t1bT.ctypes.data_as(ctypes.c_void_p),
            t2aaT.ctypes.data_as(ctypes.c_void_p),
            t2abT.ctypes.data_as(ctypes.c_void_p),
            vooo.ctypes.data_as(ctypes.c_void_p),
            vOoO.ctypes.data_as(ctypes.c_void_p),
            VoOo.ctypes.data_as(ctypes.c_void_p),
            fvo.ctypes.data_as(ctypes.c_void_p),
            fVO.ctypes.data_as(ctypes.c_void_p),
            pbaa.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nocca), ctypes.c_int(noccb),
            ctypes.c_int(nvira), ctypes.c_int(nvirb),
            ctypes.c_int(nocc_cas), ctypes.c_int(nvir_cas),
            ctypes.c_int(nocc_iact), ctypes.c_int(nocc2),
            ctypes.c_int(a0), ctypes.c_int(a1),
            ctypes.c_int(b0), ctypes.c_int(b1),
            cache_row_a.ctypes.data_as(ctypes.c_void_p),
            cache_col_a.ctypes.data_as(ctypes.c_void_p),
            cache_row_b.ctypes.data_as(ctypes.c_void_p),
            cache_col_b.ctypes.data_as(ctypes.c_void_p))
        cpu2[:] = log.timer_debug1('contract %d:%d,%d:%d'%(a0,a1,b0,b1), *cpu2)
    return contract

def _sort_eri(mycc, eris, h5tmp, log):
    cpu1 = (time.clock(), time.time())
    nocc = eris.nocc
    nmo  = eris.fock.shape[0]
    nvir = nmo - nocc

    if mycc.t2 is None:
        dtype = eris.ovov.dtype
    else:
        dtype = numpy.result_type(mycc.t2[0], eris.ovov.dtype)

    if mycc.incore_complete or h5tmp is None:
        eris_vvop = numpy.empty((nvir,nvir,nocc,nmo), dtype)
    else:
        eris_vvop = h5tmp.create_dataset('vvop', (nvir,nvir,nocc,nmo), dtype)

    max_memory = max(2000, mycc.max_memory - lib.current_memory()[0])
    max_memory = min(8000, max_memory*.9)

    blksize = min(nvir, max(16, int(max_memory*1e6/8/(nvir*nocc*nmo))))
    with lib.call_in_background(eris_vvop.__setitem__, sync=not mycc.async_io) as save:
        bufopv = numpy.empty((nocc,nmo,nvir), dtype=dtype)
        buf1 = numpy.empty_like(bufopv)
        for j0, j1 in lib.prange(0, nvir, blksize):
            ovov = numpy.asarray(eris.ovov[:,j0:j1])
            ovvv = eris.get_ovvv(slice(None), slice(j0,j1))
            for j in range(j0,j1):
                bufopv[:,:nocc,:] = ovov[:,j-j0].conj()
                bufopv[:,nocc:,:] = ovvv[:,j-j0].conj()
                save(j, bufopv.transpose(2,0,1))
                bufopv, buf1 = buf1, bufopv
            ovov = ovvv = None
            cpu1 = log.timer_debug1('transpose %d:%d'%(j0,j1), *cpu1)

    return eris_vvop


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf import cc

    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -.757 , .587)],
        [1 , (0. ,  .757 , .587)]]

    mol.basis = '631g'
    mol.build()
    rhf = scf.RHF(mol)
    rhf.conv_tol = 1e-14
    rhf.scf()
    mcc = cc.CCSD(rhf)
    mcc.conv_tol = 1e-12
    mcc.ccsd()
    t1a = t1b = mcc.t1
    t2ab = mcc.t2
    t2aa = t2bb = t2ab - t2ab.transpose(1,0,2,3)
    mycc = cc.UCCSD(scf.addons.convert_to_uhf(rhf))
    eris = mycc.ao2mo()
    e3a = kernel(mycc, eris, (t1a,t1b), (t2aa,t2ab,t2bb))
    print(e3a - -0.00099642337843278096)

    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -.757 , .587)],
        [1 , (0. ,  .757 , .587)]]
    mol.spin = 2
    mol.basis = '3-21g'
    mol.build()
    mf = scf.UHF(mol).run(conv_tol=1e-14)
    nao, nmo = mf.mo_coeff[0].shape
    numpy.random.seed(10)
    mf.mo_coeff = numpy.random.random((2,nao,nmo))

    numpy.random.seed(12)
    nocca, noccb = mol.nelec
    nmo = mf.mo_occ[0].size
    nvira = nmo - nocca
    nvirb = nmo - noccb
    t1a  = .1 * numpy.random.random((nocca,nvira))
    t1b  = .1 * numpy.random.random((noccb,nvirb))
    t2aa = .1 * numpy.random.random((nocca,nocca,nvira,nvira))
    t2aa = t2aa - t2aa.transpose(0,1,3,2)
    t2aa = t2aa - t2aa.transpose(1,0,2,3)
    t2bb = .1 * numpy.random.random((noccb,noccb,nvirb,nvirb))
    t2bb = t2bb - t2bb.transpose(0,1,3,2)
    t2bb = t2bb - t2bb.transpose(1,0,2,3)
    t2ab = .1 * numpy.random.random((nocca,noccb,nvira,nvirb))
    t1 = t1a, t1b
    t2 = t2aa, t2ab, t2bb
    mycc = cc.UCCSD(mf)
    eris = mycc.ao2mo(mf.mo_coeff)
    e3a = kernel(mycc, eris, [t1a,t1b], [t2aa, t2ab, t2bb])
    print(e3a - 9877.2780859693339)

    mycc = cc.GCCSD(scf.addons.convert_to_ghf(mf))
    eris = mycc.ao2mo()
    t1 = mycc.spatial2spin(t1, eris.orbspin)
    t2 = mycc.spatial2spin(t2, eris.orbspin)
    from pyscf.cc import gccsd_t_slow
    et = gccsd_t_slow.kernel(mycc, eris, t1, t2)
    print(et - 9877.2780859693339)

