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

import numpy
from pyscf import lib
from pyscf import ao2mo
from pyscf.lib import logger
from pyscf.cc import ccsd
import time

'''
ecCCSD(T)
'''

def kernel(mcc, eris, coeff, t1=None, t2=None, verbose=logger.NOTE, ecTCCSD=False):
    cput1 = (time.clock(), time.time())
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(mcc.stdout, verbose)

    if t1 is None or t2 is None:
        t1, t2 = mcc.t1, mcc.t2

    def p6(t):
        return (t + t.transpose(1,2,0,4,5,3) +
                t.transpose(2,0,1,5,3,4) + t.transpose(0,2,1,3,5,4) +
                t.transpose(2,1,0,5,4,3) + t.transpose(1,0,2,4,3,5))
    def r6(w):
        return (w + w.transpose(2,0,1,3,4,5) + w.transpose(1,2,0,3,4,5)
                - w.transpose(2,1,0,3,4,5) - w.transpose(0,2,1,3,4,5)
                - w.transpose(1,0,2,3,4,5))

    if ecTCCSD: coeff.get_Pmat_ectccsdt() 
    else: coeff.get_Pmat_ccsdt_slow() 

    t1a = t1
    t1b = t1
    t2ab= t2
    t2aa= t2 - t2.transpose(0,1,3,2)
    t2bb= t2aa
    nocca, nvira = t1.shape 
    noccb, nvirb = t1.shape
    nmoa = nocca + nvira 
    nmob = noccb + nvirb
    mo_ea = eris.fock.diagonal()
    mo_eb = eris.fock.diagonal()
    eia = mo_ea[:nocca,None] - mo_ea[nocca:]
    eIA = mo_eb[:noccb,None] - mo_eb[noccb:]
    fvo = eris.fock[nocca:,:nocca]
    fVO = eris.fock[noccb:,:noccb]

    #lsh test
    t_contrib = numpy.zeros((nocca,nocca,nocca,nvira,nvira,nvira)) 

    # aaa
    d3 = lib.direct_sum('ia+jb+kc->ijkabc', eia, eia, eia)
    w = numpy.einsum('ijae,kceb->ijkabc', t2aa, numpy.asarray(eris.get_ovvv()).conj())
    w-= numpy.einsum('mkbc,iajm->ijkabc', t2aa, numpy.asarray(eris.ovoo.conj()))
    w = w * coeff.Paaa
    r = r6(w)
    v = numpy.einsum('jbkc,ia->ijkabc', numpy.asarray(eris.ovov).conj(), t1a)
    v+= numpy.einsum('jkbc,ai->ijkabc', t2aa, fvo) * .5
    wvd = p6(w + v) / d3
    et = numpy.einsum('ijkabc,ijkabc', wvd.conj(), r)
    print('(T) correction from aaa, bbb: ',.5*et)

    #lsh test
    t_contrib = numpy.multiply(wvd.conj(), r) 

#    # bbb
#    d3 = lib.direct_sum('ia+jb+kc->ijkabc', eIA, eIA, eIA)
#    w = numpy.einsum('ijae,kceb->ijkabc', t2bb, numpy.asarray(eris.get_ovvv()).conj())
#    w-= numpy.einsum('imab,kcjm->ijkabc', t2bb, numpy.asarray(eris.ovoo.conj()))
#    w = w * coeff.Paaa
#    r = r6(w)
#    v = numpy.einsum('jbkc,ia->ijkabc', numpy.asarray(eris.ovov).conj(), t1b)
#    v+= numpy.einsum('jkbc,ai->ijkabc', t2bb, fVO) * .5
#    wvd = p6(w + v) / d3
#    et2 = numpy.einsum('ijkabc,ijkabc', wvd.conj(), r)
#    et += et2
#    print('(T) correction from bbb: ',.25*et2)

    # baa
    w  = numpy.einsum('jIeA,kceb->IjkAbc', t2ab, numpy.asarray(eris.get_ovvv()).conj()) * 2
    w += numpy.einsum('jIbE,kcEA->IjkAbc', t2ab, numpy.asarray(eris.get_ovvv()).conj()) * 2
    w += numpy.einsum('jkbe,IAec->IjkAbc', t2aa, numpy.asarray(eris.get_ovvv()).conj())
    w -= numpy.einsum('mIbA,kcjm->IjkAbc', t2ab, numpy.asarray(eris.ovoo).conj()) * 2
    w -= numpy.einsum('jMbA,kcIM->IjkAbc', t2ab, numpy.asarray(eris.ovoo).conj()) * 2
    w -= numpy.einsum('jmbc,IAkm->IjkAbc', t2aa, numpy.asarray(eris.ovoo).conj())
    w  = w * coeff.Pabb
    r = w - w.transpose(0,2,1,3,4,5)
    r = r + r.transpose(0,2,1,3,5,4)
    v  = numpy.einsum('jbkc,IA->IjkAbc', numpy.asarray(eris.ovov).conj(), t1b)
    v += numpy.einsum('kcIA,jb->IjkAbc', numpy.asarray(eris.ovov).conj(), t1a)
    v += numpy.einsum('kcIA,jb->IjkAbc', numpy.asarray(eris.ovov).conj(), t1a)
    v += numpy.einsum('jkbc,AI->IjkAbc', t2aa, fVO) * .5
    v += numpy.einsum('kIcA,bj->IjkAbc', t2ab, fvo) * 2
    w += v
    d3 = lib.direct_sum('ia+jb+kc->ijkabc', eIA, eia, eia)
    r /= d3
    et2 = numpy.einsum('ijkabc,ijkabc', w.conj(), r)
    et += et2
    print('(T) correction from baa, bba: ',.5*et2)

    #lsh test
    t_contrib = numpy.add(t_contrib, numpy.multiply(w.conj(), r)) 
#    for i in range(nocca):
#        for j in range(nocca):
#            for k in range(nocca):
#                for a in range(nvira):
#                    for b in range(nvira):
#                        for c in range(nvira):
#                            print (i,j,k,a,b,c, t_contrib[i,j,k,a,b,c])

    t_contrib_sort = t_contrib.reshape(-1)

    idx = numpy.argsort(-abs(t_contrib_sort))
    #print(t_contrib_sort[idx])
    for lt in range(nocca*nocca*nocca*nvira*nvira*nvira):
        l = idx[lt]
        #l = lt
        # lt = ((((i*nocca + j)*nocca + k)*nvria + a)*nvira + b)*nvira + c
        c = int(l % nvira)
        b = int(((l-c)/nvira) % nvira)
        a = int(((l-c)/nvira - b)/nvira % nvira )
        k = int((((l-c)/nvira - b)/nvira - a)/nvira % nocca)
        j = int(((((l-c)/nvira - b)/nvira - a)/nvira - k )/nocca % nocca)
        i = int((((((l-c)/nvira - b)/nvira - a)/nvira - k )/nocca - j)/nocca)
        print (i+1,j+1,k+1,a+nocca+1,b+nocca+1,c+nocca+1, t_contrib_sort[l])


#    # bba
#    w  = numpy.einsum('ijae,kceb->ijkabc', t2ab, numpy.asarray(eris.get_ovvv()).conj()) * 2
#    w += numpy.einsum('ijeb,kcea->ijkabc', t2ab, numpy.asarray(eris.get_ovvv()).conj()) * 2
#    w += numpy.einsum('jkbe,iaec->ijkabc', t2bb, numpy.asarray(eris.get_ovvv()).conj())
#    w -= numpy.einsum('imab,kcjm->ijkabc', t2ab, numpy.asarray(eris.ovoo).conj()) * 2
#    w -= numpy.einsum('mjab,kcim->ijkabc', t2ab, numpy.asarray(eris.ovoo).conj()) * 2
#    w -= numpy.einsum('jmbc,iakm->ijkabc', t2bb, numpy.asarray(eris.ovoo).conj())
#    w  = w * coeff.Pabb
#    r = w - w.transpose(0,2,1,3,4,5)
#    r = r + r.transpose(0,2,1,3,5,4)
#    v  = numpy.einsum('jbkc,ia->ijkabc', numpy.asarray(eris.ovov).conj(), t1a)
#    v += numpy.einsum('iakc,jb->ijkabc', numpy.asarray(eris.ovov).conj(), t1b)
#    v += numpy.einsum('iakc,jb->ijkabc', numpy.asarray(eris.ovov).conj(), t1b)
#    v += numpy.einsum('JKBC,ai->iJKaBC', t2bb, fvo) * .5
#    v += numpy.einsum('iKaC,BJ->iJKaBC', t2ab, fVO) * 2
#    w += v
#    d3 = lib.direct_sum('ia+jb+kc->ijkabc', eia, eIA, eIA)
#    r /= d3
#    et2 = numpy.einsum('ijkabc,ijkabc', w.conj(), r)
#    et += et2
#    print('(T) correction from bba: ',.25*et2)

    et *= .5

    log.info('max_memory %d MB (current use %d MB)',
             mcc.max_memory, lib.current_memory()[0])
    cput1 = log.timer('(T) correction', *cput1)

    return et

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
    mcc.conv_tol = 1e-14
    mcc.ccsd()
    t1a = t1b = mcc.t1
    t2ab = mcc.t2
    t2aa = t2bb = t2ab - t2ab.transpose(1,0,2,3)
    mcc = uccsd.UCCSD(scf.addons.convert_to_uhf(rhf))
    e3a = kernel(mcc, mcc.ao2mo(), (t1a,t1b), (t2aa,t2ab,t2bb))
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
    mcc = uccsd.UCCSD(scf.addons.convert_to_uhf(mf))
    e3a = kernel(mcc, mcc.ao2mo(), [t1a,t1b], [t2aa, t2ab, t2bb])
    print(e3a - 9877.2780859693339)

    mycc = cc.GCCSD(scf.addons.convert_to_ghf(mf))
    eris = mycc.ao2mo()
    t1 = mycc.spatial2spin(t1, eris.orbspin)
    t2 = mycc.spatial2spin(t2, eris.orbspin)
    from pyscf.cc import gccsd_t
    et = gccsd_t.kernel(mycc, eris, t1, t2)
    print(et - 9877.2780859693339)


    mol = gto.M()
    numpy.random.seed(12)
    nocca, noccb, nvira, nvirb = 3, 2, 4, 5
    nmo = nocca + nvira
    eris = cc.uccsd._ChemistsERIs()
    eri1 = (numpy.random.random((3,nmo,nmo,nmo,nmo)) +
            numpy.random.random((3,nmo,nmo,nmo,nmo)) * .8j - .5-.4j)
    eri1 = eri1 + eri1.transpose(0,2,1,4,3).conj()
    eri1[0] = eri1[0] + eri1[0].transpose(2,3,0,1)
    eri1[2] = eri1[2] + eri1[2].transpose(2,3,0,1)
    eri1 *= .1
    eris.ovov = eri1[0,nocca:,:nocca,nocca:,:nocca].transpose(1,0,3,2).conj()
    eris.ovvv = eri1[0,nocca:,:nocca,nocca:,nocca:].transpose(1,0,3,2).conj()
    eris.ovoo = eri1[0,nocca:,:nocca,:nocca,:nocca].transpose(1,0,3,2).conj()
    eris.OVOV = eri1[2,noccb:,:noccb,noccb:,:noccb].transpose(1,0,3,2).conj()
    eris.OVVV = eri1[2,noccb:,:noccb,noccb:,noccb:].transpose(1,0,3,2).conj()
    eris.OVOO = eri1[2,noccb:,:noccb,:noccb,:noccb].transpose(1,0,3,2).conj()
    eris.ovOV = eri1[1,nocca:,:nocca,noccb:,:noccb].transpose(1,0,3,2).conj()
    eris.ovVV = eri1[1,nocca:,:nocca,noccb:,noccb:].transpose(1,0,3,2).conj()
    eris.ovOO = eri1[1,nocca:,:nocca,:noccb,:noccb].transpose(1,0,3,2).conj()
    eris.OVov = eri1[1,nocca:,:nocca,noccb:,:noccb].transpose(3,2,1,0).conj()
    eris.OVvv = eri1[1,nocca:,nocca:,noccb:,:noccb].transpose(3,2,1,0).conj()
    eris.OVoo = eri1[1,:nocca,:nocca,noccb:,:noccb].transpose(3,2,1,0).conj()
    t1a  = .1 * numpy.random.random((nocca,nvira)) + numpy.random.random((nocca,nvira))*.1j
    t1b  = .1 * numpy.random.random((noccb,nvirb)) + numpy.random.random((noccb,nvirb))*.1j
    t2aa = .1 * numpy.random.random((nocca,nocca,nvira,nvira)) + numpy.random.random((nocca,nocca,nvira,nvira))*.1j
    t2aa = t2aa - t2aa.transpose(0,1,3,2)
    t2aa = t2aa - t2aa.transpose(1,0,2,3)
    t2bb = .1 * numpy.random.random((noccb,noccb,nvirb,nvirb)) + numpy.random.random((noccb,noccb,nvirb,nvirb))*.1j
    t2bb = t2bb - t2bb.transpose(0,1,3,2)
    t2bb = t2bb - t2bb.transpose(1,0,2,3)
    t2ab = .1 * numpy.random.random((nocca,noccb,nvira,nvirb)) + numpy.random.random((nocca,noccb,nvira,nvirb))*.1j
    f = (numpy.random.random((2,nmo,nmo)) * .4 +
         numpy.random.random((2,nmo,nmo)) * .4j)
    eris.focka = f[0]+f[0].T.conj() + numpy.diag(numpy.arange(nmo))
    eris.fockb = f[1]+f[1].T.conj() + numpy.diag(numpy.arange(nmo))
    t1 = t1a, t1b
    t2 = t2aa, t2ab, t2bb
    mcc = cc.UCCSD(scf.UHF(mol))
    print(kernel(mcc, eris, t1, t2) - (-0.056092415718338388-0.011390417704868244j))
