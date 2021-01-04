#!/usr/bin/env python
#
# Author: Seunghoon Lee <seunghoonlee89@gmail.com>
#

'''
Extracting coupled cluster amplitude
from configuration interaction coefficient
'''

import numpy 
import itertools
from pyscf.cc.fci_index import parity, reorder, fci_index
from pyscf.cc import _ccsd 
from pyscf import lib
import ctypes

import pandas as pd
from pandas import DataFrame

#class spinMO:
#    def __init__(self, p, spin='a'):
#        self.p = p  
#        self.spin = spin 
#        self.dic = {'a':0, 'b':1}
#
#    @staticmethod
#    def takeSpin(mo):
#        return mo.spin
#
#    @staticmethod
#    def countSpin(list_mo):
#        beta_num = 0
#        for mo in list_mo:
#            beta_num += mo.dic[mo.spin]
#        return beta_num
#
#    @staticmethod
#    def sort(list_mo):
#        list_spin = [ mo.dic[mo.spin] for mo in list_mo ]
#        p = parity(list_spin)
#        list_mo.sort(key=spinMO.takeSpin)
#        list_mo.append(p)
#        return list_mo 

class fci_to_cc:
    def __init__(self, nocc, nvir, idx, c0):
        self.dbg = False
        self.nocc = nocc
        self.nvir = nvir
        self.idx  = idx 
        self.t0 = numpy.zeros((1), dtype=numpy.float64)
        self.t0[0] = c0[0] 
        self.t1 = None 
        self.t2 = None 
        self.t3 = None 
        self.t3p= None 
        self.t4 = None 
        self.t2aa   = None 
        self.t2ab   = None      #  t2 = t2ab in restricted ccsd
        self.t3aaa  = None 
        self.t3aab  = None 
        self.t4aaab = None 
        self.t4aabb = None 

#    def t1s(self, i, a):
#        idx_o = [i]
#        idx_v = [a]
#        b_num_o  = spinMO.countSpin(idx_o)  
#        b_num_v  = spinMO.countSpin(idx_v)  
#        if (b_num_o != b_num_v):
#            return 0.0
#        else:
#            return self.t1[i.p][a.p] 
#
#    def t2s(self, i, j, a, b):
#        idx_o = [i, j]
#        idx_v = [a, b]
#        b_num_o  = spinMO.countSpin(idx_o)  
#        b_num_v  = spinMO.countSpin(idx_v)  
#        if (b_num_o != b_num_v):
#            return 0.0
#
#        ip, jp, po = spinMO.sort(idx_o)
#        ap, bp, pv = spinMO.sort(idx_v)
#        if (b_num_o == 0 or b_num_o == 2):
#            assert po == 1 and pv == 1
#            return self.t2aa[ip.p][jp.p][ap.p][bp.p]
#        elif (b_num_o == 1):
#            return po * pv * self.t2ab[ip.p][jp.p][ap.p][bp.p]          
#        else:
#            assert False
#
#    def t3s(self, i, j, k, a, b, c):
#        idx_o = [i, j, k]
#        idx_v = [a, b, c]
#        b_num_o  = spinMO.countSpin(idx_o)  
#        b_num_v  = spinMO.countSpin(idx_v)  
#        if (b_num_o != b_num_v):
#            return 0.0
#
#        ip, jp, kp, po = spinMO.sort(idx_o)
#        ap, bp, cp, pv = spinMO.sort(idx_v)
#        if (b_num_o == 0):
#            assert po == 1 and pv == 1
#            return self.t3aaa[ip.p][jp.p][kp.p][ap.p][bp.p][cp.p]
#        elif (b_num_o == 1):
#            return po * pv * self.t3aab[ip.p][jp.p][kp.p][ap.p][bp.p][cp.p]
#        elif (b_num_o == 2):
#            return po * pv * self.t3aab[jp.p][kp.p][ip.p][bp.p][cp.p][ap.p]
#        else:
#            assert False

    def c1_to_t1(self, c1):
        nocc = self.nocc
        nvir = self.nvir
        self.t1 = numpy.zeros((nocc,nvir), dtype=numpy.float64)
        rg_i = [ p for p in range(nocc) ]  
        rg_a = [ p for p in range(nvir) ]  
        for a, i in itertools.product(rg_a, rg_i): 
            ia = self.idx.S(i, a)
            self.t1[i][a] = c1[ia].copy()
            if self.dbg and abs(self.t1[i][a])>0.001: print ('%d %d %20.10f %20.10f'%(i+1,a+nocc+1,self.t1[i][a], c1[ia]))

    def c2_to_t2(self, c2aa, c2ab):
        nocc = self.nocc
        nvir = self.nvir
        self.t2   = numpy.zeros((nocc,nocc,nvir,nvir), dtype=numpy.float64)
        self.t2aa = numpy.zeros((nocc,nocc,nvir,nvir), dtype=numpy.float64)
        self.t2ab = numpy.zeros((nocc,nocc,nvir,nvir), dtype=numpy.float64)
        rg_i = [ p for p in range(nocc) ]  
        rg_j = rg_i  
        rg_a = [ p for p in range(nvir) ]  
        rg_b = rg_a

        for b, a, j, i in itertools.product(rg_b, rg_a, rg_j, rg_i): 
            ia   = self.idx.S(i, a) 
            jb   = self.idx.S(j, b) 

            t2ind_ab = self.t1xt1ab(i, j, a, b)
#            t2ind_ab = self.t1xt1(i, j, a, b, 'ab')
            self.t2ab[i][j][a][b] = c2ab[ia,jb].copy() - t2ind_ab
            if self.dbg and abs(self.t2ab[i][j][a][b])>0.001: print ('ab %d %d %d %d %20.10f %20.10f'%(i+1,j+1,a+nocc+1,b+nocc+1,self.t2ab[i][j][a][b], c2ab[ia,jb]))

            if i != j and a != b:
                idx_o = [i, j]
                idx_v = [a, b]
                ip, jp, po = reorder(idx_o)
                ap, bp, pv = reorder(idx_v)

                ijab = self.idx.D(ip, jp, ap, bp) 
                t2ind_aa = self.t1xt1aa(i, j, a, b)
#                t2ind_aa = self.t1xt1(i, j, a, b, 'aa')
                self.t2aa[i][j][a][b] = po*pv*c2aa[ijab].copy() - t2ind_aa

        self.t2 = self.t2ab 

    def c3_to_t3(self, c3aaa, c3aab):
        nocc = self.nocc
        nvir = self.nvir
        self.t3    = numpy.zeros((nocc,nocc,nocc,nvir,nvir,nvir), dtype=numpy.float64)
        self.t3p   = numpy.zeros((nocc,nocc,nocc,nvir,nvir,nvir), dtype=numpy.float64)

        self.t3aaa = numpy.zeros((nocc,nocc,nocc,nvir,nvir,nvir), dtype=numpy.float64)
        self.t3aab = numpy.zeros((nocc,nocc,nocc,nvir,nvir,nvir), dtype=numpy.float64)

        rg_i = [ p for p in range(nocc) ]  
        rg_j = rg_i  
        rg_k = rg_i  
        rg_a = [ p for p in range(nvir) ]  
        rg_b = rg_a
        rg_c = rg_a

#lsh TODO: using symmetry btw t3aaa and t3aab by restricted singlet state 
        for c, b, a, k, j, i in itertools.product(rg_c, rg_b, rg_a, rg_k, rg_j, rg_i): 

            if i != j and a != b:
                idx_o = [i, j]
                idx_v = [a, b]
                ip, jp, po = reorder(idx_o)
                ap, bp, pv = reorder(idx_v)
                ijab = self.idx.D(ip, jp, ap, bp) 
                kc   = self.idx.S(k, c) 
                t3ind_aab  = self.t1xt2aab(i, j, k, a, b, c)
                t3ind_aab += self.t1xt1xt1aab(i, j, k, a, b, c)
#                t3ind_aab  = self.t1xt2(i, j, k, a, b, c, 'aab')
#                t3ind_aab += self.t1xt1xt1(i, j, k, a, b, c, 'aab')
                self.t3aab[i][j][k][a][b][c] = po*pv*c3aab[ijab,kc].copy() - t3ind_aab
                if self.dbg and abs(self.t3aab[i][j][k][a][b][c])>0.001: print ('aab %d %d %d %d %d %d %20.10f %20.10f'%(i+1,j+1,k+1,a+nocc+1,b+nocc+1,c+nocc+1,self.t3aab[i][j][k][a][b][c], po*pv*c3aab[ijab,kc]))

            if i != j and j != k and k != i \
            and a != b and b != c and c != a:
                idx_o = [i, j, k]
                idx_v = [a, b, c]
                ip, jp, kp, po = reorder(idx_o)
                ap, bp, cp, pv = reorder(idx_v)

                ijkabc = self.idx.T(ip, jp, kp, ap, bp, cp) 
                t3ind_aaa  = self.t1xt2aaa(i, j, k, a, b, c)
                t3ind_aaa += self.t1xt1xt1aaa(i, j, k, a, b, c)
#                t3ind_aaa  = self.t1xt2(i, j, k, a, b, c, 'aaa')
#                t3ind_aaa += self.t1xt1xt1(i, j, k, a, b, c, 'aaa')
                self.t3aaa[i][j][k][a][b][c] = po*pv*c3aaa[ijkabc].copy() - t3ind_aaa
#lsh TODO: don't make t3aaa, t3aab to make t3 and t3p
        self.t3_rccsd()

    def t3_rccsd(self):
        rg_i = [ p for p in range(self.nocc) ]  
        rg_j = rg_i  
        rg_k = rg_i  
        rg_a = [ p for p in range(self.nvir) ]  
        rg_b = rg_a
        rg_c = rg_a

        for c, b, a, k, j, i in itertools.product(rg_c, rg_b, rg_a, rg_k, rg_j, rg_i): 
            self.t3[i][j][k][a][b][c]  = self.t3aab[i][j][k][a][b][c].copy()
            self.t3[i][j][k][a][b][c] += self.t3aab[i][k][j][a][c][b].copy()
            self.t3p[i][j][k][a][b][c]  = self.t3aaa[i][j][k][a][b][c].copy()
            self.t3p[i][j][k][a][b][c] += self.t3[i][j][k][a][b][c].copy()
            self.t3p[i][j][k][a][b][c] += self.t3aab[j][k][i][b][c][a].copy()
            self.t3p[i][j][k][a][b][c] /= 0.5 

    def c3_to_t3_ecT(self, c3aaa, c3aab):
        nacto_ref = self.nacto_ref
        nacte_ref = self.nacte_ref 
        nc_ref    = self.ncoreo_ref
        nocc_ref  = nacte_ref // 2
        nvir_ref  = nacto_ref - nocc_ref 
        nc        = self.nc
        print ('nc_ref, nocc_ref, nvir_ref', nc_ref, nocc_ref, nvir_ref)

        nocc = self.nocc
        nvir = self.nvir
        self.t3    = numpy.zeros((nocc,nocc,nocc,nvir,nvir,nvir), dtype=numpy.float64)
        self.t3p   = numpy.zeros((nocc,nocc,nocc,nvir,nvir,nvir), dtype=numpy.float64)

        self.t3aaa = numpy.zeros((nocc,nocc,nocc,nvir,nvir,nvir), dtype=numpy.float64)
        self.t3aab = numpy.zeros((nocc,nocc,nocc,nvir,nvir,nvir), dtype=numpy.float64)

        rg_i = [ p for p in range(nocc) ]  
        rg_j = rg_i  
        rg_k = rg_i  
        rg_a = [ p for p in range(nvir) ]  
        rg_b = rg_a
        rg_c = rg_a

#lsh TODO: using symmetry btw t3aaa and t3aab by restricted singlet state 
        for c, b, a, k, j, i in itertools.product(rg_c, rg_b, rg_a, rg_k, rg_j, rg_i): 
            if i <= nc_ref and j <= nc_ref and k <= nc_ref \
           and a >nvir_ref and b >nvir_ref and c >nvir_ref: continue

            if i != j and a != b:
                idx_o = [i, j]
                idx_v = [a, b]
                ip, jp, po = reorder(idx_o)
                ap, bp, pv = reorder(idx_v)
                ijab = self.idx.D(ip, jp, ap, bp) 
                kc   = self.idx.S(k, c) 
                t3ind_aab  = self.t1xt2aab(i, j, k, a, b, c)
                t3ind_aab += self.t1xt1xt1aab(i, j, k, a, b, c)
#                t3ind_aab  = self.t1xt2(i, j, k, a, b, c, 'aab')
#                t3ind_aab += self.t1xt1xt1(i, j, k, a, b, c, 'aab')
                self.t3aab[i][j][k][a][b][c] = po*pv*c3aab[ijab,kc].copy() - t3ind_aab
                if self.dbg and abs(self.t3aab[i][j][k][a][b][c])>0.001: print ('aab %d %d %d %d %d %d %20.10f %20.10f'%(i+1,j+1,k+1,a+nocc+1,b+nocc+1,c+nocc+1,self.t3aab[i][j][k][a][b][c], po*pv*c3aab[ijab,kc]))

            if i != j and j != k and k != i \
            and a != b and b != c and c != a:
                idx_o = [i, j, k]
                idx_v = [a, b, c]
                ip, jp, kp, po = reorder(idx_o)
                ap, bp, cp, pv = reorder(idx_v)

                ijkabc = self.idx.T(ip, jp, kp, ap, bp, cp) 
                t3ind_aaa  = self.t1xt2aaa(i, j, k, a, b, c)
                t3ind_aaa += self.t1xt1xt1aaa(i, j, k, a, b, c)
#                t3ind_aaa  = self.t1xt2(i, j, k, a, b, c, 'aaa')
#                t3ind_aaa += self.t1xt1xt1(i, j, k, a, b, c, 'aaa')
                self.t3aaa[i][j][k][a][b][c] = po*pv*c3aaa[ijkabc].copy() - t3ind_aaa
#lsh TODO: don't make t3aaa, t3aab to make t3 and t3p
        self.t3_rccsd()

    def c4_to_t4(self, c4aaab, c4aabb):
        nocc = self.nocc
        nvir = self.nvir
        self.t4 = numpy.zeros((nocc,nocc,nocc,nocc,nvir,nvir,nvir,nvir), dtype=numpy.float64)
        self.t4aaab = numpy.zeros((nocc,nocc,nocc,nocc,nvir,nvir,nvir,nvir), dtype=numpy.float64)
        self.t4aabb = numpy.zeros((nocc,nocc,nocc,nocc,nvir,nvir,nvir,nvir), dtype=numpy.float64)

        rg_i = [ p for p in range(self.nocc) ]  
        rg_j = rg_i  
        rg_k = rg_i  
        rg_l = rg_i  
        rg_a = [ p for p in range(self.nvir) ]  
        rg_b = rg_a
        rg_c = rg_a
        rg_d = rg_a

#lsh TODO: using symmetry btw t4aaab and t4aabb by restricted singlet state 
        for d, c, b, a, l, k, j, i in itertools.product(rg_d, rg_c, rg_b, rg_a, rg_l, rg_k, rg_j, rg_i): 

            if i != j and a != b and k != l and c != d:
                idx_o = [i, j]
                idx_v = [a, b]
                ip, jp, po1 = reorder(idx_o)
                ap, bp, pv1 = reorder(idx_v)
                ijab = self.idx.D(ip, jp, ap, bp) 
                idx_o = [k, l]
                idx_v = [c, d]
                kp, lp, po2 = reorder(idx_o)
                cp, dp, pv2 = reorder(idx_v)
                klcd = self.idx.D(kp, lp, cp, dp) 
                parity = po1 * po2 * pv1 * pv2

                t4ind_aabb = 0.0
                t4ind_aabb  = self.t1xt3aabb(i, j, k, l, a, b, c, d)
                t4ind_aabb += self.t2xt2aabb(i, j, k, l, a, b, c, d)
                t4ind_aabb += self.t1xt1xt2aabb(i, j, k, l, a, b, c, d)
                t4ind_aabb += self.t1xt1xt1xt1aabb(i, j, k, l, a, b, c, d)

#                t4ind_aabb  = self.t1xt3(i, j, k, l, a, b, c, d, 'aabb')
#                t4ind_aabb += self.t2xt2(i, j, k, l, a, b, c, d, 'aabb')
#                t4ind_aabb += self.t1xt1xt2(i, j, k, l, a, b, c, d, 'aabb')
#                t4ind_aabb += self.t1xt1xt1xt1(i, j, k, l, a, b, c, d, 'aabb')
#                self.t4aabb[i][j][k][l][a][b][c][d] = parity*c4aabb[ijab,klcd]
#                self.t4aabb[i][j][k][l][a][b][c][d] = 0.0 
                self.t4aabb[i][j][k][l][a][b][c][d] = parity*c4aabb[ijab,klcd].copy() - t4ind_aabb
#lsh test
                if self.dbg and abs(self.t4aabb[i][j][k][l][a][b][c][d])>0.001: print ('aabb %d %d %d %d %d %d %d %d %20.10f %20.10f'%(i+1,j+1,k+1,l+1,a+self.nocc+1,b+self.nocc+1,c+self.nocc+1,d+self.nocc+1,self.t4aabb[i][j][k][l][a][b][c][d], parity*c4aabb[ijab,klcd]))

            if i != j and j != k and k != i \
            and a != b and b != c and c != a:
                idx_o = [i, j, k]
                idx_v = [a, b, c]
                ip, jp, kp, po = reorder(idx_o)
                ap, bp, cp, pv = reorder(idx_v)
                ijkabc = self.idx.T(ip, jp, kp, ap, bp, cp) 
                ld   = self.idx.S(l, d) 


                t4ind_aaab = 0.0
                t4ind_aaab  = self.t1xt3aaab(i, j, k, l, a, b, c, d)
                t4ind_aaab += self.t2xt2aaab(i, j, k, l, a, b, c, d)
                t4ind_aaab += self.t1xt1xt2aaab(i, j, k, l, a, b, c, d)
                t4ind_aaab += self.t1xt1xt1xt1aaab(i, j, k, l, a, b, c, d)

#                t4ind_aaab  = self.t1xt3(i, j, k, l, a, b, c, d, 'aaab')
#                t4ind_aaab += self.t2xt2(i, j, k, l, a, b, c, d, 'aaab')
#                t4ind_aaab += self.t1xt1xt2(i, j, k, l, a, b, c, d, 'aaab')
#                t4ind_aaab += self.t1xt1xt1xt1(i, j, k, l, a, b, c, d, 'aaab')
#                self.t4aaab[i][j][k][l][a][b][c][d] = po*pv*c4aaab[ijkabc,ld]
                self.t4aaab[i][j][k][l][a][b][c][d] = po*pv*c4aaab[ijkabc,ld].copy() - t4ind_aaab
                if self.dbg and abs(self.t4aaab[i][j][k][l][a][b][c][d])>0.001: print ('aaab %d %d %d %d %d %d %d %d %20.10f %20.10f'%(i+1,j+1,k+1,l+1,a+self.nocc+1,b+self.nocc+1,c+self.nocc+1,d+self.nocc+1,self.t4aaab[i][j][k][l][a][b][c][d], parity*c4aaab[ijab,klcd]))
#lsh TODO: don't make t4aaaa, t4aaab to make t4
        self.t4_rccsd()

    def t4_rccsd(self):
        rg_i = [ p for p in range(self.nocc) ]  
        rg_j = rg_i  
        rg_k = rg_i  
        rg_l = rg_i  
        rg_a = [ p for p in range(self.nvir) ]  
        rg_b = rg_a
        rg_c = rg_a
        rg_d = rg_a

        for b, a, d, c, j, i, l, k in itertools.product(rg_b, rg_a, rg_d, rg_c, rg_j, rg_i, rg_l, rg_k): 
            self.t4[k][l][i][j][c][d][a][b]  = self.t4aaab[k][l][i][j][c][d][a][b].copy()
            self.t4[k][l][i][j][c][d][a][b] += self.t4aaab[k][l][j][i][c][d][b][a].copy()
            self.t4[k][l][i][j][c][d][a][b] += self.t4aabb[k][i][l][j][c][a][d][b].copy()
            self.t4[k][l][i][j][c][d][a][b] += self.t4aabb[l][i][k][j][d][a][c][b].copy()
#            self.t4[k][l][i][j][c][d][a][b] *= 0.5 

    def interm_norm(self):
        self.t1  = self.t1 /self.t0[0]
        self.t2  = self.t2 /self.t0[0]
        self.t3  = self.t3 /self.t0[0]
        self.t3p = self.t3p/self.t0[0]
        self.t4  = self.t4 /self.t0[0]

    def t1xt1aa(self, i, j, a, b):
        t1xt1 = self.t1[i][a] * self.t1[j][b] - self.t1[j][a] * self.t1[i][b]
        return t1xt1

    def t1xt1ab(self, i, j, a, b):
        t1xt1 = self.t1[i][a] * self.t1[j][b]
        return t1xt1

#    def t1xt1(self, i_spt, j_spt, a_spt, b_spt, spin='aa'):
#        i = spinMO(i_spt, 'a')
#        a = spinMO(a_spt, 'a')
#        if spin == 'aa':
#            j = spinMO(j_spt, 'a')
#            b = spinMO(b_spt, 'a')
#        elif spin == 'ab':
#            j = spinMO(j_spt, 'b')
#            b = spinMO(b_spt, 'b')
#        else:
#            assert False
#
#        t1xt1 = self.t1s(i,a) * self.t1s(j,b) - self.t1s(j,a) * self.t1s(i,b)
#        return t1xt1

    def t1xt2aaa(self, i, j, k, a, b, c):
        t1xt2 = self.t1[i][a] * self.t2aa[j][k][b][c] - self.t1[i][b] * self.t2aa[j][k][a][c] + self.t1[i][c] * self.t2aa[j][k][a][b] \
              - self.t1[j][a] * self.t2aa[i][k][b][c] + self.t1[j][b] * self.t2aa[i][k][a][c] - self.t1[j][c] * self.t2aa[i][k][a][b] \
              + self.t1[k][a] * self.t2aa[i][j][b][c] - self.t1[k][b] * self.t2aa[i][j][a][c] + self.t1[k][c] * self.t2aa[i][j][a][b]
        return t1xt2

    def t1xt2aab(self, i, j, k, a, b, c):
        t1xt2 = self.t1[i][a] * self.t2ab[j][k][b][c] - self.t1[i][b] * self.t2ab[j][k][a][c] \
              - self.t1[j][a] * self.t2ab[i][k][b][c] + self.t1[j][b] * self.t2ab[i][k][a][c] \
              + self.t1[k][c] * self.t2aa[i][j][a][b]
        return t1xt2

#    def t1xt2(self, i_spt, j_spt, k_spt, a_spt, b_spt, c_spt, spin='aaa'):
#        i = spinMO(i_spt, 'a')
#        j = spinMO(j_spt, 'a')
#        a = spinMO(a_spt, 'a')
#        b = spinMO(b_spt, 'a')
#        if spin == 'aaa':
#            k = spinMO(k_spt, 'a')
#            c = spinMO(c_spt, 'a')
#        elif spin == 'aab':
#            k = spinMO(k_spt, 'b')
#            c = spinMO(c_spt, 'b')
#        else:
#            assert False
#
#        t1xt2 = self.t1s(i,a) * self.t2s(j,k,b,c) - self.t1s(i,b) * self.t2s(j,k,a,c) + self.t1s(i,c) * self.t2s(j,k,a,b) \
#              - self.t1s(j,a) * self.t2s(i,k,b,c) + self.t1s(j,b) * self.t2s(i,k,a,c) - self.t1s(j,c) * self.t2s(i,k,a,b) \
#              + self.t1s(k,a) * self.t2s(i,j,b,c) - self.t1s(k,b) * self.t2s(i,j,a,c) + self.t1s(k,c) * self.t2s(i,j,a,b)
#        return t1xt2

    def t1xt1xt1aaa(self, i, j, k, a, b, c):
        t1xt1xt1 = self.t1[i][a] * self.t1[j][b] * self.t1[k][c] - self.t1[i][a] * self.t1[j][c] * self.t1[k][b] \
                 - self.t1[i][b] * self.t1[j][a] * self.t1[k][c] + self.t1[i][b] * self.t1[j][c] * self.t1[k][a] \
                 + self.t1[i][c] * self.t1[j][a] * self.t1[k][b] - self.t1[i][c] * self.t1[j][b] * self.t1[k][a]
        return t1xt1xt1

    def t1xt1xt1aab(self, i, j, k, a, b, c):
        t1xt1xt1 = self.t1[i][a] * self.t1[j][b] * self.t1[k][c] - self.t1[i][b] * self.t1[j][a] * self.t1[k][c]
        return t1xt1xt1

#    def t1xt1xt1(self, i_spt, j_spt, k_spt, a_spt, b_spt, c_spt, spin='aaa'):
#        i = spinMO(i_spt, 'a')
#        j = spinMO(j_spt, 'a')
#        a = spinMO(a_spt, 'a')
#        b = spinMO(b_spt, 'a')
#        if spin == 'aaa':
#            k = spinMO(k_spt, 'a')
#            c = spinMO(c_spt, 'a')
#        elif spin == 'aab':
#            k = spinMO(k_spt, 'b')
#            c = spinMO(c_spt, 'b')
#        else:
#            assert False
#
#        t1xt1xt1 = self.t1s(i,a) * self.t1s(j,b) * self.t1s(k,c) - self.t1s(i,a) * self.t1s(j,c) * self.t1s(k,b) \
#                 - self.t1s(i,b) * self.t1s(j,a) * self.t1s(k,c) + self.t1s(i,b) * self.t1s(j,c) * self.t1s(k,a) \
#                 + self.t1s(i,c) * self.t1s(j,a) * self.t1s(k,b) - self.t1s(i,c) * self.t1s(j,b) * self.t1s(k,a)
#        return t1xt1xt1

    def t1xt3aaab(self, i, j, k, l, a, b, c, d):
        t1xt3 = self.t1[i][a]*self.t3aab[j][k][l][b][c][d] - self.t1[i][b]*self.t3aab[j][k][l][a][c][d] + self.t1[i][c]*self.t3aab[j][k][l][a][b][d] \
              - self.t1[j][a]*self.t3aab[i][k][l][b][c][d] + self.t1[j][b]*self.t3aab[i][k][l][a][c][d] - self.t1[j][c]*self.t3aab[i][k][l][a][b][d] \
              + self.t1[k][a]*self.t3aab[i][j][l][b][c][d] - self.t1[k][b]*self.t3aab[i][j][l][a][c][d] + self.t1[k][c]*self.t3aab[i][j][l][a][b][d] \
              + self.t1[l][d]*self.t3aaa[i][j][k][a][b][c]
        return t1xt3

    def t1xt3aabb(self, i, j, k, l, a, b, c, d):
        t1xt3 = self.t1[i][a]*self.t3aab[k][l][j][c][d][b] - self.t1[i][b]*self.t3aab[k][l][j][c][d][a] \
              - self.t1[j][a]*self.t3aab[k][l][i][c][d][b] + self.t1[j][b]*self.t3aab[k][l][i][c][d][a] \
              + self.t1[k][c]*self.t3aab[i][j][l][a][b][d] - self.t1[k][d]*self.t3aab[i][j][l][a][b][c] \
              - self.t1[l][c]*self.t3aab[i][j][k][a][b][d] + self.t1[l][d]*self.t3aab[i][j][k][a][b][c]
        return t1xt3

#    def t1xt3(self, i_spt, j_spt, k_spt, l_spt, a_spt, b_spt, c_spt, d_spt, spin='aaab'):
#        i = spinMO(i_spt, 'a')
#        j = spinMO(j_spt, 'a')
#        l = spinMO(l_spt, 'b')
#        a = spinMO(a_spt, 'a')
#        b = spinMO(b_spt, 'a')
#        d = spinMO(d_spt, 'b')
#        if spin == 'aaab':
#            k = spinMO(k_spt, 'a')
#            c = spinMO(c_spt, 'a')
#        elif spin == 'aabb':
#            k = spinMO(k_spt, 'b')
#            c = spinMO(c_spt, 'b')
#        else:
#            assert False
#
#        t1xt3 = self.t1s(i,a)*self.t3s(j,k,l,b,c,d) - self.t1s(i,b)*self.t3s(j,k,l,a,c,d) + self.t1s(i,c)*self.t3s(j,k,l,a,b,d) - self.t1s(i,d)*self.t3s(j,k,l,a,b,c) \
#              - self.t1s(j,a)*self.t3s(i,k,l,b,c,d) + self.t1s(j,b)*self.t3s(i,k,l,a,c,d) - self.t1s(j,c)*self.t3s(i,k,l,a,b,d) + self.t1s(j,d)*self.t3s(i,k,l,a,b,c) \
#              + self.t1s(k,a)*self.t3s(i,j,l,b,c,d) - self.t1s(k,b)*self.t3s(i,j,l,a,c,d) + self.t1s(k,c)*self.t3s(i,j,l,a,b,d) - self.t1s(k,d)*self.t3s(i,j,l,a,b,c) \
#              - self.t1s(l,a)*self.t3s(i,j,k,b,c,d) + self.t1s(l,b)*self.t3s(i,j,k,a,c,d) - self.t1s(l,c)*self.t3s(i,j,k,a,b,d) + self.t1s(l,d)*self.t3s(i,j,k,a,b,c)
#        return t1xt3

    def t2xt2aaab(self, i, j, k, l, a, b, c, d):
        t2xt2 = self.t2aa[i][j][a][b] * self.t2ab[k][l][c][d] - self.t2aa[i][j][a][c] * self.t2ab[k][l][b][d]  + self.t2aa[i][j][b][c] * self.t2ab[k][l][a][d] \
              - self.t2aa[i][k][a][b] * self.t2ab[j][l][c][d] + self.t2aa[i][k][a][c] * self.t2ab[j][l][b][d] - self.t2aa[i][k][b][c] * self.t2ab[j][l][a][d] \
              + self.t2ab[i][l][a][d] * self.t2aa[j][k][b][c] - self.t2ab[i][l][b][d] * self.t2aa[j][k][a][c] + self.t2ab[i][l][c][d] * self.t2aa[j][k][a][b]
        return t2xt2

    def t2xt2aabb(self, i, j, k, l, a, b, c, d):
        t2xt2 = self.t2aa[i][j][a][b] * self.t2aa[k][l][c][d] + self.t2ab[i][k][a][c] * self.t2ab[j][l][b][d] - self.t2ab[i][k][a][d] * self.t2ab[j][l][b][c] \
              - self.t2ab[i][k][b][c] * self.t2ab[j][l][a][d] + self.t2ab[i][k][b][d] * self.t2ab[j][l][a][c] - self.t2ab[i][l][a][c] * self.t2ab[j][k][b][d] \
              + self.t2ab[i][l][a][d] * self.t2ab[j][k][b][c] + self.t2ab[i][l][b][c] * self.t2ab[j][k][a][d] - self.t2ab[i][l][b][d] * self.t2ab[j][k][a][c]
        return t2xt2

#    def t2xt2(self, i_spt, j_spt, k_spt, l_spt, a_spt, b_spt, c_spt, d_spt, spin='aaab'):
#        i = spinMO(i_spt, 'a')
#        j = spinMO(j_spt, 'a')
#        l = spinMO(l_spt, 'b')
#        a = spinMO(a_spt, 'a')
#        b = spinMO(b_spt, 'a')
#        d = spinMO(d_spt, 'b')
#        if spin == 'aaab':
#            k = spinMO(k_spt, 'a')
#            c = spinMO(c_spt, 'a')
#        elif spin == 'aabb':
#            k = spinMO(k_spt, 'b')
#            c = spinMO(c_spt, 'b')
#        else:
#            assert False
#
#        t2xt2 = self.t2s(i,j,a,b) * self.t2s(k,l,c,d) - self.t2s(i,j,a,c) * self.t2s(k,l,b,d) + self.t2s(i,j,a,d) * self.t2s(k,l,b,c) \
#              + self.t2s(i,j,b,c) * self.t2s(k,l,a,d) - self.t2s(i,j,b,d) * self.t2s(k,l,a,c) + self.t2s(i,j,c,d) * self.t2s(k,l,a,b) \
#              - self.t2s(i,k,a,b) * self.t2s(j,l,c,d) + self.t2s(i,k,a,c) * self.t2s(j,l,b,d) - self.t2s(i,k,a,d) * self.t2s(j,l,b,c) \
#              - self.t2s(i,k,b,c) * self.t2s(j,l,a,d) + self.t2s(i,k,b,d) * self.t2s(j,l,a,c) - self.t2s(i,k,c,d) * self.t2s(j,l,a,b) \
#              + self.t2s(i,l,a,b) * self.t2s(j,k,c,d) - self.t2s(i,l,a,c) * self.t2s(j,k,b,d) + self.t2s(i,l,a,d) * self.t2s(j,k,b,c) \
#              + self.t2s(i,l,b,c) * self.t2s(j,k,a,d) - self.t2s(i,l,b,d) * self.t2s(j,k,a,c) + self.t2s(i,l,c,d) * self.t2s(j,k,a,b)
#        return t2xt2

    def t1xt1xt2aaab(self, i, j, k, l, a, b, c, d):
        t1xt1xt2 = self.t1[i][a]*self.t1[j][b]*self.t2ab[k][l][c][d] - self.t1[i][a]*self.t1[j][c]*self.t2ab[k][l][b][d] + self.t1[i][b]*self.t1[j][c]*self.t2ab[k][l][a][d] \
                 - self.t1[i][a]*self.t1[k][b]*self.t2ab[j][l][c][d] + self.t1[i][a]*self.t1[k][c]*self.t2ab[j][l][b][d] - self.t1[i][b]*self.t1[k][c]*self.t2ab[j][l][a][d] \
                 + self.t1[i][a]*self.t1[l][d]*self.t2aa[j][k][b][c] - self.t1[i][b]*self.t1[l][d]*self.t2aa[j][k][a][c] + self.t1[i][c]*self.t1[l][d]*self.t2aa[j][k][a][b] \
                 + self.t1[j][a]*self.t1[k][b]*self.t2ab[i][l][c][d] - self.t1[j][a]*self.t1[k][c]*self.t2ab[i][l][b][d] + self.t1[j][b]*self.t1[k][c]*self.t2ab[i][l][a][d] \
                 - self.t1[j][a]*self.t1[l][d]*self.t2aa[i][k][b][c] + self.t1[j][b]*self.t1[l][d]*self.t2aa[i][k][a][c] - self.t1[j][c]*self.t1[l][d]*self.t2aa[i][k][a][b] \
                 + self.t1[k][a]*self.t1[l][d]*self.t2aa[i][j][b][c] - self.t1[k][b]*self.t1[l][d]*self.t2aa[i][j][a][c] + self.t1[k][c]*self.t1[l][d]*self.t2aa[i][j][a][b] \
                 - self.t1[j][a]*self.t1[i][b]*self.t2ab[k][l][c][d] + self.t1[j][a]*self.t1[i][c]*self.t2ab[k][l][b][d] - self.t1[j][b]*self.t1[i][c]*self.t2ab[k][l][a][d] \
                 + self.t1[k][a]*self.t1[i][b]*self.t2ab[j][l][c][d] - self.t1[k][a]*self.t1[i][c]*self.t2ab[j][l][b][d] + self.t1[k][b]*self.t1[i][c]*self.t2ab[j][l][a][d] \
                 - self.t1[k][a]*self.t1[j][b]*self.t2ab[i][l][c][d] + self.t1[k][a]*self.t1[j][c]*self.t2ab[i][l][b][d] - self.t1[k][b]*self.t1[j][c]*self.t2ab[i][l][a][d] 
        return t1xt1xt2

    def t1xt1xt2aabb(self, i, j, k, l, a, b, c, d):
        t1xt1xt2 = self.t1[i][a]*self.t1[j][b]*self.t2aa[k][l][c][d] + self.t1[k][c]*self.t1[l][d]*self.t2aa[i][j][a][b] \
                 - self.t1[j][a]*self.t1[i][b]*self.t2aa[k][l][c][d] - self.t1[l][c]*self.t1[k][d]*self.t2aa[i][j][a][b] \
                 + self.t1[i][a]*self.t1[k][c]*self.t2ab[j][l][b][d] - self.t1[i][a]*self.t1[k][d]*self.t2ab[j][l][b][c] \
                 - self.t1[i][b]*self.t1[k][c]*self.t2ab[j][l][a][d] + self.t1[i][b]*self.t1[k][d]*self.t2ab[j][l][a][c] \
                 - self.t1[i][a]*self.t1[l][c]*self.t2ab[j][k][b][d] + self.t1[i][a]*self.t1[l][d]*self.t2ab[j][k][b][c] \
                 + self.t1[i][b]*self.t1[l][c]*self.t2ab[j][k][a][d] - self.t1[i][b]*self.t1[l][d]*self.t2ab[j][k][a][c] \
                 - self.t1[j][a]*self.t1[k][c]*self.t2ab[i][l][b][d] + self.t1[j][a]*self.t1[k][d]*self.t2ab[i][l][b][c] \
                 + self.t1[j][b]*self.t1[k][c]*self.t2ab[i][l][a][d] - self.t1[j][b]*self.t1[k][d]*self.t2ab[i][l][a][c] \
                 + self.t1[j][a]*self.t1[l][c]*self.t2ab[i][k][b][d] - self.t1[j][a]*self.t1[l][d]*self.t2ab[i][k][b][c] \
                 - self.t1[j][b]*self.t1[l][c]*self.t2ab[i][k][a][d] + self.t1[j][b]*self.t1[l][d]*self.t2ab[i][k][a][c] 
        return t1xt1xt2

#    def t1xt1xt2(self, i_spt, j_spt, k_spt, l_spt, a_spt, b_spt, c_spt, d_spt, spin='aaab'):
#        i = spinMO(i_spt, 'a')
#        j = spinMO(j_spt, 'a')
#        l = spinMO(l_spt, 'b')
#        a = spinMO(a_spt, 'a')
#        b = spinMO(b_spt, 'a')
#        d = spinMO(d_spt, 'b')
#        if spin == 'aaab':
#            k = spinMO(k_spt, 'a')
#            c = spinMO(c_spt, 'a')
#        elif spin == 'aabb':
#            k = spinMO(k_spt, 'b')
#            c = spinMO(c_spt, 'b')
#        else:
#            assert False
#
#        t1xt1xt2 = self.t1s(i,a)*self.t1s(j,b)*self.t2s(k,l,c,d) - self.t1s(i,a)*self.t1s(j,c)*self.t2s(k,l,b,d) + self.t1s(i,a)*self.t1s(j,d)*self.t2s(k,l,b,c) \
#                 + self.t1s(i,b)*self.t1s(j,c)*self.t2s(k,l,a,d) - self.t1s(i,b)*self.t1s(j,d)*self.t2s(k,l,a,c) + self.t1s(i,c)*self.t1s(j,d)*self.t2s(k,l,a,b) \
#                 - self.t1s(i,a)*self.t1s(k,b)*self.t2s(j,l,c,d) + self.t1s(i,a)*self.t1s(k,c)*self.t2s(j,l,b,d) - self.t1s(i,a)*self.t1s(k,d)*self.t2s(j,l,b,c) \
#                 - self.t1s(i,b)*self.t1s(k,c)*self.t2s(j,l,a,d) + self.t1s(i,b)*self.t1s(k,d)*self.t2s(j,l,a,c) - self.t1s(i,c)*self.t1s(k,d)*self.t2s(j,l,a,b) \
#                 + self.t1s(i,a)*self.t1s(l,b)*self.t2s(j,k,c,d) - self.t1s(i,a)*self.t1s(l,c)*self.t2s(j,k,b,d) + self.t1s(i,a)*self.t1s(l,d)*self.t2s(j,k,b,c) \
#                 + self.t1s(i,b)*self.t1s(l,c)*self.t2s(j,k,a,d) - self.t1s(i,b)*self.t1s(l,d)*self.t2s(j,k,a,c) + self.t1s(i,c)*self.t1s(l,d)*self.t2s(j,k,a,b) \
#                 + self.t1s(j,a)*self.t1s(k,b)*self.t2s(i,l,c,d) - self.t1s(j,a)*self.t1s(k,c)*self.t2s(i,l,b,d) + self.t1s(j,a)*self.t1s(k,d)*self.t2s(i,l,b,c) \
#                 + self.t1s(j,b)*self.t1s(k,c)*self.t2s(i,l,a,d) - self.t1s(j,b)*self.t1s(k,d)*self.t2s(i,l,a,c) + self.t1s(j,c)*self.t1s(k,d)*self.t2s(i,l,a,b) \
#                 - self.t1s(j,a)*self.t1s(l,b)*self.t2s(i,k,c,d) + self.t1s(j,a)*self.t1s(l,c)*self.t2s(i,k,b,d) - self.t1s(j,a)*self.t1s(l,d)*self.t2s(i,k,b,c) \
#                 - self.t1s(j,b)*self.t1s(l,c)*self.t2s(i,k,a,d) + self.t1s(j,b)*self.t1s(l,d)*self.t2s(i,k,a,c) - self.t1s(j,c)*self.t1s(l,d)*self.t2s(i,k,a,b) \
#                 + self.t1s(k,a)*self.t1s(l,b)*self.t2s(i,j,c,d) - self.t1s(k,a)*self.t1s(l,c)*self.t2s(i,j,b,d) + self.t1s(k,a)*self.t1s(l,d)*self.t2s(i,j,b,c) \
#                 + self.t1s(k,b)*self.t1s(l,c)*self.t2s(i,j,a,d) - self.t1s(k,b)*self.t1s(l,d)*self.t2s(i,j,a,c) + self.t1s(k,c)*self.t1s(l,d)*self.t2s(i,j,a,b) \
#                 - self.t1s(j,a)*self.t1s(i,b)*self.t2s(k,l,c,d) + self.t1s(j,a)*self.t1s(i,c)*self.t2s(k,l,b,d) - self.t1s(j,a)*self.t1s(i,d)*self.t2s(k,l,b,c) \
#                 - self.t1s(j,b)*self.t1s(i,c)*self.t2s(k,l,a,d) + self.t1s(j,b)*self.t1s(i,d)*self.t2s(k,l,a,c) - self.t1s(j,c)*self.t1s(i,d)*self.t2s(k,l,a,b) \
#                 + self.t1s(k,a)*self.t1s(i,b)*self.t2s(j,l,c,d) - self.t1s(k,a)*self.t1s(i,c)*self.t2s(j,l,b,d) + self.t1s(k,a)*self.t1s(i,d)*self.t2s(j,l,b,c) \
#                 + self.t1s(k,b)*self.t1s(i,c)*self.t2s(j,l,a,d) - self.t1s(k,b)*self.t1s(i,d)*self.t2s(j,l,a,c) + self.t1s(k,c)*self.t1s(i,d)*self.t2s(j,l,a,b) \
#                 - self.t1s(l,a)*self.t1s(i,b)*self.t2s(j,k,c,d) + self.t1s(l,a)*self.t1s(i,c)*self.t2s(j,k,b,d) - self.t1s(l,a)*self.t1s(i,d)*self.t2s(j,k,b,c) \
#                 - self.t1s(l,b)*self.t1s(i,c)*self.t2s(j,k,a,d) + self.t1s(l,b)*self.t1s(i,d)*self.t2s(j,k,a,c) - self.t1s(l,c)*self.t1s(i,d)*self.t2s(j,k,a,b) \
#                 - self.t1s(k,a)*self.t1s(j,b)*self.t2s(i,l,c,d) + self.t1s(k,a)*self.t1s(j,c)*self.t2s(i,l,b,d) - self.t1s(k,a)*self.t1s(j,d)*self.t2s(i,l,b,c) \
#                 - self.t1s(k,b)*self.t1s(j,c)*self.t2s(i,l,a,d) + self.t1s(k,b)*self.t1s(j,d)*self.t2s(i,l,a,c) - self.t1s(k,c)*self.t1s(j,d)*self.t2s(i,l,a,b) \
#                 + self.t1s(l,a)*self.t1s(j,b)*self.t2s(i,k,c,d) - self.t1s(l,a)*self.t1s(j,c)*self.t2s(i,k,b,d) + self.t1s(l,a)*self.t1s(j,d)*self.t2s(i,k,b,c) \
#                 + self.t1s(l,b)*self.t1s(j,c)*self.t2s(i,k,a,d) - self.t1s(l,b)*self.t1s(j,d)*self.t2s(i,k,a,c) + self.t1s(l,c)*self.t1s(j,d)*self.t2s(i,k,a,b) \
#                 - self.t1s(l,a)*self.t1s(k,b)*self.t2s(i,j,c,d) + self.t1s(l,a)*self.t1s(k,c)*self.t2s(i,j,b,d) - self.t1s(l,a)*self.t1s(k,d)*self.t2s(i,j,b,c) \
#                 - self.t1s(l,b)*self.t1s(k,c)*self.t2s(i,j,a,d) + self.t1s(l,b)*self.t1s(k,d)*self.t2s(i,j,a,c) - self.t1s(l,c)*self.t1s(k,d)*self.t2s(i,j,a,b)
#        return t1xt1xt2

    def t1xt1xt1xt1aaab(self, i, j, k, l, a, b, c, d):
        t1xt1xt1xt1 = self.t1[i][a]*self.t1[j][b]*self.t1[k][c]*self.t1[l][d] \
                    - self.t1[i][a]*self.t1[j][c]*self.t1[k][b]*self.t1[l][d] \
                    - self.t1[i][b]*self.t1[j][a]*self.t1[k][c]*self.t1[l][d] \
                    + self.t1[i][b]*self.t1[j][c]*self.t1[k][a]*self.t1[l][d] \
                    + self.t1[i][c]*self.t1[j][a]*self.t1[k][b]*self.t1[l][d] \
                    - self.t1[i][c]*self.t1[j][b]*self.t1[k][a]*self.t1[l][d] 
        return t1xt1xt1xt1

    def t1xt1xt1xt1aabb(self, i, j, k, l, a, b, c, d):
        t1xt1xt1xt1 = self.t1[i][a]*self.t1[j][b]*self.t1[k][c]*self.t1[l][d] - self.t1[i][a]*self.t1[j][b]*self.t1[k][d]*self.t1[l][c] 
        return t1xt1xt1xt1

#    def t1xt1xt1xt1(self, i_spt, j_spt, k_spt, l_spt, a_spt, b_spt, c_spt, d_spt, spin='aaab'):
#        i = spinMO(i_spt, 'a')
#        j = spinMO(j_spt, 'a')
#        l = spinMO(l_spt, 'b')
#        a = spinMO(a_spt, 'a')
#        b = spinMO(b_spt, 'a')
#        d = spinMO(d_spt, 'b')
#        if spin == 'aaab':
#            k = spinMO(k_spt, 'a')
#            c = spinMO(c_spt, 'a')
#        elif spin == 'aabb':
#            k = spinMO(k_spt, 'b')
#            c = spinMO(c_spt, 'b')
#        else:
#            assert False
#
#        t1xt1xt1xt1 = self.t1s(i,a)*self.t1s(j,b)*self.t1s(k,c)*self.t1s(l,d) - self.t1s(i,a)*self.t1s(j,b)*self.t1s(k,d)*self.t1s(l,c) \
#                    - self.t1s(i,a)*self.t1s(j,c)*self.t1s(k,b)*self.t1s(l,d) + self.t1s(i,a)*self.t1s(j,c)*self.t1s(k,d)*self.t1s(l,b) \
#                    + self.t1s(i,a)*self.t1s(j,d)*self.t1s(k,b)*self.t1s(l,c) - self.t1s(i,a)*self.t1s(j,d)*self.t1s(k,c)*self.t1s(l,b) \
#                    - self.t1s(i,b)*self.t1s(j,a)*self.t1s(k,c)*self.t1s(l,d) + self.t1s(i,b)*self.t1s(j,a)*self.t1s(k,d)*self.t1s(l,c) \
#                    + self.t1s(i,b)*self.t1s(j,c)*self.t1s(k,a)*self.t1s(l,d) - self.t1s(i,b)*self.t1s(j,c)*self.t1s(k,d)*self.t1s(l,a) \
#                    - self.t1s(i,b)*self.t1s(j,d)*self.t1s(k,a)*self.t1s(l,c) + self.t1s(i,b)*self.t1s(j,d)*self.t1s(k,c)*self.t1s(l,a) \
#                    + self.t1s(i,c)*self.t1s(j,a)*self.t1s(k,b)*self.t1s(l,d) - self.t1s(i,c)*self.t1s(j,a)*self.t1s(k,d)*self.t1s(l,b) \
#                    - self.t1s(i,c)*self.t1s(j,b)*self.t1s(k,a)*self.t1s(l,d) + self.t1s(i,c)*self.t1s(j,b)*self.t1s(k,d)*self.t1s(l,a) \
#                    + self.t1s(i,c)*self.t1s(j,d)*self.t1s(k,a)*self.t1s(l,b) - self.t1s(i,c)*self.t1s(j,d)*self.t1s(k,b)*self.t1s(l,a) \
#                    - self.t1s(i,d)*self.t1s(j,a)*self.t1s(k,b)*self.t1s(l,c) + self.t1s(i,d)*self.t1s(j,a)*self.t1s(k,c)*self.t1s(l,b) \
#                    + self.t1s(i,d)*self.t1s(j,b)*self.t1s(k,a)*self.t1s(l,c) - self.t1s(i,d)*self.t1s(j,b)*self.t1s(k,c)*self.t1s(l,a) \
#                    - self.t1s(i,d)*self.t1s(j,c)*self.t1s(k,a)*self.t1s(l,b) + self.t1s(i,d)*self.t1s(j,c)*self.t1s(k,b)*self.t1s(l,a) 
#        return t1xt1xt1xt1

#    def fraction_t4_t2xt2_test(self, c4aaab, c4aabb):
##        from scipy.stats import pearsonr
##        rg_i = [ p for p in range(self.nocc) ]  
##        rg_j = rg_i  
##        rg_k = rg_i  
##        rg_l = rg_i  
##        rg_a = [ p for p in range(self.nvir) ]  
##        rg_b = rg_a
##        rg_c = rg_a
##        rg_d = rg_a
##
##        t4 = []
##        t2xt2 = []
##        for d, c, b, a, l, k, j, i in itertools.product(rg_d, rg_c, rg_b, rg_a, rg_l, rg_k, rg_j, rg_i): 
###            c4tmp = 0.0
###            if i != j and a != b and k != l and c != d:
###                idx_o = [i, j, k]
###                idx_v = [a, b, c]
###                ip, jp, kp, po = reorder(idx_o)
###                ap, bp, cp, pv = reorder(idx_v)
###                ijkabc = self.idx.T(ip, jp, kp, ap, bp, cp) 
###                ld   = self.idx.S(l, d) 
###
###                c4tmp = po*pv*c4aaab[ijkabc,ld]
###            t4=abs(self.t4aaab[i][j][k][l][a][b][c][d])
###            t2xt2=abs(self.t2xt2aaab(i,j,k,l,a,b,c,d))
###            if(t4 > 0.00000000001 and t2xt2 > 0.00000000001):
###                print i,j,k,l,a,b,c,d,t4, t2xt2 
###            print i, j, k, l, a, b, c, d, self.t4aaab[i][j][k][l][a][b][c][d], self.t2xt2aaab(i,j,k,l,a,b,c,d), c4tmp
##            t4.append(abs(self.t4aaab[i][j][k][l][a][b][c][d]))
##            t2xt2.append(abs(self.t2xt2aaab(i,j,k,l,a,b,c,d)))
##
##        corr,p_value=pearsonr(t4,t2xt2)
##        print 't4aaab'
##        print 'corr =', corr
##        print 'p_val=', p_value
##        print 'max t4, t2xt2 =', max(t4), max(t2xt2) 
##
##        t4p = []
##        t2xt2p = []
##        for d, c, b, a, l, k, j, i in itertools.product(rg_d, rg_c, rg_b, rg_a, rg_l, rg_k, rg_j, rg_i): 
###            c4tmp = 0.0
###            if i != j and a != b and k != l and c != d:
###                idx_o = [i, j]
###                idx_v = [a, b]
###                ip, jp, po1 = reorder(idx_o)
###                ap, bp, pv1 = reorder(idx_v)
###                ijab = self.idx.D(ip, jp, ap, bp) 
###                idx_o = [k, l]
###                idx_v = [c, d]
###                kp, lp, po2 = reorder(idx_o)
###                cp, dp, pv2 = reorder(idx_v)
###                klcd = self.idx.D(kp, lp, cp, dp) 
###                parity = po1 * po2 * pv1 * pv2
###
###                c4tmp = parity*c4aabb[ijab,klcd]
###            print i, j, k, l, a, b, c, d, self.t4aabb[i][j][k][l][a][b][c][d], self.t2xt2aabb(i,j,k,l,a,b,c,d), c4tmp
###            t4=abs(self.t4aabb[i][j][k][l][a][b][c][d])
###            t2xt2=abs(self.t2xt2aabb(i,j,k,l,a,b,c,d))
###            if(t4 > 0.00000000001 and t2xt2 > 0.00000000001):
###                print i,j,k,l,a,b,c,d,t4, t2xt2 
##
##            t4.append(abs(self.t4aabb[i][j][k][l][a][b][c][d]))
##            t2xt2.append(abs(self.t2xt2aabb(i,j,k,l,a,b,c,d)))
##            t4p.append(abs(self.t4aabb[i][j][k][l][a][b][c][d]))
##            t2xt2p.append(abs(self.t2xt2aabb(i,j,k,l,a,b,c,d)))
##
##        corr,p_value=pearsonr(t4p,t2xt2p)
##        print 't4aabb'
##        print 'corr =', corr
##        print 'p_val=', p_value
##        print 'max t4, t2xt2 =', max(t4p), max(t2xt2p)
##
##        corr,p_value=pearsonr(t4,t2xt2)
##        print 't4 all'
##        print 'corr =', corr
##        print 'p_val=', p_value
##        print 'max t4, t2xt2 =', max(t4), max(t2xt2) 
#
#        from t4_screen import t4_screen
#        t4scr = t4_screen(self.nocc, self.nvir)
#        t4scr.argsort(self.t2ab)
#        t4scr.gen_idx(10)
#
##        for itmp in range(5):
##            i, j, a, b = t4scr.argsort_t2ab[itmp]
##            i = t4scr.argsort_t2ab[0][itmp][0]
##            j = t4scr.argsort_t2ab[0][itmp][1]
##            a = t4scr.argsort_t2ab[0][itmp][2]
##            b = t4scr.argsort_t2ab[0][itmp][3]
##            print i,j,a,b, self.t2ab[i][j][a][b]
#
#        for i,j,k,l,a,b,c,d in t4scr.idx_t4aabb:
#            t4=abs(self.t4aabb[i][j][k][l][a][b][c][d])
#            t2xt2=abs(self.t2xt2aabb(i,j,k,l,a,b,c,d))
#            if(t4 > 0.00000000001 or t2xt2 > 0.00000000001):
#                print (i,j,k,l,a,b,c,d,t4, t2xt2 )
#

class fci_to_cc_c:
    def __init__(self, nocc, nvir, idx, c0):
        self.dbg = True 
        self.nocc = nocc
        self.nvir = nvir
        self.idx  = idx 
        self.t0 = numpy.zeros((1), dtype=numpy.float64)
        self.t0[0] = c0[0] 
#        assert self.t0[0] == 1.0     # check intermed normalization
        self.t1 = None 
        self.t2 = None 
        self.t3 = None 
        self.t3p= None 
        self.t4 = None 
        self.t2aa   = None 
        self.t2ab   = None      #  t2 = t2ab in restricted ccsd
        self.t3aaa  = None 
        self.t3aab  = None 
        self.t4aaab = None 
        self.t4aabb = None 

    def c1_to_t1(self, c1):
        nocc = self.nocc
        nvir = self.nvir
        self.t1 = numpy.zeros((nocc,nvir), dtype=numpy.float64)
        _ccsd.libcc.c1_to_t1(self.t1.ctypes.data_as(ctypes.c_void_p),
                             c1.ctypes.data_as(ctypes.c_void_p),
                             ctypes.c_int(nocc),ctypes.c_int(nvir))

    def c2_to_t2(self, c2aa, c2ab):
        nocc = self.nocc
        nvir = self.nvir
        self.t2aa = numpy.zeros((nocc,nocc,nvir,nvir), dtype=numpy.float64)
        self.t2ab = numpy.zeros((nocc,nocc,nvir,nvir), dtype=numpy.float64)
        _ccsd.libcc.c2_to_t2(self.t2aa.ctypes.data_as(ctypes.c_void_p),
                             self.t2ab.ctypes.data_as(ctypes.c_void_p),
                             c2aa.ctypes.data_as(ctypes.c_void_p),
                             c2ab.ctypes.data_as(ctypes.c_void_p),
                             self.t1.ctypes.data_as(ctypes.c_void_p),
                             ctypes.c_int(nocc),ctypes.c_int(nvir))

    def c3_to_t3(self, c3aaa, c3aab, numzero=1e-5):
        nocc = self.nocc
        nvir = self.nvir
#        self.t3    = numpy.zeros((nocc,nocc,nocc,nvir,nvir,nvir), dtype=numpy.float64)

#        self.t3p   = numpy.zeros((nocc,nocc,nocc,nvir,nvir,nvir), dtype=numpy.float64)

        print('start extracting t3 from c3')
        print('shape of c3aaa, c3aab', c3aaa.shape, c3aab.shape) 

        self.t3aaa = numpy.zeros((nocc,nocc,nocc,nvir,nvir,nvir), dtype=numpy.float64)
        self.t3aab = numpy.zeros((nocc,nocc,nocc,nvir,nvir,nvir), dtype=numpy.float64)
        _ccsd.libcc.c3_to_t3(self.t3aaa.ctypes.data_as(ctypes.c_void_p),
                             self.t3aab.ctypes.data_as(ctypes.c_void_p),
                             c3aaa.ctypes.data_as(ctypes.c_void_p),
                             c3aab.ctypes.data_as(ctypes.c_void_p),
                             self.t1.ctypes.data_as(ctypes.c_void_p),
                             self.t2aa.ctypes.data_as(ctypes.c_void_p),
                             self.t2ab.ctypes.data_as(ctypes.c_void_p),
                             ctypes.c_int(nocc),ctypes.c_int(nvir),
                             ctypes.c_double(numzero))

        print('end extracting t3 from c3')
#lsh TODO: don't make t3aaa, t3aab to make t3 and t3p
#        self.t3_rccsd()

    def c3_to_t3_ecT(self, c3aaa, c3aab, nc_ref, nvir_ref, numzero=1e-5):
        nocc = self.nocc
        nvir = self.nvir
#        self.t3    = numpy.zeros((nocc,nocc,nocc,nvir,nvir,nvir), dtype=numpy.float64)

#        self.t3p   = numpy.zeros((nocc,nocc,nocc,nvir,nvir,nvir), dtype=numpy.float64)

        print('start extracting t3 from c3')
        print('shape of c3aaa, c3aab', c3aaa.shape, c3aab.shape) 

        self.t3aaa = numpy.zeros((nocc,nocc,nocc,nvir,nvir,nvir), dtype=numpy.float64)
        self.t3aab = numpy.zeros((nocc,nocc,nocc,nvir,nvir,nvir), dtype=numpy.float64)
        _ccsd.libcc.c3_to_t3_ecT(self.t3aaa.ctypes.data_as(ctypes.c_void_p),
                             self.t3aab.ctypes.data_as(ctypes.c_void_p),
                             c3aaa.ctypes.data_as(ctypes.c_void_p),
                             c3aab.ctypes.data_as(ctypes.c_void_p),
                             self.t1.ctypes.data_as(ctypes.c_void_p),
                             self.t2aa.ctypes.data_as(ctypes.c_void_p),
                             self.t2ab.ctypes.data_as(ctypes.c_void_p),
                             ctypes.c_int(nc_ref),ctypes.c_int(nvir_ref),
                             ctypes.c_int(nocc),ctypes.c_int(nvir),
                             ctypes.c_double(numzero))

        print('end extracting t3 from c3')

    def c3_to_t3_thresh(self, c3aaa, c3aab, numzero=5e-5):
        nocc = self.nocc
        nvir = self.nvir
#        self.t3    = numpy.zeros((nocc,nocc,nocc,nvir,nvir,nvir), dtype=numpy.float64)

#        self.t3p   = numpy.zeros((nocc,nocc,nocc,nvir,nvir,nvir), dtype=numpy.float64)

        print('start extracting t3 from c3')
        print('shape of c3aaa, c3aab', c3aaa.shape, c3aab.shape) 

        self.t3aaa = numpy.zeros((nocc,nocc,nocc,nvir,nvir,nvir), dtype=numpy.float64)
        self.t3aab = numpy.zeros((nocc,nocc,nocc,nvir,nvir,nvir), dtype=numpy.float64)
        _ccsd.libcc.c3_to_t3_thresh(self.t3aaa.ctypes.data_as(ctypes.c_void_p),
                             self.t3aab.ctypes.data_as(ctypes.c_void_p),
                             c3aaa.ctypes.data_as(ctypes.c_void_p),
                             c3aab.ctypes.data_as(ctypes.c_void_p),
                             self.t1.ctypes.data_as(ctypes.c_void_p),
                             self.t2aa.ctypes.data_as(ctypes.c_void_p),
                             self.t2ab.ctypes.data_as(ctypes.c_void_p),
                             ctypes.c_int(nocc),ctypes.c_int(nvir),
                             ctypes.c_double(numzero))

    def t3_rccsd(self):
        rg_i = [ p for p in range(self.nocc) ]  
        rg_j = rg_i  
        rg_k = rg_i  
        rg_a = [ p for p in range(self.nvir) ]  
        rg_b = rg_a
        rg_c = rg_a

        for c, b, a, k, j, i in itertools.product(rg_c, rg_b, rg_a, rg_k, rg_j, rg_i): 
            self.t3[i][j][k][a][b][c]  = self.t3aab[i][j][k][a][b][c].copy()
            self.t3[i][j][k][a][b][c] += self.t3aab[i][k][j][a][c][b].copy()
            self.t3p[i][j][k][a][b][c]  = self.t3aaa[i][j][k][a][b][c].copy()
            self.t3p[i][j][k][a][b][c] += self.t3[i][j][k][a][b][c].copy()
            self.t3p[i][j][k][a][b][c] += self.t3aab[j][k][i][b][c][a].copy()
            self.t3p[i][j][k][a][b][c] /= 0.5 

    def c4_to_t4(self, c4aaab, c4aabb, numzero=1e-5):
        nocc = self.nocc
        nvir = self.nvir
#        self.t4 = numpy.zeros((nocc,nocc,nocc,nocc,nvir,nvir,nvir,nvir), dtype=numpy.float64)
        self.t4aaab = numpy.zeros((nocc,nocc,nocc,nocc,nvir,nvir,nvir,nvir), dtype=numpy.float64)
        self.t4aabb = numpy.zeros((nocc,nocc,nocc,nocc,nvir,nvir,nvir,nvir), dtype=numpy.float64)

        _ccsd.libcc.c4_to_t4(self.t4aaab.ctypes.data_as(ctypes.c_void_p),
                             self.t4aabb.ctypes.data_as(ctypes.c_void_p),
                             c4aaab.ctypes.data_as(ctypes.c_void_p),
                             c4aabb.ctypes.data_as(ctypes.c_void_p),
                             self.t1.ctypes.data_as(ctypes.c_void_p),
                             self.t2aa.ctypes.data_as(ctypes.c_void_p),
                             self.t2ab.ctypes.data_as(ctypes.c_void_p),
                             self.t3aaa.ctypes.data_as(ctypes.c_void_p),
                             self.t3aab.ctypes.data_as(ctypes.c_void_p),
                             ctypes.c_int(nocc),ctypes.c_int(nvir),
                             ctypes.c_double(numzero))

#lsh TODO: don't make t4aaaa, t4aaab to make t4
        self.t4_rccsd()


    def c4_to_t4_test(self, c4aaab, c4aabb, numzero=1e-5):
        nocc = self.nocc
        nvir = self.nvir
#        self.t4 = numpy.zeros((nocc,nocc,nocc,nocc,nvir,nvir,nvir,nvir), dtype=numpy.float64)
        self.t4aaab = numpy.zeros((nocc,nocc,nocc,nocc,nvir,nvir,nvir,nvir), dtype=numpy.float64)
        self.t4aabb = numpy.zeros((nocc,nocc,nocc,nocc,nvir,nvir,nvir,nvir), dtype=numpy.float64)

        _ccsd.libcc.c4_to_t4_test(self.t4aaab.ctypes.data_as(ctypes.c_void_p),
                             self.t4aabb.ctypes.data_as(ctypes.c_void_p),
                             c4aaab.ctypes.data_as(ctypes.c_void_p),
                             c4aabb.ctypes.data_as(ctypes.c_void_p),
                             self.t1.ctypes.data_as(ctypes.c_void_p),
                             self.t2aa.ctypes.data_as(ctypes.c_void_p),
                             self.t2ab.ctypes.data_as(ctypes.c_void_p),
                             self.t3aaa.ctypes.data_as(ctypes.c_void_p),
                             self.t3aab.ctypes.data_as(ctypes.c_void_p),
                             ctypes.c_int(nocc),ctypes.c_int(nvir),
                             ctypes.c_double(numzero))

    def c4_to_t4_thresh(self, c4aaab, c4aabb, numzero=5e-5):
        nocc = self.nocc
        nvir = self.nvir
#        self.t4 = numpy.zeros((nocc,nocc,nocc,nocc,nvir,nvir,nvir,nvir), dtype=numpy.float64)
        self.t4aaab = numpy.zeros((nocc,nocc,nocc,nocc,nvir,nvir,nvir,nvir), dtype=numpy.float64)
        self.t4aabb = numpy.zeros((nocc,nocc,nocc,nocc,nvir,nvir,nvir,nvir), dtype=numpy.float64)

        _ccsd.libcc.c4_to_t4_thresh(self.t4aaab.ctypes.data_as(ctypes.c_void_p),
                             self.t4aabb.ctypes.data_as(ctypes.c_void_p),
                             c4aaab.ctypes.data_as(ctypes.c_void_p),
                             c4aabb.ctypes.data_as(ctypes.c_void_p),
                             self.t1.ctypes.data_as(ctypes.c_void_p),
                             self.t2aa.ctypes.data_as(ctypes.c_void_p),
                             self.t2ab.ctypes.data_as(ctypes.c_void_p),
                             self.t3aaa.ctypes.data_as(ctypes.c_void_p),
                             self.t3aab.ctypes.data_as(ctypes.c_void_p),
                             ctypes.c_int(nocc),ctypes.c_int(nvir),
                             ctypes.c_double(numzero))

#lsh TODO: don't make t4aaaa, t4aaab to make t4
        self.t4_rccsd()

    def t4_rccsd(self):
#        rg_i = [ p for p in range(self.nocc) ]  
#        rg_j = rg_i  
#        rg_k = rg_i  
#        rg_l = rg_i  
#        rg_a = [ p for p in range(self.nvir) ]  
#        rg_b = rg_a
#        rg_c = rg_a
#        rg_d = rg_a
#
#        for b, a, d, c, j, i, l, k in itertools.product(rg_b, rg_a, rg_d, rg_c, rg_j, rg_i, rg_l, rg_k): 
#            self.t4[k][l][i][j][c][d][a][b]  = self.t4aaab[k][l][i][j][c][d][a][b].copy()
#            self.t4[k][l][i][j][c][d][a][b] += self.t4aaab[k][l][j][i][c][d][b][a].copy()
#            self.t4[k][l][i][j][c][d][a][b] += self.t4aabb[k][i][l][j][c][a][d][b].copy()
#            self.t4[k][l][i][j][c][d][a][b] += self.t4aabb[l][i][k][j][d][a][c][b].copy()
#            self.t4[k][l][i][j][c][d][a][b] *= 0.5 
        self.t4 = self.t4aaab.copy()
        self.t4+= self.t4aaab.copy().transpose(0,1,3,2,4,5,7,6)
        self.t4+= self.t4aabb.copy().transpose(0,2,1,3,4,6,5,7)
        self.t4+= self.t4aabb.copy().transpose(1,2,0,3,5,6,4,7)

    def read_t3aab(self, path='t3aab.csv'):
        data = pd.read_csv(path)
        self.t3aab = numpy.array(list(data.loc[:,"t3aab"]))
        self.t3aab = self.t3aab.reshape(self.nocc,self.nocc,self.nocc,self.nvir,self.nvir,self.nvir)

    def read_t1_t3c(self, path='t1_t3c.csv'):
        data = pd.read_csv(path)
        t1_t3c = numpy.array(list(data.loc[:,"t1_t3c"]))
        t1_t3c = t1_t3c.reshape(self.nocc,self.nvir)
        return t1_t3c
 
    def read_t2_t4c(self, path='t2_t4c.csv'):
        data = pd.read_csv(path)
        t2_t4c = numpy.array(list(data.loc[:,"t2_t4c"]))
        t2_t4c = t2_t4c.reshape(self.nocc,self.nocc,self.nvir,self.nvir)
        return t2_t4c

#    def fraction_t4_t2xt2_test(self, c4aaab, c4aabb):
##        from scipy.stats import pearsonr
##        rg_i = [ p for p in range(self.nocc) ]  
##        rg_j = rg_i  
##        rg_k = rg_i  
##        rg_l = rg_i  
##        rg_a = [ p for p in range(self.nvir) ]  
##        rg_b = rg_a
##        rg_c = rg_a
##        rg_d = rg_a
##
##        t4 = []
##        t2xt2 = []
##        for d, c, b, a, l, k, j, i in itertools.product(rg_d, rg_c, rg_b, rg_a, rg_l, rg_k, rg_j, rg_i): 
###            c4tmp = 0.0
###            if i != j and a != b and k != l and c != d:
###                idx_o = [i, j, k]
###                idx_v = [a, b, c]
###                ip, jp, kp, po = reorder(idx_o)
###                ap, bp, cp, pv = reorder(idx_v)
###                ijkabc = self.idx.T(ip, jp, kp, ap, bp, cp) 
###                ld   = self.idx.S(l, d) 
###
###                c4tmp = po*pv*c4aaab[ijkabc,ld]
###            t4=abs(self.t4aaab[i][j][k][l][a][b][c][d])
###            t2xt2=abs(self.t2xt2aaab(i,j,k,l,a,b,c,d))
###            if(t4 > 0.00000000001 and t2xt2 > 0.00000000001):
###                print i,j,k,l,a,b,c,d,t4, t2xt2 
###            print i, j, k, l, a, b, c, d, self.t4aaab[i][j][k][l][a][b][c][d], self.t2xt2aaab(i,j,k,l,a,b,c,d), c4tmp
##            t4.append(abs(self.t4aaab[i][j][k][l][a][b][c][d]))
##            t2xt2.append(abs(self.t2xt2aaab(i,j,k,l,a,b,c,d)))
##
##        corr,p_value=pearsonr(t4,t2xt2)
##        print 't4aaab'
##        print 'corr =', corr
##        print 'p_val=', p_value
##        print 'max t4, t2xt2 =', max(t4), max(t2xt2) 
##
##        t4p = []
##        t2xt2p = []
##        for d, c, b, a, l, k, j, i in itertools.product(rg_d, rg_c, rg_b, rg_a, rg_l, rg_k, rg_j, rg_i): 
###            c4tmp = 0.0
###            if i != j and a != b and k != l and c != d:
###                idx_o = [i, j]
###                idx_v = [a, b]
###                ip, jp, po1 = reorder(idx_o)
###                ap, bp, pv1 = reorder(idx_v)
###                ijab = self.idx.D(ip, jp, ap, bp) 
###                idx_o = [k, l]
###                idx_v = [c, d]
###                kp, lp, po2 = reorder(idx_o)
###                cp, dp, pv2 = reorder(idx_v)
###                klcd = self.idx.D(kp, lp, cp, dp) 
###                parity = po1 * po2 * pv1 * pv2
###
###                c4tmp = parity*c4aabb[ijab,klcd]
###            print i, j, k, l, a, b, c, d, self.t4aabb[i][j][k][l][a][b][c][d], self.t2xt2aabb(i,j,k,l,a,b,c,d), c4tmp
###            t4=abs(self.t4aabb[i][j][k][l][a][b][c][d])
###            t2xt2=abs(self.t2xt2aabb(i,j,k,l,a,b,c,d))
###            if(t4 > 0.00000000001 and t2xt2 > 0.00000000001):
###                print i,j,k,l,a,b,c,d,t4, t2xt2 
##
##            t4.append(abs(self.t4aabb[i][j][k][l][a][b][c][d]))
##            t2xt2.append(abs(self.t2xt2aabb(i,j,k,l,a,b,c,d)))
##            t4p.append(abs(self.t4aabb[i][j][k][l][a][b][c][d]))
##            t2xt2p.append(abs(self.t2xt2aabb(i,j,k,l,a,b,c,d)))
##
##        corr,p_value=pearsonr(t4p,t2xt2p)
##        print 't4aabb'
##        print 'corr =', corr
##        print 'p_val=', p_value
##        print 'max t4, t2xt2 =', max(t4p), max(t2xt2p)
##
##        corr,p_value=pearsonr(t4,t2xt2)
##        print 't4 all'
##        print 'corr =', corr
##        print 'p_val=', p_value
##        print 'max t4, t2xt2 =', max(t4), max(t2xt2) 
#
#        from t4_screen import t4_screen
#        t4scr = t4_screen(self.nocc, self.nvir)
#        t4scr.argsort(self.t2ab)
#        t4scr.gen_idx(10)
#
##        for itmp in range(5):
##            i, j, a, b = t4scr.argsort_t2ab[itmp]
##            i = t4scr.argsort_t2ab[0][itmp][0]
##            j = t4scr.argsort_t2ab[0][itmp][1]
##            a = t4scr.argsort_t2ab[0][itmp][2]
##            b = t4scr.argsort_t2ab[0][itmp][3]
##            print i,j,a,b, self.t2ab[i][j][a][b]
#
#        for i,j,k,l,a,b,c,d in t4scr.idx_t4aabb:
#            t4=abs(self.t4aabb[i][j][k][l][a][b][c][d])
#            t2xt2=abs(self.t2xt2aabb(i,j,k,l,a,b,c,d))
#            if(t4 > 0.00000000001 or t2xt2 > 0.00000000001):
#                print (i,j,k,l,a,b,c,d,t4, t2xt2 )
#

class ci2cc_mem:
    def __init__(self, nocc_corr, nvir_corr, nocc_cas, nvir_cas, nocc_iact, idx, c0):
        self.nocc_corr = nocc_corr
        self.nvir_corr = nvir_corr
        self.nocc_cas  = nocc_cas 
        self.nvir_cas  = nvir_cas 
        self.nocc_iact = nocc_iact
        self.idx  = idx 
        self.t1 = None 
        self.t2 = None 
        self.t3 = None 
        self.t3p= None 
        self.t2aa   = None 
        self.t2ab   = None
        self.t3aaa  = None 
        self.t3aab  = None 

    def c1_to_t1(self, c1):
        self.t1 = numpy.zeros((self.nocc_corr,self.nvir_corr), dtype=numpy.float64)
        _ccsd.libcc.c1_to_t1_mem(self.t1.ctypes.data_as(ctypes.c_void_p),
                             c1.ctypes.data_as(ctypes.c_void_p),
                             ctypes.c_int(self.nocc_cas),
                             ctypes.c_int(self.nvir_cas),
                             ctypes.c_int(self.nvir_corr),
                             ctypes.c_int(self.nocc_iact))

    def c2_to_t2(self, c2aa, c2ab, numzero=1e-12):
        nocc_corr = self.nocc_corr
        nvir_corr = self.nvir_corr
        self.t2aa = numpy.zeros((nocc_corr,nocc_corr,nvir_corr,nvir_corr), dtype=numpy.float64)
        self.t2ab = numpy.zeros((nocc_corr,nocc_corr,nvir_corr,nvir_corr), dtype=numpy.float64)
        _ccsd.libcc.c2_to_t2_mem(self.t2aa.ctypes.data_as(ctypes.c_void_p),
                             self.t2ab.ctypes.data_as(ctypes.c_void_p),
                             c2aa.ctypes.data_as(ctypes.c_void_p),
                             c2ab.ctypes.data_as(ctypes.c_void_p),
                             self.t1.ctypes.data_as(ctypes.c_void_p),
                             ctypes.c_int(self.nocc_cas),
                             ctypes.c_int(self.nvir_cas),
                             ctypes.c_int(self.nocc_corr),
                             ctypes.c_int(self.nvir_corr),
                             ctypes.c_int(self.nocc_iact),
                             ctypes.c_double(numzero))

    def c3_to_t3(self, c3aaa, c3aab, numzero=1e-12):
        raise NotImplementedError
#        nocc = self.nocc
#        nvir = self.nvir
##        self.t3    = numpy.zeros((nocc,nocc,nocc,nvir,nvir,nvir), dtype=numpy.float64)
#
##        self.t3p   = numpy.zeros((nocc,nocc,nocc,nvir,nvir,nvir), dtype=numpy.float64)
#
#        self.t3aaa = numpy.zeros((nocc,nocc,nocc,nvir,nvir,nvir), dtype=numpy.float64)
#        self.t3aab = numpy.zeros((nocc,nocc,nocc,nvir,nvir,nvir), dtype=numpy.float64)
#        _ccsd.libcc.c3_to_t3(self.t3aaa.ctypes.data_as(ctypes.c_void_p),
#                             self.t3aab.ctypes.data_as(ctypes.c_void_p),
#                             c3aaa.ctypes.data_as(ctypes.c_void_p),
#                             c3aab.ctypes.data_as(ctypes.c_void_p),
#                             self.t1.ctypes.data_as(ctypes.c_void_p),
#                             self.t2aa.ctypes.data_as(ctypes.c_void_p),
#                             self.t2ab.ctypes.data_as(ctypes.c_void_p),
#                             ctypes.c_int(nocc),ctypes.c_int(nvir),
#                             ctypes.c_double(numzero))

#        self.t3_rccsd()

    def c3_to_t3_ecT(self, c3aaa, c3aab, nc_ref, nvir_ref, numzero=1e-5):
        raise NotImplementedError
#        nocc = self.nocc
#        nvir = self.nvir
##        self.t3    = numpy.zeros((nocc,nocc,nocc,nvir,nvir,nvir), dtype=numpy.float64)
#
##        self.t3p   = numpy.zeros((nocc,nocc,nocc,nvir,nvir,nvir), dtype=numpy.float64)
#
#        print('start extracting t3 from c3')
#        print('shape of c3aaa, c3aab', c3aaa.shape, c3aab.shape) 
#
#        self.t3aaa = numpy.zeros((nocc,nocc,nocc,nvir,nvir,nvir), dtype=numpy.float64)
#        self.t3aab = numpy.zeros((nocc,nocc,nocc,nvir,nvir,nvir), dtype=numpy.float64)
#        _ccsd.libcc.c3_to_t3_ecT(self.t3aaa.ctypes.data_as(ctypes.c_void_p),
#                             self.t3aab.ctypes.data_as(ctypes.c_void_p),
#                             c3aaa.ctypes.data_as(ctypes.c_void_p),
#                             c3aab.ctypes.data_as(ctypes.c_void_p),
#                             self.t1.ctypes.data_as(ctypes.c_void_p),
#                             self.t2aa.ctypes.data_as(ctypes.c_void_p),
#                             self.t2ab.ctypes.data_as(ctypes.c_void_p),
#                             ctypes.c_int(nc_ref),ctypes.c_int(nvir_ref),
#                             ctypes.c_int(nocc),ctypes.c_int(nvir),
#                             ctypes.c_double(numzero))
#
#        print('end extracting t3 from c3')

    def t3_rccsd(self):
        raise NotImplementedError
#        rg_i = [ p for p in range(self.nocc) ]  
#        rg_j = rg_i  
#        rg_k = rg_i  
#        rg_a = [ p for p in range(self.nvir) ]  
#        rg_b = rg_a
#        rg_c = rg_a
#
#        for c, b, a, k, j, i in itertools.product(rg_c, rg_b, rg_a, rg_k, rg_j, rg_i): 
#            self.t3[i][j][k][a][b][c]  = self.t3aab[i][j][k][a][b][c].copy()
#            self.t3[i][j][k][a][b][c] += self.t3aab[i][k][j][a][c][b].copy()
#            self.t3p[i][j][k][a][b][c]  = self.t3aaa[i][j][k][a][b][c].copy()
#            self.t3p[i][j][k][a][b][c] += self.t3[i][j][k][a][b][c].copy()
#            self.t3p[i][j][k][a][b][c] += self.t3aab[j][k][i][b][c][a].copy()
#            self.t3p[i][j][k][a][b][c] /= 0.5 

    def read_t3aab(self, path='t3aab.csv'):
        raise NotImplementedError
#        data = pd.read_csv(path)
#        self.t3aab = numpy.array(list(data.loc[:,"t3aab"]))
#        self.t3aab = self.t3aab.reshape(self.nocc,self.nocc,self.nocc,self.nvir,self.nvir,self.nvir)

    def read_t1_t3c(self, path='t1_t3c.csv'):
        raise NotImplementedError
#        data = pd.read_csv(path)
#        t1_t3c = numpy.array(list(data.loc[:,"t1_t3c"]))
#        t1_t3c = t1_t3c.reshape(self.nocc,self.nvir)
#        return t1_t3c
 
    def read_t2_t4c(self, path='t2_t4c.csv'):
        raise NotImplementedError
#        data = pd.read_csv(path)
#        t2_t4c = numpy.array(list(data.loc[:,"t2_t4c"]))
#        t2_t4c = t2_t4c.reshape(self.nocc,self.nocc,self.nvir,self.nvir)
#        return t2_t4c
