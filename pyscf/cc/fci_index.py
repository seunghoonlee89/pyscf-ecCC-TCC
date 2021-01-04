#!/usr/bin/env python
#
# Author: Seunghoon Lee <seunghoonlee89@gmail.com>
#

'''
Connecting index of FCI method and that of CC method
FCI index: See def tn_addrs_signs in cisd.py
'''

import numpy 
from pyscf.fci import cistring
from ctypes import *
import ctypes

#from pyscf.ci.cisd import tn_addrs_signs

def tn_addrs_signs(norb, nelec, n_excite):
    '''Compute the FCI strings (address) for CIS n-excitation amplitudes and
    the signs of the coefficients when transferring the reference from physics
    vacuum to HF vacuum.
    '''
    if n_excite > nelec:
        print("Warning: Not enough occupied orbitals to excite.")
        return [0], [0]
    nocc = nelec

    hole_strs = cistring.gen_strings4orblist(range(nocc), nocc - n_excite)
    # For HF vacuum, hole operators are ordered from low-lying to high-lying
    # orbitals. It leads to the opposite string ordering.
    hole_strs = hole_strs[::-1]
    hole_sum = numpy.zeros(len(hole_strs), dtype=int)
    for i in range(nocc):
        hole_at_i = (hole_strs & (1<<i)) == 0
        hole_sum[hole_at_i] += i

    # The hole operators are listed from low-lying to high-lying orbitals
    # (from left to right).  For i-th (0-based) hole operator, the number of
    # orbitals which are higher than i determines the sign.  This number
    # equals to nocc-(i+1).  After removing the highest hole operator, nocc
    # becomes nocc-1, the sign for next hole operator j will be associated to
    # nocc-1-(j+1).  By iteratively calling this procedure, the overall sign
    # for annihilating three holes is (-1)**(3*nocc - 6 - sum i)
    sign = (-1) ** (n_excite * nocc - n_excite*(n_excite+1)//2 - hole_sum)

    particle_strs = cistring.gen_strings4orblist(range(nocc, norb), n_excite)
    strs = hole_strs[:,None] ^ particle_strs
    addrs = cistring.strs2addr(norb, nocc, strs.ravel())
    signs = numpy.vstack([sign] * len(particle_strs)).T.ravel()
    return addrs, signs

def parity(values):
    values = list(values)
    N = len(values)
    num_swaps = 0
    for i in range(N-1):
        for j in range(i+1, N):
            if values[i] > values[j]:
                values[i], values[j] = values[j], values[i]
                num_swaps += 1
    p = (-1)**(num_swaps % 2)
    return p

def reorder(idx):
    p = parity(idx)
    idx.sort()
    idx.append(p)
    return idx

class fci_coeff: 
    def __init__(self, fcivec, nmo, nocc):
        self.nmo = nmo
        self.nocc = nocc
        self.nvir = nmo - nocc
        self.fcivec = fcivec 
        self.Ref = numpy.array([], dtype=numpy.float64) 
        self.S_a = numpy.array([], dtype=numpy.float64) 
        self.S_b = numpy.array([], dtype=numpy.float64) 
        self.D_aa = numpy.array([], dtype=numpy.float64) 
        self.D_ab = numpy.array([], dtype=numpy.float64) 
        self.D_bb = numpy.array([], dtype=numpy.float64) 
        self.T_aaa = numpy.array([], dtype=numpy.float64) 
        self.T_aab = numpy.array([], dtype=numpy.float64) 
        self.T_abb = numpy.array([], dtype=numpy.float64) 
        self.T_bbb = numpy.array([], dtype=numpy.float64) 
        self.Q_aaaa = numpy.array([], dtype=numpy.float64) 
        self.Q_aaab = numpy.array([], dtype=numpy.float64) 
        self.Q_aabb = numpy.array([], dtype=numpy.float64) 
        self.Q_abbb = numpy.array([], dtype=numpy.float64) 
        self.Q_bbbb = numpy.array([], dtype=numpy.float64) 
        self.t1addrs, self.t1signs = tn_addrs_signs(self.nmo, self.nocc, 1)
        self.t2addrs, self.t2signs = tn_addrs_signs(self.nmo, self.nocc, 2)
        self.t3addrs, self.t3signs = tn_addrs_signs(self.nmo, self.nocc, 3)
        self.t4addrs, self.t4signs = tn_addrs_signs(self.nmo, self.nocc, 4)
        self.index1 = numpy.argsort(self.t1addrs)
        self.index2 = numpy.argsort(self.t2addrs)
        self.index3 = numpy.argsort(self.t3addrs)
        self.index4 = numpy.argsort(self.t4addrs)

        self.flagS = False 
        self.flagD = False
        self.flagT = False
        self.flagQ = False

    def get_S(self):
        self.flagS = True 
        self.Ref  = [self.fcivec[0, 0]]
        S_a_us    = self.fcivec[self.t1addrs, 0] * self.t1signs
        S_b_us    = self.fcivec[0, self.t1addrs] * self.t1signs 
#        S_a_us    = self.fcivec[self.t1addrs, 0]
#        S_b_us    = self.fcivec[0, self.t1addrs]

        self.S_a    = S_a_us[self.index1]
        self.S_b    = S_b_us[self.index1]

    def get_D(self):
        self.flagD = True
        D_aa_us   = self.fcivec[self.t2addrs, 0] * self.t2signs
        D_ab_us   = numpy.einsum('ij,i,j->ij', 
                             self.fcivec[self.t1addrs[:,None], self.t1addrs], 
                             self.t1signs, self.t1signs)
        D_bb_us   = self.fcivec[0, self.t2addrs] * self.t2signs

#        D_aa_us   = self.fcivec[self.t2addrs, 0]
#        D_ab_us   = self.fcivec[self.t1addrs[:,None], self.t1addrs]
#        D_bb_us   = self.fcivec[0, self.t2addrs]

        self.D_aa   = D_aa_us[self.index2]
        D_ab_t      = D_ab_us[self.index1,:]
        self.D_ab   = D_ab_t[:,self.index1]
        self.D_bb   = D_bb_us[self.index2]

    def get_T(self):
        self.flagT = True
        T_aaa_us  = self.fcivec[self.t3addrs, 0] * self.t3signs
        T_aab_us  = numpy.einsum('ij,i,j->ij', self.fcivec[self.t2addrs[:,None], self.t1addrs], self.t2signs, self.t1signs)
        T_abb_us  = numpy.einsum('ij,i,j->ij', self.fcivec[self.t1addrs[:,None], self.t2addrs], self.t1signs, self.t2signs)
        T_bbb_us  = self.fcivec[0, self.t3addrs] * self.t3signs

        self.T_aaa  = T_aaa_us[self.index3]
        T_aab_t     = T_aab_us[self.index2,:]
        self.T_aab  = T_aab_t[:,self.index1]
        T_abb_t     = T_abb_us[self.index1,:]
        self.T_abb  = T_abb_t[:,self.index2]
        self.T_bbb  = T_bbb_us[self.index3]

    def get_Q(self):
        self.flagQ = True
        Q_aaaa_us = self.fcivec[self.t4addrs, 0] * self.t4signs
        Q_aaab_us = numpy.einsum('ij,i,j->ij', self.fcivec[self.t3addrs[:,None], self.t1addrs], self.t3signs, self.t1signs)
        Q_aabb_us = numpy.einsum('ij,i,j->ij', self.fcivec[self.t2addrs[:,None], self.t2addrs], self.t2signs, self.t2signs)
        Q_abbb_us = numpy.einsum('ij,i,j->ij', self.fcivec[self.t1addrs[:,None], self.t3addrs], self.t1signs, self.t3signs)
        Q_bbbb_us = self.fcivec[0, self.t4addrs] * self.t4signs

        self.Q_aaaa = Q_aaaa_us[self.index4]
        Q_aaab_t    = Q_aaab_us[self.index3,:]
        self.Q_aaab = Q_aaab_t[:,self.index1]
        Q_aabb_t    = Q_aabb_us[self.index2,:]
        self.Q_aabb = Q_aabb_t[:,self.index2]
        Q_abbb_t    = Q_abbb_us[self.index1,:]
        self.Q_abbb = Q_abbb_t[:,self.index3]
        self.Q_bbbb = Q_bbbb_us[self.index4]

    def interm_norm(self, Q=True):
        self.S_a    = self.S_a    / self.Ref[0]
        self.S_b    = self.S_b    / self.Ref[0]
        self.D_aa   = self.D_aa   / self.Ref[0]
        self.D_ab   = self.D_ab   / self.Ref[0]
        self.D_bb   = self.D_bb   / self.Ref[0]
        self.T_aaa  = self.T_aaa  / self.Ref[0]
        self.T_aab  = self.T_aab  / self.Ref[0]
        self.T_abb  = self.T_abb  / self.Ref[0]
        self.T_bbb  = self.T_bbb  / self.Ref[0]
        if(Q): 
#           self.Q_aaaa = self.Q_aaaa/ self.Ref[0]
           self.Q_aaab = self.Q_aaab/ self.Ref[0]
           self.Q_aabb = self.Q_aabb/ self.Ref[0]
#           self.Q_abbb = self.Q_abbb/ self.Ref[0]
#           self.Q_bbbb = self.Q_bbbb/ self.Ref[0]
        self.Ref[0]    = 1.0 

class fci_index:
    def __init__(self, nocc, nvir):
        self.nocc = nocc
        self.nvir = nvir
        self.idx1 = {} 
        self.idx2 = {}
        self.idx3 = {}
        self.idx4 = {}

    @classmethod
    def ia_str(self, i, a):
        idx = str(i) + '->' + str(a)
        return idx 

    @classmethod
    def ijab_str(self, i, j, a, b):
        assert i < j and a < b
        idx = str(i) + ',' + str(j) + '->' + str(a) + ',' + str(b)
        return idx 
 
    @classmethod
    def ijkabc_str(self, i, j, k, a, b, c):
        if not (i < j and j < k and a < b and b < c): print ('i,j,k,a,b,c',i,j,k,a,b,c)
        assert i < j and j < k and a < b and b < c
        idx = str(i) + ',' + str(j) + ',' + str(k) + '->' + str(a) + ',' + str(b) + ',' + str(c)
        return idx 

    @classmethod
    def ijklabcd_str(self, i, j, k, l, a, b, c, d):
        assert i < j and j < k and k < l and a < b and b < c and c < d
        idx = str(i) + ',' + str(j) + ',' + str(k) + ',' + str(l) + '->' + str(a) + ',' + str(b) + ',' + str(c) + ',' + str(d)
        return idx 

    def S(self, i, a):
        return self.idx1[self.ia_str(i,a)] 

    def D(self, i, j, a, b):
        return self.idx2[self.ijab_str(i,j,a,b)] 

    def T(self, i, j, k, a, b, c):
        return self.idx3[self.ijkabc_str(i,j,k,a,b,c)] 

    def Q(self, i, j, k, l, a, b, c, d):
        return self.idx4[self.ijklabcd_str(i,j,k,l,a,b,c,d)] 

    def get_S(self):
        ia = -1 
        for a in range(0,self.nvir,1):
            for i in range(self.nocc-1,-1,-1):
                ia += 1
                self.idx1[self.ia_str(i, a)] = ia
#                print 'ia, i, a =', ia, i, a

    def get_D(self):
        ijab = -1 
        for b in range(1,self.nvir,1):
            for a in range(0,b,1):
                for j in range(self.nocc-1,0,-1):
                    for i in range(j-1,-1,-1):
                        ijab += 1
                        self.idx2[self.ijab_str(i, j, a, b)] = ijab
#                        print 'ijab, i, j, a, b =', ijab, i, j, a, b

    def get_T(self):
        ijkabc = -1 
        for c in range(2,self.nvir,1):
            for b in range(1,c,1):
                for a in range(0,b,1):
                    for k in range(self.nocc-1,1,-1):
                        for j in range(k-1,0,-1):
                            for i in range(j-1,-1,-1):
                                ijkabc += 1
                                self.idx3[self.ijkabc_str(i, j, k, a, b, c)] = ijkabc
#                                print 'ijkabc, i, j, k, a, b, c =', ijkabc, i, j, k, a, b, c

    def get_Q(self):
        ijklabcd = -1 
        for d in range(3,self.nvir,1):
            for c in range(2,d,1):
                for b in range(1,c,1):
                    for a in range(0,b,1):
                        for l in range(self.nocc-1,2,-1):
                            for k in range(l-1,1,-1):
                                for j in range(k-1,0,-1):
                                    for i in range(j-1,-1,-1):
                                        ijklabcd += 1
                                        self.idx4[self.ijklabcd_str(i, j, k, l, a, b, c, d)] = ijklabcd
#                                        print 'ijkabc, i, j, k, l, a, b, c, d =', ijklabcd, i, j, k, l, a, b, c, d

class fci_index_c:
    def __init__(self, nocc, nvir):
        self.nocc = nocc
        self.nvir = nvir
        self.nmo  = nocc+nvir

        #TODO change this part as a simple formula
        self.t1addrs, self.t1signs = tn_addrs_signs(self.nmo, self.nocc, 1)
        self.t2addrs, self.t2signs = tn_addrs_signs(self.nmo, self.nocc, 2)
        self.t3addrs, self.t3signs = tn_addrs_signs(self.nmo, self.nocc, 3)
        self.t4addrs, self.t4signs = tn_addrs_signs(self.nmo, self.nocc, 4)

        self.S = numpy.zeros((2,len(self.t1addrs)), dtype=numpy.int64)
        self.D = numpy.zeros((4,len(self.t2addrs)), dtype=numpy.int64)
        self.T = numpy.zeros((6,len(self.t3addrs)), dtype=numpy.int64)
        self.Q = numpy.zeros((8,len(self.t4addrs)), dtype=numpy.int64)

    def get_S(self):
        ia = -1 
        for a in range(0,self.nvir,1):
            for i in range(self.nocc-1,-1,-1):
                ia += 1
                self.S[0][ia] = i
                self.S[1][ia] = a

    def get_D(self):
        ijab = -1 
        for b in range(1,self.nvir,1):
            for a in range(0,b,1):
                for j in range(self.nocc-1,0,-1):
                    for i in range(j-1,-1,-1):
                        ijab += 1
                        self.D[0][ijab] = (i)
                        self.D[1][ijab] = (j)
                        self.D[2][ijab] = (a)
                        self.D[3][ijab] = (b)

    def get_T(self):
        ijkabc = -1 
        for c in range(2,self.nvir,1):
            for b in range(1,c,1):
                for a in range(0,b,1):
                    for k in range(self.nocc-1,1,-1):
                        for j in range(k-1,0,-1):
                            for i in range(j-1,-1,-1):
                                ijkabc += 1
                                self.T[0][ijkabc] = (i)
                                self.T[1][ijkabc] = (j)
                                self.T[2][ijkabc] = (k)
                                self.T[3][ijkabc] = (a)
                                self.T[4][ijkabc] = (b)
                                self.T[5][ijkabc] = (c)

    def get_Q(self):
        ijklabcd = -1 
        for d in range(3,self.nvir,1):
            for c in range(2,d,1):
                for b in range(1,c,1):
                    for a in range(0,b,1):
                        for l in range(self.nocc-1,2,-1):
                            for k in range(l-1,1,-1):
                                for j in range(k-1,0,-1):
                                    for i in range(j-1,-1,-1):
                                        ijklabcd += 1
                                        self.Q[0][ijklabcd] = (i)
                                        self.Q[1][ijklabcd] = (j)
                                        self.Q[2][ijklabcd] = (k)
                                        self.Q[3][ijklabcd] = (l)
                                        self.Q[4][ijklabcd] = (a)
                                        self.Q[5][ijklabcd] = (b)
                                        self.Q[6][ijklabcd] = (c)
                                        self.Q[7][ijklabcd] = (d)

class fci_index_nomem:
    def __init__(self, nocc, nvir):
        self.nocc = nocc
        self.nocc2 = nocc*(nocc-1)/2
        self.nocc3 = nocc*(nocc-1)*(nocc-2)/6 

    def S(self, i, a):
        return int(self.nocc * ( a + 1 ) - ( i + 1 )) 

    def D(self, i, j, a, b):
        return int(self.nocc2 * ( b*(b-1)/2 + a + 1 ) - ( j*(j-1)/2 + i + 1 )) 
               
    def T(self, i, j, k, a, b, c):
        return int(self.nocc3 * ( c*(c-1)*(c-2)/6 + b*(b-1)/2 + a + 1 ) \
                   - ( k*(k-1)*(k-2)/6 + j*(j-1)/2 + i + 1 ))

#    def Q(self, i, j, k, l, a, b, c, d):
#        return self.idx4[self.ijklabcd_str(i,j,k,l,a,b,c,d)] 

