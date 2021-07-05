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

