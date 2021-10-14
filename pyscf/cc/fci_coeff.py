import pandas as pd
import numpy 
from pyscf.cc.fci_index import tn_addrs_signs
from pyscf.cc import _ccsd 
from pyscf import lib
import ctypes

class fci_coeff: 
    def __init__(self, fcivec, nocc, nvir, idx, nc = 0):
        self.rcas = True 
        self.nocc = nocc
        self.nvir = nvir
        self.nmo  = nocc + nvir 
        self.idx  = idx
        self.nc     = nc
        self.nacto_ref = None
        self.nacte_ref = None
        self.ncoreo_ref= None
        self.top_w_idx = [] 
        self.top_w_typ = "" 
        self.top_w     = 0.0 
        self.numzero = 0.0 
        self.skip_idx= None 

        self.nocc_corr= None
        self.nvir_corr= None
        self.nocc_cas = None
        self.nvir_cas = None
        self.nocc_iact= None
        self.fcivec = fcivec 

        self.Pmat = None
        self.Ref = None
        self.S_a = None
        self.S_b = None
        self.D_aa = None
        self.D_ab = None
        self.D_bb = None
        self.T_aaa = None
        self.T_aab = None
        self.T_abb = None
        self.T_bbb = None
        self.Q_aaaa = None
        self.Q_aaab = None
        self.Q_aabb = None
        self.Q_abbb = None
        self.Q_bbbb = None

        self.flagS = False
        self.flagD = False
        self.flagT = False
        self.flagQ = False

        self.interm_norm_S = False 
        self.interm_norm_D = False 
        self.interm_norm_T = False 
        self.interm_norm_Q = False 

    def get_S(self):
        self.flagS = True 

        self.t1addrs, self.t1signs = tn_addrs_signs(self.nocc_cas + self.nvir_cas, self.nocc_cas, 1)
        self.index1 = numpy.argsort(self.t1addrs)
        self.Ref  = [self.fcivec[0, 0]]
        S_a_us    = self.fcivec[self.t1addrs, 0] * self.t1signs
        S_b_us    = self.fcivec[0, self.t1addrs] * self.t1signs 
#        S_a_us    = self.fcivec[self.t1addrs, 0]
#        S_b_us    = self.fcivec[0, self.t1addrs]

        self.S_a    = S_a_us[self.index1]
        self.S_b    = S_b_us[self.index1]

    def get_D(self):
        self.flagD = True
        self.t2addrs, self.t2signs = tn_addrs_signs(self.nocc_cas + self.nvir_cas, self.nocc_cas, 2)
        self.index2 = numpy.argsort(self.t2addrs)
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

    def get_SD(self):
        self.flagS = True 
        self.flagD = True

        self.t1addrs, self.t1signs = tn_addrs_signs(self.nocc_cas + self.nvir_cas, self.nocc_cas, 1)
        self.index1 = numpy.argsort(self.t1addrs)

        self.Ref  = [self.fcivec[0, 0]]
        S_a_us    = self.fcivec[self.t1addrs, 0] * self.t1signs
        self.S_a    = S_a_us[self.index1]
        self.S_b  = self.S_a

        D_ab_us   = numpy.einsum('ij,i,j->ij', 
                             self.fcivec[self.t1addrs[:,None], self.t1addrs], 
                             self.t1signs, self.t1signs)
        D_ab_t      = D_ab_us[self.index1,:]
        self.D_ab   = D_ab_t[:,self.index1]

        if self.nocc_cas < 2 or self.nvir_cas < 2:
            return

        self.t2addrs, self.t2signs = tn_addrs_signs(self.nocc_cas + self.nvir_cas, self.nocc_cas, 2)
        self.index2 = numpy.argsort(self.t2addrs)

        D_aa_us   = self.fcivec[self.t2addrs, 0] * self.t2signs
        self.D_aa   = D_aa_us[self.index2]
        self.D_bb = self.D_aa

    def get_SD_test(self):
        self.flagS = True 
        self.flagD = True

        self.t1addrs, self.t1signs = tn_addrs_signs(self.nocc_cas + self.nvir_cas, self.nocc_cas, 1)
        self.index1 = numpy.argsort(self.t1addrs)

        self.Ref  = [self.fcivec[0, 0]]
        S_a_us    = self.fcivec[self.t1addrs, 0] * self.t1signs
        self.S_a    = S_a_us[self.index1]

        D_ab_us   = numpy.einsum('ij,i,j->ij', 
                             self.fcivec[self.t1addrs[:,None], self.t1addrs], 
                             self.t1signs, self.t1signs)
        D_ab_t      = D_ab_us[self.index1,:]
        self.D_ab   = D_ab_t[:,self.index1]

    def get_T(self):
        self.flagT = True

        if self.nocc_cas < 2 or self.nvir_cas < 2:
            return

        T_aab_us  = numpy.einsum('ij,i,j->ij', self.fcivec[self.t2addrs[:,None], self.t1addrs], self.t2signs, self.t1signs)
        T_abb_us  = numpy.einsum('ij,i,j->ij', self.fcivec[self.t1addrs[:,None], self.t2addrs], self.t1signs, self.t2signs)

        T_aab_t     = T_aab_us[self.index2,:]
        self.T_aab  = T_aab_t[:,self.index1]
        T_abb_t     = T_abb_us[self.index1,:]
        self.T_abb  = T_abb_t[:,self.index2]

        if self.nocc_cas < 3 or self.nvir_cas < 3:
            return

        self.t3addrs, self.t3signs = tn_addrs_signs(self.nocc_cas + self.nvir_cas, self.nocc_cas, 3)
        self.index3 = numpy.argsort(self.t3addrs)
        T_aaa_us  = self.fcivec[self.t3addrs, 0] * self.t3signs
        T_bbb_us  = self.fcivec[0, self.t3addrs] * self.t3signs
        self.T_aaa  = T_aaa_us[self.index3]
        self.T_bbb  = T_bbb_us[self.index3]


    def get_Q(self):

        if self.nocc_cas < 2 or self.nvir_cas < 2:
            return

        self.flagQ = True
        Q_aabb_us = numpy.einsum('ij,i,j->ij', self.fcivec[self.t2addrs[:,None], self.t2addrs], self.t2signs, self.t2signs)
        Q_aabb_t    = Q_aabb_us[self.index2,:]
        self.Q_aabb = Q_aabb_t[:,self.index2]

        if self.nocc_cas < 3 or self.nvir_cas < 3:
            return

        Q_aaab_us = numpy.einsum('ij,i,j->ij', self.fcivec[self.t3addrs[:,None], self.t1addrs], self.t3signs, self.t1signs)
        Q_abbb_us = numpy.einsum('ij,i,j->ij', self.fcivec[self.t1addrs[:,None], self.t3addrs], self.t1signs, self.t3signs)
        Q_aaab_t    = Q_aaab_us[self.index3,:]
        self.Q_aaab = Q_aaab_t[:,self.index1]
        Q_abbb_t    = Q_abbb_us[self.index1,:]
        self.Q_abbb = Q_abbb_t[:,self.index3]

        #self.t4addrs, self.t4signs = tn_addrs_signs(self.nocc_cas + self.nvir_cas, self.nocc_cas, 4)
        #self.index4 = numpy.argsort(self.t4addrs)
        #Q_aaaa_us = self.fcivec[self.t4addrs, 0] * self.t4signs

        #Q_bbbb_us = self.fcivec[0, self.t4addrs] * self.t4signs

        #self.Q_aaaa = Q_aaaa_us[self.index4]
        #self.Q_bbbb = Q_bbbb_us[self.index4]

    def interm_norm_test(self, T=True, Q=True):
        self.interm_norm_S = True 
        self.interm_norm_D = True 

        self.S_a    = self.S_a    / self.Ref[0]
        self.D_ab   = self.D_ab   / self.Ref[0]

        if self.nocc_cas < 2 or self.nvir_cas < 2:
            return

        self.D_aa   = self.D_aa   / self.Ref[0]
        if(T): 
           self.interm_norm_T = True 
           self.T_aab  = self.T_aab / self.Ref[0]
        if(Q): 
           self.interm_norm_Q = True 
           self.Q_aabb = self.Q_aabb/ self.Ref[0]

        if self.nocc_cas < 3 or self.nvir_cas < 3:
            return

        if(T): 
           self.interm_norm_T = True 
           self.T_aaa  = self.T_aaa / self.Ref[0]
        if(Q): 
           self.interm_norm_Q = True 
           self.Q_aaab = self.Q_aaab/ self.Ref[0]

    def interm_norm(self, T=True, Q=True):
        self.interm_norm_S = True 
        self.interm_norm_D = True 

        self.S_a    = self.S_a    / self.Ref[0]
        self.D_ab   = self.D_ab   / self.Ref[0]

        if self.nocc_cas < 2 or self.nvir_cas < 2:
            return

        self.D_aa   = self.D_aa   / self.Ref[0]
        if(T): 
           self.interm_norm_T = True 
           self.T_aaa  = self.T_aaa / self.Ref[0]
           self.T_aab  = self.T_aab / self.Ref[0]
        if(Q): 
           self.interm_norm_Q = True 
           self.Q_aaab = self.Q_aaab/ self.Ref[0]
           self.Q_aabb = self.Q_aabb/ self.Ref[0]

    def tcc_tcas_idx(self):
        t1f= numpy.zeros((self.nocc_corr,self.nvir_corr), dtype=numpy.float64)
        t2f= numpy.zeros((self.nocc_corr,self.nocc_corr,self.nvir_corr,self.nvir_corr), dtype=numpy.float64)
        n_a  = 0
        n_ab = 0

        if self.nocc_tcas is not None and self.nvir_tcas is not None:
            nocc_iact = self.nocc_corr - self.nocc_tcas 

            for i in range(self.nocc_tcas):
                ip = i+nocc_iact
                for a in range(self.nvir_tcas):
                    t1f[ip,a] = 1.0
    
            for i in range(self.nocc_tcas):
                ip = i+nocc_iact
                for j in range(self.nocc_tcas):
                    jp = j+nocc_iact
                    for a in range(self.nvir_tcas):
                        for b in range(self.nvir_tcas):
                            t2f[ip,jp,a,b] = 1.0
        else:
            for i in range(self.nocc_cas):
                ip = i+self.nocc_iact
                for a in range(self.nvir_cas):
                    t1f[ip,a] = 1.0
    
            for i in range(self.nocc_cas):
                ip = i+self.nocc_iact
                for j in range(self.nocc_cas):
                    jp = j+self.nocc_iact
                    for a in range(self.nvir_cas):
                        for b in range(self.nvir_cas):
                            t2f[ip,jp,a,b] = 1.0
        return t1f, t2f

class ufci_coeff: 
    def __init__(self, fcivec, idx, nc = 0):
        self.rcas = False 
        self.idx  = idx
        self.nc   = nc
        self.nacto_ref = None
        self.nacte_ref = None
        self.ncoreo_ref= None
        self.top_w_idx = [] 
        self.top_w_typ = "" 
        self.top_w     = 0.0 
        self.numzero = 0.0 
        self.skip_idx= None 

        self.nocc_corr= None
        self.nvir_corr= None
        self.nocc_cas = None
        self.nvir_cas = None
        self.nocc_iact= None
        self.fcivec = fcivec 

        self.Pmat = None
        self.Ref = None
        self.S_a = None
        self.S_b = None
        self.D_aa = None
        self.D_ab = None
        self.D_bb = None
        self.T_aaa = None
        self.T_aab = None
        self.T_abb = None
        self.T_bbb = None
        self.Q_aaaa = None
        self.Q_aaab = None
        self.Q_aabb = None
        self.Q_abbb = None
        self.Q_bbbb = None

        self.flagS = False
        self.flagD = False
        self.flagT = False
        self.flagQ = False

        self.interm_norm_S = False 
        self.interm_norm_D = False 
        self.interm_norm_T = False 
        self.interm_norm_Q = False 

    def get_S(self):
        self.flagS = True 

        self.t1addrs, self.t1signs = tn_addrs_signs(self.nocc_cas + self.nvir_cas, self.nocc_cas, 1)
        self.index1 = numpy.argsort(self.t1addrs)
        self.Ref  = [self.fcivec[0, 0]]
        S_a_us    = self.fcivec[self.t1addrs, 0] * self.t1signs
        S_b_us    = self.fcivec[0, self.t1addrs] * self.t1signs 

        self.S_a    = S_a_us[self.index1]
        self.S_b    = S_b_us[self.index1]

    def get_D(self):
        self.flagD = True
        self.t2addrs, self.t2signs = tn_addrs_signs(self.nocc_cas + self.nvir_cas, self.nocc_cas, 2)
        self.index2 = numpy.argsort(self.t2addrs)
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

    def get_SD(self):
        self.flagS = True 
        self.flagD = True

        t1addrs_a, t1signs_a = tn_addrs_signs(self.nocc_cas[0] + self.nvir_cas[0], self.nocc_cas[0], 1)
        t2addrs_a, t2signs_a = tn_addrs_signs(self.nocc_cas[0] + self.nvir_cas[0], self.nocc_cas[0], 2)
        t1addrs_b, t1signs_b = tn_addrs_signs(self.nocc_cas[1] + self.nvir_cas[1], self.nocc_cas[1], 1)
        t2addrs_b, t2signs_b = tn_addrs_signs(self.nocc_cas[1] + self.nvir_cas[1], self.nocc_cas[1], 2)
        index1_a = numpy.argsort(t1addrs_a)
        index2_a = numpy.argsort(t2addrs_a)
        index1_b = numpy.argsort(t1addrs_b)
        index2_b = numpy.argsort(t2addrs_b)

        print('S size, D size=',t1addrs_a.size, t2addrs_a.size)

        self.Ref  = [self.fcivec[0, 0]]
        S_a_us    = self.fcivec[t1addrs_a, 0] * t1signs_a
        S_b_us    = self.fcivec[0, t1addrs_b] * t1signs_b 
        self.S_a    = S_a_us[index1_a]
        self.S_b    = S_b_us[index1_b]

        D_aa_us   = self.fcivec[t2addrs_a, 0] * t2signs_a
        self.D_aa   = D_aa_us[index2_a]
        D_ab_us   = numpy.einsum('ij,i,j->ij', 
                             self.fcivec[t1addrs_a[:,None], t1addrs_b], 
                             t1signs_a, t1signs_b)
        D_ab_t      = D_ab_us[index1_a,:]
        self.D_ab   = D_ab_t[:,index1_b]
        D_bb_us   = self.fcivec[0, t2addrs_b] * t2signs_b
        self.D_bb   = D_bb_us[index2_b]

    def get_T(self):
        self.flagT = True
        self.t3addrs, self.t3signs = tn_addrs_signs(self.nocc_cas + self.nvir_cas, self.nocc_cas, 3)
        self.index3 = numpy.argsort(self.t3addrs)
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
        self.t4addrs, self.t4signs = tn_addrs_signs(self.nocc_cas + self.nvir_cas, self.nocc_cas, 4)
        self.index4 = numpy.argsort(self.t4addrs)
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

    def interm_norm(self, T=True, Q=True):
        self.interm_norm_S = True 
        self.interm_norm_D = True 

        self.S_a    = self.S_a    / self.Ref[0]
        self.D_aa   = self.D_aa   / self.Ref[0]
        self.D_ab   = self.D_ab   / self.Ref[0]
        if(T): 
           self.interm_norm_T = True 
           self.T_aaa  = self.T_aaa / self.Ref[0]
           self.T_aab  = self.T_aab / self.Ref[0]
        if(Q): 
           self.interm_norm_Q = True 
           self.Q_aaab = self.Q_aaab/ self.Ref[0]
           self.Q_aabb = self.Q_aabb/ self.Ref[0]

    def tcc_tcas_idx(self):
        nocca_corr = self.nocc_corr[0]
        nvira_corr = self.nvir_corr[0]
        nocca_cas  = self.nocc_cas [0]
        nvira_cas  = self.nvir_cas [0]
        nocca_iact = self.nocc_iact[0]
        noccb_corr = self.nocc_corr[1]
        nvirb_corr = self.nvir_corr[1]
        noccb_cas  = self.nocc_cas [1]
        nvirb_cas  = self.nvir_cas [1]
        noccb_iact = self.nocc_iact[1]

        t1af = numpy.zeros((nocca_corr,nvira_corr), dtype=numpy.float64)
        t1bf = numpy.zeros((noccb_corr,nvirb_corr), dtype=numpy.float64)
        t2aaf= numpy.zeros((nocca_corr,nocca_corr,nvira_corr,nvira_corr), dtype=numpy.float64)
        t2abf= numpy.zeros((nocca_corr,noccb_corr,nvira_corr,nvirb_corr), dtype=numpy.float64)
        t2bbf= numpy.zeros((noccb_corr,noccb_corr,nvirb_corr,nvirb_corr), dtype=numpy.float64)

        n_a  = 0
        n_ab = 0

        for i in range(nocca_cas):
            ip = i+nocca_iact
            for a in range(nvira_cas):
                t1af[ip][a] = 1.0
        for i in range(noccb_cas):
            ip = i+noccb_iact
            for a in range(nvirb_cas):
                t1bf[ip][a] = 1.0

        for i in range(nocca_cas):
            ip = i+nocca_iact
            for j in range(nocca_cas):
                jp = j+nocca_iact
                for a in range(nvira_cas):
                    for b in range(nvira_cas):
                        t2aaf[ip][jp][a][b] = 1.0

        for i in range(nocca_cas):
            ip = i+nocca_iact
            for j in range(noccb_cas):
                jp = j+noccb_iact
                for a in range(nvira_cas):
                    for b in range(nvirb_cas):
                        t2abf[ip][jp][a][b] = 1.0

        for i in range(noccb_cas):
            ip = i+noccb_iact
            for j in range(noccb_cas):
                jp = j+noccb_iact
                for a in range(nvirb_cas):
                    for b in range(nvirb_cas):
                        t2bbf[ip][jp][a][b] = 1.0

        t1f = (t1af, t1bf)
        t2f = (t2aaf, t2abf, t2bbf)
        return t1f, t2f
