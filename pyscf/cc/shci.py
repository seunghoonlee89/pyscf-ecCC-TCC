import pandas as pd
import numpy 
from pyscf.cc.fci_index import tn_addrs_signs
from pyscf.cc import _ccsd 
from pyscf import lib
import ctypes

class shci_coeff: 
    def __init__(self, shci_out, nocc, nvir, idx, nc = 0):
        self.rcas = True 
        self.nocc = nocc
        self.nvir = nvir
        self.nmo  = nocc + nvir 
        self.idx  = idx
        self.shci_out = shci_out 
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

        self.data=pd.read_csv(shci_out,sep=",")
        #self.data.info()

        self.typ_det = list(self.data.loc[:,"typ"])
        self.num_det = len(self.typ_det)

#        self.num_a    = self.typ_det.count("a")
#        self.num_aa   = self.typ_det.count("aa")
#        self.num_ab   = self.typ_det.count("ab")
#        self.num_aaa  = self.typ_det.count("aaa")
#        self.num_aab  = self.typ_det.count("aab")
#        self.num_aaab = self.typ_det.count("aaab")
#        self.num_aabb = self.typ_det.count("aabb")

        #print('total number of determinants', self.num_det)
#lsh test
#        print(self.data.head())
#        for i in range(self.num_det):
#            print(i,typ_det[i],self.data.loc[i,"1"])
        self.HFdet_str = [ 'ab' if i < self.nocc else 'v' for i in range(self.nmo) ] 

        self.flagS = False
        self.flagD = False
        self.flagT = False
        self.flagQ = False

        self.interm_norm_S = False 
        self.interm_norm_D = False 
        self.interm_norm_T = False 
        self.interm_norm_Q = False 

        self.Pmat = None

#        self.t1addrs = None
#        self.t2addrs = None
#        self.t3addrs = None
#        self.t4addrs = None
#        self.t1signs = None
#        self.t2signs = None
#        self.t3signs = None
#        self.t4signs = None
        self.E_shci = None
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

#        self.t1addrs, self.t1signs = tn_addrs_signs(self.nmo, self.nocc, 1)
#        self.t2addrs, self.t2signs = tn_addrs_signs(self.nmo, self.nocc, 2)
#        self.t3addrs, self.t3signs = tn_addrs_signs(self.nmo, self.nocc, 3)
#        self.t4addrs, self.t4signs = tn_addrs_signs(self.nmo, self.nocc, 4)
#        self.index1 = numpy.argsort(self.t1addrs)
#        self.index2 = numpy.argsort(self.t2addrs)
#        self.index3 = numpy.argsort(self.t3addrs)
#        self.index4 = numpy.argsort(self.t4addrs)

    def alpha_count(self, det_str, qi):
        ncount = 0 
        for qj in range(qi):
            if(det_str[qj]=="ab" or det_str[qj]=="b"): ncount += 1
        return ncount

    def parity_ab_str(self, det_str):
        """
        | noc_alpha noc_beta ... 1_alpha 1_beta > 
         = parity_ab_str * | noc_alpha ...  1_alpha > | noc_beta ...  1_beta >
        """
        n=0
        for qi in range(0, len(det_str)):
            if (det_str[qi]=="ab" or det_str[qi]=="a"):
                n += self.alpha_count(det_str, qi) 
#        return (-1)**n
        return 1 

    def parity_ci_to_cc(self, sum_ijkl, n_excite):
        """
        For example of singly-excited configuration, parity is
        | a noc ... i+1 i-1 ... 2 1 > = parity_ci_to_cc * | noc ... i+1 a i-1 ... 2 1 >
        """
        return (-1) ** ( n_excite*self.nocc - sum_ijkl - n_excite*(n_excite+1)/2 )

    def get_All(self, nc):
        self.nc = nc
        self.flagS = True 
        self.flagD = True
        self.flagT = True
        self.flagQ = True

        dS = int(self.nocc * self.nvir) 
        dD = int(self.nocc*(self.nocc-1)*self.nvir*(self.nvir-1)/4) 
        dT = int(self.nocc*(self.nocc-1)*(self.nocc-2)*self.nvir*(self.nvir-1)*(self.nvir-2)/36) 
#        dQ = len(self.idx.idx4.keys())
        #print('dS, dD, dT =',dS,dD,dT)

        nex = self.nocc * self.nvir
        self.Ref    = numpy.zeros((1), dtype=numpy.float64)

        self.S_a    = numpy.zeros((dS), dtype=numpy.float64)
        self.S_b    = numpy.zeros((dS), dtype=numpy.float64)
        self.D_aa   = numpy.zeros((dD), dtype=numpy.float64)
        self.D_ab   = numpy.zeros((dS,dS), dtype=numpy.float64)
        self.D_bb   = numpy.zeros((dD), dtype=numpy.float64)
        self.T_aaa  = numpy.zeros((dT), dtype=numpy.float64)
        self.T_aab  = numpy.zeros((dD,dS), dtype=numpy.float64)
        self.T_abb  = numpy.zeros((dS,dD), dtype=numpy.float64)
        self.T_bbb  = numpy.zeros((dT), dtype=numpy.float64)
#        self.Q_aaaa = numpy.zeros((dQ), dtype=numpy.float64)
        self.Q_aaab = numpy.zeros((dT,dS), dtype=numpy.float64)
        self.Q_aabb = numpy.zeros((dD,dD), dtype=numpy.float64)
#        self.Q_abbb = numpy.zeros((dS,dT), dtype=numpy.float64)
#        self.Q_bbbb = numpy.zeros((dQ), dtype=numpy.float64)

        test_max = 0.0
        test_max_idx =[] 
        for idet in range(self.num_det):
            typ = self.typ_det[idet]
            det_str = self.HFdet_str.copy()
            parity  = 1
            if (typ == "shci"):
               self.E_shci = self.data.loc[idet,"1"] 
            elif (typ == "rf"):
               self.Ref[0] = self.data.loc[idet,"1"] 
               if abs(self.data.loc[idet,"1"]) > self.top_w:
                   self.top_w     = self.data.loc[idet,"1"]
                   self.top_w_idx = [0] 
                   self.top_w_typ = typ 
            elif (typ == "a"):
               i = int(self.data.loc[idet,"1"]) + nc
               a = int(self.data.loc[idet,"2"]) - (self.nocc) + nc
               ia = self.idx.S(i, a) 
               parity = self.parity_ci_to_cc(i, 1)
               det_str[i] = 'b'
               det_str[a + self.nocc] = 'a'
               parity *= self.parity_ab_str(det_str)
               self.S_a[ia] = parity * self.data.loc[idet,"3"] 
               if abs(self.data.loc[idet,"3"]) > self.top_w:
                   self.top_w     = self.data.loc[idet,"3"]
                   self.top_w_idx = [i, a] 
                   self.top_w_typ = typ 
            elif (typ == "b"):
               i = int(self.data.loc[idet,"1"]) + nc
               a = int(self.data.loc[idet,"2"]) - (self.nocc) + nc
               ia = self.idx.S(i, a) 
               parity = self.parity_ci_to_cc(i, 1)
               det_str[i] = 'a'
               det_str[a + self.nocc] = 'b'
               parity *= self.parity_ab_str(det_str)
               self.S_b[ia] = parity * self.data.loc[idet,"3"] 
            elif (typ == "aa"):
               i = int(self.data.loc[idet,"1"]) + nc
               j = int(self.data.loc[idet,"2"]) + nc
               a = int(self.data.loc[idet,"3"]) - (self.nocc) + nc
               b = int(self.data.loc[idet,"4"]) - (self.nocc) + nc
               ijab = self.idx.D(i, j, a, b) 
               parity = self.parity_ci_to_cc(i+j, 2)
               det_str[i] = 'b'
               det_str[j] = 'b'
               det_str[a + self.nocc] = 'a'
               det_str[b + self.nocc] = 'a'
               parity *= self.parity_ab_str(det_str)
               self.D_aa[ijab] = parity * self.data.loc[idet,"5"] 
               if abs(self.data.loc[idet,"5"]) > self.top_w:
                   self.top_w     = self.data.loc[idet,"5"]
                   self.top_w_idx = [i, j, a, b] 
                   self.top_w_typ = typ 
            elif (typ == "ab"):
               i = int(self.data.loc[idet,"1"]) + nc
               a = int(self.data.loc[idet,"2"]) - (self.nocc) + nc
               j = int(self.data.loc[idet,"3"]) + nc
               b = int(self.data.loc[idet,"4"]) - (self.nocc) + nc
#               print (idet,'i,a,j,b',i,a,j,b)
               ia = self.idx.S(i, a) 
               jb = self.idx.S(j, b) 
               parity  = self.parity_ci_to_cc(i, 1)
               parity *= self.parity_ci_to_cc(j, 1)
               det_str[i] = 'b'
               det_str[a + self.nocc] = 'a'
               det_str[j] = 'a' if i != j else 'v'
               det_str[b + self.nocc] = 'b' if a != b else 'ab'
               parity *= self.parity_ab_str(det_str)
               self.D_ab[ia][jb] = parity * self.data.loc[idet,"5"] 
               if abs(self.data.loc[idet,"5"]) > self.top_w:
                   self.top_w     = self.data.loc[idet,"5"]
                   self.top_w_idx = [i, j, a, b] 
                   self.top_w_typ = typ 
            elif (typ == "bb"):
               i = int(self.data.loc[idet,"1"]) + nc
               j = int(self.data.loc[idet,"2"]) + nc
               a = int(self.data.loc[idet,"3"]) - (self.nocc) + nc
               b = int(self.data.loc[idet,"4"]) - (self.nocc) + nc
               ijab = self.idx.D(i, j, a, b) 
               parity = self.parity_ci_to_cc(i+j, 2)
               det_str[i] = 'a'
               det_str[j] = 'a'
               det_str[a + self.nocc] = 'b'
               det_str[b + self.nocc] = 'b'
               parity *= self.parity_ab_str(det_str)
               self.D_bb[ijab] = parity * self.data.loc[idet,"5"] 
            elif (typ == "aaa"):
               i = int(self.data.loc[idet,"1"]) + nc
               j = int(self.data.loc[idet,"2"]) + nc
               k = int(self.data.loc[idet,"3"]) + nc
               a = int(self.data.loc[idet,"4"]) - (self.nocc) + nc
               b = int(self.data.loc[idet,"5"]) - (self.nocc) + nc
               c = int(self.data.loc[idet,"6"]) - (self.nocc) + nc
               ijkabc = self.idx.T(i, j, k, a, b, c) 
               parity = self.parity_ci_to_cc(i+j+k, 3)
               det_str[i] = 'b'
               det_str[j] = 'b'
               det_str[k] = 'b'
               det_str[a + self.nocc] = 'a'
               det_str[b + self.nocc] = 'a'
               det_str[c + self.nocc] = 'a'
               parity *= self.parity_ab_str(det_str)
               self.T_aaa[ijkabc] = parity * self.data.loc[idet,"7"] 
               if abs(self.data.loc[idet,"7"]) > self.top_w:
                   self.top_w     = self.data.loc[idet,"7"]
                   self.top_w_idx = [i, j, k, a, b, c] 
                   self.top_w_typ = typ 
            elif (typ == "aab"):
               i = int(self.data.loc[idet,"1"]) + nc
               j = int(self.data.loc[idet,"2"]) + nc
               a = int(self.data.loc[idet,"3"]) - (self.nocc) + nc
               b = int(self.data.loc[idet,"4"]) - (self.nocc) + nc
               k = int(self.data.loc[idet,"5"]) + nc
               c = int(self.data.loc[idet,"6"]) - (self.nocc) + nc
               ijab = self.idx.D(i, j, a, b) 
               kc   = self.idx.S(k, c) 
               parity  = self.parity_ci_to_cc(i+j, 2)
               parity *= self.parity_ci_to_cc(k, 1)
               det_str[i] = 'b'
               det_str[j] = 'b'
               det_str[a + self.nocc] = 'a'
               det_str[b + self.nocc] = 'a'
               det_str[k] = 'a' if i != k and j != k else 'v'
               det_str[c + self.nocc] = 'b' if a != c and b != c else 'ab'
               parity *= self.parity_ab_str(det_str)
               self.T_aab[ijab][kc] = parity * self.data.loc[idet,"7"] 
               if abs(self.data.loc[idet,"7"]) > self.top_w:
                   self.top_w     = self.data.loc[idet,"7"]
                   self.top_w_idx = [i, j, k, a, b, c] 
                   self.top_w_typ = typ 

            elif (typ == "abb"):
               i = int(self.data.loc[idet,"1"]) + nc
               a = int(self.data.loc[idet,"2"]) - (self.nocc) + nc
               j = int(self.data.loc[idet,"3"]) + nc
               k = int(self.data.loc[idet,"4"]) + nc
               b = int(self.data.loc[idet,"5"]) - (self.nocc) + nc
               c = int(self.data.loc[idet,"6"]) - (self.nocc) + nc
               ia   = self.idx.S(i, a) 
               jkab = self.idx.D(j, k, b, c) 
               parity = self.parity_ci_to_cc(i, 1)
               parity *= self.parity_ci_to_cc(j+k, 2)
               det_str[i] = 'b'
               det_str[a + self.nocc] = 'a'
               det_str[j] = 'a' if i != j else 'v'
               det_str[k] = 'a' if i != k else 'v'
               det_str[b + self.nocc] = 'b' if a != b else 'ab'
               det_str[c + self.nocc] = 'b' if a != c else 'ab'
               parity *= self.parity_ab_str(det_str)
               self.T_abb[ia][jkab] = parity * self.data.loc[idet,"7"] 
            elif (typ == "bbb"):
               i = int(self.data.loc[idet,"1"]) + nc
               j = int(self.data.loc[idet,"2"]) + nc
               k = int(self.data.loc[idet,"3"]) + nc
               a = int(self.data.loc[idet,"4"]) - (self.nocc) + nc
               b = int(self.data.loc[idet,"5"]) - (self.nocc) + nc
               c = int(self.data.loc[idet,"6"]) - (self.nocc) + nc
               ijkabc = self.idx.T(i, j, k, a, b, c) 
               parity = self.parity_ci_to_cc(i+j+k, 3)
               det_str[i] = 'a'
               det_str[j] = 'a'
               det_str[k] = 'a'
               det_str[a + self.nocc] = 'b'
               det_str[b + self.nocc] = 'b'
               det_str[c + self.nocc] = 'b'
               parity *= self.parity_ab_str(det_str)
               self.T_bbb[ijkabc] = parity * self.data.loc[idet,"7"] 
#            elif (typ == "aaaa"):
#               i = int(self.data.loc[idet,"1"]) + nc
#               j = int(self.data.loc[idet,"2"]) + nc
#               k = int(self.data.loc[idet,"3"]) + nc
#               l = int(self.data.loc[idet,"4"]) + nc
#               a = int(self.data.loc[idet,"5"]) - (self.nocc) + nc
#               b = int(self.data.loc[idet,"6"]) - (self.nocc) + nc
#               c = int(self.data.loc[idet,"7"]) - (self.nocc) + nc
#               d = int(self.data.loc[idet,"8"]) - (self.nocc) + nc
#               ijklabcd = self.idx.Q(i, j, k, l, a, b, c, d) 
#               parity = self.parity_ci_to_cc(i+j+k+l, 4)
#               det_str[i] = 'b'
#               det_str[j] = 'b'
#               det_str[k] = 'b'
#               det_str[l] = 'b'
#               det_str[a + self.nocc] = 'a'
#               det_str[b + self.nocc] = 'a'
#               det_str[c + self.nocc] = 'a'
#               det_str[d + self.nocc] = 'a'
#               parity *= self.parity_ab_str(det_str)
#               self.Q_aaaa[ijklabcd] = parity * self.data.loc[idet,"9"] 
            elif (typ == "aaab"):
               i = int(self.data.loc[idet,"1"]) + nc
               j = int(self.data.loc[idet,"2"]) + nc
               k = int(self.data.loc[idet,"3"]) + nc
               a = int(self.data.loc[idet,"4"]) - (self.nocc) + nc
               b = int(self.data.loc[idet,"5"]) - (self.nocc) + nc
               c = int(self.data.loc[idet,"6"]) - (self.nocc) + nc
               l = int(self.data.loc[idet,"7"]) + nc
               d = int(self.data.loc[idet,"8"]) - (self.nocc) + nc
               ijkabc = self.idx.T(i, j, k, a, b, c) 
               ld     = self.idx.S(l, d) 
               parity = self.parity_ci_to_cc(i+j+k, 3)
               parity *= self.parity_ci_to_cc(l, 1)
               det_str[i] = 'b'
               det_str[j] = 'b'
               det_str[k] = 'b'
               det_str[a + self.nocc] = 'a'
               det_str[b + self.nocc] = 'a'
               det_str[c + self.nocc] = 'a'
               det_str[l] = 'a' if i != l and j != l and k != l else 'v'
               det_str[d + self.nocc] = 'b' if a != d and b != d and c != d else 'ab'
               parity *= self.parity_ab_str(det_str)
               self.Q_aaab[ijkabc][ld] = parity * self.data.loc[idet,"9"] 
               if abs(self.data.loc[idet,"9"]) > self.top_w:
                   self.top_w     = self.data.loc[idet,"9"]
                   self.top_w_idx = [i, j, k, l, a, b, c, d] 
                   self.top_w_typ = typ 
            elif (typ == "aabb"):
               i = int(self.data.loc[idet,"1"]) + nc
               j = int(self.data.loc[idet,"2"]) + nc
               a = int(self.data.loc[idet,"3"]) - (self.nocc) + nc
               b = int(self.data.loc[idet,"4"]) - (self.nocc) + nc
               k = int(self.data.loc[idet,"5"]) + nc
               l = int(self.data.loc[idet,"6"]) + nc
               c = int(self.data.loc[idet,"7"]) - (self.nocc) + nc
               d = int(self.data.loc[idet,"8"]) - (self.nocc) + nc
               #lsh test
#               if i == 2 and j == 3 and a == 0 and b == 1 \
#              and k == 2 and l == 3 and c == 0 and d == 1:
#               if (i == 2 and j == 3 and a == 0 and b == 4 \
#              and  k == 1 and l == 3 and c == 0 and d == 4) or \
#                  (i == 1 and j == 3 and a == 0 and b == 4 \
#              and  k == 2 and l == 3 and c == 0 and d == 4):
               if True:
                   ijab = self.idx.D(i, j, a, b) 
                   klcd = self.idx.D(k, l, c, d) 
                   parity  = self.parity_ci_to_cc(i+j, 2)
                   parity *= self.parity_ci_to_cc(k+l, 2)
                   det_str[i] = 'b'
                   det_str[j] = 'b'
                   det_str[a + self.nocc] = 'a'
                   det_str[b + self.nocc] = 'a'
                   det_str[k] = 'a' if i != k and j != k else 'v'
                   det_str[l] = 'a' if i != l and j != l else 'v'
                   det_str[c + self.nocc] = 'b' if a != c and b != c else 'ab'
                   det_str[d + self.nocc] = 'b' if a != d and b != d else 'ab'
                   parity *= self.parity_ab_str(det_str)
                   self.Q_aabb[ijab][klcd] = parity * self.data.loc[idet,"9"] 
                   #lsh test
#                   print('c4 org',parity * self.data.loc[idet,"9"]/self.Ref[0])

               if abs(self.data.loc[idet,"9"]) > self.top_w:
                   self.top_w     = self.data.loc[idet,"9"]
                   self.top_w_idx = [i, j, k, l, a, b, c, d] 
                   self.top_w_typ = typ 

#            elif (typ == "abbb"):
#               i = int(self.data.loc[idet,"1"]) + nc
#               a = int(self.data.loc[idet,"2"]) - (self.nocc) + nc
#               j = int(self.data.loc[idet,"3"]) + nc
#               k = int(self.data.loc[idet,"4"]) + nc
#               l = int(self.data.loc[idet,"5"]) + nc
#               b = int(self.data.loc[idet,"6"]) - (self.nocc) + nc
#               c = int(self.data.loc[idet,"7"]) - (self.nocc) + nc
#               d = int(self.data.loc[idet,"8"]) - (self.nocc) + nc
#               ia     = self.idx.S(i, a) 
#               jklbcd = self.idx.T(j, k, l, b, c, d) 
#               parity = self.parity_ci_to_cc(i, 1)
#               parity *= self.parity_ci_to_cc(j+k+l, 3)
#               det_str[i] = 'b'
#               det_str[a + self.nocc] = 'a'
#               det_str[j] = 'a' if i != j else 'v'
#               det_str[k] = 'a' if i != k else 'v'
#               det_str[l] = 'a' if i != l else 'v'
#               det_str[b + self.nocc] = 'b' if a != b else 'ab'
#               det_str[c + self.nocc] = 'b' if a != c else 'ab'
#               det_str[d + self.nocc] = 'b' if a != d else 'ab'
#               parity *= self.parity_ab_str(det_str)
#               self.Q_abbb[ia][jklbcd] = parity * self.data.loc[idet,"9"]
#            elif (typ == "bbbb"):
#               i = int(self.data.loc[idet,"1"]) + nc
#               j = int(self.data.loc[idet,"2"]) + nc
#               k = int(self.data.loc[idet,"3"]) + nc
#               l = int(self.data.loc[idet,"4"]) + nc
#               a = int(self.data.loc[idet,"5"]) - (self.nocc) + nc
#               b = int(self.data.loc[idet,"6"]) - (self.nocc) + nc
#               c = int(self.data.loc[idet,"7"]) - (self.nocc) + nc
#               d = int(self.data.loc[idet,"8"]) - (self.nocc) + nc
#               ijklabcd = self.idx.Q(i, j, k, l, a, b, c, d) 
#               parity = self.parity_ci_to_cc(i+j+k+l, 4)
#               det_str[i] = 'a'
#               det_str[j] = 'a'
#               det_str[k] = 'a'
#               det_str[l] = 'a'
#               det_str[a + self.nocc] = 'b'
#               det_str[b + self.nocc] = 'b'
#               det_str[c + self.nocc] = 'b'
#               det_str[d + self.nocc] = 'b'
#               parity *= self.parity_ab_str(det_str)
#               self.Q_bbbb[ijklabcd] = parity * self.data.loc[idet,"9"] 
#        print ('test max:',test_max, test_max_idx) 

    get_S=get_All
    get_D=get_All
    get_T=get_All
    get_Q=get_All

    def get_SD(self):
        self.flagS = True 
        self.flagD = True

        dS = int(self.nocc_cas*self.nvir_cas)
        dD = int(self.nocc_cas*(self.nocc_cas-1)*self.nvir_cas*(self.nvir_cas-1)/4) 
        #print('dS, dD =',dS,dD)

        self.Ref    = numpy.zeros((1), dtype=numpy.float64)

        self.S_a    = numpy.zeros((dS), dtype=numpy.float64)
        self.D_aa   = numpy.zeros((dD), dtype=numpy.float64)
        self.D_ab   = numpy.zeros((dS,dS), dtype=numpy.float64)

        for idet in range(self.num_det):
            typ = self.typ_det[idet]
            det_str = self.HFdet_str.copy()
            parity  = 1
            if (typ == "shci"):
               self.E_shci = self.data.loc[idet,"1"] 
            elif (typ == "rf"):
               self.Ref[0] = self.data.loc[idet,"1"] 
#               if abs(self.data.loc[idet,"1"]) > self.top_w:
#                   self.top_w     = self.data.loc[idet,"1"]
#                   self.top_w_idx = [0] 
#                   self.top_w_typ = typ 
            elif (typ == "a"):
               i = int(self.data.loc[idet,"1"]) 
               a = int(self.data.loc[idet,"2"]) - self.nocc_cas 
               ia = self.idx.S(i, a) 
               parity = self.parity_ci_to_cc(i, 1)
               self.S_a[ia] = parity * self.data.loc[idet,"3"] 
#               if abs(self.data.loc[idet,"3"]) > self.top_w:
#                   self.top_w     = self.data.loc[idet,"3"]
#                   self.top_w_idx = [i, a] 
#                   self.top_w_typ = typ 

            elif (typ == "aa"):
               i = int(self.data.loc[idet,"1"])
               j = int(self.data.loc[idet,"2"])
               a = int(self.data.loc[idet,"3"]) - self.nocc_cas
               b = int(self.data.loc[idet,"4"]) - self.nocc_cas
               ijab = self.idx.D(i, j, a, b) 
               parity = self.parity_ci_to_cc(i+j, 2)
               self.D_aa[ijab] = parity * self.data.loc[idet,"5"] 
#               if abs(self.data.loc[idet,"5"]) > self.top_w:
#                   self.top_w     = self.data.loc[idet,"5"]
#                   self.top_w_idx = [i, j, a, b] 
#                   self.top_w_typ = typ 
            elif (typ == "ab"):
               i = int(self.data.loc[idet,"1"])
               a = int(self.data.loc[idet,"2"]) - self.nocc_cas
               j = int(self.data.loc[idet,"3"])
               b = int(self.data.loc[idet,"4"]) - self.nocc_cas
               #print (idet,'i,a,j,b, ',i,a,j,b)
               ia = self.idx.S(i, a) 
               jb = self.idx.S(j, b) 
               parity  = self.parity_ci_to_cc(i, 1)
               parity *= self.parity_ci_to_cc(j, 1)
               self.D_ab[ia][jb] = parity * self.data.loc[idet,"5"] 
#               if abs(self.data.loc[idet,"5"]) > self.top_w:
#                   self.top_w     = self.data.loc[idet,"5"]
#                   self.top_w_idx = [i, j, a, b] 
#                   self.top_w_typ = typ 

        self.S_b    = self.S_a 
        self.D_bb   = self.D_aa


    def get_SDT(self, nc, numzero=1e-9):
        self.flagS = True 
        self.flagD = True
        self.flagT = True
        self.numzero = numzero

        self.nc = nc

        dS = int(self.nocc*self.nvir)
        dD = int(self.nocc*(self.nocc-1)*self.nvir*(self.nvir-1)/4) 
        dT = int(self.nocc*(self.nocc-1)*(self.nocc-2)*self.nvir*(self.nvir-1)*(self.nvir-2)/36) 
        #print('dS, dD, dT =',dS,dD,dT)

        nex = self.nocc * self.nvir
        self.Ref    = numpy.zeros((1), dtype=numpy.float64)
        self.S_a    = numpy.zeros((dS), dtype=numpy.float64)
        self.D_aa   = numpy.zeros((dD), dtype=numpy.float64)
        self.D_ab   = numpy.zeros((dS,dS), dtype=numpy.float64)
        self.T_aaa  = numpy.zeros((dT), dtype=numpy.float64)
        self.T_aab  = numpy.zeros((dD,dS), dtype=numpy.float64)
        for idet in range(self.num_det):
            typ = self.typ_det[idet]
            det_str = self.HFdet_str.copy()
            parity  = 1
            if (typ == "shci"):
               self.E_shci = self.data.loc[idet,"1"] 
            elif (typ == "rf"):
               self.Ref[0] = self.data.loc[idet,"1"] 
#               if abs(self.data.loc[idet,"1"]) > self.top_w:
#                   self.top_w     = self.data.loc[idet,"1"]
#                   self.top_w_idx = [0] 
#                   self.top_w_typ = typ 
            elif (typ == "a"):
               i = int(self.data.loc[idet,"1"]) + nc
               a = int(self.data.loc[idet,"2"]) - (self.nocc) + nc
               ia = self.idx.S(i, a) 
               parity = self.parity_ci_to_cc(i, 1)
               det_str[i] = 'b'
               det_str[a + self.nocc] = 'a'
               parity *= self.parity_ab_str(det_str)
               self.S_a[ia] = parity * self.data.loc[idet,"3"] 
#               if abs(self.data.loc[idet,"3"]) > self.top_w:
#                   self.top_w     = self.data.loc[idet,"3"]
#                   self.top_w_idx = [i, a] 
#                   self.top_w_typ = typ 
            elif (typ == "aa"):
               i = int(self.data.loc[idet,"1"]) + nc
               j = int(self.data.loc[idet,"2"]) + nc
               a = int(self.data.loc[idet,"3"]) - (self.nocc) + nc
               b = int(self.data.loc[idet,"4"]) - (self.nocc) + nc
               ijab = self.idx.D(i, j, a, b) 
               parity = self.parity_ci_to_cc(i+j, 2)
               det_str[i] = 'b'
               det_str[j] = 'b'
               det_str[a + self.nocc] = 'a'
               det_str[b + self.nocc] = 'a'
               parity *= self.parity_ab_str(det_str)
               self.D_aa[ijab] = parity * self.data.loc[idet,"5"] 
#               if abs(self.data.loc[idet,"5"]) > self.top_w:
#                   self.top_w     = self.data.loc[idet,"5"]
#                   self.top_w_idx = [i, j, a, b] 
#                   self.top_w_typ = typ 
            elif (typ == "ab"):
               i = int(self.data.loc[idet,"1"]) + nc
               a = int(self.data.loc[idet,"2"]) - (self.nocc) + nc
               j = int(self.data.loc[idet,"3"]) + nc
               b = int(self.data.loc[idet,"4"]) - (self.nocc) + nc
#               print (idet,'i,a,j,b',i,a,j,b)
               ia = self.idx.S(i, a) 
               jb = self.idx.S(j, b) 
               parity  = self.parity_ci_to_cc(i, 1)
               parity *= self.parity_ci_to_cc(j, 1)
               det_str[i] = 'b'
               det_str[a + self.nocc] = 'a'
               det_str[j] = 'a' if i != j else 'v'
               det_str[b + self.nocc] = 'b' if a != b else 'ab'
               parity *= self.parity_ab_str(det_str)
               self.D_ab[ia][jb] = parity * self.data.loc[idet,"5"] 
#               if abs(self.data.loc[idet,"5"]) > self.top_w:
#                   self.top_w     = self.data.loc[idet,"5"]
#                   self.top_w_idx = [i, j, a, b] 
#                   self.top_w_typ = typ 
            elif (typ == "aaa"):
               i = int(self.data.loc[idet,"1"]) + nc
               j = int(self.data.loc[idet,"2"]) + nc
               k = int(self.data.loc[idet,"3"]) + nc
               a = int(self.data.loc[idet,"4"]) - (self.nocc) + nc
               b = int(self.data.loc[idet,"5"]) - (self.nocc) + nc
               c = int(self.data.loc[idet,"6"]) - (self.nocc) + nc

#               #lsh test
#               if i == 1 and j == 2 and k == 3 \
#              and a == 0 and b == 2 and c == 4:
               if abs(float(self.data.loc[idet,"7"])) > numzero: 
                   ijkabc = self.idx.T(i, j, k, a, b, c) 
                   parity = self.parity_ci_to_cc(i+j+k, 3)
                   det_str[i] = 'b'
                   det_str[j] = 'b'
                   det_str[k] = 'b'
                   det_str[a + self.nocc] = 'a'
                   det_str[b + self.nocc] = 'a'
                   det_str[c + self.nocc] = 'a'
                   parity *= self.parity_ab_str(det_str)
                   self.T_aaa[ijkabc] = parity * self.data.loc[idet,"7"] 
#                   if abs(self.data.loc[idet,"7"]) > self.top_w:
#                       self.top_w     = self.data.loc[idet,"7"]
#                       self.top_w_idx = [i, j, k, a, b, c] 
#                       self.top_w_typ = typ 

#                   print("c3 in getSDT: ",self.T_aaa[ijkabc]) 

            elif (typ == "aab"):
               i = int(self.data.loc[idet,"1"]) + nc
               j = int(self.data.loc[idet,"2"]) + nc
               a = int(self.data.loc[idet,"3"]) - (self.nocc) + nc
               b = int(self.data.loc[idet,"4"]) - (self.nocc) + nc
               k = int(self.data.loc[idet,"5"]) + nc
               c = int(self.data.loc[idet,"6"]) - (self.nocc) + nc

               #lsh test
#               if i == 2 and j == 3 and k == 3 \
#              and a == 0 and b == 1 and c == 1:
               if abs(float(self.data.loc[idet,"7"])) > numzero: 
                   ijab = self.idx.D(i, j, a, b) 
                   kc   = self.idx.S(k, c) 
                   parity  = self.parity_ci_to_cc(i+j, 2)
                   parity *= self.parity_ci_to_cc(k, 1)
                   det_str[i] = 'b'
                   det_str[j] = 'b'
                   det_str[a + self.nocc] = 'a'
                   det_str[b + self.nocc] = 'a'
                   det_str[k] = 'a' if i != k and j != k else 'v'
                   det_str[c + self.nocc] = 'b' if a != c and b != c else 'ab'
                   parity *= self.parity_ab_str(det_str)
                   self.T_aab[ijab][kc] = parity * self.data.loc[idet,"7"] 
#                   if abs(self.data.loc[idet,"7"]) > self.top_w:
#                       self.top_w     = self.data.loc[idet,"7"]
#                       self.top_w_idx = [i, j, k, a, b, c] 
#                       self.top_w_typ = typ 
                   #print("c3 in getSDT: ",self.T_aab[ijab][kc]) 

#    def get_All_new_Fermi_vacuum(self, nc, typ_ref, idx_ref):
#        self.nc = nc
#        self.flagS = True 
#        self.flagD = True
#        self.flagT = True
#        self.flagQ = True
#
#        def ex_idx(va, va_ref):
#            idx_h = [] # index for hole
#            idx_p = [] # index for particle 
#            it = 0
#            for p in zip(va_ref, va): 
#                if p[0] == p[1]:
#                    continue
#                elif p[0] > p[1]:
#                    idx_h.append(it)
#                elif p[0] < p[1]:
#                    idx_p.append(it)
#                it += 1
#            return idx_h + idx_p, len(idx_h)
#
#        # Get a new Fermi vacuum
#        va_old = [1 for i in range(self.nocc)] + [0 for i in range(self.nvir)]
#        vb_old = [1 for i in range(self.nocc)] + [0 for i in range(self.nvir)]
#        va_ref = va_old.copy() 
#        vb_ref = vb_old.copy() 
#        if ( typ_ref == "shci"):
#            mo_occ = [sum(x) for x in zip(va_ref, vb_ref)]
#            return mo_occ
#
#        if typ_ref == "a":
#            raise NotImplementedError
#        elif typ_ref == "aa":
#            raise NotImplementedError
#        elif typ_ref == "ab":
#            ia, aa, ib, ab = idx_ref 
#            va_ref[ia] = 0 
#            va_ref[aa+self.nvir] = 1 
#            vb_ref[ib] = 0 
#            vb_ref[ab+self.nvir] = 1 
#        elif typ_ref == "aaa":
#            raise NotImplementedError
#        elif typ_ref == "aab":
#            raise NotImplementedError
#        elif typ_ref == "aabb":
#            raise NotImplementedError
#        elif typ_ref == "aaab":
#            raise NotImplementedError
#        else:
#            assert False
#
#        dS = int(self.nocc * self.nvir) 
#        dD = int(self.nocc*(self.nocc-1)*self.nvir*(self.nvir-1)/4) 
#        dT = int(self.nocc*(self.nocc-1)*(self.nocc-2)*self.nvir*(self.nvir-1)*(self.nvir-2)/36) 
#        print('dS, dD, dT =',dS,dD,dT)
#
#        self.Ref    = numpy.zeros((1), dtype=numpy.float64)
#        self.S_a    = numpy.zeros((dS), dtype=numpy.float64)
#        self.S_b    = numpy.zeros((dS), dtype=numpy.float64)
#        self.D_aa   = numpy.zeros((dD), dtype=numpy.float64)
#        self.D_ab   = numpy.zeros((dS,dS), dtype=numpy.float64)
#        self.D_bb   = numpy.zeros((dD), dtype=numpy.float64)
#        self.T_aaa  = numpy.zeros((dT), dtype=numpy.float64)
#        self.T_aab  = numpy.zeros((dD,dS), dtype=numpy.float64)
#        self.T_abb  = numpy.zeros((dS,dD), dtype=numpy.float64)
#        self.T_bbb  = numpy.zeros((dT), dtype=numpy.float64)
#        self.Q_aaab = numpy.zeros((dT,dS), dtype=numpy.float64)
#        self.Q_aabb = numpy.zeros((dD,dD), dtype=numpy.float64)
#
#        for idet in range(self.num_det):
#            typ = self.typ_det[idet]
#            va = va_old.copy()
#            vb = vb_old.copy()
#            parity  = 1
#            typ_new = ""
#            if (typ == "rf"):
#               idx_a, ndx_a = ex_idx(va, va_ref)
#               idx_b, ndx_b = ex_idx(vb, vb_ref)
#               for i in range(ndx_a): typ_new += "a" 
#               for i in range(ndx_b): typ_new += "b" 
#
#               val = self.data.loc[idet,"1"] 
#               if typ_new == "":
#                   self.Ref[0] = val 
#               elif typ_new== "a":
#                   i, a = idx_a
#                   ia = self.idx.S(i, a) 
#                   self.S_a[ia] = val
#               elif typ_new == "aa":
#               elif typ_new == "ab":
#               elif typ_new == "aaa":
#               elif typ_new == "aab":
#               elif typ_new == "aabb":
#               elif typ_new == "aaab":
#               else:
#                   print(typ, typ_new)
#
#
#            elif (typ == "a"):
#               i = int(self.data.loc[idet,"1"]) + nc
#               a = int(self.data.loc[idet,"2"]) - (self.nocc) + nc
#               ia = self.idx.S(i, a) 
#               parity = self.parity_ci_to_cc(i, 1)
#               det_str[i] = 'b'
#               det_str[a + self.nocc] = 'a'
#               parity *= self.parity_ab_str(det_str)
#               self.S_a[ia] = parity * self.data.loc[idet,"3"] 
#            elif (typ == "b"):
#               i = int(self.data.loc[idet,"1"]) + nc
#               a = int(self.data.loc[idet,"2"]) - (self.nocc) + nc
#               ia = self.idx.S(i, a) 
#               parity = self.parity_ci_to_cc(i, 1)
#               det_str[i] = 'a'
#               det_str[a + self.nocc] = 'b'
#               parity *= self.parity_ab_str(det_str)
#               self.S_b[ia] = parity * self.data.loc[idet,"3"] 
#            elif (typ == "aa"):
#               i = int(self.data.loc[idet,"1"]) + nc
#               j = int(self.data.loc[idet,"2"]) + nc
#               a = int(self.data.loc[idet,"3"]) - (self.nocc) + nc
#               b = int(self.data.loc[idet,"4"]) - (self.nocc) + nc
#               ijab = self.idx.D(i, j, a, b) 
#               parity = self.parity_ci_to_cc(i+j, 2)
#               det_str[i] = 'b'
#               det_str[j] = 'b'
#               det_str[a + self.nocc] = 'a'
#               det_str[b + self.nocc] = 'a'
#               parity *= self.parity_ab_str(det_str)
#               self.D_aa[ijab] = parity * self.data.loc[idet,"5"] 
#            elif (typ == "ab"):
#               i = int(self.data.loc[idet,"1"]) + nc
#               a = int(self.data.loc[idet,"2"]) - (self.nocc) + nc
#               j = int(self.data.loc[idet,"3"]) + nc
#               b = int(self.data.loc[idet,"4"]) - (self.nocc) + nc
##               print (idet,'i,a,j,b',i,a,j,b)
#               ia = self.idx.S(i, a) 
#               jb = self.idx.S(j, b) 
#               parity  = self.parity_ci_to_cc(i, 1)
#               parity *= self.parity_ci_to_cc(j, 1)
#               det_str[i] = 'b'
#               det_str[a + self.nocc] = 'a'
#               det_str[j] = 'a' if i != j else 'v'
#               det_str[b + self.nocc] = 'b' if a != b else 'ab'
#               parity *= self.parity_ab_str(det_str)
#               self.D_ab[ia][jb] = parity * self.data.loc[idet,"5"] 
#            elif (typ == "bb"):
#               i = int(self.data.loc[idet,"1"]) + nc
#               j = int(self.data.loc[idet,"2"]) + nc
#               a = int(self.data.loc[idet,"3"]) - (self.nocc) + nc
#               b = int(self.data.loc[idet,"4"]) - (self.nocc) + nc
#               ijab = self.idx.D(i, j, a, b) 
#               parity = self.parity_ci_to_cc(i+j, 2)
#               det_str[i] = 'a'
#               det_str[j] = 'a'
#               det_str[a + self.nocc] = 'b'
#               det_str[b + self.nocc] = 'b'
#               parity *= self.parity_ab_str(det_str)
#               self.D_bb[ijab] = parity * self.data.loc[idet,"5"] 
#            elif (typ == "aaa"):
#               i = int(self.data.loc[idet,"1"]) + nc
#               j = int(self.data.loc[idet,"2"]) + nc
#               k = int(self.data.loc[idet,"3"]) + nc
#               a = int(self.data.loc[idet,"4"]) - (self.nocc) + nc
#               b = int(self.data.loc[idet,"5"]) - (self.nocc) + nc
#               c = int(self.data.loc[idet,"6"]) - (self.nocc) + nc
#               ijkabc = self.idx.T(i, j, k, a, b, c) 
#               parity = self.parity_ci_to_cc(i+j+k, 3)
#               det_str[i] = 'b'
#               det_str[j] = 'b'
#               det_str[k] = 'b'
#               det_str[a + self.nocc] = 'a'
#               det_str[b + self.nocc] = 'a'
#               det_str[c + self.nocc] = 'a'
#               parity *= self.parity_ab_str(det_str)
#               self.T_aaa[ijkabc] = parity * self.data.loc[idet,"7"] 
#            elif (typ == "aab"):
#               i = int(self.data.loc[idet,"1"]) + nc
#               j = int(self.data.loc[idet,"2"]) + nc
#               a = int(self.data.loc[idet,"3"]) - (self.nocc) + nc
#               b = int(self.data.loc[idet,"4"]) - (self.nocc) + nc
#               k = int(self.data.loc[idet,"5"]) + nc
#               c = int(self.data.loc[idet,"6"]) - (self.nocc) + nc
#               ijab = self.idx.D(i, j, a, b) 
#               kc   = self.idx.S(k, c) 
#               parity  = self.parity_ci_to_cc(i+j, 2)
#               parity *= self.parity_ci_to_cc(k, 1)
#               det_str[i] = 'b'
#               det_str[j] = 'b'
#               det_str[a + self.nocc] = 'a'
#               det_str[b + self.nocc] = 'a'
#               det_str[k] = 'a' if i != k and j != k else 'v'
#               det_str[c + self.nocc] = 'b' if a != c and b != c else 'ab'
#               parity *= self.parity_ab_str(det_str)
#               self.T_aab[ijab][kc] = parity * self.data.loc[idet,"7"] 
#            elif (typ == "abb"):
#               i = int(self.data.loc[idet,"1"]) + nc
#               a = int(self.data.loc[idet,"2"]) - (self.nocc) + nc
#               j = int(self.data.loc[idet,"3"]) + nc
#               k = int(self.data.loc[idet,"4"]) + nc
#               b = int(self.data.loc[idet,"5"]) - (self.nocc) + nc
#               c = int(self.data.loc[idet,"6"]) - (self.nocc) + nc
#               ia   = self.idx.S(i, a) 
#               jkab = self.idx.D(j, k, b, c) 
#               parity = self.parity_ci_to_cc(i, 1)
#               parity *= self.parity_ci_to_cc(j+k, 2)
#               det_str[i] = 'b'
#               det_str[a + self.nocc] = 'a'
#               det_str[j] = 'a' if i != j else 'v'
#               det_str[k] = 'a' if i != k else 'v'
#               det_str[b + self.nocc] = 'b' if a != b else 'ab'
#               det_str[c + self.nocc] = 'b' if a != c else 'ab'
#               parity *= self.parity_ab_str(det_str)
#               self.T_abb[ia][jkab] = parity * self.data.loc[idet,"7"] 
#            elif (typ == "bbb"):
#               i = int(self.data.loc[idet,"1"]) + nc
#               j = int(self.data.loc[idet,"2"]) + nc
#               k = int(self.data.loc[idet,"3"]) + nc
#               a = int(self.data.loc[idet,"4"]) - (self.nocc) + nc
#               b = int(self.data.loc[idet,"5"]) - (self.nocc) + nc
#               c = int(self.data.loc[idet,"6"]) - (self.nocc) + nc
#               ijkabc = self.idx.T(i, j, k, a, b, c) 
#               parity = self.parity_ci_to_cc(i+j+k, 3)
#               det_str[i] = 'a'
#               det_str[j] = 'a'
#               det_str[k] = 'a'
#               det_str[a + self.nocc] = 'b'
#               det_str[b + self.nocc] = 'b'
#               det_str[c + self.nocc] = 'b'
#               parity *= self.parity_ab_str(det_str)
#               self.T_bbb[ijkabc] = parity * self.data.loc[idet,"7"] 
#            elif (typ == "aaaa"):
#               i = int(self.data.loc[idet,"1"]) + nc
#               j = int(self.data.loc[idet,"2"]) + nc
#               k = int(self.data.loc[idet,"3"]) + nc
#               l = int(self.data.loc[idet,"4"]) + nc
#               a = int(self.data.loc[idet,"5"]) - (self.nocc) + nc
#               b = int(self.data.loc[idet,"6"]) - (self.nocc) + nc
#               c = int(self.data.loc[idet,"7"]) - (self.nocc) + nc
#               d = int(self.data.loc[idet,"8"]) - (self.nocc) + nc
#               ijklabcd = self.idx.Q(i, j, k, l, a, b, c, d) 
#               parity = self.parity_ci_to_cc(i+j+k+l, 4)
#               det_str[i] = 'b'
#               det_str[j] = 'b'
#               det_str[k] = 'b'
#               det_str[l] = 'b'
#               det_str[a + self.nocc] = 'a'
#               det_str[b + self.nocc] = 'a'
#               det_str[c + self.nocc] = 'a'
#               det_str[d + self.nocc] = 'a'
#               parity *= self.parity_ab_str(det_str)
#               self.Q_aaaa[ijklabcd] = parity * self.data.loc[idet,"9"] 
#            elif (typ == "aaab"):
#               i = int(self.data.loc[idet,"1"]) + nc
#               j = int(self.data.loc[idet,"2"]) + nc
#               k = int(self.data.loc[idet,"3"]) + nc
#               a = int(self.data.loc[idet,"4"]) - (self.nocc) + nc
#               b = int(self.data.loc[idet,"5"]) - (self.nocc) + nc
#               c = int(self.data.loc[idet,"6"]) - (self.nocc) + nc
#               l = int(self.data.loc[idet,"7"]) + nc
#               d = int(self.data.loc[idet,"8"]) - (self.nocc) + nc
#               ijkabc = self.idx.T(i, j, k, a, b, c) 
#               ld     = self.idx.S(l, d) 
#               parity = self.parity_ci_to_cc(i+j+k, 3)
#               parity *= self.parity_ci_to_cc(l, 1)
#               det_str[i] = 'b'
#               det_str[j] = 'b'
#               det_str[k] = 'b'
#               det_str[a + self.nocc] = 'a'
#               det_str[b + self.nocc] = 'a'
#               det_str[c + self.nocc] = 'a'
#               det_str[l] = 'a' if i != l and j != l and k != l else 'v'
#               det_str[d + self.nocc] = 'b' if a != d and b != d and c != d else 'ab'
#               parity *= self.parity_ab_str(det_str)
#               self.Q_aaab[ijkabc][ld] = parity * self.data.loc[idet,"9"] 
#            elif (typ == "aabb"):
#               i = int(self.data.loc[idet,"1"]) + nc
#               j = int(self.data.loc[idet,"2"]) + nc
#               a = int(self.data.loc[idet,"3"]) - (self.nocc) + nc
#               b = int(self.data.loc[idet,"4"]) - (self.nocc) + nc
#               k = int(self.data.loc[idet,"5"]) + nc
#               l = int(self.data.loc[idet,"6"]) + nc
#               c = int(self.data.loc[idet,"7"]) - (self.nocc) + nc
#               d = int(self.data.loc[idet,"8"]) - (self.nocc) + nc
#               #lsh test
##               if i == 2 and j == 3 and a == 0 and b == 1 \
##              and k == 2 and l == 3 and c == 0 and d == 1:
##               if i == 0 and j == 2 and a == 0 and b == 3 \
##              and k == 0 and l == 2 and c == 2 and d == 5:
#               if True:
#                   ijab = self.idx.D(i, j, a, b) 
#                   klcd = self.idx.D(k, l, c, d) 
#                   parity = self.parity_ci_to_cc(i+j, 2)
#                   parity *= self.parity_ci_to_cc(k+l, 2)
#                   det_str[i] = 'b'
#                   det_str[j] = 'b'
#                   det_str[a + self.nocc] = 'a'
#                   det_str[b + self.nocc] = 'a'
#                   det_str[k] = 'a' if i != k and j != k else 'v'
#                   det_str[l] = 'a' if i != l and j != l else 'v'
#                   det_str[c + self.nocc] = 'b' if a != c and b != c else 'ab'
#                   det_str[d + self.nocc] = 'b' if a != d and b != d else 'ab'
#                   parity *= self.parity_ab_str(det_str)
#                   self.Q_aabb[ijab][klcd] = parity * self.data.loc[idet,"9"] 
#
#                   #lsh test
##                   print('c4 org',parity * self.data.loc[idet,"9"]/self.Ref[0])
#
##            elif (typ == "abbb"):
##               i = int(self.data.loc[idet,"1"]) + nc
##               a = int(self.data.loc[idet,"2"]) - (self.nocc) + nc
##               j = int(self.data.loc[idet,"3"]) + nc
##               k = int(self.data.loc[idet,"4"]) + nc
##               l = int(self.data.loc[idet,"5"]) + nc
##               b = int(self.data.loc[idet,"6"]) - (self.nocc) + nc
##               c = int(self.data.loc[idet,"7"]) - (self.nocc) + nc
##               d = int(self.data.loc[idet,"8"]) - (self.nocc) + nc
##               ia     = self.idx.S(i, a) 
##               jklbcd = self.idx.T(j, k, l, b, c, d) 
##               parity = self.parity_ci_to_cc(i, 1)
##               parity *= self.parity_ci_to_cc(j+k+l, 3)
##               det_str[i] = 'b'
##               det_str[a + self.nocc] = 'a'
##               det_str[j] = 'a' if i != j else 'v'
##               det_str[k] = 'a' if i != k else 'v'
##               det_str[l] = 'a' if i != l else 'v'
##               det_str[b + self.nocc] = 'b' if a != b else 'ab'
##               det_str[c + self.nocc] = 'b' if a != c else 'ab'
##               det_str[d + self.nocc] = 'b' if a != d else 'ab'
##               parity *= self.parity_ab_str(det_str)
##               self.Q_abbb[ia][jklbcd] = parity * self.data.loc[idet,"9"]
##            elif (typ == "bbbb"):
##               i = int(self.data.loc[idet,"1"]) + nc
##               j = int(self.data.loc[idet,"2"]) + nc
##               k = int(self.data.loc[idet,"3"]) + nc
##               l = int(self.data.loc[idet,"4"]) + nc
##               a = int(self.data.loc[idet,"5"]) - (self.nocc) + nc
##               b = int(self.data.loc[idet,"6"]) - (self.nocc) + nc
##               c = int(self.data.loc[idet,"7"]) - (self.nocc) + nc
##               d = int(self.data.loc[idet,"8"]) - (self.nocc) + nc
##               ijklabcd = self.idx.Q(i, j, k, l, a, b, c, d) 
##               parity = self.parity_ci_to_cc(i+j+k+l, 4)
##               det_str[i] = 'a'
##               det_str[j] = 'a'
##               det_str[k] = 'a'
##               det_str[l] = 'a'
##               det_str[a + self.nocc] = 'b'
##               det_str[b + self.nocc] = 'b'
##               det_str[c + self.nocc] = 'b'
##               det_str[d + self.nocc] = 'b'
##               parity *= self.parity_ab_str(det_str)
##               self.Q_bbbb[ijklabcd] = parity * self.data.loc[idet,"9"] 
##        print ('test max:',test_max, test_max_idx) 
#
    def get_t2t4(self, t2t4c, e2ovov, ci2cc, numzero, nc):
        self.flagQ = True 
        tmp   = numpy.zeros((self.nocc,self.nocc,self.nvir,self.nvir), dtype=numpy.float64)
        self.interm_norm(Q=False)

        #TODO: put this assertion to cn_to_tn
        assert self.interm_norm_S and self.interm_norm_D and self.interm_norm_T
        ci2cc.c1_to_t1(self.S_a.copy())
        print('=== END extracting S t amplitudes ===')
        ci2cc.c2_to_t2(self.D_aa.copy(),self.D_ab.copy())
        print('=== END extracting D t amplitudes ===')
        #TODO: argument numzero as instance of ci2cc 
        #TODO: avoide gen t3 amplitudes 
        ci2cc.c3_to_t3(self.T_aaa.copy(), self.T_aab.copy(),numzero=numzero)
        print('=== END extracting T t amplitudes ===')

        test_max = 0.0
        test_max_idx = [] 

        #for dbg
        #self.t4aabb = numpy.zeros((self.nocc,self.nocc,self.nocc,self.nocc,self.nvir,self.nvir,self.nvir,self.nvir), dtype=numpy.float64)

        for idet in range(self.num_det):
            typ = self.typ_det[idet]
            det_str = self.HFdet_str.copy()
            parity  = 1
            if (typ == "aaab"):
               p = int(self.data.loc[idet,"1"]) + nc
               q = int(self.data.loc[idet,"2"]) + nc
               r = int(self.data.loc[idet,"3"]) + nc
               t = int(self.data.loc[idet,"4"]) - (self.nocc) + nc
               u = int(self.data.loc[idet,"5"]) - (self.nocc) + nc
               v = int(self.data.loc[idet,"6"]) - (self.nocc) + nc
               s = int(self.data.loc[idet,"7"]) + nc
               w = int(self.data.loc[idet,"8"]) - (self.nocc) + nc
               pqrtuv = self.idx.T(p, q, r, t, u, v) 
               sw     = self.idx.S(s, w) 
               parity = self.parity_ci_to_cc(p+q+r, 3)
               parity *= self.parity_ci_to_cc(s, 1)
               det_str[p] = 'b'
               det_str[q] = 'b'
               det_str[r] = 'b'
               det_str[t + self.nocc] = 'a'
               det_str[u + self.nocc] = 'a'
               det_str[v + self.nocc] = 'a'
               det_str[s] = 'a' if p != s and q != s and r != s else 'v'
               det_str[w + self.nocc] = 'b' if t != w and u != w and v != w else 'ab'
               parity *= self.parity_ab_str(det_str)
               c4_internorm = parity * self.data.loc[idet,"9"] / self.Ref[0] 

               t4ind_aaab  = ci2cc.t1xt3aaab(p, q, r, s, t, u, v, w)
               t4ind_aaab += ci2cc.t2xt2aaab(p, q, r, s, t, u, v, w)
               t4ind_aaab += ci2cc.t1xt1xt2aaab(p, q, r, s, t, u, v, w)
               t4ind_aaab += ci2cc.t1xt1xt1xt1aaab(p, q, r, s, t, u, v, w)
               t4 = c4_internorm - t4ind_aaab

               tmp.fill(0)
               tmp[r][s][v][w] += (e2ovov[p][t][q][u]-e2ovov[p][u][q][t]) * t4  
               tmp[q][s][v][w] -= (e2ovov[p][t][r][u]-e2ovov[p][u][r][t]) * t4  
               tmp[p][s][v][w] += (e2ovov[q][t][s][u]-e2ovov[q][u][s][t]) * t4  

               tmp[r][s][u][w] -= (e2ovov[p][t][q][v]-e2ovov[p][v][q][t]) * t4  
               tmp[q][s][u][w] += (e2ovov[p][t][r][v]-e2ovov[p][v][r][t]) * t4  
               tmp[p][s][u][w] -= (e2ovov[q][t][s][v]-e2ovov[q][v][s][t]) * t4  

               tmp[r][s][t][w] += (e2ovov[p][u][q][v]-e2ovov[p][v][q][u]) * t4  
               tmp[q][s][t][w] -= (e2ovov[p][u][r][v]-e2ovov[p][v][r][u]) * t4  
               tmp[p][s][t][w] += (e2ovov[q][u][s][v]-e2ovov[q][v][s][u]) * t4  

               t2t4c += tmp + tmp.transpose(1,0,3,2) 

            elif (typ == "aabb"):
               p = int(self.data.loc[idet,"1"]) + nc
               q = int(self.data.loc[idet,"2"]) + nc
               t = int(self.data.loc[idet,"3"]) - (self.nocc) + nc
               u = int(self.data.loc[idet,"4"]) - (self.nocc) + nc
               r = int(self.data.loc[idet,"5"]) + nc
               s = int(self.data.loc[idet,"6"]) + nc
               v = int(self.data.loc[idet,"7"]) - (self.nocc) + nc
               w = int(self.data.loc[idet,"8"]) - (self.nocc) + nc
               #lsh test
#               if p == 0 and q == 2 and t == 0 and u == 3 \
#              and r == 0 and s == 2 and v == 2 and w == 5:
               if True:
                   pqtu = self.idx.D(p, q, t, u) 
                   rsvw = self.idx.D(r, s, v, w) 
                   parity = self.parity_ci_to_cc(p+q, 2)
                   parity *= self.parity_ci_to_cc(r+s, 2)
                   det_str[p] = 'b'
                   det_str[q] = 'b'
                   det_str[t + self.nocc] = 'a'
                   det_str[u + self.nocc] = 'a'
                   det_str[r] = 'a' if p != r and q != r else 'v'
                   det_str[s] = 'a' if p != s and q != s else 'v'
                   det_str[v + self.nocc] = 'b' if t != v and u != v else 'ab'
                   det_str[w + self.nocc] = 'b' if t != w and u != w else 'ab'
                   parity *= self.parity_ab_str(det_str)
    
#                   # lsh test
#                   if abs(self.data.loc[idet,"9"]) > test_max:
#                       test_max = self.data.loc[idet,"9"]
#                       test_max_idx = [p, q, t, u, r, s, v, w] 
    
                   c4_internorm = parity * self.data.loc[idet,"9"] / self.Ref[0]  
#                   print('c4mem =', p, q, r, s, t, u, v, w, c4_internorm )
    
                   t4ind_aabb  = ci2cc.t1xt3aabb(p, q, r, s, t, u, v, w)
                   t4ind_aabb += ci2cc.t2xt2aabb(p, q, r, s, t, u, v, w)
                   t4ind_aabb += ci2cc.t1xt1xt2aabb(p, q, r, s, t, u, v, w)
                   t4ind_aabb += ci2cc.t1xt1xt1xt1aabb(p, q, r, s, t, u, v, w)

                   t4 = c4_internorm - t4ind_aabb
                   #for dbg   
#                   self.t4aabb[p][q][r][s][t][u][v][w] = t4 
#                   print('t4mem =', p, q, r, s, t, u, v, w, t4 )

                   tmp.fill(0)
                   if p<r and t<v: tmp[q][s][u][w] += e2ovov[p][t][r][v] * t4
                   if q<r and t<v: tmp[p][s][u][w] -= e2ovov[q][t][r][v] * t4
                   if p<s and t<v: tmp[q][r][u][w] -= e2ovov[p][t][s][v] * t4
                   if q<s and t<v: tmp[p][r][u][w] += e2ovov[q][t][s][v] * t4
                   if p<r and u<v: tmp[q][s][t][w] -= e2ovov[p][u][r][v] * t4
                   if q<r and u<v: tmp[p][s][t][w] += e2ovov[q][u][r][v] * t4
                   if p<s and u<v: tmp[q][r][t][w] += e2ovov[p][u][s][v] * t4
                   if q<s and u<v: tmp[p][r][t][w] -= e2ovov[q][u][s][v] * t4
                   if p<r and t<w: tmp[q][s][u][v] -= e2ovov[p][t][r][w] * t4
                   if q<r and t<w: tmp[p][s][u][v] += e2ovov[q][t][r][w] * t4
                   if p<s and t<w: tmp[q][r][u][v] += e2ovov[p][t][s][w] * t4
                   if q<s and t<w: tmp[p][r][u][v] -= e2ovov[q][t][s][w] * t4
                   if p<r and u<w: tmp[q][s][t][v] += e2ovov[p][u][r][w] * t4
                   if q<r and u<w: tmp[p][s][t][v] -= e2ovov[q][u][r][w] * t4
                   if p<s and u<w: tmp[q][r][t][v] -= e2ovov[p][u][s][w] * t4
                   if q<s and u<w: tmp[p][r][t][v] += e2ovov[q][u][s][w] * t4
                   if p<r and v<t: tmp[q][s][u][w] += e2ovov[p][t][r][v] * t4
                   if q<r and v<t: tmp[p][s][u][w] -= e2ovov[q][t][r][v] * t4
                   if p<s and v<t: tmp[q][r][u][w] -= e2ovov[p][t][s][v] * t4
                   if q<s and v<t: tmp[p][r][u][w] += e2ovov[q][t][s][v] * t4
                   if p<r and v<u: tmp[q][s][t][w] -= e2ovov[p][u][r][v] * t4
                   if q<r and v<u: tmp[p][s][t][w] += e2ovov[q][u][r][v] * t4
                   if p<s and v<u: tmp[q][r][t][w] += e2ovov[p][u][s][v] * t4
                   if q<s and v<u: tmp[p][r][t][w] -= e2ovov[q][u][s][v] * t4
                   if p<r and w<t: tmp[q][s][u][v] -= e2ovov[p][t][r][w] * t4
                   if q<r and w<t: tmp[p][s][u][v] += e2ovov[q][t][r][w] * t4
                   if p<s and w<t: tmp[q][r][u][v] += e2ovov[p][t][s][w] * t4
                   if q<s and w<t: tmp[p][r][u][v] -= e2ovov[q][t][s][w] * t4
                   if p<r and w<u: tmp[q][s][t][v] += e2ovov[p][u][r][w] * t4
                   if q<r and w<u: tmp[p][s][t][v] -= e2ovov[q][u][r][w] * t4
                   if p<s and w<u: tmp[q][r][t][v] -= e2ovov[p][u][s][w] * t4
                   if q<s and w<u: tmp[p][r][t][v] += e2ovov[q][u][s][w] * t4

                   scale = numpy.float64(0.5)
                   if p==r and t<v: tmp[q][s][u][w] += e2ovov[p][t][r][v] * t4 * scale 
                   if q==r and t<v: tmp[p][s][u][w] -= e2ovov[q][t][r][v] * t4 * scale  
                   if p==s and t<v: tmp[q][r][u][w] -= e2ovov[p][t][s][v] * t4 * scale  
                   if q==s and t<v: tmp[p][r][u][w] += e2ovov[q][t][s][v] * t4 * scale  
                   if p==r and u<v: tmp[q][s][t][w] -= e2ovov[p][u][r][v] * t4 * scale   
                   if q==r and u<v: tmp[p][s][t][w] += e2ovov[q][u][r][v] * t4 * scale   
                   if p==s and u<v: tmp[q][r][t][w] += e2ovov[p][u][s][v] * t4 * scale   
                   if q==s and u<v: tmp[p][r][t][w] -= e2ovov[q][u][s][v] * t4 * scale   
                   if p==r and t<w: tmp[q][s][u][v] -= e2ovov[p][t][r][w] * t4 * scale    
                   if q==r and t<w: tmp[p][s][u][v] += e2ovov[q][t][r][w] * t4 * scale    
                   if p==s and t<w: tmp[q][r][u][v] += e2ovov[p][t][s][w] * t4 * scale    
                   if q==s and t<w: tmp[p][r][u][v] -= e2ovov[q][t][s][w] * t4 * scale    
                   if p==r and u<w: tmp[q][s][t][v] += e2ovov[p][u][r][w] * t4 * scale    
                   if q==r and u<w: tmp[p][s][t][v] -= e2ovov[q][u][r][w] * t4 * scale    
                   if p==s and u<w: tmp[q][r][t][v] -= e2ovov[p][u][s][w] * t4 * scale    
                   if q==s and u<w: tmp[p][r][t][v] += e2ovov[q][u][s][w] * t4 * scale    
                   if p==r and v<t: tmp[q][s][u][w] += e2ovov[p][t][r][v] * t4 * scale    
                   if q==r and v<t: tmp[p][s][u][w] -= e2ovov[q][t][r][v] * t4 * scale    
                   if p==s and v<t: tmp[q][r][u][w] -= e2ovov[p][t][s][v] * t4 * scale    
                   if q==s and v<t: tmp[p][r][u][w] += e2ovov[q][t][s][v] * t4 * scale    
                   if p==r and v<u: tmp[q][s][t][w] -= e2ovov[p][u][r][v] * t4 * scale    
                   if q==r and v<u: tmp[p][s][t][w] += e2ovov[q][u][r][v] * t4 * scale    
                   if p==s and v<u: tmp[q][r][t][w] += e2ovov[p][u][s][v] * t4 * scale    
                   if q==s and v<u: tmp[p][r][t][w] -= e2ovov[q][u][s][v] * t4 * scale    
                   if p==r and w<t: tmp[q][s][u][v] -= e2ovov[p][t][r][w] * t4 * scale    
                   if q==r and w<t: tmp[p][s][u][v] += e2ovov[q][t][r][w] * t4 * scale    
                   if p==s and w<t: tmp[q][r][u][v] += e2ovov[p][t][s][w] * t4 * scale    
                   if q==s and w<t: tmp[p][r][u][v] -= e2ovov[q][t][s][w] * t4 * scale    
                   if p==r and w<u: tmp[q][s][t][v] += e2ovov[p][u][r][w] * t4 * scale    
                   if q==r and w<u: tmp[p][s][t][v] -= e2ovov[q][u][r][w] * t4 * scale    
                   if p==s and w<u: tmp[q][r][t][v] -= e2ovov[p][u][s][w] * t4 * scale    
                   if q==s and w<u: tmp[p][r][t][v] += e2ovov[q][u][s][w] * t4 * scale    

                   scale = numpy.float64(0.5)
                   if p<r and t==v: tmp[q][s][u][w] += e2ovov[p][t][r][v] * t4 * scale    
                   if q<r and t==v: tmp[p][s][u][w] -= e2ovov[q][t][r][v] * t4 * scale    
                   if p<s and t==v: tmp[q][r][u][w] -= e2ovov[p][t][s][v] * t4 * scale    
                   if q<s and t==v: tmp[p][r][u][w] += e2ovov[q][t][s][v] * t4 * scale    
                   if p<r and u==v: tmp[q][s][t][w] -= e2ovov[p][u][r][v] * t4 * scale    
                   if q<r and u==v: tmp[p][s][t][w] += e2ovov[q][u][r][v] * t4 * scale    
                   if p<s and u==v: tmp[q][r][t][w] += e2ovov[p][u][s][v] * t4 * scale    
                   if q<s and u==v: tmp[p][r][t][w] -= e2ovov[q][u][s][v] * t4 * scale    
                   if p<r and t==w: tmp[q][s][u][v] -= e2ovov[p][t][r][w] * t4 * scale    
                   if q<r and t==w: tmp[p][s][u][v] += e2ovov[q][t][r][w] * t4 * scale    
                   if p<s and t==w: tmp[q][r][u][v] += e2ovov[p][t][s][w] * t4 * scale    
                   if q<s and t==w: tmp[p][r][u][v] -= e2ovov[q][t][s][w] * t4 * scale    
                   if p<r and u==w: tmp[q][s][t][v] += e2ovov[p][u][r][w] * t4 * scale    
                   if q<r and u==w: tmp[p][s][t][v] -= e2ovov[q][u][r][w] * t4 * scale    
                   if p<s and u==w: tmp[q][r][t][v] -= e2ovov[p][u][s][w] * t4 * scale    
                   if q<s and u==w: tmp[p][r][t][v] += e2ovov[q][u][s][w] * t4 * scale    
                   if p<r and v==t: tmp[q][s][u][w] += e2ovov[p][t][r][v] * t4 * scale    
                   if q<r and v==t: tmp[p][s][u][w] -= e2ovov[q][t][r][v] * t4 * scale    
                   if p<s and v==t: tmp[q][r][u][w] -= e2ovov[p][t][s][v] * t4 * scale    
                   if q<s and v==t: tmp[p][r][u][w] += e2ovov[q][t][s][v] * t4 * scale    
                   if p<r and v==u: tmp[q][s][t][w] -= e2ovov[p][u][r][v] * t4 * scale    
                   if q<r and v==u: tmp[p][s][t][w] += e2ovov[q][u][r][v] * t4 * scale    
                   if p<s and v==u: tmp[q][r][t][w] += e2ovov[p][u][s][v] * t4 * scale    
                   if q<s and v==u: tmp[p][r][t][w] -= e2ovov[q][u][s][v] * t4 * scale    
                   if p<r and w==t: tmp[q][s][u][v] -= e2ovov[p][t][r][w] * t4 * scale    
                   if q<r and w==t: tmp[p][s][u][v] += e2ovov[q][t][r][w] * t4 * scale    
                   if p<s and w==t: tmp[q][r][u][v] += e2ovov[p][t][s][w] * t4 * scale    
                   if q<s and w==t: tmp[p][r][u][v] -= e2ovov[q][t][s][w] * t4 * scale    
                   if p<r and w==u: tmp[q][s][t][v] += e2ovov[p][u][r][w] * t4 * scale    
                   if q<r and w==u: tmp[p][s][t][v] -= e2ovov[q][u][r][w] * t4 * scale    
                   if p<s and w==u: tmp[q][r][t][v] -= e2ovov[p][u][s][w] * t4 * scale    
                   if q<s and w==u: tmp[p][r][t][v] += e2ovov[q][u][s][w] * t4 * scale    

                   scale = numpy.float64(0.25)
                   if p==r and t==v: tmp[q][s][u][w] += e2ovov[p][t][r][v] * t4 * scale    
                   if q==r and t==v: tmp[p][s][u][w] -= e2ovov[q][t][r][v] * t4 * scale    
                   if p==s and t==v: tmp[q][r][u][w] -= e2ovov[p][t][s][v] * t4 * scale    
                   if q==s and t==v: tmp[p][r][u][w] += e2ovov[q][t][s][v] * t4 * scale    
                   if p==r and u==v: tmp[q][s][t][w] -= e2ovov[p][u][r][v] * t4 * scale    
                   if q==r and u==v: tmp[p][s][t][w] += e2ovov[q][u][r][v] * t4 * scale    
                   if p==s and u==v: tmp[q][r][t][w] += e2ovov[p][u][s][v] * t4 * scale    
                   if q==s and u==v: tmp[p][r][t][w] -= e2ovov[q][u][s][v] * t4 * scale    
                   if p==r and t==w: tmp[q][s][u][v] -= e2ovov[p][t][r][w] * t4 * scale    
                   if q==r and t==w: tmp[p][s][u][v] += e2ovov[q][t][r][w] * t4 * scale    
                   if p==s and t==w: tmp[q][r][u][v] += e2ovov[p][t][s][w] * t4 * scale    
                   if q==s and t==w: tmp[p][r][u][v] -= e2ovov[q][t][s][w] * t4 * scale    
                   if p==r and u==w: tmp[q][s][t][v] += e2ovov[p][u][r][w] * t4 * scale    
                   if q==r and u==w: tmp[p][s][t][v] -= e2ovov[q][u][r][w] * t4 * scale    
                   if p==s and u==w: tmp[q][r][t][v] -= e2ovov[p][u][s][w] * t4 * scale    
                   if q==s and u==w: tmp[p][r][t][v] += e2ovov[q][u][s][w] * t4 * scale    
                   if p==r and v==t: tmp[q][s][u][w] += e2ovov[p][t][r][v] * t4 * scale    
                   if q==r and v==t: tmp[p][s][u][w] -= e2ovov[q][t][r][v] * t4 * scale    
                   if p==s and v==t: tmp[q][r][u][w] -= e2ovov[p][t][s][v] * t4 * scale    
                   if q==s and v==t: tmp[p][r][u][w] += e2ovov[q][t][s][v] * t4 * scale    
                   if p==r and v==u: tmp[q][s][t][w] -= e2ovov[p][u][r][v] * t4 * scale    
                   if q==r and v==u: tmp[p][s][t][w] += e2ovov[q][u][r][v] * t4 * scale    
                   if p==s and v==u: tmp[q][r][t][w] += e2ovov[p][u][s][v] * t4 * scale    
                   if q==s and v==u: tmp[p][r][t][w] -= e2ovov[q][u][s][v] * t4 * scale    
                   if p==r and w==t: tmp[q][s][u][v] -= e2ovov[p][t][r][w] * t4 * scale    
                   if q==r and w==t: tmp[p][s][u][v] += e2ovov[q][t][r][w] * t4 * scale    
                   if p==s and w==t: tmp[q][r][u][v] += e2ovov[p][t][s][w] * t4 * scale    
                   if q==s and w==t: tmp[p][r][u][v] -= e2ovov[q][t][s][w] * t4 * scale    
                   if p==r and w==u: tmp[q][s][t][v] += e2ovov[p][u][r][w] * t4 * scale    
                   if q==r and w==u: tmp[p][s][t][v] -= e2ovov[q][u][r][w] * t4 * scale    
                   if p==s and w==u: tmp[q][r][t][v] -= e2ovov[p][u][s][w] * t4 * scale    
                   if q==s and w==u: tmp[p][r][t][v] += e2ovov[q][u][s][w] * t4 * scale    
    
                   t2t4c += tmp + tmp.transpose(1,0,3,2) 
        print('=== END contracting Q t amplitudes ===')

    def get_t2t4c(self, t2t4c, e2ovov, ci2cc, numzero, nc, norm):
        _ccsd.libcc.t2t4c_shci(t2t4c.ctypes.data_as(ctypes.c_void_p),
                          ci2cc.t1.ctypes.data_as(ctypes.c_void_p),
                          ci2cc.t2aa.ctypes.data_as(ctypes.c_void_p),
                          ci2cc.t2ab.ctypes.data_as(ctypes.c_void_p),
                          ci2cc.t3aaa.ctypes.data_as(ctypes.c_void_p),
                          ci2cc.t3aab.ctypes.data_as(ctypes.c_void_p),
                          e2ovov.ctypes.data_as(ctypes.c_void_p),
                          ctypes.c_int(nc),
                          ctypes.c_int(self.nocc),ctypes.c_int(self.nvir),
                          ctypes.c_double(numzero),ctypes.c_double(self.Ref[0]),
                          ctypes.c_double(norm)) 

    def get_t2t3c(self, t2_t3t4c, tmp1, tmp2, tmp3, ci2cc, numzero, nc):
        _ccsd.libcc.t2t3c_shci(t2_t3t4c.ctypes.data_as(ctypes.c_void_p),
                          ci2cc.t1.ctypes.data_as(ctypes.c_void_p),
                          ci2cc.t2aa.ctypes.data_as(ctypes.c_void_p),
                          ci2cc.t2ab.ctypes.data_as(ctypes.c_void_p),
                          tmp1.ctypes.data_as(ctypes.c_void_p),
                          tmp2.ctypes.data_as(ctypes.c_void_p),
                          tmp3.ctypes.data_as(ctypes.c_void_p),
                          ctypes.c_int(nc),
                          ctypes.c_int(self.nocc),ctypes.c_int(self.nvir),
                          ctypes.c_double(numzero),ctypes.c_double(self.Ref[0])) 

    def get_t2t3c_omp(self, t2_t3t4c, tmp1, tmp2, tmp3, ci2cc, numzero, nc):
        _ccsd.libcc.t2t3c_shci_omp(t2_t3t4c.ctypes.data_as(ctypes.c_void_p),
                          ci2cc.t1.ctypes.data_as(ctypes.c_void_p),
                          ci2cc.t2aa.ctypes.data_as(ctypes.c_void_p),
                          ci2cc.t2ab.ctypes.data_as(ctypes.c_void_p),
                          tmp1.ctypes.data_as(ctypes.c_void_p),
                          tmp2.ctypes.data_as(ctypes.c_void_p),
                          tmp3.ctypes.data_as(ctypes.c_void_p),
                          ctypes.c_int(nc),
                          ctypes.c_int(self.nocc),ctypes.c_int(self.nvir),
                          ctypes.c_double(numzero),ctypes.c_double(self.Ref[0])) 

    def get_t2t3c_omp_mem(self, t2_t3t4c, tmp1, tmp2, tmp3, ci2cc, numzero=1e-9):
        _ccsd.libcc.t2t3c_shci_omp(t2_t3t4c.ctypes.data_as(ctypes.c_void_p),
                          ci2cc.t1.ctypes.data_as(ctypes.c_void_p),
                          ci2cc.t2aa.ctypes.data_as(ctypes.c_void_p),
                          ci2cc.t2ab.ctypes.data_as(ctypes.c_void_p),
                          tmp1.ctypes.data_as(ctypes.c_void_p),
                          tmp2.ctypes.data_as(ctypes.c_void_p),
                          tmp3.ctypes.data_as(ctypes.c_void_p),
                          ctypes.c_int(self.nocc_iact),
                          ctypes.c_int(self.nocc_corr),ctypes.c_int(self.nvir_corr),
                          ctypes.c_double(numzero),ctypes.c_double(self.Ref[0])) 


    def c4_div_omp(self, num_threads, numzero=0.0):
        df_tot = self.data 
        dfaabb = df_tot[df_tot['typ'] == 'aabb']
        dfaaab = df_tot[df_tot['typ'] == 'aaab']

        dfaabb['1'] = dfaabb['1'].map(lambda d: int(d))
        dfaabb['2'] = dfaabb['2'].map(lambda d: int(d))
        dfaabb['3'] = dfaabb['3'].map(lambda d: int(d))
        dfaabb['4'] = dfaabb['4'].map(lambda d: int(d))
        dfaabb['5'] = dfaabb['5'].map(lambda d: int(d))
        dfaabb['6'] = dfaabb['6'].map(lambda d: int(d))
        dfaabb['7'] = dfaabb['7'].map(lambda d: int(d))
        dfaabb['8'] = dfaabb['8'].map(lambda d: int(d))
        dfaabb['9'] = dfaabb['9'].map(lambda d: float(d))
        dfaaab['1'] = dfaaab['1'].map(lambda d: int(d))
        dfaaab['2'] = dfaaab['2'].map(lambda d: int(d))
        dfaaab['3'] = dfaaab['3'].map(lambda d: int(d))
        dfaaab['4'] = dfaaab['4'].map(lambda d: int(d))
        dfaaab['5'] = dfaaab['5'].map(lambda d: int(d))
        dfaaab['6'] = dfaaab['6'].map(lambda d: int(d))
        dfaaab['7'] = dfaaab['7'].map(lambda d: int(d))
        dfaaab['8'] = dfaaab['8'].map(lambda d: int(d))
        dfaaab['9'] = dfaaab['9'].map(lambda d: float(d))

        if self.skip_idx == None:
            dfaabb = dfaabb[abs(dfaabb['9']) > numzero]
            dfaaab = dfaaab[abs(dfaaab['9']) > numzero]
            df = pd.concat([dfaabb, dfaaab])
        else:
            list_df = []
            for idx in self.skip_idx:
                tmp1 = dfaabb['1'] == idx[0]
                tmp2 = dfaabb['2'] == idx[1]
                tmp3 = dfaabb['3'] == idx[2]
                tmp4 = dfaabb['4'] == idx[3]
                tmp5 = dfaabb['5'] == idx[4]
                tmp6 = dfaabb['6'] == idx[5]
                tmp7 = dfaabb['7'] == idx[6]
                tmp8 = dfaabb['8'] == idx[7]

                list_df.append( dfaabb[tmp1&tmp2&tmp3&tmp4&tmp5&tmp6&tmp7&tmp8] )
                tmp1 = dfaaab['1'] == idx[0]
                tmp2 = dfaaab['2'] == idx[1]
                tmp3 = dfaaab['3'] == idx[2]
                tmp4 = dfaaab['4'] == idx[3]
                tmp5 = dfaaab['5'] == idx[4]
                tmp6 = dfaaab['6'] == idx[5]
                tmp7 = dfaaab['7'] == idx[6]
                tmp8 = dfaaab['8'] == idx[7]

                list_df.append( dfaaab[tmp1&tmp2&tmp3&tmp4&tmp5&tmp6&tmp7&tmp8] )
            df = pd.concat(list_df)


        for j in range(8):
            js = str(j+1)
            df[js] = df[js].map(lambda d: int(d))
        
        tot_num = int(df['typ'].count())
        div_num = int(tot_num / num_threads) + 1
        
        #print('tot t4',tot_num)
        
        line_in    = [0 for i in range(num_threads)]
        line_limit = [0 for i in range(num_threads)]
        for i in range(num_threads):
            line_in[i] = div_num*i
            if i != num_threads-1: line_limit[i] = div_num*(i+1)
            else:                  line_limit[i] = tot_num
            #print(i, line_in[i], line_limit[i])
        
        df_div = [0 for i in range(num_threads)]
        for i in range(num_threads):
            df_div[i] = df[line_in[i]:line_limit[i]]
            #print(df_div[i]['typ'].count())
            result = 't4.%d'%(i)
            df_div[i].to_csv(result, index=False, header=False, float_format='%15.8f')

    def get_t2t4c_omp(self, t2t4c, e2ovov, ci2cc, numzero, nc, norm):
        _ccsd.libcc.t2t4c_shci_omp(t2t4c.ctypes.data_as(ctypes.c_void_p),
                          ci2cc.t1.ctypes.data_as(ctypes.c_void_p),
                          ci2cc.t2aa.ctypes.data_as(ctypes.c_void_p),
                          ci2cc.t2ab.ctypes.data_as(ctypes.c_void_p),
                          ci2cc.t3aaa.ctypes.data_as(ctypes.c_void_p),
                          ci2cc.t3aab.ctypes.data_as(ctypes.c_void_p),
                          e2ovov.ctypes.data_as(ctypes.c_void_p),
                          ctypes.c_int(nc),ctypes.c_int(self.num_det),
                          ctypes.c_int(self.nocc),ctypes.c_int(self.nvir),
                          ctypes.c_double(numzero),ctypes.c_double(self.Ref[0]),
                          ctypes.c_double(norm)) 

        print('=== END contracting Q t amplitudes ===')

    def get_t2t4c_omp_otf(self, t2t4c, e2ovov, ci2cc, numzero, nc, norm):
        _ccsd.libcc.t2t4c_shci_omp_otf(t2t4c.ctypes.data_as(ctypes.c_void_p),
                          ci2cc.t1.ctypes.data_as(ctypes.c_void_p),
                          ci2cc.t2aa.ctypes.data_as(ctypes.c_void_p),
                          ci2cc.t2ab.ctypes.data_as(ctypes.c_void_p),
                          self.T_aaa.ctypes.data_as(ctypes.c_void_p),
                          self.T_aab.ctypes.data_as(ctypes.c_void_p),
                          e2ovov.ctypes.data_as(ctypes.c_void_p),
                          ctypes.c_int(nc),ctypes.c_int(self.num_det),
                          ctypes.c_int(self.nocc),ctypes.c_int(self.nvir),
                          ctypes.c_double(numzero),ctypes.c_double(self.Ref[0]),
                          ctypes.c_double(norm)) 

    def get_t2t4c_omp_otf_mem(self, t2t4c, e2ovov, ci2cc, norm, numzero=1e-9):
        _ccsd.libcc.t2t4c_shci_omp_otf_mem(t2t4c.ctypes.data_as(ctypes.c_void_p),
                          ci2cc.t1.ctypes.data_as(ctypes.c_void_p),
                          ci2cc.t2aa.ctypes.data_as(ctypes.c_void_p),
                          ci2cc.t2ab.ctypes.data_as(ctypes.c_void_p),
                          self.T_aaa.ctypes.data_as(ctypes.c_void_p),
                          self.T_aab.ctypes.data_as(ctypes.c_void_p),
                          e2ovov.ctypes.data_as(ctypes.c_void_p),
                          ctypes.c_int(self.nocc_iact),
                          ctypes.c_int(self.nocc_corr),
                          ctypes.c_int(self.nvir_corr),
                          ctypes.c_int(self.nocc_cas),
                          ctypes.c_int(self.nvir_cas),
                          ctypes.c_double(numzero),ctypes.c_double(self.Ref[0]),
                          ctypes.c_double(norm)) 

    def get_t2t4c_otf_mem(self, t2t4c, e2ovov, ci2cc, numzero, norm):
        _ccsd.libcc.t2t4c_shci_otf_mem(t2t4c.ctypes.data_as(ctypes.c_void_p),
                          ci2cc.t1.ctypes.data_as(ctypes.c_void_p),
                          ci2cc.t2aa.ctypes.data_as(ctypes.c_void_p),
                          ci2cc.t2ab.ctypes.data_as(ctypes.c_void_p),
                          self.T_aaa.ctypes.data_as(ctypes.c_void_p),
                          self.T_aab.ctypes.data_as(ctypes.c_void_p),
                          e2ovov.ctypes.data_as(ctypes.c_void_p),
                          ctypes.c_int(self.nocc_iact),
                          ctypes.c_int(self.nocc_corr),
                          ctypes.c_int(self.nvir_corr),
                          ctypes.c_int(self.nocc_cas),
                          ctypes.c_int(self.nvir_cas),
                          ctypes.c_double(numzero),ctypes.c_double(self.Ref[0]),
                          ctypes.c_double(norm)) 

    def del_df(self):
        # release memory of previous dataframe
        import gc
        del [[self.data]]
        gc.collect()
        self.data=pd.DataFrame() 

    def get_t2t4c_ecT(self, t2t4c, e2ovov, ci2cc, numzero, nc):
        nacto_ref = self.nacto_ref
        nacte_ref = self.nacte_ref 
        nc_ref    = self.ncoreo_ref
        nocc_ref  = nacte_ref // 2
        nvir_ref  = nacto_ref - nocc_ref
        nc        = self.nc
        print ('nc_ref, nocc_ref, nvir_ref', nc_ref, nocc_ref, nvir_ref)

        norm = numpy.square(self.Ref[0])
        print ('        0 =', norm )
        norm0 = norm
        norm += 2.0*numpy.sum(numpy.square (self.S_a))
        print ('   0S (S) =', norm,'(', norm-norm0,')')
        norm0S = norm
        norm += 2.0*numpy.sum(numpy.square ( self.D_aa ))
        norm += numpy.sum(numpy.square ( self.D_ab ))
        print ('  0SD (D) =', norm,'(', norm-norm0S,')')
        norm0SD = norm
        norm += 2.0*numpy.sum(numpy.square ( self.T_aaa ))
        norm += 2.0*numpy.sum(numpy.square ( self.T_aab ))
        print (' 0SDT (T) =', norm,'(', norm-norm0SD,')')

        self.flagQ = True 
        self.interm_norm(Q=False)

        #TODO: put this assertion to cn_to_tn
        assert self.interm_norm_S and self.interm_norm_D and self.interm_norm_T
        ci2cc.c1_to_t1(self.S_a.copy())
        print('=== END extracting S t amplitudes ===')
        ci2cc.c2_to_t2(self.D_aa.copy(),self.D_ab.copy())
        print('=== END extracting D t amplitudes ===')
        #TODO: argument numzero as instance of ci2cc 
        #TODO: avoide gen t3 amplitudes 
        ci2cc.c3_to_t3_ecT(self.T_aaa.copy(), self.T_aab.copy(), nc_ref, nvir_ref, numzero=numzero)
        print('=== END extracting T t amplitudes ===')

#lsh test
#        # release memory of previous dataframe
#        import gc
#        del [[self.data]]
#        gc.collect()
#        self.data=pd.DataFrame() 

        _ccsd.libcc.t2t4c_shci_ecT(t2t4c.ctypes.data_as(ctypes.c_void_p),
                          ci2cc.t1.ctypes.data_as(ctypes.c_void_p),
                          ci2cc.t2aa.ctypes.data_as(ctypes.c_void_p),
                          ci2cc.t2ab.ctypes.data_as(ctypes.c_void_p),
                          ci2cc.t3aaa.ctypes.data_as(ctypes.c_void_p),
                          ci2cc.t3aab.ctypes.data_as(ctypes.c_void_p),
                          e2ovov.ctypes.data_as(ctypes.c_void_p),
                          ctypes.c_int(nc),
                          ctypes.c_int(nc_ref),ctypes.c_int(nvir_ref),
                          ctypes.c_int(self.nocc),ctypes.c_int(self.nvir),
                          ctypes.c_double(numzero),ctypes.c_double(self.Ref[0]),
                          ctypes.c_double(norm)) 
#                          tmp.ctypes.data_as(ctypes.c_void_p),

        for i in range(nocc_ref):
            for j in range(nocc_ref):
                for k in range(nocc_ref):
                    for a in range(nvir_ref):
                        for b in range(nvir_ref):
                            for c in range(nvir_ref):
                                ci2cc.t3aaa[i+nc_ref][j+nc_ref][k+nc_ref][a][b][c] = 0.0
                                ci2cc.t3aab[i+nc_ref][j+nc_ref][k+nc_ref][a][b][c] = 0.0


#        _ccsd.libcc.t2t4c_shci_omp(t2t4c.ctypes.data_as(ctypes.c_void_p),
#                          ci2cc.t1.ctypes.data_as(ctypes.c_void_p),
#                          ci2cc.t2aa.ctypes.data_as(ctypes.c_void_p),
#                          ci2cc.t2ab.ctypes.data_as(ctypes.c_void_p),
#                          ci2cc.t3aaa.ctypes.data_as(ctypes.c_void_p),
#                          ci2cc.t3aab.ctypes.data_as(ctypes.c_void_p),
#                          e2ovov.ctypes.data_as(ctypes.c_void_p),
#                          ctypes.c_int(nc),ctypes.c_int(self.num_det),
#                          ctypes.c_int(self.nocc),ctypes.c_int(self.nvir),
#                          ctypes.c_double(numzero),ctypes.c_double(self.Ref[0]),
#                          ctypes.c_double(norm)) 
##                          tmp.ctypes.data_as(ctypes.c_void_p),

        print('=== END contracting t4 amplitudes ===')

    def c3_div_omp(self, num_threads, numzero=0.0):

        df_tot = self.data 
        
        dfaab = df_tot[df_tot['typ'] == 'aab']
        dfaaa = df_tot[df_tot['typ'] == 'aaa']

        dfaab['7'] = dfaab['7'].map(lambda d: float(d))
        dfaaa['7'] = dfaaa['7'].map(lambda d: float(d))
        dfaab = dfaab[abs(dfaab['7']) > numzero]
        dfaaa = dfaaa[abs(dfaaa['7']) > numzero]

        df = pd.concat([dfaab, dfaaa])
        for j in range(6):
            js = str(j+1)
            df[js] = df[js].map(lambda d: int(d))

        #print(df)
        df = df.drop(['8', '9'], axis=1)
        #print(df)
        
        tot_num = int(df['typ'].count())
        div_num = int(tot_num / num_threads) + 1
        
        #print('tot t4',tot_num)
        
        line_in    = [0 for i in range(num_threads)]
        line_limit = [0 for i in range(num_threads)]
        for i in range(num_threads):
            line_in[i] = div_num*i
            if i != num_threads-1: line_limit[i] = div_num*(i+1)
            else:                  line_limit[i] = tot_num
            #print(i, line_in[i], line_limit[i])
        
        df_div = [0 for i in range(num_threads)]
        for i in range(num_threads):
            df_div[i] = df[line_in[i]:line_limit[i]]
            #print(df_div[i]['typ'].count())
            result = 't3.%d'%(i)
            df_div[i].to_csv(result, index=False, header=False, float_format='%15.8f')

    def get_t1t3c(self, t1t3c, e2ovov, ci2cc, numzero, nc):
        #norm0SD = norm
        norm = 0.0
        _ccsd.libcc.t1t3c_shci(t1t3c.ctypes.data_as(ctypes.c_void_p),
                          ci2cc.t1.ctypes.data_as(ctypes.c_void_p),
                          ci2cc.t2aa.ctypes.data_as(ctypes.c_void_p),
                          ci2cc.t2ab.ctypes.data_as(ctypes.c_void_p),
                          e2ovov.ctypes.data_as(ctypes.c_void_p),
                          ctypes.c_int(nc),
                          ctypes.c_int(self.nocc),ctypes.c_int(self.nvir),
                          ctypes.c_double(numzero),ctypes.c_double(self.Ref[0]),
                          ctypes.c_double(norm)) 

    def get_t1t3c_omp(self, t1t3c, e2ovov, ci2cc, numzero, nc):
        #norm0SD = norm
        norm = 0.0 
        _ccsd.libcc.t1t3c_shci_omp(t1t3c.ctypes.data_as(ctypes.c_void_p),
                          ci2cc.t1.ctypes.data_as(ctypes.c_void_p),
                          ci2cc.t2aa.ctypes.data_as(ctypes.c_void_p),
                          ci2cc.t2ab.ctypes.data_as(ctypes.c_void_p),
                          e2ovov.ctypes.data_as(ctypes.c_void_p),
                          ctypes.c_int(nc),
                          ctypes.c_int(self.nocc),ctypes.c_int(self.nvir),
                          ctypes.c_double(numzero),ctypes.c_double(self.Ref[0]),
                          ctypes.c_double(norm)) 

    def get_t1t3c_omp_mem(self, t1t3c, e2ovov, ci2cc, numzero=1e-9):
        #norm0SD = norm
        norm = 0.0 
        _ccsd.libcc.t1t3c_shci_omp(t1t3c.ctypes.data_as(ctypes.c_void_p),
                          ci2cc.t1.ctypes.data_as(ctypes.c_void_p),
                          ci2cc.t2aa.ctypes.data_as(ctypes.c_void_p),
                          ci2cc.t2ab.ctypes.data_as(ctypes.c_void_p),
                          e2ovov.ctypes.data_as(ctypes.c_void_p),
                          ctypes.c_int(self.nocc_iact),
                          ctypes.c_int(self.nocc_corr),ctypes.c_int(self.nvir_corr),
                          ctypes.c_double(numzero),ctypes.c_double(self.Ref[0]),
                          ctypes.c_double(norm)) 

    def denom_t3(self, denom, t1, t2aa, t2ab, numzero):
        _ccsd.libcc.denom_t3_shci(t1.ctypes.data_as(ctypes.c_void_p),
                          t2aa.ctypes.data_as(ctypes.c_void_p),
                          t2ab.ctypes.data_as(ctypes.c_void_p),
                          ctypes.c_int(self.nocc_iact),
                          ctypes.c_int(self.nocc_corr),ctypes.c_int(self.nvir_corr),
                          ctypes.c_double(numzero),ctypes.c_double(self.Ref[0]),
                          ctypes.c_double(denom)) 

    def denom_t4(self, denom, t1, t2aa, t2ab, numzero):
        self.get_paaa_paab() 
        _ccsd.libcc.denom_t4_shci(t1.ctypes.data_as(ctypes.c_void_p),
                          t2aa.ctypes.data_as(ctypes.c_void_p),
                          t2ab.ctypes.data_as(ctypes.c_void_p),
                          self.T_aaa.ctypes.data_as(ctypes.c_void_p),
                          self.T_aab.ctypes.data_as(ctypes.c_void_p),
                          self.Paaa.ctypes.data_as(ctypes.c_void_p),
                          self.Paab.ctypes.data_as(ctypes.c_void_p),
                          ctypes.c_int(self.nocc_iact),
                          ctypes.c_int(self.nocc_corr),ctypes.c_int(self.nvir_corr),
                          ctypes.c_int(self.nocc_cas),ctypes.c_int(self.nvir_cas),
                          ctypes.c_double(numzero),ctypes.c_double(self.Ref[0]),
                          ctypes.c_double(denom)) 


    def interm_norm(self, T=True, Q=True):   
        self.interm_norm_S = True 
        self.interm_norm_D = True 

        self.S_a    = self.S_a   / self.Ref[0] 
        self.D_aa   = self.D_aa  / self.Ref[0]
        self.D_ab   = self.D_ab  / self.Ref[0]
        if(T): 
           self.interm_norm_T = True 
           self.T_aaa  = self.T_aaa / self.Ref[0]
           self.T_aab  = self.T_aab / self.Ref[0]
        if(Q): 
           self.interm_norm_Q = True 
           self.Q_aaab = self.Q_aaab/ self.Ref[0]
           self.Q_aabb = self.Q_aabb/ self.Ref[0]

    def get_Pmat_ccsdt_slow(self):
        nc = self.nocc_iact
        nocc_cas = self.nocc_cas
        self.Paaa = numpy.full((self.nocc_corr,self.nocc_corr,self.nocc_corr,self.nvir_corr,self.nvir_corr,self.nvir_corr), 1.0)
        self.Pabb = numpy.full((self.nocc_corr,self.nocc_corr,self.nocc_corr,self.nvir_corr,self.nvir_corr,self.nvir_corr), 1.0)
        ncount_aaa = 0
        ncount_abb = 0

        def asgn_zero_t6(t, i, j, k, a, b, c):
            t[i][j][k][a][b][c] = 0.0
            t[i][k][j][a][b][c] = 0.0
            t[j][i][k][a][b][c] = 0.0
            t[j][k][i][a][b][c] = 0.0
            t[k][i][j][a][b][c] = 0.0
            t[k][j][i][a][b][c] = 0.0

            t[i][j][k][a][c][b] = 0.0
            t[i][k][j][a][c][b] = 0.0
            t[j][i][k][a][c][b] = 0.0
            t[j][k][i][a][c][b] = 0.0
            t[k][i][j][a][c][b] = 0.0
            t[k][j][i][a][c][b] = 0.0

            t[i][j][k][b][a][c] = 0.0
            t[i][k][j][b][a][c] = 0.0
            t[j][i][k][b][a][c] = 0.0
            t[j][k][i][b][a][c] = 0.0
            t[k][i][j][b][a][c] = 0.0
            t[k][j][i][b][a][c] = 0.0

            t[i][j][k][b][c][a] = 0.0
            t[i][k][j][b][c][a] = 0.0
            t[j][i][k][b][c][a] = 0.0
            t[j][k][i][b][c][a] = 0.0
            t[k][i][j][b][c][a] = 0.0
            t[k][j][i][b][c][a] = 0.0

            t[i][j][k][c][a][b] = 0.0
            t[i][k][j][c][a][b] = 0.0
            t[j][i][k][c][a][b] = 0.0
            t[j][k][i][c][a][b] = 0.0
            t[k][i][j][c][a][b] = 0.0
            t[k][j][i][c][a][b] = 0.0

            t[i][j][k][c][b][a] = 0.0
            t[i][k][j][c][b][a] = 0.0
            t[j][i][k][c][b][a] = 0.0
            t[j][k][i][c][b][a] = 0.0
            t[k][i][j][c][b][a] = 0.0
            t[k][j][i][c][b][a] = 0.0

        def asgn_zero_t1_2(t, i, j, k, a, b, c):
            t[i][j][k][a][b][c] = 0.0
            t[i][k][j][a][b][c] = 0.0

            t[i][j][k][a][c][b] = 0.0
            t[i][k][j][a][c][b] = 0.0

        for idet in range(self.num_det):
            typ = self.typ_det[idet]

            if (typ == "aaa"):
               if abs(float(self.data.loc[idet,"7"])) > self.numzero:
                   i = int(self.data.loc[idet,"1"]) + nc
                   j = int(self.data.loc[idet,"2"]) + nc
                   k = int(self.data.loc[idet,"3"]) + nc
                   a = int(self.data.loc[idet,"4"]) - nocc_cas
                   b = int(self.data.loc[idet,"5"]) - nocc_cas
                   c = int(self.data.loc[idet,"6"]) - nocc_cas
    
                   asgn_zero_t6(self.Paaa, i, j, k, a, b, c)
                   ncount_aaa += 1

            elif (typ == "abb"):
               if abs(float(self.data.loc[idet,"7"])) > self.numzero:
                   i = int(self.data.loc[idet,"1"]) + nc
                   a = int(self.data.loc[idet,"2"]) - nocc_cas
                   j = int(self.data.loc[idet,"3"]) + nc
                   k = int(self.data.loc[idet,"4"]) + nc
                   b = int(self.data.loc[idet,"5"]) - nocc_cas
                   c = int(self.data.loc[idet,"6"]) - nocc_cas
    
                   asgn_zero_t1_2(self.Pabb, i, j, k, a, b, c)
                   ncount_abb += 1

        print ('n_aaa, n_abb=', ncount_aaa, ncount_abb)

#    def exclude_taaa(self, w):
#        nc = self.nocc_iact
#        nocc_cas = self.nocc_cas
#        ncount_aaa = 0
#        ncount_abb = 0
#
#        def asgn_zero_t6(t, i, j, k, a, b, c):
#            t[i][j][k][a][b][c] = 0.0
#            t[i][k][j][a][b][c] = 0.0
#            t[j][i][k][a][b][c] = 0.0
#            t[j][k][i][a][b][c] = 0.0
#            t[k][i][j][a][b][c] = 0.0
#            t[k][j][i][a][b][c] = 0.0
#
#            t[i][j][k][a][c][b] = 0.0
#            t[i][k][j][a][c][b] = 0.0
#            t[j][i][k][a][c][b] = 0.0
#            t[j][k][i][a][c][b] = 0.0
#            t[k][i][j][a][c][b] = 0.0
#            t[k][j][i][a][c][b] = 0.0
#
#            t[i][j][k][b][a][c] = 0.0
#            t[i][k][j][b][a][c] = 0.0
#            t[j][i][k][b][a][c] = 0.0
#            t[j][k][i][b][a][c] = 0.0
#            t[k][i][j][b][a][c] = 0.0
#            t[k][j][i][b][a][c] = 0.0
#
#            t[i][j][k][b][c][a] = 0.0
#            t[i][k][j][b][c][a] = 0.0
#            t[j][i][k][b][c][a] = 0.0
#            t[j][k][i][b][c][a] = 0.0
#            t[k][i][j][b][c][a] = 0.0
#            t[k][j][i][b][c][a] = 0.0
#
#            t[i][j][k][c][a][b] = 0.0
#            t[i][k][j][c][a][b] = 0.0
#            t[j][i][k][c][a][b] = 0.0
#            t[j][k][i][c][a][b] = 0.0
#            t[k][i][j][c][a][b] = 0.0
#            t[k][j][i][c][a][b] = 0.0
#
#            t[i][j][k][c][b][a] = 0.0
#            t[i][k][j][c][b][a] = 0.0
#            t[j][i][k][c][b][a] = 0.0
#            t[j][k][i][c][b][a] = 0.0
#            t[k][i][j][c][b][a] = 0.0
#            t[k][j][i][c][b][a] = 0.0
#
#        def asgn_zero_t1_2(t, i, j, k, a, b, c):
#            t[i][j][k][a][b][c] = 0.0
#            t[i][k][j][a][b][c] = 0.0
#
#            t[i][j][k][a][c][b] = 0.0
#            t[i][k][j][a][c][b] = 0.0
#
#        for idet in range(self.num_det):
#            typ = self.typ_det[idet]
#
#            if (typ == "aaa"):
#               i = int(self.data.loc[idet,"1"]) + nc
#               j = int(self.data.loc[idet,"2"]) + nc
#               k = int(self.data.loc[idet,"3"]) + nc
#               a = int(self.data.loc[idet,"4"]) - nocc_cas
#               b = int(self.data.loc[idet,"5"]) - nocc_cas
#               c = int(self.data.loc[idet,"6"]) - nocc_cas
#
#               asgn_zero_t6(w, i, j, k, a, b, c)
#               ncount_aaa += 1
#
##            elif (typ == "abb"):
##               i = int(self.data.loc[idet,"1"]) + nc
##               a = int(self.data.loc[idet,"2"]) - nocc_cas
##               j = int(self.data.loc[idet,"3"]) + nc
##               k = int(self.data.loc[idet,"4"]) + nc
##               b = int(self.data.loc[idet,"5"]) - nocc_cas
##               c = int(self.data.loc[idet,"6"]) - nocc_cas
##
##               asgn_zero_t1_2(self.Pabb, i, j, k, a, b, c)
##               ncount_abb += 1

    def exclude_taaa(self, w):
        nc = self.nocc_iact
        nocc_cas = self.nocc_cas
        def asgn_zero_t6(t, i, j, k, a, b, c):
            t[i][j][k][a][b][c] = 0.0
            t[i][k][j][a][b][c] = 0.0
            t[j][i][k][a][b][c] = 0.0
            t[j][k][i][a][b][c] = 0.0
            t[k][i][j][a][b][c] = 0.0
            t[k][j][i][a][b][c] = 0.0

            t[i][j][k][a][c][b] = 0.0
            t[i][k][j][a][c][b] = 0.0
            t[j][i][k][a][c][b] = 0.0
            t[j][k][i][a][c][b] = 0.0
            t[k][i][j][a][c][b] = 0.0
            t[k][j][i][a][c][b] = 0.0

            t[i][j][k][b][a][c] = 0.0
            t[i][k][j][b][a][c] = 0.0
            t[j][i][k][b][a][c] = 0.0
            t[j][k][i][b][a][c] = 0.0
            t[k][i][j][b][a][c] = 0.0
            t[k][j][i][b][a][c] = 0.0

            t[i][j][k][b][c][a] = 0.0
            t[i][k][j][b][c][a] = 0.0
            t[j][i][k][b][c][a] = 0.0
            t[j][k][i][b][c][a] = 0.0
            t[k][i][j][b][c][a] = 0.0
            t[k][j][i][b][c][a] = 0.0

            t[i][j][k][c][a][b] = 0.0
            t[i][k][j][c][a][b] = 0.0
            t[j][i][k][c][a][b] = 0.0
            t[j][k][i][c][a][b] = 0.0
            t[k][i][j][c][a][b] = 0.0
            t[k][j][i][c][a][b] = 0.0

            t[i][j][k][c][b][a] = 0.0
            t[i][k][j][c][b][a] = 0.0
            t[j][i][k][c][b][a] = 0.0
            t[j][k][i][c][b][a] = 0.0
            t[k][i][j][c][b][a] = 0.0
            t[k][j][i][c][b][a] = 0.0

        for idet in range(self.num_det):
            typ = self.typ_det[idet]
            if (typ == "aaa"):
               if abs(float(self.data.loc[idet,"7"])) > self.numzero:
                   i = int(self.data.loc[idet,"1"]) + nc
                   j = int(self.data.loc[idet,"2"]) + nc
                   k = int(self.data.loc[idet,"3"]) + nc
                   a = int(self.data.loc[idet,"4"]) - nocc_cas
                   b = int(self.data.loc[idet,"5"]) - nocc_cas
                   c = int(self.data.loc[idet,"6"]) - nocc_cas
                   asgn_zero_t6(w, i, j, k, a, b, c)

    def exclude_tbaa(self, w):
        nc = self.nocc_iact
        nocc_cas = self.nocc_cas
        def asgn_zero_t1_2(t, i, j, k, a, b, c):
            t[i][j][k][a][b][c] = 0.0
            t[i][k][j][a][b][c] = 0.0
            t[i][j][k][a][c][b] = 0.0
            t[i][k][j][a][c][b] = 0.0

        for idet in range(self.num_det):
            typ = self.typ_det[idet]
            if (typ == "abb"):
               if abs(float(self.data.loc[idet,"7"])) > self.numzero:
                   i = int(self.data.loc[idet,"1"]) + nc
                   a = int(self.data.loc[idet,"2"]) - nocc_cas
                   j = int(self.data.loc[idet,"3"]) + nc
                   k = int(self.data.loc[idet,"4"]) + nc
                   b = int(self.data.loc[idet,"5"]) - nocc_cas
                   c = int(self.data.loc[idet,"6"]) - nocc_cas
                   asgn_zero_t1_2(w, i, j, k, a, b, c)

    def get_paaa_paab(self):
        self.Paaa = numpy.full((self.nocc_cas,self.nocc_cas,self.nocc_cas,self.nvir_cas,self.nvir_cas,self.nvir_cas), 1.0)
        self.Paab = numpy.full((self.nocc_cas,self.nocc_cas,self.nocc_cas,self.nvir_cas,self.nvir_cas,self.nvir_cas), 1.0)
        ncount_aaa = 0
        ncount_aab = 0

        def asgn_zero_t6(t, i, j, k, a, b, c):
            t[i][j][k][a][b][c] = 0.0
            t[i][k][j][a][b][c] = 0.0
            t[j][i][k][a][b][c] = 0.0
            t[j][k][i][a][b][c] = 0.0
            t[k][i][j][a][b][c] = 0.0
            t[k][j][i][a][b][c] = 0.0

            t[i][j][k][a][c][b] = 0.0
            t[i][k][j][a][c][b] = 0.0
            t[j][i][k][a][c][b] = 0.0
            t[j][k][i][a][c][b] = 0.0
            t[k][i][j][a][c][b] = 0.0
            t[k][j][i][a][c][b] = 0.0

            t[i][j][k][b][a][c] = 0.0
            t[i][k][j][b][a][c] = 0.0
            t[j][i][k][b][a][c] = 0.0
            t[j][k][i][b][a][c] = 0.0
            t[k][i][j][b][a][c] = 0.0
            t[k][j][i][b][a][c] = 0.0

            t[i][j][k][b][c][a] = 0.0
            t[i][k][j][b][c][a] = 0.0
            t[j][i][k][b][c][a] = 0.0
            t[j][k][i][b][c][a] = 0.0
            t[k][i][j][b][c][a] = 0.0
            t[k][j][i][b][c][a] = 0.0

            t[i][j][k][c][a][b] = 0.0
            t[i][k][j][c][a][b] = 0.0
            t[j][i][k][c][a][b] = 0.0
            t[j][k][i][c][a][b] = 0.0
            t[k][i][j][c][a][b] = 0.0
            t[k][j][i][c][a][b] = 0.0

            t[i][j][k][c][b][a] = 0.0
            t[i][k][j][c][b][a] = 0.0
            t[j][i][k][c][b][a] = 0.0
            t[j][k][i][c][b][a] = 0.0
            t[k][i][j][c][b][a] = 0.0
            t[k][j][i][c][b][a] = 0.0

        def asgn_zero_t2_1(t, i, j, k, a, b, c):
            t[i][j][k][a][b][c] = 0.0
            t[j][i][k][a][b][c] = 0.0
            t[i][j][k][b][a][c] = 0.0
            t[j][i][k][b][a][c] = 0.0

        for idet in range(self.num_det):
            typ = self.typ_det[idet]
            if (typ == "aaa"):
               if abs(float(self.data.loc[idet,"7"])) > self.numzero:
                   i = int(self.data.loc[idet,"1"])
                   j = int(self.data.loc[idet,"2"])
                   k = int(self.data.loc[idet,"3"])
                   a = int(self.data.loc[idet,"4"]) - (self.nocc_cas)
                   b = int(self.data.loc[idet,"5"]) - (self.nocc_cas)
                   c = int(self.data.loc[idet,"6"]) - (self.nocc_cas)
    
                   asgn_zero_t6(self.Paaa, i, j, k, a, b, c)
                   ncount_aaa += 1

            elif (typ == "aab"):
               if abs(float(self.data.loc[idet,"7"])) > self.numzero:
                   i = int(self.data.loc[idet,"1"]) 
                   j = int(self.data.loc[idet,"2"]) 
                   a = int(self.data.loc[idet,"3"]) - (self.nocc_cas) 
                   b = int(self.data.loc[idet,"4"]) - (self.nocc_cas) 
                   k = int(self.data.loc[idet,"5"]) 
                   c = int(self.data.loc[idet,"6"]) - (self.nocc_cas) 
    
                   asgn_zero_t2_1(self.Paab, i, j, k, a, b, c)
                   ncount_aab += 1

        print ('n_aaa, n_aab=', ncount_aaa, ncount_aab)

    def get_Pmat_ccsdt_cas(self):
        self.Paaa = numpy.full((self.nocc_cas,self.nocc_cas,self.nocc_cas,self.nvir_cas,self.nvir_cas,self.nvir_cas), 1.0)
        self.Pabb = numpy.full((self.nocc_cas,self.nocc_cas,self.nocc_cas,self.nvir_cas,self.nvir_cas,self.nvir_cas), 1.0)
        ncount_aaa = 0
        ncount_abb = 0

        def asgn_zero_t6(t, i, j, k, a, b, c):
            t[i][j][k][a][b][c] = 0.0
            t[i][k][j][a][b][c] = 0.0
            t[j][i][k][a][b][c] = 0.0
            t[j][k][i][a][b][c] = 0.0
            t[k][i][j][a][b][c] = 0.0
            t[k][j][i][a][b][c] = 0.0

            t[i][j][k][a][c][b] = 0.0
            t[i][k][j][a][c][b] = 0.0
            t[j][i][k][a][c][b] = 0.0
            t[j][k][i][a][c][b] = 0.0
            t[k][i][j][a][c][b] = 0.0
            t[k][j][i][a][c][b] = 0.0

            t[i][j][k][b][a][c] = 0.0
            t[i][k][j][b][a][c] = 0.0
            t[j][i][k][b][a][c] = 0.0
            t[j][k][i][b][a][c] = 0.0
            t[k][i][j][b][a][c] = 0.0
            t[k][j][i][b][a][c] = 0.0

            t[i][j][k][b][c][a] = 0.0
            t[i][k][j][b][c][a] = 0.0
            t[j][i][k][b][c][a] = 0.0
            t[j][k][i][b][c][a] = 0.0
            t[k][i][j][b][c][a] = 0.0
            t[k][j][i][b][c][a] = 0.0

            t[i][j][k][c][a][b] = 0.0
            t[i][k][j][c][a][b] = 0.0
            t[j][i][k][c][a][b] = 0.0
            t[j][k][i][c][a][b] = 0.0
            t[k][i][j][c][a][b] = 0.0
            t[k][j][i][c][a][b] = 0.0

            t[i][j][k][c][b][a] = 0.0
            t[i][k][j][c][b][a] = 0.0
            t[j][i][k][c][b][a] = 0.0
            t[j][k][i][c][b][a] = 0.0
            t[k][i][j][c][b][a] = 0.0
            t[k][j][i][c][b][a] = 0.0

        def asgn_zero_t1_2(t, i, j, k, a, b, c):
            t[i][j][k][a][b][c] = 0.0
            t[i][k][j][a][b][c] = 0.0

            t[i][j][k][a][c][b] = 0.0
            t[i][k][j][a][c][b] = 0.0

        for idet in range(self.num_det):
            typ = self.typ_det[idet]

            if (typ == "aaa"):
               if abs(float(self.data.loc[idet,"7"])) > self.numzero:
                   i = int(self.data.loc[idet,"1"])
                   j = int(self.data.loc[idet,"2"])
                   k = int(self.data.loc[idet,"3"])
                   a = int(self.data.loc[idet,"4"]) - (self.nocc_cas)
                   b = int(self.data.loc[idet,"5"]) - (self.nocc_cas)
                   c = int(self.data.loc[idet,"6"]) - (self.nocc_cas)
    
                   asgn_zero_t6(self.Paaa, i, j, k, a, b, c)
                   ncount_aaa += 1

            elif (typ == "abb"):
               if abs(float(self.data.loc[idet,"7"])) > self.numzero:
                   i = int(self.data.loc[idet,"1"]) 
                   a = int(self.data.loc[idet,"2"]) - (self.nocc_cas) 
                   j = int(self.data.loc[idet,"3"]) 
                   k = int(self.data.loc[idet,"4"]) 
                   b = int(self.data.loc[idet,"5"]) - (self.nocc_cas) 
                   c = int(self.data.loc[idet,"6"]) - (self.nocc_cas) 
    
                   asgn_zero_t1_2(self.Pabb, i, j, k, a, b, c)
                   ncount_abb += 1

        print ('n_aaa, n_abb=', ncount_aaa, ncount_abb)

    def exclude_t_ecCCSDt(self):
        nocc = self.nocc_cas
        nvir = self.nvir_cas
        nocc2 = nocc*(nocc-1)/2
        nocc3 = nocc*(nocc-1)*(nocc-2)/6 

        dS = int(nocc * nvir) 
        dD = int(nocc*(nocc-1)*nvir*(nvir-1)/4) 
        dT = int(nocc*(nocc-1)*(nocc-2)*nvir*(nvir-1)*(nvir-2)/36) 

        self.Paaa = numpy.full((dT), 1.0)
        self.Pbaa = numpy.full((dS,dD), 1.0)
        ncount_aaa = 0
        ncount_baa = 0

        # idx for i<j<k, a<b<c
        def S(i, a):
            return int(nocc * ( a+1 ) - ( i+1 ))
        def D(i, j, a, b):
            return int(nocc2 * ( b*(b-1)/2 + a+1 ) - ( j*(j-1)/2 + i+1 )) 
        def T(i, j, k, a, b, c):
            return int(nocc3 * ( c*(c-1)*(c-2)/6 + b*(b-1)/2 + a+1 ) \
                       - ( k*(k-1)*(k-2)/6 + j*(j-1)/2 + i+1 ))

        for idet in range(self.num_det):
            typ = self.typ_det[idet]
            if (typ == "aaa"):
               if abs(float(self.data.loc[idet,"7"])) > self.numzero:
                   i = int(self.data.loc[idet,"1"])
                   j = int(self.data.loc[idet,"2"])
                   k = int(self.data.loc[idet,"3"])
                   a = int(self.data.loc[idet,"4"]) - nocc
                   b = int(self.data.loc[idet,"5"]) - nocc
                   c = int(self.data.loc[idet,"6"]) - nocc
                   self.Paaa[T(i, j, k, a, b, c)] = 0.0 
                   ncount_aaa += 1
            elif (typ == "abb"):
               if abs(float(self.data.loc[idet,"7"])) > self.numzero:
                   i = int(self.data.loc[idet,"1"]) 
                   a = int(self.data.loc[idet,"2"]) - nocc
                   j = int(self.data.loc[idet,"3"])
                   k = int(self.data.loc[idet,"4"])
                   b = int(self.data.loc[idet,"5"]) - nocc
                   c = int(self.data.loc[idet,"6"]) - nocc
                   self.Pbaa[S(i,a)][D(j,k,b,c)] = 0.0
                   ncount_baa += 1

        print ('n_aaa, n_baa=', ncount_aaa, ncount_baa)

    def get_Pmat_ectccsdt(self):
        nacto_ref = self.nacto_ref
        nacte_ref = self.nacte_ref 
        nc_ref    = self.ncoreo_ref
        nocc_ref  = nacte_ref // 2
        nvir_ref  = nacto_ref - nocc_ref 
        nc        = self.nc

        self.Paaa = numpy.full((self.nocc,self.nocc,self.nocc,self.nvir,self.nvir,self.nvir), 1.0)
#        self.Paab = numpy.full((self.nocc,self.nocc,self.nocc,self.nvir,self.nvir,self.nvir), 1.0)
        self.Pabb = numpy.full((self.nocc,self.nocc,self.nocc,self.nvir,self.nvir,self.nvir), 1.0)
#        self.Pbbb = numpy.full((self.nocc,self.nocc,self.nocc,self.nvir,self.nvir,self.nvir), 1.0)
        ncount_aaa = 0
#        ncount_aab = 0
        ncount_abb = 0
#        ncount_bbb = 0

        def asgn_zero_t6(t, i, j, k, a, b, c):
            t[i][j][k][a][b][c] = 0.0
            t[i][k][j][a][b][c] = 0.0
            t[j][i][k][a][b][c] = 0.0
            t[j][k][i][a][b][c] = 0.0
            t[k][i][j][a][b][c] = 0.0
            t[k][j][i][a][b][c] = 0.0

            t[i][j][k][a][c][b] = 0.0
            t[i][k][j][a][c][b] = 0.0
            t[j][i][k][a][c][b] = 0.0
            t[j][k][i][a][c][b] = 0.0
            t[k][i][j][a][c][b] = 0.0
            t[k][j][i][a][c][b] = 0.0

            t[i][j][k][b][a][c] = 0.0
            t[i][k][j][b][a][c] = 0.0
            t[j][i][k][b][a][c] = 0.0
            t[j][k][i][b][a][c] = 0.0
            t[k][i][j][b][a][c] = 0.0
            t[k][j][i][b][a][c] = 0.0

            t[i][j][k][b][c][a] = 0.0
            t[i][k][j][b][c][a] = 0.0
            t[j][i][k][b][c][a] = 0.0
            t[j][k][i][b][c][a] = 0.0
            t[k][i][j][b][c][a] = 0.0
            t[k][j][i][b][c][a] = 0.0

            t[i][j][k][c][a][b] = 0.0
            t[i][k][j][c][a][b] = 0.0
            t[j][i][k][c][a][b] = 0.0
            t[j][k][i][c][a][b] = 0.0
            t[k][i][j][c][a][b] = 0.0
            t[k][j][i][c][a][b] = 0.0

            t[i][j][k][c][b][a] = 0.0
            t[i][k][j][c][b][a] = 0.0
            t[j][i][k][c][b][a] = 0.0
            t[j][k][i][c][b][a] = 0.0
            t[k][i][j][c][b][a] = 0.0
            t[k][j][i][c][b][a] = 0.0

        def asgn_zero_t2_1(t, i, j, k, a, b, c):
            t[i][j][k][a][b][c] = 0.0
            t[j][i][k][a][b][c] = 0.0

            t[i][j][k][b][a][c] = 0.0
            t[j][i][k][b][a][c] = 0.0

        def asgn_zero_t1_2(t, i, j, k, a, b, c):
            t[i][j][k][a][b][c] = 0.0
            t[i][k][j][a][b][c] = 0.0

            t[i][j][k][a][c][b] = 0.0
            t[i][k][j][a][c][b] = 0.0

        for idet in range(self.num_det):
            typ = self.typ_det[idet]

            if (typ == "aaa"):
               i = int(self.data.loc[idet,"1"]) + nc
               j = int(self.data.loc[idet,"2"]) + nc
               k = int(self.data.loc[idet,"3"]) + nc
               a = int(self.data.loc[idet,"4"]) - (self.nocc) + nc
               b = int(self.data.loc[idet,"5"]) - (self.nocc) + nc
               c = int(self.data.loc[idet,"6"]) - (self.nocc) + nc

               asgn_zero_t6(self.Paaa, i, j, k, a, b, c)
               ncount_aaa += 1

#            elif (typ == "aab"):
#               i = int(self.data.loc[idet,"1"]) + nc
#               j = int(self.data.loc[idet,"2"]) + nc
#               a = int(self.data.loc[idet,"3"]) - (self.nocc) + nc
#               b = int(self.data.loc[idet,"4"]) - (self.nocc) + nc
#               k = int(self.data.loc[idet,"5"]) + nc
#               c = int(self.data.loc[idet,"6"]) - (self.nocc) + nc
#
#               asgn_zero_t2_1(self.Paab, i, j, k, a, b, c)
#               ncount_aab += 1

            elif (typ == "abb"):
               i = int(self.data.loc[idet,"1"]) + nc
               a = int(self.data.loc[idet,"2"]) - (self.nocc) + nc
               j = int(self.data.loc[idet,"3"]) + nc
               k = int(self.data.loc[idet,"4"]) + nc
               b = int(self.data.loc[idet,"5"]) - (self.nocc) + nc
               c = int(self.data.loc[idet,"6"]) - (self.nocc) + nc

               asgn_zero_t1_2(self.Pabb, i, j, k, a, b, c)
               ncount_abb += 1

#            elif (typ == "bbb"):
#               i = int(self.data.loc[idet,"1"]) + nc
#               j = int(self.data.loc[idet,"2"]) + nc
#               k = int(self.data.loc[idet,"3"]) + nc
#               a = int(self.data.loc[idet,"4"]) - (self.nocc) + nc
#               b = int(self.data.loc[idet,"5"]) - (self.nocc) + nc
#               c = int(self.data.loc[idet,"6"]) - (self.nocc) + nc
#
#               asgn_zero_t6(self.Pbbb, i, j, k, a, b, c)
#               ncount_bbb += 1

        print ('n_aaa, n_abb=', ncount_aaa, ncount_abb)
        #print ('n_aaa, naab, nabb, nbbb =', ncount_aaa, ncount_aab, ncount_abb, ncount_bbb)

        for i in range(nocc_ref):
            for j in range(nocc_ref):
                for k in range(nocc_ref):
                    for a in range(nvir_ref):
                        for b in range(nvir_ref):
                            for c in range(nvir_ref):
                                self.Paaa[i+nc_ref][j+nc_ref][k+nc_ref][a][b][c] = 0.0
                                self.Pabb[i+nc_ref][j+nc_ref][k+nc_ref][a][b][c] = 0.0

    def tcc_tcas_idx(self):
        t1f= numpy.zeros((self.nocc_corr,self.nvir_corr), dtype=numpy.float64)
        t2f= numpy.zeros((self.nocc_corr,self.nocc_corr,self.nvir_corr,self.nvir_corr), dtype=numpy.float64)
        n_a  = 0
        n_ab = 0

        for idet in range(self.num_det):
            typ = self.typ_det[idet]
            if typ == "a":
                i = int(self.data.loc[idet,"1"]) + self.nocc_iact
                a = int(self.data.loc[idet,"2"]) - self.nocc_cas
                t1f[i][a] = 1.0
                n_a += 1 
            elif typ == "ab":
                i = int(self.data.loc[idet,"1"]) + self.nocc_iact
                a = int(self.data.loc[idet,"2"]) - self.nocc_cas
                j = int(self.data.loc[idet,"3"]) + self.nocc_iact
                b = int(self.data.loc[idet,"4"]) - self.nocc_cas
                t2f[i][j][a][b] = 1.0
                n_ab += 1 

        return t1f, t2f

    def ectcc_tcas_idx(self):
        nacto_ref = self.nacto_ref
        nacte_ref = self.nacte_ref 
        nc_ref    = self.ncoreo_ref
        nocc_ref  = nacte_ref // 2
        nvir_ref  = nacto_ref - nocc_ref 
        nc        = self.nc
        print ('nc_ref, nocc_ref, nvir_ref', nc_ref, nocc_ref, nvir_ref)
        t1f= numpy.zeros((self.nocc,self.nvir), dtype=numpy.float64)
        t2f= numpy.zeros((self.nocc,self.nocc,self.nvir,self.nvir), dtype=numpy.float64)
        n_a  = 0
        n_ab = 0
        for idet in range(self.num_det):
            typ = self.typ_det[idet]
            if typ == "a":
                i = int(self.data.loc[idet,"1"]) + nc
                a = int(self.data.loc[idet,"2"]) - (self.nocc) + nc
                if i >= nc_ref and a < nvir_ref:
                    t1f[i][a] = 1.0
                    #print ('i, a:', i, a)
                n_a += 1 
            elif typ == "ab":
                i = int(self.data.loc[idet,"1"]) + nc
                a = int(self.data.loc[idet,"2"]) - (self.nocc) + nc
                j = int(self.data.loc[idet,"3"]) + nc
                b = int(self.data.loc[idet,"4"]) - (self.nocc) + nc
                if i >= nc_ref and j >= nc_ref and a < nvir_ref and b < nvir_ref:
                    t2f[i][j][a][b] = 1.0
                    #print ('i, j, a, b:', i, j, a, b)
                n_ab += 1 

        print ('CIcoeff   a and ab =',self.typ_det.count('a'),self.typ_det.count('ab'))
        print ('number of a and ab =', n_a, n_ab)
        return t1f, t2f

    def align(self):   
        signref = 1
        if self.Ref[0] < 0: signref = -1
        self.S_a    = self.S_a   * signref 
        self.D_aa   = self.D_aa  * signref 
        self.D_ab   = self.D_ab  * signref 
        self.T_aaa  = self.T_aaa * signref 
        self.T_aab  = self.T_aab * signref 
        self.Q_aaab = self.Q_aaab * signref 
        self.Q_aabb = self.Q_aabb * signref 

    def top_weighted_Fermi_vacuum(self):
        self.S_a    = numpy.zeros((dS), dtype=numpy.float64)
        self.S_b    = numpy.zeros((dS), dtype=numpy.float64)
        self.D_aa   = numpy.zeros((dD), dtype=numpy.float64)
        self.D_ab   = numpy.zeros((dS,dS), dtype=numpy.float64)
        self.D_bb   = numpy.zeros((dD), dtype=numpy.float64)
        self.T_aaa  = numpy.zeros((dT), dtype=numpy.float64)
        self.T_aab  = numpy.zeros((dD,dS), dtype=numpy.float64)
        self.T_abb  = numpy.zeros((dS,dD), dtype=numpy.float64)
        self.T_bbb  = numpy.zeros((dT), dtype=numpy.float64)

    def get_All_test(self, nc):
        self.nc = nc
        self.flagS = True 
        self.flagD = True
        self.flagT = True
        self.flagQ = True

        dS = int(self.nocc * self.nvir) 
        dD = int(self.nocc*(self.nocc-1)*self.nvir*(self.nvir-1)/4) 
        dT = int(self.nocc*(self.nocc-1)*(self.nocc-2)*self.nvir*(self.nvir-1)*(self.nvir-2)/36) 
#        dQ = len(self.idx.idx4.keys())
        print('dS, dD, dT =',dS,dD,dT)

        nex = self.nocc * self.nvir
        self.Ref    = numpy.zeros((1), dtype=numpy.float64)

        self.S_a    = numpy.zeros((dS), dtype=numpy.float64)
        self.S_b    = numpy.zeros((dS), dtype=numpy.float64)
        self.D_aa   = numpy.zeros((dD), dtype=numpy.float64)
        self.D_ab   = numpy.zeros((dS,dS), dtype=numpy.float64)
        self.D_bb   = numpy.zeros((dD), dtype=numpy.float64)
        self.T_aaa  = numpy.zeros((dT), dtype=numpy.float64)
        self.T_aab  = numpy.zeros((dD,dS), dtype=numpy.float64)
        self.T_abb  = numpy.zeros((dS,dD), dtype=numpy.float64)
        self.T_bbb  = numpy.zeros((dT), dtype=numpy.float64)
#        self.Q_aaaa = numpy.zeros((dQ), dtype=numpy.float64)
        self.Q_aaab = numpy.zeros((dT,dS), dtype=numpy.float64)
        self.Q_aabb = numpy.zeros((dD,dD), dtype=numpy.float64)
#        self.Q_abbb = numpy.zeros((dS,dT), dtype=numpy.float64)
#        self.Q_bbbb = numpy.zeros((dQ), dtype=numpy.float64)

        test_max = 0.0
        test_max_idx =[] 
        for idet in range(self.num_det):
            typ = self.typ_det[idet]
            det_str = self.HFdet_str.copy()
            parity  = 1
            if (typ == "shci"):
               self.E_shci = self.data.loc[idet,"1"] 
            elif (typ == "rf"):
               self.Ref[0] = self.data.loc[idet,"1"] 
               if abs(self.data.loc[idet,"1"]) > self.top_w:
                   self.top_w     = self.data.loc[idet,"1"]
                   self.top_w_idx = [0] 
                   self.top_w_typ = typ 
            elif (typ == "a"):
               i = int(self.data.loc[idet,"1"]) + nc
               a = int(self.data.loc[idet,"2"]) - (self.nocc) + nc
               ia = self.idx.S(i, a) 
               det_str[i] = 'b'
               det_str[a + self.nocc] = 'a'
               self.S_a[ia] = parity * self.data.loc[idet,"3"] 
               if abs(self.data.loc[idet,"3"]) > self.top_w:
                   self.top_w     = self.data.loc[idet,"3"]
                   self.top_w_idx = [i, a] 
                   self.top_w_typ = typ 
            elif (typ == "b"):
               i = int(self.data.loc[idet,"1"]) + nc
               a = int(self.data.loc[idet,"2"]) - (self.nocc) + nc
               ia = self.idx.S(i, a) 
               det_str[i] = 'a'
               det_str[a + self.nocc] = 'b'
               self.S_b[ia] = parity * self.data.loc[idet,"3"] 
            elif (typ == "aa"):
               i = int(self.data.loc[idet,"1"]) + nc
               j = int(self.data.loc[idet,"2"]) + nc
               a = int(self.data.loc[idet,"3"]) - (self.nocc) + nc
               b = int(self.data.loc[idet,"4"]) - (self.nocc) + nc
               ijab = self.idx.D(i, j, a, b) 
               det_str[i] = 'b'
               det_str[j] = 'b'
               det_str[a + self.nocc] = 'a'
               det_str[b + self.nocc] = 'a'
               self.D_aa[ijab] = parity * self.data.loc[idet,"5"] 
               if abs(self.data.loc[idet,"5"]) > self.top_w:
                   self.top_w     = self.data.loc[idet,"5"]
                   self.top_w_idx = [i, j, a, b] 
                   self.top_w_typ = typ 
            elif (typ == "ab"):
               i = int(self.data.loc[idet,"1"]) + nc
               a = int(self.data.loc[idet,"2"]) - (self.nocc) + nc
               j = int(self.data.loc[idet,"3"]) + nc
               b = int(self.data.loc[idet,"4"]) - (self.nocc) + nc
#               print (idet,'i,a,j,b',i,a,j,b)
               ia = self.idx.S(i, a) 
               jb = self.idx.S(j, b) 
               det_str[i] = 'b'
               det_str[a + self.nocc] = 'a'
               det_str[j] = 'a' if i != j else 'v'
               det_str[b + self.nocc] = 'b' if a != b else 'ab'
               self.D_ab[ia][jb] = parity * self.data.loc[idet,"5"] 
               if abs(self.data.loc[idet,"5"]) > self.top_w:
                   self.top_w     = self.data.loc[idet,"5"]
                   self.top_w_idx = [i, j, a, b] 
                   self.top_w_typ = typ 
            elif (typ == "bb"):
               i = int(self.data.loc[idet,"1"]) + nc
               j = int(self.data.loc[idet,"2"]) + nc
               a = int(self.data.loc[idet,"3"]) - (self.nocc) + nc
               b = int(self.data.loc[idet,"4"]) - (self.nocc) + nc
               ijab = self.idx.D(i, j, a, b) 
               det_str[i] = 'a'
               det_str[j] = 'a'
               det_str[a + self.nocc] = 'b'
               det_str[b + self.nocc] = 'b'
               self.D_bb[ijab] = parity * self.data.loc[idet,"5"] 
            elif (typ == "aaa"):
               i = int(self.data.loc[idet,"1"]) + nc
               j = int(self.data.loc[idet,"2"]) + nc
               k = int(self.data.loc[idet,"3"]) + nc
               a = int(self.data.loc[idet,"4"]) - (self.nocc) + nc
               b = int(self.data.loc[idet,"5"]) - (self.nocc) + nc
               c = int(self.data.loc[idet,"6"]) - (self.nocc) + nc
               ijkabc = self.idx.T(i, j, k, a, b, c) 
               det_str[i] = 'b'
               det_str[j] = 'b'
               det_str[k] = 'b'
               det_str[a + self.nocc] = 'a'
               det_str[b + self.nocc] = 'a'
               det_str[c + self.nocc] = 'a'
               self.T_aaa[ijkabc] = parity * self.data.loc[idet,"7"] 
               if abs(self.data.loc[idet,"7"]) > self.top_w:
                   self.top_w     = self.data.loc[idet,"7"]
                   self.top_w_idx = [i, j, k, a, b, c] 
                   self.top_w_typ = typ 
            elif (typ == "aab"):
               i = int(self.data.loc[idet,"1"]) + nc
               j = int(self.data.loc[idet,"2"]) + nc
               a = int(self.data.loc[idet,"3"]) - (self.nocc) + nc
               b = int(self.data.loc[idet,"4"]) - (self.nocc) + nc
               k = int(self.data.loc[idet,"5"]) + nc
               c = int(self.data.loc[idet,"6"]) - (self.nocc) + nc
               ijab = self.idx.D(i, j, a, b) 
               kc   = self.idx.S(k, c) 
               det_str[i] = 'b'
               det_str[j] = 'b'
               det_str[a + self.nocc] = 'a'
               det_str[b + self.nocc] = 'a'
               det_str[k] = 'a' if i != k and j != k else 'v'
               det_str[c + self.nocc] = 'b' if a != c and b != c else 'ab'
               self.T_aab[ijab][kc] = parity * self.data.loc[idet,"7"] 
               if abs(self.data.loc[idet,"7"]) > self.top_w:
                   self.top_w     = self.data.loc[idet,"7"]
                   self.top_w_idx = [i, j, k, a, b, c] 
                   self.top_w_typ = typ 

            elif (typ == "abb"):
               i = int(self.data.loc[idet,"1"]) + nc
               a = int(self.data.loc[idet,"2"]) - (self.nocc) + nc
               j = int(self.data.loc[idet,"3"]) + nc
               k = int(self.data.loc[idet,"4"]) + nc
               b = int(self.data.loc[idet,"5"]) - (self.nocc) + nc
               c = int(self.data.loc[idet,"6"]) - (self.nocc) + nc
               ia   = self.idx.S(i, a) 
               jkab = self.idx.D(j, k, b, c) 
               det_str[i] = 'b'
               det_str[a + self.nocc] = 'a'
               det_str[j] = 'a' if i != j else 'v'
               det_str[k] = 'a' if i != k else 'v'
               det_str[b + self.nocc] = 'b' if a != b else 'ab'
               det_str[c + self.nocc] = 'b' if a != c else 'ab'
               self.T_abb[ia][jkab] = parity * self.data.loc[idet,"7"] 
            elif (typ == "bbb"):
               i = int(self.data.loc[idet,"1"]) + nc
               j = int(self.data.loc[idet,"2"]) + nc
               k = int(self.data.loc[idet,"3"]) + nc
               a = int(self.data.loc[idet,"4"]) - (self.nocc) + nc
               b = int(self.data.loc[idet,"5"]) - (self.nocc) + nc
               c = int(self.data.loc[idet,"6"]) - (self.nocc) + nc
               ijkabc = self.idx.T(i, j, k, a, b, c) 
               det_str[i] = 'a'
               det_str[j] = 'a'
               det_str[k] = 'a'
               det_str[a + self.nocc] = 'b'
               det_str[b + self.nocc] = 'b'
               det_str[c + self.nocc] = 'b'
               self.T_bbb[ijkabc] = parity * self.data.loc[idet,"7"] 
#            elif (typ == "aaaa"):
#               i = int(self.data.loc[idet,"1"]) + nc
#               j = int(self.data.loc[idet,"2"]) + nc
#               k = int(self.data.loc[idet,"3"]) + nc
#               l = int(self.data.loc[idet,"4"]) + nc
#               a = int(self.data.loc[idet,"5"]) - (self.nocc) + nc
#               b = int(self.data.loc[idet,"6"]) - (self.nocc) + nc
#               c = int(self.data.loc[idet,"7"]) - (self.nocc) + nc
#               d = int(self.data.loc[idet,"8"]) - (self.nocc) + nc
#               ijklabcd = self.idx.Q(i, j, k, l, a, b, c, d) 
#               parity = self.parity_ci_to_cc(i+j+k+l, 4)
#               det_str[i] = 'b'
#               det_str[j] = 'b'
#               det_str[k] = 'b'
#               det_str[l] = 'b'
#               det_str[a + self.nocc] = 'a'
#               det_str[b + self.nocc] = 'a'
#               det_str[c + self.nocc] = 'a'
#               det_str[d + self.nocc] = 'a'
#               parity *= self.parity_ab_str(det_str)
#               self.Q_aaaa[ijklabcd] = parity * self.data.loc[idet,"9"] 
            elif (typ == "aaab"):
               i = int(self.data.loc[idet,"1"]) + nc
               j = int(self.data.loc[idet,"2"]) + nc
               k = int(self.data.loc[idet,"3"]) + nc
               a = int(self.data.loc[idet,"4"]) - (self.nocc) + nc
               b = int(self.data.loc[idet,"5"]) - (self.nocc) + nc
               c = int(self.data.loc[idet,"6"]) - (self.nocc) + nc
               l = int(self.data.loc[idet,"7"]) + nc
               d = int(self.data.loc[idet,"8"]) - (self.nocc) + nc
               ijkabc = self.idx.T(i, j, k, a, b, c) 
               ld     = self.idx.S(l, d) 
               det_str[i] = 'b'
               det_str[j] = 'b'
               det_str[k] = 'b'
               det_str[a + self.nocc] = 'a'
               det_str[b + self.nocc] = 'a'
               det_str[c + self.nocc] = 'a'
               det_str[l] = 'a' if i != l and j != l and k != l else 'v'
               det_str[d + self.nocc] = 'b' if a != d and b != d and c != d else 'ab'
               self.Q_aaab[ijkabc][ld] = parity * self.data.loc[idet,"9"] 
               if abs(self.data.loc[idet,"9"]) > self.top_w:
                   self.top_w     = self.data.loc[idet,"9"]
                   self.top_w_idx = [i, j, k, l, a, b, c, d] 
                   self.top_w_typ = typ 
            elif (typ == "aabb"):
               i = int(self.data.loc[idet,"1"]) + nc
               j = int(self.data.loc[idet,"2"]) + nc
               a = int(self.data.loc[idet,"3"]) - (self.nocc) + nc
               b = int(self.data.loc[idet,"4"]) - (self.nocc) + nc
               k = int(self.data.loc[idet,"5"]) + nc
               l = int(self.data.loc[idet,"6"]) + nc
               c = int(self.data.loc[idet,"7"]) - (self.nocc) + nc
               d = int(self.data.loc[idet,"8"]) - (self.nocc) + nc
               #lsh test
#               if i == 2 and j == 3 and a == 0 and b == 1 \
#              and k == 2 and l == 3 and c == 0 and d == 1:
#               if i == 0 and j == 2 and a == 0 and b == 3 \
#              and k == 0 and l == 2 and c == 2 and d == 5:
               if True:
                   ijab = self.idx.D(i, j, a, b) 
                   klcd = self.idx.D(k, l, c, d) 
                   det_str[i] = 'b'
                   det_str[j] = 'b'
                   det_str[a + self.nocc] = 'a'
                   det_str[b + self.nocc] = 'a'
                   det_str[k] = 'a' if i != k and j != k else 'v'
                   det_str[l] = 'a' if i != l and j != l else 'v'
                   det_str[c + self.nocc] = 'b' if a != c and b != c else 'ab'
                   det_str[d + self.nocc] = 'b' if a != d and b != d else 'ab'
                   self.Q_aabb[ijab][klcd] = parity * self.data.loc[idet,"9"] 

                   #lsh test
#                   print('c4 org',parity * self.data.loc[idet,"9"]/self.Ref[0])

               if abs(self.data.loc[idet,"9"]) > self.top_w:
                   self.top_w     = self.data.loc[idet,"9"]
                   self.top_w_idx = [i, j, k, l, a, b, c, d] 
                   self.top_w_typ = typ 

#            elif (typ == "abbb"):
#               i = int(self.data.loc[idet,"1"]) + nc
#               a = int(self.data.loc[idet,"2"]) - (self.nocc) + nc
#               j = int(self.data.loc[idet,"3"]) + nc
#               k = int(self.data.loc[idet,"4"]) + nc
#               l = int(self.data.loc[idet,"5"]) + nc
#               b = int(self.data.loc[idet,"6"]) - (self.nocc) + nc
#               c = int(self.data.loc[idet,"7"]) - (self.nocc) + nc
#               d = int(self.data.loc[idet,"8"]) - (self.nocc) + nc
#               ia     = self.idx.S(i, a) 
#               jklbcd = self.idx.T(j, k, l, b, c, d) 
#               parity = self.parity_ci_to_cc(i, 1)
#               parity *= self.parity_ci_to_cc(j+k+l, 3)
#               det_str[i] = 'b'
#               det_str[a + self.nocc] = 'a'
#               det_str[j] = 'a' if i != j else 'v'
#               det_str[k] = 'a' if i != k else 'v'
#               det_str[l] = 'a' if i != l else 'v'
#               det_str[b + self.nocc] = 'b' if a != b else 'ab'
#               det_str[c + self.nocc] = 'b' if a != c else 'ab'
#               det_str[d + self.nocc] = 'b' if a != d else 'ab'
#               parity *= self.parity_ab_str(det_str)
#               self.Q_abbb[ia][jklbcd] = parity * self.data.loc[idet,"9"]
#            elif (typ == "bbbb"):
#               i = int(self.data.loc[idet,"1"]) + nc
#               j = int(self.data.loc[idet,"2"]) + nc
#               k = int(self.data.loc[idet,"3"]) + nc
#               l = int(self.data.loc[idet,"4"]) + nc
#               a = int(self.data.loc[idet,"5"]) - (self.nocc) + nc
#               b = int(self.data.loc[idet,"6"]) - (self.nocc) + nc
#               c = int(self.data.loc[idet,"7"]) - (self.nocc) + nc
#               d = int(self.data.loc[idet,"8"]) - (self.nocc) + nc
#               ijklabcd = self.idx.Q(i, j, k, l, a, b, c, d) 
#               parity = self.parity_ci_to_cc(i+j+k+l, 4)
#               det_str[i] = 'a'
#               det_str[j] = 'a'
#               det_str[k] = 'a'
#               det_str[l] = 'a'
#               det_str[a + self.nocc] = 'b'
#               det_str[b + self.nocc] = 'b'
#               det_str[c + self.nocc] = 'b'
#               det_str[d + self.nocc] = 'b'
#               parity *= self.parity_ab_str(det_str)
#               self.Q_bbbb[ijklabcd] = parity * self.data.loc[idet,"9"] 
#        print ('test max:',test_max, test_max_idx) 

    def dbg_expnd(self, paaa, pbaa):

        nocc = self.nocc_cas
        nvir = self.nvir_cas

        dD = nocc*(nocc-1)/2 * nvir*(nvir-1)/2
        nocc3 = nocc*(nocc-1)*(nocc-2)/6 

        paaa_f = numpy.full((nocc,nocc,nocc,nvir,nvir,nvir), 1.0)
        pabb_f = numpy.full((nocc,nocc,nocc,nvir,nvir,nvir), 1.0)

        def asgn_val_t6(t, i, j, k, a, b, c, val):
            t[i][j][k][a][b][c] = val 
            t[i][k][j][a][b][c] = val
            t[j][i][k][a][b][c] = val
            t[j][k][i][a][b][c] = val
            t[k][i][j][a][b][c] = val
            t[k][j][i][a][b][c] = val
            t[i][j][k][a][c][b] = val
            t[i][k][j][a][c][b] = val
            t[j][i][k][a][c][b] = val
            t[j][k][i][a][c][b] = val
            t[k][i][j][a][c][b] = val
            t[k][j][i][a][c][b] = val
            t[i][j][k][b][a][c] = val
            t[i][k][j][b][a][c] = val
            t[j][i][k][b][a][c] = val
            t[j][k][i][b][a][c] = val
            t[k][i][j][b][a][c] = val
            t[k][j][i][b][a][c] = val
            t[i][j][k][b][c][a] = val
            t[i][k][j][b][c][a] = val
            t[j][i][k][b][c][a] = val
            t[j][k][i][b][c][a] = val
            t[k][i][j][b][c][a] = val
            t[k][j][i][b][c][a] = val
            t[i][j][k][c][a][b] = val
            t[i][k][j][c][a][b] = val
            t[j][i][k][c][a][b] = val
            t[j][k][i][c][a][b] = val
            t[k][i][j][c][a][b] = val
            t[k][j][i][c][a][b] = val
            t[i][j][k][c][b][a] = val
            t[i][k][j][c][b][a] = val
            t[j][i][k][c][b][a] = val
            t[j][k][i][c][b][a] = val
            t[k][i][j][c][b][a] = val
            t[k][j][i][c][b][a] = val

        def asgn_val_t1_2(t, i, j, k, a, b, c, val):
            t[i][j][k][a][b][c] = val 
            t[i][k][j][a][b][c] = val
            t[i][j][k][a][c][b] = val
            t[i][k][j][a][c][b] = val

        ijkabc = -1 
        for c in range(2,nvir,1):
            for b in range(1,c,1):
                for a in range(0,b,1):
                    for k in range(nocc-1,1,-1):
                        for j in range(k-1,0,-1):
                            for i in range(j-1,-1,-1):
                                ijkabc += 1
                                asgn_val_t6(paaa_f,i,j,k,a,b,c,paaa[ijkabc])

        def SDidx(ia, jkbc):
            return int(ia * dD + jkbc)

        pbaa = pbaa.flatten()
        ia = -1 
        for a in range(0,nvir,1):
            for i in range(nocc-1,-1,-1):
                ia += 1

                jkbc = -1 
                for c in range(1,nvir,1):
                    for b in range(0,c,1):
                        for k in range(nocc-1,0,-1):
                            for j in range(k-1,-1,-1):
                                jkbc += 1
                                asgn_val_t1_2(pabb_f,i,j,k,a,b,c,pbaa[SDidx(ia,jkbc)]) 

        return paaa_f, pabb_f 

    #tmp
    def align_phase(self, dmrg):
        dS = int(self.nocc * self.nvir) 
        dD = int(self.nocc*(self.nocc-1)*self.nvir*(self.nvir-1)/4) 
        dT = int(self.nocc*(self.nocc-1)*(self.nocc-2)*self.nvir*(self.nvir-1)*(self.nvir-2)/36) 
#        dQ = len(self.idx.idx4.keys())
        print('dS, dD, dT =',dS,dD,dT)

        nex = self.nocc * self.nvir
#        self.Ref    = numpy.zeros((1), dtype=numpy.float64)
#
#        self.S_a    = numpy.zeros((dS), dtype=numpy.float64)
#        self.D_aa   = numpy.zeros((dD), dtype=numpy.float64)
#        self.D_ab   = numpy.zeros((dS,dS), dtype=numpy.float64)
#        self.T_aaa  = numpy.zeros((dT), dtype=numpy.float64)
#        self.T_aab  = numpy.zeros((dD,dS), dtype=numpy.float64)
#        self.Q_aaab = numpy.zeros((dT,dS), dtype=numpy.float64)
#        self.Q_aabb = numpy.zeros((dD,dD), dtype=numpy.float64)

        print('ref sign', self.Ref[0], dmrg.Ref[0])
        s_dmrg = 1.0
        if dmrg.Ref[0] < 0: s_dmrg = -1.0
        self.Ref[0] = s_dmrg * abs( self.Ref[0] )
        print('ref sign', self.Ref[0], dmrg.Ref[0], s_dmrg)


        for i in range(dS):
            s_dmrg = 1.0
            if abs(dmrg.S_a[i]) < 1e-8:
                s_dmrg = 0.0
            else:
                if dmrg.S_a[i] < 0: s_dmrg = -1.0
            self.S_a[i] = s_dmrg * abs( self.S_a[i] )

        for i in range(dD):
            s_dmrg = 1.0
            if abs(dmrg.D_aa[i]) < 1e-8:
                s_dmrg = 0.0
            else:
                if dmrg.D_aa[i] < 0: s_dmrg = -1.0
            self.D_aa[i] = s_dmrg * abs( self.D_aa[i] )

        for i in range(dS):
            for j in range(dS):
                s_dmrg = 1.0
                if abs(dmrg.D_ab[i][j]) < 1e-8:
                    s_dmrg = 0.0
                else:
                    if dmrg.D_ab[i][j] < 0: s_dmrg = -1.0
                self.D_ab[i][j] = s_dmrg * abs( self.D_ab[i][j] )

        for i in range(dT):
            s_dmrg = 1.0
            if abs(dmrg.T_aaa[i]) < 1e-8:
                s_dmrg = 0.0
            else:
                if dmrg.T_aaa[i] < 0: s_dmrg = -1.0
            self.T_aaa[i] = s_dmrg * abs( self.T_aaa[i] )

        for i in range(dD):
            for j in range(dS):
                s_dmrg = 1.0
                if abs(dmrg.T_aab[i][j]) < 1e-8:
                    s_dmrg = 0.0
                else:
                    if dmrg.T_aab[i][j] < 0: s_dmrg = -1.0
                self.T_aab[i][j] = s_dmrg * abs( self.T_aab[i][j] )

        for i in range(dT):
            for j in range(dS):
                s_dmrg = 1.0
                if abs(dmrg.Q_aaab[i][j]) < 1e-8:
                    s_dmrg = 0.0
                else:
                    if dmrg.Q_aaab[i][j] < 0: s_dmrg = -1.0
                self.Q_aaab[i][j] = s_dmrg * abs( self.Q_aaab[i][j] )

        for i in range(dD):
            for j in range(dD):
                s_dmrg = 1.0
                if abs(dmrg.Q_aabb[i][j]) < 1e-8:
                    s_dmrg = 0.0
                else:
                    if dmrg.Q_aabb[i][j] < 0: s_dmrg = -1.0
                self.Q_aabb[i][j] = s_dmrg * abs( self.Q_aabb[i][j] )

