import pandas as pd
import numpy 
from pyscf.cc.fci_index import tn_addrs_signs
from pyscf.cc import _ccsd 
from pyscf import lib
import ctypes

class dmrg_coeff: 
    def __init__(self, dmrg_out, nocc, nvir, idx, nc = 0):
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
        self.nvir_iact= None

        self.data=pd.read_csv(dmrg_out,sep=",")
        #self.data.info()

        self.typ_det = list(self.data.loc[:,"typ"])
        self.num_det = len(self.typ_det)
        #print('total number of determinants', self.num_det)

        self.HFdet_str = [ 'ab' if i < self.nocc else 'v' for i in range(self.nmo) ] 
        self.flagS = False
        self.flagD = False
        self.flagT = False
        self.flagQ = False

        self.E_dmrg = None 
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

        self.interm_norm_S = False 
        self.interm_norm_D = False 
        self.interm_norm_T = False 
        self.interm_norm_Q = False 

        self.Pmat = None

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
        return (-1)**n

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

        nocc=self.nocc
        nvir=self.nvir
        dS = int(self.nocc * self.nvir) 
        dD = int(self.nocc*(self.nocc-1)*self.nvir*(self.nvir-1)/4) 
        dT = int(self.nocc*(self.nocc-1)*(self.nocc-2)*self.nvir*(self.nvir-1)*(self.nvir-2)/36) 
        #print('dS, dD, dT =',dS,dD,dT)

        self.Ref    = numpy.zeros((1), dtype=numpy.float64)
        self.S_a    = numpy.zeros((dS), dtype=numpy.float64)
        self.D_aa   = numpy.zeros((dD), dtype=numpy.float64)
        self.D_ab   = numpy.zeros((dS,dS), dtype=numpy.float64)
        self.T_aaa  = numpy.zeros((dT), dtype=numpy.float64)
        self.T_aab  = numpy.zeros((dD,dS), dtype=numpy.float64)
        self.Q_aaab = numpy.zeros((dT,dS), dtype=numpy.float64)
        self.Q_aabb = numpy.zeros((dD,dD), dtype=numpy.float64)
        nocc_cas = self.nocc_cas

        for idet in range(self.num_det):
            typ = self.typ_det[idet]
            if (typ == "rf"):
               self.Ref[0] = self.data.loc[idet,"1"] 
            elif (typ == "a"):
               i = int(self.data.loc[idet,"1"]) + nc
               a = int(self.data.loc[idet,"2"]) - nocc_cas 
               ia = self.idx.S(i, a) 
               self.S_a[ia] = self.data.loc[idet,"3"] 
            elif (typ == "aa"):
               i = int(self.data.loc[idet,"1"]) + nc
               j = int(self.data.loc[idet,"2"]) + nc
               a = int(self.data.loc[idet,"3"]) - nocc_cas 
               b = int(self.data.loc[idet,"4"]) - nocc_cas 
               ijab = self.idx.D(i, j, a, b) 
               self.D_aa[ijab] = self.data.loc[idet,"5"] 
            elif (typ == "ab"):
               i = int(self.data.loc[idet,"1"]) + nc
               a = int(self.data.loc[idet,"2"]) - nocc_cas
               j = int(self.data.loc[idet,"3"]) + nc
               b = int(self.data.loc[idet,"4"]) - nocc_cas
               ia = self.idx.S(i, a) 
               jb = self.idx.S(j, b) 
               self.D_ab[ia][jb] = self.data.loc[idet,"5"] 
            elif (typ == "aaa"):
               i = int(self.data.loc[idet,"1"]) + nc
               j = int(self.data.loc[idet,"2"]) + nc
               k = int(self.data.loc[idet,"3"]) + nc
               a = int(self.data.loc[idet,"4"]) - nocc_cas 
               b = int(self.data.loc[idet,"5"]) - nocc_cas
               c = int(self.data.loc[idet,"6"]) - nocc_cas
               ijkabc = self.idx.T(i, j, k, a, b, c) 
               self.T_aaa[ijkabc] = self.data.loc[idet,"7"] 
            elif (typ == "aab"):
               i = int(self.data.loc[idet,"1"]) + nc
               j = int(self.data.loc[idet,"2"]) + nc
               a = int(self.data.loc[idet,"3"]) - nocc_cas
               b = int(self.data.loc[idet,"4"]) - nocc_cas
               k = int(self.data.loc[idet,"5"]) + nc
               c = int(self.data.loc[idet,"6"]) - nocc_cas 
               ijab = self.idx.D(i, j, a, b) 
               kc   = self.idx.S(k, c) 
               self.T_aab[ijab][kc] = self.data.loc[idet,"7"] 
            elif (typ == "aaab"):
               i = int(self.data.loc[idet,"1"]) + nc
               j = int(self.data.loc[idet,"2"]) + nc
               k = int(self.data.loc[idet,"3"]) + nc
               a = int(self.data.loc[idet,"4"]) - nocc_cas
               b = int(self.data.loc[idet,"5"]) - nocc_cas
               c = int(self.data.loc[idet,"6"]) - nocc_cas
               l = int(self.data.loc[idet,"7"]) + nc
               d = int(self.data.loc[idet,"8"]) - nocc_cas 
               ijkabc = self.idx.T(i, j, k, a, b, c) 
               ld     = self.idx.S(l, d) 
               self.Q_aaab[ijkabc][ld] = self.data.loc[idet,"9"] 
            elif (typ == "aabb"):
               i = int(self.data.loc[idet,"1"]) + nc
               j = int(self.data.loc[idet,"2"]) + nc
               a = int(self.data.loc[idet,"3"]) - nocc_cas
               b = int(self.data.loc[idet,"4"]) - nocc_cas
               k = int(self.data.loc[idet,"5"]) + nc
               l = int(self.data.loc[idet,"6"]) + nc
               c = int(self.data.loc[idet,"7"]) - nocc_cas
               d = int(self.data.loc[idet,"8"]) - nocc_cas

#               if i == 2 and j == 3 and a == 0 and b == 1 \
#              and k == 2 and l == 3 and c == 0 and d == 1:
               if True:
                   ijab = self.idx.D(i, j, a, b) 
                   klcd = self.idx.D(k, l, c, d) 
                   self.Q_aabb[ijab][klcd] = self.data.loc[idet,"9"] 


        self.S_b    = self.S_a
        self.D_bb   = self.D_aa
        self.T_abb  = self.T_aab
        self.T_bbb  = self.T_aaa
        self.Q_abbb = self.Q_aaab
        self.Q_bbbb = self.Q_aaaa

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
            if (typ == "rf"):
               self.Ref[0] = self.data.loc[idet,"1"] 
            elif (typ == "a"):
               i = int(self.data.loc[idet,"1"])
               a = int(self.data.loc[idet,"2"]) - self.nocc_cas
               ia = self.idx.S(i, a) 
               parity = self.parity_ci_to_cc(i, 1)
               self.S_a[ia] = parity * self.data.loc[idet,"3"] 
            elif (typ == "aa"):
               i = int(self.data.loc[idet,"1"])
               j = int(self.data.loc[idet,"2"])
               a = int(self.data.loc[idet,"3"]) - self.nocc_cas
               b = int(self.data.loc[idet,"4"]) - self.nocc_cas
               ijab = self.idx.D(i, j, a, b) 
               parity = self.parity_ci_to_cc(i+j, 2)
               self.D_aa[ijab] = parity * self.data.loc[idet,"5"] 
            elif (typ == "ab"):
               i = int(self.data.loc[idet,"1"])
               a = int(self.data.loc[idet,"2"]) - self.nocc_cas
               j = int(self.data.loc[idet,"3"])
               b = int(self.data.loc[idet,"4"]) - self.nocc_cas
               ia = self.idx.S(i, a) 
               jb = self.idx.S(j, b) 
               parity  = self.parity_ci_to_cc(i, 1)
               parity *= self.parity_ci_to_cc(j, 1)
               self.D_ab[ia][jb] = parity * self.data.loc[idet,"5"] 
        self.S_b    = self.S_a 
        self.D_bb   = self.D_aa


    def get_SDT(self, nc, numzero=1e-9):
        self.nc = nc
        self.flagS = True 
        self.flagD = True
        self.flagT = True
        self.numzero = numzero

        dS = int(self.nocc * self.nvir)
        dD = int(self.nocc*(self.nocc-1)*self.nvir*(self.nvir-1)/4) 
        dT = int(self.nocc*(self.nocc-1)*(self.nocc-2)*self.nvir*(self.nvir-1)*(self.nvir-2)/36) 
        print('dS, dD, dT =',dS,dD,dT)

        self.Ref    = numpy.zeros((1), dtype=numpy.float64)
        self.S_a    = numpy.zeros((dS), dtype=numpy.float64)
        self.D_aa   = numpy.zeros((dD), dtype=numpy.float64)
        self.D_ab   = numpy.zeros((dS,dS), dtype=numpy.float64)
        self.T_aaa  = numpy.zeros((dT), dtype=numpy.float64)
        self.T_aab  = numpy.zeros((dD,dS), dtype=numpy.float64)
        nocc_cas = self.nocc_cas

        for idet in range(self.num_det):
            typ = self.typ_det[idet]
            det_str = self.HFdet_str.copy()
            parity  = 1
            if (typ == "rf"):
               self.Ref[0] = self.data.loc[idet,"1"] 
            elif (typ == "a"):
               i = int(self.data.loc[idet,"1"]) + nc
               a = int(self.data.loc[idet,"2"]) - nocc_cas 
               ia = self.idx.S(i, a) 
               parity = self.parity_ci_to_cc(i, 1)
               det_str[i] = 'b'
               det_str[a + self.nocc] = 'a'
               #parity *= self.parity_ab_str(det_str)
               self.S_a[ia] = parity * self.data.loc[idet,"3"] 
            elif (typ == "aa"):
               i = int(self.data.loc[idet,"1"]) + nc
               j = int(self.data.loc[idet,"2"]) + nc
               a = int(self.data.loc[idet,"3"]) - nocc_cas 
               b = int(self.data.loc[idet,"4"]) - nocc_cas 
               ijab = self.idx.D(i, j, a, b) 
               parity = self.parity_ci_to_cc(i+j, 2)
               det_str[i] = 'b'
               det_str[j] = 'b'
               det_str[a + self.nocc] = 'a'
               det_str[b + self.nocc] = 'a'
               #parity *= self.parity_ab_str(det_str)
               self.D_aa[ijab] = parity * self.data.loc[idet,"5"] 
            elif (typ == "ab"):
               i = int(self.data.loc[idet,"1"]) + nc
               a = int(self.data.loc[idet,"2"]) - nocc_cas 
               j = int(self.data.loc[idet,"3"]) + nc
               b = int(self.data.loc[idet,"4"]) - nocc_cas 
               ia = self.idx.S(i, a) 
               jb = self.idx.S(j, b) 
               parity  = self.parity_ci_to_cc(i, 1)
               parity *= self.parity_ci_to_cc(j, 1)
               det_str[i] = 'b'
               det_str[a + self.nocc] = 'a'
               det_str[j] = 'a' if i != j else 'v'
               det_str[b + self.nocc] = 'b' if a != b else 'ab'
               #parity *= self.parity_ab_str(det_str)
               self.D_ab[ia][jb] = parity * self.data.loc[idet,"5"] 
            elif (typ == "aaa"):
               i = int(self.data.loc[idet,"1"]) + nc
               j = int(self.data.loc[idet,"2"]) + nc
               k = int(self.data.loc[idet,"3"]) + nc
               a = int(self.data.loc[idet,"4"]) - nocc_cas 
               b = int(self.data.loc[idet,"5"]) - nocc_cas 
               c = int(self.data.loc[idet,"6"]) - nocc_cas 

               if abs(float(self.data.loc[idet,"7"])) > numzero: 
                   ijkabc = self.idx.T(i, j, k, a, b, c) 
                   parity = self.parity_ci_to_cc(i+j+k, 3)
                   det_str[i] = 'b'
                   det_str[j] = 'b'
                   det_str[k] = 'b'
                   det_str[a + self.nocc] = 'a'
                   det_str[b + self.nocc] = 'a'
                   det_str[c + self.nocc] = 'a'
                   #parity *= self.parity_ab_str(det_str)
                   self.T_aaa[ijkabc] = parity * self.data.loc[idet,"7"] 
            elif (typ == "aab"):
               i = int(self.data.loc[idet,"1"]) + nc
               j = int(self.data.loc[idet,"2"]) + nc
               a = int(self.data.loc[idet,"3"]) - nocc_cas
               b = int(self.data.loc[idet,"4"]) - nocc_cas
               k = int(self.data.loc[idet,"5"]) + nc
               c = int(self.data.loc[idet,"6"]) - nocc_cas

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
                   #parity *= self.parity_ab_str(det_str)
                   self.T_aab[ijab][kc] = parity * self.data.loc[idet,"7"] 

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

    def get_t2t4c(self, t2t4c, e2ovov, ci2cc, numzero, nc):
        self.nc = nc
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
        #print('=== END extracting S t amplitudes ===')
        ci2cc.c2_to_t2(self.D_aa.copy(),self.D_ab.copy())
        #print('=== END extracting D t amplitudes ===')
        #TODO: argument numzero as instance of ci2cc 
        #TODO: avoide gen t3 amplitudes 
        ci2cc.c3_to_t3(self.T_aaa.copy(), self.T_aab.copy(),numzero=numzero)
        #print('=== END extracting T t amplitudes ===')

        test_max = 0.0
        test_max_idx = [] 

#        # release memory of previous dataframe
#        import gc
#        del [[self.data]]
#        gc.collect()
#        self.data=pd.DataFrame() 

        _ccsd.libcc.t2t4c_dmrg(t2t4c.ctypes.data_as(ctypes.c_void_p),
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
        #print('=== END contracting Q t amplitudes ===')

    def get_t2t4c_omp_otf_mem(self, t2t4c, e2ovov, ci2cc, norm, numzero=1e-9):
        _ccsd.libcc.t2t4c_dmrg_omp_otf_mem(t2t4c.ctypes.data_as(ctypes.c_void_p),
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

    def get_t1t3c_omp_mem(self, t1t3c, e2ovov, ci2cc, numzero=1e-9):
        #norm0SD = norm
        norm = 0.0 
        _ccsd.libcc.t1t3c_dmrg_omp(t1t3c.ctypes.data_as(ctypes.c_void_p),
                          ci2cc.t1.ctypes.data_as(ctypes.c_void_p),
                          ci2cc.t2aa.ctypes.data_as(ctypes.c_void_p),
                          ci2cc.t2ab.ctypes.data_as(ctypes.c_void_p),
                          e2ovov.ctypes.data_as(ctypes.c_void_p),
                          ctypes.c_int(self.nocc_iact),
                          ctypes.c_int(self.nocc_corr),ctypes.c_int(self.nvir_corr),
                          ctypes.c_double(numzero),ctypes.c_double(self.Ref[0]),
                          ctypes.c_double(norm)) 

    def get_t2t3c_omp_mem(self, t2_t3t4c, tmp1, tmp2, tmp3, ci2cc, numzero=1e-9):
        _ccsd.libcc.t2t3c_dmrg_omp(t2_t3t4c.ctypes.data_as(ctypes.c_void_p),
                          ci2cc.t1.ctypes.data_as(ctypes.c_void_p),
                          ci2cc.t2aa.ctypes.data_as(ctypes.c_void_p),
                          ci2cc.t2ab.ctypes.data_as(ctypes.c_void_p),
                          tmp1.ctypes.data_as(ctypes.c_void_p),
                          tmp2.ctypes.data_as(ctypes.c_void_p),
                          tmp3.ctypes.data_as(ctypes.c_void_p),
                          ctypes.c_int(self.nocc_iact),
                          ctypes.c_int(self.nocc_corr),ctypes.c_int(self.nvir_corr),
                          ctypes.c_double(numzero),ctypes.c_double(self.Ref[0])) 

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

    def align(self):   
        signref = 1
        if self.Ref[0] < 0: signref = -1

        self.Ref    = self.Ref   * signref 
        self.S_a    = self.S_a   * signref 
        self.D_aa   = self.D_aa  * signref 
        self.D_ab   = self.D_ab  * signref 
        self.T_aaa  = self.T_aaa * signref 
        self.T_aab  = self.T_aab * signref 
        self.Q_aaab = self.Q_aaab * signref 
        self.Q_aabb = self.Q_aabb * signref 

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

            elif (typ == "aab"):
               if abs(float(self.data.loc[idet,"7"])) > self.numzero:
                   i = int(self.data.loc[idet,"1"]) + nc
                   j = int(self.data.loc[idet,"2"]) + nc
                   a = int(self.data.loc[idet,"3"]) - nocc_cas
                   b = int(self.data.loc[idet,"4"]) - nocc_cas
                   k = int(self.data.loc[idet,"5"]) + nc
                   c = int(self.data.loc[idet,"6"]) - nocc_cas
    
                   asgn_zero_t1_2(self.Pabb, k, i, j, c, a, b)
                   ncount_abb += 1

        #print ('n_aaa, n_abb=', ncount_aaa, ncount_abb)

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
            elif (typ == "aab"):
               if abs(float(self.data.loc[idet,"7"])) > self.numzero:
                   i = int(self.data.loc[idet,"1"]) 
                   j = int(self.data.loc[idet,"2"])
                   a = int(self.data.loc[idet,"3"]) - nocc
                   b = int(self.data.loc[idet,"4"]) - nocc
                   k = int(self.data.loc[idet,"5"])
                   c = int(self.data.loc[idet,"6"]) - nocc
                   self.Pbaa[S(k,c)][D(i,j,a,b)] = 0.0
                   ncount_baa += 1

        #print ('n_aaa, n_baa=', ncount_aaa, ncount_baa)

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
            if (typ == "aab"):
               if abs(float(self.data.loc[idet,"7"])) > self.numzero:
                   i = int(self.data.loc[idet,"1"]) + nc
                   j = int(self.data.loc[idet,"2"]) + nc
                   a = int(self.data.loc[idet,"3"]) - nocc_cas
                   b = int(self.data.loc[idet,"4"]) - nocc_cas
                   k = int(self.data.loc[idet,"5"]) + nc
                   c = int(self.data.loc[idet,"6"]) - nocc_cas
                   asgn_zero_t1_2(w, k, i, j, c, a, b)

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

        #print ('n_aaa, n_aab=', ncount_aaa, ncount_aab)


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



