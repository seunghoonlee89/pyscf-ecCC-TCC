/* Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
  
   Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
 
        http://www.apache.org/licenses/LICENSE-2.0
 
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

 *
 * Author: Qiming Sun <osirpt.sun@gmail.com>
 */

#include <stdio.h>
#include <stdlib.h>
#include "config.h"
#include "np_helper/np_helper.h"
#include "vhf/fblas.h"
#include <assert.h>

typedef struct {
        void *cache[6];
        short a;
        short b;
        short c;
        short _padding;
} CacheJob;

int Sc (int i, int a, int nocc);
int Dc (int i, int j, int a, int b, int nocc2);
int Tc (int i, int j, int k, int a, int b, int c, int nocc3);
int DSc(int i, int j, int k, int a, int b, int c, int nocc, int nvir, int nocc2);
int SDc(int i, int j, int k, int a, int b, int c, int nocc, int nvir, int nocc2)
{ return Sc(i, a, nocc) * nocc2 * nvir*(nvir-1)/2 + Dc(j, k, b, c, nocc2); }

int S(int i, int a, int nvir);
int D(int i, int j, int a, int b, int nocc, int nvir);
size_t T(int i, int j, int k, int a, int b, int c, int nocc, int nvir);
int SD(int i, int j, int k, int a, int b, int c, int nocc, int nvir)
{ return S(i, a, nvir) * nocc * nocc * nvir * nvir + D(j, k, b, c, nocc, nvir); }

double _ccsd_t_get_energy(double *w, double *v, double *mo_energy, int nocc,
                          int a, int b, int c, double fac);

size_t _ccsd_t_gen_jobs(CacheJob *jobs, int nocc, int nvir,
                        int a0, int a1, int b0, int b1,
                        void *cache_row_a, void *cache_col_a,
                        void *cache_row_b, void *cache_col_b, size_t stride);

void _make_permute_indices(int *idx, int n);

double _ccsd_t_zget_energy(double complex *w, double complex *v,
                           double *mo_energy, int nocc,
                           int a, int b, int c, double fac);

int comparator (const void * p1, const void * p2)
{
  return (*(int*)p1 - *(int*)p2);
}

double _ecccsd_t_get_energy(double *w, double *v, double *mo_energy, double *paaa, int nocc, int nocc3, int nocc_iact,
                          int a, int b, int c, double fac)
{
        int i, j, k, n;
        int it, jt, kt, at, bt, ct;
        double abc = mo_energy[nocc+a] + mo_energy[nocc+b] + mo_energy[nocc+c];
        double et = 0;
        double exclude = 1.0;
        int arr[3] = {a, b, c}; 
        qsort(arr, 3, sizeof(int), comparator);
        at = arr[0];
        bt = arr[1];
        ct = arr[2];

        for (n = 0, i = 0; i < nocc; i++) {
        for (j = 0; j < nocc; j++) {
        for (k = 0; k < nocc; k++, n++) {
                arr[0]=i;
                arr[1]=j;
                arr[2]=k;
                qsort(arr, 3, sizeof(int), comparator);
                it = arr[0]-nocc_iact;
                jt = arr[1]-nocc_iact;
                kt = arr[2]-nocc_iact;
                if ( it != jt && jt != kt ) exclude = paaa[Tc(it, jt, kt, at, bt, ct, nocc3)];

                et += fac * w[n] * v[n] / (mo_energy[i] + mo_energy[j] + mo_energy[k] - abc) * exclude;
        } } }
        return et;
}



/*
 * w + w.transpose(1,2,0) + w.transpose(2,0,1)
 * - w.transpose(2,1,0) - w.transpose(0,2,1) - w.transpose(1,0,2)
 */
static void add_and_permute(double *out, double *w, double *v, int n)
{
        int nn = n * n;
        int nnn = nn * n;
        int i, j, k;

        for (i = 0; i < nnn; i++) {
                v[i] += w[i];
        }

        for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
        for (k = 0; k < n; k++) {
                out[i*nn+j*n+k] = v[i*nn+j*n+k] + v[j*nn+k*n+i] + v[k*nn+i*n+j]
                                - v[k*nn+j*n+i] - v[i*nn+k*n+j] - v[j*nn+i*n+k];
        } } }
}

/*
 * t2T = t2.transpose(2,3,0,1)
 * ov = vv_op[:,nocc:]
 * oo = vv_op[:,:nocc]
 * w = numpy.einsum('if,fjk->ijk', -ov, t2T[c])
 * w-= numpy.einsum('ijm,mk->ijk', vooo[a], t2T[b,c])
 * v = numpy.einsum('ij,k->ijk', oo, t1T[c])
 * v+= w
 */
static void get_wv(double *w, double *v, double *cache,
                   double *fvohalf, double *vooo,
                   double *vv_op, double *t1T, double *t2T,
                   int nocc, int nvir, int a, int b, int c, int *idx)
{
        const double D0 = 0;
        const double D1 = 1;
        const double DN1 =-1;
        const char TRANS_N = 'N';
        const char TRANS_T = 'T';
        const int nmo = nocc + nvir;
        const int noo = nocc * nocc;
        const size_t nooo = nocc * noo;
        const size_t nvoo = nvir * noo;
        int i, j, k, n;
        double *pt2T;

        dgemm_(&TRANS_N, &TRANS_N, &noo, &nocc, &nvir,
               &DN1, t2T+c*nvoo, &noo, vv_op+nocc, &nmo,
               &D0, cache, &noo);
        dgemm_(&TRANS_N, &TRANS_T, &nocc, &noo, &nocc,
               &DN1, t2T+b*nvoo+c*noo, &nocc, vooo+a*nooo, &noo,
               &D1, cache, &nocc);

        pt2T = t2T + a * nvoo + b * noo;
        for (n = 0, i = 0; i < nocc; i++) {
        for (j = 0; j < nocc; j++) {
        for (k = 0; k < nocc; k++, n++) {
                w[idx[n]] += cache[n];
                v[idx[n]] +=(vv_op[i*nmo+j] * t1T[c*nocc+k]
                           + pt2T[i*nocc+j] * fvohalf[c*nocc+k]);
        } } }
}

////full p mat version
//static void get_wv_ec(double *w, double *v, double *cache,
//                   double *fvohalf, double *vooo,
//                   double *vv_op, double *t1T, double *t2T,
//                   double *paaa, int nocc, int nocc3, int nocc_iact, 
//                   int nvir, int a, int b, int c, int *idx)
//{
//        const double D0 = 0;
//        const double D1 = 1;
//        const double DN1 =-1;
//        const char TRANS_N = 'N';
//        const char TRANS_T = 'T';
//        const int nmo = nocc + nvir;
//        const int noo = nocc * nocc;
//        const size_t nooo = nocc * noo;
//        const size_t nvoo = nvir * noo;
//        int i, j, k, n;
//        double *pt2T;
//
//        dgemm_(&TRANS_N, &TRANS_N, &noo, &nocc, &nvir,
//               &DN1, t2T+c*nvoo, &noo, vv_op+nocc, &nmo,
//               &D0, cache, &noo);
//        dgemm_(&TRANS_N, &TRANS_T, &nocc, &noo, &nocc,
//               &DN1, t2T+b*nvoo+c*noo, &nocc, vooo+a*nooo, &noo,
//               &D1, cache, &nocc);
//
//        pt2T = t2T + a * nvoo + b * noo;
//        for (n = 0, i = 0; i < nocc; i++) {
//        for (j = 0; j < nocc; j++) {
//        for (k = 0; k < nocc; k++, n++) {
//                w[idx[n]] += cache[n] * paaa[T(i-nocc_iact, j-nocc_iact, k-nocc_iact, a, b, c, nocc, nvir)];
//                v[idx[n]] +=(vv_op[i*nmo+j] * t1T[c*nocc+k]
//                           + pt2T[i*nocc+j] * fvohalf[c*nocc+k]) * paaa[T(i-nocc_iact, j-nocc_iact, k-nocc_iact, a, b, c, nocc, nvir)];
//        } } }
//}

// off diag version
static void get_wv_ec(double *w, double *v, double *cache,
                   double *fvohalf, double *vooo,
                   double *vv_op, double *t1T, double *t2T,
                   double *paaa, int nocc, int nocc3, int nocc_iact, 
                   int nvir, int nvir_cas, int a, int b, int c, int *idx)
{
        const double D0 = 0;
        const double D1 = 1;
        const double DN1 =-1;
        const char TRANS_N = 'N';
        const char TRANS_T = 'T';
        const int nmo = nocc + nvir;
        const int noo = nocc * nocc;
        const size_t nooo = nocc * noo;
        const size_t nvoo = nvir * noo;
        int i, j, k, n;
        double *pt2T;

        int it, jt, kt, at, bt, ct;
        double exclude = 1.0;
        int arr[3] = {a, b, c}; 
        qsort(arr, 3, sizeof(int), comparator);
        at = arr[0];
        bt = arr[1];
        ct = arr[2];

        dgemm_(&TRANS_N, &TRANS_N, &noo, &nocc, &nvir,
               &DN1, t2T+c*nvoo, &noo, vv_op+nocc, &nmo,
               &D0, cache, &noo);
        dgemm_(&TRANS_N, &TRANS_T, &nocc, &noo, &nocc,
               &DN1, t2T+b*nvoo+c*noo, &nocc, vooo+a*nooo, &noo,
               &D1, cache, &nocc);

        pt2T = t2T + a * nvoo + b * noo;
        for (n = 0, i = 0; i < nocc; i++) {
        for (j = 0; j < nocc; j++) {
        for (k = 0; k < nocc; k++, n++) {
                arr[0]=i;
                arr[1]=j;
                arr[2]=k;
                qsort(arr, 3, sizeof(int), comparator);
                it = arr[0]-nocc_iact;
                jt = arr[1]-nocc_iact;
                kt = arr[2]-nocc_iact;
                if ( it >= 0 && jt >= 0 && kt >= 0 
                  && it != jt && jt != kt && it != kt
                  && at != bt && bt != ct && at != ct
                  && at < nvir_cas && bt < nvir_cas && ct < nvir_cas ){
                    exclude = paaa[Tc(it, jt, kt, at, bt, ct, nocc3)];
                    //printf("idx: %d for %d %d %d %d %d %d\n",Tc(it, jt, kt, at, bt, ct, nocc3),it, jt, kt, at, bt, ct);
                    //if (exclude == 1.0) printf("no exclude: %d %d %d %d %d %d\n",it+nocc_iact,jt+nocc_iact,kt+nocc_iact,at+nocc,bt+nocc,ct+nocc);
                }
                else
                    exclude = 1.0;

                w[idx[n]] += cache[n] * exclude;
                v[idx[n]] +=(vv_op[i*nmo+j] * t1T[c*nocc+k]
                           + pt2T[i*nocc+j] * fvohalf[c*nocc+k]) * exclude;
        } } }
}

// off diag version
static void get_wv_ecr(double *w, double *v, double *y, double *cache,
                   double *fvohalf, double *vooo,
                   double *vv_op, double *t1T, double *t2T,
                   double *paaa, int nocc, int nocc3, int nocc_iact, 
                   int nvir, int nvir_cas, int a, int b, int c, int *idx)
{
        const double D0 = 0;
        const double D1 = 1;
        const double DN1 =-1;
        const char TRANS_N = 'N';
        const char TRANS_T = 'T';
        const int nmo = nocc + nvir;
        const int noo = nocc * nocc;
        const size_t nooo = nocc * noo;
        const size_t nvoo = nvir * noo;
        int i, j, k, n;
        double *pt2Tab, *pt2Tbc, *pt2Tac;

        int it, jt, kt, at, bt, ct;
        double exclude = 1.0;
        int arr[3] = {a, b, c}; 
        qsort(arr, 3, sizeof(int), comparator);
        at = arr[0];
        bt = arr[1];
        ct = arr[2];

        dgemm_(&TRANS_N, &TRANS_N, &noo, &nocc, &nvir,
               &DN1, t2T+c*nvoo, &noo, vv_op+nocc, &nmo,
               &D0, cache, &noo);
        dgemm_(&TRANS_N, &TRANS_T, &nocc, &noo, &nocc,
               &DN1, t2T+b*nvoo+c*noo, &nocc, vooo+a*nooo, &noo,
               &D1, cache, &nocc);

        pt2Tab = t2T + a * nvoo + b * noo;
        pt2Tbc = t2T + b * nvoo + c * noo;
        pt2Tac = t2T + a * nvoo + c * noo;
        for (n = 0, i = 0; i < nocc; i++) {
        for (j = 0; j < nocc; j++) {
        for (k = 0; k < nocc; k++, n++) {
                arr[0]=i;
                arr[1]=j;
                arr[2]=k;
                qsort(arr, 3, sizeof(int), comparator);
                it = arr[0]-nocc_iact;
                jt = arr[1]-nocc_iact;
                kt = arr[2]-nocc_iact;
                if ( it >= 0 && jt >= 0 && kt >= 0 
                  && it != jt && jt != kt && it != kt
                  && at != bt && bt != ct && at != ct
                  && at < nvir_cas && bt < nvir_cas && ct < nvir_cas ){
                    exclude = paaa[Tc(it, jt, kt, at, bt, ct, nocc3)];
                    //printf("idx: %d for %d %d %d %d %d %d\n",Tc(it, jt, kt, at, bt, ct, nocc3),it, jt, kt, at, bt, ct);
                    //if (exclude == 1.0) printf("no exclude: %d %d %d %d %d %d\n",it+nocc_iact,jt+nocc_iact,kt+nocc_iact,at+nocc,bt+nocc,ct+nocc);
                }
                else
                    exclude = 1.0;

                w[idx[n]] += cache[n] * exclude;
                v[idx[n]] +=(vv_op[i*nmo+j]  * t1T[c*nocc+k]
                           + pt2Tab[i*nocc+j]* fvohalf[c*nocc+k]) * exclude;
                y[idx[n]] += (t1T[a*nocc+i] * t1T[b*nocc+j] * t1T[c*nocc+k] 
                             +t1T[a*nocc+i] * pt2Tbc[j*nocc+k]
                             +t1T[b*nocc+j] * pt2Tac[i*nocc+k]
                             +t1T[c*nocc+k] * pt2Tab[i*nocc+j])* exclude;
        } } }
}

static void sym_wv(double *w, double *v, double *cache,
                   double *fvohalf, double *vooo,
                   double *vv_op, double *t1T, double *t2T,
                   int nocc, int nvir, int a, int b, int c, int nirrep,
                   int *o_ir_loc, int *v_ir_loc, int *oo_ir_loc, int *orbsym,
                   int *idx)
{
        const double D0 = 0;
        const double D1 = 1;
        const char TRANS_N = 'N';
        const int nmo = nocc + nvir;
        const int noo = nocc * nocc;
        const int nooo = nocc * noo;
        const int nvoo = nvir * noo;
        int a_irrep = orbsym[nocc+a];
        int b_irrep = orbsym[nocc+b];
        int c_irrep = orbsym[nocc+c];
        int ab_irrep = a_irrep ^ b_irrep;
        int bc_irrep = c_irrep ^ b_irrep;
        int i, j, k, n;
        int fr, f0, f1, df, mr, m0, m1, dm, mk0;
        int ir, i0, i1, di, kr, k0, k1, dk, jr;
        int ijr, ij0, ij1, dij, jkr, jk0, jk1, djk;
        double *pt2T;

/* symmetry adapted
 * w = numpy.einsum('if,fjk->ijk', ov, t2T[c]) */
        pt2T = t2T + c * nvoo;
        for (ir = 0; ir < nirrep; ir++) {
                i0 = o_ir_loc[ir];
                i1 = o_ir_loc[ir+1];
                di = i1 - i0;
                if (di > 0) {
                        fr = ir ^ ab_irrep;
                        f0 = v_ir_loc[fr];
                        f1 = v_ir_loc[fr+1];
                        df = f1 - f0;
                        if (df > 0) {
                                jkr = fr ^ c_irrep;
                                jk0 = oo_ir_loc[jkr];
                                jk1 = oo_ir_loc[jkr+1];
                                djk = jk1 - jk0;
                                if (djk > 0) {

        dgemm_(&TRANS_N, &TRANS_N, &djk, &di, &df,
               &D1, pt2T+f0*noo+jk0, &noo, vv_op+i0*nmo+nocc+f0, &nmo,
               &D0, cache, &djk);
        for (n = 0, i = o_ir_loc[ir]; i < o_ir_loc[ir+1]; i++) {
        for (jr = 0; jr < nirrep; jr++) {
                kr = jkr ^ jr;
                for (j = o_ir_loc[jr]; j < o_ir_loc[jr+1]; j++) {
                for (k = o_ir_loc[kr]; k < o_ir_loc[kr+1]; k++, n++) {
                        w[idx[i*noo+j*nocc+k]] -= cache[n];
                } }
        } }
                                }
                        }
                }
        }

/* symmetry adapted
 * w-= numpy.einsum('ijm,mk->ijk', eris_vooo[a], t2T[c,b]) */
        pt2T = t2T + c * nvoo + b * noo;
        vooo += a * nooo;
        mk0 = oo_ir_loc[bc_irrep];
        for (mr = 0; mr < nirrep; mr++) {
                m0 = o_ir_loc[mr];
                m1 = o_ir_loc[mr+1];
                dm = m1 - m0;
                if (dm > 0) {
                        kr = mr ^ bc_irrep;
                        k0 = o_ir_loc[kr];
                        k1 = o_ir_loc[kr+1];
                        dk = k1 - k0;
                        if (dk > 0) {
                                ijr = mr ^ a_irrep;
                                ij0 = oo_ir_loc[ijr];
                                ij1 = oo_ir_loc[ijr+1];
                                dij = ij1 - ij0;
                                if (dij > 0) {

        dgemm_(&TRANS_N, &TRANS_N, &dk, &dij, &dm,
               &D1, pt2T+mk0, &dk, vooo+ij0*nocc+m0, &nocc,
               &D0, cache, &dk);
        for (n = 0, ir = 0; ir < nirrep; ir++) {
                jr = ijr ^ ir;
                for (i = o_ir_loc[ir]; i < o_ir_loc[ir+1]; i++) {
                for (j = o_ir_loc[jr]; j < o_ir_loc[jr+1]; j++) {
                for (k = o_ir_loc[kr]; k < o_ir_loc[kr+1]; k++, n++) {
                        w[idx[i*noo+j*nocc+k]] -= cache[n];
                } }
        } }
                                }
                                mk0 += dm * dk;
                        }
                }
        }

        pt2T = t2T + a * nvoo + b * noo;
        for (n = 0, i = 0; i < nocc; i++) {
        for (j = 0; j < nocc; j++) {
        for (k = 0; k < nocc; k++, n++) {
                v[idx[n]] +=(vv_op[i*nmo+j] * t1T[c*nocc+k]
                           + pt2T[i*nocc+j] * fvohalf[c*nocc+k]);
        } } }
}

static double contract6_aaa(int nocc, int nvir, int a, int b, int c,
                            double *mo_energy, double *t1T, double *t2T,
                            int nirrep, int *o_ir_loc, int *v_ir_loc,
                            int *oo_ir_loc, int *orbsym, double *fvo,
                            double *vooo, double *cache1, void **cache,
                            int *permute_idx)
{
        int nooo = nocc * nocc * nocc;
        int *idx0 = permute_idx;
        int *idx1 = idx0 + nooo;
        int *idx2 = idx1 + nooo;
        int *idx3 = idx2 + nooo;
        int *idx4 = idx3 + nooo;
        int *idx5 = idx4 + nooo;
        double *v0 = cache1;
        double *w0 = v0 + nooo;
        double *z0 = w0 + nooo;
        double *wtmp = z0;
        int i;

        for (i = 0; i < nooo; i++) {
                w0[i] = 0;
                v0[i] = 0;
        }

        if (nirrep == 1) {
                get_wv(w0, v0, wtmp, fvo, vooo, cache[0], t1T, t2T, nocc, nvir, a, b, c, idx0);
                get_wv(w0, v0, wtmp, fvo, vooo, cache[1], t1T, t2T, nocc, nvir, a, c, b, idx1);
                get_wv(w0, v0, wtmp, fvo, vooo, cache[2], t1T, t2T, nocc, nvir, b, a, c, idx2);
                get_wv(w0, v0, wtmp, fvo, vooo, cache[3], t1T, t2T, nocc, nvir, b, c, a, idx3);
                get_wv(w0, v0, wtmp, fvo, vooo, cache[4], t1T, t2T, nocc, nvir, c, a, b, idx4);
                get_wv(w0, v0, wtmp, fvo, vooo, cache[5], t1T, t2T, nocc, nvir, c, b, a, idx5);
        } else {
                sym_wv(w0, v0, wtmp, fvo, vooo, cache[0], t1T, t2T, nocc, nvir, a, b, c,
                       nirrep, o_ir_loc, v_ir_loc, oo_ir_loc, orbsym, idx0);
                sym_wv(w0, v0, wtmp, fvo, vooo, cache[1], t1T, t2T, nocc, nvir, a, c, b,
                       nirrep, o_ir_loc, v_ir_loc, oo_ir_loc, orbsym, idx1);
                sym_wv(w0, v0, wtmp, fvo, vooo, cache[2], t1T, t2T, nocc, nvir, b, a, c,
                       nirrep, o_ir_loc, v_ir_loc, oo_ir_loc, orbsym, idx2);
                sym_wv(w0, v0, wtmp, fvo, vooo, cache[3], t1T, t2T, nocc, nvir, b, c, a,
                       nirrep, o_ir_loc, v_ir_loc, oo_ir_loc, orbsym, idx3);
                sym_wv(w0, v0, wtmp, fvo, vooo, cache[4], t1T, t2T, nocc, nvir, c, a, b,
                       nirrep, o_ir_loc, v_ir_loc, oo_ir_loc, orbsym, idx4);
                sym_wv(w0, v0, wtmp, fvo, vooo, cache[5], t1T, t2T, nocc, nvir, c, b, a,
                       nirrep, o_ir_loc, v_ir_loc, oo_ir_loc, orbsym, idx5);
        }
        add_and_permute(z0, w0, v0, nocc);

        double et;
        if (a == c) {
                et = _ccsd_t_get_energy(w0, z0, mo_energy, nocc, a, b, c, 1./6);
        } else if (a == b || b == c) {
                et = _ccsd_t_get_energy(w0, z0, mo_energy, nocc, a, b, c, .5);
        } else {
                et = _ccsd_t_get_energy(w0, z0, mo_energy, nocc, a, b, c, 1.);
        }
        return et;
}

static double eccontract6_aaa(int nocc, int nocc3, int nocc_iact, 
                            int nvir, int nvir_cas, int a, int b, int c,
                            double *mo_energy, double *t1T, double *t2T,
                            int nirrep, int *o_ir_loc, int *v_ir_loc,
                            int *oo_ir_loc, int *orbsym, double *fvo,
                            double *vooo, double *cache1, void **cache,
                            int *permute_idx, double *paaa)
{
        int nooo = nocc * nocc * nocc;
        int *idx0 = permute_idx;
        int *idx1 = idx0 + nooo;
        int *idx2 = idx1 + nooo;
        int *idx3 = idx2 + nooo;
        int *idx4 = idx3 + nooo;
        int *idx5 = idx4 + nooo;
        double *v0 = cache1;
        double *w0 = v0 + nooo;
        double *z0 = w0 + nooo;
        double *wtmp = z0;
        int i;

        for (i = 0; i < nooo; i++) {
                w0[i] = 0;
                v0[i] = 0;
        }

        if (nirrep == 1) {
                get_wv_ec(w0, v0, wtmp, fvo, vooo, cache[0], t1T, t2T, paaa, nocc, nocc3, nocc_iact, nvir, nvir_cas, a, b, c, idx0);
                get_wv_ec(w0, v0, wtmp, fvo, vooo, cache[1], t1T, t2T, paaa, nocc, nocc3, nocc_iact, nvir, nvir_cas, a, c, b, idx1);
                get_wv_ec(w0, v0, wtmp, fvo, vooo, cache[2], t1T, t2T, paaa, nocc, nocc3, nocc_iact, nvir, nvir_cas, b, a, c, idx2);
                get_wv_ec(w0, v0, wtmp, fvo, vooo, cache[3], t1T, t2T, paaa, nocc, nocc3, nocc_iact, nvir, nvir_cas, b, c, a, idx3);
                get_wv_ec(w0, v0, wtmp, fvo, vooo, cache[4], t1T, t2T, paaa, nocc, nocc3, nocc_iact, nvir, nvir_cas, c, a, b, idx4);
                get_wv_ec(w0, v0, wtmp, fvo, vooo, cache[5], t1T, t2T, paaa, nocc, nocc3, nocc_iact, nvir, nvir_cas, c, b, a, idx5);

//                get_wv(w0, v0, wtmp, fvo, vooo, cache[0], t1T, t2T, nocc, nvir, a, b, c, idx0);
//                get_wv(w0, v0, wtmp, fvo, vooo, cache[1], t1T, t2T, nocc, nvir, a, c, b, idx1);
//                get_wv(w0, v0, wtmp, fvo, vooo, cache[2], t1T, t2T, nocc, nvir, b, a, c, idx2);
//                get_wv(w0, v0, wtmp, fvo, vooo, cache[3], t1T, t2T, nocc, nvir, b, c, a, idx3);
//                get_wv(w0, v0, wtmp, fvo, vooo, cache[4], t1T, t2T, nocc, nvir, c, a, b, idx4);
//                get_wv(w0, v0, wtmp, fvo, vooo, cache[5], t1T, t2T, nocc, nvir, c, b, a, idx5);
        } else {
                printf("Not implemented yet");
                sym_wv(w0, v0, wtmp, fvo, vooo, cache[0], t1T, t2T, nocc, nvir, a, b, c,
                       nirrep, o_ir_loc, v_ir_loc, oo_ir_loc, orbsym, idx0);
                sym_wv(w0, v0, wtmp, fvo, vooo, cache[1], t1T, t2T, nocc, nvir, a, c, b,
                       nirrep, o_ir_loc, v_ir_loc, oo_ir_loc, orbsym, idx1);
                sym_wv(w0, v0, wtmp, fvo, vooo, cache[2], t1T, t2T, nocc, nvir, b, a, c,
                       nirrep, o_ir_loc, v_ir_loc, oo_ir_loc, orbsym, idx2);
                sym_wv(w0, v0, wtmp, fvo, vooo, cache[3], t1T, t2T, nocc, nvir, b, c, a,
                       nirrep, o_ir_loc, v_ir_loc, oo_ir_loc, orbsym, idx3);
                sym_wv(w0, v0, wtmp, fvo, vooo, cache[4], t1T, t2T, nocc, nvir, c, a, b,
                       nirrep, o_ir_loc, v_ir_loc, oo_ir_loc, orbsym, idx4);
                sym_wv(w0, v0, wtmp, fvo, vooo, cache[5], t1T, t2T, nocc, nvir, c, b, a,
                       nirrep, o_ir_loc, v_ir_loc, oo_ir_loc, orbsym, idx5);
        }
        add_and_permute(z0, w0, v0, nocc);

        double et;
        if (a == c) {
                et = _ccsd_t_get_energy(w0, z0, mo_energy, nocc, a, b, c, 1./6);
        } else if (a == b || b == c) {
                et = _ccsd_t_get_energy(w0, z0, mo_energy, nocc, a, b, c, .5);
        } else {
                et = _ccsd_t_get_energy(w0, z0, mo_energy, nocc, a, b, c, 1.);
        }
        return et;
}

static void permute(double *out, double *v, int n)
{
        int nn = n * n;
        int nnn = nn * n;
        int i, j, k;

        for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
        for (k = 0; k < n; k++) {
                out[i*nn+j*n+k] = v[i*nn+j*n+k] * 4
                                + v[j*nn+k*n+i]
                                + v[k*nn+i*n+j]
                                - v[k*nn+j*n+i] * 2
                                - v[i*nn+k*n+j] * 2
                                - v[j*nn+i*n+k] * 2;
        } } }
}

double _rccsd_t_get_denom(double *w, double *v, double *mo_energy, int nocc,
                          int a, int b, int c, double fac)
{
        int i, j, k, n;
        double abc = mo_energy[nocc+a] + mo_energy[nocc+b] + mo_energy[nocc+c];
        double dn = 0;

        for (n = 0, i = 0; i < nocc; i++) {
        for (j = 0; j < nocc; j++) {
        for (k = 0; k < nocc; k++, n++) {
                dn += fac * w[n] * v[n] / (mo_energy[i] + mo_energy[j] + mo_energy[k] - abc);
        } } }
        return dn;
}

static void ecrcontract6_aaa(double et, double dn, int nocc, int nocc3, int nocc_iact, 
                      int nvir, int nvir_cas, int a, int b, int c,
                      double *mo_energy, double *t1T, double *t2T,
                      int nirrep, int *o_ir_loc, int *v_ir_loc,
                      int *oo_ir_loc, int *orbsym, double *fvo,
                      double *vooo, double *cache1, void **cache,
                      int *permute_idx, double *paaa)
{
        int nooo = nocc * nocc * nocc;
        int *idx0 = permute_idx;
        int *idx1 = idx0 + nooo;
        int *idx2 = idx1 + nooo;
        int *idx3 = idx2 + nooo;
        int *idx4 = idx3 + nooo;
        int *idx5 = idx4 + nooo;
        double *v0 = cache1;
        double *w0 = v0 + nooo;
        double *z0 = w0 + nooo;
        double *y0 = z0 + nooo;
        double *wtmp = y0;
        int i;

        for (i = 0; i < nooo; i++) {
                w0[i] = 0;
                v0[i] = 0;
                y0[i] = 0;
        }

        if (nirrep == 1) {
                get_wv_ecr(w0, v0, y0, wtmp, fvo, vooo, cache[0], t1T, t2T, paaa, nocc, nocc3, nocc_iact, nvir, nvir_cas, a, b, c, idx0);
                get_wv_ecr(w0, v0, y0, wtmp, fvo, vooo, cache[1], t1T, t2T, paaa, nocc, nocc3, nocc_iact, nvir, nvir_cas, a, c, b, idx1);
                get_wv_ecr(w0, v0, y0, wtmp, fvo, vooo, cache[2], t1T, t2T, paaa, nocc, nocc3, nocc_iact, nvir, nvir_cas, b, a, c, idx2);
                get_wv_ecr(w0, v0, y0, wtmp, fvo, vooo, cache[3], t1T, t2T, paaa, nocc, nocc3, nocc_iact, nvir, nvir_cas, b, c, a, idx3);
                get_wv_ecr(w0, v0, y0, wtmp, fvo, vooo, cache[4], t1T, t2T, paaa, nocc, nocc3, nocc_iact, nvir, nvir_cas, c, a, b, idx4);
                get_wv_ecr(w0, v0, y0, wtmp, fvo, vooo, cache[5], t1T, t2T, paaa, nocc, nocc3, nocc_iact, nvir, nvir_cas, c, b, a, idx5);

        } else {
                printf("Not implemented yet");
                sym_wv(w0, v0, wtmp, fvo, vooo, cache[0], t1T, t2T, nocc, nvir, a, b, c,
                       nirrep, o_ir_loc, v_ir_loc, oo_ir_loc, orbsym, idx0);
                sym_wv(w0, v0, wtmp, fvo, vooo, cache[1], t1T, t2T, nocc, nvir, a, c, b,
                       nirrep, o_ir_loc, v_ir_loc, oo_ir_loc, orbsym, idx1);
                sym_wv(w0, v0, wtmp, fvo, vooo, cache[2], t1T, t2T, nocc, nvir, b, a, c,
                       nirrep, o_ir_loc, v_ir_loc, oo_ir_loc, orbsym, idx2);
                sym_wv(w0, v0, wtmp, fvo, vooo, cache[3], t1T, t2T, nocc, nvir, b, c, a,
                       nirrep, o_ir_loc, v_ir_loc, oo_ir_loc, orbsym, idx3);
                sym_wv(w0, v0, wtmp, fvo, vooo, cache[4], t1T, t2T, nocc, nvir, c, a, b,
                       nirrep, o_ir_loc, v_ir_loc, oo_ir_loc, orbsym, idx4);
                sym_wv(w0, v0, wtmp, fvo, vooo, cache[5], t1T, t2T, nocc, nvir, c, b, a,
                       nirrep, o_ir_loc, v_ir_loc, oo_ir_loc, orbsym, idx5);
        }
        add_and_permute(z0, w0, v0, nocc);

        if (a == c) {
                et = _ccsd_t_get_energy(w0, z0, mo_energy, nocc, a, b, c, 1./6);
        } else if (a == b || b == c) {
                et = _ccsd_t_get_energy(w0, z0, mo_energy, nocc, a, b, c, .5);
        } else {
                et = _ccsd_t_get_energy(w0, z0, mo_energy, nocc, a, b, c, 1.);
        }

        for (i = 0; i < nooo; i++) {
                w0[i] = 0;
        }
        permute(w0, y0, nocc);
        if (a == c) {
                dn = _rccsd_t_get_denom(w0, z0, mo_energy, nocc, a, b, c, 1./6);
        } else if (a == b || b == c) {
                dn = _rccsd_t_get_denom(w0, z0, mo_energy, nocc, a, b, c, .5);
        } else {
                dn = _rccsd_t_get_denom(w0, z0, mo_energy, nocc, a, b, c, 1.);
        }
}

void CCuccsd_t_aaa(double complex *e_tot,
                   double *mo_energy, double *t1T, double *t2T,
                   double *vooo, double *fvo,
                   int nocc, int nvir, int a0, int a1, int b0, int b1,
                   int nirrep, int *o_ir_loc, int *v_ir_loc,
                   int *oo_ir_loc, int *orbsym,
                   double *cache_row_a, double *cache_col_a,
                   double *cache_row_b, double *cache_col_b)
{
        int da = a1 - a0;
        int db = b1 - b0;
        CacheJob *jobs = malloc(sizeof(CacheJob) * da*db*b1);
        size_t njobs = _ccsd_t_gen_jobs(jobs, nocc, nvir, a0, a1, b0, b1,
                                        cache_row_a, cache_col_a,
                                        cache_row_b, cache_col_b, sizeof(double));
        double fvohalf[nvir*nocc];
        int i;
        for (i = 0; i < nvir*nocc; i++) {
                fvohalf[i] = fvo[i] * .5;
        }

        int *permute_idx = malloc(sizeof(int) * nocc*nocc*nocc * 6);
        _make_permute_indices(permute_idx, nocc);

#pragma omp parallel default(none) \
        shared(njobs, nocc, nvir, mo_energy, t1T, t2T, nirrep, o_ir_loc, \
               v_ir_loc, oo_ir_loc, orbsym, vooo, fvohalf, jobs, e_tot, \
               permute_idx)
{
        int a, b, c;
        size_t k;
        double *cache1 = malloc(sizeof(double) * (nocc*nocc*nocc*3+2));
        double e = 0;
#pragma omp for schedule (dynamic, 4)
        for (k = 0; k < njobs; k++) {
                a = jobs[k].a;
                b = jobs[k].b;
                c = jobs[k].c;
                e += contract6_aaa(nocc, nvir, a, b, c, mo_energy, t1T, t2T,
                                   nirrep, o_ir_loc, v_ir_loc, oo_ir_loc, orbsym,
                                   fvohalf, vooo, cache1, jobs[k].cache,
                                   permute_idx);
        }
        free(cache1);
#pragma omp critical
        *e_tot += e;
}
        free(permute_idx);
}


void CCecccsd_t_aaa(double complex *e_tot,
                   double *mo_energy, double *t1T, double *t2T,
                   double *vooo, double *fvo, double *paaa,
                   int nocc, int nvir, const int nocc_iact, const int nvir_cas, const int nocc3,
                   int a0, int a1, int b0, int b1,
                   int nirrep, int *o_ir_loc, int *v_ir_loc,
                   int *oo_ir_loc, int *orbsym,
                   double *cache_row_a, double *cache_col_a,
                   double *cache_row_b, double *cache_col_b)
{
        int da = a1 - a0;
        int db = b1 - b0;
        CacheJob *jobs = malloc(sizeof(CacheJob) * da*db*b1);
        size_t njobs = _ccsd_t_gen_jobs(jobs, nocc, nvir, a0, a1, b0, b1,
                                        cache_row_a, cache_col_a,
                                        cache_row_b, cache_col_b, sizeof(double));
        double fvohalf[nvir*nocc];
        int i;
        for (i = 0; i < nvir*nocc; i++) {
                fvohalf[i] = fvo[i] * .5;
        }

        int *permute_idx = malloc(sizeof(int) * nocc*nocc*nocc * 6);
        _make_permute_indices(permute_idx, nocc);

#pragma omp parallel default(none) \
        shared(njobs, nocc, nvir, mo_energy, t1T, t2T, nirrep, o_ir_loc, \
               v_ir_loc, oo_ir_loc, orbsym, vooo, fvohalf, jobs, e_tot, \
               permute_idx, paaa)
{
        int a, b, c;
        size_t k;
        double *cache1 = malloc(sizeof(double) * (nocc*nocc*nocc*3+2));
        double e = 0;
#pragma omp for schedule (dynamic, 4)
        for (k = 0; k < njobs; k++) {
                a = jobs[k].a;
                b = jobs[k].b;
                c = jobs[k].c;
                e += eccontract6_aaa(nocc, nocc3, nocc_iact, nvir, nvir_cas, a, b, c, mo_energy, t1T, t2T,
                                   nirrep, o_ir_loc, v_ir_loc, oo_ir_loc, orbsym,
                                   fvohalf, vooo, cache1, jobs[k].cache,
                                   permute_idx, paaa);
        }
        free(cache1);
#pragma omp critical
        *e_tot += e;
}
        free(permute_idx);
}

void CCecrccsd_t_aaa(double complex *e_tot, double complex *denom,
                   double *mo_energy, double *t1T, double *t2T,
                   double *vooo, double *fvo, double *paaa,
                   int nocc, int nvir, const int nocc_iact, const int nvir_cas, const int nocc3,
                   int a0, int a1, int b0, int b1,
                   int nirrep, int *o_ir_loc, int *v_ir_loc,
                   int *oo_ir_loc, int *orbsym,
                   double *cache_row_a, double *cache_col_a,
                   double *cache_row_b, double *cache_col_b)
{
        int da = a1 - a0;
        int db = b1 - b0;
        CacheJob *jobs = malloc(sizeof(CacheJob) * da*db*b1);
        size_t njobs = _ccsd_t_gen_jobs(jobs, nocc, nvir, a0, a1, b0, b1,
                                        cache_row_a, cache_col_a,
                                        cache_row_b, cache_col_b, sizeof(double));
        double fvohalf[nvir*nocc];
        int i;
        for (i = 0; i < nvir*nocc; i++) {
                fvohalf[i] = fvo[i] * .5;
        }

        int *permute_idx = malloc(sizeof(int) * nocc*nocc*nocc * 6);
        _make_permute_indices(permute_idx, nocc);

#pragma omp parallel default(none) \
        shared(njobs, nocc, nvir, mo_energy, t1T, t2T, nirrep, o_ir_loc, \
               v_ir_loc, oo_ir_loc, orbsym, vooo, fvohalf, jobs, e_tot, \
               denom, permute_idx, paaa)
{
        int a, b, c;
        size_t k;
        double *cache1 = malloc(sizeof(double) * (nocc*nocc*nocc*4+2));
        double e = 0;
        double d = 0;
        double etmp, dtmp;
#pragma omp for schedule (dynamic, 4)
        for (k = 0; k < njobs; k++) {
                a = jobs[k].a;
                b = jobs[k].b;
                c = jobs[k].c;
                ecrcontract6_aaa(etmp, dtmp, nocc, nocc3, nocc_iact, nvir, 
                                   nvir_cas, a, b, c, mo_energy, t1T, t2T,
                                   nirrep, o_ir_loc, v_ir_loc, oo_ir_loc, orbsym,
                                   fvohalf, vooo, cache1, jobs[k].cache,
                                   permute_idx, paaa);
                e += etmp;
                d += dtmp;
        }
        free(cache1);
#pragma omp critical
        *e_tot += e;
        *denom += d;
}
        free(permute_idx);
}
/*************************************************
 *
 * UCCSD(T) beta-alpha-alpha
 *
 *************************************************/
static void get_wv_baa(double *w, double *v, double **vs_ts, double **cache,
                       int nocca, int noccb, int nvira, int nvirb,
                       int a, int b, int c)
{
        double *fvo = vs_ts[2];
        double *fVO = vs_ts[3];
        double *vooo = vs_ts[4];
        double *vOoO = vs_ts[5];
        double *VoOo = vs_ts[6];
        double *t1aT = vs_ts[7];
        double *t1bT = vs_ts[8];
        double *t2aaT = vs_ts[9];
        double *t2abT = vs_ts[10];
        double *vvop = cache[0];
        double *vVoP = cache[1];
        double *VvOp = cache[2];
        const double D0 = 0;
        const double D1 = 1;
        const double D2 = 2;
        const char TRANS_T = 'T';
        const char TRANS_N = 'N';
        const int nmoa = nocca + nvira;
        const int nmob = noccb + nvirb;
        const int noo = nocca * nocca;
        const int nOo = noccb * nocca;
        const size_t nooo = nocca * noo;
        const size_t noOo = nocca * nOo;
        const size_t nOoO = noccb * nOo;
        const size_t nvoo = nvira * noo;
        const int nVoO = nvirb * nOo;
        int i, j, k, n;

/*
 * t2aaT = t2aa.transpose(2,3,0,1)
 * w  = numpy.einsum('ejI,ke->Ijk', t2abT[:,a], vvov) * 2
 * w += numpy.einsum('EjI,kE->Ijk', t2abT[b,:], vVoV) * 2
 * w += numpy.einsum('mj,mIk->Ijk', t2aaT[b,c], VoOo[a,:])
 * w += numpy.einsum('kM,MjI->Ijk', t2abT[b,a], vOoO[c,:]) * 2
 * w += numpy.einsum('ejk,Ie->Ijk', t2aaT[b,:], VvOv)
 * w += numpy.einsum('mI,mjk->Ijk', t2abT[b,a], vooo[c,:]) * 2
 * v  = numpy.einsum('kj,I->Ijk', vvoo, t1bT[a])
 * v += numpy.einsum('Ik,j->Ijk', VvOo, t1aT[b]) * 2
 * v += numpy.einsum('jk,I->Ijk', t2aaT[b,c], fVO[a]) * .5
 * v += numpy.einsum('kI,j->Ijk', t2abT[c,a], fvo[b]) * 2
 * v += w
 */
        dgemm_(&TRANS_T, &TRANS_T, &nocca, &nOo, &nvira,
               &D2, vvop+nocca, &nmoa, t2abT+a*nOo, &nVoO,
               &D0, v, &nocca);
        dgemm_(&TRANS_T, &TRANS_T, &nocca, &nOo, &nvirb,
               &D2, vVoP+noccb, &nmob, t2abT+b*(size_t)nVoO, &nOo,
               &D1, v, &nocca);
        dgemm_(&TRANS_N, &TRANS_T, &nOo, &nocca, &nocca,
               &D1, VoOo+a*noOo, &nOo, t2aaT+b*nvoo+c*noo, &nocca,
               &D1, v, &nOo);
        dgemm_(&TRANS_T, &TRANS_T, &nocca, &nOo, &noccb,
               &D2, t2abT+b*(size_t)nVoO+a*nOo, &noccb, vOoO+c*nOoO, &nOo,
               &D1, v, &nocca);
        for (n = 0, i = 0; i < noccb; i++) {
        for (j = 0; j < nocca; j++) {
        for (k = 0; k < nocca; k++, n++) {
                w[n] = v[j*nOo+i*nocca+k];
        } } }
        dgemm_(&TRANS_N, &TRANS_N, &noo, &noccb, &nvira,
               &D1, t2aaT+b*nvoo, &noo, VvOp+nocca, &nmoa,
               &D1, w, &noo);
        dgemm_(&TRANS_N, &TRANS_T, &noo, &noccb, &nocca,
               &D2, vooo+c*nooo, &noo, t2abT+b*(size_t)nVoO+a*nOo, &noccb,
               &D1, w, &noo);

        double t1aT2[nocca];
        double fvo2[nocca];
        double fVOhalf[noccb];
        for (i = 0; i < nocca; i++) {
                t1aT2[i] = t1aT[b*nocca+i] * 2;
                fvo2[i] = fvo[b*nocca+i] * 2;
        }
        for (i = 0; i < noccb; i++) {
                fVOhalf[i] = fVO[a*noccb+i] * .5;
        }
        double *pt2aaT = t2aaT + b * nvoo + c * noo;
        double *pt2abT = t2abT + (c*nvirb+a) * nOo;
        for (n = 0, i = 0; i < noccb; i++) {
        for (j = 0; j < nocca; j++) {
        for (k = 0; k < nocca; k++, n++) {
                v[n] = (w[n] + vvop[k*nmoa+j] * t1bT[a*noccb+i]
                        + VvOp[i*nmoa+k] * t1aT2[j]
                        + pt2aaT[j*nocca+k] * fVOhalf[i]
                        + pt2abT[k*noccb+i] * fvo2[j]);
        } } }
}

//static void get_wv_baa_ec(double *w, double *v, double **vs_ts, double **cache,
//                       int nocca, int noccb, int nvira, int nvirb,
//                       int nocc_cas, int nvir_cas, int nocc_iact, int nocc2,
//                       int a, int b, int c)
//{
//        double *fvo = vs_ts[2];
//        double *fVO = vs_ts[3];
//        double *vooo = vs_ts[4];
//        double *vOoO = vs_ts[5];
//        double *VoOo = vs_ts[6];
//        double *t1aT = vs_ts[7];
//        double *t1bT = vs_ts[8];
//        double *t2aaT = vs_ts[9];
//        double *t2abT = vs_ts[10];
//        double *pbaa  = vs_ts[11];
//        double *vvop = cache[0];
//        double *vVoP = cache[1];
//        double *VvOp = cache[2];
//        const double D0 = 0;
//        const double D1 = 1;
//        const double D2 = 2;
//        const char TRANS_T = 'T';
//        const char TRANS_N = 'N';
//        const int nmoa = nocca + nvira;
//        const int nmob = noccb + nvirb;
//        const int noo = nocca * nocca;
//        const int nOo = noccb * nocca;
//        const size_t nooo = nocca * noo;
//        const size_t noOo = nocca * nOo;
//        const size_t nOoO = noccb * nOo;
//        const size_t nvoo = nvira * noo;
//        const int nVoO = nvirb * nOo;
//        int i, j, k, n;
//
///*
// * t2aaT = t2aa.transpose(2,3,0,1)
// * w  = numpy.einsum('ejI,ke->Ijk', t2abT[:,a], vvov) * 2
// * w += numpy.einsum('EjI,kE->Ijk', t2abT[b,:], vVoV) * 2
// * w += numpy.einsum('mj,mIk->Ijk', t2aaT[b,c], VoOo[a,:])
// * w += numpy.einsum('kM,MjI->Ijk', t2abT[b,a], vOoO[c,:]) * 2
// * w += numpy.einsum('ejk,Ie->Ijk', t2aaT[b,:], VvOv)
// * w += numpy.einsum('mI,mjk->Ijk', t2abT[b,a], vooo[c,:]) * 2
// * v  = numpy.einsum('kj,I->Ijk', vvoo, t1bT[a])
// * v += numpy.einsum('Ik,j->Ijk', VvOo, t1aT[b]) * 2
// * v += numpy.einsum('jk,I->Ijk', t2aaT[b,c], fVO[a]) * .5
// * v += numpy.einsum('kI,j->Ijk', t2abT[c,a], fvo[b]) * 2
// * v += w
// */
//        dgemm_(&TRANS_T, &TRANS_T, &nocca, &nOo, &nvira,
//               &D2, vvop+nocca, &nmoa, t2abT+a*nOo, &nVoO,
//               &D0, v, &nocca);
//        dgemm_(&TRANS_T, &TRANS_T, &nocca, &nOo, &nvirb,
//               &D2, vVoP+noccb, &nmob, t2abT+b*(size_t)nVoO, &nOo,
//               &D1, v, &nocca);
//        dgemm_(&TRANS_N, &TRANS_T, &nOo, &nocca, &nocca,
//               &D1, VoOo+a*noOo, &nOo, t2aaT+b*nvoo+c*noo, &nocca,
//               &D1, v, &nOo);
//        dgemm_(&TRANS_T, &TRANS_T, &nocca, &nOo, &noccb,
//               &D2, t2abT+b*(size_t)nVoO+a*nOo, &noccb, vOoO+c*nOoO, &nOo,
//               &D1, v, &nocca);
//        for (n = 0, i = 0; i < noccb; i++) {
//        for (j = 0; j < nocca; j++) {
//        for (k = 0; k < nocca; k++, n++) {
//                w[n] = v[j*nOo+i*nocca+k] * pbaa[T(i-nocc_iact, j-nocc_iact, k-nocc_iact, a, b, c, nocca, nvira)];
//        } } }
//        dgemm_(&TRANS_N, &TRANS_N, &noo, &noccb, &nvira,
//               &D1, t2aaT+b*nvoo, &noo, VvOp+nocca, &nmoa,
//               &D1, w, &noo);
//        dgemm_(&TRANS_N, &TRANS_T, &noo, &noccb, &nocca,
//               &D2, vooo+c*nooo, &noo, t2abT+b*(size_t)nVoO+a*nOo, &noccb,
//               &D1, w, &noo);
//
//        double t1aT2[nocca];
//        double fvo2[nocca];
//        double fVOhalf[noccb];
//        for (i = 0; i < nocca; i++) {
//                t1aT2[i] = t1aT[b*nocca+i] * 2;
//                fvo2[i] = fvo[b*nocca+i] * 2;
//        }
//        for (i = 0; i < noccb; i++) {
//                fVOhalf[i] = fVO[a*noccb+i] * .5;
//        }
//        double *pt2aaT = t2aaT + b * nvoo + c * noo;
//        double *pt2abT = t2abT + (c*nvirb+a) * nOo;
//        for (n = 0, i = 0; i < noccb; i++) {
//        for (j = 0; j < nocca; j++) {
//        for (k = 0; k < nocca; k++, n++) {
//                v[n] = (w[n] + vvop[k*nmoa+j] * t1bT[a*noccb+i]
//                        + VvOp[i*nmoa+k] * t1aT2[j]
//                        + pt2aaT[j*nocca+k] * fVOhalf[i]
//                        + pt2abT[k*noccb+i] * fvo2[j]) * pbaa[T(i-nocc_iact, j-nocc_iact, k-nocc_iact, a, b, c, nocca, nvira)];
//        } } }
//}

// off diag version
static void get_wv_baa_ec(double *w, double *v, double **vs_ts, double **cache,
                       int nocca, int noccb, int nvira, int nvirb,
                       int nocc_cas, int nvir_cas, int nocc_iact, int nocc2,
                       int a, int b, int c)
{
        double *fvo = vs_ts[2];
        double *fVO = vs_ts[3];
        double *vooo = vs_ts[4];
        double *vOoO = vs_ts[5];
        double *VoOo = vs_ts[6];
        double *t1aT = vs_ts[7];
        double *t1bT = vs_ts[8];
        double *t2aaT = vs_ts[9];
        double *t2abT = vs_ts[10];
        double *pbaa  = vs_ts[11];
        double *vvop = cache[0];
        double *vVoP = cache[1];
        double *VvOp = cache[2];
        const double D0 = 0;
        const double D1 = 1;
        const double D2 = 2;
        const char TRANS_T = 'T';
        const char TRANS_N = 'N';
        const int nmoa = nocca + nvira;
        const int nmob = noccb + nvirb;
        const int noo = nocca * nocca;
        const int nOo = noccb * nocca;
        const size_t nooo = nocca * noo;
        const size_t noOo = nocca * nOo;
        const size_t nOoO = noccb * nOo;
        const size_t nvoo = nvira * noo;
        const int nVoO = nvirb * nOo;
        int i, j, k, n;

        int it, jt, kt, at, bt, ct;
        double exclude = 1.0;
        int arr[2] = {b, c}; 
        qsort(arr, 2, sizeof(int), comparator);
        at = a;
        bt = arr[0];
        ct = arr[1];


/*
 * t2aaT = t2aa.transpose(2,3,0,1)
 * w  = numpy.einsum('ejI,ke->Ijk', t2abT[:,a], vvov) * 2
 * w += numpy.einsum('EjI,kE->Ijk', t2abT[b,:], vVoV) * 2
 * w += numpy.einsum('mj,mIk->Ijk', t2aaT[b,c], VoOo[a,:])
 * w += numpy.einsum('kM,MjI->Ijk', t2abT[b,a], vOoO[c,:]) * 2
 * w += numpy.einsum('ejk,Ie->Ijk', t2aaT[b,:], VvOv)
 * w += numpy.einsum('mI,mjk->Ijk', t2abT[b,a], vooo[c,:]) * 2
 * v  = numpy.einsum('kj,I->Ijk', vvoo, t1bT[a])
 * v += numpy.einsum('Ik,j->Ijk', VvOo, t1aT[b]) * 2
 * v += numpy.einsum('jk,I->Ijk', t2aaT[b,c], fVO[a]) * .5
 * v += numpy.einsum('kI,j->Ijk', t2abT[c,a], fvo[b]) * 2
 * v += w
 */
        dgemm_(&TRANS_T, &TRANS_T, &nocca, &nOo, &nvira,
               &D2, vvop+nocca, &nmoa, t2abT+a*nOo, &nVoO,
               &D0, v, &nocca);
        dgemm_(&TRANS_T, &TRANS_T, &nocca, &nOo, &nvirb,
               &D2, vVoP+noccb, &nmob, t2abT+b*(size_t)nVoO, &nOo,
               &D1, v, &nocca);
        dgemm_(&TRANS_N, &TRANS_T, &nOo, &nocca, &nocca,
               &D1, VoOo+a*noOo, &nOo, t2aaT+b*nvoo+c*noo, &nocca,
               &D1, v, &nOo);
        dgemm_(&TRANS_T, &TRANS_T, &nocca, &nOo, &noccb,
               &D2, t2abT+b*(size_t)nVoO+a*nOo, &noccb, vOoO+c*nOoO, &nOo,
               &D1, v, &nocca);

        for (n = 0, i = 0; i < noccb; i++) {
        for (j = 0; j < nocca; j++) {
        for (k = 0; k < nocca; k++, n++) {
                arr[0]=j;
                arr[1]=k;
                qsort(arr, 2, sizeof(int), comparator);
                it = i-nocc_iact;
                jt = arr[0]-nocc_iact;
                kt = arr[1]-nocc_iact;
                if ( it >=0 && jt >= 0 && kt >= 0
                  && jt != kt && bt != ct 
                  && at < nvir_cas && bt < nvir_cas && ct < nvir_cas ) 
                    exclude = pbaa[SDc(it, jt, kt, at, bt, ct, nocc_cas, nvir_cas, nocc2)];
                else
                    exclude = 1.0;
                //printf("%d %d %d %d %d %d, exclude: %f\n",i,j,k,a,b,c,exclude);
                assert(exclude == 1.0 || exclude == 0.0);

                w[n] = v[j*nOo+i*nocca+k] * exclude;
        } } }
        dgemm_(&TRANS_N, &TRANS_N, &noo, &noccb, &nvira,
               &D1, t2aaT+b*nvoo, &noo, VvOp+nocca, &nmoa,
               &D1, w, &noo);
        dgemm_(&TRANS_N, &TRANS_T, &noo, &noccb, &nocca,
               &D2, vooo+c*nooo, &noo, t2abT+b*(size_t)nVoO+a*nOo, &noccb,
               &D1, w, &noo);

        double t1aT2[nocca];
        double fvo2[nocca];
        double fVOhalf[noccb];
        for (i = 0; i < nocca; i++) {
                t1aT2[i] = t1aT[b*nocca+i] * 2;
                fvo2[i]  = fvo[b*nocca+i] * 2;
        }
        for (i = 0; i < noccb; i++) {
                fVOhalf[i] = fVO[a*noccb+i] * .5;
        }
        double *pt2aaT = t2aaT + b * nvoo + c * noo;
        double *pt2abT = t2abT + (c*nvirb+a) * nOo;
        for (n = 0, i = 0; i < noccb; i++) {
        for (j = 0; j < nocca; j++) {
        for (k = 0; k < nocca; k++, n++) {
                arr[0]=j;
                arr[1]=k;
                qsort(arr, 2, sizeof(int), comparator);
                it = i-nocc_iact;
                jt = arr[0]-nocc_iact;
                kt = arr[1]-nocc_iact;

                if ( it >=0 && jt >= 0 && kt >= 0
                  && jt != kt && bt != ct 
                  && at < nvir_cas && bt < nvir_cas && ct < nvir_cas ) 
                    exclude = pbaa[SDc(it, jt, kt, at, bt, ct, nocc_cas, nvir_cas, nocc2)];
                else
                    exclude = 1.0;
                assert(exclude == 1.0 || exclude == 0.0);

                v[n] = (w[n] + vvop[k*nmoa+j] * t1bT[a*noccb+i]
                        + VvOp[i*nmoa+k] * t1aT2[j]
                        + pt2aaT[j*nocca+k] * fVOhalf[i]
                        + pt2abT[k*noccb+i] * fvo2[j]) * exclude;
        } } }
}

/*
 * w - w.transpose(0,2,1)
 */
static void permute_baa(double *out, double *w, int nocca, int noccb)
{
        int noo = nocca * nocca;
        int n;
        int i, j, k;

        for (n = 0, i = 0; i < noccb; i++) {
        for (j = 0; j < nocca; j++) {
        for (k = 0; k < nocca; k++, n++) {
                out[n] = w[i*noo+j*nocca+k] - w[i*noo+k*nocca+j];
        } } }
}

static double _get_energy_baa_ec(double *z0, double *z1, double *w0, double *w1,
                              double *mo_ea, double *mo_eb, double *pbaa,
                              int nocca, int noccb, int nocc_cas, int nvir_cas, int nocc_iact, int nocc2,
                              int a, int b, int c, double fac)
{
        int noo = nocca * nocca;
        int i, j, k;
        int it, jt, kt, at, bt, ct;
        double abc = mo_eb[noccb+a] + mo_ea[nocca+b] + mo_ea[nocca+c];
        double et = 0;
        double exclude = 1.0;
        int arr[3] = {a, b, c}; 
        qsort(arr, 3, sizeof(int), comparator);
        at = arr[0];
        bt = arr[1];
        ct = arr[2];

        for (i = 0; i < noccb; i++) {
        for (j = 0; j < nocca; j++) {
        for (k = 0; k < nocca; k++) {
                arr[0]=i;
                arr[1]=j;
                arr[2]=k;
                qsort(arr, 3, sizeof(int), comparator);
                it = arr[0]-nocc_iact;
                jt = arr[1]-nocc_iact;
                kt = arr[2]-nocc_iact;
                if ( it != jt && jt != kt ) exclude = pbaa[DSc(it, jt, kt, at, bt, ct, nocc_cas, nvir_cas, nocc2)];

                et += (z0[i*noo+j*nocca+k] + z1[i*noo+k*nocca+j])
                    * (w0[i*noo+j*nocca+k] + w1[i*noo+k*nocca+j])
                    * fac / (mo_eb[i] + mo_ea[j] + mo_ea[k] - abc) * exclude;
        } } }
        return et;
}

static double _get_energy_baa(double *z0, double *z1, double *w0, double *w1,
                              double *mo_ea, double *mo_eb, int nocca, int noccb,
                              int a, int b, int c, double fac)
{
        int noo = nocca * nocca;
        int i, j, k;
        double abc = mo_eb[noccb+a] + mo_ea[nocca+b] + mo_ea[nocca+c];
        double et = 0;

        for (i = 0; i < noccb; i++) {
        for (j = 0; j < nocca; j++) {
        for (k = 0; k < nocca; k++) {
                et += (z0[i*noo+j*nocca+k] + z1[i*noo+k*nocca+j])
                    * (w0[i*noo+j*nocca+k] + w1[i*noo+k*nocca+j])
                    * fac / (mo_eb[i] + mo_ea[j] + mo_ea[k] - abc);
        } } }
        return et;
}

static double contract6_baa(int nocca, int noccb, int nvira, int nvirb,
                            int a, int b, int c,
                            double **vs_ts, void **cache, double *cache1)
{
        int nOoo = noccb * nocca * nocca;
        double *v0 = cache1;
        double *v1 = v0 + nOoo;
        double *w0 = v1 + nOoo;
        double *w1 = w0 + nOoo;
        double *z0 = w1 + nOoo;
        double *z1 = v0;

        get_wv_baa(w0, v0, vs_ts, ((double **)cache)  , nocca, noccb, nvira, nvirb, a, b, c);
        get_wv_baa(w1, v1, vs_ts, ((double **)cache)+3, nocca, noccb, nvira, nvirb, a, c, b);
        permute_baa(z0, v0, nocca, noccb);
        permute_baa(z1, v1, nocca, noccb);

        double *mo_ea = vs_ts[0];
        double *mo_eb = vs_ts[1];
        double et;
        if (b == c) {
                et = _get_energy_baa(z0, z1, w0, w1, mo_ea, mo_eb, nocca, noccb, a, b, c, .5);
        } else {
                et = _get_energy_baa(z0, z1, w0, w1, mo_ea, mo_eb, nocca, noccb, a, b, c, 1.);
        }
        return et;
}

static double eccontract6_baa(int nocca, int noccb, int nvira, int nvirb,
                            int nocc_cas, int nvir_cas, int nocc_iact, int nocc2,
                            int a, int b, int c,
                            double **vs_ts, void **cache, double *cache1)
{
        int nOoo = noccb * nocca * nocca;
        double *v0 = cache1;
        double *v1 = v0 + nOoo;
        double *w0 = v1 + nOoo;
        double *w1 = w0 + nOoo;
        double *z0 = w1 + nOoo;
        double *z1 = v0;

        get_wv_baa_ec(w0, v0, vs_ts, ((double **)cache)  , nocca, noccb, nvira, nvirb, nocc_cas, nvir_cas, nocc_iact, nocc2, a, b, c);
        get_wv_baa_ec(w1, v1, vs_ts, ((double **)cache)+3, nocca, noccb, nvira, nvirb, nocc_cas, nvir_cas, nocc_iact, nocc2, a, c, b);

//        get_wv_baa(w0, v0, vs_ts, ((double **)cache)  , nocca, noccb, nvira, nvirb, a, b, c);
//        get_wv_baa(w1, v1, vs_ts, ((double **)cache)+3, nocca, noccb, nvira, nvirb, a, c, b);

        permute_baa(z0, v0, nocca, noccb);
        permute_baa(z1, v1, nocca, noccb);

        double *mo_ea = vs_ts[0];
        double *mo_eb = vs_ts[1];
        double et;
        if (b == c) {
//                printf("b==c in tabb\n"); 
                et = _get_energy_baa(z0, z1, w0, w1, mo_ea, mo_eb, nocca, noccb, a, b, c, .5);
        } else {
                et = _get_energy_baa(z0, z1, w0, w1, mo_ea, mo_eb, nocca, noccb, a, b, c, 1.);
                //et = _get_energy_baa_ec(z0, z1, w0, w1, mo_ea, mo_eb, pbaa, nocca, noccb,
                //     nocc_cas, nvir_cas, nocc_iact, nocc2, a, b, c, 1.);
        }
        return et;
}

static size_t gen_baa_jobs(CacheJob *jobs,
                           int nocca, int noccb, int nvira, int nvirb,
                           int a0, int a1, int b0, int b1,
                           void *cache_row_a, void *cache_col_a,
                           void *cache_row_b, void *cache_col_b, size_t stride)
{
        size_t nov = nocca * (nocca+nvira) * stride;
        size_t noV = nocca * (noccb+nvirb) * stride;
        size_t nOv = noccb * (nocca+nvira) * stride;
        int da = a1 - a0;
        int db = b1 - b0;
        int a, b, c;

        size_t m = 0;
        for (a = a0; a < a1; a++) {
        for (b = b0; b < b1; b++) {
        for (c = 0; c <= b; c++, m++) {
                jobs[m].a = a;
                jobs[m].b = b;
                jobs[m].c = c;
                if (c < b0) {
                        jobs[m].cache[0] = cache_col_b + nov*(db*(c   )+b-b0);
                } else {
                        jobs[m].cache[0] = cache_row_b + nov*(b1*(c-b0)+b   );
                }
                jobs[m].cache[1] = cache_col_a + noV*(da   *(c   )+a-a0);
                jobs[m].cache[2] = cache_row_a + nOv*(nvira*(a-a0)+c   );
                jobs[m].cache[3] = cache_row_b + nov*(b1   *(b-b0)+c   );
                jobs[m].cache[4] = cache_col_a + noV*(da   *(b   )+a-a0);
                jobs[m].cache[5] = cache_row_a + nOv*(nvira*(a-a0)+b   );
        } } }
        return m;
}

void CCuccsd_t_baa(double complex *e_tot,
                   double *mo_ea, double *mo_eb,
                   double *t1aT, double *t1bT, double *t2aaT, double *t2abT,
                   double *vooo, double *vOoO, double *VoOo,
                   double *fvo, double *fVO,
                   int nocca, int noccb, int nvira, int nvirb,
                   int a0, int a1, int b0, int b1,
                   void *cache_row_a, void *cache_col_a,
                   void *cache_row_b, void *cache_col_b)
{
        int da = a1 - a0;
        int db = b1 - b0;
        CacheJob *jobs = malloc(sizeof(CacheJob) * da*db*b1);
        size_t njobs = gen_baa_jobs(jobs, nocca, noccb, nvira, nvirb,
                                    a0, a1, b0, b1,
                                    cache_row_a, cache_col_a,
                                    cache_row_b, cache_col_b, sizeof(double));
        double *vs_ts[] = {mo_ea, mo_eb, fvo, fVO, vooo, vOoO, VoOo,
                t1aT, t1bT, t2aaT, t2abT};

#pragma omp parallel default(none) \
        shared(njobs, nocca, noccb, nvira, nvirb, vs_ts, jobs, e_tot)
{
        int a, b, c;
        size_t k;
        double *cache1 = malloc(sizeof(double) * (noccb*nocca*nocca*5+1));
        double e = 0;
#pragma omp for schedule (dynamic, 4)
        for (k = 0; k < njobs; k++) {
                a = jobs[k].a;
                b = jobs[k].b;
                c = jobs[k].c;
                e += contract6_baa(nocca, noccb, nvira, nvirb, a, b, c, vs_ts,
                                   jobs[k].cache, cache1);
        }
        free(cache1);
#pragma omp critical
        *e_tot += e;
}
}

void CCecccsd_t_baa(double complex *e_tot,
                   double *mo_ea, double *mo_eb,
                   double *t1aT, double *t1bT, double *t2aaT, double *t2abT,
                   double *vooo, double *vOoO, double *VoOo,
                   double *fvo, double *fVO, double *pbaa,
                   int nocca, int noccb, int nvira, int nvirb,
                   const int nocc_cas, const int nvir_cas, const int nocc_iact, const int nocc2,
                   int a0, int a1, int b0, int b1,
                   void *cache_row_a, void *cache_col_a,
                   void *cache_row_b, void *cache_col_b)
{
        int da = a1 - a0;
        int db = b1 - b0;
        CacheJob *jobs = malloc(sizeof(CacheJob) * da*db*b1);
        size_t njobs = gen_baa_jobs(jobs, nocca, noccb, nvira, nvirb,
                                    a0, a1, b0, b1,
                                    cache_row_a, cache_col_a,
                                    cache_row_b, cache_col_b, sizeof(double));
        double *vs_ts[] = {mo_ea, mo_eb, fvo, fVO, vooo, vOoO, VoOo,
                t1aT, t1bT, t2aaT, t2abT, pbaa};

#pragma omp parallel default(none) \
        shared(njobs, nocca, noccb, nvira, nvirb, vs_ts, jobs, e_tot)
{
        int a, b, c;
        size_t k;
        double *cache1 = malloc(sizeof(double) * (noccb*nocca*nocca*5+1));
        double e = 0;
#pragma omp for schedule (dynamic, 4)
        for (k = 0; k < njobs; k++) {
                a = jobs[k].a;
                b = jobs[k].b;
                c = jobs[k].c;
                e += eccontract6_baa(nocca, noccb, nvira, nvirb,
                                   nocc_cas, nvir_cas, nocc_iact, nocc2,
                                   a, b, c, vs_ts,
                                   jobs[k].cache, cache1);
        }
        free(cache1);
#pragma omp critical
        *e_tot += e;
}
}

/*
 * Complex version of all functions
 */
static void zadd_and_permute(double complex *out, double complex *w,
                             double complex *v, int n)
{
        int nn = n * n;
        int nnn = nn * n;
        int i, j, k;

        for (i = 0; i < nnn; i++) {
                v[i] += w[i];
        }

        for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
        for (k = 0; k < n; k++) {
                out[i*nn+j*n+k] = v[i*nn+j*n+k] + v[j*nn+k*n+i] + v[k*nn+i*n+j]
                                - v[k*nn+j*n+i] - v[i*nn+k*n+j] - v[j*nn+i*n+k];
        } } }
}

static void zget_wv(double complex *w, double complex *v, double complex *cache,
                    double complex *fvohalf, double complex *vooo,
                    double complex *vv_op, double complex *t1T, double complex *t2T,
                    int nocc, int nvir, int a, int b, int c, int *idx)
{
        const double complex D0 = 0;
        const double complex D1 = 1;
        const double complex DN1 =-1;
        const char TRANS_N = 'N';
        const char TRANS_T = 'T';
        const int nmo = nocc + nvir;
        const int noo = nocc * nocc;
        const size_t nooo = nocc * noo;
        const size_t nvoo = nvir * noo;
        int i, j, k, n;
        double complex *pt2T;

        zgemm_(&TRANS_N, &TRANS_N, &noo, &nocc, &nvir,
               &DN1, t2T+c*nvoo, &noo, vv_op+nocc, &nmo,
               &D0, cache, &noo);
        zgemm_(&TRANS_N, &TRANS_T, &nocc, &noo, &nocc,
               &DN1, t2T+b*nvoo+c*noo, &nocc, vooo+a*nooo, &noo,
               &D1, cache, &nocc);

        pt2T = t2T + a * nvoo + b * noo;
        for (n = 0, i = 0; i < nocc; i++) {
        for (j = 0; j < nocc; j++) {
        for (k = 0; k < nocc; k++, n++) {
                w[idx[n]] += cache[n];
                v[idx[n]] +=(vv_op[i*nmo+j] * t1T[c*nocc+k]
                           + pt2T[i*nocc+j] * fvohalf[c*nocc+k]);
        } } }
}

static double complex
zcontract6_aaa(int nocc, int nvir, int a, int b, int c,
               double *mo_energy, double complex *t1T, double complex *t2T,
               int nirrep, int *o_ir_loc, int *v_ir_loc,
               int *oo_ir_loc, int *orbsym, double complex *fvo,
               double complex *vooo, double complex *cache1, void **cache,
               int *permute_idx)
{
        int nooo = nocc * nocc * nocc;
        int *idx0 = permute_idx;
        int *idx1 = idx0 + nooo;
        int *idx2 = idx1 + nooo;
        int *idx3 = idx2 + nooo;
        int *idx4 = idx3 + nooo;
        int *idx5 = idx4 + nooo;
        double complex *v0 = cache1;
        double complex *w0 = v0 + nooo;
        double complex *z0 = w0 + nooo;
        double complex *wtmp = z0;
        int i;

        for (i = 0; i < nooo; i++) {
                w0[i] = 0;
                v0[i] = 0;
        }

        zget_wv(w0, v0, wtmp, fvo, vooo, cache[0], t1T, t2T, nocc, nvir, a, b, c, idx0);
        zget_wv(w0, v0, wtmp, fvo, vooo, cache[1], t1T, t2T, nocc, nvir, a, c, b, idx1);
        zget_wv(w0, v0, wtmp, fvo, vooo, cache[2], t1T, t2T, nocc, nvir, b, a, c, idx2);
        zget_wv(w0, v0, wtmp, fvo, vooo, cache[3], t1T, t2T, nocc, nvir, b, c, a, idx3);
        zget_wv(w0, v0, wtmp, fvo, vooo, cache[4], t1T, t2T, nocc, nvir, c, a, b, idx4);
        zget_wv(w0, v0, wtmp, fvo, vooo, cache[5], t1T, t2T, nocc, nvir, c, b, a, idx5);
        zadd_and_permute(z0, w0, v0, nocc);

        double complex et;
        if (a == c) {
                et = _ccsd_t_zget_energy(w0, z0, mo_energy, nocc, a, b, c, 1./6);
        } else if (a == b || b == c) {
                et = _ccsd_t_zget_energy(w0, z0, mo_energy, nocc, a, b, c, .5);
        } else {
                et = _ccsd_t_zget_energy(w0, z0, mo_energy, nocc, a, b, c, 1.);
        }
        return et;
}

void CCuccsd_t_zaaa(double complex *e_tot,
                    double *mo_energy, double complex *t1T, double complex *t2T,
                    double complex *vooo, double complex *fvo,
                    int nocc, int nvir, int a0, int a1, int b0, int b1,
                    int nirrep, int *o_ir_loc, int *v_ir_loc,
                    int *oo_ir_loc, int *orbsym,
                    void *cache_row_a, void *cache_col_a,
                    void *cache_row_b, void *cache_col_b)
{
        int da = a1 - a0;
        int db = b1 - b0;
        CacheJob *jobs = malloc(sizeof(CacheJob) * da*db*b1);
        size_t njobs = _ccsd_t_gen_jobs(jobs, nocc, nvir, a0, a1, b0, b1,
                                        cache_row_a, cache_col_a,
                                        cache_row_b, cache_col_b,
                                        sizeof(double complex));
        double complex fvohalf[nvir*nocc];
        int i;
        for (i = 0; i < nvir*nocc; i++) {
                fvohalf[i] = fvo[i] * .5;
        }

        int *permute_idx = malloc(sizeof(int) * nocc*nocc*nocc * 6);
        _make_permute_indices(permute_idx, nocc);

#pragma omp parallel default(none) \
        shared(njobs, nocc, nvir, mo_energy, t1T, t2T, nirrep, o_ir_loc, \
               v_ir_loc, oo_ir_loc, orbsym, vooo, fvohalf, jobs, e_tot, \
               permute_idx)
{
        int a, b, c;
        size_t k;
        double complex *cache1 = malloc(sizeof(double complex) *
                                        (nocc*nocc*nocc*3+2));
        double complex e = 0;
#pragma omp for schedule (dynamic, 4)
        for (k = 0; k < njobs; k++) {
                a = jobs[k].a;
                b = jobs[k].b;
                c = jobs[k].c;
                e += zcontract6_aaa(nocc, nvir, a, b, c, mo_energy, t1T, t2T,
                                    nirrep, o_ir_loc, v_ir_loc, oo_ir_loc, orbsym,
                                    fvohalf, vooo, cache1, jobs[k].cache,
                                    permute_idx);
        }
        free(cache1);
#pragma omp critical
        *e_tot += e;
}
        free(permute_idx);
}


/*************************************************
 *
 * UCCSD(T) beta-alpha-alpha
 *
 *************************************************/
static void zget_wv_baa(double complex *w, double complex *v,
                        double complex **vs_ts, double complex **cache,
                        int nocca, int noccb, int nvira, int nvirb,
                        int a, int b, int c)
{
        double complex *fvo = vs_ts[2];
        double complex *fVO = vs_ts[3];
        double complex *vooo = vs_ts[4];
        double complex *vOoO = vs_ts[5];
        double complex *VoOo = vs_ts[6];
        double complex *t1aT = vs_ts[7];
        double complex *t1bT = vs_ts[8];
        double complex *t2aaT = vs_ts[9];
        double complex *t2abT = vs_ts[10];
        double complex *vvop = cache[0];
        double complex *vVoP = cache[1];
        double complex *VvOp = cache[2];
        const double complex D0 = 0;
        const double complex D1 = 1;
        const double complex D2 = 2;
        const char TRANS_T = 'T';
        const char TRANS_N = 'N';
        const int nmoa = nocca + nvira;
        const int nmob = noccb + nvirb;
        const int noo = nocca * nocca;
        const int nOo = noccb * nocca;
        const size_t nooo = nocca * noo;
        const size_t noOo = nocca * nOo;
        const size_t nOoO = noccb * nOo;
        const size_t nvoo = nvira * noo;
        const int nVoO = nvirb * nOo;
        int i, j, k, n;

        zgemm_(&TRANS_T, &TRANS_T, &nocca, &nOo, &nvira,
               &D2, vvop+nocca, &nmoa, t2abT+a*nOo, &nVoO,
               &D0, v, &nocca);
        zgemm_(&TRANS_T, &TRANS_T, &nocca, &nOo, &nvirb,
               &D2, vVoP+noccb, &nmob, t2abT+b*(size_t)nVoO, &nOo,
               &D1, v, &nocca);
        zgemm_(&TRANS_N, &TRANS_T, &nOo, &nocca, &nocca,
               &D1, VoOo+a*noOo, &nOo, t2aaT+b*nvoo+c*noo, &nocca,
               &D1, v, &nOo);
        zgemm_(&TRANS_T, &TRANS_T, &nocca, &nOo, &noccb,
               &D2, t2abT+b*(size_t)nVoO+a*nOo, &noccb, vOoO+c*nOoO, &nOo,
               &D1, v, &nocca);
        for (n = 0, i = 0; i < noccb; i++) {
        for (j = 0; j < nocca; j++) {
        for (k = 0; k < nocca; k++, n++) {
                w[n] = v[j*nOo+i*nocca+k];
        } } }
        zgemm_(&TRANS_N, &TRANS_N, &noo, &noccb, &nvira,
               &D1, t2aaT+b*nvoo, &noo, VvOp+nocca, &nmoa,
               &D1, w, &noo);
        zgemm_(&TRANS_N, &TRANS_T, &noo, &noccb, &nocca,
               &D2, vooo+c*nooo, &noo, t2abT+b*(size_t)nVoO+a*nOo, &noccb,
               &D1, w, &noo);

        double complex t1aT2[nocca];
        double complex fvo2[nocca];
        double complex fVOhalf[noccb];
        for (i = 0; i < nocca; i++) {
                t1aT2[i] = t1aT[b*nocca+i] * 2;
                fvo2[i] = fvo[b*nocca+i] * 2;
        }
        for (i = 0; i < noccb; i++) {
                fVOhalf[i] = fVO[a*noccb+i] * .5;
        }
        double complex *pt2aaT = t2aaT + b * nvoo + c * noo;
        double complex *pt2abT = t2abT + (c*nvirb+a) * nOo;
        for (n = 0, i = 0; i < noccb; i++) {
        for (j = 0; j < nocca; j++) {
        for (k = 0; k < nocca; k++, n++) {
                v[n] = (w[n] + vvop[k*nmoa+j] * t1bT[a*noccb+i]
                        + VvOp[i*nmoa+k] * t1aT2[j]
                        + pt2aaT[j*nocca+k] * fVOhalf[i]
                        + pt2abT[k*noccb+i] * fvo2[j]);
        } } }
}

/*
 * w - w.transpose(0,2,1)
 */
static void zpermute_baa(double complex *out, double complex *w, int nocca, int noccb)
{
        int noo = nocca * nocca;
        int n;
        int i, j, k;

        for (n = 0, i = 0; i < noccb; i++) {
        for (j = 0; j < nocca; j++) {
        for (k = 0; k < nocca; k++, n++) {
                out[n] = w[i*noo+j*nocca+k] - w[i*noo+k*nocca+j];
        } } }
}

static double complex
_zget_energy_baa(double complex *z0, double complex *z1,
                 double complex *w0, double complex *w1,
                 double *mo_ea, double *mo_eb, int nocca, int noccb,
                 int a, int b, int c, double fac)
{
        int noo = nocca * nocca;
        int i, j, k;
        double abc = mo_eb[noccb+a] + mo_ea[nocca+b] + mo_ea[nocca+c];
        double complex et = 0;

        for (i = 0; i < noccb; i++) {
        for (j = 0; j < nocca; j++) {
        for (k = 0; k < nocca; k++) {
                et += conj(z0[i*noo+j*nocca+k] + z1[i*noo+k*nocca+j])
                    * (w0[i*noo+j*nocca+k] + w1[i*noo+k*nocca+j])
                    * (fac / (mo_eb[i] + mo_ea[j] + mo_ea[k] - abc));
        } } }
        return et;
}

static double complex
zcontract6_baa(int nocca, int noccb, int nvira, int nvirb,
               int a, int b, int c,
               double complex **vs_ts, void **cache, double complex *cache1)
{
        int nOoo = noccb * nocca * nocca;
        double complex *v0 = cache1;
        double complex *v1 = v0 + nOoo;
        double complex *w0 = v1 + nOoo;
        double complex *w1 = w0 + nOoo;
        double complex *z0 = w1 + nOoo;
        double complex *z1 = v0;

        zget_wv_baa(w0, v0, vs_ts, ((double complex **)cache)  , nocca, noccb, nvira, nvirb, a, b, c);
        zget_wv_baa(w1, v1, vs_ts, ((double complex **)cache)+3, nocca, noccb, nvira, nvirb, a, c, b);
        zpermute_baa(z0, v0, nocca, noccb);
        zpermute_baa(z1, v1, nocca, noccb);

        double *mo_ea = (double *)vs_ts[0];
        double *mo_eb = (double *)vs_ts[1];
        double complex et;
        if (b == c) {
                et = _zget_energy_baa(z0, z1, w0, w1, mo_ea, mo_eb, nocca, noccb, a, b, c, .5);
        } else {
                et = _zget_energy_baa(z0, z1, w0, w1, mo_ea, mo_eb, nocca, noccb, a, b, c, 1.);
        }
        return et;
}


void CCuccsd_t_zbaa(double complex *e_tot,
                    double *mo_ea, double *mo_eb,
                    double complex *t1aT, double complex *t1bT,
                    double complex *t2aaT, double complex *t2abT,
                    double complex *vooo, double complex *vOoO, double complex *VoOo,
                    double complex *fvo, double complex *fVO,
                    int nocca, int noccb, int nvira, int nvirb,
                    int a0, int a1, int b0, int b1,
                    void *cache_row_a, void *cache_col_a,
                    void *cache_row_b, void *cache_col_b)
{
        int da = a1 - a0;
        int db = b1 - b0;
        CacheJob *jobs = malloc(sizeof(CacheJob) * da*db*b1);
        size_t njobs = gen_baa_jobs(jobs, nocca, noccb, nvira, nvirb,
                                    a0, a1, b0, b1,
                                    cache_row_a, cache_col_a,
                                    cache_row_b, cache_col_b,
                                    sizeof(double complex));
        double complex *vs_ts[] = {(double complex *)mo_ea,
                (double complex *)mo_eb, fvo, fVO, vooo, vOoO, VoOo,
                t1aT, t1bT, t2aaT, t2abT};

#pragma omp parallel default(none) \
        shared(njobs, nocca, noccb, nvira, nvirb, vs_ts, jobs, e_tot)
{
        int a, b, c;
        size_t k;
        double complex *cache1 = malloc(sizeof(double complex) *
                                        (noccb*nocca*nocca*5+1));
        double complex e = 0;
#pragma omp for schedule (dynamic, 4)
        for (k = 0; k < njobs; k++) {
                a = jobs[k].a;
                b = jobs[k].b;
                c = jobs[k].c;
                e += zcontract6_baa(nocca, noccb, nvira, nvirb, a, b, c, vs_ts,
                                   jobs[k].cache, cache1);
        }
        free(cache1);
#pragma omp critical
        *e_tot += e;
}
}

