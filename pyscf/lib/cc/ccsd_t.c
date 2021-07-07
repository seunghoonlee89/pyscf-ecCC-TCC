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
#include <stdint.h>
#include <complex.h>
#include "config.h"
#include "np_helper/np_helper.h"
#include "vhf/fblas.h"
#include <math.h>
#include <string.h>

typedef struct {
        void *cache[6];
        short a;
        short b;
        short c;
        short _padding;
} CacheJob;

/*
 * 4 * w + w.transpose(1,2,0) + w.transpose(2,0,1)
 * - 2 * w.transpose(2,1,0) - 2 * w.transpose(0,2,1)
 * - 2 * w.transpose(1,0,2)
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
                out[i*nn+j*n+k] = v[i*nn+j*n+k] * 4
                                + v[j*nn+k*n+i]
                                + v[k*nn+i*n+j]
                                - v[k*nn+j*n+i] * 2
                                - v[i*nn+k*n+j] * 2
                                - v[j*nn+i*n+k] * 2;
        } } }
}

/*
 * t2T = t2.transpose(2,3,1,0)
 * ov = vv_op[:,nocc:]
 * oo = vv_op[:,:nocc]
 * w = numpy.einsum('if,fjk->ijk', ov, t2T[c])
 * w-= numpy.einsum('ijm,mk->ijk', vooo[a], t2T[c,b])
 * v = numpy.einsum('ij,k->ijk', oo, t1T[c]*.5)
 * v+= numpy.einsum('ij,k->ijk', t2T[b,a], fov[:,c]*.5)
 * v+= w
 */
static void get_wv(double *w, double *v, double *cache,
                   double *fvohalf, double *vooo,
                   double *vv_op, double *t1Thalf, double *t2T,
                   int nocc, int nvir, int a, int b, int c, int *idx)
{
        const double D0 = 0;
        const double D1 = 1;
        const double DN1 =-1;
        const char TRANS_N = 'N';
        const int nmo = nocc + nvir;
        const int noo = nocc * nocc;
        const size_t nooo = nocc * noo;
        const size_t nvoo = nvir * noo;
        int i, j, k, n;
        double *pt2T;

        dgemm_(&TRANS_N, &TRANS_N, &noo, &nocc, &nvir,
               &D1, t2T+c*nvoo, &noo, vv_op+nocc, &nmo,
               &D0, cache, &noo);
        dgemm_(&TRANS_N, &TRANS_N, &nocc, &noo, &nocc,
               &DN1, t2T+c*nvoo+b*noo, &nocc, vooo+a*nooo, &nocc,
               &D1, cache, &nocc);

        pt2T = t2T + b * nvoo + a * noo;
        for (n = 0, i = 0; i < nocc; i++) {
        for (j = 0; j < nocc; j++) {
        for (k = 0; k < nocc; k++, n++) {
                w[idx[n]] += cache[n];
                v[idx[n]] +=(vv_op[i*nmo+j] * t1Thalf[c*nocc+k]
                           + pt2T[i*nocc+j] * fvohalf[c*nocc+k]);
        } } }
}

static void sym_wv(double *w, double *v, double *cache,
                   double *fvohalf, double *vooo,
                   double *vv_op, double *t1Thalf, double *t2T,
                   int nocc, int nvir, int a, int b, int c, int nirrep,
                   int *o_ir_loc, int *v_ir_loc, int *oo_ir_loc, int *orbsym,
                   int *idx)
{
        const double D0 = 0;
        const double D1 = 1;
        const char TRANS_N = 'N';
        const int nmo = nocc + nvir;
        const int noo = nocc * nocc;
        const size_t nooo = nocc * noo;
        const size_t nvoo = nvir * noo;
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
                        w[idx[i*noo+j*nocc+k]] += cache[n];
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

        pt2T = t2T + b * nvoo + a * noo;
        for (n = 0, i = 0; i < nocc; i++) {
        for (j = 0; j < nocc; j++) {
        for (k = 0; k < nocc; k++, n++) {
                v[idx[n]] +=(vv_op[i*nmo+j] * t1Thalf[c*nocc+k]
                           + pt2T[i*nocc+j] * fvohalf[c*nocc+k]);
        } } }
}

double _ccsd_t_get_energy(double *w, double *v, double *mo_energy, int nocc,
                          int a, int b, int c, double fac)
{
        int i, j, k, n;
        double abc = mo_energy[nocc+a] + mo_energy[nocc+b] + mo_energy[nocc+c];
        double et = 0;

        for (n = 0, i = 0; i < nocc; i++) {
        for (j = 0; j < nocc; j++) {
        for (k = 0; k < nocc; k++, n++) {
                et += fac * w[n] * v[n] / (mo_energy[i] + mo_energy[j] + mo_energy[k] - abc);
        } } }
        return et;
}

static double contract6(int nocc, int nvir, int a, int b, int c,
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

size_t _ccsd_t_gen_jobs(CacheJob *jobs, int nocc, int nvir,
                        int a0, int a1, int b0, int b1,
                        void *cache_row_a, void *cache_col_a,
                        void *cache_row_b, void *cache_col_b, size_t stride)
{
        size_t nov = nocc * (nocc+nvir) * stride;
        int da = a1 - a0;
        int db = b1 - b0;
        size_t m, a, b, c;

        if (b1 <= a0) {
                m = 0;
                for (a = a0; a < a1; a++) {
                for (b = b0; b < b1; b++) {
                        for (c = 0; c < b0; c++, m++) {
                                jobs[m].a = a;
                                jobs[m].b = b;
                                jobs[m].c = c;
                                jobs[m].cache[0] = cache_row_a + nov*(a1*(a-a0)+b   );
                                jobs[m].cache[1] = cache_row_a + nov*(a1*(a-a0)+c   );
                                jobs[m].cache[2] = cache_col_a + nov*(da*(b)   +a-a0);
                                jobs[m].cache[3] = cache_row_b + nov*(b1*(b-b0)+c   );
                                jobs[m].cache[4] = cache_col_a + nov*(da*(c)   +a-a0);
                                jobs[m].cache[5] = cache_col_b + nov*(db*(c)   +b-b0);
                        }
                        for (c = b0; c <= b; c++, m++) {
                                jobs[m].a = a;
                                jobs[m].b = b;
                                jobs[m].c = c;
                                jobs[m].cache[0] = cache_row_a + nov*(a1*(a-a0)+b   );
                                jobs[m].cache[1] = cache_row_a + nov*(a1*(a-a0)+c   );
                                jobs[m].cache[2] = cache_col_a + nov*(da*(b)   +a-a0);
                                jobs[m].cache[3] = cache_row_b + nov*(b1*(b-b0)+c   );
                                jobs[m].cache[4] = cache_col_a + nov*(da*(c)   +a-a0);
                                jobs[m].cache[5] = cache_row_b + nov*(b1*(c-b0)+b   );
                        }
                } }
        } else {
                m = 0;
                for (a = a0; a < a1; a++) {
                for (b = a0; b <= a; b++) {
                        for (c = 0; c < a0; c++, m++) {
                                jobs[m].a = a;
                                jobs[m].b = b;
                                jobs[m].c = c;
                                jobs[m].cache[0] = cache_row_a + nov*(a1*(a-a0)+b);
                                jobs[m].cache[1] = cache_row_a + nov*(a1*(a-a0)+c);
                                jobs[m].cache[2] = cache_row_a + nov*(a1*(b-a0)+a);
                                jobs[m].cache[3] = cache_row_a + nov*(a1*(b-a0)+c);
                                jobs[m].cache[4] = cache_col_a + nov*(da*(c)+a-a0);
                                jobs[m].cache[5] = cache_col_a + nov*(da*(c)+b-a0);
                        }
                        for (c = a0; c <= b; c++, m++) {
                                jobs[m].a = a;
                                jobs[m].b = b;
                                jobs[m].c = c;
                                jobs[m].cache[0] = cache_row_a + nov*(a1*(a-a0)+b);
                                jobs[m].cache[1] = cache_row_a + nov*(a1*(a-a0)+c);
                                jobs[m].cache[2] = cache_row_a + nov*(a1*(b-a0)+a);
                                jobs[m].cache[3] = cache_row_a + nov*(a1*(b-a0)+c);
                                jobs[m].cache[4] = cache_row_a + nov*(a1*(c-a0)+a);
                                jobs[m].cache[5] = cache_row_a + nov*(a1*(c-a0)+b);
                        }
                } }
        }
        return m;
}

void _make_permute_indices(int *idx, int n)
{
        const int nn = n * n;
        const int nnn = nn * n;
        int *idx0 = idx;
        int *idx1 = idx0 + nnn;
        int *idx2 = idx1 + nnn;
        int *idx3 = idx2 + nnn;
        int *idx4 = idx3 + nnn;
        int *idx5 = idx4 + nnn;
        int i, j, k, m;

        for (m = 0, i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
        for (k = 0; k < n; k++, m++) {
                idx0[m] = i * nn + j * n + k;
                idx1[m] = i * nn + k * n + j;
                idx2[m] = j * nn + i * n + k;
                idx3[m] = k * nn + i * n + j;
                idx4[m] = j * nn + k * n + i;
                idx5[m] = k * nn + j * n + i;
        } } }
}

void CCsd_t_contract(double *e_tot,
                     double *mo_energy, double *t1T, double *t2T,
                     double *vooo, double *fvo,
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
                                        cache_row_b, cache_col_b, sizeof(double));
        int *permute_idx = malloc(sizeof(int) * nocc*nocc*nocc * 6);
        _make_permute_indices(permute_idx, nocc);
#pragma omp parallel default(none) \
        shared(njobs, nocc, nvir, mo_energy, t1T, t2T, nirrep, o_ir_loc, \
               v_ir_loc, oo_ir_loc, orbsym, vooo, fvo, jobs, e_tot, permute_idx)
{
        int a, b, c;
        size_t k;
        double *cache1 = malloc(sizeof(double) * (nocc*nocc*nocc*3+2));
        double *t1Thalf = malloc(sizeof(double) * nvir*nocc * 2);
        double *fvohalf = t1Thalf + nvir*nocc;
        for (k = 0; k < nvir*nocc; k++) {
                t1Thalf[k] = t1T[k] * .5;
                fvohalf[k] = fvo[k] * .5;
        }
        double e = 0;
#pragma omp for schedule (dynamic, 4)
        for (k = 0; k < njobs; k++) {
                a = jobs[k].a;
                b = jobs[k].b;
                c = jobs[k].c;
                e += contract6(nocc, nvir, a, b, c, mo_energy, t1Thalf, t2T,
                               nirrep, o_ir_loc, v_ir_loc, oo_ir_loc, orbsym,
                               fvohalf, vooo, cache1, jobs[k].cache, permute_idx);
        }
        free(t1Thalf);
        free(cache1);
#pragma omp critical
        *e_tot += e;
}
        free(permute_idx);
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
                out[i*nn+j*n+k] = v[i*nn+j*n+k] * 4
                                + v[j*nn+k*n+i]
                                + v[k*nn+i*n+j]
                                - v[k*nn+j*n+i] * 2
                                - v[i*nn+k*n+j] * 2
                                - v[j*nn+i*n+k] * 2;
        } } }
}

static void zget_wv(double complex *w, double complex *v,
                    double complex *cache, double complex *fvohalf,
                    double complex *vooo, double complex *vv_op,
                    double complex *t1Thalf, double complex *t2T,
                    int nocc, int nvir, int a, int b, int c, int *idx)
{
        const double complex D0 = 0;
        const double complex D1 = 1;
        const double complex DN1 =-1;
        const char TRANS_N = 'N';
        const int nmo = nocc + nvir;
        const int noo = nocc * nocc;
        const size_t nooo = nocc * noo;
        const size_t nvoo = nvir * noo;
        int i, j, k, n;
        double complex *pt2T;

        zgemm_(&TRANS_N, &TRANS_N, &noo, &nocc, &nvir,
               &D1, t2T+c*nvoo, &noo, vv_op+nocc, &nmo,
               &D0, cache, &noo);
        zgemm_(&TRANS_N, &TRANS_N, &nocc, &noo, &nocc,
               &DN1, t2T+c*nvoo+b*noo, &nocc, vooo+a*nooo, &nocc,
               &D1, cache, &nocc);

        pt2T = t2T + b * nvoo + a * noo;
        for (n = 0, i = 0; i < nocc; i++) {
        for (j = 0; j < nocc; j++) {
        for (k = 0; k < nocc; k++, n++) {
                w[idx[n]] += cache[n];
                v[idx[n]] +=(vv_op[i*nmo+j] * t1Thalf[c*nocc+k]
                           + pt2T[i*nocc+j] * fvohalf[c*nocc+k]);
        } } }
}

double _ccsd_t_zget_energy(double complex *w, double complex *v,
                           double *mo_energy, int nocc,
                           int a, int b, int c, double fac)
{
        int i, j, k, n;
        double abc = mo_energy[nocc+a] + mo_energy[nocc+b] + mo_energy[nocc+c];
        double et = 0;

        for (n = 0, i = 0; i < nocc; i++) {
        for (j = 0; j < nocc; j++) {
        for (k = 0; k < nocc; k++, n++) {
                et += fac / (mo_energy[i] + mo_energy[j] + mo_energy[k] - abc) * w[n] * conj(v[n]);
        } } }
        return et;
}

static double complex
zcontract6(int nocc, int nvir, int a, int b, int c,
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

void CCsd_t_zcontract(double complex *e_tot,
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

        int *permute_idx = malloc(sizeof(int) * nocc*nocc*nocc * 6);
        _make_permute_indices(permute_idx, nocc);

#pragma omp parallel default(none) \
        shared(njobs, nocc, nvir, mo_energy, t1T, t2T, nirrep, o_ir_loc, \
               v_ir_loc, oo_ir_loc, orbsym, vooo, fvo, jobs, e_tot, permute_idx)
{
        int a, b, c;
        size_t k;
        double complex *cache1 = malloc(sizeof(double complex) * (nocc*nocc*nocc*3+2));
        double complex *t1Thalf = malloc(sizeof(double complex) * nvir*nocc * 2);
        double complex *fvohalf = t1Thalf + nvir*nocc;
        for (k = 0; k < nvir*nocc; k++) {
                t1Thalf[k] = t1T[k] * .5;
                fvohalf[k] = fvo[k] * .5;
        }
        double complex e = 0;
#pragma omp for schedule (dynamic, 4)
        for (k = 0; k < njobs; k++) {
                a = jobs[k].a;
                b = jobs[k].b;
                c = jobs[k].c;
                e += zcontract6(nocc, nvir, a, b, c, mo_energy, t1Thalf, t2T,
                               nirrep, o_ir_loc, v_ir_loc, oo_ir_loc, orbsym,
                               fvohalf, vooo, cache1, jobs[k].cache, permute_idx);
        }
        free(t1Thalf);
        free(cache1);
#pragma omp critical
        *e_tot += e;
}
        free(permute_idx);
}


/*****************************************************************************
 *
 * mpi4pyscf
 *
 *****************************************************************************/
static void MPICCget_wv(double *w, double *v, double *cache,
                        double *fvohalf, double *vooo,
                        double *vv_op, double *t1Thalf,
                        double *t2T_a, double *t2T_c,
                        int nocc, int nvir, int a, int b, int c,
                        int a0, int b0, int c0, int *idx)
{
        const double D0 = 0;
        const double D1 = 1;
        const double DN1 = -1;
        const char TRANS_N = 'N';
        const int nmo = nocc + nvir;
        const int noo = nocc * nocc;
        const size_t nooo = nocc * noo;
        const size_t nvoo = nvir * noo;
        int i, j, k, n;
        double *pt2T;

        dgemm_(&TRANS_N, &TRANS_N, &noo, &nocc, &nvir,
               &D1, t2T_c+(c-c0)*nvoo, &noo, vv_op+nocc, &nmo,
               &D0, cache, &noo);
        dgemm_(&TRANS_N, &TRANS_N, &nocc, &noo, &nocc,
               &DN1, t2T_c+(c-c0)*nvoo+b*noo, &nocc, vooo+(a-a0)*nooo, &nocc,
               &D1, cache, &nocc);

        pt2T = t2T_a + (a-a0) * nvoo + b * noo;
        for (n = 0, i = 0; i < nocc; i++) {
        for (j = 0; j < nocc; j++) {
        for (k = 0; k < nocc; k++, n++) {
                w[idx[n]] += cache[n];
                v[idx[n]] +=(vv_op[i*nmo+j] * t1Thalf[c*nocc+k]
                           + pt2T[i*nocc+j] * fvohalf[c*nocc+k]);
        } } }
}

static double MPICCcontract6(int nocc, int nvir, int a, int b, int c,
                             double *mo_energy, double *t1T, double *fvo,
                             int *slices, double **data_ptrs, double *cache1,
                             int *permute_idx)
{
        const int a0 = slices[0];
        const int a1 = slices[1];
        const int b0 = slices[2];
        const int b1 = slices[3];
        const int c0 = slices[4];
        const int c1 = slices[5];
        const int da = a1 - a0;
        const int db = b1 - b0;
        const int dc = c1 - c0;
        const int nooo = nocc * nocc * nocc;
        const int nmo = nocc + nvir;
        const size_t nop = nocc * nmo;
        int *idx0 = permute_idx;
        int *idx1 = idx0 + nooo;
        int *idx2 = idx1 + nooo;
        int *idx3 = idx2 + nooo;
        int *idx4 = idx3 + nooo;
        int *idx5 = idx4 + nooo;
        double *vvop_ab = data_ptrs[0] + ((a-a0)*db+b-b0) * nop;
        double *vvop_ac = data_ptrs[1] + ((a-a0)*dc+c-c0) * nop;
        double *vvop_ba = data_ptrs[2] + ((b-b0)*da+a-a0) * nop;
        double *vvop_bc = data_ptrs[3] + ((b-b0)*dc+c-c0) * nop;
        double *vvop_ca = data_ptrs[4] + ((c-c0)*da+a-a0) * nop;
        double *vvop_cb = data_ptrs[5] + ((c-c0)*db+b-b0) * nop;
        double *vooo_a = data_ptrs[6];
        double *vooo_b = data_ptrs[7];
        double *vooo_c = data_ptrs[8];
        double *t2T_a = data_ptrs[9 ];
        double *t2T_b = data_ptrs[10];
        double *t2T_c = data_ptrs[11];

        double *v0 = cache1;
        double *w0 = v0 + nooo;
        double *z0 = w0 + nooo;
        double *wtmp = z0;
        int i;

        for (i = 0; i < nooo; i++) {
                w0[i] = 0;
                v0[i] = 0;
        }

        MPICCget_wv(w0, v0, wtmp, fvo, vooo_a, vvop_ab, t1T, t2T_a, t2T_c, nocc, nvir, a, b, c, a0, b0, c0, idx0);
        MPICCget_wv(w0, v0, wtmp, fvo, vooo_a, vvop_ac, t1T, t2T_a, t2T_b, nocc, nvir, a, c, b, a0, c0, b0, idx1);
        MPICCget_wv(w0, v0, wtmp, fvo, vooo_b, vvop_ba, t1T, t2T_b, t2T_c, nocc, nvir, b, a, c, b0, a0, c0, idx2);
        MPICCget_wv(w0, v0, wtmp, fvo, vooo_b, vvop_bc, t1T, t2T_b, t2T_a, nocc, nvir, b, c, a, b0, c0, a0, idx3);
        MPICCget_wv(w0, v0, wtmp, fvo, vooo_c, vvop_ca, t1T, t2T_c, t2T_b, nocc, nvir, c, a, b, c0, a0, b0, idx4);
        MPICCget_wv(w0, v0, wtmp, fvo, vooo_c, vvop_cb, t1T, t2T_c, t2T_a, nocc, nvir, c, b, a, c0, b0, a0, idx5);
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

size_t _MPICCsd_t_gen_jobs(CacheJob *jobs, int nocc, int nvir,
                           int *slices, double **data_ptrs)
{
        const int a0 = slices[0];
        const int a1 = slices[1];
        const int b0 = slices[2];
        const int b1 = slices[3];
        const int c0 = slices[4];
        const int c1 = slices[5];
        size_t m, a, b, c;

        m = 0;
        for (a = a0; a < a1; a++) {
        for (b = b0; b < MIN(b1, a+1); b++) {
        for (c = c0; c < MIN(c1, b+1); c++, m++) {
                jobs[m].a = a;
                jobs[m].b = b;
                jobs[m].c = c;
        } } }
        return m;
}

void MPICCsd_t_contract(double *e_tot, double *mo_energy, double *t1T,
                        double *fvo, int nocc, int nvir,
                        int *slices, double **data_ptrs)
{
        const int a0 = slices[0];
        const int a1 = slices[1];
        const int b0 = slices[2];
        const int b1 = slices[3];
        const int c0 = slices[4];
        const int c1 = slices[5];
        int da = a1 - a0;
        int db = b1 - b0;
        int dc = c1 - c0;
        CacheJob *jobs = malloc(sizeof(CacheJob) * da*db*dc);
        size_t njobs = _MPICCsd_t_gen_jobs(jobs, nocc, nvir, slices, data_ptrs);

        int *permute_idx = malloc(sizeof(int) * nocc*nocc*nocc * 6);
        _make_permute_indices(permute_idx, nocc);

#pragma omp parallel default(none) \
        shared(njobs, nocc, nvir, mo_energy, t1T, fvo, jobs, e_tot, slices, \
               data_ptrs, permute_idx)
{
        int a, b, c;
        size_t k;
        double *cache1 = malloc(sizeof(double) * (nocc*nocc*nocc*3+2));
        double *t1Thalf = malloc(sizeof(double) * nvir*nocc * 2);
        double *fvohalf = t1Thalf + nvir*nocc;
        for (k = 0; k < nvir*nocc; k++) {
                t1Thalf[k] = t1T[k] * .5;
                fvohalf[k] = fvo[k] * .5;
        }
        double e = 0;
#pragma omp for schedule (dynamic, 4)
        for (k = 0; k < njobs; k++) {
                a = jobs[k].a;
                b = jobs[k].b;
                c = jobs[k].c;
                e += MPICCcontract6(nocc, nvir, a, b, c, mo_energy, t1Thalf,
                                    fvohalf, slices, data_ptrs, cache1,
                                    permute_idx);
        }
        free(t1Thalf);
        free(cache1);
#pragma omp critical
        *e_tot += e;
}
        free(permute_idx);
}

/*****************************************************************************
 *
 * pyscf periodic ccsd(t) with k-points
 *
 *****************************************************************************/

size_t _CCsd_t_gen_jobs_full(CacheJob *jobs, int nocc, int nvir,
                             int *slices)
{
        const int a0 = slices[0];
        const int a1 = slices[1];
        const int b0 = slices[2];
        const int b1 = slices[3];
        const int c0 = slices[4];
        const int c1 = slices[5];
        size_t m, a, b, c;

        m = 0;
        for (a = a0; a < a1; a++) {
        for (b = b0; b < b1; b++) {
        for (c = c0; c < c1; c++, m++) {
                jobs[m].a = a;
                jobs[m].b = b;
                jobs[m].c = c;
        } } }
        return m;
}

static void CCzget_wv(double complex *w, double complex *v, double complex *cache,
                      double complex *fvohalf, double complex *vooo,
                      double complex *vv_op, double complex *vv_op2,
                      double complex *t1Thalf, double complex *t2T_c1,
                      double complex *t2T_c2, double complex *t2T_c3,
                      int nocc, int nvir, int a, int b, int c,
                      int a0, int b0, int c0, int *idx, int bool_add_v)
{
        const double complex D0 = 0;
        const double complex D1 = 1;
        const double complex DN1 = -1;
        const char TRANS_N = 'N';
        const int nmo = nocc + nvir;
        const int noo = nocc * nocc;
        const size_t nooo = nocc * noo;
        const size_t nvoo = nvir * noo;
        int i, j, k, n;
        double complex *pt2T;


        zgemm_(&TRANS_N, &TRANS_N, &noo, &nocc, &nvir,
               &D1, t2T_c1+(c-c0)*nvoo, &noo, vv_op+nocc, &nmo,
               &D0, cache, &noo);
        zgemm_(&TRANS_N, &TRANS_N, &nocc, &noo, &nocc,
               &DN1, t2T_c2+(c-c0)*nvoo+b*noo, &nocc, vooo+(a-a0)*nooo, &nocc,
               &D1, cache, &nocc);

        pt2T = t2T_c3 + (b-b0)*nvoo + a*noo;
        for (n = 0, i = 0; i < nocc; i++) {
        for (j = 0; j < nocc; j++) {
        for (k = 0; k < nocc; k++, n++) {
                w[idx[n]] += cache[n];
                if(bool_add_v == 1){
                    v[idx[n]] += (vv_op2[j*nmo+i] * t1Thalf[c*nocc+k]
                                 + pt2T[i*nocc+j] * fvohalf[c*nocc+k]);
                }
        } } }
}

static void zcontract6_t3T(int nocc, int nvir, int a, int b, int c,
                           int *mo_offset, double complex *t3Tw,
                           double complex *t3Tv, double *mo_energy,
                           double complex *t1T, double complex *fvo, int *slices,
                           double complex **data_ptrs, double complex *cache1,
                           int *permute_idx)
{
        const int a0 = slices[0];
        const int a1 = slices[1];
        const int b0 = slices[2];
        const int b1 = slices[3];
        const int c0 = slices[4];
        const int c1 = slices[5];
        const int da = a1 - a0;
        const int db = b1 - b0;
        const int dc = c1 - c0;
        const int nooo = nocc * nocc * nocc;
        const int nmo = nocc + nvir;
        const int nop = nocc * nmo;
        const int nov = nocc * nvir;
        int *idx0 = permute_idx;
        int *idx1 = idx0 + nooo;
        int *idx2 = idx1 + nooo;
        int *idx3 = idx2 + nooo;
        int *idx4 = idx3 + nooo;
        int *idx5 = idx4 + nooo;
        int ki = mo_offset[0];
        int kj = mo_offset[1];
        int kk = mo_offset[2];
        int ka = mo_offset[3];
        int kb = mo_offset[4];
        int kc = mo_offset[5];
        double complex *t1T_a = t1T + ka * nov;
        double complex *t1T_b = t1T + kb * nov;
        double complex *t1T_c = t1T + kc * nov;
        double complex *fvo_a = fvo + ka * nov;
        double complex *fvo_b = fvo + kb * nov;
        double complex *fvo_c = fvo + kc * nov;
        double complex *vvop_ab = data_ptrs[0] + ((a-a0)*db+b-b0) * nop;
        double complex *vvop_ac = data_ptrs[1] + ((a-a0)*dc+c-c0) * nop;
        double complex *vvop_ba = data_ptrs[2] + ((b-b0)*da+a-a0) * nop;
        double complex *vvop_bc = data_ptrs[3] + ((b-b0)*dc+c-c0) * nop;
        double complex *vvop_ca = data_ptrs[4] + ((c-c0)*da+a-a0) * nop;
        double complex *vvop_cb = data_ptrs[5] + ((c-c0)*db+b-b0) * nop;
        double complex *vooo_aj = data_ptrs[6];
        double complex *vooo_ak = data_ptrs[7];
        double complex *vooo_bi = data_ptrs[8];
        double complex *vooo_bk = data_ptrs[9];
        double complex *vooo_ci = data_ptrs[10];
        double complex *vooo_cj = data_ptrs[11];
        double complex *t2T_cj = data_ptrs[12];
        double complex *t2T_cb = data_ptrs[13];
        double complex *t2T_bk = data_ptrs[14];
        double complex *t2T_bc = data_ptrs[15];
        double complex *t2T_ci = data_ptrs[16];
        double complex *t2T_ca = data_ptrs[17];
        double complex *t2T_ak = data_ptrs[18];
        double complex *t2T_ac = data_ptrs[19];
        double complex *t2T_bi = data_ptrs[20];
        double complex *t2T_ba = data_ptrs[21];
        double complex *t2T_aj = data_ptrs[22];
        double complex *t2T_ab = data_ptrs[23];
        double abc = mo_energy[nocc+a+ka*nmo] + mo_energy[nocc+b+kb*nmo] + mo_energy[nocc+c+kc*nmo];

        double div;
        double complex *v0 = cache1;
        double complex *w0 = v0 + nooo;
        double complex *z0 = w0 + nooo;
        double complex *wtmp = z0;
        int i, j, k, n;
        int offset;

        for (i = 0; i < nooo; i++) {
                w0[i] = 0;
                v0[i] = 0;
        }

/*
 * t2T = t2.transpose(2,3,1,0)
 * ov = vv_op[:,nocc:]
 * oo = vv_op[:,:nocc]
 * w = numpy.einsum('if,fjk->ijk', ov, t2T[c])
 * w-= numpy.einsum('ijm,mk->ijk', vooo[a], t2T[c,b])
 * v = numpy.einsum('ij,k->ijk', oo, t1T[c]*.5)
 * v+= numpy.einsum('ij,k->ijk', t2T[b,a], fov[:,c]*.5)
 * v+= w
 */

        CCzget_wv(w0, v0, wtmp, fvo_c, vooo_aj, vvop_ab, vvop_ba, t1T_c, t2T_cj, t2T_cb, t2T_ba,
                  nocc, nvir, a, b, c, a0, b0, c0, idx0, (kk==kc));
        CCzget_wv(w0, v0, wtmp, fvo_b, vooo_ak, vvop_ac, vvop_ca, t1T_b, t2T_bk, t2T_bc, t2T_ca,
                  nocc, nvir, a, c, b, a0, c0, b0, idx1, (kj==kb));
        CCzget_wv(w0, v0, wtmp, fvo_c, vooo_bi, vvop_ba, vvop_ab, t1T_c, t2T_ci, t2T_ca, t2T_ab,
                  nocc, nvir, b, a, c, b0, a0, c0, idx2, (kk==kc));
        CCzget_wv(w0, v0, wtmp, fvo_a, vooo_bk, vvop_bc, vvop_cb, t1T_a, t2T_ak, t2T_ac, t2T_cb,
                  nocc, nvir, b, c, a, b0, c0, a0, idx3, (ka==ki));
        CCzget_wv(w0, v0, wtmp, fvo_b, vooo_ci, vvop_ca, vvop_ac, t1T_b, t2T_bi, t2T_ba, t2T_ac,
                  nocc, nvir, c, a, b, c0, a0, b0, idx4, (kb==kj));
        CCzget_wv(w0, v0, wtmp, fvo_a, vooo_cj, vvop_cb, vvop_bc, t1T_a, t2T_aj, t2T_ab, t2T_bc,
                  nocc, nvir, c, b, a, c0, b0, a0, idx5, (ka==ki));

        offset = (((a-a0)*db + b-b0)*dc + c-c0)*nooo;
        for (n = 0, i = 0; i < nocc; i++) {
        for (j = 0; j < nocc; j++) {
        for (k = 0; k < nocc; k++, n++) {
            //div = 1. / (mo_energy[i+ki*nmo] + mo_energy[j+kj*nmo] + mo_energy[k+kk*nmo] - abc);
            t3Tw[offset + n] = w0[n];
            t3Tv[offset + n] = v0[n];
        } } }

}

void CCsd_zcontract_t3T(double complex *t3Tw, double complex *t3Tv, double *mo_energy,
                        double complex *t1T, double complex *fvo, int nocc, int nvir, int nkpts,
                        int *mo_offset, int *slices, double complex **data_ptrs)
{
        const int a0 = slices[0];
        const int a1 = slices[1];
        const int b0 = slices[2];
        const int b1 = slices[3];
        const int c0 = slices[4];
        const int c1 = slices[5];
        int da = a1 - a0;
        int db = b1 - b0;
        int dc = c1 - c0;
        CacheJob *jobs = malloc(sizeof(CacheJob) * da*db*dc);
        size_t njobs = _CCsd_t_gen_jobs_full(jobs, nocc, nvir, slices);

        int *permute_idx = malloc(sizeof(int) * nocc*nocc*nocc * 6);
        _make_permute_indices(permute_idx, nocc);

#pragma omp parallel default(none) \
        shared(njobs, nocc, nvir, nkpts, t3Tw, t3Tv, mo_offset, mo_energy, t1T, fvo, jobs, slices, \
               data_ptrs, permute_idx)
{
        int a, b, c;
        size_t k;
        complex double *cache1 = malloc(sizeof(double complex) * (nocc*nocc*nocc*3+2));
        complex double *t1Thalf = malloc(sizeof(double complex) * nkpts*nvir*nocc*2);
        complex double *fvohalf = t1Thalf + nkpts*nvir*nocc;
        for (k = 0; k < nkpts*nvir*nocc; k++) {
                t1Thalf[k] = t1T[k] * .5;
                fvohalf[k] = fvo[k] * .5;
        }
#pragma omp for schedule (dynamic, 4)
        for (k = 0; k < njobs; k++) {
                a = jobs[k].a;
                b = jobs[k].b;
                c = jobs[k].c;
                zcontract6_t3T(nocc, nvir, a, b, c, mo_offset, t3Tw, t3Tv, mo_energy, t1Thalf,
                               fvohalf, slices, data_ptrs, cache1,
                               permute_idx);
        }
        free(t1Thalf);
        free(cache1);
}
        free(jobs);
        free(permute_idx);
}

int Sc(int i, int a, int nocc) { return nocc * ( a + 1 ) - ( i + 1 ); }
int Dc(int i, int j, int a, int b, int nocc2)
{ return (int) (nocc2 * ( b*(b-1)/2 + a + 1 ) - ( j*(j-1)/2 + i + 1 )); }
int Tc(int i, int j, int k, int a, int b, int c, int nocc3)
{ return (int) (nocc3 * ( c*(c-1)*(c-2)/6 + b*(b-1)/2 + a + 1 ) - ( k*(k-1)*(k-2)/6 + j*(j-1)/2 + i + 1 )); }

int DSc(int i, int j, int k, int a, int b, int c, int nocc, int nvir, int nocc2)
{ return Dc(i, j, a, b, nocc2) * nocc * nvir + Sc(k, c, nocc); }

int S(int i, int a, int nvir) { return i*nvir+a; } 
int D(int i, int j, int a, int b, int nocc, int nvir)
{ return ((i*nocc+j)*nvir+a)*nvir+b; } 

int De(int i, int a, int j, int b, int nocc, int nvir)
{ return ((i*nvir+a)*nocc+j)*nvir+b; } 

int Dtmp1(int i, int j, int k, int a, int nocc, int nvir)
{ return ((i*nocc+j)*nocc+k)*nvir+a; } 
//{ return ((a*nocc+k)*nocc+j)*nocc+i; } 

int Dtmp2(int i, int a, int b, int c, int nvir)
{ return ((i*nvir+a)*nvir+b)*nvir+c; } 

size_t T(int i, int j, int k, int a, int b, int c, int nocc, int nvir)
{ return ((((i*nocc+(size_t)(j))*nocc+k)*nvir+a)*nvir+b)*nvir+c; } 
int Q(int i, int j, int k, int l, int a, int b, int c, int d, int nocc, int nvir)
{ return ((((((i*nocc+j)*nocc+k)*nocc+l)*nvir+a)*nvir+b)*nvir+c)*nvir+d; } 

double t1xt1aa(int i, int j, int a, int b, int nocc, int nvir, double *t1)
{
    double t1xt1 = 0.0;

    t1xt1 += t1[S(i, a, nvir)] * t1[S(j, b, nvir)];
    t1xt1 -= t1[S(i, b, nvir)] * t1[S(j, a, nvir)];

    return t1xt1;
}

double t1xt1ab(int i, int j, int a, int b, int nocc, int nvir, double *t1)
{
    double t1xt1;

    t1xt1 = t1[S(i, a, nvir)] * t1[S(j, b, nvir)];

    return t1xt1;
}

double t1xt1ab_u(int i, int j, int a, int b, int nvira, int nvirb, double *t1a, double *t1b)
{
    double t1xt1;

    t1xt1 = t1a[S(i, a, nvira)] * t1b[S(j, b, nvirb)];

    return t1xt1;
}

double t1xt2aaa(int i, int j, int k, int a, int b, int c, int nocc, int nvir, double *t1, double *t2aa)
{
    double t1xt2 = 0.0;
    t1xt2 += t1[S(i, a, nvir)] * t2aa[D(j, k, b, c, nocc, nvir)];
    t1xt2 -= t1[S(i, b, nvir)] * t2aa[D(j, k, a, c, nocc, nvir)];
    t1xt2 += t1[S(i, c, nvir)] * t2aa[D(j, k, a, b, nocc, nvir)];
    t1xt2 -= t1[S(j, a, nvir)] * t2aa[D(i, k, b, c, nocc, nvir)];
    t1xt2 += t1[S(j, b, nvir)] * t2aa[D(i, k, a, c, nocc, nvir)];
    t1xt2 -= t1[S(j, c, nvir)] * t2aa[D(i, k, a, b, nocc, nvir)];
    t1xt2 += t1[S(k, a, nvir)] * t2aa[D(i, j, b, c, nocc, nvir)];
    t1xt2 -= t1[S(k, b, nvir)] * t2aa[D(i, j, a, c, nocc, nvir)];
    t1xt2 += t1[S(k, c, nvir)] * t2aa[D(i, j, a, b, nocc, nvir)];
    return t1xt2;
}

double t1xt1xt1aaa(int i, int j, int k, int a, int b, int c, int nocc, int nvir, double *t1)
{
    double t1xt1xt1 = 0.0;
    t1xt1xt1 += t1[S(i, a, nvir)] * t1[S(j, b, nvir)] * t1[S(k, c, nvir)];
    t1xt1xt1 -= t1[S(i, a, nvir)] * t1[S(j, c, nvir)] * t1[S(k, b, nvir)];
    t1xt1xt1 -= t1[S(i, b, nvir)] * t1[S(j, a, nvir)] * t1[S(k, c, nvir)];
    t1xt1xt1 += t1[S(i, b, nvir)] * t1[S(j, c, nvir)] * t1[S(k, a, nvir)];
    t1xt1xt1 += t1[S(i, c, nvir)] * t1[S(j, a, nvir)] * t1[S(k, b, nvir)];
    t1xt1xt1 -= t1[S(i, c, nvir)] * t1[S(j, b, nvir)] * t1[S(k, a, nvir)];
    return t1xt1xt1;
}

double t1xt2aab(int i, int j, int k, int a, int b, int c, int nocc, int nvir, double *t1, double *t2aa, double *t2ab)
{
    double t1xt2 = 0.0;
    t1xt2 += t1[S(i, a, nvir)] * t2ab[D(j, k, b, c, nocc, nvir)];
    t1xt2 -= t1[S(i, b, nvir)] * t2ab[D(j, k, a, c, nocc, nvir)];
    t1xt2 -= t1[S(j, a, nvir)] * t2ab[D(i, k, b, c, nocc, nvir)];
    t1xt2 += t1[S(j, b, nvir)] * t2ab[D(i, k, a, c, nocc, nvir)];
    t1xt2 += t1[S(k, c, nvir)] * t2aa[D(i, j, a, b, nocc, nvir)];
    return t1xt2;
}

double t1xt1xt1aab(int i, int j, int k, int a, int b, int c, int nocc, int nvir, double *t1)
{
    double t1xt1xt1 = 0.0;
    t1xt1xt1 += t1[S(i, a, nvir)] * t1[S(j, b, nvir)] * t1[S(k, c, nvir)];
    t1xt1xt1 -= t1[S(i, b, nvir)] * t1[S(j, a, nvir)] * t1[S(k, c, nvir)];
    return t1xt1xt1;
}

double t1xt3aaab(int i, int j, int k, int l, int a, int b, int c, int d, int nocc, int nvir, double *t1, double *t3aaa, double *t3aab)
{
    double t1xt3 = 0.0;
    t1xt3 += t1[S(i, a, nvir)] * t3aab[T(j, k, l, b, c, d, nocc, nvir)];
    t1xt3 -= t1[S(i, b, nvir)] * t3aab[T(j, k, l, a, c, d, nocc, nvir)];
    t1xt3 += t1[S(i, c, nvir)] * t3aab[T(j, k, l, a, b, d, nocc, nvir)];
    t1xt3 -= t1[S(j, a, nvir)] * t3aab[T(i, k, l, b, c, d, nocc, nvir)];
    t1xt3 += t1[S(j, b, nvir)] * t3aab[T(i, k, l, a, c, d, nocc, nvir)];
    t1xt3 -= t1[S(j, c, nvir)] * t3aab[T(i, k, l, a, b, d, nocc, nvir)];
    t1xt3 += t1[S(k, a, nvir)] * t3aab[T(i, j, l, b, c, d, nocc, nvir)];
    t1xt3 -= t1[S(k, b, nvir)] * t3aab[T(i, j, l, a, c, d, nocc, nvir)];
    t1xt3 += t1[S(k, c, nvir)] * t3aab[T(i, j, l, a, b, d, nocc, nvir)];
    t1xt3 += t1[S(l, d, nvir)] * t3aaa[T(i, j, k, a, b, c, nocc, nvir)];
    return t1xt3;
}

double c3tot3aab(int i, int j, int k, int a, int b, int c, int nocc, int nocc2, int nvir, double *t1, double *t2aa, double *t2ab, double *c3aab, double c0)
{
    double t3 = 0.0;

    double parity = 1;
    if ( i > j ) {
        printf("parity change!");
    } 
    if ( a > b ) {
        printf("parity change!");
    } 

    // interm norm of c3
    t3 = c3aab[DSc(i, j, k, a, b, c, nocc, nvir, nocc2)] / c0;
    t3-= t1xt2aab(i, j, k, a, b, c, nocc, nvir, t1, t2aa, t2ab); 
    t3-= t1xt1xt1aab(i, j, k, a, b, c, nocc, nvir, t1); 

    return t3 * parity;
}

double c3tot3aaa(int i, int j, int k, int a, int b, int c, int nocc, int nocc3, int nvir, double *t1, double *t2aa, double *c3aaa, double c0)
{
    double t3 = 0.0;

    double parity = 1;
    if ( i > j || j > k || i > k ) {
        printf("parity change!: c3tott3aaa");
    } 
    if ( a > b || b > c || a > c ) {
        printf("parity change!: c3tott3aaa");
    } 

    // interm norm of c3
    t3 = c3aaa[Tc(i, j, k, a, b, c, nocc3)] / c0;
    t3-= t1xt2aaa (i, j, k, a, b, c, nocc, nvir, t1, t2aa); 
    t3-= t1xt1xt1aaa (i, j, k, a, b, c, nocc, nvir, t1); 

    return t3;
}



double t1xc3aaab(int i, int j, int k, int l, int a, int b, int c, int d, int nocc, int nocc2, int nocc3, int nvir, double *t1, double *t2aa, double *t2ab, double *c3aaa, double *c3aab, double c0)
{
    double t1xt3 = 0.0;
    t1xt3 += t1[S(i, a, nvir)] * c3tot3aab(j, k, l, b, c, d, nocc, nocc2, nvir, t1, t2aa, t2ab, c3aab, c0);
    t1xt3 -= t1[S(i, b, nvir)] * c3tot3aab(j, k, l, a, c, d, nocc, nocc2, nvir, t1, t2aa, t2ab, c3aab, c0);
    t1xt3 += t1[S(i, c, nvir)] * c3tot3aab(j, k, l, a, b, d, nocc, nocc2, nvir, t1, t2aa, t2ab, c3aab, c0);
    t1xt3 -= t1[S(j, a, nvir)] * c3tot3aab(i, k, l, b, c, d, nocc, nocc2, nvir, t1, t2aa, t2ab, c3aab, c0);
    t1xt3 += t1[S(j, b, nvir)] * c3tot3aab(i, k, l, a, c, d, nocc, nocc2, nvir, t1, t2aa, t2ab, c3aab, c0);
    t1xt3 -= t1[S(j, c, nvir)] * c3tot3aab(i, k, l, a, b, d, nocc, nocc2, nvir, t1, t2aa, t2ab, c3aab, c0);
    t1xt3 += t1[S(k, a, nvir)] * c3tot3aab(i, j, l, b, c, d, nocc, nocc2, nvir, t1, t2aa, t2ab, c3aab, c0);
    t1xt3 -= t1[S(k, b, nvir)] * c3tot3aab(i, j, l, a, c, d, nocc, nocc2, nvir, t1, t2aa, t2ab, c3aab, c0);
    t1xt3 += t1[S(k, c, nvir)] * c3tot3aab(i, j, l, a, b, d, nocc, nocc2, nvir, t1, t2aa, t2ab, c3aab, c0);
    t1xt3 += t1[S(l, d, nvir)] * c3tot3aaa(i, j, k, a, b, c, nocc, nocc3, nvir, t1, t2aa, c3aaa, c0);
    return t1xt3;
}

double t1xt3aabb(int i, int j, int k, int l, int a, int b, int c, int d, int nocc, int nvir, double *t1, double *t3aab)
{
    double t1xt3 = 0.0;
    t1xt3 += t1[S(i, a, nvir)] * t3aab[T(k, l, j, c, d, b, nocc, nvir)];
    t1xt3 -= t1[S(i, b, nvir)] * t3aab[T(k, l, j, c, d, a, nocc, nvir)];
    t1xt3 -= t1[S(j, a, nvir)] * t3aab[T(k, l, i, c, d, b, nocc, nvir)];
    t1xt3 += t1[S(j, b, nvir)] * t3aab[T(k, l, i, c, d, a, nocc, nvir)];
    t1xt3 += t1[S(k, c, nvir)] * t3aab[T(i, j, l, a, b, d, nocc, nvir)];
    t1xt3 -= t1[S(k, d, nvir)] * t3aab[T(i, j, l, a, b, c, nocc, nvir)];
    t1xt3 -= t1[S(l, c, nvir)] * t3aab[T(i, j, k, a, b, d, nocc, nvir)];
    t1xt3 += t1[S(l, d, nvir)] * t3aab[T(i, j, k, a, b, c, nocc, nvir)];
    return t1xt3;
}

double t1xc3aabb(int i, int j, int k, int l, int a, int b, int c, int d, int nocc, int nocc2, int nvir, double *t1, double *t2aa, double *t2ab, double *c3aab, double c0)
{
    double t1xt3 = 0.0;
    t1xt3 += t1[S(i, a, nvir)] * c3tot3aab(k, l, j, c, d, b, nocc, nocc2, nvir, t1, t2aa, t2ab, c3aab, c0);
    t1xt3 -= t1[S(i, b, nvir)] * c3tot3aab(k, l, j, c, d, a, nocc, nocc2, nvir, t1, t2aa, t2ab, c3aab, c0);
    t1xt3 -= t1[S(j, a, nvir)] * c3tot3aab(k, l, i, c, d, b, nocc, nocc2, nvir, t1, t2aa, t2ab, c3aab, c0);
    t1xt3 += t1[S(j, b, nvir)] * c3tot3aab(k, l, i, c, d, a, nocc, nocc2, nvir, t1, t2aa, t2ab, c3aab, c0);
    t1xt3 += t1[S(k, c, nvir)] * c3tot3aab(i, j, l, a, b, d, nocc, nocc2, nvir, t1, t2aa, t2ab, c3aab, c0);
    t1xt3 -= t1[S(k, d, nvir)] * c3tot3aab(i, j, l, a, b, c, nocc, nocc2, nvir, t1, t2aa, t2ab, c3aab, c0);
    t1xt3 -= t1[S(l, c, nvir)] * c3tot3aab(i, j, k, a, b, d, nocc, nocc2, nvir, t1, t2aa, t2ab, c3aab, c0);
    t1xt3 += t1[S(l, d, nvir)] * c3tot3aab(i, j, k, a, b, c, nocc, nocc2, nvir, t1, t2aa, t2ab, c3aab, c0);
    return t1xt3;
}

double t2xt2aaab(int i, int j, int k, int l, int a, int b, int c, int d, int nocc, int nvir, double *t2aa, double *t2ab)
{
    double t2xt2 = 0.0;
    t2xt2 += t2aa[D(i, j, a, b, nocc, nvir)] * t2ab[D(k, l, c, d, nocc, nvir)];
    t2xt2 -= t2aa[D(i, j, a, c, nocc, nvir)] * t2ab[D(k, l, b, d, nocc, nvir)];
    t2xt2 += t2aa[D(i, j, b, c, nocc, nvir)] * t2ab[D(k, l, a, d, nocc, nvir)];
    t2xt2 -= t2aa[D(i, k, a, b, nocc, nvir)] * t2ab[D(j, l, c, d, nocc, nvir)];
    t2xt2 += t2aa[D(i, k, a, c, nocc, nvir)] * t2ab[D(j, l, b, d, nocc, nvir)];
    t2xt2 -= t2aa[D(i, k, b, c, nocc, nvir)] * t2ab[D(j, l, a, d, nocc, nvir)];
    t2xt2 += t2ab[D(i, l, a, d, nocc, nvir)] * t2aa[D(j, k, b, c, nocc, nvir)];
    t2xt2 -= t2ab[D(i, l, b, d, nocc, nvir)] * t2aa[D(j, k, a, c, nocc, nvir)];
    t2xt2 += t2ab[D(i, l, c, d, nocc, nvir)] * t2aa[D(j, k, a, b, nocc, nvir)];
    return t2xt2;
}

double t2xt2aabb(int i, int j, int k, int l, int a, int b, int c, int d, int nocc, int nvir, double *t2aa, double *t2ab)
{
    double t2xt2 = 0.0;
    t2xt2 += t2aa[D(i, j, a, b, nocc, nvir)] * t2aa[D(k, l, c, d, nocc, nvir)];
    t2xt2 += t2ab[D(i, k, a, c, nocc, nvir)] * t2ab[D(j, l, b, d, nocc, nvir)];
    t2xt2 -= t2ab[D(i, k, a, d, nocc, nvir)] * t2ab[D(j, l, b, c, nocc, nvir)];
    t2xt2 -= t2ab[D(i, k, b, c, nocc, nvir)] * t2ab[D(j, l, a, d, nocc, nvir)];
    t2xt2 += t2ab[D(i, k, b, d, nocc, nvir)] * t2ab[D(j, l, a, c, nocc, nvir)];
    t2xt2 -= t2ab[D(i, l, a, c, nocc, nvir)] * t2ab[D(j, k, b, d, nocc, nvir)];
    t2xt2 += t2ab[D(i, l, a, d, nocc, nvir)] * t2ab[D(j, k, b, c, nocc, nvir)];
    t2xt2 += t2ab[D(i, l, b, c, nocc, nvir)] * t2ab[D(j, k, a, d, nocc, nvir)];
    t2xt2 -= t2ab[D(i, l, b, d, nocc, nvir)] * t2ab[D(j, k, a, c, nocc, nvir)];
    return t2xt2;
}

double t1xt1xt2aaab(int i, int j, int k, int l, int a, int b, int c, int d, int nocc, int nvir, double *t1, double *t2aa, double *t2ab)
{
    double t1xt1xt2 = 0.0;
    t1xt1xt2 += t1[S(i,a,nvir)] * t1[S(j,b,nvir)] * t2ab[D(k,l,c,d,nocc,nvir)];
    t1xt1xt2 -= t1[S(i,a,nvir)] * t1[S(j,c,nvir)] * t2ab[D(k,l,b,d,nocc,nvir)];
    t1xt1xt2 += t1[S(i,b,nvir)] * t1[S(j,c,nvir)] * t2ab[D(k,l,a,d,nocc,nvir)];
    t1xt1xt2 -= t1[S(i,a,nvir)] * t1[S(k,b,nvir)] * t2ab[D(j,l,c,d,nocc,nvir)];
    t1xt1xt2 += t1[S(i,a,nvir)] * t1[S(k,c,nvir)] * t2ab[D(j,l,b,d,nocc,nvir)];
    t1xt1xt2 -= t1[S(i,b,nvir)] * t1[S(k,c,nvir)] * t2ab[D(j,l,a,d,nocc,nvir)];
    t1xt1xt2 += t1[S(i,a,nvir)] * t1[S(l,d,nvir)] * t2aa[D(j,k,b,c,nocc,nvir)];
    t1xt1xt2 -= t1[S(i,b,nvir)] * t1[S(l,d,nvir)] * t2aa[D(j,k,a,c,nocc,nvir)];
    t1xt1xt2 += t1[S(i,c,nvir)] * t1[S(l,d,nvir)] * t2aa[D(j,k,a,b,nocc,nvir)];
    t1xt1xt2 += t1[S(j,a,nvir)] * t1[S(k,b,nvir)] * t2ab[D(i,l,c,d,nocc,nvir)];
    t1xt1xt2 -= t1[S(j,a,nvir)] * t1[S(k,c,nvir)] * t2ab[D(i,l,b,d,nocc,nvir)];
    t1xt1xt2 += t1[S(j,b,nvir)] * t1[S(k,c,nvir)] * t2ab[D(i,l,a,d,nocc,nvir)];
    t1xt1xt2 -= t1[S(j,a,nvir)] * t1[S(l,d,nvir)] * t2aa[D(i,k,b,c,nocc,nvir)];
    t1xt1xt2 += t1[S(j,b,nvir)] * t1[S(l,d,nvir)] * t2aa[D(i,k,a,c,nocc,nvir)];
    t1xt1xt2 -= t1[S(j,c,nvir)] * t1[S(l,d,nvir)] * t2aa[D(i,k,a,b,nocc,nvir)];
    t1xt1xt2 += t1[S(k,a,nvir)] * t1[S(l,d,nvir)] * t2aa[D(i,j,b,c,nocc,nvir)];
    t1xt1xt2 -= t1[S(k,b,nvir)] * t1[S(l,d,nvir)] * t2aa[D(i,j,a,c,nocc,nvir)];
    t1xt1xt2 += t1[S(k,c,nvir)] * t1[S(l,d,nvir)] * t2aa[D(i,j,a,b,nocc,nvir)];
    t1xt1xt2 -= t1[S(j,a,nvir)] * t1[S(i,b,nvir)] * t2ab[D(k,l,c,d,nocc,nvir)];
    t1xt1xt2 += t1[S(j,a,nvir)] * t1[S(i,c,nvir)] * t2ab[D(k,l,b,d,nocc,nvir)];
    t1xt1xt2 -= t1[S(j,b,nvir)] * t1[S(i,c,nvir)] * t2ab[D(k,l,a,d,nocc,nvir)];
    t1xt1xt2 += t1[S(k,a,nvir)] * t1[S(i,b,nvir)] * t2ab[D(j,l,c,d,nocc,nvir)];
    t1xt1xt2 -= t1[S(k,a,nvir)] * t1[S(i,c,nvir)] * t2ab[D(j,l,b,d,nocc,nvir)];
    t1xt1xt2 += t1[S(k,b,nvir)] * t1[S(i,c,nvir)] * t2ab[D(j,l,a,d,nocc,nvir)];
    t1xt1xt2 -= t1[S(k,a,nvir)] * t1[S(j,b,nvir)] * t2ab[D(i,l,c,d,nocc,nvir)];
    t1xt1xt2 += t1[S(k,a,nvir)] * t1[S(j,c,nvir)] * t2ab[D(i,l,b,d,nocc,nvir)];
    t1xt1xt2 -= t1[S(k,b,nvir)] * t1[S(j,c,nvir)] * t2ab[D(i,l,a,d,nocc,nvir)];
    return t1xt1xt2;
}

double t1xt1xt2aabb(int i, int j, int k, int l, int a, int b, int c, int d, int nocc, int nvir, double *t1, double *t2aa, double *t2ab)
{
    double t1xt1xt2 = 0.0;
    t1xt1xt2 += t1[S(i,a,nvir)] * t1[S(j,b,nvir)] * t2aa[D(k,l,c,d,nocc,nvir)];
    t1xt1xt2 += t1[S(k,c,nvir)] * t1[S(l,d,nvir)] * t2aa[D(i,j,a,b,nocc,nvir)];
    t1xt1xt2 -= t1[S(j,a,nvir)] * t1[S(i,b,nvir)] * t2aa[D(k,l,c,d,nocc,nvir)];
    t1xt1xt2 -= t1[S(l,c,nvir)] * t1[S(k,d,nvir)] * t2aa[D(i,j,a,b,nocc,nvir)];
    t1xt1xt2 += t1[S(i,a,nvir)] * t1[S(k,c,nvir)] * t2ab[D(j,l,b,d,nocc,nvir)];
    t1xt1xt2 -= t1[S(i,a,nvir)] * t1[S(k,d,nvir)] * t2ab[D(j,l,b,c,nocc,nvir)];
    t1xt1xt2 -= t1[S(i,b,nvir)] * t1[S(k,c,nvir)] * t2ab[D(j,l,a,d,nocc,nvir)];
    t1xt1xt2 += t1[S(i,b,nvir)] * t1[S(k,d,nvir)] * t2ab[D(j,l,a,c,nocc,nvir)];
    t1xt1xt2 -= t1[S(i,a,nvir)] * t1[S(l,c,nvir)] * t2ab[D(j,k,b,d,nocc,nvir)];
    t1xt1xt2 += t1[S(i,a,nvir)] * t1[S(l,d,nvir)] * t2ab[D(j,k,b,c,nocc,nvir)];
    t1xt1xt2 += t1[S(i,b,nvir)] * t1[S(l,c,nvir)] * t2ab[D(j,k,a,d,nocc,nvir)];
    t1xt1xt2 -= t1[S(i,b,nvir)] * t1[S(l,d,nvir)] * t2ab[D(j,k,a,c,nocc,nvir)];
    t1xt1xt2 -= t1[S(j,a,nvir)] * t1[S(k,c,nvir)] * t2ab[D(i,l,b,d,nocc,nvir)];
    t1xt1xt2 += t1[S(j,a,nvir)] * t1[S(k,d,nvir)] * t2ab[D(i,l,b,c,nocc,nvir)];
    t1xt1xt2 += t1[S(j,b,nvir)] * t1[S(k,c,nvir)] * t2ab[D(i,l,a,d,nocc,nvir)];
    t1xt1xt2 -= t1[S(j,b,nvir)] * t1[S(k,d,nvir)] * t2ab[D(i,l,a,c,nocc,nvir)];
    t1xt1xt2 += t1[S(j,a,nvir)] * t1[S(l,c,nvir)] * t2ab[D(i,k,b,d,nocc,nvir)];
    t1xt1xt2 -= t1[S(j,a,nvir)] * t1[S(l,d,nvir)] * t2ab[D(i,k,b,c,nocc,nvir)];
    t1xt1xt2 -= t1[S(j,b,nvir)] * t1[S(l,c,nvir)] * t2ab[D(i,k,a,d,nocc,nvir)];
    t1xt1xt2 += t1[S(j,b,nvir)] * t1[S(l,d,nvir)] * t2ab[D(i,k,a,c,nocc,nvir)];
    return t1xt1xt2;
}

double t1xt1xt1xt1aaab(int i, int j, int k, int l, int a, int b, int c, int d, int nocc, int nvir, double *t1)
{
    double t1xt1xt1xt1 = 0.0;
    t1xt1xt1xt1 += t1[S(i, a, nvir)] * t1[S(j, b, nvir)] * t1[S(k, c, nvir)] * t1[S(l, d, nvir)];
    t1xt1xt1xt1 -= t1[S(i, a, nvir)] * t1[S(j, c, nvir)] * t1[S(k, b, nvir)] * t1[S(l, d, nvir)];
    t1xt1xt1xt1 -= t1[S(i, b, nvir)] * t1[S(j, a, nvir)] * t1[S(k, c, nvir)] * t1[S(l, d, nvir)];
    t1xt1xt1xt1 += t1[S(i, b, nvir)] * t1[S(j, c, nvir)] * t1[S(k, a, nvir)] * t1[S(l, d, nvir)];
    t1xt1xt1xt1 += t1[S(i, c, nvir)] * t1[S(j, a, nvir)] * t1[S(k, b, nvir)] * t1[S(l, d, nvir)];
    t1xt1xt1xt1 -= t1[S(i, c, nvir)] * t1[S(j, b, nvir)] * t1[S(k, a, nvir)] * t1[S(l, d, nvir)];
    return t1xt1xt1xt1;
}

double t1xt1xt1xt1aabb(int i, int j, int k, int l, int a, int b, int c, int d, int nocc, int nvir, double *t1)
{
    double t1xt1xt1xt1 = 0.0;
    t1xt1xt1xt1 += t1[S(i, a, nvir)] * t1[S(j, b, nvir)] * t1[S(k, c, nvir)] * t1[S(l, d, nvir)];
    t1xt1xt1xt1 -= t1[S(i, a, nvir)] * t1[S(j, b, nvir)] * t1[S(k, d, nvir)] * t1[S(l, c, nvir)];
    return t1xt1xt1xt1;
}

void c1_to_t1(double *t1, double *c1, int nocc, int nvir) 
{
    int i, a, ia_c, ia_t;
    ia_c = -1;
    for (a = 0; a < nvir; a++) {
    for (i = nocc-1; i > -1; i--) {
        ia_c += 1;
        ia_t  = i * nvir + a;
        t1[ia_t] = c1[ia_c];
    }
    }
}

void c2_to_t2(double *t2aa, double *t2ab, double *c2aa, double *c2ab, double *t1, int nocc, int nvir) 
{
    double numzero = 1e-7;
    int i, j, a, b, ijab_c, ijab_t1, ijab_t2, ijab_t3, ijab_t4;
    int ia, jb, iajb_c, ijab_t;
    double tmp;

    // t2aa
    ijab_c = -1;
    for (b = 1; b < nvir; b++) {
    for (a = 0; a < b; a++) {
    for (j = nocc-1; j > 0; j--) {
    for (i = j-1; i > -1; i--) {
        ijab_c += 1;
        ijab_t1 = ((i*nocc+j)*nvir+a)*nvir+b;
        ijab_t2 = ((i*nocc+j)*nvir+b)*nvir+a;
        ijab_t3 = ((j*nocc+i)*nvir+a)*nvir+b;
        ijab_t4 = ((j*nocc+i)*nvir+b)*nvir+a;

        tmp = c2aa[ijab_c]; 
        if(fabs(tmp) > numzero) 
        {
            tmp -= t1xt1aa (i, j, a, b, nocc, nvir, t1); 
            t2aa[ijab_t1] =  tmp;
            t2aa[ijab_t2] = -tmp;
            t2aa[ijab_t3] = -tmp;
            t2aa[ijab_t4] =  tmp;
        }
    }
    }
    }
    }

    ia = -1;
    for (a = 0; a < nvir; a++) {
    for (i = nocc-1; i > -1; i--) {
        ia += 1;
        jb  =-1;
        for (b = 0; b < nvir; b++) {
        for (j = nocc-1; j > -1; j--) {
            jb += 1;
            iajb_c = ia * nocc*nvir + jb;
            ijab_t = ((i*nocc+j)*nvir+a)*nvir+b;

            tmp = c2ab[iajb_c]; 
            if(fabs(tmp) > numzero) 
            {
                tmp -= t1xt1ab (i, j, a, b, nocc, nvir, t1); 
                t2ab[ijab_t] = tmp;
            } 
        }
        }
    }
    }

}

void c3_to_t3(double *t3aaa, double *t3aab, double *c3aaa, double *c3aab, double *t1, double *t2aa, double *t2ab, int nocc, int nvir, double numzero) 
{
    int i, j, k, a, b, c;
    size_t ijkabc_t11, ijkabc_t21, ijkabc_t31, ijkabc_t41, ijkabc_t51, ijkabc_t61;
    size_t ijkabc_t12, ijkabc_t22, ijkabc_t32, ijkabc_t42, ijkabc_t52, ijkabc_t62;
    size_t ijkabc_t13, ijkabc_t23, ijkabc_t33, ijkabc_t43, ijkabc_t53, ijkabc_t63;
    size_t ijkabc_t14, ijkabc_t24, ijkabc_t34, ijkabc_t44, ijkabc_t54, ijkabc_t64;
    size_t ijkabc_t15, ijkabc_t25, ijkabc_t35, ijkabc_t45, ijkabc_t55, ijkabc_t65;
    size_t ijkabc_t16, ijkabc_t26, ijkabc_t36, ijkabc_t46, ijkabc_t56, ijkabc_t66;
    size_t ijab, kc, ijabkc_c, ijkabc_c;

    double tmp, tmp2;

    // t3aaa
    ijkabc_c = -1;
    for (c = 2; c < nvir; c++) {
    for (b = 1; b < c; b++) {
    for (a = 0; a < b; a++) {
    for (k = nocc-1; k > 1; k--) {
    for (j = k-1; j > 0; j--) {
    for (i = j-1; i > -1; i--) {
        ijkabc_c += 1;

//        //lsh dbg
//        printf("c3aaa, %d \n", ijkabc_c);        
        tmp = c3aaa[ijkabc_c]; 

//        if(fabs(tmp)-fabs(tmp2) > numzero) 
        if(fabs(tmp) > numzero) 
        {
            tmp2 = t1xt2aaa (i, j, k, a, b, c, nocc, nvir, t1, t2aa); 
            tmp2+= t1xt1xt1aaa (i, j, k, a, b, c, nocc, nvir, t1); 
            tmp -= tmp2; 
            ijkabc_t11 = T(i, j, k, a, b, c, nocc, nvir);
            ijkabc_t12 = T(i, j, k, b, c, a, nocc, nvir);
            ijkabc_t13 = T(i, j, k, c, a, b, nocc, nvir);
            ijkabc_t14 = T(i, j, k, a, c, b, nocc, nvir);
            ijkabc_t15 = T(i, j, k, b, a, c, nocc, nvir);
            ijkabc_t16 = T(i, j, k, c, b, a, nocc, nvir);
    
            t3aaa[ijkabc_t11] =  tmp;
            t3aaa[ijkabc_t12] =  tmp;
            t3aaa[ijkabc_t13] =  tmp;
            t3aaa[ijkabc_t14] = -tmp;
            t3aaa[ijkabc_t15] = -tmp;
            t3aaa[ijkabc_t16] = -tmp;
    
            ijkabc_t21 = T(j, k, i, a, b, c, nocc, nvir);
            ijkabc_t22 = T(j, k, i, b, c, a, nocc, nvir);
            ijkabc_t23 = T(j, k, i, c, a, b, nocc, nvir);
            ijkabc_t24 = T(j, k, i, a, c, b, nocc, nvir);
            ijkabc_t25 = T(j, k, i, b, a, c, nocc, nvir);
            ijkabc_t26 = T(j, k, i, c, b, a, nocc, nvir);
    
            t3aaa[ijkabc_t21] =  tmp;
            t3aaa[ijkabc_t22] =  tmp;
            t3aaa[ijkabc_t23] =  tmp;
            t3aaa[ijkabc_t24] = -tmp;
            t3aaa[ijkabc_t25] = -tmp;
            t3aaa[ijkabc_t26] = -tmp;
    
            ijkabc_t31 = T(k, i, j, a, b, c, nocc, nvir);
            ijkabc_t32 = T(k, i, j, b, c, a, nocc, nvir);
            ijkabc_t33 = T(k, i, j, c, a, b, nocc, nvir);
            ijkabc_t34 = T(k, i, j, a, c, b, nocc, nvir);
            ijkabc_t35 = T(k, i, j, b, a, c, nocc, nvir);
            ijkabc_t36 = T(k, i, j, c, b, a, nocc, nvir);
    
            t3aaa[ijkabc_t31] =  tmp;
            t3aaa[ijkabc_t32] =  tmp;
            t3aaa[ijkabc_t33] =  tmp;
            t3aaa[ijkabc_t34] = -tmp;
            t3aaa[ijkabc_t35] = -tmp;
            t3aaa[ijkabc_t36] = -tmp;
    
            ijkabc_t41 = T(i, k, j, a, b, c, nocc, nvir);
            ijkabc_t42 = T(i, k, j, b, c, a, nocc, nvir);
            ijkabc_t43 = T(i, k, j, c, a, b, nocc, nvir);
            ijkabc_t44 = T(i, k, j, a, c, b, nocc, nvir);
            ijkabc_t45 = T(i, k, j, b, a, c, nocc, nvir);
            ijkabc_t46 = T(i, k, j, c, b, a, nocc, nvir);
    
            t3aaa[ijkabc_t41] = -tmp;
            t3aaa[ijkabc_t42] = -tmp;
            t3aaa[ijkabc_t43] = -tmp;
            t3aaa[ijkabc_t44] =  tmp;
            t3aaa[ijkabc_t45] =  tmp;
            t3aaa[ijkabc_t46] =  tmp;
    
            ijkabc_t51 = T(j, i, k, a, b, c, nocc, nvir);
            ijkabc_t52 = T(j, i, k, b, c, a, nocc, nvir);
            ijkabc_t53 = T(j, i, k, c, a, b, nocc, nvir);
            ijkabc_t54 = T(j, i, k, a, c, b, nocc, nvir);
            ijkabc_t55 = T(j, i, k, b, a, c, nocc, nvir);
            ijkabc_t56 = T(j, i, k, c, b, a, nocc, nvir);
    
            t3aaa[ijkabc_t51] = -tmp;
            t3aaa[ijkabc_t52] = -tmp;
            t3aaa[ijkabc_t53] = -tmp;
            t3aaa[ijkabc_t54] =  tmp;
            t3aaa[ijkabc_t55] =  tmp;
            t3aaa[ijkabc_t56] =  tmp;
    
            ijkabc_t61 = T(k, j, i, a, b, c, nocc, nvir);
            ijkabc_t62 = T(k, j, i, b, c, a, nocc, nvir);
            ijkabc_t63 = T(k, j, i, c, a, b, nocc, nvir);
            ijkabc_t64 = T(k, j, i, a, c, b, nocc, nvir);
            ijkabc_t65 = T(k, j, i, b, a, c, nocc, nvir);
            ijkabc_t66 = T(k, j, i, c, b, a, nocc, nvir);
    
            t3aaa[ijkabc_t61] = -tmp;
            t3aaa[ijkabc_t62] = -tmp;
            t3aaa[ijkabc_t63] = -tmp;
            t3aaa[ijkabc_t64] =  tmp;
            t3aaa[ijkabc_t65] =  tmp;
            t3aaa[ijkabc_t66] =  tmp;
        }
    }
    }
    }
    }
    }
    }

    ijab = -1;
    for (b = 1; b < nvir; b++) {
    for (a = 0; a < b; a++) {
    for (j = nocc-1; j > 0; j--) {
    for (i = j-1; i > -1; i--) {
        ijab += 1;
        kc  =-1;
        for (c = 0; c < nvir; c++) {
        for (k = nocc-1; k > -1; k--) {
            kc += 1;
            ijabkc_c = ijab * nocc*nvir + kc;

//            //lsh dbg
//            printf("c3aab, %d \n", ijabkc_c);        

            tmp = c3aab[ijabkc_c]; 

//            if(fabs(tmp)-fabs(tmp2) > numzero) 
            if(fabs(tmp) > numzero) 
            {
                tmp2 = t1xt2aab(i, j, k, a, b, c, nocc, nvir, t1, t2aa, t2ab); 
                tmp2+= t1xt1xt1aab(i, j, k, a, b, c, nocc, nvir, t1); 

                tmp -= tmp2;    
                ijkabc_t11 = T(i, j, k, a, b, c, nocc, nvir);
                ijkabc_t12 = T(i, j, k, b, a, c, nocc, nvir);
        
                t3aab[ijkabc_t11] =  tmp;
                t3aab[ijkabc_t12] = -tmp;
       
                ijkabc_t21 = T(j, i, k, a, b, c, nocc, nvir);
                ijkabc_t22 = T(j, i, k, b, a, c, nocc, nvir);
       
                t3aab[ijkabc_t21] = -tmp;
                t3aab[ijkabc_t22] =  tmp;
            }
        }
        }
    }
    }
    }
    }

}

void c3_to_t3_ecT(double *t3aaa, double *t3aab, double *c3aaa, double *c3aab, double *t1, double *t2aa, double *t2ab, int nc_ref, int nvir_ref, int nocc, int nvir, double numzero) 
{
    int i, j, k, a, b, c;
    size_t ijkabc_t11, ijkabc_t21, ijkabc_t31, ijkabc_t41, ijkabc_t51, ijkabc_t61;
    size_t ijkabc_t12, ijkabc_t22, ijkabc_t32, ijkabc_t42, ijkabc_t52, ijkabc_t62;
    size_t ijkabc_t13, ijkabc_t23, ijkabc_t33, ijkabc_t43, ijkabc_t53, ijkabc_t63;
    size_t ijkabc_t14, ijkabc_t24, ijkabc_t34, ijkabc_t44, ijkabc_t54, ijkabc_t64;
    size_t ijkabc_t15, ijkabc_t25, ijkabc_t35, ijkabc_t45, ijkabc_t55, ijkabc_t65;
    size_t ijkabc_t16, ijkabc_t26, ijkabc_t36, ijkabc_t46, ijkabc_t56, ijkabc_t66;
    size_t ijab, kc, ijabkc_c, ijkabc_c;

    double tmp, tmp2;

    // t3aaa
    ijkabc_c = -1;
    for (c = 2; c < nvir; c++) {
    for (b = 1; b < c; b++) {
    for (a = 0; a < b; a++) {
    for (k = nocc-1; k > 1; k--) {
    for (j = k-1; j > 0; j--) {
    for (i = j-1; i > -1; i--) {
        ijkabc_c += 1;

        // exclude inactive space
        if (i <= nc_ref && j <= nc_ref && k <= nc_ref && \
            a > nvir_ref && b > nvir_ref && c > nvir_ref) continue; 

//        //lsh dbg
//        printf("c3aaa, %d \n", ijkabc_c);        
        tmp = c3aaa[ijkabc_c]; 

//        if(fabs(tmp)-fabs(tmp2) > numzero) 
        if(fabs(tmp) > numzero) 
        {
            tmp2 = t1xt2aaa (i, j, k, a, b, c, nocc, nvir, t1, t2aa); 
            tmp2+= t1xt1xt1aaa (i, j, k, a, b, c, nocc, nvir, t1); 
            tmp -= tmp2; 
            ijkabc_t11 = T(i, j, k, a, b, c, nocc, nvir);
            ijkabc_t12 = T(i, j, k, b, c, a, nocc, nvir);
            ijkabc_t13 = T(i, j, k, c, a, b, nocc, nvir);
            ijkabc_t14 = T(i, j, k, a, c, b, nocc, nvir);
            ijkabc_t15 = T(i, j, k, b, a, c, nocc, nvir);
            ijkabc_t16 = T(i, j, k, c, b, a, nocc, nvir);
    
            t3aaa[ijkabc_t11] =  tmp;
            t3aaa[ijkabc_t12] =  tmp;
            t3aaa[ijkabc_t13] =  tmp;
            t3aaa[ijkabc_t14] = -tmp;
            t3aaa[ijkabc_t15] = -tmp;
            t3aaa[ijkabc_t16] = -tmp;
    
            ijkabc_t21 = T(j, k, i, a, b, c, nocc, nvir);
            ijkabc_t22 = T(j, k, i, b, c, a, nocc, nvir);
            ijkabc_t23 = T(j, k, i, c, a, b, nocc, nvir);
            ijkabc_t24 = T(j, k, i, a, c, b, nocc, nvir);
            ijkabc_t25 = T(j, k, i, b, a, c, nocc, nvir);
            ijkabc_t26 = T(j, k, i, c, b, a, nocc, nvir);
    
            t3aaa[ijkabc_t21] =  tmp;
            t3aaa[ijkabc_t22] =  tmp;
            t3aaa[ijkabc_t23] =  tmp;
            t3aaa[ijkabc_t24] = -tmp;
            t3aaa[ijkabc_t25] = -tmp;
            t3aaa[ijkabc_t26] = -tmp;
    
            ijkabc_t31 = T(k, i, j, a, b, c, nocc, nvir);
            ijkabc_t32 = T(k, i, j, b, c, a, nocc, nvir);
            ijkabc_t33 = T(k, i, j, c, a, b, nocc, nvir);
            ijkabc_t34 = T(k, i, j, a, c, b, nocc, nvir);
            ijkabc_t35 = T(k, i, j, b, a, c, nocc, nvir);
            ijkabc_t36 = T(k, i, j, c, b, a, nocc, nvir);
    
            t3aaa[ijkabc_t31] =  tmp;
            t3aaa[ijkabc_t32] =  tmp;
            t3aaa[ijkabc_t33] =  tmp;
            t3aaa[ijkabc_t34] = -tmp;
            t3aaa[ijkabc_t35] = -tmp;
            t3aaa[ijkabc_t36] = -tmp;
    
            ijkabc_t41 = T(i, k, j, a, b, c, nocc, nvir);
            ijkabc_t42 = T(i, k, j, b, c, a, nocc, nvir);
            ijkabc_t43 = T(i, k, j, c, a, b, nocc, nvir);
            ijkabc_t44 = T(i, k, j, a, c, b, nocc, nvir);
            ijkabc_t45 = T(i, k, j, b, a, c, nocc, nvir);
            ijkabc_t46 = T(i, k, j, c, b, a, nocc, nvir);
    
            t3aaa[ijkabc_t41] = -tmp;
            t3aaa[ijkabc_t42] = -tmp;
            t3aaa[ijkabc_t43] = -tmp;
            t3aaa[ijkabc_t44] =  tmp;
            t3aaa[ijkabc_t45] =  tmp;
            t3aaa[ijkabc_t46] =  tmp;
    
            ijkabc_t51 = T(j, i, k, a, b, c, nocc, nvir);
            ijkabc_t52 = T(j, i, k, b, c, a, nocc, nvir);
            ijkabc_t53 = T(j, i, k, c, a, b, nocc, nvir);
            ijkabc_t54 = T(j, i, k, a, c, b, nocc, nvir);
            ijkabc_t55 = T(j, i, k, b, a, c, nocc, nvir);
            ijkabc_t56 = T(j, i, k, c, b, a, nocc, nvir);
    
            t3aaa[ijkabc_t51] = -tmp;
            t3aaa[ijkabc_t52] = -tmp;
            t3aaa[ijkabc_t53] = -tmp;
            t3aaa[ijkabc_t54] =  tmp;
            t3aaa[ijkabc_t55] =  tmp;
            t3aaa[ijkabc_t56] =  tmp;
    
            ijkabc_t61 = T(k, j, i, a, b, c, nocc, nvir);
            ijkabc_t62 = T(k, j, i, b, c, a, nocc, nvir);
            ijkabc_t63 = T(k, j, i, c, a, b, nocc, nvir);
            ijkabc_t64 = T(k, j, i, a, c, b, nocc, nvir);
            ijkabc_t65 = T(k, j, i, b, a, c, nocc, nvir);
            ijkabc_t66 = T(k, j, i, c, b, a, nocc, nvir);
    
            t3aaa[ijkabc_t61] = -tmp;
            t3aaa[ijkabc_t62] = -tmp;
            t3aaa[ijkabc_t63] = -tmp;
            t3aaa[ijkabc_t64] =  tmp;
            t3aaa[ijkabc_t65] =  tmp;
            t3aaa[ijkabc_t66] =  tmp;
        }
    }
    }
    }
    }
    }
    }

    ijab = -1;
    for (b = 1; b < nvir; b++) {
    for (a = 0; a < b; a++) {
    for (j = nocc-1; j > 0; j--) {
    for (i = j-1; i > -1; i--) {
        ijab += 1;
        kc  =-1;
        for (c = 0; c < nvir; c++) {
        for (k = nocc-1; k > -1; k--) {
            kc += 1;
            ijabkc_c = ijab * nocc*nvir + kc;

            // exclude inactive space
            if (i <= nc_ref && j <= nc_ref && k <= nc_ref && \
                a > nvir_ref && b > nvir_ref && c > nvir_ref) continue; 

//            //lsh dbg
//            printf("c3aab, %d \n", ijabkc_c);        

            tmp = c3aab[ijabkc_c]; 

//            if(fabs(tmp)-fabs(tmp2) > numzero) 
            if(fabs(tmp) > numzero) 
            {
                tmp2 = t1xt2aab(i, j, k, a, b, c, nocc, nvir, t1, t2aa, t2ab); 
                tmp2+= t1xt1xt1aab(i, j, k, a, b, c, nocc, nvir, t1); 

                tmp -= tmp2;    
                ijkabc_t11 = T(i, j, k, a, b, c, nocc, nvir);
                ijkabc_t12 = T(i, j, k, b, a, c, nocc, nvir);
        
                t3aab[ijkabc_t11] =  tmp;
                t3aab[ijkabc_t12] = -tmp;
       
                ijkabc_t21 = T(j, i, k, a, b, c, nocc, nvir);
                ijkabc_t22 = T(j, i, k, b, a, c, nocc, nvir);
       
                t3aab[ijkabc_t21] = -tmp;
                t3aab[ijkabc_t22] =  tmp;
            }
        }
        }
    }
    }
    }
    }

}



void c4_to_t4(double *t4aaab, double *t4aabb, double *c4aaab, double *c4aabb, double *t1, double *t2aa, double *t2ab, double *t3aaa, double *t3aab, int nocc, int nvir, double numzero) 
{
    int i, j, k, l, a, b, c, d, m_ijab;
    int ijkabc, ld, ijkabcld_c;
    int ijklabcd_t11, ijklabcd_t21, ijklabcd_t31, ijklabcd_t41, ijklabcd_t51, ijklabcd_t61;
    int ijklabcd_t12, ijklabcd_t22, ijklabcd_t32, ijklabcd_t42, ijklabcd_t52, ijklabcd_t62;
    int ijklabcd_t13, ijklabcd_t23, ijklabcd_t33, ijklabcd_t43, ijklabcd_t53, ijklabcd_t63;
    int ijklabcd_t14, ijklabcd_t24, ijklabcd_t34, ijklabcd_t44, ijklabcd_t54, ijklabcd_t64;
    int ijklabcd_t15, ijklabcd_t25, ijklabcd_t35, ijklabcd_t45, ijklabcd_t55, ijklabcd_t65;
    int ijklabcd_t16, ijklabcd_t26, ijklabcd_t36, ijklabcd_t46, ijklabcd_t56, ijklabcd_t66;
    int ijab, klcd, ijabklcd_c;

    double tmp, tmp2;

    // t4aaab
    ijkabc = -1;
    for (c = 2; c < nvir; c++) {
    for (b = 1; b < c; b++) {
    for (a = 0; a < b; a++) {
    for (k = nocc-1; k > 1; k--) {
    for (j = k-1; j > 0; j--) {
    for (i = j-1; i > -1; i--) {
        ijkabc += 1;
        ld = -1;
        for (d = 0; d < nvir; d++) {
        for (l = nocc-1; l > -1; l--) {
            ld += 1;
            ijkabcld_c = ijkabc * nocc*nvir + ld;
            tmp = c4aaab[ijkabcld_c]; 

//            if(fabs(tmp)-fabs(tmp2) > numzero) 
            if(fabs(tmp) > numzero) 
            {
                tmp2 = t1xt3aaab (i, j, k, l, a, b, c, d, nocc, nvir, t1, t3aaa, t3aab);   // may have 1e-5 bug 
                tmp2+= t2xt2aaab (i, j, k, l, a, b, c, d, nocc, nvir, t2aa, t2ab);         // may have 1e-3 bug 
                tmp2+= t1xt1xt2aaab (i, j, k, l, a, b, c, d, nocc, nvir, t1, t2aa, t2ab);  // may have 1e-5 bug 
                tmp2+= t1xt1xt1xt1aaab (i, j, k, l, a, b, c, d, nocc, nvir, t1);           // may have 1e-6 bug

                tmp -= tmp2; 
                ijklabcd_t11 = Q(i, j, k, l, a, b, c, d, nocc, nvir);
                ijklabcd_t12 = Q(i, j, k, l, b, c, a, d, nocc, nvir);
                ijklabcd_t13 = Q(i, j, k, l, c, a, b, d, nocc, nvir);
                ijklabcd_t14 = Q(i, j, k, l, a, c, b, d, nocc, nvir);
                ijklabcd_t15 = Q(i, j, k, l, b, a, c, d, nocc, nvir);
                ijklabcd_t16 = Q(i, j, k, l, c, b, a, d, nocc, nvir);
        
                t4aaab[ijklabcd_t11] =  tmp;
                t4aaab[ijklabcd_t12] =  tmp;
                t4aaab[ijklabcd_t13] =  tmp;
                t4aaab[ijklabcd_t14] = -tmp;
                t4aaab[ijklabcd_t15] = -tmp;
                t4aaab[ijklabcd_t16] = -tmp;
        
                ijklabcd_t21 = Q(j, k, i, l, a, b, c, d, nocc, nvir);
                ijklabcd_t22 = Q(j, k, i, l, b, c, a, d, nocc, nvir);
                ijklabcd_t23 = Q(j, k, i, l, c, a, b, d, nocc, nvir);
                ijklabcd_t24 = Q(j, k, i, l, a, c, b, d, nocc, nvir);
                ijklabcd_t25 = Q(j, k, i, l, b, a, c, d, nocc, nvir);
                ijklabcd_t26 = Q(j, k, i, l, c, b, a, d, nocc, nvir);
        
                t4aaab[ijklabcd_t21] =  tmp;
                t4aaab[ijklabcd_t22] =  tmp;
                t4aaab[ijklabcd_t23] =  tmp;
                t4aaab[ijklabcd_t24] = -tmp;
                t4aaab[ijklabcd_t25] = -tmp;
                t4aaab[ijklabcd_t26] = -tmp;
        
                ijklabcd_t31 = Q(k, i, j, l, a, b, c, d, nocc, nvir);
                ijklabcd_t32 = Q(k, i, j, l, b, c, a, d, nocc, nvir);
                ijklabcd_t33 = Q(k, i, j, l, c, a, b, d, nocc, nvir);
                ijklabcd_t34 = Q(k, i, j, l, a, c, b, d, nocc, nvir);
                ijklabcd_t35 = Q(k, i, j, l, b, a, c, d, nocc, nvir);
                ijklabcd_t36 = Q(k, i, j, l, c, b, a, d, nocc, nvir);
        
                t4aaab[ijklabcd_t31] =  tmp;
                t4aaab[ijklabcd_t32] =  tmp;
                t4aaab[ijklabcd_t33] =  tmp;
                t4aaab[ijklabcd_t34] = -tmp;
                t4aaab[ijklabcd_t35] = -tmp;
                t4aaab[ijklabcd_t36] = -tmp;
        
                ijklabcd_t41 = Q(i, k, j, l, a, b, c, d, nocc, nvir);
                ijklabcd_t42 = Q(i, k, j, l, b, c, a, d, nocc, nvir);
                ijklabcd_t43 = Q(i, k, j, l, c, a, b, d, nocc, nvir);
                ijklabcd_t44 = Q(i, k, j, l, a, c, b, d, nocc, nvir);
                ijklabcd_t45 = Q(i, k, j, l, b, a, c, d, nocc, nvir);
                ijklabcd_t46 = Q(i, k, j, l, c, b, a, d, nocc, nvir);
        
                t4aaab[ijklabcd_t41] = -tmp;
                t4aaab[ijklabcd_t42] = -tmp;
                t4aaab[ijklabcd_t43] = -tmp;
                t4aaab[ijklabcd_t44] =  tmp;
                t4aaab[ijklabcd_t45] =  tmp;
                t4aaab[ijklabcd_t46] =  tmp;
        
                ijklabcd_t51 = Q(j, i, k, l, a, b, c, d, nocc, nvir);
                ijklabcd_t52 = Q(j, i, k, l, b, c, a, d, nocc, nvir);
                ijklabcd_t53 = Q(j, i, k, l, c, a, b, d, nocc, nvir);
                ijklabcd_t54 = Q(j, i, k, l, a, c, b, d, nocc, nvir);
                ijklabcd_t55 = Q(j, i, k, l, b, a, c, d, nocc, nvir);
                ijklabcd_t56 = Q(j, i, k, l, c, b, a, d, nocc, nvir);
        
                t4aaab[ijklabcd_t51] = -tmp;
                t4aaab[ijklabcd_t52] = -tmp;
                t4aaab[ijklabcd_t53] = -tmp;
                t4aaab[ijklabcd_t54] =  tmp;
                t4aaab[ijklabcd_t55] =  tmp;
                t4aaab[ijklabcd_t56] =  tmp;
        
                ijklabcd_t61 = Q(k, j, i, l, a, b, c, d, nocc, nvir);
                ijklabcd_t62 = Q(k, j, i, l, b, c, a, d, nocc, nvir);
                ijklabcd_t63 = Q(k, j, i, l, c, a, b, d, nocc, nvir);
                ijklabcd_t64 = Q(k, j, i, l, a, c, b, d, nocc, nvir);
                ijklabcd_t65 = Q(k, j, i, l, b, a, c, d, nocc, nvir);
                ijklabcd_t66 = Q(k, j, i, l, c, b, a, d, nocc, nvir);
        
                t4aaab[ijklabcd_t61] = -tmp;
                t4aaab[ijklabcd_t62] = -tmp;
                t4aaab[ijklabcd_t63] = -tmp;
                t4aaab[ijklabcd_t64] =  tmp;
                t4aaab[ijklabcd_t65] =  tmp;
                t4aaab[ijklabcd_t66] =  tmp;
            }
        }
        }
    }
    }
    }
    }
    }
    }

    // TODO lsh: reduce symmetry of t4, t3

    //numzero = 1e-3;
    // t4aabb
    m_ijab = nocc*(nocc-1)/2 * nvir*(nvir-1)/2;
    ijab = -1;
    for (b = 1; b < nvir; b++) {
    for (a = 0; a < b; a++) {
    for (j = nocc-1; j > 0; j--) {
    for (i = j-1; i > -1; i--) {
        ijab += 1;
        klcd  =-1;
        for (d = 1; d < nvir; d++) {
        for (c = 0; c < d; c++) {
        for (l = nocc-1; l > 0; l--) {
        for (k = l-1; k > -1; k--) {
            klcd += 1;
            ijabklcd_c = ijab * m_ijab + klcd;
            tmp = c4aabb[ijabklcd_c]; 

//            if(fabs(tmp)-fabs(tmp2) > numzero) 
            if(fabs(tmp) > numzero) 
            {
                tmp2 = 0.0;
                tmp2 = t1xt3aabb(i, j, k, l, a, b, c, d, nocc, nvir, t1, t3aab); 
                tmp2+= t2xt2aabb(i, j, k, l, a, b, c, d, nocc, nvir, t2aa, t2ab); 
                tmp2+= t1xt1xt2aabb(i, j, k, l, a, b, c, d, nocc, nvir, t1, t2aa, t2ab); 
                tmp2+= t1xt1xt1xt1aabb(i, j, k, l, a, b, c, d, nocc, nvir, t1);   // may have bug 

                tmp -= tmp2; 
                //printf("t4 slow %d %d %d %d %d %d %d %d %20.10f \n",i,j,k,l,a,b,c,d,tmp);
                ijklabcd_t11 = Q(i, j, k, l, a, b, c, d, nocc, nvir);
                ijklabcd_t12 = Q(j, i, k, l, b, a, c, d, nocc, nvir);
                ijklabcd_t13 = Q(i, j, k, l, b, a, c, d, nocc, nvir);
                ijklabcd_t14 = Q(j, i, k, l, a, b, c, d, nocc, nvir);
        
                t4aabb[ijklabcd_t11] =  tmp;
                t4aabb[ijklabcd_t12] =  tmp;
                t4aabb[ijklabcd_t13] = -tmp;
                t4aabb[ijklabcd_t14] = -tmp;  
    
                ijklabcd_t21 = Q(i, j, l, k, a, b, d, c, nocc, nvir);
                ijklabcd_t22 = Q(j, i, l, k, b, a, d, c, nocc, nvir);
                ijklabcd_t23 = Q(i, j, l, k, b, a, d, c, nocc, nvir);
                ijklabcd_t24 = Q(j, i, l, k, a, b, d, c, nocc, nvir);
        
                t4aabb[ijklabcd_t21] =  tmp;
                t4aabb[ijklabcd_t22] =  tmp;
                t4aabb[ijklabcd_t23] = -tmp;
                t4aabb[ijklabcd_t24] = -tmp;  
    
                ijklabcd_t31 = Q(i, j, k, l, a, b, d, c, nocc, nvir);
                ijklabcd_t32 = Q(j, i, k, l, b, a, d, c, nocc, nvir);
                ijklabcd_t33 = Q(i, j, k, l, b, a, d, c, nocc, nvir);
                ijklabcd_t34 = Q(j, i, k, l, a, b, d, c, nocc, nvir);
        
                t4aabb[ijklabcd_t31] = -tmp;
                t4aabb[ijklabcd_t32] = -tmp;
                t4aabb[ijklabcd_t33] =  tmp;
                t4aabb[ijklabcd_t34] =  tmp;  
    
                ijklabcd_t41 = Q(i, j, l, k, a, b, c, d, nocc, nvir);
                ijklabcd_t42 = Q(j, i, l, k, b, a, c, d, nocc, nvir);
                ijklabcd_t43 = Q(i, j, l, k, b, a, c, d, nocc, nvir);
                ijklabcd_t44 = Q(j, i, l, k, a, b, c, d, nocc, nvir);
        
                t4aabb[ijklabcd_t41] = -tmp;
                t4aabb[ijklabcd_t42] = -tmp;
                t4aabb[ijklabcd_t43] =  tmp;
                t4aabb[ijklabcd_t44] =  tmp;  
            }
        }
        }
        }
        }
    }
    }
    }
    }

}


void c3_to_t3_thresh(double *t3aaa, double *t3aab, double *c3aaa, double *c3aab, double *t1, double *t2aa, double *t2ab, int nocc, int nvir, double numzero) 
{
    int i, j, k, a, b, c;
    size_t ijkabc_t11, ijkabc_t21, ijkabc_t31, ijkabc_t41, ijkabc_t51, ijkabc_t61;
    size_t ijkabc_t12, ijkabc_t22, ijkabc_t32, ijkabc_t42, ijkabc_t52, ijkabc_t62;
    size_t ijkabc_t13, ijkabc_t23, ijkabc_t33, ijkabc_t43, ijkabc_t53, ijkabc_t63;
    size_t ijkabc_t14, ijkabc_t24, ijkabc_t34, ijkabc_t44, ijkabc_t54, ijkabc_t64;
    size_t ijkabc_t15, ijkabc_t25, ijkabc_t35, ijkabc_t45, ijkabc_t55, ijkabc_t65;
    size_t ijkabc_t16, ijkabc_t26, ijkabc_t36, ijkabc_t46, ijkabc_t56, ijkabc_t66;
    size_t ijab, kc, ijabkc_c, ijkabc_c;

    double tmp, tmp2;

    // t3aaa
    ijkabc_c = -1;
    for (c = 2; c < nvir; c++) {
    for (b = 1; b < c; b++) {
    for (a = 0; a < b; a++) {
    for (k = nocc-1; k > 1; k--) {
    for (j = k-1; j > 0; j--) {
    for (i = j-1; i > -1; i--) {
        ijkabc_c += 1;

//        //lsh dbg
//        printf("c3aaa, %d \n", ijkabc_c);        

        tmp = c3aaa[ijkabc_c]; 

        tmp2 = t1xt2aaa (i, j, k, a, b, c, nocc, nvir, t1, t2aa); 
        tmp2+= t1xt1xt1aaa (i, j, k, a, b, c, nocc, nvir, t1); 

        //if(fabs((tmp-tmp2)/tmp) < numzero) 
        if(fabs(tmp)-fabs(tmp2) > -numzero) 
        {
            tmp -= tmp2; 
            ijkabc_t11 = T(i, j, k, a, b, c, nocc, nvir);
            ijkabc_t12 = T(i, j, k, b, c, a, nocc, nvir);
            ijkabc_t13 = T(i, j, k, c, a, b, nocc, nvir);
            ijkabc_t14 = T(i, j, k, a, c, b, nocc, nvir);
            ijkabc_t15 = T(i, j, k, b, a, c, nocc, nvir);
            ijkabc_t16 = T(i, j, k, c, b, a, nocc, nvir);
    
            t3aaa[ijkabc_t11] =  tmp;
            t3aaa[ijkabc_t12] =  tmp;
            t3aaa[ijkabc_t13] =  tmp;
            t3aaa[ijkabc_t14] = -tmp;
            t3aaa[ijkabc_t15] = -tmp;
            t3aaa[ijkabc_t16] = -tmp;
    
            ijkabc_t21 = T(j, k, i, a, b, c, nocc, nvir);
            ijkabc_t22 = T(j, k, i, b, c, a, nocc, nvir);
            ijkabc_t23 = T(j, k, i, c, a, b, nocc, nvir);
            ijkabc_t24 = T(j, k, i, a, c, b, nocc, nvir);
            ijkabc_t25 = T(j, k, i, b, a, c, nocc, nvir);
            ijkabc_t26 = T(j, k, i, c, b, a, nocc, nvir);
    
            t3aaa[ijkabc_t21] =  tmp;
            t3aaa[ijkabc_t22] =  tmp;
            t3aaa[ijkabc_t23] =  tmp;
            t3aaa[ijkabc_t24] = -tmp;
            t3aaa[ijkabc_t25] = -tmp;
            t3aaa[ijkabc_t26] = -tmp;
    
            ijkabc_t31 = T(k, i, j, a, b, c, nocc, nvir);
            ijkabc_t32 = T(k, i, j, b, c, a, nocc, nvir);
            ijkabc_t33 = T(k, i, j, c, a, b, nocc, nvir);
            ijkabc_t34 = T(k, i, j, a, c, b, nocc, nvir);
            ijkabc_t35 = T(k, i, j, b, a, c, nocc, nvir);
            ijkabc_t36 = T(k, i, j, c, b, a, nocc, nvir);
    
            t3aaa[ijkabc_t31] =  tmp;
            t3aaa[ijkabc_t32] =  tmp;
            t3aaa[ijkabc_t33] =  tmp;
            t3aaa[ijkabc_t34] = -tmp;
            t3aaa[ijkabc_t35] = -tmp;
            t3aaa[ijkabc_t36] = -tmp;
    
            ijkabc_t41 = T(i, k, j, a, b, c, nocc, nvir);
            ijkabc_t42 = T(i, k, j, b, c, a, nocc, nvir);
            ijkabc_t43 = T(i, k, j, c, a, b, nocc, nvir);
            ijkabc_t44 = T(i, k, j, a, c, b, nocc, nvir);
            ijkabc_t45 = T(i, k, j, b, a, c, nocc, nvir);
            ijkabc_t46 = T(i, k, j, c, b, a, nocc, nvir);
    
            t3aaa[ijkabc_t41] = -tmp;
            t3aaa[ijkabc_t42] = -tmp;
            t3aaa[ijkabc_t43] = -tmp;
            t3aaa[ijkabc_t44] =  tmp;
            t3aaa[ijkabc_t45] =  tmp;
            t3aaa[ijkabc_t46] =  tmp;
    
            ijkabc_t51 = T(j, i, k, a, b, c, nocc, nvir);
            ijkabc_t52 = T(j, i, k, b, c, a, nocc, nvir);
            ijkabc_t53 = T(j, i, k, c, a, b, nocc, nvir);
            ijkabc_t54 = T(j, i, k, a, c, b, nocc, nvir);
            ijkabc_t55 = T(j, i, k, b, a, c, nocc, nvir);
            ijkabc_t56 = T(j, i, k, c, b, a, nocc, nvir);
    
            t3aaa[ijkabc_t51] = -tmp;
            t3aaa[ijkabc_t52] = -tmp;
            t3aaa[ijkabc_t53] = -tmp;
            t3aaa[ijkabc_t54] =  tmp;
            t3aaa[ijkabc_t55] =  tmp;
            t3aaa[ijkabc_t56] =  tmp;
    
            ijkabc_t61 = T(k, j, i, a, b, c, nocc, nvir);
            ijkabc_t62 = T(k, j, i, b, c, a, nocc, nvir);
            ijkabc_t63 = T(k, j, i, c, a, b, nocc, nvir);
            ijkabc_t64 = T(k, j, i, a, c, b, nocc, nvir);
            ijkabc_t65 = T(k, j, i, b, a, c, nocc, nvir);
            ijkabc_t66 = T(k, j, i, c, b, a, nocc, nvir);
    
            t3aaa[ijkabc_t61] = -tmp;
            t3aaa[ijkabc_t62] = -tmp;
            t3aaa[ijkabc_t63] = -tmp;
            t3aaa[ijkabc_t64] =  tmp;
            t3aaa[ijkabc_t65] =  tmp;
            t3aaa[ijkabc_t66] =  tmp;
        }
    }
    }
    }
    }
    }
    }

    ijab = -1;
    for (b = 1; b < nvir; b++) {
    for (a = 0; a < b; a++) {
    for (j = nocc-1; j > 0; j--) {
    for (i = j-1; i > -1; i--) {
        ijab += 1;
        kc  =-1;
        for (c = 0; c < nvir; c++) {
        for (k = nocc-1; k > -1; k--) {
            kc += 1;
            ijabkc_c = ijab * nocc*nvir + kc;

//            //lsh dbg
//            printf("c3aab, %d \n", ijabkc_c);        

            tmp = c3aab[ijabkc_c]; 

            tmp2 = t1xt2aab(i, j, k, a, b, c, nocc, nvir, t1, t2aa, t2ab); 
            tmp2+= t1xt1xt1aab(i, j, k, a, b, c, nocc, nvir, t1); 

            //if(fabs((tmp-tmp2)/tmp) < numzero) 
            //if(fabs(fabs(tmp)-fabs(tmp2)) < numzero) 
            if(fabs(tmp)-fabs(tmp2) > -numzero) 
            {
                tmp -= tmp2;    
                ijkabc_t11 = T(i, j, k, a, b, c, nocc, nvir);
                ijkabc_t12 = T(i, j, k, b, a, c, nocc, nvir);
        
                t3aab[ijkabc_t11] =  tmp;
                t3aab[ijkabc_t12] = -tmp;
       
                ijkabc_t21 = T(j, i, k, a, b, c, nocc, nvir);
                ijkabc_t22 = T(j, i, k, b, a, c, nocc, nvir);
       
                t3aab[ijkabc_t21] = -tmp;
                t3aab[ijkabc_t22] =  tmp;
            }
        }
        }
    }
    }
    }
    }

}

void c4_to_t4_thresh(double *t4aaab, double *t4aabb, double *c4aaab, double *c4aabb, double *t1, double *t2aa, double *t2ab, double *t3aaa, double *t3aab, int nocc, int nvir, double numzero) 
{
    int i, j, k, l, a, b, c, d, m_ijab;
    int ijkabc, ld, ijkabcld_c;
    int ijklabcd_t11, ijklabcd_t21, ijklabcd_t31, ijklabcd_t41, ijklabcd_t51, ijklabcd_t61;
    int ijklabcd_t12, ijklabcd_t22, ijklabcd_t32, ijklabcd_t42, ijklabcd_t52, ijklabcd_t62;
    int ijklabcd_t13, ijklabcd_t23, ijklabcd_t33, ijklabcd_t43, ijklabcd_t53, ijklabcd_t63;
    int ijklabcd_t14, ijklabcd_t24, ijklabcd_t34, ijklabcd_t44, ijklabcd_t54, ijklabcd_t64;
    int ijklabcd_t15, ijklabcd_t25, ijklabcd_t35, ijklabcd_t45, ijklabcd_t55, ijklabcd_t65;
    int ijklabcd_t16, ijklabcd_t26, ijklabcd_t36, ijklabcd_t46, ijklabcd_t56, ijklabcd_t66;
    int ijab, klcd, ijabklcd_c;

    double tmp, tmp2;

    // t4aaab
    ijkabc = -1;
    for (c = 2; c < nvir; c++) {
    for (b = 1; b < c; b++) {
    for (a = 0; a < b; a++) {
    for (k = nocc-1; k > 1; k--) {
    for (j = k-1; j > 0; j--) {
    for (i = j-1; i > -1; i--) {
        ijkabc += 1;
        ld = -1;
        for (d = 0; d < nvir; d++) {
        for (l = nocc-1; l > -1; l--) {
            ld += 1;
            ijkabcld_c = ijkabc * nocc*nvir + ld;
            tmp = c4aaab[ijkabcld_c]; 

            tmp2 = t1xt3aaab (i, j, k, l, a, b, c, d, nocc, nvir, t1, t3aaa, t3aab);   // may have 1e-5 bug 
            tmp2+= t2xt2aaab (i, j, k, l, a, b, c, d, nocc, nvir, t2aa, t2ab);         // may have 1e-3 bug 
            tmp2+= t1xt1xt2aaab (i, j, k, l, a, b, c, d, nocc, nvir, t1, t2aa, t2ab);  // may have 1e-5 bug 
            tmp2+= t1xt1xt1xt1aaab (i, j, k, l, a, b, c, d, nocc, nvir, t1);           // may have 1e-6 bug

            //if(fabs((tmp-tmp2)/tmp) < numzero) 
            //if(fabs(fabs(tmp)-fabs(tmp2)) < numzero) 
            if(fabs(tmp)-fabs(tmp2) > -numzero) 
            {
                tmp -= tmp2; 
                ijklabcd_t11 = Q(i, j, k, l, a, b, c, d, nocc, nvir);
                ijklabcd_t12 = Q(i, j, k, l, b, c, a, d, nocc, nvir);
                ijklabcd_t13 = Q(i, j, k, l, c, a, b, d, nocc, nvir);
                ijklabcd_t14 = Q(i, j, k, l, a, c, b, d, nocc, nvir);
                ijklabcd_t15 = Q(i, j, k, l, b, a, c, d, nocc, nvir);
                ijklabcd_t16 = Q(i, j, k, l, c, b, a, d, nocc, nvir);
        
                t4aaab[ijklabcd_t11] =  tmp;
                t4aaab[ijklabcd_t12] =  tmp;
                t4aaab[ijklabcd_t13] =  tmp;
                t4aaab[ijklabcd_t14] = -tmp;
                t4aaab[ijklabcd_t15] = -tmp;
                t4aaab[ijklabcd_t16] = -tmp;
        
                ijklabcd_t21 = Q(j, k, i, l, a, b, c, d, nocc, nvir);
                ijklabcd_t22 = Q(j, k, i, l, b, c, a, d, nocc, nvir);
                ijklabcd_t23 = Q(j, k, i, l, c, a, b, d, nocc, nvir);
                ijklabcd_t24 = Q(j, k, i, l, a, c, b, d, nocc, nvir);
                ijklabcd_t25 = Q(j, k, i, l, b, a, c, d, nocc, nvir);
                ijklabcd_t26 = Q(j, k, i, l, c, b, a, d, nocc, nvir);
        
                t4aaab[ijklabcd_t21] =  tmp;
                t4aaab[ijklabcd_t22] =  tmp;
                t4aaab[ijklabcd_t23] =  tmp;
                t4aaab[ijklabcd_t24] = -tmp;
                t4aaab[ijklabcd_t25] = -tmp;
                t4aaab[ijklabcd_t26] = -tmp;
        
                ijklabcd_t31 = Q(k, i, j, l, a, b, c, d, nocc, nvir);
                ijklabcd_t32 = Q(k, i, j, l, b, c, a, d, nocc, nvir);
                ijklabcd_t33 = Q(k, i, j, l, c, a, b, d, nocc, nvir);
                ijklabcd_t34 = Q(k, i, j, l, a, c, b, d, nocc, nvir);
                ijklabcd_t35 = Q(k, i, j, l, b, a, c, d, nocc, nvir);
                ijklabcd_t36 = Q(k, i, j, l, c, b, a, d, nocc, nvir);
        
                t4aaab[ijklabcd_t31] =  tmp;
                t4aaab[ijklabcd_t32] =  tmp;
                t4aaab[ijklabcd_t33] =  tmp;
                t4aaab[ijklabcd_t34] = -tmp;
                t4aaab[ijklabcd_t35] = -tmp;
                t4aaab[ijklabcd_t36] = -tmp;
        
                ijklabcd_t41 = Q(i, k, j, l, a, b, c, d, nocc, nvir);
                ijklabcd_t42 = Q(i, k, j, l, b, c, a, d, nocc, nvir);
                ijklabcd_t43 = Q(i, k, j, l, c, a, b, d, nocc, nvir);
                ijklabcd_t44 = Q(i, k, j, l, a, c, b, d, nocc, nvir);
                ijklabcd_t45 = Q(i, k, j, l, b, a, c, d, nocc, nvir);
                ijklabcd_t46 = Q(i, k, j, l, c, b, a, d, nocc, nvir);
        
                t4aaab[ijklabcd_t41] = -tmp;
                t4aaab[ijklabcd_t42] = -tmp;
                t4aaab[ijklabcd_t43] = -tmp;
                t4aaab[ijklabcd_t44] =  tmp;
                t4aaab[ijklabcd_t45] =  tmp;
                t4aaab[ijklabcd_t46] =  tmp;
        
                ijklabcd_t51 = Q(j, i, k, l, a, b, c, d, nocc, nvir);
                ijklabcd_t52 = Q(j, i, k, l, b, c, a, d, nocc, nvir);
                ijklabcd_t53 = Q(j, i, k, l, c, a, b, d, nocc, nvir);
                ijklabcd_t54 = Q(j, i, k, l, a, c, b, d, nocc, nvir);
                ijklabcd_t55 = Q(j, i, k, l, b, a, c, d, nocc, nvir);
                ijklabcd_t56 = Q(j, i, k, l, c, b, a, d, nocc, nvir);
        
                t4aaab[ijklabcd_t51] = -tmp;
                t4aaab[ijklabcd_t52] = -tmp;
                t4aaab[ijklabcd_t53] = -tmp;
                t4aaab[ijklabcd_t54] =  tmp;
                t4aaab[ijklabcd_t55] =  tmp;
                t4aaab[ijklabcd_t56] =  tmp;
        
                ijklabcd_t61 = Q(k, j, i, l, a, b, c, d, nocc, nvir);
                ijklabcd_t62 = Q(k, j, i, l, b, c, a, d, nocc, nvir);
                ijklabcd_t63 = Q(k, j, i, l, c, a, b, d, nocc, nvir);
                ijklabcd_t64 = Q(k, j, i, l, a, c, b, d, nocc, nvir);
                ijklabcd_t65 = Q(k, j, i, l, b, a, c, d, nocc, nvir);
                ijklabcd_t66 = Q(k, j, i, l, c, b, a, d, nocc, nvir);
        
                t4aaab[ijklabcd_t61] = -tmp;
                t4aaab[ijklabcd_t62] = -tmp;
                t4aaab[ijklabcd_t63] = -tmp;
                t4aaab[ijklabcd_t64] =  tmp;
                t4aaab[ijklabcd_t65] =  tmp;
                t4aaab[ijklabcd_t66] =  tmp;
            }
        }
        }
    }
    }
    }
    }
    }
    }

    // TODO lsh: reduce symmetry of t4, t3

    // t4aabb
    m_ijab = nocc*(nocc-1)/2 * nvir*(nvir-1)/2;
    ijab = -1;
    for (b = 1; b < nvir; b++) {
    for (a = 0; a < b; a++) {
    for (j = nocc-1; j > 0; j--) {
    for (i = j-1; i > -1; i--) {
        ijab += 1;
        klcd  =-1;
        for (d = 1; d < nvir; d++) {
        for (c = 0; c < d; c++) {
        for (l = nocc-1; l > 0; l--) {
        for (k = l-1; k > -1; k--) {
            klcd += 1;
            ijabklcd_c = ijab * m_ijab + klcd;
            tmp = c4aabb[ijabklcd_c]; 

            tmp2 = t1xt3aabb(i, j, k, l, a, b, c, d, nocc, nvir, t1, t3aab); 
            tmp2+= t2xt2aabb(i, j, k, l, a, b, c, d, nocc, nvir, t2aa, t2ab); 
            tmp2+= t1xt1xt2aabb(i, j, k, l, a, b, c, d, nocc, nvir, t1, t2aa, t2ab); 
            tmp2+= t1xt1xt1xt1aabb(i, j, k, l, a, b, c, d, nocc, nvir, t1);   // may have bug 

            //if(fabs((tmp-tmp2)/tmp) < numzero) 
            //if(fabs(fabs(tmp)-fabs(tmp2)) < numzero) 
            if(fabs(tmp)-fabs(tmp2) > -numzero) 
            {
                tmp -= tmp2; 
                ijklabcd_t11 = Q(i, j, k, l, a, b, c, d, nocc, nvir);
                ijklabcd_t12 = Q(j, i, k, l, b, a, c, d, nocc, nvir);
                ijklabcd_t13 = Q(i, j, k, l, b, a, c, d, nocc, nvir);
                ijklabcd_t14 = Q(j, i, k, l, a, b, c, d, nocc, nvir);
        
                t4aabb[ijklabcd_t11] =  tmp;
                t4aabb[ijklabcd_t12] =  tmp;
                t4aabb[ijklabcd_t13] = -tmp;
                t4aabb[ijklabcd_t14] = -tmp;  
    
                ijklabcd_t21 = Q(i, j, l, k, a, b, d, c, nocc, nvir);
                ijklabcd_t22 = Q(j, i, l, k, b, a, d, c, nocc, nvir);
                ijklabcd_t23 = Q(i, j, l, k, b, a, d, c, nocc, nvir);
                ijklabcd_t24 = Q(j, i, l, k, a, b, d, c, nocc, nvir);
        
                t4aabb[ijklabcd_t21] =  tmp;
                t4aabb[ijklabcd_t22] =  tmp;
                t4aabb[ijklabcd_t23] = -tmp;
                t4aabb[ijklabcd_t24] = -tmp;  
    
                ijklabcd_t31 = Q(i, j, k, l, a, b, d, c, nocc, nvir);
                ijklabcd_t32 = Q(j, i, k, l, b, a, d, c, nocc, nvir);
                ijklabcd_t33 = Q(i, j, k, l, b, a, d, c, nocc, nvir);
                ijklabcd_t34 = Q(j, i, k, l, a, b, d, c, nocc, nvir);
        
                t4aabb[ijklabcd_t31] = -tmp;
                t4aabb[ijklabcd_t32] = -tmp;
                t4aabb[ijklabcd_t33] =  tmp;
                t4aabb[ijklabcd_t34] =  tmp;  
    
                ijklabcd_t41 = Q(i, j, l, k, a, b, c, d, nocc, nvir);
                ijklabcd_t42 = Q(j, i, l, k, b, a, c, d, nocc, nvir);
                ijklabcd_t43 = Q(i, j, l, k, b, a, c, d, nocc, nvir);
                ijklabcd_t44 = Q(j, i, l, k, a, b, c, d, nocc, nvir);
        
                t4aabb[ijklabcd_t41] = -tmp;
                t4aabb[ijklabcd_t42] = -tmp;
                t4aabb[ijklabcd_t43] =  tmp;
                t4aabb[ijklabcd_t44] =  tmp;  
            }
        }
        }
        }
        }
    }
    }
    }
    }

}

int alpha_count(uint8_t det[], const uint8_t qi)
{
    int ncount = 0;
    int i;
    for (i=0; i<qi; i++)
        if (det[i] == 3 || det[i] == 1)
            ncount += 1;
    return ncount;
}

double parity_ab_str(uint8_t det[], int nmo)
{
    // | 1_alpha 1_beta ... noc_alpha noc_beta > 
    // = (-1)**n * | 1_beta ...  noc_beta > | 1_alpha ...  noc_alpha > 
    // = (-1)**n * | noc_beta ...  1_beta > | noc_alpha ...  1_alpha > 
    int n=0, i;
    for (i=0; i<nmo; i++)
        if (det[i]==3 || det[i]==2)
           n += alpha_count(det, i);
    return pow((double)-1, n);
    //return 1;
}

double parity_ci_to_cc(int sum_ijkl, int n_excite, int nocc)
{
    // For example of singly-excited configuration, parity is
    // | a noc noc-1 ... i+1 i-1 ... 2 1 > = parity_ci_to_cc * | 1 2 ... i-1 a i+1 ... noc-1 noc >
    return pow((double)-1, n_excite * nocc - sum_ijkl - n_excite*(n_excite+1)/2);
}

typedef struct _darray
{
    size_t size;
    size_t actual_size;
    double *content;
} darray;

void darray_create(darray *d)
{

    d->actual_size = d->size = 0;
    d->content = NULL;
}

void darray_append(darray *d, double v)
{
    if (d->size+1 > d->actual_size)
    {
    size_t new_size;
    if (!d->actual_size) 
    { 
        new_size = 1;
    }
    else
    {
        new_size = d->actual_size * 2;
    }
    double *temp = realloc(d->content, sizeof(double) * new_size);
    if (!temp)
    {
        fprintf(stderr, "Failed to extend array (new_size=%zu)\n", new_size);
        exit(EXIT_FAILURE);
    }
    d->actual_size = new_size;
    d->content = temp;
    }
    d->content[d->size] = v;
    d->size++;
}

const double* darray_data(darray *d)
{
    return d->content;
}

void darray_destroy(darray *d)
{
    free(d->content);
    d->content = NULL;
    d->size = d->actual_size = 0;
}

size_t darray_size(darray *d)
{
    return d->size;
}

typedef struct _iarray
{
    size_t size;
    size_t actual_size;
    int *content;
} iarray;

void iarray_create(iarray *d)
{

    d->actual_size = d->size = 0;
    d->content = NULL;
}

void iarray_append(iarray *d, int v)
{
    if (d->size+1 > d->actual_size)
    {
    size_t new_size;
    if (!d->actual_size) 
    { 
        new_size = 1;
    }
    else
    {
        new_size = d->actual_size * 2;
    }
    int *temp = realloc(d->content, sizeof(int) * new_size);
    if (!temp)
    {
        fprintf(stderr, "Failed to extend array (new_size=%zu)\n", new_size);
        exit(EXIT_FAILURE);
    }
    d->actual_size = new_size;
    d->content = temp;
    }
    d->content[d->size] = v;
    d->size++;
}

const int* iarray_data(iarray *d)
{
    return d->content;
}

void iarray_destroy(iarray *d)
{
    free(d->content);
    d->content = NULL;
    d->size = d->actual_size = 0;
}

size_t iarray_size(iarray *d)
{
    return d->size;
}

//void t2t4c(double *t2t4c, int *p_aabb, int *q_aabb, int *r_aabb, int *s_aabb, int *t_aabb, int *u_aabb, int *v_aabb, int *w_aabb, double *c4_aabb, int *p_aaab, int *q_aaab, int *r_aaab, int *s_aaab, int *t_aaab, int *u_aaab, int *v_aaab, int *w_aaab, double *c4_aaab, double *t1, double *t2aa, double *t2ab, double *t3aaa, double *t3aab, double *e2ovov, double *tmp, int n_aabb, int n_aaab, int nocc, int nvir, double numzero, double c0) 
void t2t4c_shci(double *t2t4c, double *t1, double *t2aa, double *t2ab, double *t3aaa, double *t3aab, double *e2ovov, const int nc, const int nocc, const int nvir, const double numzero, const double c0, double norm) 
{
    //double numzero = 1e-7;
    int p, q, r, s, t, u, v, w, itmp, it, jt, at, bt;
    double t4, parity, scale;
    uint8_t Refdet[nocc+nvir], det_str[nocc+nvir];
    for (itmp = 0; itmp < nocc+nvir; itmp++){
        if (itmp<nocc) Refdet[itmp] = 3;  
        else           Refdet[itmp] = 0;
    }

    double norm0SDT = norm;

    double ****tmp;
    tmp = (double ****)malloc(sizeof(double ***) * nocc); 
    for (it=0; it< nocc; it++){
        tmp[it] = (double ***)malloc(sizeof(double **) * nocc);
        for (jt=0; jt< nocc; jt++){
            tmp[it][jt] = (double **)malloc(sizeof(double *) * nvir);
            for (at=0; at< nvir; at++){
                tmp[it][jt][at] = (double *)malloc(sizeof(double) * nvir);
            }
        }
    }

//    for (p=0; p<nocc; p++) {
//    for (q=0; q<nocc; q++) {
//    for (r=0; r<nvir; r++) {
//    for (s=0; s<nvir; s++) {
//        printf("%d %d %d %d %20.10lf\n",p,q,r,s,e2ovov[De(p,r,q,s,nocc,nvir)]); 
//    }
//    }
//    }
//    }

    FILE *fp;
    char typ[4], line[255];
    fp = fopen("CIcoeff_shci.out", "r");
    fscanf(fp, "%s\n", line);
    if (fp) {
       while ( !feof(fp) ){
           fscanf(fp, "%c%c%c%c,%s\n", &(typ[0]), &(typ[1]), &(typ[2]), &(typ[3]), line);
           fscanf(fp, "%lf\n", &t4);
           //printf ("typ=%c%c%c%c line=%s\n",typ[0],typ[1],typ[2],typ[3], line);
           if (strncmp(typ, "aabb", 4) == 0 && fabs(t4) > numzero){
               norm += t4*t4;
               sscanf(line,"%d,%d,%d,%d,%d,%d,%d,%d",&p,&q,&t,&u,&r,&s,&v,&w);
               p += nc;
               q += nc;
               r += nc;
               s += nc;
               t += - nocc + nc;
               u += - nocc + nc;
               v += - nocc + nc;
               w += - nocc + nc;

//               if(!(p == 2 && q == 3 && t == 0 && u == 1 && \
//                    r == 2 && s == 3 && v == 0 && w == 1)) continue;
   
               for (itmp = 0; itmp < nocc+nvir; itmp++)
                   det_str[itmp] = Refdet[itmp];  
   
               det_str[p] = 2;
               det_str[q] = 2;
               det_str[t+nocc] = 1;
               det_str[u+nocc] = 1;
   
               if (p != r && q != r) det_str[r] = 1;
               else  det_str[r] = 0;
               if (p != s && q != s) det_str[s] = 1;
               else  det_str[s] = 0;
               if (t != v && u != v) det_str[v+nocc] = 2;
               else  det_str[v+nocc] = 3;
               if (t != w && u != w) det_str[w+nocc] = 2;
               else  det_str[w+nocc] = 3;
   
               //parity  = parity_ab_str(det_str, nocc+nvir);
               parity = parity_ci_to_cc(p+q, 2, nocc);
               parity *= parity_ci_to_cc(r+s, 2, nocc);
   
               // interm norm of c4
               t4 = parity * t4 / c0;
               // lsh test 
//               printf("c4 mem %20.10f \n",t4);
   
               // extract t4 
               t4-= t1xt3aabb(p, q, r, s, t, u, v, w, nocc, nvir, t1, t3aab); 
               t4-= t2xt2aabb(p, q, r, s, t, u, v, w, nocc, nvir, t2aa, t2ab); 
               t4-= t1xt1xt2aabb(p, q, r, s, t, u, v, w, nocc, nvir, t1, t2aa, t2ab); 
               t4-= t1xt1xt1xt1aabb(p, q, r, s, t, u, v, w, nocc, nvir, t1);   // may have bug 
   
               // lsh test 
//               printf("t4 mem %20.10f \n",t4);
   
               for (it=0; it< nocc; it++){
               for (jt=0; jt< nocc; jt++){
               for (at=0; at< nvir; at++){
               for (bt=0; bt< nvir; bt++){
                   tmp[it][jt][at][bt] = 0.0;
               }
               }
               }
               }
//               printf("eris_ovov mem %20.10f \n",e2ovov[De(p,t,r,v,nocc,nvir)]);
   
               if (p<r && t<v) tmp[q][s][u][w] += e2ovov[De(p,t,r,v,nocc,nvir)] * t4;
               if (q<r && t<v) tmp[p][s][u][w] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t4;
               if (p<s && t<v) tmp[q][r][u][w] -= e2ovov[De(p,t,s,v,nocc,nvir)] * t4;
               if (q<s && t<v) tmp[p][r][u][w] += e2ovov[De(q,t,s,v,nocc,nvir)] * t4;
               if (p<r && u<v) tmp[q][s][t][w] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t4;
               if (q<r && u<v) tmp[p][s][t][w] += e2ovov[De(q,u,r,v,nocc,nvir)] * t4;
               if (p<s && u<v) tmp[q][r][t][w] += e2ovov[De(p,u,s,v,nocc,nvir)] * t4;
               if (q<s && u<v) tmp[p][r][t][w] -= e2ovov[De(q,u,s,v,nocc,nvir)] * t4;
               if (p<r && t<w) tmp[q][s][u][v] -= e2ovov[De(p,t,r,w,nocc,nvir)] * t4;
               if (q<r && t<w) tmp[p][s][u][v] += e2ovov[De(q,t,r,w,nocc,nvir)] * t4;
               if (p<s && t<w) tmp[q][r][u][v] += e2ovov[De(p,t,s,w,nocc,nvir)] * t4;
               if (q<s && t<w) tmp[p][r][u][v] -= e2ovov[De(q,t,s,w,nocc,nvir)] * t4;
               if (p<r && u<w) tmp[q][s][t][v] += e2ovov[De(p,u,r,w,nocc,nvir)] * t4;
               if (q<r && u<w) tmp[p][s][t][v] -= e2ovov[De(q,u,r,w,nocc,nvir)] * t4;
               if (p<s && u<w) tmp[q][r][t][v] -= e2ovov[De(p,u,s,w,nocc,nvir)] * t4;
               if (q<s && u<w) tmp[p][r][t][v] += e2ovov[De(q,u,s,w,nocc,nvir)] * t4;
               if (p<r && v<t) tmp[q][s][u][w] += e2ovov[De(p,t,r,v,nocc,nvir)] * t4;
               if (q<r && v<t) tmp[p][s][u][w] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t4;
               if (p<s && v<t) tmp[q][r][u][w] -= e2ovov[De(p,t,s,v,nocc,nvir)] * t4;
               if (q<s && v<t) tmp[p][r][u][w] += e2ovov[De(q,t,s,v,nocc,nvir)] * t4;
               if (p<r && v<u) tmp[q][s][t][w] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t4;
               if (q<r && v<u) tmp[p][s][t][w] += e2ovov[De(q,u,r,v,nocc,nvir)] * t4;
               if (p<s && v<u) tmp[q][r][t][w] += e2ovov[De(p,u,s,v,nocc,nvir)] * t4;
               if (q<s && v<u) tmp[p][r][t][w] -= e2ovov[De(q,u,s,v,nocc,nvir)] * t4;
               if (p<r && w<t) tmp[q][s][u][v] -= e2ovov[De(p,t,r,w,nocc,nvir)] * t4;
               if (q<r && w<t) tmp[p][s][u][v] += e2ovov[De(q,t,r,w,nocc,nvir)] * t4;
               if (p<s && w<t) tmp[q][r][u][v] += e2ovov[De(p,t,s,w,nocc,nvir)] * t4;
               if (q<s && w<t) tmp[p][r][u][v] -= e2ovov[De(q,t,s,w,nocc,nvir)] * t4;
               if (p<r && w<u) tmp[q][s][t][v] += e2ovov[De(p,u,r,w,nocc,nvir)] * t4;
               if (q<r && w<u) tmp[p][s][t][v] -= e2ovov[De(q,u,r,w,nocc,nvir)] * t4;
               if (p<s && w<u) tmp[q][r][t][v] -= e2ovov[De(p,u,s,w,nocc,nvir)] * t4;
               if (q<s && w<u) tmp[p][r][t][v] += e2ovov[De(q,u,s,w,nocc,nvir)] * t4;
   
               scale = 0.5;
               if (p==r && t<v) tmp[q][s][u][w] += e2ovov[De(p,t,r,v,nocc,nvir)] * t4 * scale;
               if (q==r && t<v) tmp[p][s][u][w] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t4 * scale;
               if (p==s && t<v) tmp[q][r][u][w] -= e2ovov[De(p,t,s,v,nocc,nvir)] * t4 * scale;
               if (q==s && t<v) tmp[p][r][u][w] += e2ovov[De(q,t,s,v,nocc,nvir)] * t4 * scale;
               if (p==r && u<v) tmp[q][s][t][w] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t4 * scale;
               if (q==r && u<v) tmp[p][s][t][w] += e2ovov[De(q,u,r,v,nocc,nvir)] * t4 * scale;
               if (p==s && u<v) tmp[q][r][t][w] += e2ovov[De(p,u,s,v,nocc,nvir)] * t4 * scale;
               if (q==s && u<v) tmp[p][r][t][w] -= e2ovov[De(q,u,s,v,nocc,nvir)] * t4 * scale;
               if (p==r && t<w) tmp[q][s][u][v] -= e2ovov[De(p,t,r,w,nocc,nvir)] * t4 * scale;
               if (q==r && t<w) tmp[p][s][u][v] += e2ovov[De(q,t,r,w,nocc,nvir)] * t4 * scale;
               if (p==s && t<w) tmp[q][r][u][v] += e2ovov[De(p,t,s,w,nocc,nvir)] * t4 * scale;
               if (q==s && t<w) tmp[p][r][u][v] -= e2ovov[De(q,t,s,w,nocc,nvir)] * t4 * scale;
               if (p==r && u<w) tmp[q][s][t][v] += e2ovov[De(p,u,r,w,nocc,nvir)] * t4 * scale;
               if (q==r && u<w) tmp[p][s][t][v] -= e2ovov[De(q,u,r,w,nocc,nvir)] * t4 * scale;
               if (p==s && u<w) tmp[q][r][t][v] -= e2ovov[De(p,u,s,w,nocc,nvir)] * t4 * scale;
               if (q==s && u<w) tmp[p][r][t][v] += e2ovov[De(q,u,s,w,nocc,nvir)] * t4 * scale;
               if (p==r && v<t) tmp[q][s][u][w] += e2ovov[De(p,t,r,v,nocc,nvir)] * t4 * scale;
               if (q==r && v<t) tmp[p][s][u][w] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t4 * scale;
               if (p==s && v<t) tmp[q][r][u][w] -= e2ovov[De(p,t,s,v,nocc,nvir)] * t4 * scale;
               if (q==s && v<t) tmp[p][r][u][w] += e2ovov[De(q,t,s,v,nocc,nvir)] * t4 * scale;
               if (p==r && v<u) tmp[q][s][t][w] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t4 * scale;
               if (q==r && v<u) tmp[p][s][t][w] += e2ovov[De(q,u,r,v,nocc,nvir)] * t4 * scale;
               if (p==s && v<u) tmp[q][r][t][w] += e2ovov[De(p,u,s,v,nocc,nvir)] * t4 * scale;
               if (q==s && v<u) tmp[p][r][t][w] -= e2ovov[De(q,u,s,v,nocc,nvir)] * t4 * scale;
               if (p==r && w<t) tmp[q][s][u][v] -= e2ovov[De(p,t,r,w,nocc,nvir)] * t4 * scale;
               if (q==r && w<t) tmp[p][s][u][v] += e2ovov[De(q,t,r,w,nocc,nvir)] * t4 * scale;
               if (p==s && w<t) tmp[q][r][u][v] += e2ovov[De(p,t,s,w,nocc,nvir)] * t4 * scale;
               if (q==s && w<t) tmp[p][r][u][v] -= e2ovov[De(q,t,s,w,nocc,nvir)] * t4 * scale;
               if (p==r && w<u) tmp[q][s][t][v] += e2ovov[De(p,u,r,w,nocc,nvir)] * t4 * scale;
               if (q==r && w<u) tmp[p][s][t][v] -= e2ovov[De(q,u,r,w,nocc,nvir)] * t4 * scale;
               if (p==s && w<u) tmp[q][r][t][v] -= e2ovov[De(p,u,s,w,nocc,nvir)] * t4 * scale;
               if (q==s && w<u) tmp[p][r][t][v] += e2ovov[De(q,u,s,w,nocc,nvir)] * t4 * scale;
   
               scale = 0.5;
               if (p<r && t==v) tmp[q][s][u][w] += e2ovov[De(p,t,r,v,nocc,nvir)] * t4 * scale;
               if (q<r && t==v) tmp[p][s][u][w] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t4 * scale;
               if (p<s && t==v) tmp[q][r][u][w] -= e2ovov[De(p,t,s,v,nocc,nvir)] * t4 * scale;
               if (q<s && t==v) tmp[p][r][u][w] += e2ovov[De(q,t,s,v,nocc,nvir)] * t4 * scale;
               if (p<r && u==v) tmp[q][s][t][w] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t4 * scale;
               if (q<r && u==v) tmp[p][s][t][w] += e2ovov[De(q,u,r,v,nocc,nvir)] * t4 * scale;
               if (p<s && u==v) tmp[q][r][t][w] += e2ovov[De(p,u,s,v,nocc,nvir)] * t4 * scale;
               if (q<s && u==v) tmp[p][r][t][w] -= e2ovov[De(q,u,s,v,nocc,nvir)] * t4 * scale;
               if (p<r && t==w) tmp[q][s][u][v] -= e2ovov[De(p,t,r,w,nocc,nvir)] * t4 * scale;
               if (q<r && t==w) tmp[p][s][u][v] += e2ovov[De(q,t,r,w,nocc,nvir)] * t4 * scale;
               if (p<s && t==w) tmp[q][r][u][v] += e2ovov[De(p,t,s,w,nocc,nvir)] * t4 * scale;
               if (q<s && t==w) tmp[p][r][u][v] -= e2ovov[De(q,t,s,w,nocc,nvir)] * t4 * scale;
               if (p<r && u==w) tmp[q][s][t][v] += e2ovov[De(p,u,r,w,nocc,nvir)] * t4 * scale;
               if (q<r && u==w) tmp[p][s][t][v] -= e2ovov[De(q,u,r,w,nocc,nvir)] * t4 * scale;
               if (p<s && u==w) tmp[q][r][t][v] -= e2ovov[De(p,u,s,w,nocc,nvir)] * t4 * scale;
               if (q<s && u==w) tmp[p][r][t][v] += e2ovov[De(q,u,s,w,nocc,nvir)] * t4 * scale;
               if (p<r && v==t) tmp[q][s][u][w] += e2ovov[De(p,t,r,v,nocc,nvir)] * t4 * scale;
               if (q<r && v==t) tmp[p][s][u][w] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t4 * scale;
               if (p<s && v==t) tmp[q][r][u][w] -= e2ovov[De(p,t,s,v,nocc,nvir)] * t4 * scale;
               if (q<s && v==t) tmp[p][r][u][w] += e2ovov[De(q,t,s,v,nocc,nvir)] * t4 * scale;
               if (p<r && v==u) tmp[q][s][t][w] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t4 * scale;
               if (q<r && v==u) tmp[p][s][t][w] += e2ovov[De(q,u,r,v,nocc,nvir)] * t4 * scale;
               if (p<s && v==u) tmp[q][r][t][w] += e2ovov[De(p,u,s,v,nocc,nvir)] * t4 * scale;
               if (q<s && v==u) tmp[p][r][t][w] -= e2ovov[De(q,u,s,v,nocc,nvir)] * t4 * scale;
               if (p<r && w==t) tmp[q][s][u][v] -= e2ovov[De(p,t,r,w,nocc,nvir)] * t4 * scale;
               if (q<r && w==t) tmp[p][s][u][v] += e2ovov[De(q,t,r,w,nocc,nvir)] * t4 * scale;
               if (p<s && w==t) tmp[q][r][u][v] += e2ovov[De(p,t,s,w,nocc,nvir)] * t4 * scale;
               if (q<s && w==t) tmp[p][r][u][v] -= e2ovov[De(q,t,s,w,nocc,nvir)] * t4 * scale;
               if (p<r && w==u) tmp[q][s][t][v] += e2ovov[De(p,u,r,w,nocc,nvir)] * t4 * scale;
               if (q<r && w==u) tmp[p][s][t][v] -= e2ovov[De(q,u,r,w,nocc,nvir)] * t4 * scale;
               if (p<s && w==u) tmp[q][r][t][v] -= e2ovov[De(p,u,s,w,nocc,nvir)] * t4 * scale;
               if (q<s && w==u) tmp[p][r][t][v] += e2ovov[De(q,u,s,w,nocc,nvir)] * t4 * scale;
   
               scale = 0.25;
               if (p==r && t==v) tmp[q][s][u][w] += e2ovov[De(p,t,r,v,nocc,nvir)] * t4 * scale;
               if (q==r && t==v) tmp[p][s][u][w] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t4 * scale;
               if (p==s && t==v) tmp[q][r][u][w] -= e2ovov[De(p,t,s,v,nocc,nvir)] * t4 * scale;
               if (q==s && t==v) tmp[p][r][u][w] += e2ovov[De(q,t,s,v,nocc,nvir)] * t4 * scale;
               if (p==r && u==v) tmp[q][s][t][w] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t4 * scale;
               if (q==r && u==v) tmp[p][s][t][w] += e2ovov[De(q,u,r,v,nocc,nvir)] * t4 * scale;
               if (p==s && u==v) tmp[q][r][t][w] += e2ovov[De(p,u,s,v,nocc,nvir)] * t4 * scale;
               if (q==s && u==v) tmp[p][r][t][w] -= e2ovov[De(q,u,s,v,nocc,nvir)] * t4 * scale;
               if (p==r && t==w) tmp[q][s][u][v] -= e2ovov[De(p,t,r,w,nocc,nvir)] * t4 * scale;
               if (q==r && t==w) tmp[p][s][u][v] += e2ovov[De(q,t,r,w,nocc,nvir)] * t4 * scale;
               if (p==s && t==w) tmp[q][r][u][v] += e2ovov[De(p,t,s,w,nocc,nvir)] * t4 * scale;
               if (q==s && t==w) tmp[p][r][u][v] -= e2ovov[De(q,t,s,w,nocc,nvir)] * t4 * scale;
               if (p==r && u==w) tmp[q][s][t][v] += e2ovov[De(p,u,r,w,nocc,nvir)] * t4 * scale;
               if (q==r && u==w) tmp[p][s][t][v] -= e2ovov[De(q,u,r,w,nocc,nvir)] * t4 * scale;
               if (p==s && u==w) tmp[q][r][t][v] -= e2ovov[De(p,u,s,w,nocc,nvir)] * t4 * scale;
               if (q==s && u==w) tmp[p][r][t][v] += e2ovov[De(q,u,s,w,nocc,nvir)] * t4 * scale;
               if (p==r && v==t) tmp[q][s][u][w] += e2ovov[De(p,t,r,v,nocc,nvir)] * t4 * scale;
               if (q==r && v==t) tmp[p][s][u][w] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t4 * scale;
               if (p==s && v==t) tmp[q][r][u][w] -= e2ovov[De(p,t,s,v,nocc,nvir)] * t4 * scale;
               if (q==s && v==t) tmp[p][r][u][w] += e2ovov[De(q,t,s,v,nocc,nvir)] * t4 * scale;
               if (p==r && v==u) tmp[q][s][t][w] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t4 * scale;
               if (q==r && v==u) tmp[p][s][t][w] += e2ovov[De(q,u,r,v,nocc,nvir)] * t4 * scale;
               if (p==s && v==u) tmp[q][r][t][w] += e2ovov[De(p,u,s,v,nocc,nvir)] * t4 * scale;
               if (q==s && v==u) tmp[p][r][t][w] -= e2ovov[De(q,u,s,v,nocc,nvir)] * t4 * scale;
               if (p==r && w==t) tmp[q][s][u][v] -= e2ovov[De(p,t,r,w,nocc,nvir)] * t4 * scale;
               if (q==r && w==t) tmp[p][s][u][v] += e2ovov[De(q,t,r,w,nocc,nvir)] * t4 * scale;
               if (p==s && w==t) tmp[q][r][u][v] += e2ovov[De(p,t,s,w,nocc,nvir)] * t4 * scale;
               if (q==s && w==t) tmp[p][r][u][v] -= e2ovov[De(q,t,s,w,nocc,nvir)] * t4 * scale;
               if (p==r && w==u) tmp[q][s][t][v] += e2ovov[De(p,u,r,w,nocc,nvir)] * t4 * scale;
               if (q==r && w==u) tmp[p][s][t][v] -= e2ovov[De(q,u,r,w,nocc,nvir)] * t4 * scale;
               if (p==s && w==u) tmp[q][r][t][v] -= e2ovov[De(p,u,s,w,nocc,nvir)] * t4 * scale;
               if (q==s && w==u) tmp[p][r][t][v] += e2ovov[De(q,u,s,w,nocc,nvir)] * t4 * scale;
   
               for (it = 0; it < nocc; it++)
               for (jt = 0; jt < nocc; jt++)
               for (at = 0; at < nvir; at++)
               for (bt = 0; bt < nvir; bt++)
                   t2t4c[D(it,jt,at,bt,nocc,nvir)] += tmp[it][jt][at][bt] + tmp[jt][it][bt][at];

           }
           else if (strncmp(typ, "aaab", 4) == 0 && fabs(t4) > numzero){
               norm += 2.0*t4*t4; 
               //lsh test
               //printf ("typ=%c%c%c%c line=%s c4=%lf\n",typ[0],typ[1],typ[2],typ[3], line, t4);

               sscanf(line,"%d,%d,%d,%d,%d,%d,%d,%d",&p,&q,&r,&t,&u,&v,&s,&w);
               p += nc;
               q += nc;
               r += nc;
               s += nc;
               t += - nocc + nc;
               u += - nocc + nc;
               v += - nocc + nc;
               w += - nocc + nc;

               //printf ("p=%d, q=%d, r=%d, t=%d, u=%d, v=%d, s=%d, w=%d",p,q,r,t,u,v,s,w);

               for (itmp = 0; itmp < nocc+nvir; itmp++)
                   det_str[itmp] = Refdet[itmp];  
               det_str[p] = 2;
               det_str[q] = 2;
               det_str[r] = 2;
               det_str[t+nocc] = 1;
               det_str[u+nocc] = 1;
               det_str[v+nocc] = 1;
   
               if (p != s && q != s && r != s) det_str[s] = 1;
               else  det_str[s] = 0;
               if (t != w && u != w && v != w) det_str[w+nocc] = 2;
               else  det_str[w+nocc] = 3;
               //parity  = parity_ab_str(det_str, nocc+nvir);
               parity = parity_ci_to_cc(p+q+r, 3, nocc);
               parity *= parity_ci_to_cc(s, 1, nocc);
   
               // interm norm of c4
               t4 = parity * t4 / c0;
   
               // extract t4 
               t4-= t1xt3aaab (p, q, r, s, t, u, v, w, nocc, nvir, t1, t3aaa, t3aab);   // may have 1e-5 bug 
               t4-= t2xt2aaab (p, q, r, s, t, u, v, w, nocc, nvir, t2aa, t2ab);         // may have 1e-3 bug 
               t4-= t1xt1xt2aaab (p, q, r, s, t, u, v, w, nocc, nvir, t1, t2aa, t2ab);  // may have 1e-5 bug 
               t4-= t1xt1xt1xt1aaab (p, q, r, s, t, u, v, w, nocc, nvir, t1);           // may have 1e-6 bug
   
               for (it=0; it< nocc; it++){
               for (jt=0; jt< nocc; jt++){
               for (at=0; at< nvir; at++){
               for (bt=0; bt< nvir; bt++){
                   tmp[it][jt][at][bt] = 0.0;
               }
               }
               }
               }
   
               tmp[r][s][v][w] += (e2ovov[De(p,t,q,u,nocc,nvir)]-e2ovov[De(p,u,q,t,nocc,nvir)]) * t4; 
               tmp[q][s][v][w] -= (e2ovov[De(p,t,r,u,nocc,nvir)]-e2ovov[De(p,u,r,t,nocc,nvir)]) * t4; 
               tmp[p][s][v][w] += (e2ovov[De(q,t,s,u,nocc,nvir)]-e2ovov[De(q,u,s,t,nocc,nvir)]) * t4; 
               tmp[r][s][u][w] -= (e2ovov[De(p,t,q,v,nocc,nvir)]-e2ovov[De(p,v,q,t,nocc,nvir)]) * t4; 
               tmp[q][s][u][w] += (e2ovov[De(p,t,r,v,nocc,nvir)]-e2ovov[De(p,v,r,t,nocc,nvir)]) * t4; 
               tmp[p][s][u][w] -= (e2ovov[De(q,t,s,v,nocc,nvir)]-e2ovov[De(q,v,s,t,nocc,nvir)]) * t4; 
               tmp[r][s][t][w] += (e2ovov[De(p,u,q,v,nocc,nvir)]-e2ovov[De(p,v,q,u,nocc,nvir)]) * t4; 
               tmp[q][s][t][w] -= (e2ovov[De(p,u,r,v,nocc,nvir)]-e2ovov[De(p,v,r,u,nocc,nvir)]) * t4; 
               tmp[p][s][t][w] += (e2ovov[De(q,u,s,v,nocc,nvir)]-e2ovov[De(q,v,s,u,nocc,nvir)]) * t4; 
   
               for (it = 0; it < nocc; it++)
               for (jt = 0; jt < nocc; jt++)
               for (at = 0; at < nvir; at++)
               for (bt = 0; bt < nvir; bt++)
                   t2t4c[D(it,jt,at,bt,nocc,nvir)] += tmp[it][jt][at][bt] + tmp[jt][it][bt][at];

           }
       }
       fclose(fp);
    }
    else
    {
       // error message
    }


    for (it=0; it< nocc; it++){
        for (jt=0; jt< nocc; jt++){
            for (at=0; at< nvir; at++){
                free(tmp[it][jt][at]);
            }
            free(tmp[it][jt]);
        }
        free(tmp[it]);
    }   
    free(tmp);

    printf ("0SDTQ (Q) =    %f   ( %f )\n", norm, norm-norm0SDT);

//    for (itmp = 0; itmp < nocc+nvir; itmp++){
//        if (itmp<nocc) Refdet[itmp] = 3;  
//        else           Refdet[itmp] = 0;
//    }
//
//    // t2t4c += e2ovov * t4aabb
//    for (idet = 0; idet < n_aabb; idet++) {
//        t4 = c4_aabb[idet]; 
//        if(fabs(t4) > numzero) 
//        {
//            p = p_aabb[idet];
//            q = q_aabb[idet];
//            r = r_aabb[idet];
//            s = s_aabb[idet];
//            t = t_aabb[idet];
//            u = u_aabb[idet];
//            v = v_aabb[idet];
//            w = w_aabb[idet];
//
//
//            if(!(p == 2 && q == 3 && t == 0 && u == 1 && \
//                 r == 2 && s == 3 && v == 0 && w == 1)) continue;
//
//            for (itmp = 0; itmp < nocc+nvir; itmp++)
//                det_str[itmp] = Refdet[itmp];  
//
//            det_str[p] = 2;
//            det_str[q] = 2;
//            det_str[t+nocc] = 1;
//            det_str[u+nocc] = 1;
//
//            if (p != r && q != r) det_str[r] = 1;
//            else  det_str[r] = 0;
//            if (p != s && q != s) det_str[s] = 1;
//            else  det_str[s] = 0;
//            if (t != v && u != v) det_str[v+nocc] = 2;
//            else  det_str[v+nocc] = 3;
//            if (t != w && u != w) det_str[w+nocc] = 2;
//            else  det_str[w+nocc] = 3;
//
//            parity  = parity_ab_str(det_str, nocc+nvir);
//            parity *= parity_ci_to_cc(p+q, 2, nocc);
//            parity *= parity_ci_to_cc(r+s, 2, nocc);
//
//            // interm norm of c4
//            t4 = parity * t4 / c0;
//            // lsh test 
//            printf("c4 mem %20.10f \n",t4);
//
//
//            // extract t4 
//            t4-= t1xt3aabb(p, q, r, s, t, u, v, w, nocc, nvir, t1, t3aab); 
//            t4-= t2xt2aabb(p, q, r, s, t, u, v, w, nocc, nvir, t2aa, t2ab); 
//            t4-= t1xt1xt2aabb(p, q, r, s, t, u, v, w, nocc, nvir, t1, t2aa, t2ab); 
//            t4-= t1xt1xt1xt1aabb(p, q, r, s, t, u, v, w, nocc, nvir, t1);   // may have bug 
//
//            // lsh test 
//            printf("t4 mem %20.10f \n",t4);
//
//            for (itmp = 0; itmp < dlen; itmp++)
//                tmp[itmp] = 0.0;
//
//            if (p<r && t<v) tmp[D(q,s,u,w,nocc,nvir)] += e2ovov[De(p,t,r,v,nocc,nvir)] * t4;
//            if (q<r && t<v) tmp[D(p,s,u,w,nocc,nvir)] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t4;
//            if (p<s && t<v) tmp[D(q,r,u,w,nocc,nvir)] -= e2ovov[De(p,t,s,v,nocc,nvir)] * t4;
//            if (q<s && t<v) tmp[D(p,r,u,w,nocc,nvir)] += e2ovov[De(q,t,s,v,nocc,nvir)] * t4;
//            if (p<r && u<v) tmp[D(q,s,t,w,nocc,nvir)] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t4;
//            if (q<r && u<v) tmp[D(p,s,t,w,nocc,nvir)] += e2ovov[De(q,u,r,v,nocc,nvir)] * t4;
//            if (p<s && u<v) tmp[D(q,r,t,w,nocc,nvir)] += e2ovov[De(p,u,s,v,nocc,nvir)] * t4;
//            if (q<s && u<v) tmp[D(p,r,t,w,nocc,nvir)] -= e2ovov[De(q,u,s,v,nocc,nvir)] * t4;
//            if (p<r && t<w) tmp[D(q,s,u,v,nocc,nvir)] -= e2ovov[De(p,t,r,w,nocc,nvir)] * t4;
//            if (q<r && t<w) tmp[D(p,s,u,v,nocc,nvir)] += e2ovov[De(q,t,r,w,nocc,nvir)] * t4;
//            if (p<s && t<w) tmp[D(q,r,u,v,nocc,nvir)] += e2ovov[De(p,t,s,w,nocc,nvir)] * t4;
//            if (q<s && t<w) tmp[D(p,r,u,v,nocc,nvir)] -= e2ovov[De(q,t,s,w,nocc,nvir)] * t4;
//            if (p<r && u<w) tmp[D(q,s,t,v,nocc,nvir)] += e2ovov[De(p,u,r,w,nocc,nvir)] * t4;
//            if (q<r && u<w) tmp[D(p,s,t,v,nocc,nvir)] -= e2ovov[De(q,u,r,w,nocc,nvir)] * t4;
//            if (p<s && u<w) tmp[D(q,r,t,v,nocc,nvir)] -= e2ovov[De(p,u,s,w,nocc,nvir)] * t4;
//            if (q<s && u<w) tmp[D(p,r,t,v,nocc,nvir)] += e2ovov[De(q,u,s,w,nocc,nvir)] * t4;
//            if (p<r && v<t) tmp[D(q,s,u,w,nocc,nvir)] += e2ovov[De(p,t,r,v,nocc,nvir)] * t4;
//            if (q<r && v<t) tmp[D(p,s,u,w,nocc,nvir)] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t4;
//            if (p<s && v<t) tmp[D(q,r,u,w,nocc,nvir)] -= e2ovov[De(p,t,s,v,nocc,nvir)] * t4;
//            if (q<s && v<t) tmp[D(p,r,u,w,nocc,nvir)] += e2ovov[De(q,t,s,v,nocc,nvir)] * t4;
//            if (p<r && v<u) tmp[D(q,s,t,w,nocc,nvir)] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t4;
//            if (q<r && v<u) tmp[D(p,s,t,w,nocc,nvir)] += e2ovov[De(q,u,r,v,nocc,nvir)] * t4;
//            if (p<s && v<u) tmp[D(q,r,t,w,nocc,nvir)] += e2ovov[De(p,u,s,v,nocc,nvir)] * t4;
//            if (q<s && v<u) tmp[D(p,r,t,w,nocc,nvir)] -= e2ovov[De(q,u,s,v,nocc,nvir)] * t4;
//            if (p<r && w<t) tmp[D(q,s,u,v,nocc,nvir)] -= e2ovov[De(p,t,r,w,nocc,nvir)] * t4;
//            if (q<r && w<t) tmp[D(p,s,u,v,nocc,nvir)] += e2ovov[De(q,t,r,w,nocc,nvir)] * t4;
//            if (p<s && w<t) tmp[D(q,r,u,v,nocc,nvir)] += e2ovov[De(p,t,s,w,nocc,nvir)] * t4;
//            if (q<s && w<t) tmp[D(p,r,u,v,nocc,nvir)] -= e2ovov[De(q,t,s,w,nocc,nvir)] * t4;
//            if (p<r && w<u) tmp[D(q,s,t,v,nocc,nvir)] += e2ovov[De(p,u,r,w,nocc,nvir)] * t4;
//            if (q<r && w<u) tmp[D(p,s,t,v,nocc,nvir)] -= e2ovov[De(q,u,r,w,nocc,nvir)] * t4;
//            if (p<s && w<u) tmp[D(q,r,t,v,nocc,nvir)] -= e2ovov[De(p,u,s,w,nocc,nvir)] * t4;
//            if (q<s && w<u) tmp[D(p,r,t,v,nocc,nvir)] += e2ovov[De(q,u,s,w,nocc,nvir)] * t4;
//
//            scale = 0.5;
//            if (p==r && t<v) tmp[D(q,s,u,w,nocc,nvir)] += e2ovov[De(p,t,r,v,nocc,nvir)] * t4 * scale;
//            if (q==r && t<v) tmp[D(p,s,u,w,nocc,nvir)] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t4 * scale;
//            if (p==s && t<v) tmp[D(q,r,u,w,nocc,nvir)] -= e2ovov[De(p,t,s,v,nocc,nvir)] * t4 * scale;
//            if (q==s && t<v) tmp[D(p,r,u,w,nocc,nvir)] += e2ovov[De(q,t,s,v,nocc,nvir)] * t4 * scale;
//            if (p==r && u<v) tmp[D(q,s,t,w,nocc,nvir)] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t4 * scale;
//            if (q==r && u<v) tmp[D(p,s,t,w,nocc,nvir)] += e2ovov[De(q,u,r,v,nocc,nvir)] * t4 * scale;
//            if (p==s && u<v) tmp[D(q,r,t,w,nocc,nvir)] += e2ovov[De(p,u,s,v,nocc,nvir)] * t4 * scale;
//            if (q==s && u<v) tmp[D(p,r,t,w,nocc,nvir)] -= e2ovov[De(q,u,s,v,nocc,nvir)] * t4 * scale;
//            if (p==r && t<w) tmp[D(q,s,u,v,nocc,nvir)] -= e2ovov[De(p,t,r,w,nocc,nvir)] * t4 * scale;
//            if (q==r && t<w) tmp[D(p,s,u,v,nocc,nvir)] += e2ovov[De(q,t,r,w,nocc,nvir)] * t4 * scale;
//            if (p==s && t<w) tmp[D(q,r,u,v,nocc,nvir)] += e2ovov[De(p,t,s,w,nocc,nvir)] * t4 * scale;
//            if (q==s && t<w) tmp[D(p,r,u,v,nocc,nvir)] -= e2ovov[De(q,t,s,w,nocc,nvir)] * t4 * scale;
//            if (p==r && u<w) tmp[D(q,s,t,v,nocc,nvir)] += e2ovov[De(p,u,r,w,nocc,nvir)] * t4 * scale;
//            if (q==r && u<w) tmp[D(p,s,t,v,nocc,nvir)] -= e2ovov[De(q,u,r,w,nocc,nvir)] * t4 * scale;
//            if (p==s && u<w) tmp[D(q,r,t,v,nocc,nvir)] -= e2ovov[De(p,u,s,w,nocc,nvir)] * t4 * scale;
//            if (q==s && u<w) tmp[D(p,r,t,v,nocc,nvir)] += e2ovov[De(q,u,s,w,nocc,nvir)] * t4 * scale;
//            if (p==r && v<t) tmp[D(q,s,u,w,nocc,nvir)] += e2ovov[De(p,t,r,v,nocc,nvir)] * t4 * scale;
//            if (q==r && v<t) tmp[D(p,s,u,w,nocc,nvir)] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t4 * scale;
//            if (p==s && v<t) tmp[D(q,r,u,w,nocc,nvir)] -= e2ovov[De(p,t,s,v,nocc,nvir)] * t4 * scale;
//            if (q==s && v<t) tmp[D(p,r,u,w,nocc,nvir)] += e2ovov[De(q,t,s,v,nocc,nvir)] * t4 * scale;
//            if (p==r && v<u) tmp[D(q,s,t,w,nocc,nvir)] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t4 * scale;
//            if (q==r && v<u) tmp[D(p,s,t,w,nocc,nvir)] += e2ovov[De(q,u,r,v,nocc,nvir)] * t4 * scale;
//            if (p==s && v<u) tmp[D(q,r,t,w,nocc,nvir)] += e2ovov[De(p,u,s,v,nocc,nvir)] * t4 * scale;
//            if (q==s && v<u) tmp[D(p,r,t,w,nocc,nvir)] -= e2ovov[De(q,u,s,v,nocc,nvir)] * t4 * scale;
//            if (p==r && w<t) tmp[D(q,s,u,v,nocc,nvir)] -= e2ovov[De(p,t,r,w,nocc,nvir)] * t4 * scale;
//            if (q==r && w<t) tmp[D(p,s,u,v,nocc,nvir)] += e2ovov[De(q,t,r,w,nocc,nvir)] * t4 * scale;
//            if (p==s && w<t) tmp[D(q,r,u,v,nocc,nvir)] += e2ovov[De(p,t,s,w,nocc,nvir)] * t4 * scale;
//            if (q==s && w<t) tmp[D(p,r,u,v,nocc,nvir)] -= e2ovov[De(q,t,s,w,nocc,nvir)] * t4 * scale;
//            if (p==r && w<u) tmp[D(q,s,t,v,nocc,nvir)] += e2ovov[De(p,u,r,w,nocc,nvir)] * t4 * scale;
//            if (q==r && w<u) tmp[D(p,s,t,v,nocc,nvir)] -= e2ovov[De(q,u,r,w,nocc,nvir)] * t4 * scale;
//            if (p==s && w<u) tmp[D(q,r,t,v,nocc,nvir)] -= e2ovov[De(p,u,s,w,nocc,nvir)] * t4 * scale;
//            if (q==s && w<u) tmp[D(p,r,t,v,nocc,nvir)] += e2ovov[De(q,u,s,w,nocc,nvir)] * t4 * scale;
//
//            scale = 0.5;
//            if (p<r && t==v) tmp[D(q,s,u,w,nocc,nvir)] += e2ovov[De(p,t,r,v,nocc,nvir)] * t4 * scale;
//            if (q<r && t==v) tmp[D(p,s,u,w,nocc,nvir)] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t4 * scale;
//            if (p<s && t==v) tmp[D(q,r,u,w,nocc,nvir)] -= e2ovov[De(p,t,s,v,nocc,nvir)] * t4 * scale;
//            if (q<s && t==v) tmp[D(p,r,u,w,nocc,nvir)] += e2ovov[De(q,t,s,v,nocc,nvir)] * t4 * scale;
//            if (p<r && u==v) tmp[D(q,s,t,w,nocc,nvir)] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t4 * scale;
//            if (q<r && u==v) tmp[D(p,s,t,w,nocc,nvir)] += e2ovov[De(q,u,r,v,nocc,nvir)] * t4 * scale;
//            if (p<s && u==v) tmp[D(q,r,t,w,nocc,nvir)] += e2ovov[De(p,u,s,v,nocc,nvir)] * t4 * scale;
//            if (q<s && u==v) tmp[D(p,r,t,w,nocc,nvir)] -= e2ovov[De(q,u,s,v,nocc,nvir)] * t4 * scale;
//            if (p<r && t==w) tmp[D(q,s,u,v,nocc,nvir)] -= e2ovov[De(p,t,r,w,nocc,nvir)] * t4 * scale;
//            if (q<r && t==w) tmp[D(p,s,u,v,nocc,nvir)] += e2ovov[De(q,t,r,w,nocc,nvir)] * t4 * scale;
//            if (p<s && t==w) tmp[D(q,r,u,v,nocc,nvir)] += e2ovov[De(p,t,s,w,nocc,nvir)] * t4 * scale;
//            if (q<s && t==w) tmp[D(p,r,u,v,nocc,nvir)] -= e2ovov[De(q,t,s,w,nocc,nvir)] * t4 * scale;
//            if (p<r && u==w) tmp[D(q,s,t,v,nocc,nvir)] += e2ovov[De(p,u,r,w,nocc,nvir)] * t4 * scale;
//            if (q<r && u==w) tmp[D(p,s,t,v,nocc,nvir)] -= e2ovov[De(q,u,r,w,nocc,nvir)] * t4 * scale;
//            if (p<s && u==w) tmp[D(q,r,t,v,nocc,nvir)] -= e2ovov[De(p,u,s,w,nocc,nvir)] * t4 * scale;
//            if (q<s && u==w) tmp[D(p,r,t,v,nocc,nvir)] += e2ovov[De(q,u,s,w,nocc,nvir)] * t4 * scale;
//            if (p<r && v==t) tmp[D(q,s,u,w,nocc,nvir)] += e2ovov[De(p,t,r,v,nocc,nvir)] * t4 * scale;
//            if (q<r && v==t) tmp[D(p,s,u,w,nocc,nvir)] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t4 * scale;
//            if (p<s && v==t) tmp[D(q,r,u,w,nocc,nvir)] -= e2ovov[De(p,t,s,v,nocc,nvir)] * t4 * scale;
//            if (q<s && v==t) tmp[D(p,r,u,w,nocc,nvir)] += e2ovov[De(q,t,s,v,nocc,nvir)] * t4 * scale;
//            if (p<r && v==u) tmp[D(q,s,t,w,nocc,nvir)] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t4 * scale;
//            if (q<r && v==u) tmp[D(p,s,t,w,nocc,nvir)] += e2ovov[De(q,u,r,v,nocc,nvir)] * t4 * scale;
//            if (p<s && v==u) tmp[D(q,r,t,w,nocc,nvir)] += e2ovov[De(p,u,s,v,nocc,nvir)] * t4 * scale;
//            if (q<s && v==u) tmp[D(p,r,t,w,nocc,nvir)] -= e2ovov[De(q,u,s,v,nocc,nvir)] * t4 * scale;
//            if (p<r && w==t) tmp[D(q,s,u,v,nocc,nvir)] -= e2ovov[De(p,t,r,w,nocc,nvir)] * t4 * scale;
//            if (q<r && w==t) tmp[D(p,s,u,v,nocc,nvir)] += e2ovov[De(q,t,r,w,nocc,nvir)] * t4 * scale;
//            if (p<s && w==t) tmp[D(q,r,u,v,nocc,nvir)] += e2ovov[De(p,t,s,w,nocc,nvir)] * t4 * scale;
//            if (q<s && w==t) tmp[D(p,r,u,v,nocc,nvir)] -= e2ovov[De(q,t,s,w,nocc,nvir)] * t4 * scale;
//            if (p<r && w==u) tmp[D(q,s,t,v,nocc,nvir)] += e2ovov[De(p,u,r,w,nocc,nvir)] * t4 * scale;
//            if (q<r && w==u) tmp[D(p,s,t,v,nocc,nvir)] -= e2ovov[De(q,u,r,w,nocc,nvir)] * t4 * scale;
//            if (p<s && w==u) tmp[D(q,r,t,v,nocc,nvir)] -= e2ovov[De(p,u,s,w,nocc,nvir)] * t4 * scale;
//            if (q<s && w==u) tmp[D(p,r,t,v,nocc,nvir)] += e2ovov[De(q,u,s,w,nocc,nvir)] * t4 * scale;
//
//            scale = 0.25;
//            if (p==r && t==v) tmp[D(q,s,u,w,nocc,nvir)] += e2ovov[De(p,t,r,v,nocc,nvir)] * t4 * scale;
//            if (q==r && t==v) tmp[D(p,s,u,w,nocc,nvir)] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t4 * scale;
//            if (p==s && t==v) tmp[D(q,r,u,w,nocc,nvir)] -= e2ovov[De(p,t,s,v,nocc,nvir)] * t4 * scale;
//            if (q==s && t==v) tmp[D(p,r,u,w,nocc,nvir)] += e2ovov[De(q,t,s,v,nocc,nvir)] * t4 * scale;
//            if (p==r && u==v) tmp[D(q,s,t,w,nocc,nvir)] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t4 * scale;
//            if (q==r && u==v) tmp[D(p,s,t,w,nocc,nvir)] += e2ovov[De(q,u,r,v,nocc,nvir)] * t4 * scale;
//            if (p==s && u==v) tmp[D(q,r,t,w,nocc,nvir)] += e2ovov[De(p,u,s,v,nocc,nvir)] * t4 * scale;
//            if (q==s && u==v) tmp[D(p,r,t,w,nocc,nvir)] -= e2ovov[De(q,u,s,v,nocc,nvir)] * t4 * scale;
//            if (p==r && t==w) tmp[D(q,s,u,v,nocc,nvir)] -= e2ovov[De(p,t,r,w,nocc,nvir)] * t4 * scale;
//            if (q==r && t==w) tmp[D(p,s,u,v,nocc,nvir)] += e2ovov[De(q,t,r,w,nocc,nvir)] * t4 * scale;
//            if (p==s && t==w) tmp[D(q,r,u,v,nocc,nvir)] += e2ovov[De(p,t,s,w,nocc,nvir)] * t4 * scale;
//            if (q==s && t==w) tmp[D(p,r,u,v,nocc,nvir)] -= e2ovov[De(q,t,s,w,nocc,nvir)] * t4 * scale;
//            if (p==r && u==w) tmp[D(q,s,t,v,nocc,nvir)] += e2ovov[De(p,u,r,w,nocc,nvir)] * t4 * scale;
//            if (q==r && u==w) tmp[D(p,s,t,v,nocc,nvir)] -= e2ovov[De(q,u,r,w,nocc,nvir)] * t4 * scale;
//            if (p==s && u==w) tmp[D(q,r,t,v,nocc,nvir)] -= e2ovov[De(p,u,s,w,nocc,nvir)] * t4 * scale;
//            if (q==s && u==w) tmp[D(p,r,t,v,nocc,nvir)] += e2ovov[De(q,u,s,w,nocc,nvir)] * t4 * scale;
//            if (p==r && v==t) tmp[D(q,s,u,w,nocc,nvir)] += e2ovov[De(p,t,r,v,nocc,nvir)] * t4 * scale;
//            if (q==r && v==t) tmp[D(p,s,u,w,nocc,nvir)] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t4 * scale;
//            if (p==s && v==t) tmp[D(q,r,u,w,nocc,nvir)] -= e2ovov[De(p,t,s,v,nocc,nvir)] * t4 * scale;
//            if (q==s && v==t) tmp[D(p,r,u,w,nocc,nvir)] += e2ovov[De(q,t,s,v,nocc,nvir)] * t4 * scale;
//            if (p==r && v==u) tmp[D(q,s,t,w,nocc,nvir)] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t4 * scale;
//            if (q==r && v==u) tmp[D(p,s,t,w,nocc,nvir)] += e2ovov[De(q,u,r,v,nocc,nvir)] * t4 * scale;
//            if (p==s && v==u) tmp[D(q,r,t,w,nocc,nvir)] += e2ovov[De(p,u,s,v,nocc,nvir)] * t4 * scale;
//            if (q==s && v==u) tmp[D(p,r,t,w,nocc,nvir)] -= e2ovov[De(q,u,s,v,nocc,nvir)] * t4 * scale;
//            if (p==r && w==t) tmp[D(q,s,u,v,nocc,nvir)] -= e2ovov[De(p,t,r,w,nocc,nvir)] * t4 * scale;
//            if (q==r && w==t) tmp[D(p,s,u,v,nocc,nvir)] += e2ovov[De(q,t,r,w,nocc,nvir)] * t4 * scale;
//            if (p==s && w==t) tmp[D(q,r,u,v,nocc,nvir)] += e2ovov[De(p,t,s,w,nocc,nvir)] * t4 * scale;
//            if (q==s && w==t) tmp[D(p,r,u,v,nocc,nvir)] -= e2ovov[De(q,t,s,w,nocc,nvir)] * t4 * scale;
//            if (p==r && w==u) tmp[D(q,s,t,v,nocc,nvir)] += e2ovov[De(p,u,r,w,nocc,nvir)] * t4 * scale;
//            if (q==r && w==u) tmp[D(p,s,t,v,nocc,nvir)] -= e2ovov[De(q,u,r,w,nocc,nvir)] * t4 * scale;
//            if (p==s && w==u) tmp[D(q,r,t,v,nocc,nvir)] -= e2ovov[De(p,u,s,w,nocc,nvir)] * t4 * scale;
//            if (q==s && w==u) tmp[D(p,r,t,v,nocc,nvir)] += e2ovov[De(q,u,s,w,nocc,nvir)] * t4 * scale;
//
//            for (it = 0; it < nocc; it++)
//            for (jt = 0; jt < nocc; jt++)
//            for (at = 0; at < nvir; at++)
//            for (bt = 0; bt < nvir; bt++)
//                t2t4c[D(it,jt,at,bt,nocc,nvir)] += tmp[D(it,jt,at,bt,nocc,nvir)] + tmp[D(jt,it,bt,at,nocc,nvir)];
//        }
//    }


//    // t2t4c += e2ovov * t4aaab
//    for (idet = 0; idet < n_aaab; idet++) {
//        t4 = c4_aaab[idet]; 
//        if(fabs(t4) > numzero) 
//        {
//            p = p_aaab[idet];
//            q = q_aaab[idet];
//            r = r_aaab[idet];
//            s = s_aaab[idet];
//            t = t_aaab[idet];
//            u = u_aaab[idet];
//            v = v_aaab[idet];
//            w = w_aaab[idet];
//
//            for (itmp = 0; itmp < nocc+nvir; itmp++)
//                det_str[itmp] = Refdet[itmp];  
//            det_str[p] = 2;
//            det_str[q] = 2;
//            det_str[r] = 2;
//            det_str[t+nocc] = 1;
//            det_str[u+nocc] = 1;
//            det_str[v+nocc] = 1;
//
//            if (p != s && q != s && r != s) det_str[s] = 1;
//            else  det_str[s] = 0;
//            if (t != w && u != w && v != w) det_str[w+nocc] = 2;
//            else  det_str[w+nocc] = 3;
//            parity  = parity_ab_str(det_str, nocc+nvir);
//            parity *= parity_ci_to_cc(p+q+r, 3, nocc);
//            parity *= parity_ci_to_cc(s, 1, nocc);
//
//            // interm norm of c4
//            t4 = parity * t4 / c0;
//
//            // extract t4 
//            t4-= t1xt3aaab (p, q, r, s, t, u, v, w, nocc, nvir, t1, t3aaa, t3aab);   // may have 1e-5 bug 
//            t4-= t2xt2aaab (p, q, r, s, t, u, v, w, nocc, nvir, t2aa, t2ab);         // may have 1e-3 bug 
//            t4-= t1xt1xt2aaab (p, q, r, s, t, u, v, w, nocc, nvir, t1, t2aa, t2ab);  // may have 1e-5 bug 
//            t4-= t1xt1xt1xt1aaab (p, q, r, s, t, u, v, w, nocc, nvir, t1);           // may have 1e-6 bug
//
//            for (itmp = 0; itmp < dlen; itmp++)
//                tmp[itmp] = 0.0;
//
//            tmp[D(r,s,v,w,nocc,nvir)] += (e2ovov[De(p,t,q,u,nocc,nvir)]-e2ovov[De(p,u,q,t,nocc,nvir)]) * t4; 
//            tmp[D(q,s,v,w,nocc,nvir)] -= (e2ovov[De(p,t,r,u,nocc,nvir)]-e2ovov[De(p,u,r,t,nocc,nvir)]) * t4; 
//            tmp[D(p,s,v,w,nocc,nvir)] += (e2ovov[De(q,t,s,u,nocc,nvir)]-e2ovov[De(q,u,s,t,nocc,nvir)]) * t4; 
//            tmp[D(r,s,u,w,nocc,nvir)] -= (e2ovov[De(p,t,q,v,nocc,nvir)]-e2ovov[De(p,v,q,t,nocc,nvir)]) * t4; 
//            tmp[D(q,s,u,w,nocc,nvir)] += (e2ovov[De(p,t,r,v,nocc,nvir)]-e2ovov[De(p,v,r,t,nocc,nvir)]) * t4; 
//            tmp[D(p,s,u,w,nocc,nvir)] -= (e2ovov[De(q,t,s,v,nocc,nvir)]-e2ovov[De(q,v,s,t,nocc,nvir)]) * t4; 
//            tmp[D(r,s,t,w,nocc,nvir)] += (e2ovov[De(p,u,q,v,nocc,nvir)]-e2ovov[De(p,v,q,u,nocc,nvir)]) * t4; 
//            tmp[D(q,s,t,w,nocc,nvir)] -= (e2ovov[De(p,u,r,v,nocc,nvir)]-e2ovov[De(p,v,r,u,nocc,nvir)]) * t4; 
//            tmp[D(p,s,t,w,nocc,nvir)] += (e2ovov[De(q,u,s,v,nocc,nvir)]-e2ovov[De(q,v,s,u,nocc,nvir)]) * t4; 
//
//            for (it = 0; it < nocc; it++)
//            for (jt = 0; jt < nocc; jt++)
//            for (at = 0; at < nvir; at++)
//            for (bt = 0; bt < nvir; bt++)
//                t2t4c[D(it,jt,at,bt,nocc,nvir)] += tmp[D(it,jt,at,bt,nocc,nvir)] + tmp[D(jt,it,bt,at,nocc,nvir)];
//        }
//    }

}

void t2t4c_shci_ecT(double *t2t4c, double *t1, double *t2aa, double *t2ab, double *t3aaa, double *t3aab, double *e2ovov, const int nc, const int nc_ref, const int nvir_ref, const int nocc, const int nvir, const double numzero, const double c0, double norm) 
{
    //double numzero = 1e-7;
    int p, q, r, s, t, u, v, w, itmp, it, jt, at, bt;
    double t4, parity, scale;
    uint8_t Refdet[nocc+nvir], det_str[nocc+nvir];
    for (itmp = 0; itmp < nocc+nvir; itmp++){
        if (itmp<nocc) Refdet[itmp] = 3;  
        else           Refdet[itmp] = 0;
    }

    double norm0SDT = norm;

    double ****tmp;
    tmp = (double ****)malloc(sizeof(double ***) * nocc); 
    for (it=0; it< nocc; it++){
        tmp[it] = (double ***)malloc(sizeof(double **) * nocc);
        for (jt=0; jt< nocc; jt++){
            tmp[it][jt] = (double **)malloc(sizeof(double *) * nvir);
            for (at=0; at< nvir; at++){
                tmp[it][jt][at] = (double *)malloc(sizeof(double) * nvir);
            }
        }
    }

//    for (p=0; p<nocc; p++) {
//    for (q=0; q<nocc; q++) {
//    for (r=0; r<nvir; r++) {
//    for (s=0; s<nvir; s++) {
//        printf("%d %d %d %d %20.10lf\n",p,q,r,s,e2ovov[De(p,r,q,s,nocc,nvir)]); 
//    }
//    }
//    }
//    }

    FILE *fp;
    char typ[4], line[255];
    fp = fopen("CIcoeff_shci.out", "r");
    fscanf(fp, "%s\n", line);
    if (fp) {
       while ( !feof(fp) ){
           fscanf(fp, "%c%c%c%c,%s\n", &(typ[0]), &(typ[1]), &(typ[2]), &(typ[3]), line);
           fscanf(fp, "%lf\n", &t4);
           //printf ("typ=%c%c%c%c line=%s\n",typ[0],typ[1],typ[2],typ[3], line);
           if (strncmp(typ, "aabb", 4) == 0 && fabs(t4) > numzero){
               norm += t4*t4;
               sscanf(line,"%d,%d,%d,%d,%d,%d,%d,%d",&p,&q,&t,&u,&r,&s,&v,&w);
               p += nc;
               q += nc;
               r += nc;
               s += nc;
               t += - nocc + nc;
               u += - nocc + nc;
               v += - nocc + nc;
               w += - nocc + nc;

               // exclude active space (reference space)
               //if (p >= nc_ref && q >= nc_ref && r >= nc_ref && s >= nc_ref && \
               //    r < nvir_ref && s < nvir_ref && t < nvir_ref && u < nvir_ref) continue; 

               // exclude inactive space
               if (p <= nc_ref && q <= nc_ref && r <= nc_ref && s <= nc_ref && \
                   t > nvir_ref && u > nvir_ref && v > nvir_ref && w > nvir_ref) continue; 

//               if(!(p == 2 && q == 3 && t == 0 && u == 1 && \
//                    r == 2 && s == 3 && v == 0 && w == 1)) continue;
   
               for (itmp = 0; itmp < nocc+nvir; itmp++)
                   det_str[itmp] = Refdet[itmp];  
   
               det_str[p] = 2;
               det_str[q] = 2;
               det_str[t+nocc] = 1;
               det_str[u+nocc] = 1;
   
               if (p != r && q != r) det_str[r] = 1;
               else  det_str[r] = 0;
               if (p != s && q != s) det_str[s] = 1;
               else  det_str[s] = 0;
               if (t != v && u != v) det_str[v+nocc] = 2;
               else  det_str[v+nocc] = 3;
               if (t != w && u != w) det_str[w+nocc] = 2;
               else  det_str[w+nocc] = 3;
   
               //parity  = parity_ab_str(det_str, nocc+nvir);
               parity = parity_ci_to_cc(p+q, 2, nocc);
               parity *= parity_ci_to_cc(r+s, 2, nocc);
   
               // interm norm of c4
               t4 = parity * t4 / c0;
               // lsh test 
//               printf("c4 mem %20.10f \n",t4);
   
               // extract t4 
               t4-= t1xt3aabb(p, q, r, s, t, u, v, w, nocc, nvir, t1, t3aab); 
               t4-= t2xt2aabb(p, q, r, s, t, u, v, w, nocc, nvir, t2aa, t2ab); 
               t4-= t1xt1xt2aabb(p, q, r, s, t, u, v, w, nocc, nvir, t1, t2aa, t2ab); 
               t4-= t1xt1xt1xt1aabb(p, q, r, s, t, u, v, w, nocc, nvir, t1);   // may have bug 
   
               // lsh test 
//               printf("t4 mem %20.10f \n",t4);
   
               for (it=0; it< nocc; it++){
               for (jt=0; jt< nocc; jt++){
               for (at=0; at< nvir; at++){
               for (bt=0; bt< nvir; bt++){
                   tmp[it][jt][at][bt] = 0.0;
               }
               }
               }
               }
//               printf("eris_ovov mem %20.10f \n",e2ovov[De(p,t,r,v,nocc,nvir)]);
   
               if (p<r && t<v) tmp[q][s][u][w] += e2ovov[De(p,t,r,v,nocc,nvir)] * t4;
               if (q<r && t<v) tmp[p][s][u][w] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t4;
               if (p<s && t<v) tmp[q][r][u][w] -= e2ovov[De(p,t,s,v,nocc,nvir)] * t4;
               if (q<s && t<v) tmp[p][r][u][w] += e2ovov[De(q,t,s,v,nocc,nvir)] * t4;
               if (p<r && u<v) tmp[q][s][t][w] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t4;
               if (q<r && u<v) tmp[p][s][t][w] += e2ovov[De(q,u,r,v,nocc,nvir)] * t4;
               if (p<s && u<v) tmp[q][r][t][w] += e2ovov[De(p,u,s,v,nocc,nvir)] * t4;
               if (q<s && u<v) tmp[p][r][t][w] -= e2ovov[De(q,u,s,v,nocc,nvir)] * t4;
               if (p<r && t<w) tmp[q][s][u][v] -= e2ovov[De(p,t,r,w,nocc,nvir)] * t4;
               if (q<r && t<w) tmp[p][s][u][v] += e2ovov[De(q,t,r,w,nocc,nvir)] * t4;
               if (p<s && t<w) tmp[q][r][u][v] += e2ovov[De(p,t,s,w,nocc,nvir)] * t4;
               if (q<s && t<w) tmp[p][r][u][v] -= e2ovov[De(q,t,s,w,nocc,nvir)] * t4;
               if (p<r && u<w) tmp[q][s][t][v] += e2ovov[De(p,u,r,w,nocc,nvir)] * t4;
               if (q<r && u<w) tmp[p][s][t][v] -= e2ovov[De(q,u,r,w,nocc,nvir)] * t4;
               if (p<s && u<w) tmp[q][r][t][v] -= e2ovov[De(p,u,s,w,nocc,nvir)] * t4;
               if (q<s && u<w) tmp[p][r][t][v] += e2ovov[De(q,u,s,w,nocc,nvir)] * t4;
               if (p<r && v<t) tmp[q][s][u][w] += e2ovov[De(p,t,r,v,nocc,nvir)] * t4;
               if (q<r && v<t) tmp[p][s][u][w] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t4;
               if (p<s && v<t) tmp[q][r][u][w] -= e2ovov[De(p,t,s,v,nocc,nvir)] * t4;
               if (q<s && v<t) tmp[p][r][u][w] += e2ovov[De(q,t,s,v,nocc,nvir)] * t4;
               if (p<r && v<u) tmp[q][s][t][w] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t4;
               if (q<r && v<u) tmp[p][s][t][w] += e2ovov[De(q,u,r,v,nocc,nvir)] * t4;
               if (p<s && v<u) tmp[q][r][t][w] += e2ovov[De(p,u,s,v,nocc,nvir)] * t4;
               if (q<s && v<u) tmp[p][r][t][w] -= e2ovov[De(q,u,s,v,nocc,nvir)] * t4;
               if (p<r && w<t) tmp[q][s][u][v] -= e2ovov[De(p,t,r,w,nocc,nvir)] * t4;
               if (q<r && w<t) tmp[p][s][u][v] += e2ovov[De(q,t,r,w,nocc,nvir)] * t4;
               if (p<s && w<t) tmp[q][r][u][v] += e2ovov[De(p,t,s,w,nocc,nvir)] * t4;
               if (q<s && w<t) tmp[p][r][u][v] -= e2ovov[De(q,t,s,w,nocc,nvir)] * t4;
               if (p<r && w<u) tmp[q][s][t][v] += e2ovov[De(p,u,r,w,nocc,nvir)] * t4;
               if (q<r && w<u) tmp[p][s][t][v] -= e2ovov[De(q,u,r,w,nocc,nvir)] * t4;
               if (p<s && w<u) tmp[q][r][t][v] -= e2ovov[De(p,u,s,w,nocc,nvir)] * t4;
               if (q<s && w<u) tmp[p][r][t][v] += e2ovov[De(q,u,s,w,nocc,nvir)] * t4;
   
               scale = 0.5;
               if (p==r && t<v) tmp[q][s][u][w] += e2ovov[De(p,t,r,v,nocc,nvir)] * t4 * scale;
               if (q==r && t<v) tmp[p][s][u][w] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t4 * scale;
               if (p==s && t<v) tmp[q][r][u][w] -= e2ovov[De(p,t,s,v,nocc,nvir)] * t4 * scale;
               if (q==s && t<v) tmp[p][r][u][w] += e2ovov[De(q,t,s,v,nocc,nvir)] * t4 * scale;
               if (p==r && u<v) tmp[q][s][t][w] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t4 * scale;
               if (q==r && u<v) tmp[p][s][t][w] += e2ovov[De(q,u,r,v,nocc,nvir)] * t4 * scale;
               if (p==s && u<v) tmp[q][r][t][w] += e2ovov[De(p,u,s,v,nocc,nvir)] * t4 * scale;
               if (q==s && u<v) tmp[p][r][t][w] -= e2ovov[De(q,u,s,v,nocc,nvir)] * t4 * scale;
               if (p==r && t<w) tmp[q][s][u][v] -= e2ovov[De(p,t,r,w,nocc,nvir)] * t4 * scale;
               if (q==r && t<w) tmp[p][s][u][v] += e2ovov[De(q,t,r,w,nocc,nvir)] * t4 * scale;
               if (p==s && t<w) tmp[q][r][u][v] += e2ovov[De(p,t,s,w,nocc,nvir)] * t4 * scale;
               if (q==s && t<w) tmp[p][r][u][v] -= e2ovov[De(q,t,s,w,nocc,nvir)] * t4 * scale;
               if (p==r && u<w) tmp[q][s][t][v] += e2ovov[De(p,u,r,w,nocc,nvir)] * t4 * scale;
               if (q==r && u<w) tmp[p][s][t][v] -= e2ovov[De(q,u,r,w,nocc,nvir)] * t4 * scale;
               if (p==s && u<w) tmp[q][r][t][v] -= e2ovov[De(p,u,s,w,nocc,nvir)] * t4 * scale;
               if (q==s && u<w) tmp[p][r][t][v] += e2ovov[De(q,u,s,w,nocc,nvir)] * t4 * scale;
               if (p==r && v<t) tmp[q][s][u][w] += e2ovov[De(p,t,r,v,nocc,nvir)] * t4 * scale;
               if (q==r && v<t) tmp[p][s][u][w] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t4 * scale;
               if (p==s && v<t) tmp[q][r][u][w] -= e2ovov[De(p,t,s,v,nocc,nvir)] * t4 * scale;
               if (q==s && v<t) tmp[p][r][u][w] += e2ovov[De(q,t,s,v,nocc,nvir)] * t4 * scale;
               if (p==r && v<u) tmp[q][s][t][w] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t4 * scale;
               if (q==r && v<u) tmp[p][s][t][w] += e2ovov[De(q,u,r,v,nocc,nvir)] * t4 * scale;
               if (p==s && v<u) tmp[q][r][t][w] += e2ovov[De(p,u,s,v,nocc,nvir)] * t4 * scale;
               if (q==s && v<u) tmp[p][r][t][w] -= e2ovov[De(q,u,s,v,nocc,nvir)] * t4 * scale;
               if (p==r && w<t) tmp[q][s][u][v] -= e2ovov[De(p,t,r,w,nocc,nvir)] * t4 * scale;
               if (q==r && w<t) tmp[p][s][u][v] += e2ovov[De(q,t,r,w,nocc,nvir)] * t4 * scale;
               if (p==s && w<t) tmp[q][r][u][v] += e2ovov[De(p,t,s,w,nocc,nvir)] * t4 * scale;
               if (q==s && w<t) tmp[p][r][u][v] -= e2ovov[De(q,t,s,w,nocc,nvir)] * t4 * scale;
               if (p==r && w<u) tmp[q][s][t][v] += e2ovov[De(p,u,r,w,nocc,nvir)] * t4 * scale;
               if (q==r && w<u) tmp[p][s][t][v] -= e2ovov[De(q,u,r,w,nocc,nvir)] * t4 * scale;
               if (p==s && w<u) tmp[q][r][t][v] -= e2ovov[De(p,u,s,w,nocc,nvir)] * t4 * scale;
               if (q==s && w<u) tmp[p][r][t][v] += e2ovov[De(q,u,s,w,nocc,nvir)] * t4 * scale;
   
               scale = 0.5;
               if (p<r && t==v) tmp[q][s][u][w] += e2ovov[De(p,t,r,v,nocc,nvir)] * t4 * scale;
               if (q<r && t==v) tmp[p][s][u][w] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t4 * scale;
               if (p<s && t==v) tmp[q][r][u][w] -= e2ovov[De(p,t,s,v,nocc,nvir)] * t4 * scale;
               if (q<s && t==v) tmp[p][r][u][w] += e2ovov[De(q,t,s,v,nocc,nvir)] * t4 * scale;
               if (p<r && u==v) tmp[q][s][t][w] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t4 * scale;
               if (q<r && u==v) tmp[p][s][t][w] += e2ovov[De(q,u,r,v,nocc,nvir)] * t4 * scale;
               if (p<s && u==v) tmp[q][r][t][w] += e2ovov[De(p,u,s,v,nocc,nvir)] * t4 * scale;
               if (q<s && u==v) tmp[p][r][t][w] -= e2ovov[De(q,u,s,v,nocc,nvir)] * t4 * scale;
               if (p<r && t==w) tmp[q][s][u][v] -= e2ovov[De(p,t,r,w,nocc,nvir)] * t4 * scale;
               if (q<r && t==w) tmp[p][s][u][v] += e2ovov[De(q,t,r,w,nocc,nvir)] * t4 * scale;
               if (p<s && t==w) tmp[q][r][u][v] += e2ovov[De(p,t,s,w,nocc,nvir)] * t4 * scale;
               if (q<s && t==w) tmp[p][r][u][v] -= e2ovov[De(q,t,s,w,nocc,nvir)] * t4 * scale;
               if (p<r && u==w) tmp[q][s][t][v] += e2ovov[De(p,u,r,w,nocc,nvir)] * t4 * scale;
               if (q<r && u==w) tmp[p][s][t][v] -= e2ovov[De(q,u,r,w,nocc,nvir)] * t4 * scale;
               if (p<s && u==w) tmp[q][r][t][v] -= e2ovov[De(p,u,s,w,nocc,nvir)] * t4 * scale;
               if (q<s && u==w) tmp[p][r][t][v] += e2ovov[De(q,u,s,w,nocc,nvir)] * t4 * scale;
               if (p<r && v==t) tmp[q][s][u][w] += e2ovov[De(p,t,r,v,nocc,nvir)] * t4 * scale;
               if (q<r && v==t) tmp[p][s][u][w] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t4 * scale;
               if (p<s && v==t) tmp[q][r][u][w] -= e2ovov[De(p,t,s,v,nocc,nvir)] * t4 * scale;
               if (q<s && v==t) tmp[p][r][u][w] += e2ovov[De(q,t,s,v,nocc,nvir)] * t4 * scale;
               if (p<r && v==u) tmp[q][s][t][w] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t4 * scale;
               if (q<r && v==u) tmp[p][s][t][w] += e2ovov[De(q,u,r,v,nocc,nvir)] * t4 * scale;
               if (p<s && v==u) tmp[q][r][t][w] += e2ovov[De(p,u,s,v,nocc,nvir)] * t4 * scale;
               if (q<s && v==u) tmp[p][r][t][w] -= e2ovov[De(q,u,s,v,nocc,nvir)] * t4 * scale;
               if (p<r && w==t) tmp[q][s][u][v] -= e2ovov[De(p,t,r,w,nocc,nvir)] * t4 * scale;
               if (q<r && w==t) tmp[p][s][u][v] += e2ovov[De(q,t,r,w,nocc,nvir)] * t4 * scale;
               if (p<s && w==t) tmp[q][r][u][v] += e2ovov[De(p,t,s,w,nocc,nvir)] * t4 * scale;
               if (q<s && w==t) tmp[p][r][u][v] -= e2ovov[De(q,t,s,w,nocc,nvir)] * t4 * scale;
               if (p<r && w==u) tmp[q][s][t][v] += e2ovov[De(p,u,r,w,nocc,nvir)] * t4 * scale;
               if (q<r && w==u) tmp[p][s][t][v] -= e2ovov[De(q,u,r,w,nocc,nvir)] * t4 * scale;
               if (p<s && w==u) tmp[q][r][t][v] -= e2ovov[De(p,u,s,w,nocc,nvir)] * t4 * scale;
               if (q<s && w==u) tmp[p][r][t][v] += e2ovov[De(q,u,s,w,nocc,nvir)] * t4 * scale;
   
               scale = 0.25;
               if (p==r && t==v) tmp[q][s][u][w] += e2ovov[De(p,t,r,v,nocc,nvir)] * t4 * scale;
               if (q==r && t==v) tmp[p][s][u][w] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t4 * scale;
               if (p==s && t==v) tmp[q][r][u][w] -= e2ovov[De(p,t,s,v,nocc,nvir)] * t4 * scale;
               if (q==s && t==v) tmp[p][r][u][w] += e2ovov[De(q,t,s,v,nocc,nvir)] * t4 * scale;
               if (p==r && u==v) tmp[q][s][t][w] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t4 * scale;
               if (q==r && u==v) tmp[p][s][t][w] += e2ovov[De(q,u,r,v,nocc,nvir)] * t4 * scale;
               if (p==s && u==v) tmp[q][r][t][w] += e2ovov[De(p,u,s,v,nocc,nvir)] * t4 * scale;
               if (q==s && u==v) tmp[p][r][t][w] -= e2ovov[De(q,u,s,v,nocc,nvir)] * t4 * scale;
               if (p==r && t==w) tmp[q][s][u][v] -= e2ovov[De(p,t,r,w,nocc,nvir)] * t4 * scale;
               if (q==r && t==w) tmp[p][s][u][v] += e2ovov[De(q,t,r,w,nocc,nvir)] * t4 * scale;
               if (p==s && t==w) tmp[q][r][u][v] += e2ovov[De(p,t,s,w,nocc,nvir)] * t4 * scale;
               if (q==s && t==w) tmp[p][r][u][v] -= e2ovov[De(q,t,s,w,nocc,nvir)] * t4 * scale;
               if (p==r && u==w) tmp[q][s][t][v] += e2ovov[De(p,u,r,w,nocc,nvir)] * t4 * scale;
               if (q==r && u==w) tmp[p][s][t][v] -= e2ovov[De(q,u,r,w,nocc,nvir)] * t4 * scale;
               if (p==s && u==w) tmp[q][r][t][v] -= e2ovov[De(p,u,s,w,nocc,nvir)] * t4 * scale;
               if (q==s && u==w) tmp[p][r][t][v] += e2ovov[De(q,u,s,w,nocc,nvir)] * t4 * scale;
               if (p==r && v==t) tmp[q][s][u][w] += e2ovov[De(p,t,r,v,nocc,nvir)] * t4 * scale;
               if (q==r && v==t) tmp[p][s][u][w] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t4 * scale;
               if (p==s && v==t) tmp[q][r][u][w] -= e2ovov[De(p,t,s,v,nocc,nvir)] * t4 * scale;
               if (q==s && v==t) tmp[p][r][u][w] += e2ovov[De(q,t,s,v,nocc,nvir)] * t4 * scale;
               if (p==r && v==u) tmp[q][s][t][w] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t4 * scale;
               if (q==r && v==u) tmp[p][s][t][w] += e2ovov[De(q,u,r,v,nocc,nvir)] * t4 * scale;
               if (p==s && v==u) tmp[q][r][t][w] += e2ovov[De(p,u,s,v,nocc,nvir)] * t4 * scale;
               if (q==s && v==u) tmp[p][r][t][w] -= e2ovov[De(q,u,s,v,nocc,nvir)] * t4 * scale;
               if (p==r && w==t) tmp[q][s][u][v] -= e2ovov[De(p,t,r,w,nocc,nvir)] * t4 * scale;
               if (q==r && w==t) tmp[p][s][u][v] += e2ovov[De(q,t,r,w,nocc,nvir)] * t4 * scale;
               if (p==s && w==t) tmp[q][r][u][v] += e2ovov[De(p,t,s,w,nocc,nvir)] * t4 * scale;
               if (q==s && w==t) tmp[p][r][u][v] -= e2ovov[De(q,t,s,w,nocc,nvir)] * t4 * scale;
               if (p==r && w==u) tmp[q][s][t][v] += e2ovov[De(p,u,r,w,nocc,nvir)] * t4 * scale;
               if (q==r && w==u) tmp[p][s][t][v] -= e2ovov[De(q,u,r,w,nocc,nvir)] * t4 * scale;
               if (p==s && w==u) tmp[q][r][t][v] -= e2ovov[De(p,u,s,w,nocc,nvir)] * t4 * scale;
               if (q==s && w==u) tmp[p][r][t][v] += e2ovov[De(q,u,s,w,nocc,nvir)] * t4 * scale;
   
               for (it = 0; it < nocc; it++)
               for (jt = 0; jt < nocc; jt++)
               for (at = 0; at < nvir; at++)
               for (bt = 0; bt < nvir; bt++)
                   t2t4c[D(it,jt,at,bt,nocc,nvir)] += tmp[it][jt][at][bt] + tmp[jt][it][bt][at];

           }
           else if (strncmp(typ, "aaab", 4) == 0 && fabs(t4) > numzero){
               norm += 2.0*t4*t4; 
               //lsh test
               //printf ("typ=%c%c%c%c line=%s c4=%lf\n",typ[0],typ[1],typ[2],typ[3], line, t4);

               sscanf(line,"%d,%d,%d,%d,%d,%d,%d,%d",&p,&q,&r,&t,&u,&v,&s,&w);
               p += nc;
               q += nc;
               r += nc;
               s += nc;
               t += - nocc + nc;
               u += - nocc + nc;
               v += - nocc + nc;
               w += - nocc + nc;

               // exclude active space (reference space)
               //if (p >= nc_ref && q >= nc_ref && r >= nc_ref && s >= nc_ref && \
               //    r < nvir_ref && s < nvir_ref && t < nvir_ref && u < nvir_ref) continue; 

               // exclude inactive space
               if (p <= nc_ref && q <= nc_ref && r <= nc_ref && s <= nc_ref && \
                   t > nvir_ref && u > nvir_ref && v > nvir_ref && w > nvir_ref) continue; 

               //printf ("p=%d, q=%d, r=%d, t=%d, u=%d, v=%d, s=%d, w=%d",p,q,r,t,u,v,s,w);

               for (itmp = 0; itmp < nocc+nvir; itmp++)
                   det_str[itmp] = Refdet[itmp];  
               det_str[p] = 2;
               det_str[q] = 2;
               det_str[r] = 2;
               det_str[t+nocc] = 1;
               det_str[u+nocc] = 1;
               det_str[v+nocc] = 1;
   
               if (p != s && q != s && r != s) det_str[s] = 1;
               else  det_str[s] = 0;
               if (t != w && u != w && v != w) det_str[w+nocc] = 2;
               else  det_str[w+nocc] = 3;
               //parity  = parity_ab_str(det_str, nocc+nvir);
               parity = parity_ci_to_cc(p+q+r, 3, nocc);
               parity *= parity_ci_to_cc(s, 1, nocc);
   
               // interm norm of c4
               t4 = parity * t4 / c0;
   
               // extract t4 
               t4-= t1xt3aaab (p, q, r, s, t, u, v, w, nocc, nvir, t1, t3aaa, t3aab);   // may have 1e-5 bug 
               t4-= t2xt2aaab (p, q, r, s, t, u, v, w, nocc, nvir, t2aa, t2ab);         // may have 1e-3 bug 
               t4-= t1xt1xt2aaab (p, q, r, s, t, u, v, w, nocc, nvir, t1, t2aa, t2ab);  // may have 1e-5 bug 
               t4-= t1xt1xt1xt1aaab (p, q, r, s, t, u, v, w, nocc, nvir, t1);           // may have 1e-6 bug
   
               for (it=0; it< nocc; it++){
               for (jt=0; jt< nocc; jt++){
               for (at=0; at< nvir; at++){
               for (bt=0; bt< nvir; bt++){
                   tmp[it][jt][at][bt] = 0.0;
               }
               }
               }
               }
   
               tmp[r][s][v][w] += (e2ovov[De(p,t,q,u,nocc,nvir)]-e2ovov[De(p,u,q,t,nocc,nvir)]) * t4; 
               tmp[q][s][v][w] -= (e2ovov[De(p,t,r,u,nocc,nvir)]-e2ovov[De(p,u,r,t,nocc,nvir)]) * t4; 
               tmp[p][s][v][w] += (e2ovov[De(q,t,s,u,nocc,nvir)]-e2ovov[De(q,u,s,t,nocc,nvir)]) * t4; 
               tmp[r][s][u][w] -= (e2ovov[De(p,t,q,v,nocc,nvir)]-e2ovov[De(p,v,q,t,nocc,nvir)]) * t4; 
               tmp[q][s][u][w] += (e2ovov[De(p,t,r,v,nocc,nvir)]-e2ovov[De(p,v,r,t,nocc,nvir)]) * t4; 
               tmp[p][s][u][w] -= (e2ovov[De(q,t,s,v,nocc,nvir)]-e2ovov[De(q,v,s,t,nocc,nvir)]) * t4; 
               tmp[r][s][t][w] += (e2ovov[De(p,u,q,v,nocc,nvir)]-e2ovov[De(p,v,q,u,nocc,nvir)]) * t4; 
               tmp[q][s][t][w] -= (e2ovov[De(p,u,r,v,nocc,nvir)]-e2ovov[De(p,v,r,u,nocc,nvir)]) * t4; 
               tmp[p][s][t][w] += (e2ovov[De(q,u,s,v,nocc,nvir)]-e2ovov[De(q,v,s,u,nocc,nvir)]) * t4; 
   
               for (it = 0; it < nocc; it++)
               for (jt = 0; jt < nocc; jt++)
               for (at = 0; at < nvir; at++)
               for (bt = 0; bt < nvir; bt++)
                   t2t4c[D(it,jt,at,bt,nocc,nvir)] += tmp[it][jt][at][bt] + tmp[jt][it][bt][at];

           }
       }
       fclose(fp);
    }
    else
    {
       // error message
    }


    for (it=0; it< nocc; it++){
        for (jt=0; jt< nocc; jt++){
            for (at=0; at< nvir; at++){
                free(tmp[it][jt][at]);
            }
            free(tmp[it][jt]);
        }
        free(tmp[it]);
    }   
    free(tmp);

    printf ("0SDTQ (Q) =    %f   ( %f )\n", norm, norm-norm0SDT);

//    for (itmp = 0; itmp < nocc+nvir; itmp++){
//        if (itmp<nocc) Refdet[itmp] = 3;  
//        else           Refdet[itmp] = 0;
//    }
//
//    // t2t4c += e2ovov * t4aabb
//    for (idet = 0; idet < n_aabb; idet++) {
//        t4 = c4_aabb[idet]; 
//        if(fabs(t4) > numzero) 
//        {
//            p = p_aabb[idet];
//            q = q_aabb[idet];
//            r = r_aabb[idet];
//            s = s_aabb[idet];
//            t = t_aabb[idet];
//            u = u_aabb[idet];
//            v = v_aabb[idet];
//            w = w_aabb[idet];
//
//
//            if(!(p == 2 && q == 3 && t == 0 && u == 1 && \
//                 r == 2 && s == 3 && v == 0 && w == 1)) continue;
//
//            for (itmp = 0; itmp < nocc+nvir; itmp++)
//                det_str[itmp] = Refdet[itmp];  
//
//            det_str[p] = 2;
//            det_str[q] = 2;
//            det_str[t+nocc] = 1;
//            det_str[u+nocc] = 1;
//
//            if (p != r && q != r) det_str[r] = 1;
//            else  det_str[r] = 0;
//            if (p != s && q != s) det_str[s] = 1;
//            else  det_str[s] = 0;
//            if (t != v && u != v) det_str[v+nocc] = 2;
//            else  det_str[v+nocc] = 3;
//            if (t != w && u != w) det_str[w+nocc] = 2;
//            else  det_str[w+nocc] = 3;
//
//            parity  = parity_ab_str(det_str, nocc+nvir);
//            parity *= parity_ci_to_cc(p+q, 2, nocc);
//            parity *= parity_ci_to_cc(r+s, 2, nocc);
//
//            // interm norm of c4
//            t4 = parity * t4 / c0;
//            // lsh test 
//            printf("c4 mem %20.10f \n",t4);
//
//
//            // extract t4 
//            t4-= t1xt3aabb(p, q, r, s, t, u, v, w, nocc, nvir, t1, t3aab); 
//            t4-= t2xt2aabb(p, q, r, s, t, u, v, w, nocc, nvir, t2aa, t2ab); 
//            t4-= t1xt1xt2aabb(p, q, r, s, t, u, v, w, nocc, nvir, t1, t2aa, t2ab); 
//            t4-= t1xt1xt1xt1aabb(p, q, r, s, t, u, v, w, nocc, nvir, t1);   // may have bug 
//
//            // lsh test 
//            printf("t4 mem %20.10f \n",t4);
//
//            for (itmp = 0; itmp < dlen; itmp++)
//                tmp[itmp] = 0.0;
//
//            if (p<r && t<v) tmp[D(q,s,u,w,nocc,nvir)] += e2ovov[De(p,t,r,v,nocc,nvir)] * t4;
//            if (q<r && t<v) tmp[D(p,s,u,w,nocc,nvir)] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t4;
//            if (p<s && t<v) tmp[D(q,r,u,w,nocc,nvir)] -= e2ovov[De(p,t,s,v,nocc,nvir)] * t4;
//            if (q<s && t<v) tmp[D(p,r,u,w,nocc,nvir)] += e2ovov[De(q,t,s,v,nocc,nvir)] * t4;
//            if (p<r && u<v) tmp[D(q,s,t,w,nocc,nvir)] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t4;
//            if (q<r && u<v) tmp[D(p,s,t,w,nocc,nvir)] += e2ovov[De(q,u,r,v,nocc,nvir)] * t4;
//            if (p<s && u<v) tmp[D(q,r,t,w,nocc,nvir)] += e2ovov[De(p,u,s,v,nocc,nvir)] * t4;
//            if (q<s && u<v) tmp[D(p,r,t,w,nocc,nvir)] -= e2ovov[De(q,u,s,v,nocc,nvir)] * t4;
//            if (p<r && t<w) tmp[D(q,s,u,v,nocc,nvir)] -= e2ovov[De(p,t,r,w,nocc,nvir)] * t4;
//            if (q<r && t<w) tmp[D(p,s,u,v,nocc,nvir)] += e2ovov[De(q,t,r,w,nocc,nvir)] * t4;
//            if (p<s && t<w) tmp[D(q,r,u,v,nocc,nvir)] += e2ovov[De(p,t,s,w,nocc,nvir)] * t4;
//            if (q<s && t<w) tmp[D(p,r,u,v,nocc,nvir)] -= e2ovov[De(q,t,s,w,nocc,nvir)] * t4;
//            if (p<r && u<w) tmp[D(q,s,t,v,nocc,nvir)] += e2ovov[De(p,u,r,w,nocc,nvir)] * t4;
//            if (q<r && u<w) tmp[D(p,s,t,v,nocc,nvir)] -= e2ovov[De(q,u,r,w,nocc,nvir)] * t4;
//            if (p<s && u<w) tmp[D(q,r,t,v,nocc,nvir)] -= e2ovov[De(p,u,s,w,nocc,nvir)] * t4;
//            if (q<s && u<w) tmp[D(p,r,t,v,nocc,nvir)] += e2ovov[De(q,u,s,w,nocc,nvir)] * t4;
//            if (p<r && v<t) tmp[D(q,s,u,w,nocc,nvir)] += e2ovov[De(p,t,r,v,nocc,nvir)] * t4;
//            if (q<r && v<t) tmp[D(p,s,u,w,nocc,nvir)] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t4;
//            if (p<s && v<t) tmp[D(q,r,u,w,nocc,nvir)] -= e2ovov[De(p,t,s,v,nocc,nvir)] * t4;
//            if (q<s && v<t) tmp[D(p,r,u,w,nocc,nvir)] += e2ovov[De(q,t,s,v,nocc,nvir)] * t4;
//            if (p<r && v<u) tmp[D(q,s,t,w,nocc,nvir)] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t4;
//            if (q<r && v<u) tmp[D(p,s,t,w,nocc,nvir)] += e2ovov[De(q,u,r,v,nocc,nvir)] * t4;
//            if (p<s && v<u) tmp[D(q,r,t,w,nocc,nvir)] += e2ovov[De(p,u,s,v,nocc,nvir)] * t4;
//            if (q<s && v<u) tmp[D(p,r,t,w,nocc,nvir)] -= e2ovov[De(q,u,s,v,nocc,nvir)] * t4;
//            if (p<r && w<t) tmp[D(q,s,u,v,nocc,nvir)] -= e2ovov[De(p,t,r,w,nocc,nvir)] * t4;
//            if (q<r && w<t) tmp[D(p,s,u,v,nocc,nvir)] += e2ovov[De(q,t,r,w,nocc,nvir)] * t4;
//            if (p<s && w<t) tmp[D(q,r,u,v,nocc,nvir)] += e2ovov[De(p,t,s,w,nocc,nvir)] * t4;
//            if (q<s && w<t) tmp[D(p,r,u,v,nocc,nvir)] -= e2ovov[De(q,t,s,w,nocc,nvir)] * t4;
//            if (p<r && w<u) tmp[D(q,s,t,v,nocc,nvir)] += e2ovov[De(p,u,r,w,nocc,nvir)] * t4;
//            if (q<r && w<u) tmp[D(p,s,t,v,nocc,nvir)] -= e2ovov[De(q,u,r,w,nocc,nvir)] * t4;
//            if (p<s && w<u) tmp[D(q,r,t,v,nocc,nvir)] -= e2ovov[De(p,u,s,w,nocc,nvir)] * t4;
//            if (q<s && w<u) tmp[D(p,r,t,v,nocc,nvir)] += e2ovov[De(q,u,s,w,nocc,nvir)] * t4;
//
//            scale = 0.5;
//            if (p==r && t<v) tmp[D(q,s,u,w,nocc,nvir)] += e2ovov[De(p,t,r,v,nocc,nvir)] * t4 * scale;
//            if (q==r && t<v) tmp[D(p,s,u,w,nocc,nvir)] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t4 * scale;
//            if (p==s && t<v) tmp[D(q,r,u,w,nocc,nvir)] -= e2ovov[De(p,t,s,v,nocc,nvir)] * t4 * scale;
//            if (q==s && t<v) tmp[D(p,r,u,w,nocc,nvir)] += e2ovov[De(q,t,s,v,nocc,nvir)] * t4 * scale;
//            if (p==r && u<v) tmp[D(q,s,t,w,nocc,nvir)] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t4 * scale;
//            if (q==r && u<v) tmp[D(p,s,t,w,nocc,nvir)] += e2ovov[De(q,u,r,v,nocc,nvir)] * t4 * scale;
//            if (p==s && u<v) tmp[D(q,r,t,w,nocc,nvir)] += e2ovov[De(p,u,s,v,nocc,nvir)] * t4 * scale;
//            if (q==s && u<v) tmp[D(p,r,t,w,nocc,nvir)] -= e2ovov[De(q,u,s,v,nocc,nvir)] * t4 * scale;
//            if (p==r && t<w) tmp[D(q,s,u,v,nocc,nvir)] -= e2ovov[De(p,t,r,w,nocc,nvir)] * t4 * scale;
//            if (q==r && t<w) tmp[D(p,s,u,v,nocc,nvir)] += e2ovov[De(q,t,r,w,nocc,nvir)] * t4 * scale;
//            if (p==s && t<w) tmp[D(q,r,u,v,nocc,nvir)] += e2ovov[De(p,t,s,w,nocc,nvir)] * t4 * scale;
//            if (q==s && t<w) tmp[D(p,r,u,v,nocc,nvir)] -= e2ovov[De(q,t,s,w,nocc,nvir)] * t4 * scale;
//            if (p==r && u<w) tmp[D(q,s,t,v,nocc,nvir)] += e2ovov[De(p,u,r,w,nocc,nvir)] * t4 * scale;
//            if (q==r && u<w) tmp[D(p,s,t,v,nocc,nvir)] -= e2ovov[De(q,u,r,w,nocc,nvir)] * t4 * scale;
//            if (p==s && u<w) tmp[D(q,r,t,v,nocc,nvir)] -= e2ovov[De(p,u,s,w,nocc,nvir)] * t4 * scale;
//            if (q==s && u<w) tmp[D(p,r,t,v,nocc,nvir)] += e2ovov[De(q,u,s,w,nocc,nvir)] * t4 * scale;
//            if (p==r && v<t) tmp[D(q,s,u,w,nocc,nvir)] += e2ovov[De(p,t,r,v,nocc,nvir)] * t4 * scale;
//            if (q==r && v<t) tmp[D(p,s,u,w,nocc,nvir)] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t4 * scale;
//            if (p==s && v<t) tmp[D(q,r,u,w,nocc,nvir)] -= e2ovov[De(p,t,s,v,nocc,nvir)] * t4 * scale;
//            if (q==s && v<t) tmp[D(p,r,u,w,nocc,nvir)] += e2ovov[De(q,t,s,v,nocc,nvir)] * t4 * scale;
//            if (p==r && v<u) tmp[D(q,s,t,w,nocc,nvir)] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t4 * scale;
//            if (q==r && v<u) tmp[D(p,s,t,w,nocc,nvir)] += e2ovov[De(q,u,r,v,nocc,nvir)] * t4 * scale;
//            if (p==s && v<u) tmp[D(q,r,t,w,nocc,nvir)] += e2ovov[De(p,u,s,v,nocc,nvir)] * t4 * scale;
//            if (q==s && v<u) tmp[D(p,r,t,w,nocc,nvir)] -= e2ovov[De(q,u,s,v,nocc,nvir)] * t4 * scale;
//            if (p==r && w<t) tmp[D(q,s,u,v,nocc,nvir)] -= e2ovov[De(p,t,r,w,nocc,nvir)] * t4 * scale;
//            if (q==r && w<t) tmp[D(p,s,u,v,nocc,nvir)] += e2ovov[De(q,t,r,w,nocc,nvir)] * t4 * scale;
//            if (p==s && w<t) tmp[D(q,r,u,v,nocc,nvir)] += e2ovov[De(p,t,s,w,nocc,nvir)] * t4 * scale;
//            if (q==s && w<t) tmp[D(p,r,u,v,nocc,nvir)] -= e2ovov[De(q,t,s,w,nocc,nvir)] * t4 * scale;
//            if (p==r && w<u) tmp[D(q,s,t,v,nocc,nvir)] += e2ovov[De(p,u,r,w,nocc,nvir)] * t4 * scale;
//            if (q==r && w<u) tmp[D(p,s,t,v,nocc,nvir)] -= e2ovov[De(q,u,r,w,nocc,nvir)] * t4 * scale;
//            if (p==s && w<u) tmp[D(q,r,t,v,nocc,nvir)] -= e2ovov[De(p,u,s,w,nocc,nvir)] * t4 * scale;
//            if (q==s && w<u) tmp[D(p,r,t,v,nocc,nvir)] += e2ovov[De(q,u,s,w,nocc,nvir)] * t4 * scale;
//
//            scale = 0.5;
//            if (p<r && t==v) tmp[D(q,s,u,w,nocc,nvir)] += e2ovov[De(p,t,r,v,nocc,nvir)] * t4 * scale;
//            if (q<r && t==v) tmp[D(p,s,u,w,nocc,nvir)] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t4 * scale;
//            if (p<s && t==v) tmp[D(q,r,u,w,nocc,nvir)] -= e2ovov[De(p,t,s,v,nocc,nvir)] * t4 * scale;
//            if (q<s && t==v) tmp[D(p,r,u,w,nocc,nvir)] += e2ovov[De(q,t,s,v,nocc,nvir)] * t4 * scale;
//            if (p<r && u==v) tmp[D(q,s,t,w,nocc,nvir)] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t4 * scale;
//            if (q<r && u==v) tmp[D(p,s,t,w,nocc,nvir)] += e2ovov[De(q,u,r,v,nocc,nvir)] * t4 * scale;
//            if (p<s && u==v) tmp[D(q,r,t,w,nocc,nvir)] += e2ovov[De(p,u,s,v,nocc,nvir)] * t4 * scale;
//            if (q<s && u==v) tmp[D(p,r,t,w,nocc,nvir)] -= e2ovov[De(q,u,s,v,nocc,nvir)] * t4 * scale;
//            if (p<r && t==w) tmp[D(q,s,u,v,nocc,nvir)] -= e2ovov[De(p,t,r,w,nocc,nvir)] * t4 * scale;
//            if (q<r && t==w) tmp[D(p,s,u,v,nocc,nvir)] += e2ovov[De(q,t,r,w,nocc,nvir)] * t4 * scale;
//            if (p<s && t==w) tmp[D(q,r,u,v,nocc,nvir)] += e2ovov[De(p,t,s,w,nocc,nvir)] * t4 * scale;
//            if (q<s && t==w) tmp[D(p,r,u,v,nocc,nvir)] -= e2ovov[De(q,t,s,w,nocc,nvir)] * t4 * scale;
//            if (p<r && u==w) tmp[D(q,s,t,v,nocc,nvir)] += e2ovov[De(p,u,r,w,nocc,nvir)] * t4 * scale;
//            if (q<r && u==w) tmp[D(p,s,t,v,nocc,nvir)] -= e2ovov[De(q,u,r,w,nocc,nvir)] * t4 * scale;
//            if (p<s && u==w) tmp[D(q,r,t,v,nocc,nvir)] -= e2ovov[De(p,u,s,w,nocc,nvir)] * t4 * scale;
//            if (q<s && u==w) tmp[D(p,r,t,v,nocc,nvir)] += e2ovov[De(q,u,s,w,nocc,nvir)] * t4 * scale;
//            if (p<r && v==t) tmp[D(q,s,u,w,nocc,nvir)] += e2ovov[De(p,t,r,v,nocc,nvir)] * t4 * scale;
//            if (q<r && v==t) tmp[D(p,s,u,w,nocc,nvir)] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t4 * scale;
//            if (p<s && v==t) tmp[D(q,r,u,w,nocc,nvir)] -= e2ovov[De(p,t,s,v,nocc,nvir)] * t4 * scale;
//            if (q<s && v==t) tmp[D(p,r,u,w,nocc,nvir)] += e2ovov[De(q,t,s,v,nocc,nvir)] * t4 * scale;
//            if (p<r && v==u) tmp[D(q,s,t,w,nocc,nvir)] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t4 * scale;
//            if (q<r && v==u) tmp[D(p,s,t,w,nocc,nvir)] += e2ovov[De(q,u,r,v,nocc,nvir)] * t4 * scale;
//            if (p<s && v==u) tmp[D(q,r,t,w,nocc,nvir)] += e2ovov[De(p,u,s,v,nocc,nvir)] * t4 * scale;
//            if (q<s && v==u) tmp[D(p,r,t,w,nocc,nvir)] -= e2ovov[De(q,u,s,v,nocc,nvir)] * t4 * scale;
//            if (p<r && w==t) tmp[D(q,s,u,v,nocc,nvir)] -= e2ovov[De(p,t,r,w,nocc,nvir)] * t4 * scale;
//            if (q<r && w==t) tmp[D(p,s,u,v,nocc,nvir)] += e2ovov[De(q,t,r,w,nocc,nvir)] * t4 * scale;
//            if (p<s && w==t) tmp[D(q,r,u,v,nocc,nvir)] += e2ovov[De(p,t,s,w,nocc,nvir)] * t4 * scale;
//            if (q<s && w==t) tmp[D(p,r,u,v,nocc,nvir)] -= e2ovov[De(q,t,s,w,nocc,nvir)] * t4 * scale;
//            if (p<r && w==u) tmp[D(q,s,t,v,nocc,nvir)] += e2ovov[De(p,u,r,w,nocc,nvir)] * t4 * scale;
//            if (q<r && w==u) tmp[D(p,s,t,v,nocc,nvir)] -= e2ovov[De(q,u,r,w,nocc,nvir)] * t4 * scale;
//            if (p<s && w==u) tmp[D(q,r,t,v,nocc,nvir)] -= e2ovov[De(p,u,s,w,nocc,nvir)] * t4 * scale;
//            if (q<s && w==u) tmp[D(p,r,t,v,nocc,nvir)] += e2ovov[De(q,u,s,w,nocc,nvir)] * t4 * scale;
//
//            scale = 0.25;
//            if (p==r && t==v) tmp[D(q,s,u,w,nocc,nvir)] += e2ovov[De(p,t,r,v,nocc,nvir)] * t4 * scale;
//            if (q==r && t==v) tmp[D(p,s,u,w,nocc,nvir)] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t4 * scale;
//            if (p==s && t==v) tmp[D(q,r,u,w,nocc,nvir)] -= e2ovov[De(p,t,s,v,nocc,nvir)] * t4 * scale;
//            if (q==s && t==v) tmp[D(p,r,u,w,nocc,nvir)] += e2ovov[De(q,t,s,v,nocc,nvir)] * t4 * scale;
//            if (p==r && u==v) tmp[D(q,s,t,w,nocc,nvir)] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t4 * scale;
//            if (q==r && u==v) tmp[D(p,s,t,w,nocc,nvir)] += e2ovov[De(q,u,r,v,nocc,nvir)] * t4 * scale;
//            if (p==s && u==v) tmp[D(q,r,t,w,nocc,nvir)] += e2ovov[De(p,u,s,v,nocc,nvir)] * t4 * scale;
//            if (q==s && u==v) tmp[D(p,r,t,w,nocc,nvir)] -= e2ovov[De(q,u,s,v,nocc,nvir)] * t4 * scale;
//            if (p==r && t==w) tmp[D(q,s,u,v,nocc,nvir)] -= e2ovov[De(p,t,r,w,nocc,nvir)] * t4 * scale;
//            if (q==r && t==w) tmp[D(p,s,u,v,nocc,nvir)] += e2ovov[De(q,t,r,w,nocc,nvir)] * t4 * scale;
//            if (p==s && t==w) tmp[D(q,r,u,v,nocc,nvir)] += e2ovov[De(p,t,s,w,nocc,nvir)] * t4 * scale;
//            if (q==s && t==w) tmp[D(p,r,u,v,nocc,nvir)] -= e2ovov[De(q,t,s,w,nocc,nvir)] * t4 * scale;
//            if (p==r && u==w) tmp[D(q,s,t,v,nocc,nvir)] += e2ovov[De(p,u,r,w,nocc,nvir)] * t4 * scale;
//            if (q==r && u==w) tmp[D(p,s,t,v,nocc,nvir)] -= e2ovov[De(q,u,r,w,nocc,nvir)] * t4 * scale;
//            if (p==s && u==w) tmp[D(q,r,t,v,nocc,nvir)] -= e2ovov[De(p,u,s,w,nocc,nvir)] * t4 * scale;
//            if (q==s && u==w) tmp[D(p,r,t,v,nocc,nvir)] += e2ovov[De(q,u,s,w,nocc,nvir)] * t4 * scale;
//            if (p==r && v==t) tmp[D(q,s,u,w,nocc,nvir)] += e2ovov[De(p,t,r,v,nocc,nvir)] * t4 * scale;
//            if (q==r && v==t) tmp[D(p,s,u,w,nocc,nvir)] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t4 * scale;
//            if (p==s && v==t) tmp[D(q,r,u,w,nocc,nvir)] -= e2ovov[De(p,t,s,v,nocc,nvir)] * t4 * scale;
//            if (q==s && v==t) tmp[D(p,r,u,w,nocc,nvir)] += e2ovov[De(q,t,s,v,nocc,nvir)] * t4 * scale;
//            if (p==r && v==u) tmp[D(q,s,t,w,nocc,nvir)] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t4 * scale;
//            if (q==r && v==u) tmp[D(p,s,t,w,nocc,nvir)] += e2ovov[De(q,u,r,v,nocc,nvir)] * t4 * scale;
//            if (p==s && v==u) tmp[D(q,r,t,w,nocc,nvir)] += e2ovov[De(p,u,s,v,nocc,nvir)] * t4 * scale;
//            if (q==s && v==u) tmp[D(p,r,t,w,nocc,nvir)] -= e2ovov[De(q,u,s,v,nocc,nvir)] * t4 * scale;
//            if (p==r && w==t) tmp[D(q,s,u,v,nocc,nvir)] -= e2ovov[De(p,t,r,w,nocc,nvir)] * t4 * scale;
//            if (q==r && w==t) tmp[D(p,s,u,v,nocc,nvir)] += e2ovov[De(q,t,r,w,nocc,nvir)] * t4 * scale;
//            if (p==s && w==t) tmp[D(q,r,u,v,nocc,nvir)] += e2ovov[De(p,t,s,w,nocc,nvir)] * t4 * scale;
//            if (q==s && w==t) tmp[D(p,r,u,v,nocc,nvir)] -= e2ovov[De(q,t,s,w,nocc,nvir)] * t4 * scale;
//            if (p==r && w==u) tmp[D(q,s,t,v,nocc,nvir)] += e2ovov[De(p,u,r,w,nocc,nvir)] * t4 * scale;
//            if (q==r && w==u) tmp[D(p,s,t,v,nocc,nvir)] -= e2ovov[De(q,u,r,w,nocc,nvir)] * t4 * scale;
//            if (p==s && w==u) tmp[D(q,r,t,v,nocc,nvir)] -= e2ovov[De(p,u,s,w,nocc,nvir)] * t4 * scale;
//            if (q==s && w==u) tmp[D(p,r,t,v,nocc,nvir)] += e2ovov[De(q,u,s,w,nocc,nvir)] * t4 * scale;
//
//            for (it = 0; it < nocc; it++)
//            for (jt = 0; jt < nocc; jt++)
//            for (at = 0; at < nvir; at++)
//            for (bt = 0; bt < nvir; bt++)
//                t2t4c[D(it,jt,at,bt,nocc,nvir)] += tmp[D(it,jt,at,bt,nocc,nvir)] + tmp[D(jt,it,bt,at,nocc,nvir)];
//        }
//    }


//    // t2t4c += e2ovov * t4aaab
//    for (idet = 0; idet < n_aaab; idet++) {
//        t4 = c4_aaab[idet]; 
//        if(fabs(t4) > numzero) 
//        {
//            p = p_aaab[idet];
//            q = q_aaab[idet];
//            r = r_aaab[idet];
//            s = s_aaab[idet];
//            t = t_aaab[idet];
//            u = u_aaab[idet];
//            v = v_aaab[idet];
//            w = w_aaab[idet];
//
//            for (itmp = 0; itmp < nocc+nvir; itmp++)
//                det_str[itmp] = Refdet[itmp];  
//            det_str[p] = 2;
//            det_str[q] = 2;
//            det_str[r] = 2;
//            det_str[t+nocc] = 1;
//            det_str[u+nocc] = 1;
//            det_str[v+nocc] = 1;
//
//            if (p != s && q != s && r != s) det_str[s] = 1;
//            else  det_str[s] = 0;
//            if (t != w && u != w && v != w) det_str[w+nocc] = 2;
//            else  det_str[w+nocc] = 3;
//            parity  = parity_ab_str(det_str, nocc+nvir);
//            parity *= parity_ci_to_cc(p+q+r, 3, nocc);
//            parity *= parity_ci_to_cc(s, 1, nocc);
//
//            // interm norm of c4
//            t4 = parity * t4 / c0;
//
//            // extract t4 
//            t4-= t1xt3aaab (p, q, r, s, t, u, v, w, nocc, nvir, t1, t3aaa, t3aab);   // may have 1e-5 bug 
//            t4-= t2xt2aaab (p, q, r, s, t, u, v, w, nocc, nvir, t2aa, t2ab);         // may have 1e-3 bug 
//            t4-= t1xt1xt2aaab (p, q, r, s, t, u, v, w, nocc, nvir, t1, t2aa, t2ab);  // may have 1e-5 bug 
//            t4-= t1xt1xt1xt1aaab (p, q, r, s, t, u, v, w, nocc, nvir, t1);           // may have 1e-6 bug
//
//            for (itmp = 0; itmp < dlen; itmp++)
//                tmp[itmp] = 0.0;
//
//            tmp[D(r,s,v,w,nocc,nvir)] += (e2ovov[De(p,t,q,u,nocc,nvir)]-e2ovov[De(p,u,q,t,nocc,nvir)]) * t4; 
//            tmp[D(q,s,v,w,nocc,nvir)] -= (e2ovov[De(p,t,r,u,nocc,nvir)]-e2ovov[De(p,u,r,t,nocc,nvir)]) * t4; 
//            tmp[D(p,s,v,w,nocc,nvir)] += (e2ovov[De(q,t,s,u,nocc,nvir)]-e2ovov[De(q,u,s,t,nocc,nvir)]) * t4; 
//            tmp[D(r,s,u,w,nocc,nvir)] -= (e2ovov[De(p,t,q,v,nocc,nvir)]-e2ovov[De(p,v,q,t,nocc,nvir)]) * t4; 
//            tmp[D(q,s,u,w,nocc,nvir)] += (e2ovov[De(p,t,r,v,nocc,nvir)]-e2ovov[De(p,v,r,t,nocc,nvir)]) * t4; 
//            tmp[D(p,s,u,w,nocc,nvir)] -= (e2ovov[De(q,t,s,v,nocc,nvir)]-e2ovov[De(q,v,s,t,nocc,nvir)]) * t4; 
//            tmp[D(r,s,t,w,nocc,nvir)] += (e2ovov[De(p,u,q,v,nocc,nvir)]-e2ovov[De(p,v,q,u,nocc,nvir)]) * t4; 
//            tmp[D(q,s,t,w,nocc,nvir)] -= (e2ovov[De(p,u,r,v,nocc,nvir)]-e2ovov[De(p,v,r,u,nocc,nvir)]) * t4; 
//            tmp[D(p,s,t,w,nocc,nvir)] += (e2ovov[De(q,u,s,v,nocc,nvir)]-e2ovov[De(q,v,s,u,nocc,nvir)]) * t4; 
//
//            for (it = 0; it < nocc; it++)
//            for (jt = 0; jt < nocc; jt++)
//            for (at = 0; at < nvir; at++)
//            for (bt = 0; bt < nvir; bt++)
//                t2t4c[D(it,jt,at,bt,nocc,nvir)] += tmp[D(it,jt,at,bt,nocc,nvir)] + tmp[D(jt,it,bt,at,nocc,nvir)];
//        }
//    }

}



//void t2t4c_shci_omp(double *t2t4c, double *t1, double *t2aa, double *t2ab, double *t3aaa, double *t3aab, double *e2ovov, const int nc, const int num_det, const int nocc, const int nvir, const double numzero, const double c0, double norm) 
//{
//    //double numzero = 1e-7;
//
//    double norm0SDT = norm;
//
////    for (p=0; p<nocc; p++) {
////    for (q=0; q<nocc; q++) {
////    for (r=0; r<nvir; r++) {
////    for (s=0; s<nvir; s++) {
////        printf("%d %d %d %d %20.10lf\n",p,q,r,s,e2ovov[De(p,r,q,s,nocc,nvir)]); 
////    }
////    }
////    }
////    }
//
//    const int t2size = nocc*nocc*nvir*nvir;
//    FILE *fp;
//    char line_init[255];
//    fp = fopen("CIcoeff_shci.out", "r");
//    fscanf(fp, "%s\n", line_init);
//    if (fp) {
//
////        shared(t1, t2aa, t2ab, t3aaa, t3aab, e2ovov, nc, num_det, nocc, nvir, numzero, c0, fp, t2t4c)
//#pragma omp parallel default(none) \
//        shared(t1, t2aa, t2ab, t3aaa, t3aab, e2ovov, fp, t2t4c, norm)
//{
//       double t4, parity, scale;
//       int p, q, r, s, t, u, v, w, itmp, it, jt, at, bt, ifile;
//       char typ[4], line[255];
//       uint8_t Refdet[nocc+nvir], det_str[nocc+nvir];
//       for (itmp = 0; itmp < nocc+nvir; itmp++){
//           if (itmp<nocc) Refdet[itmp] = 3;  
//           else           Refdet[itmp] = 0;
//       }
//       double ****tmp;
//       tmp = (double ****)malloc(sizeof(double ***) * nocc); 
//       for (it=0; it< nocc; it++){
//           tmp[it] = (double ***)malloc(sizeof(double **) * nocc);
//           for (jt=0; jt< nocc; jt++){
//               tmp[it][jt] = (double **)malloc(sizeof(double *) * nvir);
//               for (at=0; at< nvir; at++){
//                   tmp[it][jt][at] = (double *)malloc(sizeof(double) * nvir);
//               }
//           }
//       }
//
//       double *t2t4c_priv;
//       t2t4c_priv = (double *)malloc(sizeof(double) * t2size); 
//       for (it=0; it< t2size; it++){
//           t2t4c_priv[it] = 0.0;
//       }
//
////       while ( !feof(fp) ){
////#pragma omp for schedule(dynamic, 100) reduction(+ : norm)
//#pragma omp for reduction(+ : norm)
//       for ( ifile=0; ifile<num_det; ifile++ ){
//           fscanf(fp, "%c%c%c%c,%s\n", &(typ[0]), &(typ[1]), &(typ[2]), &(typ[3]), line);
//           fscanf(fp, "%lf\n", &t4);
//           //printf ("typ=%c%c%c%c line=%s\n",typ[0],typ[1],typ[2],typ[3], line);
//           if (strncmp(typ, "aabb", 4) == 0 && fabs(t4) > numzero){
//               norm += t4*t4;
//               sscanf(line,"%d,%d,%d,%d,%d,%d,%d,%d",&p,&q,&t,&u,&r,&s,&v,&w);
//               p += nc;
//               q += nc;
//               r += nc;
//               s += nc;
//               t += - nocc + nc;
//               u += - nocc + nc;
//               v += - nocc + nc;
//               w += - nocc + nc;
//
////               if(!(p == 2 && q == 3 && t == 0 && u == 1 && \
////                    r == 2 && s == 3 && v == 0 && w == 1)) continue;
//   
//               for (itmp = 0; itmp < nocc+nvir; itmp++)
//                   det_str[itmp] = Refdet[itmp];  
//   
//               det_str[p] = 2;
//               det_str[q] = 2;
//               det_str[t+nocc] = 1;
//               det_str[u+nocc] = 1;
//   
//               if (p != r && q != r) det_str[r] = 1;
//               else  det_str[r] = 0;
//               if (p != s && q != s) det_str[s] = 1;
//               else  det_str[s] = 0;
//               if (t != v && u != v) det_str[v+nocc] = 2;
//               else  det_str[v+nocc] = 3;
//               if (t != w && u != w) det_str[w+nocc] = 2;
//               else  det_str[w+nocc] = 3;
//   
//               parity  = parity_ab_str(det_str, nocc+nvir);
//               parity *= parity_ci_to_cc(p+q, 2, nocc);
//               parity *= parity_ci_to_cc(r+s, 2, nocc);
//   
//               // interm norm of c4
//               t4 = parity * t4 / c0;
//               // lsh test 
////               printf("c4 mem %20.10f \n",t4);
//   
//               // extract t4 
//               t4-= t1xt3aabb(p, q, r, s, t, u, v, w, nocc, nvir, t1, t3aab); 
//               t4-= t2xt2aabb(p, q, r, s, t, u, v, w, nocc, nvir, t2aa, t2ab); 
//               t4-= t1xt1xt2aabb(p, q, r, s, t, u, v, w, nocc, nvir, t1, t2aa, t2ab); 
//               t4-= t1xt1xt1xt1aabb(p, q, r, s, t, u, v, w, nocc, nvir, t1);   // may have bug 
//   
//               // lsh test 
////               printf("t4 mem %20.10f \n",t4);
//   
//               for (it=0; it< nocc; it++){
//               for (jt=0; jt< nocc; jt++){
//               for (at=0; at< nvir; at++){
//               for (bt=0; bt< nvir; bt++){
//                   tmp[it][jt][at][bt] = 0.0;
//               }
//               }
//               }
//               }
////               printf("eris_ovov mem %20.10f \n",e2ovov[De(p,t,r,v,nocc,nvir)]);
//   
//               if (p<r && t<v) tmp[q][s][u][w] += e2ovov[De(p,t,r,v,nocc,nvir)] * t4;
//               if (q<r && t<v) tmp[p][s][u][w] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t4;
//               if (p<s && t<v) tmp[q][r][u][w] -= e2ovov[De(p,t,s,v,nocc,nvir)] * t4;
//               if (q<s && t<v) tmp[p][r][u][w] += e2ovov[De(q,t,s,v,nocc,nvir)] * t4;
//               if (p<r && u<v) tmp[q][s][t][w] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t4;
//               if (q<r && u<v) tmp[p][s][t][w] += e2ovov[De(q,u,r,v,nocc,nvir)] * t4;
//               if (p<s && u<v) tmp[q][r][t][w] += e2ovov[De(p,u,s,v,nocc,nvir)] * t4;
//               if (q<s && u<v) tmp[p][r][t][w] -= e2ovov[De(q,u,s,v,nocc,nvir)] * t4;
//               if (p<r && t<w) tmp[q][s][u][v] -= e2ovov[De(p,t,r,w,nocc,nvir)] * t4;
//               if (q<r && t<w) tmp[p][s][u][v] += e2ovov[De(q,t,r,w,nocc,nvir)] * t4;
//               if (p<s && t<w) tmp[q][r][u][v] += e2ovov[De(p,t,s,w,nocc,nvir)] * t4;
//               if (q<s && t<w) tmp[p][r][u][v] -= e2ovov[De(q,t,s,w,nocc,nvir)] * t4;
//               if (p<r && u<w) tmp[q][s][t][v] += e2ovov[De(p,u,r,w,nocc,nvir)] * t4;
//               if (q<r && u<w) tmp[p][s][t][v] -= e2ovov[De(q,u,r,w,nocc,nvir)] * t4;
//               if (p<s && u<w) tmp[q][r][t][v] -= e2ovov[De(p,u,s,w,nocc,nvir)] * t4;
//               if (q<s && u<w) tmp[p][r][t][v] += e2ovov[De(q,u,s,w,nocc,nvir)] * t4;
//               if (p<r && v<t) tmp[q][s][u][w] += e2ovov[De(p,t,r,v,nocc,nvir)] * t4;
//               if (q<r && v<t) tmp[p][s][u][w] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t4;
//               if (p<s && v<t) tmp[q][r][u][w] -= e2ovov[De(p,t,s,v,nocc,nvir)] * t4;
//               if (q<s && v<t) tmp[p][r][u][w] += e2ovov[De(q,t,s,v,nocc,nvir)] * t4;
//               if (p<r && v<u) tmp[q][s][t][w] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t4;
//               if (q<r && v<u) tmp[p][s][t][w] += e2ovov[De(q,u,r,v,nocc,nvir)] * t4;
//               if (p<s && v<u) tmp[q][r][t][w] += e2ovov[De(p,u,s,v,nocc,nvir)] * t4;
//               if (q<s && v<u) tmp[p][r][t][w] -= e2ovov[De(q,u,s,v,nocc,nvir)] * t4;
//               if (p<r && w<t) tmp[q][s][u][v] -= e2ovov[De(p,t,r,w,nocc,nvir)] * t4;
//               if (q<r && w<t) tmp[p][s][u][v] += e2ovov[De(q,t,r,w,nocc,nvir)] * t4;
//               if (p<s && w<t) tmp[q][r][u][v] += e2ovov[De(p,t,s,w,nocc,nvir)] * t4;
//               if (q<s && w<t) tmp[p][r][u][v] -= e2ovov[De(q,t,s,w,nocc,nvir)] * t4;
//               if (p<r && w<u) tmp[q][s][t][v] += e2ovov[De(p,u,r,w,nocc,nvir)] * t4;
//               if (q<r && w<u) tmp[p][s][t][v] -= e2ovov[De(q,u,r,w,nocc,nvir)] * t4;
//               if (p<s && w<u) tmp[q][r][t][v] -= e2ovov[De(p,u,s,w,nocc,nvir)] * t4;
//               if (q<s && w<u) tmp[p][r][t][v] += e2ovov[De(q,u,s,w,nocc,nvir)] * t4;
//   
//               scale = 0.5;
//               if (p==r && t<v) tmp[q][s][u][w] += e2ovov[De(p,t,r,v,nocc,nvir)] * t4 * scale;
//               if (q==r && t<v) tmp[p][s][u][w] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t4 * scale;
//               if (p==s && t<v) tmp[q][r][u][w] -= e2ovov[De(p,t,s,v,nocc,nvir)] * t4 * scale;
//               if (q==s && t<v) tmp[p][r][u][w] += e2ovov[De(q,t,s,v,nocc,nvir)] * t4 * scale;
//               if (p==r && u<v) tmp[q][s][t][w] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t4 * scale;
//               if (q==r && u<v) tmp[p][s][t][w] += e2ovov[De(q,u,r,v,nocc,nvir)] * t4 * scale;
//               if (p==s && u<v) tmp[q][r][t][w] += e2ovov[De(p,u,s,v,nocc,nvir)] * t4 * scale;
//               if (q==s && u<v) tmp[p][r][t][w] -= e2ovov[De(q,u,s,v,nocc,nvir)] * t4 * scale;
//               if (p==r && t<w) tmp[q][s][u][v] -= e2ovov[De(p,t,r,w,nocc,nvir)] * t4 * scale;
//               if (q==r && t<w) tmp[p][s][u][v] += e2ovov[De(q,t,r,w,nocc,nvir)] * t4 * scale;
//               if (p==s && t<w) tmp[q][r][u][v] += e2ovov[De(p,t,s,w,nocc,nvir)] * t4 * scale;
//               if (q==s && t<w) tmp[p][r][u][v] -= e2ovov[De(q,t,s,w,nocc,nvir)] * t4 * scale;
//               if (p==r && u<w) tmp[q][s][t][v] += e2ovov[De(p,u,r,w,nocc,nvir)] * t4 * scale;
//               if (q==r && u<w) tmp[p][s][t][v] -= e2ovov[De(q,u,r,w,nocc,nvir)] * t4 * scale;
//               if (p==s && u<w) tmp[q][r][t][v] -= e2ovov[De(p,u,s,w,nocc,nvir)] * t4 * scale;
//               if (q==s && u<w) tmp[p][r][t][v] += e2ovov[De(q,u,s,w,nocc,nvir)] * t4 * scale;
//               if (p==r && v<t) tmp[q][s][u][w] += e2ovov[De(p,t,r,v,nocc,nvir)] * t4 * scale;
//               if (q==r && v<t) tmp[p][s][u][w] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t4 * scale;
//               if (p==s && v<t) tmp[q][r][u][w] -= e2ovov[De(p,t,s,v,nocc,nvir)] * t4 * scale;
//               if (q==s && v<t) tmp[p][r][u][w] += e2ovov[De(q,t,s,v,nocc,nvir)] * t4 * scale;
//               if (p==r && v<u) tmp[q][s][t][w] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t4 * scale;
//               if (q==r && v<u) tmp[p][s][t][w] += e2ovov[De(q,u,r,v,nocc,nvir)] * t4 * scale;
//               if (p==s && v<u) tmp[q][r][t][w] += e2ovov[De(p,u,s,v,nocc,nvir)] * t4 * scale;
//               if (q==s && v<u) tmp[p][r][t][w] -= e2ovov[De(q,u,s,v,nocc,nvir)] * t4 * scale;
//               if (p==r && w<t) tmp[q][s][u][v] -= e2ovov[De(p,t,r,w,nocc,nvir)] * t4 * scale;
//               if (q==r && w<t) tmp[p][s][u][v] += e2ovov[De(q,t,r,w,nocc,nvir)] * t4 * scale;
//               if (p==s && w<t) tmp[q][r][u][v] += e2ovov[De(p,t,s,w,nocc,nvir)] * t4 * scale;
//               if (q==s && w<t) tmp[p][r][u][v] -= e2ovov[De(q,t,s,w,nocc,nvir)] * t4 * scale;
//               if (p==r && w<u) tmp[q][s][t][v] += e2ovov[De(p,u,r,w,nocc,nvir)] * t4 * scale;
//               if (q==r && w<u) tmp[p][s][t][v] -= e2ovov[De(q,u,r,w,nocc,nvir)] * t4 * scale;
//               if (p==s && w<u) tmp[q][r][t][v] -= e2ovov[De(p,u,s,w,nocc,nvir)] * t4 * scale;
//               if (q==s && w<u) tmp[p][r][t][v] += e2ovov[De(q,u,s,w,nocc,nvir)] * t4 * scale;
//   
//               scale = 0.5;
//               if (p<r && t==v) tmp[q][s][u][w] += e2ovov[De(p,t,r,v,nocc,nvir)] * t4 * scale;
//               if (q<r && t==v) tmp[p][s][u][w] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t4 * scale;
//               if (p<s && t==v) tmp[q][r][u][w] -= e2ovov[De(p,t,s,v,nocc,nvir)] * t4 * scale;
//               if (q<s && t==v) tmp[p][r][u][w] += e2ovov[De(q,t,s,v,nocc,nvir)] * t4 * scale;
//               if (p<r && u==v) tmp[q][s][t][w] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t4 * scale;
//               if (q<r && u==v) tmp[p][s][t][w] += e2ovov[De(q,u,r,v,nocc,nvir)] * t4 * scale;
//               if (p<s && u==v) tmp[q][r][t][w] += e2ovov[De(p,u,s,v,nocc,nvir)] * t4 * scale;
//               if (q<s && u==v) tmp[p][r][t][w] -= e2ovov[De(q,u,s,v,nocc,nvir)] * t4 * scale;
//               if (p<r && t==w) tmp[q][s][u][v] -= e2ovov[De(p,t,r,w,nocc,nvir)] * t4 * scale;
//               if (q<r && t==w) tmp[p][s][u][v] += e2ovov[De(q,t,r,w,nocc,nvir)] * t4 * scale;
//               if (p<s && t==w) tmp[q][r][u][v] += e2ovov[De(p,t,s,w,nocc,nvir)] * t4 * scale;
//               if (q<s && t==w) tmp[p][r][u][v] -= e2ovov[De(q,t,s,w,nocc,nvir)] * t4 * scale;
//               if (p<r && u==w) tmp[q][s][t][v] += e2ovov[De(p,u,r,w,nocc,nvir)] * t4 * scale;
//               if (q<r && u==w) tmp[p][s][t][v] -= e2ovov[De(q,u,r,w,nocc,nvir)] * t4 * scale;
//               if (p<s && u==w) tmp[q][r][t][v] -= e2ovov[De(p,u,s,w,nocc,nvir)] * t4 * scale;
//               if (q<s && u==w) tmp[p][r][t][v] += e2ovov[De(q,u,s,w,nocc,nvir)] * t4 * scale;
//               if (p<r && v==t) tmp[q][s][u][w] += e2ovov[De(p,t,r,v,nocc,nvir)] * t4 * scale;
//               if (q<r && v==t) tmp[p][s][u][w] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t4 * scale;
//               if (p<s && v==t) tmp[q][r][u][w] -= e2ovov[De(p,t,s,v,nocc,nvir)] * t4 * scale;
//               if (q<s && v==t) tmp[p][r][u][w] += e2ovov[De(q,t,s,v,nocc,nvir)] * t4 * scale;
//               if (p<r && v==u) tmp[q][s][t][w] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t4 * scale;
//               if (q<r && v==u) tmp[p][s][t][w] += e2ovov[De(q,u,r,v,nocc,nvir)] * t4 * scale;
//               if (p<s && v==u) tmp[q][r][t][w] += e2ovov[De(p,u,s,v,nocc,nvir)] * t4 * scale;
//               if (q<s && v==u) tmp[p][r][t][w] -= e2ovov[De(q,u,s,v,nocc,nvir)] * t4 * scale;
//               if (p<r && w==t) tmp[q][s][u][v] -= e2ovov[De(p,t,r,w,nocc,nvir)] * t4 * scale;
//               if (q<r && w==t) tmp[p][s][u][v] += e2ovov[De(q,t,r,w,nocc,nvir)] * t4 * scale;
//               if (p<s && w==t) tmp[q][r][u][v] += e2ovov[De(p,t,s,w,nocc,nvir)] * t4 * scale;
//               if (q<s && w==t) tmp[p][r][u][v] -= e2ovov[De(q,t,s,w,nocc,nvir)] * t4 * scale;
//               if (p<r && w==u) tmp[q][s][t][v] += e2ovov[De(p,u,r,w,nocc,nvir)] * t4 * scale;
//               if (q<r && w==u) tmp[p][s][t][v] -= e2ovov[De(q,u,r,w,nocc,nvir)] * t4 * scale;
//               if (p<s && w==u) tmp[q][r][t][v] -= e2ovov[De(p,u,s,w,nocc,nvir)] * t4 * scale;
//               if (q<s && w==u) tmp[p][r][t][v] += e2ovov[De(q,u,s,w,nocc,nvir)] * t4 * scale;
//   
//               scale = 0.25;
//               if (p==r && t==v) tmp[q][s][u][w] += e2ovov[De(p,t,r,v,nocc,nvir)] * t4 * scale;
//               if (q==r && t==v) tmp[p][s][u][w] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t4 * scale;
//               if (p==s && t==v) tmp[q][r][u][w] -= e2ovov[De(p,t,s,v,nocc,nvir)] * t4 * scale;
//               if (q==s && t==v) tmp[p][r][u][w] += e2ovov[De(q,t,s,v,nocc,nvir)] * t4 * scale;
//               if (p==r && u==v) tmp[q][s][t][w] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t4 * scale;
//               if (q==r && u==v) tmp[p][s][t][w] += e2ovov[De(q,u,r,v,nocc,nvir)] * t4 * scale;
//               if (p==s && u==v) tmp[q][r][t][w] += e2ovov[De(p,u,s,v,nocc,nvir)] * t4 * scale;
//               if (q==s && u==v) tmp[p][r][t][w] -= e2ovov[De(q,u,s,v,nocc,nvir)] * t4 * scale;
//               if (p==r && t==w) tmp[q][s][u][v] -= e2ovov[De(p,t,r,w,nocc,nvir)] * t4 * scale;
//               if (q==r && t==w) tmp[p][s][u][v] += e2ovov[De(q,t,r,w,nocc,nvir)] * t4 * scale;
//               if (p==s && t==w) tmp[q][r][u][v] += e2ovov[De(p,t,s,w,nocc,nvir)] * t4 * scale;
//               if (q==s && t==w) tmp[p][r][u][v] -= e2ovov[De(q,t,s,w,nocc,nvir)] * t4 * scale;
//               if (p==r && u==w) tmp[q][s][t][v] += e2ovov[De(p,u,r,w,nocc,nvir)] * t4 * scale;
//               if (q==r && u==w) tmp[p][s][t][v] -= e2ovov[De(q,u,r,w,nocc,nvir)] * t4 * scale;
//               if (p==s && u==w) tmp[q][r][t][v] -= e2ovov[De(p,u,s,w,nocc,nvir)] * t4 * scale;
//               if (q==s && u==w) tmp[p][r][t][v] += e2ovov[De(q,u,s,w,nocc,nvir)] * t4 * scale;
//               if (p==r && v==t) tmp[q][s][u][w] += e2ovov[De(p,t,r,v,nocc,nvir)] * t4 * scale;
//               if (q==r && v==t) tmp[p][s][u][w] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t4 * scale;
//               if (p==s && v==t) tmp[q][r][u][w] -= e2ovov[De(p,t,s,v,nocc,nvir)] * t4 * scale;
//               if (q==s && v==t) tmp[p][r][u][w] += e2ovov[De(q,t,s,v,nocc,nvir)] * t4 * scale;
//               if (p==r && v==u) tmp[q][s][t][w] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t4 * scale;
//               if (q==r && v==u) tmp[p][s][t][w] += e2ovov[De(q,u,r,v,nocc,nvir)] * t4 * scale;
//               if (p==s && v==u) tmp[q][r][t][w] += e2ovov[De(p,u,s,v,nocc,nvir)] * t4 * scale;
//               if (q==s && v==u) tmp[p][r][t][w] -= e2ovov[De(q,u,s,v,nocc,nvir)] * t4 * scale;
//               if (p==r && w==t) tmp[q][s][u][v] -= e2ovov[De(p,t,r,w,nocc,nvir)] * t4 * scale;
//               if (q==r && w==t) tmp[p][s][u][v] += e2ovov[De(q,t,r,w,nocc,nvir)] * t4 * scale;
//               if (p==s && w==t) tmp[q][r][u][v] += e2ovov[De(p,t,s,w,nocc,nvir)] * t4 * scale;
//               if (q==s && w==t) tmp[p][r][u][v] -= e2ovov[De(q,t,s,w,nocc,nvir)] * t4 * scale;
//               if (p==r && w==u) tmp[q][s][t][v] += e2ovov[De(p,u,r,w,nocc,nvir)] * t4 * scale;
//               if (q==r && w==u) tmp[p][s][t][v] -= e2ovov[De(q,u,r,w,nocc,nvir)] * t4 * scale;
//               if (p==s && w==u) tmp[q][r][t][v] -= e2ovov[De(p,u,s,w,nocc,nvir)] * t4 * scale;
//               if (q==s && w==u) tmp[p][r][t][v] += e2ovov[De(q,u,s,w,nocc,nvir)] * t4 * scale;
//   
//               for (it = 0; it < nocc; it++)
//               for (jt = 0; jt < nocc; jt++)
//               for (at = 0; at < nvir; at++)
//               for (bt = 0; bt < nvir; bt++)
//                   t2t4c_priv[D(it,jt,at,bt,nocc,nvir)] += tmp[it][jt][at][bt] + tmp[jt][it][bt][at];
//
//           }
//           else if (strncmp(typ, "aaab", 4) == 0 && fabs(t4) > numzero){
//               norm += 2.0*t4*t4; 
//               //lsh test
//               //printf ("typ=%c%c%c%c line=%s c4=%lf\n",typ[0],typ[1],typ[2],typ[3], line, t4);
//
//               sscanf(line,"%d,%d,%d,%d,%d,%d,%d,%d",&p,&q,&r,&t,&u,&v,&s,&w);
//               p += nc;
//               q += nc;
//               r += nc;
//               s += nc;
//               t += - nocc + nc;
//               u += - nocc + nc;
//               v += - nocc + nc;
//               w += - nocc + nc;
//
//               //printf ("p=%d, q=%d, r=%d, t=%d, u=%d, v=%d, s=%d, w=%d",p,q,r,t,u,v,s,w);
//
//               for (itmp = 0; itmp < nocc+nvir; itmp++)
//                   det_str[itmp] = Refdet[itmp];  
//               det_str[p] = 2;
//               det_str[q] = 2;
//               det_str[r] = 2;
//               det_str[t+nocc] = 1;
//               det_str[u+nocc] = 1;
//               det_str[v+nocc] = 1;
//   
//               if (p != s && q != s && r != s) det_str[s] = 1;
//               else  det_str[s] = 0;
//               if (t != w && u != w && v != w) det_str[w+nocc] = 2;
//               else  det_str[w+nocc] = 3;
//               parity  = parity_ab_str(det_str, nocc+nvir);
//               parity *= parity_ci_to_cc(p+q+r, 3, nocc);
//               parity *= parity_ci_to_cc(s, 1, nocc);
//   
//               // interm norm of c4
//               t4 = parity * t4 / c0;
//   
//               // extract t4 
//               t4-= t1xt3aaab (p, q, r, s, t, u, v, w, nocc, nvir, t1, t3aaa, t3aab);   // may have 1e-5 bug 
//               t4-= t2xt2aaab (p, q, r, s, t, u, v, w, nocc, nvir, t2aa, t2ab);         // may have 1e-3 bug 
//               t4-= t1xt1xt2aaab (p, q, r, s, t, u, v, w, nocc, nvir, t1, t2aa, t2ab);  // may have 1e-5 bug 
//               t4-= t1xt1xt1xt1aaab (p, q, r, s, t, u, v, w, nocc, nvir, t1);           // may have 1e-6 bug
//   
//               for (it=0; it< nocc; it++){
//               for (jt=0; jt< nocc; jt++){
//               for (at=0; at< nvir; at++){
//               for (bt=0; bt< nvir; bt++){
//                   tmp[it][jt][at][bt] = 0.0;
//               }
//               }
//               }
//               }
//   
//               tmp[r][s][v][w] += (e2ovov[De(p,t,q,u,nocc,nvir)]-e2ovov[De(p,u,q,t,nocc,nvir)]) * t4; 
//               tmp[q][s][v][w] -= (e2ovov[De(p,t,r,u,nocc,nvir)]-e2ovov[De(p,u,r,t,nocc,nvir)]) * t4; 
//               tmp[p][s][v][w] += (e2ovov[De(q,t,s,u,nocc,nvir)]-e2ovov[De(q,u,s,t,nocc,nvir)]) * t4; 
//               tmp[r][s][u][w] -= (e2ovov[De(p,t,q,v,nocc,nvir)]-e2ovov[De(p,v,q,t,nocc,nvir)]) * t4; 
//               tmp[q][s][u][w] += (e2ovov[De(p,t,r,v,nocc,nvir)]-e2ovov[De(p,v,r,t,nocc,nvir)]) * t4; 
//               tmp[p][s][u][w] -= (e2ovov[De(q,t,s,v,nocc,nvir)]-e2ovov[De(q,v,s,t,nocc,nvir)]) * t4; 
//               tmp[r][s][t][w] += (e2ovov[De(p,u,q,v,nocc,nvir)]-e2ovov[De(p,v,q,u,nocc,nvir)]) * t4; 
//               tmp[q][s][t][w] -= (e2ovov[De(p,u,r,v,nocc,nvir)]-e2ovov[De(p,v,r,u,nocc,nvir)]) * t4; 
//               tmp[p][s][t][w] += (e2ovov[De(q,u,s,v,nocc,nvir)]-e2ovov[De(q,v,s,u,nocc,nvir)]) * t4; 
//   
//               for (it = 0; it < nocc; it++)
//               for (jt = 0; jt < nocc; jt++)
//               for (at = 0; at < nvir; at++)
//               for (bt = 0; bt < nvir; bt++)
//                   t2t4c_priv[D(it,jt,at,bt,nocc,nvir)] += tmp[it][jt][at][bt] + tmp[jt][it][bt][at];
//
//           }
//       }
//
//       for (it=0; it< nocc; it++){
//           for (jt=0; jt< nocc; jt++){
//               for (at=0; at< nvir; at++){
//                   free(tmp[it][jt][at]);
//               }
//               free(tmp[it][jt]);
//           }
//           free(tmp[it]);
//       }   
//       free(tmp);
//#pragma omp critical
//       { 
//           for (it=0; it< nocc; it++){
//               for (jt=0; jt< nocc; jt++){
//                   for (at=0; at< nvir; at++){
//                       for (bt=0; bt< nvir; bt++){
//                           t2t4c[D(it,jt,at,bt,nocc,nvir)] += t2t4c_priv[D(it,jt,at,bt,nocc,nvir)];
//                       }
//                   }
//               }
//           }   
//           free(t2t4c_priv);
//       } 
//}
//
//       fclose(fp);
//    }
//    else
//    {
//       // error message
//    }
//
//    printf ("0SDTQ (Q) =    %f   ( %f )\n", norm, norm-norm0SDT);
//
//}
//

void t2t4c_shci_omp(double *t2t4c, double *t1, double *t2aa, double *t2ab, double *t3aaa, double *t3aab, double *e2ovov, const int nc, const int num_det, const int nocc, const int nvir, const double numzero, const double c0, double norm) 
{
    //double numzero = 1e-7;

    double norm0SDT = norm;

    const int t2size = nocc*nocc*nvir*nvir;

//        shared(t1, t2aa, t2ab, t3aaa, t3aab, e2ovov, nc, num_det, nocc, nvir, numzero, c0, t2t4c)
#pragma omp parallel default(none) \
        shared(t1, t2aa, t2ab, t3aaa, t3aab, e2ovov, t2t4c, norm)
{
       double t4, parity, scale;
       int p, q, r, s, t, u, v, w, itmp, it, jt, at, bt, ifile;
       char typ[4], line[255];
       uint8_t Refdet[nocc+nvir], det_str[nocc+nvir];
       for (itmp = 0; itmp < nocc+nvir; itmp++){
           if (itmp<nocc) Refdet[itmp] = 3;  
           else           Refdet[itmp] = 0;
       }
       double ****tmp;
       tmp = (double ****)malloc(sizeof(double ***) * nocc); 
       for (it=0; it< nocc; it++){
           tmp[it] = (double ***)malloc(sizeof(double **) * nocc);
           for (jt=0; jt< nocc; jt++){
               tmp[it][jt] = (double **)malloc(sizeof(double *) * nvir);
               for (at=0; at< nvir; at++){
                   tmp[it][jt][at] = (double *)malloc(sizeof(double) * nvir);
               }
           }
       }

       double *t2t4c_priv;
       t2t4c_priv = (double *)malloc(sizeof(double) * t2size); 
       for (it=0; it< t2size; it++){
           t2t4c_priv[it] = 0.0;
       }
       //lsh test
       //printf ("num_threads = %d\n",omp_get_num_threads());

       int i;
#pragma omp for reduction(+ : norm)
       for (i=0; i<omp_get_num_threads(); i++){
           char s0[20]="t4.";
           char s1[4];
           sprintf(s1, "%d", i);
           char* filename = strcat(s0,s1);
           FILE *fp = fopen(filename, "r");
           //printf ("filename = %s\n",filename);

           if (fp) {
           while ( !feof(fp) ){

           fscanf(fp, "%c%c%c%c,%s\n", &(typ[0]), &(typ[1]), &(typ[2]), &(typ[3]), line);
           fscanf(fp, "%lf\n", &t4);
           //lsh test
           //printf ("typ=%c%c%c%c line=%s\n",typ[0],typ[1],typ[2],typ[3], line);
           if (strncmp(typ, "aabb", 4) == 0 && fabs(t4) > numzero){
               norm += t4*t4;
               sscanf(line,"%d,%d,%d,%d,%d,%d,%d,%d",&p,&q,&t,&u,&r,&s,&v,&w);
               p += nc;
               q += nc;
               r += nc;
               s += nc;
               t += - nocc + nc;
               u += - nocc + nc;
               v += - nocc + nc;
               w += - nocc + nc;

//               if(!(p == 2 && q == 3 && t == 0 && u == 1 && \
//                    r == 2 && s == 3 && v == 0 && w == 1)) continue;
   
               for (itmp = 0; itmp < nocc+nvir; itmp++)
                   det_str[itmp] = Refdet[itmp];  
   
               det_str[p] = 2;
               det_str[q] = 2;
               det_str[t+nocc] = 1;
               det_str[u+nocc] = 1;
   
               if (p != r && q != r) det_str[r] = 1;
               else  det_str[r] = 0;
               if (p != s && q != s) det_str[s] = 1;
               else  det_str[s] = 0;
               if (t != v && u != v) det_str[v+nocc] = 2;
               else  det_str[v+nocc] = 3;
               if (t != w && u != w) det_str[w+nocc] = 2;
               else  det_str[w+nocc] = 3;
   
               //parity  = parity_ab_str(det_str, nocc+nvir);
               parity = parity_ci_to_cc(p+q, 2, nocc);
               parity *= parity_ci_to_cc(r+s, 2, nocc);
   
               // interm norm of c4
               t4 = parity * t4 / c0;
               // lsh test 
//               printf("c4 mem %20.10f \n",t4);
   
               // extract t4 
               t4-= t1xt3aabb(p, q, r, s, t, u, v, w, nocc, nvir, t1, t3aab); 
               t4-= t2xt2aabb(p, q, r, s, t, u, v, w, nocc, nvir, t2aa, t2ab); 
               t4-= t1xt1xt2aabb(p, q, r, s, t, u, v, w, nocc, nvir, t1, t2aa, t2ab); 
               t4-= t1xt1xt1xt1aabb(p, q, r, s, t, u, v, w, nocc, nvir, t1);   // may have bug 
   
               // lsh test 
//               printf("t4 mem %20.10f \n",t4);
   
               for (it=0; it< nocc; it++){
               for (jt=0; jt< nocc; jt++){
               for (at=0; at< nvir; at++){
               for (bt=0; bt< nvir; bt++){
                   tmp[it][jt][at][bt] = 0.0;
               }
               }
               }
               }
//               printf("eris_ovov mem %20.10f \n",e2ovov[De(p,t,r,v,nocc,nvir)]);
   
               if (p<r && t<v) tmp[q][s][u][w] += e2ovov[De(p,t,r,v,nocc,nvir)] * t4;
               if (q<r && t<v) tmp[p][s][u][w] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t4;
               if (p<s && t<v) tmp[q][r][u][w] -= e2ovov[De(p,t,s,v,nocc,nvir)] * t4;
               if (q<s && t<v) tmp[p][r][u][w] += e2ovov[De(q,t,s,v,nocc,nvir)] * t4;
               if (p<r && u<v) tmp[q][s][t][w] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t4;
               if (q<r && u<v) tmp[p][s][t][w] += e2ovov[De(q,u,r,v,nocc,nvir)] * t4;
               if (p<s && u<v) tmp[q][r][t][w] += e2ovov[De(p,u,s,v,nocc,nvir)] * t4;
               if (q<s && u<v) tmp[p][r][t][w] -= e2ovov[De(q,u,s,v,nocc,nvir)] * t4;
               if (p<r && t<w) tmp[q][s][u][v] -= e2ovov[De(p,t,r,w,nocc,nvir)] * t4;
               if (q<r && t<w) tmp[p][s][u][v] += e2ovov[De(q,t,r,w,nocc,nvir)] * t4;
               if (p<s && t<w) tmp[q][r][u][v] += e2ovov[De(p,t,s,w,nocc,nvir)] * t4;
               if (q<s && t<w) tmp[p][r][u][v] -= e2ovov[De(q,t,s,w,nocc,nvir)] * t4;
               if (p<r && u<w) tmp[q][s][t][v] += e2ovov[De(p,u,r,w,nocc,nvir)] * t4;
               if (q<r && u<w) tmp[p][s][t][v] -= e2ovov[De(q,u,r,w,nocc,nvir)] * t4;
               if (p<s && u<w) tmp[q][r][t][v] -= e2ovov[De(p,u,s,w,nocc,nvir)] * t4;
               if (q<s && u<w) tmp[p][r][t][v] += e2ovov[De(q,u,s,w,nocc,nvir)] * t4;
               if (p<r && v<t) tmp[q][s][u][w] += e2ovov[De(p,t,r,v,nocc,nvir)] * t4;
               if (q<r && v<t) tmp[p][s][u][w] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t4;
               if (p<s && v<t) tmp[q][r][u][w] -= e2ovov[De(p,t,s,v,nocc,nvir)] * t4;
               if (q<s && v<t) tmp[p][r][u][w] += e2ovov[De(q,t,s,v,nocc,nvir)] * t4;
               if (p<r && v<u) tmp[q][s][t][w] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t4;
               if (q<r && v<u) tmp[p][s][t][w] += e2ovov[De(q,u,r,v,nocc,nvir)] * t4;
               if (p<s && v<u) tmp[q][r][t][w] += e2ovov[De(p,u,s,v,nocc,nvir)] * t4;
               if (q<s && v<u) tmp[p][r][t][w] -= e2ovov[De(q,u,s,v,nocc,nvir)] * t4;
               if (p<r && w<t) tmp[q][s][u][v] -= e2ovov[De(p,t,r,w,nocc,nvir)] * t4;
               if (q<r && w<t) tmp[p][s][u][v] += e2ovov[De(q,t,r,w,nocc,nvir)] * t4;
               if (p<s && w<t) tmp[q][r][u][v] += e2ovov[De(p,t,s,w,nocc,nvir)] * t4;
               if (q<s && w<t) tmp[p][r][u][v] -= e2ovov[De(q,t,s,w,nocc,nvir)] * t4;
               if (p<r && w<u) tmp[q][s][t][v] += e2ovov[De(p,u,r,w,nocc,nvir)] * t4;
               if (q<r && w<u) tmp[p][s][t][v] -= e2ovov[De(q,u,r,w,nocc,nvir)] * t4;
               if (p<s && w<u) tmp[q][r][t][v] -= e2ovov[De(p,u,s,w,nocc,nvir)] * t4;
               if (q<s && w<u) tmp[p][r][t][v] += e2ovov[De(q,u,s,w,nocc,nvir)] * t4;
   
               scale = 0.5;
               if (p==r && t<v) tmp[q][s][u][w] += e2ovov[De(p,t,r,v,nocc,nvir)] * t4 * scale;
               if (q==r && t<v) tmp[p][s][u][w] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t4 * scale;
               if (p==s && t<v) tmp[q][r][u][w] -= e2ovov[De(p,t,s,v,nocc,nvir)] * t4 * scale;
               if (q==s && t<v) tmp[p][r][u][w] += e2ovov[De(q,t,s,v,nocc,nvir)] * t4 * scale;
               if (p==r && u<v) tmp[q][s][t][w] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t4 * scale;
               if (q==r && u<v) tmp[p][s][t][w] += e2ovov[De(q,u,r,v,nocc,nvir)] * t4 * scale;
               if (p==s && u<v) tmp[q][r][t][w] += e2ovov[De(p,u,s,v,nocc,nvir)] * t4 * scale;
               if (q==s && u<v) tmp[p][r][t][w] -= e2ovov[De(q,u,s,v,nocc,nvir)] * t4 * scale;
               if (p==r && t<w) tmp[q][s][u][v] -= e2ovov[De(p,t,r,w,nocc,nvir)] * t4 * scale;
               if (q==r && t<w) tmp[p][s][u][v] += e2ovov[De(q,t,r,w,nocc,nvir)] * t4 * scale;
               if (p==s && t<w) tmp[q][r][u][v] += e2ovov[De(p,t,s,w,nocc,nvir)] * t4 * scale;
               if (q==s && t<w) tmp[p][r][u][v] -= e2ovov[De(q,t,s,w,nocc,nvir)] * t4 * scale;
               if (p==r && u<w) tmp[q][s][t][v] += e2ovov[De(p,u,r,w,nocc,nvir)] * t4 * scale;
               if (q==r && u<w) tmp[p][s][t][v] -= e2ovov[De(q,u,r,w,nocc,nvir)] * t4 * scale;
               if (p==s && u<w) tmp[q][r][t][v] -= e2ovov[De(p,u,s,w,nocc,nvir)] * t4 * scale;
               if (q==s && u<w) tmp[p][r][t][v] += e2ovov[De(q,u,s,w,nocc,nvir)] * t4 * scale;
               if (p==r && v<t) tmp[q][s][u][w] += e2ovov[De(p,t,r,v,nocc,nvir)] * t4 * scale;
               if (q==r && v<t) tmp[p][s][u][w] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t4 * scale;
               if (p==s && v<t) tmp[q][r][u][w] -= e2ovov[De(p,t,s,v,nocc,nvir)] * t4 * scale;
               if (q==s && v<t) tmp[p][r][u][w] += e2ovov[De(q,t,s,v,nocc,nvir)] * t4 * scale;
               if (p==r && v<u) tmp[q][s][t][w] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t4 * scale;
               if (q==r && v<u) tmp[p][s][t][w] += e2ovov[De(q,u,r,v,nocc,nvir)] * t4 * scale;
               if (p==s && v<u) tmp[q][r][t][w] += e2ovov[De(p,u,s,v,nocc,nvir)] * t4 * scale;
               if (q==s && v<u) tmp[p][r][t][w] -= e2ovov[De(q,u,s,v,nocc,nvir)] * t4 * scale;
               if (p==r && w<t) tmp[q][s][u][v] -= e2ovov[De(p,t,r,w,nocc,nvir)] * t4 * scale;
               if (q==r && w<t) tmp[p][s][u][v] += e2ovov[De(q,t,r,w,nocc,nvir)] * t4 * scale;
               if (p==s && w<t) tmp[q][r][u][v] += e2ovov[De(p,t,s,w,nocc,nvir)] * t4 * scale;
               if (q==s && w<t) tmp[p][r][u][v] -= e2ovov[De(q,t,s,w,nocc,nvir)] * t4 * scale;
               if (p==r && w<u) tmp[q][s][t][v] += e2ovov[De(p,u,r,w,nocc,nvir)] * t4 * scale;
               if (q==r && w<u) tmp[p][s][t][v] -= e2ovov[De(q,u,r,w,nocc,nvir)] * t4 * scale;
               if (p==s && w<u) tmp[q][r][t][v] -= e2ovov[De(p,u,s,w,nocc,nvir)] * t4 * scale;
               if (q==s && w<u) tmp[p][r][t][v] += e2ovov[De(q,u,s,w,nocc,nvir)] * t4 * scale;
   
               scale = 0.5;
               if (p<r && t==v) tmp[q][s][u][w] += e2ovov[De(p,t,r,v,nocc,nvir)] * t4 * scale;
               if (q<r && t==v) tmp[p][s][u][w] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t4 * scale;
               if (p<s && t==v) tmp[q][r][u][w] -= e2ovov[De(p,t,s,v,nocc,nvir)] * t4 * scale;
               if (q<s && t==v) tmp[p][r][u][w] += e2ovov[De(q,t,s,v,nocc,nvir)] * t4 * scale;
               if (p<r && u==v) tmp[q][s][t][w] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t4 * scale;
               if (q<r && u==v) tmp[p][s][t][w] += e2ovov[De(q,u,r,v,nocc,nvir)] * t4 * scale;
               if (p<s && u==v) tmp[q][r][t][w] += e2ovov[De(p,u,s,v,nocc,nvir)] * t4 * scale;
               if (q<s && u==v) tmp[p][r][t][w] -= e2ovov[De(q,u,s,v,nocc,nvir)] * t4 * scale;
               if (p<r && t==w) tmp[q][s][u][v] -= e2ovov[De(p,t,r,w,nocc,nvir)] * t4 * scale;
               if (q<r && t==w) tmp[p][s][u][v] += e2ovov[De(q,t,r,w,nocc,nvir)] * t4 * scale;
               if (p<s && t==w) tmp[q][r][u][v] += e2ovov[De(p,t,s,w,nocc,nvir)] * t4 * scale;
               if (q<s && t==w) tmp[p][r][u][v] -= e2ovov[De(q,t,s,w,nocc,nvir)] * t4 * scale;
               if (p<r && u==w) tmp[q][s][t][v] += e2ovov[De(p,u,r,w,nocc,nvir)] * t4 * scale;
               if (q<r && u==w) tmp[p][s][t][v] -= e2ovov[De(q,u,r,w,nocc,nvir)] * t4 * scale;
               if (p<s && u==w) tmp[q][r][t][v] -= e2ovov[De(p,u,s,w,nocc,nvir)] * t4 * scale;
               if (q<s && u==w) tmp[p][r][t][v] += e2ovov[De(q,u,s,w,nocc,nvir)] * t4 * scale;
               if (p<r && v==t) tmp[q][s][u][w] += e2ovov[De(p,t,r,v,nocc,nvir)] * t4 * scale;
               if (q<r && v==t) tmp[p][s][u][w] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t4 * scale;
               if (p<s && v==t) tmp[q][r][u][w] -= e2ovov[De(p,t,s,v,nocc,nvir)] * t4 * scale;
               if (q<s && v==t) tmp[p][r][u][w] += e2ovov[De(q,t,s,v,nocc,nvir)] * t4 * scale;
               if (p<r && v==u) tmp[q][s][t][w] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t4 * scale;
               if (q<r && v==u) tmp[p][s][t][w] += e2ovov[De(q,u,r,v,nocc,nvir)] * t4 * scale;
               if (p<s && v==u) tmp[q][r][t][w] += e2ovov[De(p,u,s,v,nocc,nvir)] * t4 * scale;
               if (q<s && v==u) tmp[p][r][t][w] -= e2ovov[De(q,u,s,v,nocc,nvir)] * t4 * scale;
               if (p<r && w==t) tmp[q][s][u][v] -= e2ovov[De(p,t,r,w,nocc,nvir)] * t4 * scale;
               if (q<r && w==t) tmp[p][s][u][v] += e2ovov[De(q,t,r,w,nocc,nvir)] * t4 * scale;
               if (p<s && w==t) tmp[q][r][u][v] += e2ovov[De(p,t,s,w,nocc,nvir)] * t4 * scale;
               if (q<s && w==t) tmp[p][r][u][v] -= e2ovov[De(q,t,s,w,nocc,nvir)] * t4 * scale;
               if (p<r && w==u) tmp[q][s][t][v] += e2ovov[De(p,u,r,w,nocc,nvir)] * t4 * scale;
               if (q<r && w==u) tmp[p][s][t][v] -= e2ovov[De(q,u,r,w,nocc,nvir)] * t4 * scale;
               if (p<s && w==u) tmp[q][r][t][v] -= e2ovov[De(p,u,s,w,nocc,nvir)] * t4 * scale;
               if (q<s && w==u) tmp[p][r][t][v] += e2ovov[De(q,u,s,w,nocc,nvir)] * t4 * scale;
   
               scale = 0.25;
               if (p==r && t==v) tmp[q][s][u][w] += e2ovov[De(p,t,r,v,nocc,nvir)] * t4 * scale;
               if (q==r && t==v) tmp[p][s][u][w] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t4 * scale;
               if (p==s && t==v) tmp[q][r][u][w] -= e2ovov[De(p,t,s,v,nocc,nvir)] * t4 * scale;
               if (q==s && t==v) tmp[p][r][u][w] += e2ovov[De(q,t,s,v,nocc,nvir)] * t4 * scale;
               if (p==r && u==v) tmp[q][s][t][w] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t4 * scale;
               if (q==r && u==v) tmp[p][s][t][w] += e2ovov[De(q,u,r,v,nocc,nvir)] * t4 * scale;
               if (p==s && u==v) tmp[q][r][t][w] += e2ovov[De(p,u,s,v,nocc,nvir)] * t4 * scale;
               if (q==s && u==v) tmp[p][r][t][w] -= e2ovov[De(q,u,s,v,nocc,nvir)] * t4 * scale;
               if (p==r && t==w) tmp[q][s][u][v] -= e2ovov[De(p,t,r,w,nocc,nvir)] * t4 * scale;
               if (q==r && t==w) tmp[p][s][u][v] += e2ovov[De(q,t,r,w,nocc,nvir)] * t4 * scale;
               if (p==s && t==w) tmp[q][r][u][v] += e2ovov[De(p,t,s,w,nocc,nvir)] * t4 * scale;
               if (q==s && t==w) tmp[p][r][u][v] -= e2ovov[De(q,t,s,w,nocc,nvir)] * t4 * scale;
               if (p==r && u==w) tmp[q][s][t][v] += e2ovov[De(p,u,r,w,nocc,nvir)] * t4 * scale;
               if (q==r && u==w) tmp[p][s][t][v] -= e2ovov[De(q,u,r,w,nocc,nvir)] * t4 * scale;
               if (p==s && u==w) tmp[q][r][t][v] -= e2ovov[De(p,u,s,w,nocc,nvir)] * t4 * scale;
               if (q==s && u==w) tmp[p][r][t][v] += e2ovov[De(q,u,s,w,nocc,nvir)] * t4 * scale;
               if (p==r && v==t) tmp[q][s][u][w] += e2ovov[De(p,t,r,v,nocc,nvir)] * t4 * scale;
               if (q==r && v==t) tmp[p][s][u][w] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t4 * scale;
               if (p==s && v==t) tmp[q][r][u][w] -= e2ovov[De(p,t,s,v,nocc,nvir)] * t4 * scale;
               if (q==s && v==t) tmp[p][r][u][w] += e2ovov[De(q,t,s,v,nocc,nvir)] * t4 * scale;
               if (p==r && v==u) tmp[q][s][t][w] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t4 * scale;
               if (q==r && v==u) tmp[p][s][t][w] += e2ovov[De(q,u,r,v,nocc,nvir)] * t4 * scale;
               if (p==s && v==u) tmp[q][r][t][w] += e2ovov[De(p,u,s,v,nocc,nvir)] * t4 * scale;
               if (q==s && v==u) tmp[p][r][t][w] -= e2ovov[De(q,u,s,v,nocc,nvir)] * t4 * scale;
               if (p==r && w==t) tmp[q][s][u][v] -= e2ovov[De(p,t,r,w,nocc,nvir)] * t4 * scale;
               if (q==r && w==t) tmp[p][s][u][v] += e2ovov[De(q,t,r,w,nocc,nvir)] * t4 * scale;
               if (p==s && w==t) tmp[q][r][u][v] += e2ovov[De(p,t,s,w,nocc,nvir)] * t4 * scale;
               if (q==s && w==t) tmp[p][r][u][v] -= e2ovov[De(q,t,s,w,nocc,nvir)] * t4 * scale;
               if (p==r && w==u) tmp[q][s][t][v] += e2ovov[De(p,u,r,w,nocc,nvir)] * t4 * scale;
               if (q==r && w==u) tmp[p][s][t][v] -= e2ovov[De(q,u,r,w,nocc,nvir)] * t4 * scale;
               if (p==s && w==u) tmp[q][r][t][v] -= e2ovov[De(p,u,s,w,nocc,nvir)] * t4 * scale;
               if (q==s && w==u) tmp[p][r][t][v] += e2ovov[De(q,u,s,w,nocc,nvir)] * t4 * scale;
   
               for (it = 0; it < nocc; it++)
               for (jt = 0; jt < nocc; jt++)
               for (at = 0; at < nvir; at++)
               for (bt = 0; bt < nvir; bt++)
                   t2t4c_priv[D(it,jt,at,bt,nocc,nvir)] += tmp[it][jt][at][bt] + tmp[jt][it][bt][at];

           }
           else if (strncmp(typ, "aaab", 4) == 0 && fabs(t4) > numzero){
               norm += 2.0*t4*t4; 
               //lsh test
               //printf ("typ=%c%c%c%c line=%s c4=%lf\n",typ[0],typ[1],typ[2],typ[3], line, t4);

               sscanf(line,"%d,%d,%d,%d,%d,%d,%d,%d",&p,&q,&r,&t,&u,&v,&s,&w);
               p += nc;
               q += nc;
               r += nc;
               s += nc;
               t += - nocc + nc;
               u += - nocc + nc;
               v += - nocc + nc;
               w += - nocc + nc;

               //printf ("p=%d, q=%d, r=%d, t=%d, u=%d, v=%d, s=%d, w=%d",p,q,r,t,u,v,s,w);

               for (itmp = 0; itmp < nocc+nvir; itmp++)
                   det_str[itmp] = Refdet[itmp];  
               det_str[p] = 2;
               det_str[q] = 2;
               det_str[r] = 2;
               det_str[t+nocc] = 1;
               det_str[u+nocc] = 1;
               det_str[v+nocc] = 1;
   
               if (p != s && q != s && r != s) det_str[s] = 1;
               else  det_str[s] = 0;
               if (t != w && u != w && v != w) det_str[w+nocc] = 2;
               else  det_str[w+nocc] = 3;
               //parity  = parity_ab_str(det_str, nocc+nvir);
               parity = parity_ci_to_cc(p+q+r, 3, nocc);
               parity *= parity_ci_to_cc(s, 1, nocc);
   
               // interm norm of c4
               t4 = parity * t4 / c0;
   
               // extract t4 
               t4-= t1xt3aaab (p, q, r, s, t, u, v, w, nocc, nvir, t1, t3aaa, t3aab);   // may have 1e-5 bug 
               t4-= t2xt2aaab (p, q, r, s, t, u, v, w, nocc, nvir, t2aa, t2ab);         // may have 1e-3 bug 
               t4-= t1xt1xt2aaab (p, q, r, s, t, u, v, w, nocc, nvir, t1, t2aa, t2ab);  // may have 1e-5 bug 
               t4-= t1xt1xt1xt1aaab (p, q, r, s, t, u, v, w, nocc, nvir, t1);           // may have 1e-6 bug
   
               for (it=0; it< nocc; it++){
               for (jt=0; jt< nocc; jt++){
               for (at=0; at< nvir; at++){
               for (bt=0; bt< nvir; bt++){
                   tmp[it][jt][at][bt] = 0.0;
               }
               }
               }
               }
   
               tmp[r][s][v][w] += (e2ovov[De(p,t,q,u,nocc,nvir)]-e2ovov[De(p,u,q,t,nocc,nvir)]) * t4; 
               tmp[q][s][v][w] -= (e2ovov[De(p,t,r,u,nocc,nvir)]-e2ovov[De(p,u,r,t,nocc,nvir)]) * t4; 
               tmp[p][s][v][w] += (e2ovov[De(q,t,s,u,nocc,nvir)]-e2ovov[De(q,u,s,t,nocc,nvir)]) * t4; 
               tmp[r][s][u][w] -= (e2ovov[De(p,t,q,v,nocc,nvir)]-e2ovov[De(p,v,q,t,nocc,nvir)]) * t4; 
               tmp[q][s][u][w] += (e2ovov[De(p,t,r,v,nocc,nvir)]-e2ovov[De(p,v,r,t,nocc,nvir)]) * t4; 
               tmp[p][s][u][w] -= (e2ovov[De(q,t,s,v,nocc,nvir)]-e2ovov[De(q,v,s,t,nocc,nvir)]) * t4; 
               tmp[r][s][t][w] += (e2ovov[De(p,u,q,v,nocc,nvir)]-e2ovov[De(p,v,q,u,nocc,nvir)]) * t4; 
               tmp[q][s][t][w] -= (e2ovov[De(p,u,r,v,nocc,nvir)]-e2ovov[De(p,v,r,u,nocc,nvir)]) * t4; 
               tmp[p][s][t][w] += (e2ovov[De(q,u,s,v,nocc,nvir)]-e2ovov[De(q,v,s,u,nocc,nvir)]) * t4; 
   
               for (it = 0; it < nocc; it++)
               for (jt = 0; jt < nocc; jt++)
               for (at = 0; at < nvir; at++)
               for (bt = 0; bt < nvir; bt++)
                   t2t4c_priv[D(it,jt,at,bt,nocc,nvir)] += tmp[it][jt][at][bt] + tmp[jt][it][bt][at];

           }

           }
           fclose(fp);
           }
       }

       for (it=0; it< nocc; it++){
           for (jt=0; jt< nocc; jt++){
               for (at=0; at< nvir; at++){
                   free(tmp[it][jt][at]);
               }
               free(tmp[it][jt]);
           }
           free(tmp[it]);
       }   
       free(tmp);
#pragma omp critical
       { 
           for (it=0; it< nocc; it++){
               for (jt=0; jt< nocc; jt++){
                   for (at=0; at< nvir; at++){
                       for (bt=0; bt< nvir; bt++){
                           t2t4c[D(it,jt,at,bt,nocc,nvir)] += t2t4c_priv[D(it,jt,at,bt,nocc,nvir)];
                       }
                   }
               }
           }   
           free(t2t4c_priv);
       } 
}

    printf ("0SDTQ (Q) =    %f   ( %f )\n", norm, norm-norm0SDT);

}

//void t2t4c_shci_omp_otf_old(double *t2t4c, double *t1, double *t2aa, double *t2ab, double *c3aaa, double *c3aab, double *e2ovov, const int nc, const int num_det, const int nocc, const int nvir, const double numzero, const double c0, double norm) 
//{
//    //double numzero = 1e-7;
//
//    const int nocc2 = (int) nocc*(nocc-1)/2;
//    const int nocc3 = (int) nocc*(nocc-1)*(nocc-2)/6;
//    double norm0SDT = norm;
//
//    const int t2size = nocc*nocc*nvir*nvir;
//
////        shared(t1, t2aa, t2ab, c3aaa, c3aab, e2ovov, nc, num_det, nocc, nvir, numzero, c0, t2t4c)
//#pragma omp parallel default(none) \
//        shared(t1, t2aa, t2ab, c3aaa, c3aab, e2ovov, t2t4c, norm)
//{
//       double t4, parity, scale;
//       int p, q, r, s, t, u, v, w, itmp, it, jt, at, bt, ifile;
//       char typ[4], line[255];
//       uint8_t Refdet[nocc+nvir], det_str[nocc+nvir];
//       for (itmp = 0; itmp < nocc+nvir; itmp++){
//           if (itmp<nocc) Refdet[itmp] = 3;  
//           else           Refdet[itmp] = 0;
//       }
//       double ****tmp;
//       tmp = (double ****)malloc(sizeof(double ***) * nocc); 
//       for (it=0; it< nocc; it++){
//           tmp[it] = (double ***)malloc(sizeof(double **) * nocc);
//           for (jt=0; jt< nocc; jt++){
//               tmp[it][jt] = (double **)malloc(sizeof(double *) * nvir);
//               for (at=0; at< nvir; at++){
//                   tmp[it][jt][at] = (double *)malloc(sizeof(double) * nvir);
//               }
//           }
//       }
//
//       double *t2t4c_priv;
//       t2t4c_priv = (double *)malloc(sizeof(double) * t2size); 
//       for (it=0; it< t2size; it++){
//           t2t4c_priv[it] = 0.0;
//       }
//       //lsh test
//       //printf ("num_threads = %d\n",omp_get_num_threads());
//
//       int i;
//#pragma omp for reduction(+ : norm)
//       for (i=0; i<omp_get_num_threads(); i++){
//           char s0[20]="t4.";
//           char s1[4];
//           sprintf(s1, "%d", i);
//           char* filename = strcat(s0,s1);
//           FILE *fp = fopen(filename, "r");
//           //printf ("filename = %s\n",filename);
//
//           if (fp) {
//           while ( !feof(fp) ){
//
//           fscanf(fp, "%c%c%c%c,%s\n", &(typ[0]), &(typ[1]), &(typ[2]), &(typ[3]), line);
//           fscanf(fp, "%lf\n", &t4);
//           //lsh test
//           //printf ("typ=%c%c%c%c line=%s\n",typ[0],typ[1],typ[2],typ[3], line);
//           if (strncmp(typ, "aabb", 4) == 0 && fabs(t4) > numzero){
//               norm += t4*t4;
//               sscanf(line,"%d,%d,%d,%d,%d,%d,%d,%d",&p,&q,&t,&u,&r,&s,&v,&w);
//               p += nc;
//               q += nc;
//               r += nc;
//               s += nc;
//               t += - nocc + nc;
//               u += - nocc + nc;
//               v += - nocc + nc;
//               w += - nocc + nc;
//
////               if(!(p == 2 && q == 3 && t == 0 && u == 1 && \
////                    r == 2 && s == 3 && v == 0 && w == 1)) continue;
//   
//               for (itmp = 0; itmp < nocc+nvir; itmp++)
//                   det_str[itmp] = Refdet[itmp];  
//   
//               det_str[p] = 2;
//               det_str[q] = 2;
//               det_str[t+nocc] = 1;
//               det_str[u+nocc] = 1;
//   
//               if (p != r && q != r) det_str[r] = 1;
//               else  det_str[r] = 0;
//               if (p != s && q != s) det_str[s] = 1;
//               else  det_str[s] = 0;
//               if (t != v && u != v) det_str[v+nocc] = 2;
//               else  det_str[v+nocc] = 3;
//               if (t != w && u != w) det_str[w+nocc] = 2;
//               else  det_str[w+nocc] = 3;
//   
//               parity  = parity_ab_str(det_str, nocc+nvir);
//               parity *= parity_ci_to_cc(p+q, 2, nocc);
//               parity *= parity_ci_to_cc(r+s, 2, nocc);
//   
//               // interm norm of c4
//               t4 = parity * t4 / c0;
//               // lsh test 
////               printf("c4 mem %20.10f \n",t4);
//   
//               // extract t4 
//               t4-= t1xc3aabb(p, q, r, s, t, u, v, w, nocc, nocc2, nvir, t1, t2aa, t2ab, c3aab, c0); 
//               t4-= t2xt2aabb(p, q, r, s, t, u, v, w, nocc, nvir, t2aa, t2ab); 
//               t4-= t1xt1xt2aabb(p, q, r, s, t, u, v, w, nocc, nvir, t1, t2aa, t2ab); 
//               t4-= t1xt1xt1xt1aabb(p, q, r, s, t, u, v, w, nocc, nvir, t1);   // may have bug 
//   
//               // lsh test 
////               printf("t4 mem %20.10f \n",t4);
//   
//               for (it=0; it< nocc; it++)
//               for (jt=0; jt< nocc; jt++)
//               for (at=0; at< nvir; at++)
//               for (bt=0; bt< nvir; bt++)
//                   tmp[it][jt][at][bt] = 0.0;
////               printf("eris_ovov mem %20.10f \n",e2ovov[De(p,t,r,v,nocc,nvir)]);
//   
//               if (p<r && t<v) tmp[q][s][u][w] += e2ovov[De(p,t,r,v,nocc,nvir)] * t4;
//               if (q<r && t<v) tmp[p][s][u][w] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t4;
//               if (p<s && t<v) tmp[q][r][u][w] -= e2ovov[De(p,t,s,v,nocc,nvir)] * t4;
//               if (q<s && t<v) tmp[p][r][u][w] += e2ovov[De(q,t,s,v,nocc,nvir)] * t4;
//               if (p<r && u<v) tmp[q][s][t][w] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t4;
//               if (q<r && u<v) tmp[p][s][t][w] += e2ovov[De(q,u,r,v,nocc,nvir)] * t4;
//               if (p<s && u<v) tmp[q][r][t][w] += e2ovov[De(p,u,s,v,nocc,nvir)] * t4;
//               if (q<s && u<v) tmp[p][r][t][w] -= e2ovov[De(q,u,s,v,nocc,nvir)] * t4;
//               if (p<r && t<w) tmp[q][s][u][v] -= e2ovov[De(p,t,r,w,nocc,nvir)] * t4;
//               if (q<r && t<w) tmp[p][s][u][v] += e2ovov[De(q,t,r,w,nocc,nvir)] * t4;
//               if (p<s && t<w) tmp[q][r][u][v] += e2ovov[De(p,t,s,w,nocc,nvir)] * t4;
//               if (q<s && t<w) tmp[p][r][u][v] -= e2ovov[De(q,t,s,w,nocc,nvir)] * t4;
//               if (p<r && u<w) tmp[q][s][t][v] += e2ovov[De(p,u,r,w,nocc,nvir)] * t4;
//               if (q<r && u<w) tmp[p][s][t][v] -= e2ovov[De(q,u,r,w,nocc,nvir)] * t4;
//               if (p<s && u<w) tmp[q][r][t][v] -= e2ovov[De(p,u,s,w,nocc,nvir)] * t4;
//               if (q<s && u<w) tmp[p][r][t][v] += e2ovov[De(q,u,s,w,nocc,nvir)] * t4;
//               if (p<r && v<t) tmp[q][s][u][w] += e2ovov[De(p,t,r,v,nocc,nvir)] * t4;
//               if (q<r && v<t) tmp[p][s][u][w] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t4;
//               if (p<s && v<t) tmp[q][r][u][w] -= e2ovov[De(p,t,s,v,nocc,nvir)] * t4;
//               if (q<s && v<t) tmp[p][r][u][w] += e2ovov[De(q,t,s,v,nocc,nvir)] * t4;
//               if (p<r && v<u) tmp[q][s][t][w] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t4;
//               if (q<r && v<u) tmp[p][s][t][w] += e2ovov[De(q,u,r,v,nocc,nvir)] * t4;
//               if (p<s && v<u) tmp[q][r][t][w] += e2ovov[De(p,u,s,v,nocc,nvir)] * t4;
//               if (q<s && v<u) tmp[p][r][t][w] -= e2ovov[De(q,u,s,v,nocc,nvir)] * t4;
//               if (p<r && w<t) tmp[q][s][u][v] -= e2ovov[De(p,t,r,w,nocc,nvir)] * t4;
//               if (q<r && w<t) tmp[p][s][u][v] += e2ovov[De(q,t,r,w,nocc,nvir)] * t4;
//               if (p<s && w<t) tmp[q][r][u][v] += e2ovov[De(p,t,s,w,nocc,nvir)] * t4;
//               if (q<s && w<t) tmp[p][r][u][v] -= e2ovov[De(q,t,s,w,nocc,nvir)] * t4;
//               if (p<r && w<u) tmp[q][s][t][v] += e2ovov[De(p,u,r,w,nocc,nvir)] * t4;
//               if (q<r && w<u) tmp[p][s][t][v] -= e2ovov[De(q,u,r,w,nocc,nvir)] * t4;
//               if (p<s && w<u) tmp[q][r][t][v] -= e2ovov[De(p,u,s,w,nocc,nvir)] * t4;
//               if (q<s && w<u) tmp[p][r][t][v] += e2ovov[De(q,u,s,w,nocc,nvir)] * t4;
//   
//               scale = 0.5;
//               if (p==r && t<v) tmp[q][s][u][w] += e2ovov[De(p,t,r,v,nocc,nvir)] * t4 * scale;
//               if (q==r && t<v) tmp[p][s][u][w] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t4 * scale;
//               if (p==s && t<v) tmp[q][r][u][w] -= e2ovov[De(p,t,s,v,nocc,nvir)] * t4 * scale;
//               if (q==s && t<v) tmp[p][r][u][w] += e2ovov[De(q,t,s,v,nocc,nvir)] * t4 * scale;
//               if (p==r && u<v) tmp[q][s][t][w] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t4 * scale;
//               if (q==r && u<v) tmp[p][s][t][w] += e2ovov[De(q,u,r,v,nocc,nvir)] * t4 * scale;
//               if (p==s && u<v) tmp[q][r][t][w] += e2ovov[De(p,u,s,v,nocc,nvir)] * t4 * scale;
//               if (q==s && u<v) tmp[p][r][t][w] -= e2ovov[De(q,u,s,v,nocc,nvir)] * t4 * scale;
//               if (p==r && t<w) tmp[q][s][u][v] -= e2ovov[De(p,t,r,w,nocc,nvir)] * t4 * scale;
//               if (q==r && t<w) tmp[p][s][u][v] += e2ovov[De(q,t,r,w,nocc,nvir)] * t4 * scale;
//               if (p==s && t<w) tmp[q][r][u][v] += e2ovov[De(p,t,s,w,nocc,nvir)] * t4 * scale;
//               if (q==s && t<w) tmp[p][r][u][v] -= e2ovov[De(q,t,s,w,nocc,nvir)] * t4 * scale;
//               if (p==r && u<w) tmp[q][s][t][v] += e2ovov[De(p,u,r,w,nocc,nvir)] * t4 * scale;
//               if (q==r && u<w) tmp[p][s][t][v] -= e2ovov[De(q,u,r,w,nocc,nvir)] * t4 * scale;
//               if (p==s && u<w) tmp[q][r][t][v] -= e2ovov[De(p,u,s,w,nocc,nvir)] * t4 * scale;
//               if (q==s && u<w) tmp[p][r][t][v] += e2ovov[De(q,u,s,w,nocc,nvir)] * t4 * scale;
//               if (p==r && v<t) tmp[q][s][u][w] += e2ovov[De(p,t,r,v,nocc,nvir)] * t4 * scale;
//               if (q==r && v<t) tmp[p][s][u][w] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t4 * scale;
//               if (p==s && v<t) tmp[q][r][u][w] -= e2ovov[De(p,t,s,v,nocc,nvir)] * t4 * scale;
//               if (q==s && v<t) tmp[p][r][u][w] += e2ovov[De(q,t,s,v,nocc,nvir)] * t4 * scale;
//               if (p==r && v<u) tmp[q][s][t][w] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t4 * scale;
//               if (q==r && v<u) tmp[p][s][t][w] += e2ovov[De(q,u,r,v,nocc,nvir)] * t4 * scale;
//               if (p==s && v<u) tmp[q][r][t][w] += e2ovov[De(p,u,s,v,nocc,nvir)] * t4 * scale;
//               if (q==s && v<u) tmp[p][r][t][w] -= e2ovov[De(q,u,s,v,nocc,nvir)] * t4 * scale;
//               if (p==r && w<t) tmp[q][s][u][v] -= e2ovov[De(p,t,r,w,nocc,nvir)] * t4 * scale;
//               if (q==r && w<t) tmp[p][s][u][v] += e2ovov[De(q,t,r,w,nocc,nvir)] * t4 * scale;
//               if (p==s && w<t) tmp[q][r][u][v] += e2ovov[De(p,t,s,w,nocc,nvir)] * t4 * scale;
//               if (q==s && w<t) tmp[p][r][u][v] -= e2ovov[De(q,t,s,w,nocc,nvir)] * t4 * scale;
//               if (p==r && w<u) tmp[q][s][t][v] += e2ovov[De(p,u,r,w,nocc,nvir)] * t4 * scale;
//               if (q==r && w<u) tmp[p][s][t][v] -= e2ovov[De(q,u,r,w,nocc,nvir)] * t4 * scale;
//               if (p==s && w<u) tmp[q][r][t][v] -= e2ovov[De(p,u,s,w,nocc,nvir)] * t4 * scale;
//               if (q==s && w<u) tmp[p][r][t][v] += e2ovov[De(q,u,s,w,nocc,nvir)] * t4 * scale;
//   
//               scale = 0.5;
//               if (p<r && t==v) tmp[q][s][u][w] += e2ovov[De(p,t,r,v,nocc,nvir)] * t4 * scale;
//               if (q<r && t==v) tmp[p][s][u][w] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t4 * scale;
//               if (p<s && t==v) tmp[q][r][u][w] -= e2ovov[De(p,t,s,v,nocc,nvir)] * t4 * scale;
//               if (q<s && t==v) tmp[p][r][u][w] += e2ovov[De(q,t,s,v,nocc,nvir)] * t4 * scale;
//               if (p<r && u==v) tmp[q][s][t][w] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t4 * scale;
//               if (q<r && u==v) tmp[p][s][t][w] += e2ovov[De(q,u,r,v,nocc,nvir)] * t4 * scale;
//               if (p<s && u==v) tmp[q][r][t][w] += e2ovov[De(p,u,s,v,nocc,nvir)] * t4 * scale;
//               if (q<s && u==v) tmp[p][r][t][w] -= e2ovov[De(q,u,s,v,nocc,nvir)] * t4 * scale;
//               if (p<r && t==w) tmp[q][s][u][v] -= e2ovov[De(p,t,r,w,nocc,nvir)] * t4 * scale;
//               if (q<r && t==w) tmp[p][s][u][v] += e2ovov[De(q,t,r,w,nocc,nvir)] * t4 * scale;
//               if (p<s && t==w) tmp[q][r][u][v] += e2ovov[De(p,t,s,w,nocc,nvir)] * t4 * scale;
//               if (q<s && t==w) tmp[p][r][u][v] -= e2ovov[De(q,t,s,w,nocc,nvir)] * t4 * scale;
//               if (p<r && u==w) tmp[q][s][t][v] += e2ovov[De(p,u,r,w,nocc,nvir)] * t4 * scale;
//               if (q<r && u==w) tmp[p][s][t][v] -= e2ovov[De(q,u,r,w,nocc,nvir)] * t4 * scale;
//               if (p<s && u==w) tmp[q][r][t][v] -= e2ovov[De(p,u,s,w,nocc,nvir)] * t4 * scale;
//               if (q<s && u==w) tmp[p][r][t][v] += e2ovov[De(q,u,s,w,nocc,nvir)] * t4 * scale;
//               if (p<r && v==t) tmp[q][s][u][w] += e2ovov[De(p,t,r,v,nocc,nvir)] * t4 * scale;
//               if (q<r && v==t) tmp[p][s][u][w] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t4 * scale;
//               if (p<s && v==t) tmp[q][r][u][w] -= e2ovov[De(p,t,s,v,nocc,nvir)] * t4 * scale;
//               if (q<s && v==t) tmp[p][r][u][w] += e2ovov[De(q,t,s,v,nocc,nvir)] * t4 * scale;
//               if (p<r && v==u) tmp[q][s][t][w] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t4 * scale;
//               if (q<r && v==u) tmp[p][s][t][w] += e2ovov[De(q,u,r,v,nocc,nvir)] * t4 * scale;
//               if (p<s && v==u) tmp[q][r][t][w] += e2ovov[De(p,u,s,v,nocc,nvir)] * t4 * scale;
//               if (q<s && v==u) tmp[p][r][t][w] -= e2ovov[De(q,u,s,v,nocc,nvir)] * t4 * scale;
//               if (p<r && w==t) tmp[q][s][u][v] -= e2ovov[De(p,t,r,w,nocc,nvir)] * t4 * scale;
//               if (q<r && w==t) tmp[p][s][u][v] += e2ovov[De(q,t,r,w,nocc,nvir)] * t4 * scale;
//               if (p<s && w==t) tmp[q][r][u][v] += e2ovov[De(p,t,s,w,nocc,nvir)] * t4 * scale;
//               if (q<s && w==t) tmp[p][r][u][v] -= e2ovov[De(q,t,s,w,nocc,nvir)] * t4 * scale;
//               if (p<r && w==u) tmp[q][s][t][v] += e2ovov[De(p,u,r,w,nocc,nvir)] * t4 * scale;
//               if (q<r && w==u) tmp[p][s][t][v] -= e2ovov[De(q,u,r,w,nocc,nvir)] * t4 * scale;
//               if (p<s && w==u) tmp[q][r][t][v] -= e2ovov[De(p,u,s,w,nocc,nvir)] * t4 * scale;
//               if (q<s && w==u) tmp[p][r][t][v] += e2ovov[De(q,u,s,w,nocc,nvir)] * t4 * scale;
//   
//               scale = 0.25;
//               if (p==r && t==v) tmp[q][s][u][w] += e2ovov[De(p,t,r,v,nocc,nvir)] * t4 * scale;
//               if (q==r && t==v) tmp[p][s][u][w] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t4 * scale;
//               if (p==s && t==v) tmp[q][r][u][w] -= e2ovov[De(p,t,s,v,nocc,nvir)] * t4 * scale;
//               if (q==s && t==v) tmp[p][r][u][w] += e2ovov[De(q,t,s,v,nocc,nvir)] * t4 * scale;
//               if (p==r && u==v) tmp[q][s][t][w] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t4 * scale;
//               if (q==r && u==v) tmp[p][s][t][w] += e2ovov[De(q,u,r,v,nocc,nvir)] * t4 * scale;
//               if (p==s && u==v) tmp[q][r][t][w] += e2ovov[De(p,u,s,v,nocc,nvir)] * t4 * scale;
//               if (q==s && u==v) tmp[p][r][t][w] -= e2ovov[De(q,u,s,v,nocc,nvir)] * t4 * scale;
//               if (p==r && t==w) tmp[q][s][u][v] -= e2ovov[De(p,t,r,w,nocc,nvir)] * t4 * scale;
//               if (q==r && t==w) tmp[p][s][u][v] += e2ovov[De(q,t,r,w,nocc,nvir)] * t4 * scale;
//               if (p==s && t==w) tmp[q][r][u][v] += e2ovov[De(p,t,s,w,nocc,nvir)] * t4 * scale;
//               if (q==s && t==w) tmp[p][r][u][v] -= e2ovov[De(q,t,s,w,nocc,nvir)] * t4 * scale;
//               if (p==r && u==w) tmp[q][s][t][v] += e2ovov[De(p,u,r,w,nocc,nvir)] * t4 * scale;
//               if (q==r && u==w) tmp[p][s][t][v] -= e2ovov[De(q,u,r,w,nocc,nvir)] * t4 * scale;
//               if (p==s && u==w) tmp[q][r][t][v] -= e2ovov[De(p,u,s,w,nocc,nvir)] * t4 * scale;
//               if (q==s && u==w) tmp[p][r][t][v] += e2ovov[De(q,u,s,w,nocc,nvir)] * t4 * scale;
//               if (p==r && v==t) tmp[q][s][u][w] += e2ovov[De(p,t,r,v,nocc,nvir)] * t4 * scale;
//               if (q==r && v==t) tmp[p][s][u][w] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t4 * scale;
//               if (p==s && v==t) tmp[q][r][u][w] -= e2ovov[De(p,t,s,v,nocc,nvir)] * t4 * scale;
//               if (q==s && v==t) tmp[p][r][u][w] += e2ovov[De(q,t,s,v,nocc,nvir)] * t4 * scale;
//               if (p==r && v==u) tmp[q][s][t][w] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t4 * scale;
//               if (q==r && v==u) tmp[p][s][t][w] += e2ovov[De(q,u,r,v,nocc,nvir)] * t4 * scale;
//               if (p==s && v==u) tmp[q][r][t][w] += e2ovov[De(p,u,s,v,nocc,nvir)] * t4 * scale;
//               if (q==s && v==u) tmp[p][r][t][w] -= e2ovov[De(q,u,s,v,nocc,nvir)] * t4 * scale;
//               if (p==r && w==t) tmp[q][s][u][v] -= e2ovov[De(p,t,r,w,nocc,nvir)] * t4 * scale;
//               if (q==r && w==t) tmp[p][s][u][v] += e2ovov[De(q,t,r,w,nocc,nvir)] * t4 * scale;
//               if (p==s && w==t) tmp[q][r][u][v] += e2ovov[De(p,t,s,w,nocc,nvir)] * t4 * scale;
//               if (q==s && w==t) tmp[p][r][u][v] -= e2ovov[De(q,t,s,w,nocc,nvir)] * t4 * scale;
//               if (p==r && w==u) tmp[q][s][t][v] += e2ovov[De(p,u,r,w,nocc,nvir)] * t4 * scale;
//               if (q==r && w==u) tmp[p][s][t][v] -= e2ovov[De(q,u,r,w,nocc,nvir)] * t4 * scale;
//               if (p==s && w==u) tmp[q][r][t][v] -= e2ovov[De(p,u,s,w,nocc,nvir)] * t4 * scale;
//               if (q==s && w==u) tmp[p][r][t][v] += e2ovov[De(q,u,s,w,nocc,nvir)] * t4 * scale;
//   
//               for (it = 0; it < nocc; it++)
//               for (jt = 0; jt < nocc; jt++)
//               for (at = 0; at < nvir; at++)
//               for (bt = 0; bt < nvir; bt++)
//                   t2t4c_priv[D(it,jt,at,bt,nocc,nvir)] += tmp[it][jt][at][bt] + tmp[jt][it][bt][at];
//
//           }
//           else if (strncmp(typ, "aaab", 4) == 0 && fabs(t4) > numzero){
//               norm += 2.0*t4*t4; 
//               //lsh test
//               //printf ("typ=%c%c%c%c line=%s c4=%lf\n",typ[0],typ[1],typ[2],typ[3], line, t4);
//
//               sscanf(line,"%d,%d,%d,%d,%d,%d,%d,%d",&p,&q,&r,&t,&u,&v,&s,&w);
//               p += nc;
//               q += nc;
//               r += nc;
//               s += nc;
//               t += - nocc + nc;
//               u += - nocc + nc;
//               v += - nocc + nc;
//               w += - nocc + nc;
//
//               //printf ("p=%d, q=%d, r=%d, t=%d, u=%d, v=%d, s=%d, w=%d",p,q,r,t,u,v,s,w);
//
//               for (itmp = 0; itmp < nocc+nvir; itmp++)
//                   det_str[itmp] = Refdet[itmp];  
//               det_str[p] = 2;
//               det_str[q] = 2;
//               det_str[r] = 2;
//               det_str[t+nocc] = 1;
//               det_str[u+nocc] = 1;
//               det_str[v+nocc] = 1;
//   
//               if (p != s && q != s && r != s) det_str[s] = 1;
//               else  det_str[s] = 0;
//               if (t != w && u != w && v != w) det_str[w+nocc] = 2;
//               else  det_str[w+nocc] = 3;
//               parity  = parity_ab_str(det_str, nocc+nvir);
//               parity *= parity_ci_to_cc(p+q+r, 3, nocc);
//               parity *= parity_ci_to_cc(s, 1, nocc);
//   
//               // interm norm of c4
//               t4 = parity * t4 / c0;
//   
//               // extract t4 
//               t4-= t1xc3aaab (p, q, r, s, t, u, v, w, nocc, nocc2, nocc3, nvir, t1, t2aa, t2ab, c3aaa, c3aab, c0);
//               t4-= t2xt2aaab (p, q, r, s, t, u, v, w, nocc, nvir, t2aa, t2ab);         // may have 1e-3 bug 
//               t4-= t1xt1xt2aaab (p, q, r, s, t, u, v, w, nocc, nvir, t1, t2aa, t2ab);  // may have 1e-5 bug 
//               t4-= t1xt1xt1xt1aaab (p, q, r, s, t, u, v, w, nocc, nvir, t1);           // may have 1e-6 bug
//   
//               for (it=0; it< nocc; it++){
//               for (jt=0; jt< nocc; jt++){
//               for (at=0; at< nvir; at++){
//               for (bt=0; bt< nvir; bt++){
//                   tmp[it][jt][at][bt] = 0.0;
//               }
//               }
//               }
//               }
//   
//               tmp[r][s][v][w] += (e2ovov[De(p,t,q,u,nocc,nvir)]-e2ovov[De(p,u,q,t,nocc,nvir)]) * t4; 
//               tmp[q][s][v][w] -= (e2ovov[De(p,t,r,u,nocc,nvir)]-e2ovov[De(p,u,r,t,nocc,nvir)]) * t4; 
//               tmp[p][s][v][w] += (e2ovov[De(q,t,s,u,nocc,nvir)]-e2ovov[De(q,u,s,t,nocc,nvir)]) * t4; 
//               tmp[r][s][u][w] -= (e2ovov[De(p,t,q,v,nocc,nvir)]-e2ovov[De(p,v,q,t,nocc,nvir)]) * t4; 
//               tmp[q][s][u][w] += (e2ovov[De(p,t,r,v,nocc,nvir)]-e2ovov[De(p,v,r,t,nocc,nvir)]) * t4; 
//               tmp[p][s][u][w] -= (e2ovov[De(q,t,s,v,nocc,nvir)]-e2ovov[De(q,v,s,t,nocc,nvir)]) * t4; 
//               tmp[r][s][t][w] += (e2ovov[De(p,u,q,v,nocc,nvir)]-e2ovov[De(p,v,q,u,nocc,nvir)]) * t4; 
//               tmp[q][s][t][w] -= (e2ovov[De(p,u,r,v,nocc,nvir)]-e2ovov[De(p,v,r,u,nocc,nvir)]) * t4; 
//               tmp[p][s][t][w] += (e2ovov[De(q,u,s,v,nocc,nvir)]-e2ovov[De(q,v,s,u,nocc,nvir)]) * t4; 
//   
//               for (it = 0; it < nocc; it++)
//               for (jt = 0; jt < nocc; jt++)
//               for (at = 0; at < nvir; at++)
//               for (bt = 0; bt < nvir; bt++)
//                   t2t4c_priv[D(it,jt,at,bt,nocc,nvir)] += tmp[it][jt][at][bt] + tmp[jt][it][bt][at];
//
//           }
//
//           }
//           fclose(fp);
//           }
//       }
//
//       for (it=0; it< nocc; it++){
//           for (jt=0; jt< nocc; jt++){
//               for (at=0; at< nvir; at++){
//                   free(tmp[it][jt][at]);
//               }
//               free(tmp[it][jt]);
//           }
//           free(tmp[it]);
//       }   
//       free(tmp);
//#pragma omp critical
//       { 
//           for (it=0; it< nocc; it++){
//               for (jt=0; jt< nocc; jt++){
//                   for (at=0; at< nvir; at++){
//                       for (bt=0; bt< nvir; bt++){
//                           t2t4c[D(it,jt,at,bt,nocc,nvir)] += t2t4c_priv[D(it,jt,at,bt,nocc,nvir)];
//                       }
//                   }
//               }
//           }   
//           free(t2t4c_priv);
//       } 
//}
//
//
//    printf ("0SDTQ (Q) =    %f   ( %f )\n", norm, norm-norm0SDT);
//
//}

void t2t4c_shci_omp_otf(double *t2t4c, double *t1, double *t2aa, double *t2ab, double *c3aaa, double *c3aab, double *e2ovov, const int nc, const int num_det, const int nocc, const int nvir, const double numzero, const double c0, double norm) 
{
    //double numzero = 1e-7;

    const int nocc2 = (int) nocc*(nocc-1)/2;
    const int nocc3 = (int) nocc*(nocc-1)*(nocc-2)/6;
    double norm0SDT = norm;

    const int t2size = nocc*nocc*nvir*nvir;

//        shared(t1, t2aa, t2ab, c3aaa, c3aab, e2ovov, nc, num_det, nocc, nvir, numzero, c0, t2t4c)
#pragma omp parallel default(none) \
        shared(t1, t2aa, t2ab, c3aaa, c3aab, e2ovov, t2t4c, norm)
{
       double t4, parity, scale;
       int p, q, r, s, t, u, v, w, itmp, it, jt, at, bt, ifile;
       char typ[4], line[255];
       uint8_t Refdet[nocc+nvir], det_str[nocc+nvir];
       for (itmp = 0; itmp < nocc+nvir; itmp++){
           if (itmp<nocc) Refdet[itmp] = 3;  
           else           Refdet[itmp] = 0;
       }

       double *t2t4c_priv;
       t2t4c_priv = (double *)malloc(sizeof(double) * t2size); 
       for (it=0; it< t2size; it++){
           t2t4c_priv[it] = 0.0;
       }
       //lsh test
       //printf ("num_threads = %d\n",omp_get_num_threads());

       int i;
#pragma omp for reduction(+ : norm)
       for (i=0; i<omp_get_num_threads(); i++){
           char s0[20]="t4.";
           char s1[4];
           sprintf(s1, "%d", i);
           char* filename = strcat(s0,s1);
           FILE *fp = fopen(filename, "r");
           //printf ("filename = %s\n",filename);

           if (fp) {
           while ( !feof(fp) ){

           fscanf(fp, "%c%c%c%c,%s\n", &(typ[0]), &(typ[1]), &(typ[2]), &(typ[3]), line);
           fscanf(fp, "%lf\n", &t4);
           //lsh test
           //printf ("typ=%c%c%c%c line=%s\n",typ[0],typ[1],typ[2],typ[3], line);
           if (strncmp(typ, "aabb", 4) == 0 && fabs(t4) > numzero){
               norm += t4*t4;
               sscanf(line,"%d,%d,%d,%d,%d,%d,%d,%d",&p,&q,&t,&u,&r,&s,&v,&w);
               p += nc;
               q += nc;
               r += nc;
               s += nc;
               t += - nocc + nc;
               u += - nocc + nc;
               v += - nocc + nc;
               w += - nocc + nc;

//               if(!(p == 2 && q == 3 && t == 0 && u == 1 && \
//                    r == 2 && s == 3 && v == 0 && w == 1)) continue;
   
               for (itmp = 0; itmp < nocc+nvir; itmp++)
                   det_str[itmp] = Refdet[itmp];  
   
               det_str[p] = 2;
               det_str[q] = 2;
               det_str[t+nocc] = 1;
               det_str[u+nocc] = 1;
   
               if (p != r && q != r) det_str[r] = 1;
               else  det_str[r] = 0;
               if (p != s && q != s) det_str[s] = 1;
               else  det_str[s] = 0;
               if (t != v && u != v) det_str[v+nocc] = 2;
               else  det_str[v+nocc] = 3;
               if (t != w && u != w) det_str[w+nocc] = 2;
               else  det_str[w+nocc] = 3;
   
               //parity  = parity_ab_str(det_str, nocc+nvir);
               parity = parity_ci_to_cc(p+q, 2, nocc);
               parity *= parity_ci_to_cc(r+s, 2, nocc);
   
               // interm norm of c4
               t4 = parity * t4 / c0;
               // lsh test 
//               printf("c4 mem %20.10f \n",t4);
   
               // extract t4 
               t4-= t1xc3aabb(p, q, r, s, t, u, v, w, nocc, nocc2, nvir, t1, t2aa, t2ab, c3aab, c0); 
               t4-= t2xt2aabb(p, q, r, s, t, u, v, w, nocc, nvir, t2aa, t2ab); 
               t4-= t1xt1xt2aabb(p, q, r, s, t, u, v, w, nocc, nvir, t1, t2aa, t2ab); 
               t4-= t1xt1xt1xt1aabb(p, q, r, s, t, u, v, w, nocc, nvir, t1);   // may have bug 
   
               t2t4c_priv[D(q,s,u,w,nocc,nvir)] += e2ovov[De(p,t,r,v,nocc,nvir)] * t4;
               t2t4c_priv[D(p,s,u,w,nocc,nvir)] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t4;
               t2t4c_priv[D(q,s,t,w,nocc,nvir)] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t4;
               t2t4c_priv[D(p,s,t,w,nocc,nvir)] += e2ovov[De(q,u,r,v,nocc,nvir)] * t4;
               t2t4c_priv[D(q,r,u,w,nocc,nvir)] -= e2ovov[De(p,t,s,v,nocc,nvir)] * t4;
               t2t4c_priv[D(p,r,u,w,nocc,nvir)] += e2ovov[De(q,t,s,v,nocc,nvir)] * t4;
               t2t4c_priv[D(q,r,t,w,nocc,nvir)] += e2ovov[De(p,u,s,v,nocc,nvir)] * t4;
               t2t4c_priv[D(p,r,t,w,nocc,nvir)] -= e2ovov[De(q,u,s,v,nocc,nvir)] * t4;
               t2t4c_priv[D(q,s,u,v,nocc,nvir)] -= e2ovov[De(p,t,r,w,nocc,nvir)] * t4;
               t2t4c_priv[D(p,s,u,v,nocc,nvir)] += e2ovov[De(q,t,r,w,nocc,nvir)] * t4;
               t2t4c_priv[D(q,s,t,v,nocc,nvir)] += e2ovov[De(p,u,r,w,nocc,nvir)] * t4;
               t2t4c_priv[D(p,s,t,v,nocc,nvir)] -= e2ovov[De(q,u,r,w,nocc,nvir)] * t4;
               t2t4c_priv[D(q,r,u,v,nocc,nvir)] += e2ovov[De(p,t,s,w,nocc,nvir)] * t4;
               t2t4c_priv[D(p,r,u,v,nocc,nvir)] -= e2ovov[De(q,t,s,w,nocc,nvir)] * t4;
               t2t4c_priv[D(q,r,t,v,nocc,nvir)] -= e2ovov[De(p,u,s,w,nocc,nvir)] * t4;
               t2t4c_priv[D(p,r,t,v,nocc,nvir)] += e2ovov[De(q,u,s,w,nocc,nvir)] * t4;

           }
           else if (strncmp(typ, "aaab", 4) == 0 && fabs(t4) > numzero){
               norm += 2.0*t4*t4; 
               //lsh test
               //printf ("typ=%c%c%c%c line=%s c4=%lf\n",typ[0],typ[1],typ[2],typ[3], line, t4);

               sscanf(line,"%d,%d,%d,%d,%d,%d,%d,%d",&p,&q,&r,&t,&u,&v,&s,&w);
               p += nc;
               q += nc;
               r += nc;
               s += nc;
               t += - nocc + nc;
               u += - nocc + nc;
               v += - nocc + nc;
               w += - nocc + nc;

               //printf ("p=%d, q=%d, r=%d, t=%d, u=%d, v=%d, s=%d, w=%d",p,q,r,t,u,v,s,w);

               for (itmp = 0; itmp < nocc+nvir; itmp++)
                   det_str[itmp] = Refdet[itmp];  
               det_str[p] = 2;
               det_str[q] = 2;
               det_str[r] = 2;
               det_str[t+nocc] = 1;
               det_str[u+nocc] = 1;
               det_str[v+nocc] = 1;
   
               if (p != s && q != s && r != s) det_str[s] = 1;
               else  det_str[s] = 0;
               if (t != w && u != w && v != w) det_str[w+nocc] = 2;
               else  det_str[w+nocc] = 3;
               //parity  = parity_ab_str(det_str, nocc+nvir);
               parity = parity_ci_to_cc(p+q+r, 3, nocc);
               parity *= parity_ci_to_cc(s, 1, nocc);
   
               // interm norm of c4
               t4 = parity * t4 / c0;
   
               // extract t4 
               t4-= t1xc3aaab (p, q, r, s, t, u, v, w, nocc, nocc2, nocc3, nvir, t1, t2aa, t2ab, c3aaa, c3aab, c0);
               t4-= t2xt2aaab (p, q, r, s, t, u, v, w, nocc, nvir, t2aa, t2ab);         // may have 1e-3 bug 
               t4-= t1xt1xt2aaab (p, q, r, s, t, u, v, w, nocc, nvir, t1, t2aa, t2ab);  // may have 1e-5 bug 
               t4-= t1xt1xt1xt1aaab (p, q, r, s, t, u, v, w, nocc, nvir, t1);           // may have 1e-6 bug
   
               t2t4c_priv[D(r,s,v,w,nocc,nvir)] += e2ovov[De(p,t,q,u,nocc,nvir)] * t4;
               t2t4c_priv[D(q,s,v,w,nocc,nvir)] += e2ovov[De(r,t,p,u,nocc,nvir)] * t4;
               t2t4c_priv[D(p,s,v,w,nocc,nvir)] += e2ovov[De(q,t,r,u,nocc,nvir)] * t4;
               t2t4c_priv[D(r,s,v,w,nocc,nvir)] -= e2ovov[De(q,t,p,u,nocc,nvir)] * t4;
               t2t4c_priv[D(q,s,v,w,nocc,nvir)] -= e2ovov[De(p,t,r,u,nocc,nvir)] * t4;
               t2t4c_priv[D(p,s,v,w,nocc,nvir)] -= e2ovov[De(r,t,q,u,nocc,nvir)] * t4;

               t2t4c_priv[D(r,s,u,w,nocc,nvir)] += e2ovov[De(p,v,q,t,nocc,nvir)] * t4;
               t2t4c_priv[D(q,s,u,w,nocc,nvir)] += e2ovov[De(r,v,p,t,nocc,nvir)] * t4;
               t2t4c_priv[D(p,s,u,w,nocc,nvir)] += e2ovov[De(q,v,r,t,nocc,nvir)] * t4;
               t2t4c_priv[D(r,s,u,w,nocc,nvir)] -= e2ovov[De(q,v,p,t,nocc,nvir)] * t4;
               t2t4c_priv[D(q,s,u,w,nocc,nvir)] -= e2ovov[De(p,v,r,t,nocc,nvir)] * t4;
               t2t4c_priv[D(p,s,u,w,nocc,nvir)] -= e2ovov[De(r,v,q,t,nocc,nvir)] * t4;

               t2t4c_priv[D(r,s,t,w,nocc,nvir)] += e2ovov[De(p,u,q,v,nocc,nvir)] * t4;
               t2t4c_priv[D(q,s,t,w,nocc,nvir)] += e2ovov[De(r,u,p,v,nocc,nvir)] * t4;
               t2t4c_priv[D(p,s,t,w,nocc,nvir)] += e2ovov[De(q,u,r,v,nocc,nvir)] * t4;
               t2t4c_priv[D(r,s,t,w,nocc,nvir)] -= e2ovov[De(q,u,p,v,nocc,nvir)] * t4;
               t2t4c_priv[D(q,s,t,w,nocc,nvir)] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t4;
               t2t4c_priv[D(p,s,t,w,nocc,nvir)] -= e2ovov[De(r,u,q,v,nocc,nvir)] * t4;

               t2t4c_priv[D(r,s,u,w,nocc,nvir)] -= e2ovov[De(p,t,q,v,nocc,nvir)] * t4;
               t2t4c_priv[D(q,s,u,w,nocc,nvir)] -= e2ovov[De(r,t,p,v,nocc,nvir)] * t4;
               t2t4c_priv[D(p,s,u,w,nocc,nvir)] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t4;
               t2t4c_priv[D(r,s,u,w,nocc,nvir)] += e2ovov[De(q,t,p,v,nocc,nvir)] * t4;
               t2t4c_priv[D(q,s,u,w,nocc,nvir)] += e2ovov[De(p,t,r,v,nocc,nvir)] * t4;
               t2t4c_priv[D(p,s,u,w,nocc,nvir)] += e2ovov[De(r,t,q,v,nocc,nvir)] * t4;

               t2t4c_priv[D(r,s,t,w,nocc,nvir)] -= e2ovov[De(p,v,q,u,nocc,nvir)] * t4;
               t2t4c_priv[D(q,s,t,w,nocc,nvir)] -= e2ovov[De(r,v,p,u,nocc,nvir)] * t4;
               t2t4c_priv[D(p,s,t,w,nocc,nvir)] -= e2ovov[De(q,v,r,u,nocc,nvir)] * t4;
               t2t4c_priv[D(r,s,t,w,nocc,nvir)] += e2ovov[De(q,v,p,u,nocc,nvir)] * t4;
               t2t4c_priv[D(q,s,t,w,nocc,nvir)] += e2ovov[De(p,v,r,u,nocc,nvir)] * t4;
               t2t4c_priv[D(p,s,t,w,nocc,nvir)] += e2ovov[De(r,v,q,u,nocc,nvir)] * t4;

               t2t4c_priv[D(r,s,v,w,nocc,nvir)] -= e2ovov[De(p,u,q,t,nocc,nvir)] * t4;
               t2t4c_priv[D(q,s,v,w,nocc,nvir)] -= e2ovov[De(r,u,p,t,nocc,nvir)] * t4;
               t2t4c_priv[D(p,s,v,w,nocc,nvir)] -= e2ovov[De(q,u,r,t,nocc,nvir)] * t4;
               t2t4c_priv[D(r,s,v,w,nocc,nvir)] += e2ovov[De(q,u,p,t,nocc,nvir)] * t4;
               t2t4c_priv[D(q,s,v,w,nocc,nvir)] += e2ovov[De(p,u,r,t,nocc,nvir)] * t4;
               t2t4c_priv[D(p,s,v,w,nocc,nvir)] += e2ovov[De(r,u,q,t,nocc,nvir)] * t4;
           }

           }
           fclose(fp);
           }
       }

#pragma omp critical
       { 
           for (it=0; it< nocc; it++){
               for (jt=0; jt< nocc; jt++){
                   for (at=0; at< nvir; at++){
                       for (bt=0; bt< nvir; bt++){
                           t2t4c[D(it,jt,at,bt,nocc,nvir)] += 0.5*(t2t4c_priv[D(it,jt,at,bt,nocc,nvir)]+t2t4c_priv[D(jt,it,bt,at,nocc,nvir)]);
                       }
                   }
               }
           }   
           free(t2t4c_priv);
       } 
}

    printf ("0SDTQ (Q) =    %f   ( %f )\n", norm, norm-norm0SDT);

}

void t2t4c_dmrg(double *t2t4c, double *t1, double *t2aa, double *t2ab, double *t3aaa, double *t3aab, double *e2ovov, const int nc, const int nocc, const int nvir, const double numzero, const double c0, double norm) 
{
    //double numzero = 1e-7;
    int p, q, r, s, t, u, v, w, it, jt, at, bt;
    double t4, scale;

    double norm0SDT = norm;

    double ****tmp;
    tmp = (double ****)malloc(sizeof(double ***) * nocc); 
    for (it=0; it< nocc; it++){
        tmp[it] = (double ***)malloc(sizeof(double **) * nocc);
        for (jt=0; jt< nocc; jt++){
            tmp[it][jt] = (double **)malloc(sizeof(double *) * nvir);
            for (at=0; at< nvir; at++){
                tmp[it][jt][at] = (double *)malloc(sizeof(double) * nvir);
            }
        }
    }

//    for (p=0; p<nocc; p++) {
//    for (q=0; q<nocc; q++) {
//    for (r=0; r<nvir; r++) {
//    for (s=0; s<nvir; s++) {
//        printf("%d %d %d %d %20.10lf\n",p,q,r,s,e2ovov[De(p,r,q,s,nocc,nvir)]); 
//    }
//    }
//    }
//    }

    FILE *fp;
    char typ[4], line[255];
    fp = fopen("CIcoeff_dmrg.out", "r");
    fscanf(fp, "%s\n", line);
    if (fp) {

//#pragma omp parallel default(none) \
//        shared(count, m, out, v1, v2, a, b)
//{

       while ( !feof(fp) ){
           fscanf(fp, "%c%c%c%c,%s\n", &(typ[0]), &(typ[1]), &(typ[2]), &(typ[3]), line);
           fscanf(fp, "%lf\n", &t4);
           //printf ("typ=%c%c%c%c line=%s\n",typ[0],typ[1],typ[2],typ[3], line);
           if (strncmp(typ, "aabb", 4) == 0 && fabs(t4) > numzero){
               norm += t4*t4;
               sscanf(line,"%d,%d,%d,%d,%d,%d,%d,%d",&p,&q,&t,&u,&r,&s,&v,&w);
               p += nc;
               q += nc;
               r += nc;
               s += nc;
               t += nc;
               u += nc;
               v += nc;
               w += nc;

//               if(!(p == 2 && q == 3 && t == 0 && u == 1 && \
//                    r == 2 && s == 3 && v == 0 && w == 1)) continue;
   
               // interm norm of c4
               t4 = t4 / c0;
               // lsh test 
//               printf("c4 mem %20.10f \n",t4);
   
               // extract t4 
               t4-= t1xt3aabb(p, q, r, s, t, u, v, w, nocc, nvir, t1, t3aab); 
               t4-= t2xt2aabb(p, q, r, s, t, u, v, w, nocc, nvir, t2aa, t2ab); 
               t4-= t1xt1xt2aabb(p, q, r, s, t, u, v, w, nocc, nvir, t1, t2aa, t2ab); 
               t4-= t1xt1xt1xt1aabb(p, q, r, s, t, u, v, w, nocc, nvir, t1);   // may have bug 
   
               // lsh test 
//               printf("t4 mem %20.10f \n",t4);
   
               for (it=0; it< nocc; it++){
               for (jt=0; jt< nocc; jt++){
               for (at=0; at< nvir; at++){
               for (bt=0; bt< nvir; bt++){
                   tmp[it][jt][at][bt] = 0.0;
               }
               }
               }
               }
//               printf("eris_ovov mem %20.10f \n",e2ovov[De(p,t,r,v,nocc,nvir)]);
   
               if (p<r && t<v) tmp[q][s][u][w] += e2ovov[De(p,t,r,v,nocc,nvir)] * t4;
               if (q<r && t<v) tmp[p][s][u][w] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t4;
               if (p<s && t<v) tmp[q][r][u][w] -= e2ovov[De(p,t,s,v,nocc,nvir)] * t4;
               if (q<s && t<v) tmp[p][r][u][w] += e2ovov[De(q,t,s,v,nocc,nvir)] * t4;
               if (p<r && u<v) tmp[q][s][t][w] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t4;
               if (q<r && u<v) tmp[p][s][t][w] += e2ovov[De(q,u,r,v,nocc,nvir)] * t4;
               if (p<s && u<v) tmp[q][r][t][w] += e2ovov[De(p,u,s,v,nocc,nvir)] * t4;
               if (q<s && u<v) tmp[p][r][t][w] -= e2ovov[De(q,u,s,v,nocc,nvir)] * t4;
               if (p<r && t<w) tmp[q][s][u][v] -= e2ovov[De(p,t,r,w,nocc,nvir)] * t4;
               if (q<r && t<w) tmp[p][s][u][v] += e2ovov[De(q,t,r,w,nocc,nvir)] * t4;
               if (p<s && t<w) tmp[q][r][u][v] += e2ovov[De(p,t,s,w,nocc,nvir)] * t4;
               if (q<s && t<w) tmp[p][r][u][v] -= e2ovov[De(q,t,s,w,nocc,nvir)] * t4;
               if (p<r && u<w) tmp[q][s][t][v] += e2ovov[De(p,u,r,w,nocc,nvir)] * t4;
               if (q<r && u<w) tmp[p][s][t][v] -= e2ovov[De(q,u,r,w,nocc,nvir)] * t4;
               if (p<s && u<w) tmp[q][r][t][v] -= e2ovov[De(p,u,s,w,nocc,nvir)] * t4;
               if (q<s && u<w) tmp[p][r][t][v] += e2ovov[De(q,u,s,w,nocc,nvir)] * t4;
               if (p<r && v<t) tmp[q][s][u][w] += e2ovov[De(p,t,r,v,nocc,nvir)] * t4;
               if (q<r && v<t) tmp[p][s][u][w] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t4;
               if (p<s && v<t) tmp[q][r][u][w] -= e2ovov[De(p,t,s,v,nocc,nvir)] * t4;
               if (q<s && v<t) tmp[p][r][u][w] += e2ovov[De(q,t,s,v,nocc,nvir)] * t4;
               if (p<r && v<u) tmp[q][s][t][w] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t4;
               if (q<r && v<u) tmp[p][s][t][w] += e2ovov[De(q,u,r,v,nocc,nvir)] * t4;
               if (p<s && v<u) tmp[q][r][t][w] += e2ovov[De(p,u,s,v,nocc,nvir)] * t4;
               if (q<s && v<u) tmp[p][r][t][w] -= e2ovov[De(q,u,s,v,nocc,nvir)] * t4;
               if (p<r && w<t) tmp[q][s][u][v] -= e2ovov[De(p,t,r,w,nocc,nvir)] * t4;
               if (q<r && w<t) tmp[p][s][u][v] += e2ovov[De(q,t,r,w,nocc,nvir)] * t4;
               if (p<s && w<t) tmp[q][r][u][v] += e2ovov[De(p,t,s,w,nocc,nvir)] * t4;
               if (q<s && w<t) tmp[p][r][u][v] -= e2ovov[De(q,t,s,w,nocc,nvir)] * t4;
               if (p<r && w<u) tmp[q][s][t][v] += e2ovov[De(p,u,r,w,nocc,nvir)] * t4;
               if (q<r && w<u) tmp[p][s][t][v] -= e2ovov[De(q,u,r,w,nocc,nvir)] * t4;
               if (p<s && w<u) tmp[q][r][t][v] -= e2ovov[De(p,u,s,w,nocc,nvir)] * t4;
               if (q<s && w<u) tmp[p][r][t][v] += e2ovov[De(q,u,s,w,nocc,nvir)] * t4;
   
               scale = 0.5;
               if (p==r && t<v) tmp[q][s][u][w] += e2ovov[De(p,t,r,v,nocc,nvir)] * t4 * scale;
               if (q==r && t<v) tmp[p][s][u][w] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t4 * scale;
               if (p==s && t<v) tmp[q][r][u][w] -= e2ovov[De(p,t,s,v,nocc,nvir)] * t4 * scale;
               if (q==s && t<v) tmp[p][r][u][w] += e2ovov[De(q,t,s,v,nocc,nvir)] * t4 * scale;
               if (p==r && u<v) tmp[q][s][t][w] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t4 * scale;
               if (q==r && u<v) tmp[p][s][t][w] += e2ovov[De(q,u,r,v,nocc,nvir)] * t4 * scale;
               if (p==s && u<v) tmp[q][r][t][w] += e2ovov[De(p,u,s,v,nocc,nvir)] * t4 * scale;
               if (q==s && u<v) tmp[p][r][t][w] -= e2ovov[De(q,u,s,v,nocc,nvir)] * t4 * scale;
               if (p==r && t<w) tmp[q][s][u][v] -= e2ovov[De(p,t,r,w,nocc,nvir)] * t4 * scale;
               if (q==r && t<w) tmp[p][s][u][v] += e2ovov[De(q,t,r,w,nocc,nvir)] * t4 * scale;
               if (p==s && t<w) tmp[q][r][u][v] += e2ovov[De(p,t,s,w,nocc,nvir)] * t4 * scale;
               if (q==s && t<w) tmp[p][r][u][v] -= e2ovov[De(q,t,s,w,nocc,nvir)] * t4 * scale;
               if (p==r && u<w) tmp[q][s][t][v] += e2ovov[De(p,u,r,w,nocc,nvir)] * t4 * scale;
               if (q==r && u<w) tmp[p][s][t][v] -= e2ovov[De(q,u,r,w,nocc,nvir)] * t4 * scale;
               if (p==s && u<w) tmp[q][r][t][v] -= e2ovov[De(p,u,s,w,nocc,nvir)] * t4 * scale;
               if (q==s && u<w) tmp[p][r][t][v] += e2ovov[De(q,u,s,w,nocc,nvir)] * t4 * scale;
               if (p==r && v<t) tmp[q][s][u][w] += e2ovov[De(p,t,r,v,nocc,nvir)] * t4 * scale;
               if (q==r && v<t) tmp[p][s][u][w] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t4 * scale;
               if (p==s && v<t) tmp[q][r][u][w] -= e2ovov[De(p,t,s,v,nocc,nvir)] * t4 * scale;
               if (q==s && v<t) tmp[p][r][u][w] += e2ovov[De(q,t,s,v,nocc,nvir)] * t4 * scale;
               if (p==r && v<u) tmp[q][s][t][w] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t4 * scale;
               if (q==r && v<u) tmp[p][s][t][w] += e2ovov[De(q,u,r,v,nocc,nvir)] * t4 * scale;
               if (p==s && v<u) tmp[q][r][t][w] += e2ovov[De(p,u,s,v,nocc,nvir)] * t4 * scale;
               if (q==s && v<u) tmp[p][r][t][w] -= e2ovov[De(q,u,s,v,nocc,nvir)] * t4 * scale;
               if (p==r && w<t) tmp[q][s][u][v] -= e2ovov[De(p,t,r,w,nocc,nvir)] * t4 * scale;
               if (q==r && w<t) tmp[p][s][u][v] += e2ovov[De(q,t,r,w,nocc,nvir)] * t4 * scale;
               if (p==s && w<t) tmp[q][r][u][v] += e2ovov[De(p,t,s,w,nocc,nvir)] * t4 * scale;
               if (q==s && w<t) tmp[p][r][u][v] -= e2ovov[De(q,t,s,w,nocc,nvir)] * t4 * scale;
               if (p==r && w<u) tmp[q][s][t][v] += e2ovov[De(p,u,r,w,nocc,nvir)] * t4 * scale;
               if (q==r && w<u) tmp[p][s][t][v] -= e2ovov[De(q,u,r,w,nocc,nvir)] * t4 * scale;
               if (p==s && w<u) tmp[q][r][t][v] -= e2ovov[De(p,u,s,w,nocc,nvir)] * t4 * scale;
               if (q==s && w<u) tmp[p][r][t][v] += e2ovov[De(q,u,s,w,nocc,nvir)] * t4 * scale;
   
               scale = 0.5;
               if (p<r && t==v) tmp[q][s][u][w] += e2ovov[De(p,t,r,v,nocc,nvir)] * t4 * scale;
               if (q<r && t==v) tmp[p][s][u][w] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t4 * scale;
               if (p<s && t==v) tmp[q][r][u][w] -= e2ovov[De(p,t,s,v,nocc,nvir)] * t4 * scale;
               if (q<s && t==v) tmp[p][r][u][w] += e2ovov[De(q,t,s,v,nocc,nvir)] * t4 * scale;
               if (p<r && u==v) tmp[q][s][t][w] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t4 * scale;
               if (q<r && u==v) tmp[p][s][t][w] += e2ovov[De(q,u,r,v,nocc,nvir)] * t4 * scale;
               if (p<s && u==v) tmp[q][r][t][w] += e2ovov[De(p,u,s,v,nocc,nvir)] * t4 * scale;
               if (q<s && u==v) tmp[p][r][t][w] -= e2ovov[De(q,u,s,v,nocc,nvir)] * t4 * scale;
               if (p<r && t==w) tmp[q][s][u][v] -= e2ovov[De(p,t,r,w,nocc,nvir)] * t4 * scale;
               if (q<r && t==w) tmp[p][s][u][v] += e2ovov[De(q,t,r,w,nocc,nvir)] * t4 * scale;
               if (p<s && t==w) tmp[q][r][u][v] += e2ovov[De(p,t,s,w,nocc,nvir)] * t4 * scale;
               if (q<s && t==w) tmp[p][r][u][v] -= e2ovov[De(q,t,s,w,nocc,nvir)] * t4 * scale;
               if (p<r && u==w) tmp[q][s][t][v] += e2ovov[De(p,u,r,w,nocc,nvir)] * t4 * scale;
               if (q<r && u==w) tmp[p][s][t][v] -= e2ovov[De(q,u,r,w,nocc,nvir)] * t4 * scale;
               if (p<s && u==w) tmp[q][r][t][v] -= e2ovov[De(p,u,s,w,nocc,nvir)] * t4 * scale;
               if (q<s && u==w) tmp[p][r][t][v] += e2ovov[De(q,u,s,w,nocc,nvir)] * t4 * scale;
               if (p<r && v==t) tmp[q][s][u][w] += e2ovov[De(p,t,r,v,nocc,nvir)] * t4 * scale;
               if (q<r && v==t) tmp[p][s][u][w] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t4 * scale;
               if (p<s && v==t) tmp[q][r][u][w] -= e2ovov[De(p,t,s,v,nocc,nvir)] * t4 * scale;
               if (q<s && v==t) tmp[p][r][u][w] += e2ovov[De(q,t,s,v,nocc,nvir)] * t4 * scale;
               if (p<r && v==u) tmp[q][s][t][w] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t4 * scale;
               if (q<r && v==u) tmp[p][s][t][w] += e2ovov[De(q,u,r,v,nocc,nvir)] * t4 * scale;
               if (p<s && v==u) tmp[q][r][t][w] += e2ovov[De(p,u,s,v,nocc,nvir)] * t4 * scale;
               if (q<s && v==u) tmp[p][r][t][w] -= e2ovov[De(q,u,s,v,nocc,nvir)] * t4 * scale;
               if (p<r && w==t) tmp[q][s][u][v] -= e2ovov[De(p,t,r,w,nocc,nvir)] * t4 * scale;
               if (q<r && w==t) tmp[p][s][u][v] += e2ovov[De(q,t,r,w,nocc,nvir)] * t4 * scale;
               if (p<s && w==t) tmp[q][r][u][v] += e2ovov[De(p,t,s,w,nocc,nvir)] * t4 * scale;
               if (q<s && w==t) tmp[p][r][u][v] -= e2ovov[De(q,t,s,w,nocc,nvir)] * t4 * scale;
               if (p<r && w==u) tmp[q][s][t][v] += e2ovov[De(p,u,r,w,nocc,nvir)] * t4 * scale;
               if (q<r && w==u) tmp[p][s][t][v] -= e2ovov[De(q,u,r,w,nocc,nvir)] * t4 * scale;
               if (p<s && w==u) tmp[q][r][t][v] -= e2ovov[De(p,u,s,w,nocc,nvir)] * t4 * scale;
               if (q<s && w==u) tmp[p][r][t][v] += e2ovov[De(q,u,s,w,nocc,nvir)] * t4 * scale;
   
               scale = 0.25;
               if (p==r && t==v) tmp[q][s][u][w] += e2ovov[De(p,t,r,v,nocc,nvir)] * t4 * scale;
               if (q==r && t==v) tmp[p][s][u][w] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t4 * scale;
               if (p==s && t==v) tmp[q][r][u][w] -= e2ovov[De(p,t,s,v,nocc,nvir)] * t4 * scale;
               if (q==s && t==v) tmp[p][r][u][w] += e2ovov[De(q,t,s,v,nocc,nvir)] * t4 * scale;
               if (p==r && u==v) tmp[q][s][t][w] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t4 * scale;
               if (q==r && u==v) tmp[p][s][t][w] += e2ovov[De(q,u,r,v,nocc,nvir)] * t4 * scale;
               if (p==s && u==v) tmp[q][r][t][w] += e2ovov[De(p,u,s,v,nocc,nvir)] * t4 * scale;
               if (q==s && u==v) tmp[p][r][t][w] -= e2ovov[De(q,u,s,v,nocc,nvir)] * t4 * scale;
               if (p==r && t==w) tmp[q][s][u][v] -= e2ovov[De(p,t,r,w,nocc,nvir)] * t4 * scale;
               if (q==r && t==w) tmp[p][s][u][v] += e2ovov[De(q,t,r,w,nocc,nvir)] * t4 * scale;
               if (p==s && t==w) tmp[q][r][u][v] += e2ovov[De(p,t,s,w,nocc,nvir)] * t4 * scale;
               if (q==s && t==w) tmp[p][r][u][v] -= e2ovov[De(q,t,s,w,nocc,nvir)] * t4 * scale;
               if (p==r && u==w) tmp[q][s][t][v] += e2ovov[De(p,u,r,w,nocc,nvir)] * t4 * scale;
               if (q==r && u==w) tmp[p][s][t][v] -= e2ovov[De(q,u,r,w,nocc,nvir)] * t4 * scale;
               if (p==s && u==w) tmp[q][r][t][v] -= e2ovov[De(p,u,s,w,nocc,nvir)] * t4 * scale;
               if (q==s && u==w) tmp[p][r][t][v] += e2ovov[De(q,u,s,w,nocc,nvir)] * t4 * scale;
               if (p==r && v==t) tmp[q][s][u][w] += e2ovov[De(p,t,r,v,nocc,nvir)] * t4 * scale;
               if (q==r && v==t) tmp[p][s][u][w] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t4 * scale;
               if (p==s && v==t) tmp[q][r][u][w] -= e2ovov[De(p,t,s,v,nocc,nvir)] * t4 * scale;
               if (q==s && v==t) tmp[p][r][u][w] += e2ovov[De(q,t,s,v,nocc,nvir)] * t4 * scale;
               if (p==r && v==u) tmp[q][s][t][w] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t4 * scale;
               if (q==r && v==u) tmp[p][s][t][w] += e2ovov[De(q,u,r,v,nocc,nvir)] * t4 * scale;
               if (p==s && v==u) tmp[q][r][t][w] += e2ovov[De(p,u,s,v,nocc,nvir)] * t4 * scale;
               if (q==s && v==u) tmp[p][r][t][w] -= e2ovov[De(q,u,s,v,nocc,nvir)] * t4 * scale;
               if (p==r && w==t) tmp[q][s][u][v] -= e2ovov[De(p,t,r,w,nocc,nvir)] * t4 * scale;
               if (q==r && w==t) tmp[p][s][u][v] += e2ovov[De(q,t,r,w,nocc,nvir)] * t4 * scale;
               if (p==s && w==t) tmp[q][r][u][v] += e2ovov[De(p,t,s,w,nocc,nvir)] * t4 * scale;
               if (q==s && w==t) tmp[p][r][u][v] -= e2ovov[De(q,t,s,w,nocc,nvir)] * t4 * scale;
               if (p==r && w==u) tmp[q][s][t][v] += e2ovov[De(p,u,r,w,nocc,nvir)] * t4 * scale;
               if (q==r && w==u) tmp[p][s][t][v] -= e2ovov[De(q,u,r,w,nocc,nvir)] * t4 * scale;
               if (p==s && w==u) tmp[q][r][t][v] -= e2ovov[De(p,u,s,w,nocc,nvir)] * t4 * scale;
               if (q==s && w==u) tmp[p][r][t][v] += e2ovov[De(q,u,s,w,nocc,nvir)] * t4 * scale;
   
               for (it = 0; it < nocc; it++)
               for (jt = 0; jt < nocc; jt++)
               for (at = 0; at < nvir; at++)
               for (bt = 0; bt < nvir; bt++)
                   t2t4c[D(it,jt,at,bt,nocc,nvir)] += tmp[it][jt][at][bt] + tmp[jt][it][bt][at];

           }
           else if (strncmp(typ, "aaab", 4) == 0 && fabs(t4) > numzero){
               norm += 2.0*t4*t4; 
               //lsh test
               //printf ("typ=%c%c%c%c line=%s c4=%lf\n",typ[0],typ[1],typ[2],typ[3], line, t4);

               sscanf(line,"%d,%d,%d,%d,%d,%d,%d,%d",&p,&q,&r,&t,&u,&v,&s,&w);
               p += nc;
               q += nc;
               r += nc;
               s += nc;
               t += nc;
               u += nc;
               v += nc;
               w += nc;

               //printf ("p=%d, q=%d, r=%d, t=%d, u=%d, v=%d, s=%d, w=%d",p,q,r,t,u,v,s,w);
               // interm norm of c4
               t4 = t4 / c0;
   
               // extract t4 
               t4-= t1xt3aaab (p, q, r, s, t, u, v, w, nocc, nvir, t1, t3aaa, t3aab);   // may have 1e-5 bug 
               t4-= t2xt2aaab (p, q, r, s, t, u, v, w, nocc, nvir, t2aa, t2ab);         // may have 1e-3 bug 
               t4-= t1xt1xt2aaab (p, q, r, s, t, u, v, w, nocc, nvir, t1, t2aa, t2ab);  // may have 1e-5 bug 
               t4-= t1xt1xt1xt1aaab (p, q, r, s, t, u, v, w, nocc, nvir, t1);           // may have 1e-6 bug
   
               for (it=0; it< nocc; it++){
               for (jt=0; jt< nocc; jt++){
               for (at=0; at< nvir; at++){
               for (bt=0; bt< nvir; bt++){
                   tmp[it][jt][at][bt] = 0.0;
               }
               }
               }
               }
   
               tmp[r][s][v][w] += (e2ovov[De(p,t,q,u,nocc,nvir)]-e2ovov[De(p,u,q,t,nocc,nvir)]) * t4; 
               tmp[q][s][v][w] -= (e2ovov[De(p,t,r,u,nocc,nvir)]-e2ovov[De(p,u,r,t,nocc,nvir)]) * t4; 
               tmp[p][s][v][w] += (e2ovov[De(q,t,s,u,nocc,nvir)]-e2ovov[De(q,u,s,t,nocc,nvir)]) * t4; 
               tmp[r][s][u][w] -= (e2ovov[De(p,t,q,v,nocc,nvir)]-e2ovov[De(p,v,q,t,nocc,nvir)]) * t4; 
               tmp[q][s][u][w] += (e2ovov[De(p,t,r,v,nocc,nvir)]-e2ovov[De(p,v,r,t,nocc,nvir)]) * t4; 
               tmp[p][s][u][w] -= (e2ovov[De(q,t,s,v,nocc,nvir)]-e2ovov[De(q,v,s,t,nocc,nvir)]) * t4; 
               tmp[r][s][t][w] += (e2ovov[De(p,u,q,v,nocc,nvir)]-e2ovov[De(p,v,q,u,nocc,nvir)]) * t4; 
               tmp[q][s][t][w] -= (e2ovov[De(p,u,r,v,nocc,nvir)]-e2ovov[De(p,v,r,u,nocc,nvir)]) * t4; 
               tmp[p][s][t][w] += (e2ovov[De(q,u,s,v,nocc,nvir)]-e2ovov[De(q,v,s,u,nocc,nvir)]) * t4; 
   
               for (it = 0; it < nocc; it++)
               for (jt = 0; jt < nocc; jt++)
               for (at = 0; at < nvir; at++)
               for (bt = 0; bt < nvir; bt++)
                   t2t4c[D(it,jt,at,bt,nocc,nvir)] += tmp[it][jt][at][bt] + tmp[jt][it][bt][at];

           }
       }
//}

       fclose(fp);
    }
    else
    {
       // error message
    }


    for (it=0; it< nocc; it++){
        for (jt=0; jt< nocc; jt++){
            for (at=0; at< nvir; at++){
                free(tmp[it][jt][at]);
            }
            free(tmp[it][jt]);
        }
        free(tmp[it]);
    }   
    free(tmp);

    printf ("0SDTQ (Q) =    %f   ( %f )\n", norm, norm-norm0SDT);
}
//
//void c4_to_t4_test(double *t4aaab, double *t4aabb, double *c4aaab, double *c4aabb, double *t1, double *t2aa, double *t2ab, double *t3aaa, double *t3aab, int nocc, int nvir, double numzero) 
//{
//    int i, j, k, l, a, b, c, d, m_ijab;
//    int ijkabc, ld, ijkabcld_c;
//    int ijklabcd_t11, ijklabcd_t21, ijklabcd_t31, ijklabcd_t41, ijklabcd_t51, ijklabcd_t61;
//    int ijklabcd_t12, ijklabcd_t22, ijklabcd_t32, ijklabcd_t42, ijklabcd_t52, ijklabcd_t62;
//    int ijklabcd_t13, ijklabcd_t23, ijklabcd_t33, ijklabcd_t43, ijklabcd_t53, ijklabcd_t63;
//    int ijklabcd_t14, ijklabcd_t24, ijklabcd_t34, ijklabcd_t44, ijklabcd_t54, ijklabcd_t64;
//    int ijklabcd_t15, ijklabcd_t25, ijklabcd_t35, ijklabcd_t45, ijklabcd_t55, ijklabcd_t65;
//    int ijklabcd_t16, ijklabcd_t26, ijklabcd_t36, ijklabcd_t46, ijklabcd_t56, ijklabcd_t66;
//    int ijab, klcd, ijabklcd_c;
//
//    double tmp, tmp2;
//
//    // t4aaab
//    ijkabc = -1;
//    for (c = 2; c < nvir; c++) {
//    for (b = 1; b < c; b++) {
//    for (a = 0; a < b; a++) {
//    for (k = nocc-1; k > 1; k--) {
//    for (j = k-1; j > 0; j--) {
//    for (i = j-1; i > -1; i--) {
//        ijkabc += 1;
//        ld = -1;
//        for (d = 0; d < nvir; d++) {
//        for (l = nocc-1; l > -1; l--) {
//            ld += 1;
//            ijkabcld_c = ijkabc * nocc*nvir + ld;
//            tmp = c4aaab[ijkabcld_c]; 
//
////            if(fabs(tmp)-fabs(tmp2) > numzero) 
//            if(fabs(tmp) > numzero) 
//            {
//                tmp2 = t1xt3aaab (i, j, k, l, a, b, c, d, nocc, nvir, t1, t3aaa, t3aab);   // may have 1e-5 bug 
//                tmp2+= t2xt2aaab (i, j, k, l, a, b, c, d, nocc, nvir, t2aa, t2ab);         // may have 1e-3 bug 
//                tmp2+= t1xt1xt2aaab (i, j, k, l, a, b, c, d, nocc, nvir, t1, t2aa, t2ab);  // may have 1e-5 bug 
//                tmp2+= t1xt1xt1xt1aaab (i, j, k, l, a, b, c, d, nocc, nvir, t1);           // may have 1e-6 bug
//
//                tmp = tmp2; 
//                ijklabcd_t11 = Q(i, j, k, l, a, b, c, d, nocc, nvir);
//                ijklabcd_t12 = Q(i, j, k, l, b, c, a, d, nocc, nvir);
//                ijklabcd_t13 = Q(i, j, k, l, c, a, b, d, nocc, nvir);
//                ijklabcd_t14 = Q(i, j, k, l, a, c, b, d, nocc, nvir);
//                ijklabcd_t15 = Q(i, j, k, l, b, a, c, d, nocc, nvir);
//                ijklabcd_t16 = Q(i, j, k, l, c, b, a, d, nocc, nvir);
//        
//                t4aaab[ijklabcd_t11] =  tmp;
//                t4aaab[ijklabcd_t12] =  tmp;
//                t4aaab[ijklabcd_t13] =  tmp;
//                t4aaab[ijklabcd_t14] = -tmp;
//                t4aaab[ijklabcd_t15] = -tmp;
//                t4aaab[ijklabcd_t16] = -tmp;
//        
//                ijklabcd_t21 = Q(j, k, i, l, a, b, c, d, nocc, nvir);
//                ijklabcd_t22 = Q(j, k, i, l, b, c, a, d, nocc, nvir);
//                ijklabcd_t23 = Q(j, k, i, l, c, a, b, d, nocc, nvir);
//                ijklabcd_t24 = Q(j, k, i, l, a, c, b, d, nocc, nvir);
//                ijklabcd_t25 = Q(j, k, i, l, b, a, c, d, nocc, nvir);
//                ijklabcd_t26 = Q(j, k, i, l, c, b, a, d, nocc, nvir);
//        
//                t4aaab[ijklabcd_t21] =  tmp;
//                t4aaab[ijklabcd_t22] =  tmp;
//                t4aaab[ijklabcd_t23] =  tmp;
//                t4aaab[ijklabcd_t24] = -tmp;
//                t4aaab[ijklabcd_t25] = -tmp;
//                t4aaab[ijklabcd_t26] = -tmp;
//        
//                ijklabcd_t31 = Q(k, i, j, l, a, b, c, d, nocc, nvir);
//                ijklabcd_t32 = Q(k, i, j, l, b, c, a, d, nocc, nvir);
//                ijklabcd_t33 = Q(k, i, j, l, c, a, b, d, nocc, nvir);
//                ijklabcd_t34 = Q(k, i, j, l, a, c, b, d, nocc, nvir);
//                ijklabcd_t35 = Q(k, i, j, l, b, a, c, d, nocc, nvir);
//                ijklabcd_t36 = Q(k, i, j, l, c, b, a, d, nocc, nvir);
//        
//                t4aaab[ijklabcd_t31] =  tmp;
//                t4aaab[ijklabcd_t32] =  tmp;
//                t4aaab[ijklabcd_t33] =  tmp;
//                t4aaab[ijklabcd_t34] = -tmp;
//                t4aaab[ijklabcd_t35] = -tmp;
//                t4aaab[ijklabcd_t36] = -tmp;
//        
//                ijklabcd_t41 = Q(i, k, j, l, a, b, c, d, nocc, nvir);
//                ijklabcd_t42 = Q(i, k, j, l, b, c, a, d, nocc, nvir);
//                ijklabcd_t43 = Q(i, k, j, l, c, a, b, d, nocc, nvir);
//                ijklabcd_t44 = Q(i, k, j, l, a, c, b, d, nocc, nvir);
//                ijklabcd_t45 = Q(i, k, j, l, b, a, c, d, nocc, nvir);
//                ijklabcd_t46 = Q(i, k, j, l, c, b, a, d, nocc, nvir);
//        
//                t4aaab[ijklabcd_t41] = -tmp;
//                t4aaab[ijklabcd_t42] = -tmp;
//                t4aaab[ijklabcd_t43] = -tmp;
//                t4aaab[ijklabcd_t44] =  tmp;
//                t4aaab[ijklabcd_t45] =  tmp;
//                t4aaab[ijklabcd_t46] =  tmp;
//        
//                ijklabcd_t51 = Q(j, i, k, l, a, b, c, d, nocc, nvir);
//                ijklabcd_t52 = Q(j, i, k, l, b, c, a, d, nocc, nvir);
//                ijklabcd_t53 = Q(j, i, k, l, c, a, b, d, nocc, nvir);
//                ijklabcd_t54 = Q(j, i, k, l, a, c, b, d, nocc, nvir);
//                ijklabcd_t55 = Q(j, i, k, l, b, a, c, d, nocc, nvir);
//                ijklabcd_t56 = Q(j, i, k, l, c, b, a, d, nocc, nvir);
//        
//                t4aaab[ijklabcd_t51] = -tmp;
//                t4aaab[ijklabcd_t52] = -tmp;
//                t4aaab[ijklabcd_t53] = -tmp;
//                t4aaab[ijklabcd_t54] =  tmp;
//                t4aaab[ijklabcd_t55] =  tmp;
//                t4aaab[ijklabcd_t56] =  tmp;
//        
//                ijklabcd_t61 = Q(k, j, i, l, a, b, c, d, nocc, nvir);
//                ijklabcd_t62 = Q(k, j, i, l, b, c, a, d, nocc, nvir);
//                ijklabcd_t63 = Q(k, j, i, l, c, a, b, d, nocc, nvir);
//                ijklabcd_t64 = Q(k, j, i, l, a, c, b, d, nocc, nvir);
//                ijklabcd_t65 = Q(k, j, i, l, b, a, c, d, nocc, nvir);
//                ijklabcd_t66 = Q(k, j, i, l, c, b, a, d, nocc, nvir);
//        
//                t4aaab[ijklabcd_t61] = -tmp;
//                t4aaab[ijklabcd_t62] = -tmp;
//                t4aaab[ijklabcd_t63] = -tmp;
//                t4aaab[ijklabcd_t64] =  tmp;
//                t4aaab[ijklabcd_t65] =  tmp;
//                t4aaab[ijklabcd_t66] =  tmp;
//            }
//        }
//        }
//    }
//    }
//    }
//    }
//    }
//    }
//
//    // TODO lsh: reduce symmetry of t4, t3
//
//    // t4aabb
//    m_ijab = nocc*(nocc-1)/2 * nvir*(nvir-1)/2;
//    ijab = -1;
//    for (b = 1; b < nvir; b++) {
//    for (a = 0; a < b; a++) {
//    for (j = nocc-1; j > 0; j--) {
//    for (i = j-1; i > -1; i--) {
//        ijab += 1;
//        klcd  =-1;
//        for (d = 1; d < nvir; d++) {
//        for (c = 0; c < d; c++) {
//        for (l = nocc-1; l > 0; l--) {
//        for (k = l-1; k > -1; k--) {
//            klcd += 1;
//            ijabklcd_c = ijab * m_ijab + klcd;
//            tmp = c4aabb[ijabklcd_c]; 
//
////            if(fabs(tmp)-fabs(tmp2) > numzero) 
//            if(fabs(tmp) > numzero) 
//            {
//                tmp2 = t1xt3aabb(i, j, k, l, a, b, c, d, nocc, nvir, t1, t3aab); 
//                tmp2+= t2xt2aabb(i, j, k, l, a, b, c, d, nocc, nvir, t2aa, t2ab); 
//                tmp2+= t1xt1xt2aabb(i, j, k, l, a, b, c, d, nocc, nvir, t1, t2aa, t2ab); 
//                tmp2+= t1xt1xt1xt1aabb(i, j, k, l, a, b, c, d, nocc, nvir, t1);   // may have bug 
//
//                tmp = tmp2; 
//                ijklabcd_t11 = Q(i, j, k, l, a, b, c, d, nocc, nvir);
//                ijklabcd_t12 = Q(j, i, k, l, b, a, c, d, nocc, nvir);
//                ijklabcd_t13 = Q(i, j, k, l, b, a, c, d, nocc, nvir);
//                ijklabcd_t14 = Q(j, i, k, l, a, b, c, d, nocc, nvir);
//        
//                t4aabb[ijklabcd_t11] =  tmp;
//                t4aabb[ijklabcd_t12] =  tmp;
//                t4aabb[ijklabcd_t13] = -tmp;
//                t4aabb[ijklabcd_t14] = -tmp;  
//    
//                ijklabcd_t21 = Q(i, j, l, k, a, b, d, c, nocc, nvir);
//                ijklabcd_t22 = Q(j, i, l, k, b, a, d, c, nocc, nvir);
//                ijklabcd_t23 = Q(i, j, l, k, b, a, d, c, nocc, nvir);
//                ijklabcd_t24 = Q(j, i, l, k, a, b, d, c, nocc, nvir);
//        
//                t4aabb[ijklabcd_t21] =  tmp;
//                t4aabb[ijklabcd_t22] =  tmp;
//                t4aabb[ijklabcd_t23] = -tmp;
//                t4aabb[ijklabcd_t24] = -tmp;  
//    
//                ijklabcd_t31 = Q(i, j, k, l, a, b, d, c, nocc, nvir);
//                ijklabcd_t32 = Q(j, i, k, l, b, a, d, c, nocc, nvir);
//                ijklabcd_t33 = Q(i, j, k, l, b, a, d, c, nocc, nvir);
//                ijklabcd_t34 = Q(j, i, k, l, a, b, d, c, nocc, nvir);
//        
//                t4aabb[ijklabcd_t31] = -tmp;
//                t4aabb[ijklabcd_t32] = -tmp;
//                t4aabb[ijklabcd_t33] =  tmp;
//                t4aabb[ijklabcd_t34] =  tmp;  
//    
//                ijklabcd_t41 = Q(i, j, l, k, a, b, c, d, nocc, nvir);
//                ijklabcd_t42 = Q(j, i, l, k, b, a, c, d, nocc, nvir);
//                ijklabcd_t43 = Q(i, j, l, k, b, a, c, d, nocc, nvir);
//                ijklabcd_t44 = Q(j, i, l, k, a, b, c, d, nocc, nvir);
//        
//                t4aabb[ijklabcd_t41] = -tmp;
//                t4aabb[ijklabcd_t42] = -tmp;
//                t4aabb[ijklabcd_t43] =  tmp;
//                t4aabb[ijklabcd_t44] =  tmp;  
//            }
//        }
//        }
//        }
//        }
//    }
//    }
//    }
//    }
//
//}

void t1t3c_shci(double *t1t3c, double *t1, double *t2aa, double *t2ab, double *e2ovov, const int nc, const int nocc, const int nvir, const double numzero, const double c0, double norm) 
{
    int p, q, r, t, u, v, itmp, it, at;
    double t3, parity, scale;
    uint8_t Refdet[nocc+nvir], det_str[nocc+nvir];
    for (itmp = 0; itmp < nocc+nvir; itmp++){
        if (itmp<nocc) Refdet[itmp] = 3;  
        else           Refdet[itmp] = 0;
    }

    FILE *fp;
    char line[255], typ[4], tmpc[255];
    char *ptr;
    fp = fopen("CIcoeff_shci.out", "r");
    fgets(line, 255, fp);

    double norm0SD = norm;
    if (fp) {
       while ( !feof(fp) ){
           fgets(line, 255, fp);
           ptr = strtok(line, ",");

           it=0;
           while(ptr != NULL){
               if(it==0){
                   strcpy(typ, ptr);
               }
               if (it>0 && strlen(typ)==3 && strncmp(typ, "aaa", 3)==0){
                   strcpy(tmpc, ptr);
                   if(it==1) p = atoi(tmpc);
                   if(it==2) q = atoi(tmpc);
                   if(it==3) r = atoi(tmpc);
                   if(it==4) t = atoi(tmpc);
                   if(it==5) u = atoi(tmpc);
                   if(it==6) v = atoi(tmpc);
                   if(it==7) t3= atof(tmpc);
               }
               if (it>0 && strlen(typ)==3 && strncmp(typ, "aab", 3)==0){
                   strcpy(tmpc, ptr);
                   if(it==1) p = atoi(tmpc);
                   if(it==2) q = atoi(tmpc);
                   if(it==3) t = atoi(tmpc);
                   if(it==4) u = atoi(tmpc);
                   if(it==5) r = atoi(tmpc);
                   if(it==6) v = atoi(tmpc);
                   if(it==7) t3= atof(tmpc);
               }
               if (it>7) break;
               ptr = strtok(NULL, ",");
               it++;
           }

           if (strlen(typ)==3 && strncmp(typ, "aaa", 3) == 0 && fabs(t3) > numzero){
//               printf("'%s, %d, %d, %d, %d, %d, %d, %15.8f'\n", typ, p,q,r,t,u,v,t3);
               norm += 2.0*t3*t3;
               p += nc;
               q += nc;
               r += nc;
               t += - nocc + nc;
               u += - nocc + nc;
               v += - nocc + nc;

//               //lsh test
//               if(!(p == 1 && q == 2 && r == 3 && \
//                    t == 0 && u == 2 && v == 4)) continue;
//               printf("c3 in OTF: %15.8f\n",t3);

               for (itmp = 0; itmp < nocc+nvir; itmp++)
                   det_str[itmp] = Refdet[itmp];  
   
               det_str[p] = 2;
               det_str[q] = 2;
               det_str[r] = 2;
               det_str[t+nocc] = 1;
               det_str[u+nocc] = 1;
               det_str[v+nocc] = 1;
   
               //parity  = parity_ab_str(det_str, nocc+nvir);
               parity = parity_ci_to_cc(p+q+r, 3, nocc);
   
               // interm norm of c3
               t3 = parity * t3 / c0;
               // extract t3
               t3-= t1xt2aaa (p, q, r, t, u, v, nocc, nvir, t1, t2aa); 
               t3-= t1xt1xt1aaa (p, q, r, t, u, v, nocc, nvir, t1); 

//               printf("t3 in OTF: %15.8f\n",t3);

               scale = 1.0;
               t1t3c[S(p,t,nvir)] += (e2ovov[De(q,u,r,v,nocc,nvir)]-e2ovov[De(r,u,q,v,nocc,nvir)]) * t3 * scale; 
               t1t3c[S(q,t,nvir)] -= (e2ovov[De(p,u,r,v,nocc,nvir)]-e2ovov[De(r,u,p,v,nocc,nvir)]) * t3 * scale; 
               t1t3c[S(r,t,nvir)] += (e2ovov[De(p,u,q,v,nocc,nvir)]-e2ovov[De(q,u,p,v,nocc,nvir)]) * t3 * scale; 
               t1t3c[S(p,u,nvir)] -= (e2ovov[De(q,t,r,v,nocc,nvir)]-e2ovov[De(r,t,q,v,nocc,nvir)]) * t3 * scale; 
               t1t3c[S(q,u,nvir)] += (e2ovov[De(p,t,r,v,nocc,nvir)]-e2ovov[De(r,t,p,v,nocc,nvir)]) * t3 * scale; 
               t1t3c[S(r,u,nvir)] -= (e2ovov[De(p,t,q,v,nocc,nvir)]-e2ovov[De(q,t,p,v,nocc,nvir)]) * t3 * scale; 
               t1t3c[S(p,v,nvir)] += (e2ovov[De(q,t,r,u,nocc,nvir)]-e2ovov[De(r,t,q,u,nocc,nvir)]) * t3 * scale; 
               t1t3c[S(q,v,nvir)] -= (e2ovov[De(p,t,r,u,nocc,nvir)]-e2ovov[De(r,t,p,u,nocc,nvir)]) * t3 * scale; 
               t1t3c[S(r,v,nvir)] += (e2ovov[De(p,t,q,u,nocc,nvir)]-e2ovov[De(q,t,p,u,nocc,nvir)]) * t3 * scale; 

           }
           else if (strlen(typ)==3 && strncmp(typ, "aab", 3) == 0 && fabs(t3) > numzero){
               //printf("'%s, %d, %d, %d, %d, %d, %d, %15.8f'\n", typ, p,q,t,u,r,v,t3);
               norm += 2.0*t3*t3; 

               p += nc;
               q += nc;
               r += nc;
               t += - nocc + nc;
               u += - nocc + nc;
               v += - nocc + nc;

               //lsh test
               //if(!(p == 2 && q == 3 && r == 3 && \
               //     t == 0 && u == 1 && v == 1)) continue;
               //printf("c3 in OTF: %15.8f\n",t3);

               for (itmp = 0; itmp < nocc+nvir; itmp++)
                   det_str[itmp] = Refdet[itmp];  
               det_str[p] = 2;
               det_str[q] = 2;
               det_str[t+nocc] = 1;
               det_str[u+nocc] = 1;
   
               if (p != r && q != r) det_str[r] = 1;
               else  det_str[r] = 0;
               if (t != v && u != v) det_str[v+nocc] = 2;
               else  det_str[v+nocc] = 3;
               //parity  = parity_ab_str(det_str, nocc+nvir);
               parity = parity_ci_to_cc(p+q, 2, nocc);
               parity *= parity_ci_to_cc(r, 1, nocc);
   
               // interm norm of c3
               t3 = parity * t3 / c0;
   
               // extract t3 
               t3-= t1xt2aab(p, q, r, t, u, v, nocc, nvir, t1, t2aa, t2ab); 
               t3-= t1xt1xt1aab(p, q, r, t, u, v, nocc, nvir, t1); 
               //printf("t3 in OTF: %15.8f\n",t3);

               t1t3c[S(r,v,nvir)] += (e2ovov[De(p,t,q,u,nocc,nvir)] - e2ovov[De(p,u,q,t,nocc,nvir)]) * t3; 

               scale = 1.0;
               if (r<q && v<u) t1t3c[S(p,t,nvir)] += e2ovov[De(r,v,q,u,nocc,nvir)] * t3 * scale; 
               if (r<p && v<u) t1t3c[S(q,t,nvir)] -= e2ovov[De(r,v,p,u,nocc,nvir)] * t3 * scale; 
               if (r<q && v<t) t1t3c[S(p,u,nvir)] -= e2ovov[De(r,v,q,t,nocc,nvir)] * t3 * scale; 
               if (r<p && v<t) t1t3c[S(q,u,nvir)] += e2ovov[De(r,v,p,t,nocc,nvir)] * t3 * scale; 

               if (q<r && u<v) t1t3c[S(p,t,nvir)] += e2ovov[De(q,u,r,v,nocc,nvir)] * t3 * scale; 
               if (p<r && u<v) t1t3c[S(q,t,nvir)] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t3 * scale; 
               if (q<r && t<v) t1t3c[S(p,u,nvir)] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t3 * scale; 
               if (p<r && t<v) t1t3c[S(q,u,nvir)] += e2ovov[De(p,t,r,v,nocc,nvir)] * t3 * scale; 

               scale = 0.5;
               if (r==q && v<u) t1t3c[S(p,t,nvir)] += e2ovov[De(r,v,q,u,nocc,nvir)] * t3 * scale; 
               if (r==p && v<u) t1t3c[S(q,t,nvir)] -= e2ovov[De(r,v,p,u,nocc,nvir)] * t3 * scale; 
               if (r==q && v<t) t1t3c[S(p,u,nvir)] -= e2ovov[De(r,v,q,t,nocc,nvir)] * t3 * scale; 
               if (r==p && v<t) t1t3c[S(q,u,nvir)] += e2ovov[De(r,v,p,t,nocc,nvir)] * t3 * scale; 

               if (q==r && u<v) t1t3c[S(p,t,nvir)] += e2ovov[De(q,u,r,v,nocc,nvir)] * t3 * scale; 
               if (p==r && u<v) t1t3c[S(q,t,nvir)] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t3 * scale; 
               if (q==r && t<v) t1t3c[S(p,u,nvir)] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t3 * scale; 
               if (p==r && t<v) t1t3c[S(q,u,nvir)] += e2ovov[De(p,t,r,v,nocc,nvir)] * t3 * scale; 

               if (r<q && v==u) t1t3c[S(p,t,nvir)] += e2ovov[De(r,v,q,u,nocc,nvir)] * t3 * scale; 
               if (r<p && v==u) t1t3c[S(q,t,nvir)] -= e2ovov[De(r,v,p,u,nocc,nvir)] * t3 * scale; 
               if (r<q && v==t) t1t3c[S(p,u,nvir)] -= e2ovov[De(r,v,q,t,nocc,nvir)] * t3 * scale; 
               if (r<p && v==t) t1t3c[S(q,u,nvir)] += e2ovov[De(r,v,p,t,nocc,nvir)] * t3 * scale; 

               if (q<r && u==v) t1t3c[S(p,t,nvir)] += e2ovov[De(q,u,r,v,nocc,nvir)] * t3 * scale; 
               if (p<r && u==v) t1t3c[S(q,t,nvir)] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t3 * scale; 
               if (q<r && t==v) t1t3c[S(p,u,nvir)] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t3 * scale; 
               if (p<r && t==v) t1t3c[S(q,u,nvir)] += e2ovov[De(p,t,r,v,nocc,nvir)] * t3 * scale; 

               scale = 0.25;
               if (r==q && v==u) t1t3c[S(p,t,nvir)] += e2ovov[De(r,v,q,u,nocc,nvir)] * t3 * scale; 
               if (r==p && v==u) t1t3c[S(q,t,nvir)] -= e2ovov[De(r,v,p,u,nocc,nvir)] * t3 * scale; 
               if (r==q && v==t) t1t3c[S(p,u,nvir)] -= e2ovov[De(r,v,q,t,nocc,nvir)] * t3 * scale; 
               if (r==p && v==t) t1t3c[S(q,u,nvir)] += e2ovov[De(r,v,p,t,nocc,nvir)] * t3 * scale; 

               if (q==r && u==v) t1t3c[S(p,t,nvir)] += e2ovov[De(q,u,r,v,nocc,nvir)] * t3 * scale; 
               if (p==r && u==v) t1t3c[S(q,t,nvir)] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t3 * scale; 
               if (q==r && t==v) t1t3c[S(p,u,nvir)] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t3 * scale; 
               if (p==r && t==v) t1t3c[S(q,u,nvir)] += e2ovov[De(p,t,r,v,nocc,nvir)] * t3 * scale; 

               scale = 1.0;
               if (r<q && u<v) t1t3c[S(p,t,nvir)] += e2ovov[De(r,v,q,u,nocc,nvir)] * t3 * scale; 
               if (r<p && u<v) t1t3c[S(q,t,nvir)] -= e2ovov[De(r,v,p,u,nocc,nvir)] * t3 * scale; 
               if (r<q && t<v) t1t3c[S(p,u,nvir)] -= e2ovov[De(r,v,q,t,nocc,nvir)] * t3 * scale; 
               if (r<p && t<v) t1t3c[S(q,u,nvir)] += e2ovov[De(r,v,p,t,nocc,nvir)] * t3 * scale; 

               if (q<r && v<u) t1t3c[S(p,t,nvir)] += e2ovov[De(q,u,r,v,nocc,nvir)] * t3 * scale; 
               if (p<r && v<u) t1t3c[S(q,t,nvir)] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t3 * scale; 
               if (q<r && v<t) t1t3c[S(p,u,nvir)] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t3 * scale; 
               if (p<r && v<t) t1t3c[S(q,u,nvir)] += e2ovov[De(p,t,r,v,nocc,nvir)] * t3 * scale; 

               scale = 0.5;
               if (r==q && u<v) t1t3c[S(p,t,nvir)] += e2ovov[De(r,v,q,u,nocc,nvir)] * t3 * scale; 
               if (r==p && u<v) t1t3c[S(q,t,nvir)] -= e2ovov[De(r,v,p,u,nocc,nvir)] * t3 * scale; 
               if (r==q && t<v) t1t3c[S(p,u,nvir)] -= e2ovov[De(r,v,q,t,nocc,nvir)] * t3 * scale; 
               if (r==p && t<v) t1t3c[S(q,u,nvir)] += e2ovov[De(r,v,p,t,nocc,nvir)] * t3 * scale; 

               if (q==r && v<u) t1t3c[S(p,t,nvir)] += e2ovov[De(q,u,r,v,nocc,nvir)] * t3 * scale; 
               if (p==r && v<u) t1t3c[S(q,t,nvir)] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t3 * scale; 
               if (q==r && v<t) t1t3c[S(p,u,nvir)] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t3 * scale; 
               if (p==r && v<t) t1t3c[S(q,u,nvir)] += e2ovov[De(p,t,r,v,nocc,nvir)] * t3 * scale; 

               if (r<q && u==v) t1t3c[S(p,t,nvir)] += e2ovov[De(r,v,q,u,nocc,nvir)] * t3 * scale; 
               if (r<p && u==v) t1t3c[S(q,t,nvir)] -= e2ovov[De(r,v,p,u,nocc,nvir)] * t3 * scale; 
               if (r<q && t==v) t1t3c[S(p,u,nvir)] -= e2ovov[De(r,v,q,t,nocc,nvir)] * t3 * scale; 
               if (r<p && t==v) t1t3c[S(q,u,nvir)] += e2ovov[De(r,v,p,t,nocc,nvir)] * t3 * scale; 

               if (q<r && v==u) t1t3c[S(p,t,nvir)] += e2ovov[De(q,u,r,v,nocc,nvir)] * t3 * scale; 
               if (p<r && v==u) t1t3c[S(q,t,nvir)] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t3 * scale; 
               if (q<r && v==t) t1t3c[S(p,u,nvir)] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t3 * scale; 
               if (p<r && v==t) t1t3c[S(q,u,nvir)] += e2ovov[De(p,t,r,v,nocc,nvir)] * t3 * scale; 

               scale = 0.25;
               if (r==q && u==v) t1t3c[S(p,t,nvir)] += e2ovov[De(r,v,q,u,nocc,nvir)] * t3 * scale; 
               if (r==p && u==v) t1t3c[S(q,t,nvir)] -= e2ovov[De(r,v,p,u,nocc,nvir)] * t3 * scale; 
               if (r==q && t==v) t1t3c[S(p,u,nvir)] -= e2ovov[De(r,v,q,t,nocc,nvir)] * t3 * scale; 
               if (r==p && t==v) t1t3c[S(q,u,nvir)] += e2ovov[De(r,v,p,t,nocc,nvir)] * t3 * scale; 

               if (q==r && v==u) t1t3c[S(p,t,nvir)] += e2ovov[De(q,u,r,v,nocc,nvir)] * t3 * scale; 
               if (p==r && v==u) t1t3c[S(q,t,nvir)] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t3 * scale; 
               if (q==r && v==t) t1t3c[S(p,u,nvir)] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t3 * scale; 
               if (p==r && v==t) t1t3c[S(q,u,nvir)] += e2ovov[De(p,t,r,v,nocc,nvir)] * t3 * scale; 

//               scale = 1.0;
//               if (r==q && u<v) t1t3c[S(p,t,nvir)] += e2ovov[De(r,v,q,u,nocc,nvir)] * t3 * scale; 
//               if (r==p && u<v) t1t3c[S(q,t,nvir)] -= e2ovov[De(r,v,p,u,nocc,nvir)] * t3 * scale; 
//               if (r==q && t<v) t1t3c[S(p,u,nvir)] -= e2ovov[De(r,v,q,t,nocc,nvir)] * t3 * scale; 
//               if (r==p && t<v) t1t3c[S(q,u,nvir)] += e2ovov[De(r,v,p,t,nocc,nvir)] * t3 * scale; 
//
//               if (r<q && u==v) t1t3c[S(p,t,nvir)] += e2ovov[De(r,v,q,u,nocc,nvir)] * t3 * scale; 
//               if (r<p && u==v) t1t3c[S(q,t,nvir)] -= e2ovov[De(r,v,p,u,nocc,nvir)] * t3 * scale; 
//               if (r<q && t==v) t1t3c[S(p,u,nvir)] -= e2ovov[De(r,v,q,t,nocc,nvir)] * t3 * scale; 
//               if (r<p && t==v) t1t3c[S(q,u,nvir)] += e2ovov[De(r,v,p,t,nocc,nvir)] * t3 * scale; 
//
//               scale = 0.5;
//               if (r==q && u==v) t1t3c[S(p,t,nvir)] += e2ovov[De(r,v,q,u,nocc,nvir)] * t3 * scale; 
//               if (r==p && u==v) t1t3c[S(q,t,nvir)] -= e2ovov[De(r,v,p,u,nocc,nvir)] * t3 * scale; 
//               if (r==q && t==v) t1t3c[S(p,u,nvir)] -= e2ovov[De(r,v,q,t,nocc,nvir)] * t3 * scale; 
//               if (r==p && t==v) t1t3c[S(q,u,nvir)] += e2ovov[De(r,v,p,t,nocc,nvir)] * t3 * scale; 
           }
       }
       fclose(fp);
    }
    else
    {
       // error message
    }

    //printf (" 0SDT (T) =    %f   ( %f )\n", norm, norm-norm0SD);

}

void t1t3c_dmrg_omp(double *t1t3c, double *t1, double *t2aa, double *t2ab, double *e2ovov, const int nc, const int nocc, const int nvir, const double numzero, const double c0, double norm) 
{

    double norm0SD = norm;
    const int t1size = nocc*nvir;

    //printf ("nocc, nvir, nc = %d, %d, %d\n",nocc, nvir, nc);

#pragma omp parallel default(none) \
        shared(t1, t2aa, t2ab, e2ovov, t1t3c, norm)
{
    int p, q, r, t, u, v, itmp, it, at;
    double t3, parity, scale;
    uint8_t Refdet[nocc+nvir], det_str[nocc+nvir];
    for (itmp = 0; itmp < nocc+nvir; itmp++){
        if (itmp<nocc) Refdet[itmp] = 3;  
        else           Refdet[itmp] = 0;
    }

    double *t1t3c_priv;
    t1t3c_priv = (double *)malloc(sizeof(double) * t1size); 
    for (it=0; it< t1size; it++){
        t1t3c_priv[it] = 0.0;
    }

    int i;
#pragma omp for reduction(+ : norm)
    for (i=0; i<omp_get_num_threads(); i++){

        char line[255], typ[4];
        //char *ptr;
        char s0[20]="t3.";
        char s1[4];

        sprintf(s1, "%d", i);
        char* filename = strcat(s0,s1);
        FILE *fp = fopen(filename, "r");
        //printf ("filename = %s\n",filename);

        fp = fopen(filename, "r");
    
        if (fp) {
           while ( !feof(fp) ){

               fscanf(fp, "%c%c%c,%s\n", &(typ[0]), &(typ[1]), &(typ[2]), line);
               fscanf(fp, "%lf\n", &t3);
               if (strncmp(typ, "aaa", 3) == 0 && fabs(t3) > numzero){
                   sscanf(line,"%d,%d,%d,%d,%d,%d",&p,&q,&r,&t,&u,&v);
                   norm += 2.0*t3*t3;
                   p += nc;
                   q += nc;
                   r += nc;
                   t += - nocc + nc;
                   u += - nocc + nc;
                   v += - nocc + nc;
    
                   for (itmp = 0; itmp < nocc+nvir; itmp++)
                       det_str[itmp] = Refdet[itmp];  
       
                   det_str[p] = 1;
                   det_str[q] = 1;
                   det_str[r] = 1;
                   det_str[t+nocc] = 2;
                   det_str[u+nocc] = 2;
                   det_str[v+nocc] = 2;
       
                   //parity  = parity_ab_str(det_str, nocc+nvir);
                   parity = parity_ci_to_cc(p+q+r, 3, nocc);
       
                   // interm norm of c3
                   t3 = parity * t3 / c0;
                   // extract t3
                   t3-= t1xt2aaa (p, q, r, t, u, v, nocc, nvir, t1, t2aa); 
                   t3-= t1xt1xt1aaa (p, q, r, t, u, v, nocc, nvir, t1); 
    
                   scale = 1.0;
                   t1t3c_priv[S(p,t,nvir)] += (e2ovov[De(q,u,r,v,nocc,nvir)]-e2ovov[De(r,u,q,v,nocc,nvir)]) * t3 * scale; 
                   t1t3c_priv[S(q,t,nvir)] -= (e2ovov[De(p,u,r,v,nocc,nvir)]-e2ovov[De(r,u,p,v,nocc,nvir)]) * t3 * scale; 
                   t1t3c_priv[S(r,t,nvir)] += (e2ovov[De(p,u,q,v,nocc,nvir)]-e2ovov[De(q,u,p,v,nocc,nvir)]) * t3 * scale; 
                   t1t3c_priv[S(p,u,nvir)] -= (e2ovov[De(q,t,r,v,nocc,nvir)]-e2ovov[De(r,t,q,v,nocc,nvir)]) * t3 * scale; 
                   t1t3c_priv[S(q,u,nvir)] += (e2ovov[De(p,t,r,v,nocc,nvir)]-e2ovov[De(r,t,p,v,nocc,nvir)]) * t3 * scale; 
                   t1t3c_priv[S(r,u,nvir)] -= (e2ovov[De(p,t,q,v,nocc,nvir)]-e2ovov[De(q,t,p,v,nocc,nvir)]) * t3 * scale; 
                   t1t3c_priv[S(p,v,nvir)] += (e2ovov[De(q,t,r,u,nocc,nvir)]-e2ovov[De(r,t,q,u,nocc,nvir)]) * t3 * scale; 
                   t1t3c_priv[S(q,v,nvir)] -= (e2ovov[De(p,t,r,u,nocc,nvir)]-e2ovov[De(r,t,p,u,nocc,nvir)]) * t3 * scale; 
                   t1t3c_priv[S(r,v,nvir)] += (e2ovov[De(p,t,q,u,nocc,nvir)]-e2ovov[De(q,t,p,u,nocc,nvir)]) * t3 * scale; 
    
               }
               else if (strncmp(typ, "aab", 3) == 0 && fabs(t3) > numzero){
                   sscanf(line,"%d,%d,%d,%d,%d,%d",&p,&q,&t,&u,&r,&v);
                   //printf("'%s, %d, %d, %d, %d, %d, %d, %15.8f'\n", typ, p,q,t,u,r,v,t3);
                   norm += 2.0*t3*t3; 
    
                   p += nc;
                   q += nc;
                   r += nc;
                   t += - nocc + nc;
                   u += - nocc + nc;
                   v += - nocc + nc;
    
                   //lsh test
                   //if(!(p == 2 && q == 3 && r == 3 && \
                   //     t == 0 && u == 1 && v == 1)) continue;
                   //printf("c3 in OTF: %15.8f\n",t3);
    
                   for (itmp = 0; itmp < nocc+nvir; itmp++)
                       det_str[itmp] = Refdet[itmp];  
                   det_str[p] = 1;
                   det_str[q] = 1;
                   det_str[t+nocc] = 2;
                   det_str[u+nocc] = 2;
       
                   if (p != r && q != r) det_str[r] = 2;
                   else  det_str[r] = 0;
                   if (t != v && u != v) det_str[v+nocc] = 1;
                   else  det_str[v+nocc] = 3;
                   //parity  = parity_ab_str(det_str, nocc+nvir);
                   parity = parity_ci_to_cc(p+q, 2, nocc);
                   parity *= parity_ci_to_cc(r, 1, nocc);
       
                   // interm norm of c3
                   t3 = parity * t3 / c0;
       
                   // extract t3 
                   t3-= t1xt2aab(p, q, r, t, u, v, nocc, nvir, t1, t2aa, t2ab); 
                   t3-= t1xt1xt1aab(p, q, r, t, u, v, nocc, nvir, t1); 
                   //printf("t3 in OTF: %15.8f\n",t3);
    
                   t1t3c_priv[S(r,v,nvir)] += (e2ovov[De(p,t,q,u,nocc,nvir)] - e2ovov[De(p,u,q,t,nocc,nvir)]) * t3; 
    
                   scale = 1.0;
                   if (r<q && v<u) t1t3c_priv[S(p,t,nvir)] += e2ovov[De(r,v,q,u,nocc,nvir)] * t3 * scale; 
                   if (r<p && v<u) t1t3c_priv[S(q,t,nvir)] -= e2ovov[De(r,v,p,u,nocc,nvir)] * t3 * scale; 
                   if (r<q && v<t) t1t3c_priv[S(p,u,nvir)] -= e2ovov[De(r,v,q,t,nocc,nvir)] * t3 * scale; 
                   if (r<p && v<t) t1t3c_priv[S(q,u,nvir)] += e2ovov[De(r,v,p,t,nocc,nvir)] * t3 * scale; 
    
                   if (q<r && u<v) t1t3c_priv[S(p,t,nvir)] += e2ovov[De(q,u,r,v,nocc,nvir)] * t3 * scale; 
                   if (p<r && u<v) t1t3c_priv[S(q,t,nvir)] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t3 * scale; 
                   if (q<r && t<v) t1t3c_priv[S(p,u,nvir)] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t3 * scale; 
                   if (p<r && t<v) t1t3c_priv[S(q,u,nvir)] += e2ovov[De(p,t,r,v,nocc,nvir)] * t3 * scale; 
    
                   scale = 0.5;
                   if (r==q && v<u) t1t3c_priv[S(p,t,nvir)] += e2ovov[De(r,v,q,u,nocc,nvir)] * t3 * scale; 
                   if (r==p && v<u) t1t3c_priv[S(q,t,nvir)] -= e2ovov[De(r,v,p,u,nocc,nvir)] * t3 * scale; 
                   if (r==q && v<t) t1t3c_priv[S(p,u,nvir)] -= e2ovov[De(r,v,q,t,nocc,nvir)] * t3 * scale; 
                   if (r==p && v<t) t1t3c_priv[S(q,u,nvir)] += e2ovov[De(r,v,p,t,nocc,nvir)] * t3 * scale; 
    
                   if (q==r && u<v) t1t3c_priv[S(p,t,nvir)] += e2ovov[De(q,u,r,v,nocc,nvir)] * t3 * scale; 
                   if (p==r && u<v) t1t3c_priv[S(q,t,nvir)] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t3 * scale; 
                   if (q==r && t<v) t1t3c_priv[S(p,u,nvir)] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t3 * scale; 
                   if (p==r && t<v) t1t3c_priv[S(q,u,nvir)] += e2ovov[De(p,t,r,v,nocc,nvir)] * t3 * scale; 
    
                   if (r<q && v==u) t1t3c_priv[S(p,t,nvir)] += e2ovov[De(r,v,q,u,nocc,nvir)] * t3 * scale; 
                   if (r<p && v==u) t1t3c_priv[S(q,t,nvir)] -= e2ovov[De(r,v,p,u,nocc,nvir)] * t3 * scale; 
                   if (r<q && v==t) t1t3c_priv[S(p,u,nvir)] -= e2ovov[De(r,v,q,t,nocc,nvir)] * t3 * scale; 
                   if (r<p && v==t) t1t3c_priv[S(q,u,nvir)] += e2ovov[De(r,v,p,t,nocc,nvir)] * t3 * scale; 
    
                   if (q<r && u==v) t1t3c_priv[S(p,t,nvir)] += e2ovov[De(q,u,r,v,nocc,nvir)] * t3 * scale; 
                   if (p<r && u==v) t1t3c_priv[S(q,t,nvir)] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t3 * scale; 
                   if (q<r && t==v) t1t3c_priv[S(p,u,nvir)] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t3 * scale; 
                   if (p<r && t==v) t1t3c_priv[S(q,u,nvir)] += e2ovov[De(p,t,r,v,nocc,nvir)] * t3 * scale; 
    
                   scale = 0.25;
                   if (r==q && v==u) t1t3c_priv[S(p,t,nvir)] += e2ovov[De(r,v,q,u,nocc,nvir)] * t3 * scale; 
                   if (r==p && v==u) t1t3c_priv[S(q,t,nvir)] -= e2ovov[De(r,v,p,u,nocc,nvir)] * t3 * scale; 
                   if (r==q && v==t) t1t3c_priv[S(p,u,nvir)] -= e2ovov[De(r,v,q,t,nocc,nvir)] * t3 * scale; 
                   if (r==p && v==t) t1t3c_priv[S(q,u,nvir)] += e2ovov[De(r,v,p,t,nocc,nvir)] * t3 * scale; 
    
                   if (q==r && u==v) t1t3c_priv[S(p,t,nvir)] += e2ovov[De(q,u,r,v,nocc,nvir)] * t3 * scale; 
                   if (p==r && u==v) t1t3c_priv[S(q,t,nvir)] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t3 * scale; 
                   if (q==r && t==v) t1t3c_priv[S(p,u,nvir)] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t3 * scale; 
                   if (p==r && t==v) t1t3c_priv[S(q,u,nvir)] += e2ovov[De(p,t,r,v,nocc,nvir)] * t3 * scale; 
    
                   scale = 1.0;
                   if (r<q && u<v) t1t3c_priv[S(p,t,nvir)] += e2ovov[De(r,v,q,u,nocc,nvir)] * t3 * scale; 
                   if (r<p && u<v) t1t3c_priv[S(q,t,nvir)] -= e2ovov[De(r,v,p,u,nocc,nvir)] * t3 * scale; 
                   if (r<q && t<v) t1t3c_priv[S(p,u,nvir)] -= e2ovov[De(r,v,q,t,nocc,nvir)] * t3 * scale; 
                   if (r<p && t<v) t1t3c_priv[S(q,u,nvir)] += e2ovov[De(r,v,p,t,nocc,nvir)] * t3 * scale; 
    
                   if (q<r && v<u) t1t3c_priv[S(p,t,nvir)] += e2ovov[De(q,u,r,v,nocc,nvir)] * t3 * scale; 
                   if (p<r && v<u) t1t3c_priv[S(q,t,nvir)] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t3 * scale; 
                   if (q<r && v<t) t1t3c_priv[S(p,u,nvir)] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t3 * scale; 
                   if (p<r && v<t) t1t3c_priv[S(q,u,nvir)] += e2ovov[De(p,t,r,v,nocc,nvir)] * t3 * scale; 
    
                   scale = 0.5;
                   if (r==q && u<v) t1t3c_priv[S(p,t,nvir)] += e2ovov[De(r,v,q,u,nocc,nvir)] * t3 * scale; 
                   if (r==p && u<v) t1t3c_priv[S(q,t,nvir)] -= e2ovov[De(r,v,p,u,nocc,nvir)] * t3 * scale; 
                   if (r==q && t<v) t1t3c_priv[S(p,u,nvir)] -= e2ovov[De(r,v,q,t,nocc,nvir)] * t3 * scale; 
                   if (r==p && t<v) t1t3c_priv[S(q,u,nvir)] += e2ovov[De(r,v,p,t,nocc,nvir)] * t3 * scale; 
    
                   if (q==r && v<u) t1t3c_priv[S(p,t,nvir)] += e2ovov[De(q,u,r,v,nocc,nvir)] * t3 * scale; 
                   if (p==r && v<u) t1t3c_priv[S(q,t,nvir)] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t3 * scale; 
                   if (q==r && v<t) t1t3c_priv[S(p,u,nvir)] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t3 * scale; 
                   if (p==r && v<t) t1t3c_priv[S(q,u,nvir)] += e2ovov[De(p,t,r,v,nocc,nvir)] * t3 * scale; 
    
                   if (r<q && u==v) t1t3c_priv[S(p,t,nvir)] += e2ovov[De(r,v,q,u,nocc,nvir)] * t3 * scale; 
                   if (r<p && u==v) t1t3c_priv[S(q,t,nvir)] -= e2ovov[De(r,v,p,u,nocc,nvir)] * t3 * scale; 
                   if (r<q && t==v) t1t3c_priv[S(p,u,nvir)] -= e2ovov[De(r,v,q,t,nocc,nvir)] * t3 * scale; 
                   if (r<p && t==v) t1t3c_priv[S(q,u,nvir)] += e2ovov[De(r,v,p,t,nocc,nvir)] * t3 * scale; 
    
                   if (q<r && v==u) t1t3c_priv[S(p,t,nvir)] += e2ovov[De(q,u,r,v,nocc,nvir)] * t3 * scale; 
                   if (p<r && v==u) t1t3c_priv[S(q,t,nvir)] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t3 * scale; 
                   if (q<r && v==t) t1t3c_priv[S(p,u,nvir)] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t3 * scale; 
                   if (p<r && v==t) t1t3c_priv[S(q,u,nvir)] += e2ovov[De(p,t,r,v,nocc,nvir)] * t3 * scale; 
    
                   scale = 0.25;
                   if (r==q && u==v) t1t3c_priv[S(p,t,nvir)] += e2ovov[De(r,v,q,u,nocc,nvir)] * t3 * scale; 
                   if (r==p && u==v) t1t3c_priv[S(q,t,nvir)] -= e2ovov[De(r,v,p,u,nocc,nvir)] * t3 * scale; 
                   if (r==q && t==v) t1t3c_priv[S(p,u,nvir)] -= e2ovov[De(r,v,q,t,nocc,nvir)] * t3 * scale; 
                   if (r==p && t==v) t1t3c_priv[S(q,u,nvir)] += e2ovov[De(r,v,p,t,nocc,nvir)] * t3 * scale; 
    
                   if (q==r && v==u) t1t3c_priv[S(p,t,nvir)] += e2ovov[De(q,u,r,v,nocc,nvir)] * t3 * scale; 
                   if (p==r && v==u) t1t3c_priv[S(q,t,nvir)] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t3 * scale; 
                   if (q==r && v==t) t1t3c_priv[S(p,u,nvir)] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t3 * scale; 
                   if (p==r && v==t) t1t3c_priv[S(q,u,nvir)] += e2ovov[De(p,t,r,v,nocc,nvir)] * t3 * scale; 
    
               }
           }
           fclose(fp);
        }
        else
        {
           // error message
        }

    }
#pragma omp critical
    { 
        for (it=0; it< nocc; it++){
            for (at=0; at< nvir; at++){
                t1t3c[S(it,at,nvir)] += t1t3c_priv[S(it,at,nvir)];
            }
        }   
        free(t1t3c_priv);
    } 

}

    //printf (" 0SDT (T) =    %f   ( %f )\n", norm, norm-norm0SD);

}

void t1t3c_shci_omp(double *t1t3c, double *t1, double *t2aa, double *t2ab, double *e2ovov, const int nc, const int nocc, const int nvir, const double numzero, const double c0, double norm) 
{

    double norm0SD = norm;
    const int t1size = nocc*nvir;

    //printf ("nocc, nvir, nc = %d, %d, %d\n",nocc, nvir, nc);

#pragma omp parallel default(none) \
        shared(t1, t2aa, t2ab, e2ovov, t1t3c, norm)
{
    int p, q, r, t, u, v, itmp, it, at;
    double t3, parity, scale;
    uint8_t Refdet[nocc+nvir], det_str[nocc+nvir];
    for (itmp = 0; itmp < nocc+nvir; itmp++){
        if (itmp<nocc) Refdet[itmp] = 3;  
        else           Refdet[itmp] = 0;
    }

    double *t1t3c_priv;
    t1t3c_priv = (double *)malloc(sizeof(double) * t1size); 
    for (it=0; it< t1size; it++){
        t1t3c_priv[it] = 0.0;
    }

    int i;
#pragma omp for reduction(+ : norm)
    for (i=0; i<omp_get_num_threads(); i++){

        char line[255], typ[4];
        //char *ptr;
        char s0[20]="t3.";
        char s1[4];

        sprintf(s1, "%d", i);
        char* filename = strcat(s0,s1);
        FILE *fp = fopen(filename, "r");
        //printf ("filename = %s\n",filename);

        fp = fopen(filename, "r");
    
        if (fp) {
           while ( !feof(fp) ){

               fscanf(fp, "%c%c%c,%s\n", &(typ[0]), &(typ[1]), &(typ[2]), line);
               fscanf(fp, "%lf\n", &t3);
               if (strncmp(typ, "aaa", 3) == 0 && fabs(t3) > numzero){
                   sscanf(line,"%d,%d,%d,%d,%d,%d",&p,&q,&r,&t,&u,&v);
                   norm += 2.0*t3*t3;
                   p += nc;
                   q += nc;
                   r += nc;
                   t += - nocc + nc;
                   u += - nocc + nc;
                   v += - nocc + nc;
    
                   for (itmp = 0; itmp < nocc+nvir; itmp++)
                       det_str[itmp] = Refdet[itmp];  
       
                   det_str[p] = 2;
                   det_str[q] = 2;
                   det_str[r] = 2;
                   det_str[t+nocc] = 1;
                   det_str[u+nocc] = 1;
                   det_str[v+nocc] = 1;
       
                   //parity  = parity_ab_str(det_str, nocc+nvir);
                   parity = parity_ci_to_cc(p+q+r, 3, nocc);
       
                   // interm norm of c3
                   t3 = parity * t3 / c0;
                   // extract t3
                   t3-= t1xt2aaa (p, q, r, t, u, v, nocc, nvir, t1, t2aa); 
                   t3-= t1xt1xt1aaa (p, q, r, t, u, v, nocc, nvir, t1); 
    
                   scale = 1.0;
                   t1t3c_priv[S(p,t,nvir)] += (e2ovov[De(q,u,r,v,nocc,nvir)]-e2ovov[De(r,u,q,v,nocc,nvir)]) * t3 * scale; 
                   t1t3c_priv[S(q,t,nvir)] -= (e2ovov[De(p,u,r,v,nocc,nvir)]-e2ovov[De(r,u,p,v,nocc,nvir)]) * t3 * scale; 
                   t1t3c_priv[S(r,t,nvir)] += (e2ovov[De(p,u,q,v,nocc,nvir)]-e2ovov[De(q,u,p,v,nocc,nvir)]) * t3 * scale; 
                   t1t3c_priv[S(p,u,nvir)] -= (e2ovov[De(q,t,r,v,nocc,nvir)]-e2ovov[De(r,t,q,v,nocc,nvir)]) * t3 * scale; 
                   t1t3c_priv[S(q,u,nvir)] += (e2ovov[De(p,t,r,v,nocc,nvir)]-e2ovov[De(r,t,p,v,nocc,nvir)]) * t3 * scale; 
                   t1t3c_priv[S(r,u,nvir)] -= (e2ovov[De(p,t,q,v,nocc,nvir)]-e2ovov[De(q,t,p,v,nocc,nvir)]) * t3 * scale; 
                   t1t3c_priv[S(p,v,nvir)] += (e2ovov[De(q,t,r,u,nocc,nvir)]-e2ovov[De(r,t,q,u,nocc,nvir)]) * t3 * scale; 
                   t1t3c_priv[S(q,v,nvir)] -= (e2ovov[De(p,t,r,u,nocc,nvir)]-e2ovov[De(r,t,p,u,nocc,nvir)]) * t3 * scale; 
                   t1t3c_priv[S(r,v,nvir)] += (e2ovov[De(p,t,q,u,nocc,nvir)]-e2ovov[De(q,t,p,u,nocc,nvir)]) * t3 * scale; 
    
               }
               else if (strncmp(typ, "aab", 3) == 0 && fabs(t3) > numzero){
                   sscanf(line,"%d,%d,%d,%d,%d,%d",&p,&q,&t,&u,&r,&v);
                   //printf("'%s, %d, %d, %d, %d, %d, %d, %15.8f'\n", typ, p,q,t,u,r,v,t3);
                   norm += 2.0*t3*t3; 
    
                   p += nc;
                   q += nc;
                   r += nc;
                   t += - nocc + nc;
                   u += - nocc + nc;
                   v += - nocc + nc;
    
                   //lsh test
                   //if(!(p == 2 && q == 3 && r == 3 && \
                   //     t == 0 && u == 1 && v == 1)) continue;
                   //printf("c3 in OTF: %15.8f\n",t3);
    
                   for (itmp = 0; itmp < nocc+nvir; itmp++)
                       det_str[itmp] = Refdet[itmp];  
                   det_str[p] = 2;
                   det_str[q] = 2;
                   det_str[t+nocc] = 1;
                   det_str[u+nocc] = 1;
       
                   if (p != r && q != r) det_str[r] = 1;
                   else  det_str[r] = 0;
                   if (t != v && u != v) det_str[v+nocc] = 2;
                   else  det_str[v+nocc] = 3;
                   //parity  = parity_ab_str(det_str, nocc+nvir);
                   parity = parity_ci_to_cc(p+q, 2, nocc);
                   parity *= parity_ci_to_cc(r, 1, nocc);
       
                   // interm norm of c3
                   t3 = parity * t3 / c0;
       
                   // extract t3 
                   t3-= t1xt2aab(p, q, r, t, u, v, nocc, nvir, t1, t2aa, t2ab); 
                   t3-= t1xt1xt1aab(p, q, r, t, u, v, nocc, nvir, t1); 
                   //printf("t3 in OTF: %15.8f\n",t3);
    
                   t1t3c_priv[S(r,v,nvir)] += (e2ovov[De(p,t,q,u,nocc,nvir)] - e2ovov[De(p,u,q,t,nocc,nvir)]) * t3; 
    
                   scale = 1.0;
                   if (r<q && v<u) t1t3c_priv[S(p,t,nvir)] += e2ovov[De(r,v,q,u,nocc,nvir)] * t3 * scale; 
                   if (r<p && v<u) t1t3c_priv[S(q,t,nvir)] -= e2ovov[De(r,v,p,u,nocc,nvir)] * t3 * scale; 
                   if (r<q && v<t) t1t3c_priv[S(p,u,nvir)] -= e2ovov[De(r,v,q,t,nocc,nvir)] * t3 * scale; 
                   if (r<p && v<t) t1t3c_priv[S(q,u,nvir)] += e2ovov[De(r,v,p,t,nocc,nvir)] * t3 * scale; 
    
                   if (q<r && u<v) t1t3c_priv[S(p,t,nvir)] += e2ovov[De(q,u,r,v,nocc,nvir)] * t3 * scale; 
                   if (p<r && u<v) t1t3c_priv[S(q,t,nvir)] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t3 * scale; 
                   if (q<r && t<v) t1t3c_priv[S(p,u,nvir)] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t3 * scale; 
                   if (p<r && t<v) t1t3c_priv[S(q,u,nvir)] += e2ovov[De(p,t,r,v,nocc,nvir)] * t3 * scale; 
    
                   scale = 0.5;
                   if (r==q && v<u) t1t3c_priv[S(p,t,nvir)] += e2ovov[De(r,v,q,u,nocc,nvir)] * t3 * scale; 
                   if (r==p && v<u) t1t3c_priv[S(q,t,nvir)] -= e2ovov[De(r,v,p,u,nocc,nvir)] * t3 * scale; 
                   if (r==q && v<t) t1t3c_priv[S(p,u,nvir)] -= e2ovov[De(r,v,q,t,nocc,nvir)] * t3 * scale; 
                   if (r==p && v<t) t1t3c_priv[S(q,u,nvir)] += e2ovov[De(r,v,p,t,nocc,nvir)] * t3 * scale; 
    
                   if (q==r && u<v) t1t3c_priv[S(p,t,nvir)] += e2ovov[De(q,u,r,v,nocc,nvir)] * t3 * scale; 
                   if (p==r && u<v) t1t3c_priv[S(q,t,nvir)] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t3 * scale; 
                   if (q==r && t<v) t1t3c_priv[S(p,u,nvir)] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t3 * scale; 
                   if (p==r && t<v) t1t3c_priv[S(q,u,nvir)] += e2ovov[De(p,t,r,v,nocc,nvir)] * t3 * scale; 
    
                   if (r<q && v==u) t1t3c_priv[S(p,t,nvir)] += e2ovov[De(r,v,q,u,nocc,nvir)] * t3 * scale; 
                   if (r<p && v==u) t1t3c_priv[S(q,t,nvir)] -= e2ovov[De(r,v,p,u,nocc,nvir)] * t3 * scale; 
                   if (r<q && v==t) t1t3c_priv[S(p,u,nvir)] -= e2ovov[De(r,v,q,t,nocc,nvir)] * t3 * scale; 
                   if (r<p && v==t) t1t3c_priv[S(q,u,nvir)] += e2ovov[De(r,v,p,t,nocc,nvir)] * t3 * scale; 
    
                   if (q<r && u==v) t1t3c_priv[S(p,t,nvir)] += e2ovov[De(q,u,r,v,nocc,nvir)] * t3 * scale; 
                   if (p<r && u==v) t1t3c_priv[S(q,t,nvir)] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t3 * scale; 
                   if (q<r && t==v) t1t3c_priv[S(p,u,nvir)] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t3 * scale; 
                   if (p<r && t==v) t1t3c_priv[S(q,u,nvir)] += e2ovov[De(p,t,r,v,nocc,nvir)] * t3 * scale; 
    
                   scale = 0.25;
                   if (r==q && v==u) t1t3c_priv[S(p,t,nvir)] += e2ovov[De(r,v,q,u,nocc,nvir)] * t3 * scale; 
                   if (r==p && v==u) t1t3c_priv[S(q,t,nvir)] -= e2ovov[De(r,v,p,u,nocc,nvir)] * t3 * scale; 
                   if (r==q && v==t) t1t3c_priv[S(p,u,nvir)] -= e2ovov[De(r,v,q,t,nocc,nvir)] * t3 * scale; 
                   if (r==p && v==t) t1t3c_priv[S(q,u,nvir)] += e2ovov[De(r,v,p,t,nocc,nvir)] * t3 * scale; 
    
                   if (q==r && u==v) t1t3c_priv[S(p,t,nvir)] += e2ovov[De(q,u,r,v,nocc,nvir)] * t3 * scale; 
                   if (p==r && u==v) t1t3c_priv[S(q,t,nvir)] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t3 * scale; 
                   if (q==r && t==v) t1t3c_priv[S(p,u,nvir)] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t3 * scale; 
                   if (p==r && t==v) t1t3c_priv[S(q,u,nvir)] += e2ovov[De(p,t,r,v,nocc,nvir)] * t3 * scale; 
    
                   scale = 1.0;
                   if (r<q && u<v) t1t3c_priv[S(p,t,nvir)] += e2ovov[De(r,v,q,u,nocc,nvir)] * t3 * scale; 
                   if (r<p && u<v) t1t3c_priv[S(q,t,nvir)] -= e2ovov[De(r,v,p,u,nocc,nvir)] * t3 * scale; 
                   if (r<q && t<v) t1t3c_priv[S(p,u,nvir)] -= e2ovov[De(r,v,q,t,nocc,nvir)] * t3 * scale; 
                   if (r<p && t<v) t1t3c_priv[S(q,u,nvir)] += e2ovov[De(r,v,p,t,nocc,nvir)] * t3 * scale; 
    
                   if (q<r && v<u) t1t3c_priv[S(p,t,nvir)] += e2ovov[De(q,u,r,v,nocc,nvir)] * t3 * scale; 
                   if (p<r && v<u) t1t3c_priv[S(q,t,nvir)] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t3 * scale; 
                   if (q<r && v<t) t1t3c_priv[S(p,u,nvir)] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t3 * scale; 
                   if (p<r && v<t) t1t3c_priv[S(q,u,nvir)] += e2ovov[De(p,t,r,v,nocc,nvir)] * t3 * scale; 
    
                   scale = 0.5;
                   if (r==q && u<v) t1t3c_priv[S(p,t,nvir)] += e2ovov[De(r,v,q,u,nocc,nvir)] * t3 * scale; 
                   if (r==p && u<v) t1t3c_priv[S(q,t,nvir)] -= e2ovov[De(r,v,p,u,nocc,nvir)] * t3 * scale; 
                   if (r==q && t<v) t1t3c_priv[S(p,u,nvir)] -= e2ovov[De(r,v,q,t,nocc,nvir)] * t3 * scale; 
                   if (r==p && t<v) t1t3c_priv[S(q,u,nvir)] += e2ovov[De(r,v,p,t,nocc,nvir)] * t3 * scale; 
    
                   if (q==r && v<u) t1t3c_priv[S(p,t,nvir)] += e2ovov[De(q,u,r,v,nocc,nvir)] * t3 * scale; 
                   if (p==r && v<u) t1t3c_priv[S(q,t,nvir)] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t3 * scale; 
                   if (q==r && v<t) t1t3c_priv[S(p,u,nvir)] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t3 * scale; 
                   if (p==r && v<t) t1t3c_priv[S(q,u,nvir)] += e2ovov[De(p,t,r,v,nocc,nvir)] * t3 * scale; 
    
                   if (r<q && u==v) t1t3c_priv[S(p,t,nvir)] += e2ovov[De(r,v,q,u,nocc,nvir)] * t3 * scale; 
                   if (r<p && u==v) t1t3c_priv[S(q,t,nvir)] -= e2ovov[De(r,v,p,u,nocc,nvir)] * t3 * scale; 
                   if (r<q && t==v) t1t3c_priv[S(p,u,nvir)] -= e2ovov[De(r,v,q,t,nocc,nvir)] * t3 * scale; 
                   if (r<p && t==v) t1t3c_priv[S(q,u,nvir)] += e2ovov[De(r,v,p,t,nocc,nvir)] * t3 * scale; 
    
                   if (q<r && v==u) t1t3c_priv[S(p,t,nvir)] += e2ovov[De(q,u,r,v,nocc,nvir)] * t3 * scale; 
                   if (p<r && v==u) t1t3c_priv[S(q,t,nvir)] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t3 * scale; 
                   if (q<r && v==t) t1t3c_priv[S(p,u,nvir)] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t3 * scale; 
                   if (p<r && v==t) t1t3c_priv[S(q,u,nvir)] += e2ovov[De(p,t,r,v,nocc,nvir)] * t3 * scale; 
    
                   scale = 0.25;
                   if (r==q && u==v) t1t3c_priv[S(p,t,nvir)] += e2ovov[De(r,v,q,u,nocc,nvir)] * t3 * scale; 
                   if (r==p && u==v) t1t3c_priv[S(q,t,nvir)] -= e2ovov[De(r,v,p,u,nocc,nvir)] * t3 * scale; 
                   if (r==q && t==v) t1t3c_priv[S(p,u,nvir)] -= e2ovov[De(r,v,q,t,nocc,nvir)] * t3 * scale; 
                   if (r==p && t==v) t1t3c_priv[S(q,u,nvir)] += e2ovov[De(r,v,p,t,nocc,nvir)] * t3 * scale; 
    
                   if (q==r && v==u) t1t3c_priv[S(p,t,nvir)] += e2ovov[De(q,u,r,v,nocc,nvir)] * t3 * scale; 
                   if (p==r && v==u) t1t3c_priv[S(q,t,nvir)] -= e2ovov[De(p,u,r,v,nocc,nvir)] * t3 * scale; 
                   if (q==r && v==t) t1t3c_priv[S(p,u,nvir)] -= e2ovov[De(q,t,r,v,nocc,nvir)] * t3 * scale; 
                   if (p==r && v==t) t1t3c_priv[S(q,u,nvir)] += e2ovov[De(p,t,r,v,nocc,nvir)] * t3 * scale; 
    
               }
           }
           fclose(fp);
        }
        else
        {
           // error message
        }

    }
#pragma omp critical
    { 
        for (it=0; it< nocc; it++){
            for (at=0; at< nvir; at++){
                t1t3c[S(it,at,nvir)] += t1t3c_priv[S(it,at,nvir)];
            }
        }   
        free(t1t3c_priv);
    } 

}

    //printf (" 0SDT (T) =    %f   ( %f )\n", norm, norm-norm0SD);

}

void t2t3c_shci(double *t2_t3t4c, double *t1, double *t2aa, double *t2ab, double *tmp1, double *tmp2, double *tmp3, const int nc, const int nocc, const int nvir, const double numzero, const double c0) 
{
    int p, q, r, t, u, v, itmp, it, jt, at, bt;
    double t3, parity, scale;
    uint8_t Refdet[nocc+nvir], det_str[nocc+nvir];
    for (itmp = 0; itmp < nocc+nvir; itmp++){
        if (itmp<nocc) Refdet[itmp] = 3;  
        else           Refdet[itmp] = 0;
    }

//    double ****tmp;
//    tmp = (double ****)malloc(sizeof(double ***) * nocc); 
//    for (it=0; it< nocc; it++){
//        tmp[it] = (double ***)malloc(sizeof(double **) * nocc);
//        for (jt=0; jt< nocc; jt++){
//            tmp[it][jt] = (double **)malloc(sizeof(double *) * nvir);
//            for (at=0; at< nvir; at++){
//                tmp[it][jt][at] = (double *)malloc(sizeof(double) * nvir);
//            }
//        }
//    }

    FILE *fp;
    char line[255], typ[4], tmpc[255];
    char *ptr;
    fp = fopen("CIcoeff_shci.out", "r");
    fgets(line, 255, fp);

//    //test
//    int kt;
//    printf("tmp3 in c\n");
//    for (it=0; it< nocc; it++)
//    for (at=0; at< nvir; at++)
//    for (bt=0; bt< nvir; bt++)
//    for (ct=0; ct< nvir; ct++)
//        printf("%d %d %d %d %15.8f \n",it,at,bt,ct,tmp3[Dtmp34(it,at,bt,ct,nocc,nvir)]);

    if (fp) {
       while ( !feof(fp) ){
           fgets(line, 255, fp);
           ptr = strtok(line, ",");

           it=0;
           while(ptr != NULL){
               if(it==0){
                   strcpy(typ, ptr);
               }
               if (it>0 && strlen(typ)==3 && strncmp(typ, "aab", 3)==0){
                   strcpy(tmpc, ptr);
                   if(it==1) p = atoi(tmpc);
                   if(it==2) q = atoi(tmpc);
                   if(it==3) t = atoi(tmpc);
                   if(it==4) u = atoi(tmpc);
                   if(it==5) r = atoi(tmpc);
                   if(it==6) v = atoi(tmpc);
                   if(it==7) t3= atof(tmpc);
               }
               if (it>7) break;
               ptr = strtok(NULL, ",");
               it++;
           }

           if (strlen(typ)==3 && strncmp(typ, "aab", 3) == 0 && fabs(t3) > numzero){
               //printf("'%s, %d, %d, %d, %d, %d, %d, %15.8f'\n", typ, p,q,t,u,r,v,t3);

               p += nc;
               q += nc;
               r += nc;
               t += - nocc + nc;
               u += - nocc + nc;
               v += - nocc + nc;

//               //lsh test
//               if(!(p == 2 && q == 3 && r == 3 && \
//                    t == 0 && u == 1 && v == 1)) continue;
//               //printf("c3 in OTF: %15.8f\n",t3);

               for (itmp = 0; itmp < nocc+nvir; itmp++)
                   det_str[itmp] = Refdet[itmp];  
               det_str[p] = 2;
               det_str[q] = 2;
               det_str[t+nocc] = 1;
               det_str[u+nocc] = 1;
   
               if (p != r && q != r) det_str[r] = 1;
               else  det_str[r] = 0;
               if (t != v && u != v) det_str[v+nocc] = 2;
               else  det_str[v+nocc] = 3;
               //parity  = parity_ab_str(det_str, nocc+nvir);
               parity = parity_ci_to_cc(p+q, 2, nocc);
               parity *= parity_ci_to_cc(r, 1, nocc);
   
               // interm norm of c3
               t3 = parity * t3 / c0;
   
               // extract t3 
               t3-= t1xt2aab(p, q, r, t, u, v, nocc, nvir, t1, t2aa, t2ab); 
               t3-= t1xt1xt1aab(p, q, r, t, u, v, nocc, nvir, t1); 
               //printf("t3 in OTF: %15.8f\n",t3);

//               for (it=0; it< nocc; it++)
//               for (jt=0; jt< nocc; jt++)
//               for (at=0; at< nvir; at++)
//               for (bt=0; bt< nvir; bt++)
//                   tmp[it][jt][at][bt] = 0.0;

               for (it=0; it< nocc; it++){
                   t2_t3t4c[D(it,r,u,v,nocc,nvir)] -= tmp1[Dtmp1(q,it,p,t,nocc,nvir)] * t3; 
                   t2_t3t4c[D(it,r,u,v,nocc,nvir)] += tmp1[Dtmp1(p,it,q,t,nocc,nvir)] * t3; 
                   t2_t3t4c[D(it,r,t,v,nocc,nvir)] += tmp1[Dtmp1(q,it,p,u,nocc,nvir)] * t3; 
                   t2_t3t4c[D(it,r,t,v,nocc,nvir)] -= tmp1[Dtmp1(p,it,q,u,nocc,nvir)] * t3; 
                   t2_t3t4c[D(it,q,v,u,nocc,nvir)] -= tmp1[Dtmp1(r,it,p,t,nocc,nvir)] * t3; 
                   t2_t3t4c[D(it,p,v,u,nocc,nvir)] += tmp1[Dtmp1(r,it,q,t,nocc,nvir)] * t3; 
                   t2_t3t4c[D(it,q,v,t,nocc,nvir)] += tmp1[Dtmp1(r,it,p,u,nocc,nvir)] * t3; 
                   t2_t3t4c[D(it,p,v,t,nocc,nvir)] -= tmp1[Dtmp1(r,it,q,u,nocc,nvir)] * t3; 

                   t2_t3t4c[D(q,it,u,v,nocc,nvir)] -= tmp1[Dtmp1(r,it,p,t,nocc,nvir)] * t3; 
                   t2_t3t4c[D(p,it,u,v,nocc,nvir)] += tmp1[Dtmp1(r,it,q,t,nocc,nvir)] * t3; 
                   t2_t3t4c[D(q,it,t,v,nocc,nvir)] += tmp1[Dtmp1(r,it,p,u,nocc,nvir)] * t3; 
                   t2_t3t4c[D(p,it,t,v,nocc,nvir)] -= tmp1[Dtmp1(r,it,q,u,nocc,nvir)] * t3; 
                   t2_t3t4c[D(r,it,v,u,nocc,nvir)] -= tmp1[Dtmp1(q,it,p,t,nocc,nvir)] * t3; 
                   t2_t3t4c[D(r,it,v,u,nocc,nvir)] += tmp1[Dtmp1(p,it,q,t,nocc,nvir)] * t3; 
                   t2_t3t4c[D(r,it,v,t,nocc,nvir)] += tmp1[Dtmp1(q,it,p,u,nocc,nvir)] * t3; 
                   t2_t3t4c[D(r,it,v,t,nocc,nvir)] -= tmp1[Dtmp1(p,it,q,u,nocc,nvir)] * t3; 
               }

               for (at = 0; at < nvir; at++){
                   t2_t3t4c[D(q,r,u,at,nocc,nvir)] += tmp2[Dtmp2(p,t,at,v,nvir)] * t3; 
                   t2_t3t4c[D(p,r,u,at,nocc,nvir)] -= tmp2[Dtmp2(q,t,at,v,nvir)] * t3; 
                   t2_t3t4c[D(q,r,t,at,nocc,nvir)] -= tmp2[Dtmp2(p,u,at,v,nvir)] * t3; 
                   t2_t3t4c[D(p,r,t,at,nocc,nvir)] += tmp2[Dtmp2(q,u,at,v,nvir)] * t3; 
                   t2_t3t4c[D(r,q,v,at,nocc,nvir)] += tmp2[Dtmp2(p,t,at,u,nvir)] * t3; 
                   t2_t3t4c[D(r,p,v,at,nocc,nvir)] -= tmp2[Dtmp2(q,t,at,u,nvir)] * t3; 
                   t2_t3t4c[D(r,q,v,at,nocc,nvir)] -= tmp2[Dtmp2(p,u,at,t,nvir)] * t3; 
                   t2_t3t4c[D(r,p,v,at,nocc,nvir)] += tmp2[Dtmp2(q,u,at,t,nvir)] * t3; 

                   t2_t3t4c[D(r,q,at,u,nocc,nvir)] += tmp2[Dtmp2(p,t,at,v,nvir)] * t3; 
                   t2_t3t4c[D(r,p,at,u,nocc,nvir)] -= tmp2[Dtmp2(q,t,at,v,nvir)] * t3; 
                   t2_t3t4c[D(r,q,at,t,nocc,nvir)] -= tmp2[Dtmp2(p,u,at,v,nvir)] * t3; 
                   t2_t3t4c[D(r,p,at,t,nocc,nvir)] += tmp2[Dtmp2(q,u,at,v,nvir)] * t3; 
                   t2_t3t4c[D(q,r,at,v,nocc,nvir)] += tmp2[Dtmp2(p,t,at,u,nvir)] * t3; 
                   t2_t3t4c[D(p,r,at,v,nocc,nvir)] -= tmp2[Dtmp2(q,t,at,u,nvir)] * t3; 
                   t2_t3t4c[D(q,r,at,v,nocc,nvir)] -= tmp2[Dtmp2(p,u,at,t,nvir)] * t3; 
                   t2_t3t4c[D(p,r,at,v,nocc,nvir)] += tmp2[Dtmp2(q,u,at,t,nvir)] * t3; 
               }

               t2_t3t4c[D(q,r,u,v,nocc,nvir)] += tmp3[S(p,t,nvir)] * t3; 
               t2_t3t4c[D(p,r,u,v,nocc,nvir)] -= tmp3[S(q,t,nvir)] * t3; 
               t2_t3t4c[D(q,r,t,v,nocc,nvir)] -= tmp3[S(p,u,nvir)] * t3; 
               t2_t3t4c[D(p,r,t,v,nocc,nvir)] += tmp3[S(q,u,nvir)] * t3; 
               t2_t3t4c[D(r,q,v,u,nocc,nvir)] += tmp3[S(p,t,nvir)] * t3; 
               t2_t3t4c[D(r,p,v,u,nocc,nvir)] -= tmp3[S(q,t,nvir)] * t3; 
               t2_t3t4c[D(r,q,v,t,nocc,nvir)] -= tmp3[S(p,u,nvir)] * t3; 
               t2_t3t4c[D(r,p,v,t,nocc,nvir)] += tmp3[S(q,u,nvir)] * t3; 

           }
       }
       fclose(fp);
    }
    else
    {
       // error message
    }


//    for (it=0; it< nocc; it++)
//    for (jt=0; jt< nocc; jt++)
//    for (at=0; at< nvir; at++)
//    for (bt=0; bt< nvir; bt++)
//        tmp[it][jt][at][bt] = t2_t3t4c[D(it,jt,at,bt,nocc,nvir)] + t2_t3t4c[D(jt,it,at,bt,nocc,nvir)];
//
//    for (it=0; it< nocc; it++)
//    for (jt=0; jt< nocc; jt++)
//    for (at=0; at< nvir; at++)
//    for (bt=0; bt< nvir; bt++)
//        t2_t3t4c[D(it,jt,at,bt,nocc,nvir)] = tmp[it][jt][at][bt];

//    for (it=0; it< nocc; it++){
//        for (jt=0; jt< nocc; jt++){
//            for (at=0; at< nvir; at++){
//                free(tmp[it][jt][at]);
//            }
//            free(tmp[it][jt]);
//        }
//        free(tmp[it]);
//    }   
//    free(tmp);

}

void t2t3c_dmrg_omp(double *t2_t3t4c, double *t1, double *t2aa, double *t2ab, double *tmp1, double *tmp2, double *tmp3, const int nc, const int nocc, const int nvir, const double numzero, const double c0) 
{

    const int t2size = nocc*nocc*nvir*nvir;

#pragma omp parallel default(none) \
        shared(t1, t2aa, t2ab, tmp1, tmp2, tmp3, t2_t3t4c)
{

    int p, q, r, t, u, v, itmp, it, jt, at, bt;
    double t3, parity, scale;
    uint8_t Refdet[nocc+nvir], det_str[nocc+nvir];
    for (itmp = 0; itmp < nocc+nvir; itmp++){
        if (itmp<nocc) Refdet[itmp] = 3;  
        else           Refdet[itmp] = 0;
    }

    double *t2_t3t4c_priv;
    t2_t3t4c_priv = (double *)malloc(sizeof(double) * t2size); 
    for (it=0; it< t2size; it++){
        t2_t3t4c_priv [it] = 0.0;
    }

    int i;
#pragma omp for
    for (i=0; i<omp_get_num_threads(); i++){

        char line[255], typ[4];
        //char *ptr;
        char s0[20]="t3.";
        char s1[4];

        sprintf(s1, "%d", i);
        char* filename = strcat(s0,s1);
        FILE *fp = fopen(filename, "r");
        //printf ("filename = %s\n",filename);

        fp = fopen(filename, "r");
    
        if (fp) {
           while ( !feof(fp) ){
               fscanf(fp, "%c%c%c,%s\n", &(typ[0]), &(typ[1]), &(typ[2]), line);
               fscanf(fp, "%lf\n", &t3);
               if (strncmp(typ, "aab", 3) == 0 && fabs(t3) > numzero){
                   sscanf(line,"%d,%d,%d,%d,%d,%d",&p,&q,&t,&u,&r,&v);
                   p += nc;
                   q += nc;
                   r += nc;
                   t += - nocc + nc;
                   u += - nocc + nc;
                   v += - nocc + nc;
    
                   for (itmp = 0; itmp < nocc+nvir; itmp++)
                       det_str[itmp] = Refdet[itmp];  
                   det_str[p] = 1;
                   det_str[q] = 1;
                   det_str[t+nocc] = 2;
                   det_str[u+nocc] = 2;
       
                   if (p != r && q != r) det_str[r] = 2;
                   else  det_str[r] = 0;
                   if (t != v && u != v) det_str[v+nocc] = 1;
                   else  det_str[v+nocc] = 3;
                   //parity  = parity_ab_str(det_str, nocc+nvir);
                   parity = parity_ci_to_cc(p+q, 2, nocc);
                   parity *= parity_ci_to_cc(r, 1, nocc);
       
                   // interm norm of c3
                   t3 = parity * t3 / c0;
       
                   // extract t3 
                   t3-= t1xt2aab(p, q, r, t, u, v, nocc, nvir, t1, t2aa, t2ab); 
                   t3-= t1xt1xt1aab(p, q, r, t, u, v, nocc, nvir, t1); 
   
                   for (it=0; it< nocc; it++){
                       t2_t3t4c_priv[D(it,r,u,v,nocc,nvir)] -= tmp1[Dtmp1(q,it,p,t,nocc,nvir)] * t3; 
                       t2_t3t4c_priv[D(it,r,u,v,nocc,nvir)] += tmp1[Dtmp1(p,it,q,t,nocc,nvir)] * t3; 
                       t2_t3t4c_priv[D(it,r,t,v,nocc,nvir)] += tmp1[Dtmp1(q,it,p,u,nocc,nvir)] * t3; 
                       t2_t3t4c_priv[D(it,r,t,v,nocc,nvir)] -= tmp1[Dtmp1(p,it,q,u,nocc,nvir)] * t3; 
                       t2_t3t4c_priv[D(it,q,v,u,nocc,nvir)] -= tmp1[Dtmp1(r,it,p,t,nocc,nvir)] * t3; 
                       t2_t3t4c_priv[D(it,p,v,u,nocc,nvir)] += tmp1[Dtmp1(r,it,q,t,nocc,nvir)] * t3; 
                       t2_t3t4c_priv[D(it,q,v,t,nocc,nvir)] += tmp1[Dtmp1(r,it,p,u,nocc,nvir)] * t3; 
                       t2_t3t4c_priv[D(it,p,v,t,nocc,nvir)] -= tmp1[Dtmp1(r,it,q,u,nocc,nvir)] * t3; 
    
                       t2_t3t4c_priv[D(q,it,u,v,nocc,nvir)] -= tmp1[Dtmp1(r,it,p,t,nocc,nvir)] * t3; 
                       t2_t3t4c_priv[D(p,it,u,v,nocc,nvir)] += tmp1[Dtmp1(r,it,q,t,nocc,nvir)] * t3; 
                       t2_t3t4c_priv[D(q,it,t,v,nocc,nvir)] += tmp1[Dtmp1(r,it,p,u,nocc,nvir)] * t3; 
                       t2_t3t4c_priv[D(p,it,t,v,nocc,nvir)] -= tmp1[Dtmp1(r,it,q,u,nocc,nvir)] * t3; 
                       t2_t3t4c_priv[D(r,it,v,u,nocc,nvir)] -= tmp1[Dtmp1(q,it,p,t,nocc,nvir)] * t3; 
                       t2_t3t4c_priv[D(r,it,v,u,nocc,nvir)] += tmp1[Dtmp1(p,it,q,t,nocc,nvir)] * t3; 
                       t2_t3t4c_priv[D(r,it,v,t,nocc,nvir)] += tmp1[Dtmp1(q,it,p,u,nocc,nvir)] * t3; 
                       t2_t3t4c_priv[D(r,it,v,t,nocc,nvir)] -= tmp1[Dtmp1(p,it,q,u,nocc,nvir)] * t3; 
                   }
    
                   for (at = 0; at < nvir; at++){
                       t2_t3t4c_priv[D(q,r,u,at,nocc,nvir)] += tmp2[Dtmp2(p,t,at,v,nvir)] * t3; 
                       t2_t3t4c_priv[D(p,r,u,at,nocc,nvir)] -= tmp2[Dtmp2(q,t,at,v,nvir)] * t3; 
                       t2_t3t4c_priv[D(q,r,t,at,nocc,nvir)] -= tmp2[Dtmp2(p,u,at,v,nvir)] * t3; 
                       t2_t3t4c_priv[D(p,r,t,at,nocc,nvir)] += tmp2[Dtmp2(q,u,at,v,nvir)] * t3; 
                       t2_t3t4c_priv[D(r,q,v,at,nocc,nvir)] += tmp2[Dtmp2(p,t,at,u,nvir)] * t3; 
                       t2_t3t4c_priv[D(r,p,v,at,nocc,nvir)] -= tmp2[Dtmp2(q,t,at,u,nvir)] * t3; 
                       t2_t3t4c_priv[D(r,q,v,at,nocc,nvir)] -= tmp2[Dtmp2(p,u,at,t,nvir)] * t3; 
                       t2_t3t4c_priv[D(r,p,v,at,nocc,nvir)] += tmp2[Dtmp2(q,u,at,t,nvir)] * t3; 
    
                       t2_t3t4c_priv[D(r,q,at,u,nocc,nvir)] += tmp2[Dtmp2(p,t,at,v,nvir)] * t3; 
                       t2_t3t4c_priv[D(r,p,at,u,nocc,nvir)] -= tmp2[Dtmp2(q,t,at,v,nvir)] * t3; 
                       t2_t3t4c_priv[D(r,q,at,t,nocc,nvir)] -= tmp2[Dtmp2(p,u,at,v,nvir)] * t3; 
                       t2_t3t4c_priv[D(r,p,at,t,nocc,nvir)] += tmp2[Dtmp2(q,u,at,v,nvir)] * t3; 
                       t2_t3t4c_priv[D(q,r,at,v,nocc,nvir)] += tmp2[Dtmp2(p,t,at,u,nvir)] * t3; 
                       t2_t3t4c_priv[D(p,r,at,v,nocc,nvir)] -= tmp2[Dtmp2(q,t,at,u,nvir)] * t3; 
                       t2_t3t4c_priv[D(q,r,at,v,nocc,nvir)] -= tmp2[Dtmp2(p,u,at,t,nvir)] * t3; 
                       t2_t3t4c_priv[D(p,r,at,v,nocc,nvir)] += tmp2[Dtmp2(q,u,at,t,nvir)] * t3; 
                   }
    
                   t2_t3t4c_priv[D(q,r,u,v,nocc,nvir)] += tmp3[S(p,t,nvir)] * t3; 
                   t2_t3t4c_priv[D(p,r,u,v,nocc,nvir)] -= tmp3[S(q,t,nvir)] * t3; 
                   t2_t3t4c_priv[D(q,r,t,v,nocc,nvir)] -= tmp3[S(p,u,nvir)] * t3; 
                   t2_t3t4c_priv[D(p,r,t,v,nocc,nvir)] += tmp3[S(q,u,nvir)] * t3; 
                   t2_t3t4c_priv[D(r,q,v,u,nocc,nvir)] += tmp3[S(p,t,nvir)] * t3; 
                   t2_t3t4c_priv[D(r,p,v,u,nocc,nvir)] -= tmp3[S(q,t,nvir)] * t3; 
                   t2_t3t4c_priv[D(r,q,v,t,nocc,nvir)] -= tmp3[S(p,u,nvir)] * t3; 
                   t2_t3t4c_priv[D(r,p,v,t,nocc,nvir)] += tmp3[S(q,u,nvir)] * t3; 
    
               }
           }
           fclose(fp);
        }
        else
        {
           // error message
        }

    }
#pragma omp critical
    { 
        for (it=0; it< nocc; it++){
        for (jt=0; jt< nocc; jt++){
            for (at=0; at< nvir; at++){
            for (bt=0; bt< nvir; bt++){
                t2_t3t4c[D(it,jt,at,bt,nocc,nvir)] += t2_t3t4c_priv[D(it,jt,at,bt,nocc,nvir)];
            }
            }
        }   
        }   
        free(t2_t3t4c_priv);
    } 

}

}

void t2t3c_shci_omp(double *t2_t3t4c, double *t1, double *t2aa, double *t2ab, double *tmp1, double *tmp2, double *tmp3, const int nc, const int nocc, const int nvir, const double numzero, const double c0) 
{

    const int t2size = nocc*nocc*nvir*nvir;

#pragma omp parallel default(none) \
        shared(t1, t2aa, t2ab, tmp1, tmp2, tmp3, t2_t3t4c)
{

    int p, q, r, t, u, v, itmp, it, jt, at, bt;
    double t3, parity, scale;
    uint8_t Refdet[nocc+nvir], det_str[nocc+nvir];
    for (itmp = 0; itmp < nocc+nvir; itmp++){
        if (itmp<nocc) Refdet[itmp] = 3;  
        else           Refdet[itmp] = 0;
    }

    double *t2_t3t4c_priv;
    t2_t3t4c_priv = (double *)malloc(sizeof(double) * t2size); 
    for (it=0; it< t2size; it++){
        t2_t3t4c_priv [it] = 0.0;
    }

    int i;
#pragma omp for
    for (i=0; i<omp_get_num_threads(); i++){

        char line[255], typ[4];
        //char *ptr;
        char s0[20]="t3.";
        char s1[4];

        sprintf(s1, "%d", i);
        char* filename = strcat(s0,s1);
        FILE *fp = fopen(filename, "r");
        //printf ("filename = %s\n",filename);

        fp = fopen(filename, "r");
    
        if (fp) {
           while ( !feof(fp) ){
               fscanf(fp, "%c%c%c,%s\n", &(typ[0]), &(typ[1]), &(typ[2]), line);
               fscanf(fp, "%lf\n", &t3);
               if (strncmp(typ, "aab", 3) == 0 && fabs(t3) > numzero){
                   sscanf(line,"%d,%d,%d,%d,%d,%d",&p,&q,&t,&u,&r,&v);
                   p += nc;
                   q += nc;
                   r += nc;
                   t += - nocc + nc;
                   u += - nocc + nc;
                   v += - nocc + nc;
    
                   for (itmp = 0; itmp < nocc+nvir; itmp++)
                       det_str[itmp] = Refdet[itmp];  
                   det_str[p] = 2;
                   det_str[q] = 2;
                   det_str[t+nocc] = 1;
                   det_str[u+nocc] = 1;
       
                   if (p != r && q != r) det_str[r] = 1;
                   else  det_str[r] = 0;
                   if (t != v && u != v) det_str[v+nocc] = 2;
                   else  det_str[v+nocc] = 3;
                   //parity  = parity_ab_str(det_str, nocc+nvir);
                   parity = parity_ci_to_cc(p+q, 2, nocc);
                   parity *= parity_ci_to_cc(r, 1, nocc);
       
                   // interm norm of c3
                   t3 = parity * t3 / c0;
       
                   // extract t3 
                   t3-= t1xt2aab(p, q, r, t, u, v, nocc, nvir, t1, t2aa, t2ab); 
                   t3-= t1xt1xt1aab(p, q, r, t, u, v, nocc, nvir, t1); 
   
                   for (it=0; it< nocc; it++){
                       t2_t3t4c_priv[D(it,r,u,v,nocc,nvir)] -= tmp1[Dtmp1(q,it,p,t,nocc,nvir)] * t3; 
                       t2_t3t4c_priv[D(it,r,u,v,nocc,nvir)] += tmp1[Dtmp1(p,it,q,t,nocc,nvir)] * t3; 
                       t2_t3t4c_priv[D(it,r,t,v,nocc,nvir)] += tmp1[Dtmp1(q,it,p,u,nocc,nvir)] * t3; 
                       t2_t3t4c_priv[D(it,r,t,v,nocc,nvir)] -= tmp1[Dtmp1(p,it,q,u,nocc,nvir)] * t3; 
                       t2_t3t4c_priv[D(it,q,v,u,nocc,nvir)] -= tmp1[Dtmp1(r,it,p,t,nocc,nvir)] * t3; 
                       t2_t3t4c_priv[D(it,p,v,u,nocc,nvir)] += tmp1[Dtmp1(r,it,q,t,nocc,nvir)] * t3; 
                       t2_t3t4c_priv[D(it,q,v,t,nocc,nvir)] += tmp1[Dtmp1(r,it,p,u,nocc,nvir)] * t3; 
                       t2_t3t4c_priv[D(it,p,v,t,nocc,nvir)] -= tmp1[Dtmp1(r,it,q,u,nocc,nvir)] * t3; 
    
                       t2_t3t4c_priv[D(q,it,u,v,nocc,nvir)] -= tmp1[Dtmp1(r,it,p,t,nocc,nvir)] * t3; 
                       t2_t3t4c_priv[D(p,it,u,v,nocc,nvir)] += tmp1[Dtmp1(r,it,q,t,nocc,nvir)] * t3; 
                       t2_t3t4c_priv[D(q,it,t,v,nocc,nvir)] += tmp1[Dtmp1(r,it,p,u,nocc,nvir)] * t3; 
                       t2_t3t4c_priv[D(p,it,t,v,nocc,nvir)] -= tmp1[Dtmp1(r,it,q,u,nocc,nvir)] * t3; 
                       t2_t3t4c_priv[D(r,it,v,u,nocc,nvir)] -= tmp1[Dtmp1(q,it,p,t,nocc,nvir)] * t3; 
                       t2_t3t4c_priv[D(r,it,v,u,nocc,nvir)] += tmp1[Dtmp1(p,it,q,t,nocc,nvir)] * t3; 
                       t2_t3t4c_priv[D(r,it,v,t,nocc,nvir)] += tmp1[Dtmp1(q,it,p,u,nocc,nvir)] * t3; 
                       t2_t3t4c_priv[D(r,it,v,t,nocc,nvir)] -= tmp1[Dtmp1(p,it,q,u,nocc,nvir)] * t3; 
                   }
    
                   for (at = 0; at < nvir; at++){
                       t2_t3t4c_priv[D(q,r,u,at,nocc,nvir)] += tmp2[Dtmp2(p,t,at,v,nvir)] * t3; 
                       t2_t3t4c_priv[D(p,r,u,at,nocc,nvir)] -= tmp2[Dtmp2(q,t,at,v,nvir)] * t3; 
                       t2_t3t4c_priv[D(q,r,t,at,nocc,nvir)] -= tmp2[Dtmp2(p,u,at,v,nvir)] * t3; 
                       t2_t3t4c_priv[D(p,r,t,at,nocc,nvir)] += tmp2[Dtmp2(q,u,at,v,nvir)] * t3; 
                       t2_t3t4c_priv[D(r,q,v,at,nocc,nvir)] += tmp2[Dtmp2(p,t,at,u,nvir)] * t3; 
                       t2_t3t4c_priv[D(r,p,v,at,nocc,nvir)] -= tmp2[Dtmp2(q,t,at,u,nvir)] * t3; 
                       t2_t3t4c_priv[D(r,q,v,at,nocc,nvir)] -= tmp2[Dtmp2(p,u,at,t,nvir)] * t3; 
                       t2_t3t4c_priv[D(r,p,v,at,nocc,nvir)] += tmp2[Dtmp2(q,u,at,t,nvir)] * t3; 
    
                       t2_t3t4c_priv[D(r,q,at,u,nocc,nvir)] += tmp2[Dtmp2(p,t,at,v,nvir)] * t3; 
                       t2_t3t4c_priv[D(r,p,at,u,nocc,nvir)] -= tmp2[Dtmp2(q,t,at,v,nvir)] * t3; 
                       t2_t3t4c_priv[D(r,q,at,t,nocc,nvir)] -= tmp2[Dtmp2(p,u,at,v,nvir)] * t3; 
                       t2_t3t4c_priv[D(r,p,at,t,nocc,nvir)] += tmp2[Dtmp2(q,u,at,v,nvir)] * t3; 
                       t2_t3t4c_priv[D(q,r,at,v,nocc,nvir)] += tmp2[Dtmp2(p,t,at,u,nvir)] * t3; 
                       t2_t3t4c_priv[D(p,r,at,v,nocc,nvir)] -= tmp2[Dtmp2(q,t,at,u,nvir)] * t3; 
                       t2_t3t4c_priv[D(q,r,at,v,nocc,nvir)] -= tmp2[Dtmp2(p,u,at,t,nvir)] * t3; 
                       t2_t3t4c_priv[D(p,r,at,v,nocc,nvir)] += tmp2[Dtmp2(q,u,at,t,nvir)] * t3; 
                   }
    
                   t2_t3t4c_priv[D(q,r,u,v,nocc,nvir)] += tmp3[S(p,t,nvir)] * t3; 
                   t2_t3t4c_priv[D(p,r,u,v,nocc,nvir)] -= tmp3[S(q,t,nvir)] * t3; 
                   t2_t3t4c_priv[D(q,r,t,v,nocc,nvir)] -= tmp3[S(p,u,nvir)] * t3; 
                   t2_t3t4c_priv[D(p,r,t,v,nocc,nvir)] += tmp3[S(q,u,nvir)] * t3; 
                   t2_t3t4c_priv[D(r,q,v,u,nocc,nvir)] += tmp3[S(p,t,nvir)] * t3; 
                   t2_t3t4c_priv[D(r,p,v,u,nocc,nvir)] -= tmp3[S(q,t,nvir)] * t3; 
                   t2_t3t4c_priv[D(r,q,v,t,nocc,nvir)] -= tmp3[S(p,u,nvir)] * t3; 
                   t2_t3t4c_priv[D(r,p,v,t,nocc,nvir)] += tmp3[S(q,u,nvir)] * t3; 
    
               }
           }
           fclose(fp);
        }
        else
        {
           // error message
        }

    }
#pragma omp critical
    { 
        for (it=0; it< nocc; it++){
        for (jt=0; jt< nocc; jt++){
            for (at=0; at< nvir; at++){
            for (bt=0; bt< nvir; bt++){
                t2_t3t4c[D(it,jt,at,bt,nocc,nvir)] += t2_t3t4c_priv[D(it,jt,at,bt,nocc,nvir)];
            }
            }
        }   
        }   
        free(t2_t3t4c_priv);
    } 

}

}

//void t2t3c_shci(double *t2_t3t4c, double *t1, double *t2aa, double *t2ab, double *tmp1, double *tmp2, double *tmp3, double *tmp4, double *tmp5, const int nc, const int nocc, const int nvir, const double numzero, const double c0) 
//{
//    int p, q, r, t, u, v, itmp, it, jt, at, bt;
//    double t3, parity, scale;
//    uint8_t Refdet[nocc+nvir], det_str[nocc+nvir];
//    for (itmp = 0; itmp < nocc+nvir; itmp++){
//        if (itmp<nocc) Refdet[itmp] = 3;  
//        else           Refdet[itmp] = 0;
//    }
//
//    double ****tmp;
//    tmp = (double ****)malloc(sizeof(double ***) * nocc); 
//    for (it=0; it< nocc; it++){
//        tmp[it] = (double ***)malloc(sizeof(double **) * nocc);
//        for (jt=0; jt< nocc; jt++){
//            tmp[it][jt] = (double **)malloc(sizeof(double *) * nvir);
//            for (at=0; at< nvir; at++){
//                tmp[it][jt][at] = (double *)malloc(sizeof(double) * nvir);
//            }
//        }
//    }
//
//    FILE *fp;
//    char line[255], typ[4], tmpc[255];
//    char *ptr;
//    fp = fopen("CIcoeff_shci.out", "r");
//    fgets(line, 255, fp);
//
////    //test
////    int kt;
////    printf("tmp3 in c\n");
////    for (it=0; it< nocc; it++)
////    for (at=0; at< nvir; at++)
////    for (bt=0; bt< nvir; bt++)
////    for (ct=0; ct< nvir; ct++)
////        printf("%d %d %d %d %15.8f \n",it,at,bt,ct,tmp3[Dtmp34(it,at,bt,ct,nocc,nvir)]);
//
//    if (fp) {
//       while ( !feof(fp) ){
//           fgets(line, 255, fp);
//           ptr = strtok(line, ",");
//
//           it=0;
//           while(ptr != NULL){
//               if(it==0){
//                   strcpy(typ, ptr);
//               }
//               if (it>0 && strlen(typ)==3 && strncmp(typ, "aab", 3)==0){
//                   strcpy(tmpc, ptr);
//                   if(it==1) p = atoi(tmpc);
//                   if(it==2) q = atoi(tmpc);
//                   if(it==3) t = atoi(tmpc);
//                   if(it==4) u = atoi(tmpc);
//                   if(it==5) r = atoi(tmpc);
//                   if(it==6) v = atoi(tmpc);
//                   if(it==7) t3= atof(tmpc);
//               }
//               if (it>7) break;
//               ptr = strtok(NULL, ",");
//               it++;
//           }
//
//           if (strlen(typ)==3 && strncmp(typ, "aab", 3) == 0 && fabs(t3) > numzero){
//               //printf("'%s, %d, %d, %d, %d, %d, %d, %15.8f'\n", typ, p,q,t,u,r,v,t3);
//
//               p += nc;
//               q += nc;
//               r += nc;
//               t += - nocc + nc;
//               u += - nocc + nc;
//               v += - nocc + nc;
//
//               //lsh test
//               if(!(p == 2 && q == 3 && r == 3 && \
//                    t == 0 && u == 1 && v == 1)) continue;
//               //printf("c3 in OTF: %15.8f\n",t3);
//
//               for (itmp = 0; itmp < nocc+nvir; itmp++)
//                   det_str[itmp] = Refdet[itmp];  
//               det_str[p] = 2;
//               det_str[q] = 2;
//               det_str[t+nocc] = 1;
//               det_str[u+nocc] = 1;
//   
//               if (p != r && q != r) det_str[r] = 1;
//               else  det_str[r] = 0;
//               if (t != v && u != v) det_str[v+nocc] = 2;
//               else  det_str[v+nocc] = 3;
//               parity  = parity_ab_str(det_str, nocc+nvir);
//               parity *= parity_ci_to_cc(p+q, 2, nocc);
//               parity *= parity_ci_to_cc(r, 1, nocc);
//   
//               // interm norm of c3
//               t3 = parity * t3 / c0;
//   
//               // extract t3 
//               t3-= t1xt2aab(p, q, r, t, u, v, nocc, nvir, t1, t2aa, t2ab); 
//               t3-= t1xt1xt1aab(p, q, r, t, u, v, nocc, nvir, t1); 
//               //printf("t3 in OTF: %15.8f\n",t3);
//
////               for (it=0; it< nocc; it++)
////               for (jt=0; jt< nocc; jt++)
////               for (at=0; at< nvir; at++)
////               for (bt=0; bt< nvir; bt++)
////                   tmp[it][jt][at][bt] = 0.0;
//
//               for (it=0; it< nocc; it++){
////                            t2_t3t4c[D(it,r,u,v,nocc,nvir)] += tmp1[Dtmp1(p,it,q,t,nocc,nvir)] * t3; 
////                            t2_t3t4c[D(it,r,t,v,nocc,nvir)] -= tmp1[Dtmp1(p,it,q,u,nocc,nvir)] * t3; 
////                   if (r<p) t2_t3t4c[D(it,q,v,u,nocc,nvir)] -= tmp1[Dtmp1(r,it,p,t,nocc,nvir)] * t3; 
////                   if (r<q) t2_t3t4c[D(it,p,v,u,nocc,nvir)] += tmp1[Dtmp1(r,it,q,t,nocc,nvir)] * t3; 
////                   if (r<p) t2_t3t4c[D(it,q,v,t,nocc,nvir)] += tmp1[Dtmp1(r,it,p,u,nocc,nvir)] * t3; 
////                   if (r<q) t2_t3t4c[D(it,p,v,t,nocc,nvir)] -= tmp1[Dtmp1(r,it,q,u,nocc,nvir)] * t3; 
////
////                            t2_t3t4c[D(r,it,u,v,nocc,nvir)] += tmp1[Dtmp1(p,it,q,t,nocc,nvir)] * t3; 
////                            t2_t3t4c[D(r,it,t,v,nocc,nvir)] -= tmp1[Dtmp1(p,it,q,u,nocc,nvir)] * t3; 
////                   if (r<p) t2_t3t4c[D(q,it,v,u,nocc,nvir)] -= tmp1[Dtmp1(r,it,p,t,nocc,nvir)] * t3; 
////                   if (r<q) t2_t3t4c[D(p,it,v,u,nocc,nvir)] += tmp1[Dtmp1(r,it,q,t,nocc,nvir)] * t3; 
////                   if (r<p) t2_t3t4c[D(q,it,v,t,nocc,nvir)] += tmp1[Dtmp1(r,it,p,u,nocc,nvir)] * t3; 
////                   if (r<q) t2_t3t4c[D(p,it,v,t,nocc,nvir)] -= tmp1[Dtmp1(r,it,q,u,nocc,nvir)] * t3; 
////
////                   scale = 0.5;
////                   if (r==p) t2_t3t4c[D(it,q,v,u,nocc,nvir)] -= tmp1[Dtmp1(r,it,p,t,nocc,nvir)] * t3 * scale; 
////                   if (r==q) t2_t3t4c[D(it,p,v,u,nocc,nvir)] += tmp1[Dtmp1(r,it,q,t,nocc,nvir)] * t3 * scale; 
////                   if (r==p) t2_t3t4c[D(it,q,v,t,nocc,nvir)] += tmp1[Dtmp1(r,it,p,u,nocc,nvir)] * t3 * scale; 
////                   if (r==q) t2_t3t4c[D(it,p,v,t,nocc,nvir)] -= tmp1[Dtmp1(r,it,q,u,nocc,nvir)] * t3 * scale; 
////                   if (r==p) t2_t3t4c[D(q,it,v,u,nocc,nvir)] -= tmp1[Dtmp1(r,it,p,t,nocc,nvir)] * t3 * scale; 
////                   if (r==q) t2_t3t4c[D(p,it,v,u,nocc,nvir)] += tmp1[Dtmp1(r,it,q,t,nocc,nvir)] * t3 * scale; 
////                   if (r==p) t2_t3t4c[D(q,it,v,t,nocc,nvir)] += tmp1[Dtmp1(r,it,p,u,nocc,nvir)] * t3 * scale; 
////                   if (r==q) t2_t3t4c[D(p,it,v,t,nocc,nvir)] -= tmp1[Dtmp1(r,it,q,u,nocc,nvir)] * t3 * scale; 
////
////                            t2_t3t4c[D(it,r,u,v,nocc,nvir)] -= tmp2[Dtmp2(p,t,q,it,nocc,nvir)] * t3; 
////                            t2_t3t4c[D(it,r,t,v,nocc,nvir)] += tmp2[Dtmp2(p,u,q,it,nocc,nvir)] * t3; 
////                   if (p<r) t2_t3t4c[D(it,q,v,u,nocc,nvir)] -= tmp2[Dtmp2(p,t,r,it,nocc,nvir)] * t3; 
////                   if (q<r) t2_t3t4c[D(it,p,v,u,nocc,nvir)] += tmp2[Dtmp2(q,t,r,it,nocc,nvir)] * t3; 
////                   if (p<r) t2_t3t4c[D(it,q,v,t,nocc,nvir)] += tmp2[Dtmp2(p,u,r,it,nocc,nvir)] * t3; 
////                   if (q<r) t2_t3t4c[D(it,p,v,t,nocc,nvir)] -= tmp2[Dtmp2(q,u,r,it,nocc,nvir)] * t3; 
////
////                            t2_t3t4c[D(r,it,u,v,nocc,nvir)] -= tmp2[Dtmp2(p,t,q,it,nocc,nvir)] * t3; 
////                            t2_t3t4c[D(r,it,t,v,nocc,nvir)] += tmp2[Dtmp2(p,u,q,it,nocc,nvir)] * t3; 
////                   if (p<r) t2_t3t4c[D(q,it,v,u,nocc,nvir)] -= tmp2[Dtmp2(p,t,r,it,nocc,nvir)] * t3; 
////                   if (q<r) t2_t3t4c[D(p,it,v,u,nocc,nvir)] += tmp2[Dtmp2(q,t,r,it,nocc,nvir)] * t3; 
////                   if (p<r) t2_t3t4c[D(q,it,v,t,nocc,nvir)] += tmp2[Dtmp2(p,u,r,it,nocc,nvir)] * t3; 
////                   if (q<r) t2_t3t4c[D(p,it,v,t,nocc,nvir)] -= tmp2[Dtmp2(q,u,r,it,nocc,nvir)] * t3; 
////
////                   scale = 0.5;
////                   if (p==r) t2_t3t4c[D(it,q,v,u,nocc,nvir)] -= tmp2[Dtmp2(p,t,r,it,nocc,nvir)] * t3 * scale; 
////                   if (q==r) t2_t3t4c[D(it,p,v,u,nocc,nvir)] += tmp2[Dtmp2(q,t,r,it,nocc,nvir)] * t3 * scale; 
////                   if (p==r) t2_t3t4c[D(it,q,v,t,nocc,nvir)] += tmp2[Dtmp2(p,u,r,it,nocc,nvir)] * t3 * scale; 
////                   if (q==r) t2_t3t4c[D(it,p,v,t,nocc,nvir)] -= tmp2[Dtmp2(q,u,r,it,nocc,nvir)] * t3 * scale; 
////                   if (p==r) t2_t3t4c[D(q,it,v,u,nocc,nvir)] -= tmp2[Dtmp2(p,t,r,it,nocc,nvir)] * t3 * scale; 
////                   if (q==r) t2_t3t4c[D(p,it,v,u,nocc,nvir)] += tmp2[Dtmp2(q,t,r,it,nocc,nvir)] * t3 * scale; 
////                   if (p==r) t2_t3t4c[D(q,it,v,t,nocc,nvir)] += tmp2[Dtmp2(p,u,r,it,nocc,nvir)] * t3 * scale; 
////                   if (q==r) t2_t3t4c[D(p,it,v,t,nocc,nvir)] -= tmp2[Dtmp2(q,u,r,it,nocc,nvir)] * t3 * scale; 
//
//               }
//
//               for (at = 0; at < nvir; at++){
//                             t2_t3t4c[D(q,r,at,v,nocc,nvir)] += tmp3[Dtmp34(p,t,at,u,nvir)] * t3; 
//                             t2_t3t4c[D(p,r,at,v,nocc,nvir)] -= tmp3[Dtmp34(q,t,at,u,nvir)] * t3; 
//                   if (t<v)  t2_t3t4c[D(r,q,at,u,nocc,nvir)] += tmp3[Dtmp34(p,t,at,v,nvir)] * t3; 
//                   if (t<v)  t2_t3t4c[D(r,p,at,u,nocc,nvir)] -= tmp3[Dtmp34(q,t,at,v,nvir)] * t3; 
//                   if (u<v)  t2_t3t4c[D(r,q,at,t,nocc,nvir)] -= tmp3[Dtmp34(p,u,at,v,nvir)] * t3; 
//                   if (u<v)  t2_t3t4c[D(r,p,at,t,nocc,nvir)] += tmp3[Dtmp34(q,u,at,v,nvir)] * t3; 
//
//                            t2_t3t4c[D(q,r,v,at,nocc,nvir)] += tmp3[Dtmp34(p,t,at,u,nvir)] * t3; 
//                            t2_t3t4c[D(p,r,v,at,nocc,nvir)] -= tmp3[Dtmp34(q,t,at,u,nvir)] * t3; 
//                   if (t<v) t2_t3t4c[D(r,q,u,at,nocc,nvir)] += tmp3[Dtmp34(p,t,at,v,nvir)] * t3; 
//                   if (t<v) t2_t3t4c[D(r,p,u,at,nocc,nvir)] -= tmp3[Dtmp34(q,t,at,v,nvir)] * t3; 
//                   if (u<v) t2_t3t4c[D(r,q,t,at,nocc,nvir)] -= tmp3[Dtmp34(p,u,at,v,nvir)] * t3; 
//                   if (u<v) t2_t3t4c[D(r,p,t,at,nocc,nvir)] += tmp3[Dtmp34(q,u,at,v,nvir)] * t3; 
//
//                   scale = 0.5;
//                   if (t==v) t2_t3t4c[D(r,q,at,u,nocc,nvir)] += tmp3[Dtmp34(p,t,at,v,nvir)] * t3 * scale; 
//                   if (t==v) t2_t3t4c[D(r,p,at,u,nocc,nvir)] -= tmp3[Dtmp34(q,t,at,v,nvir)] * t3 * scale; 
//                   if (u==v) t2_t3t4c[D(r,q,at,t,nocc,nvir)] -= tmp3[Dtmp34(p,u,at,v,nvir)] * t3 * scale; 
//                   if (u==v) t2_t3t4c[D(r,p,at,t,nocc,nvir)] += tmp3[Dtmp34(q,u,at,v,nvir)] * t3 * scale; 
//                   if (t==v) t2_t3t4c[D(r,q,u,at,nocc,nvir)] += tmp3[Dtmp34(p,t,at,v,nvir)] * t3 * scale; 
//                   if (t==v) t2_t3t4c[D(r,p,u,at,nocc,nvir)] -= tmp3[Dtmp34(q,t,at,v,nvir)] * t3 * scale; 
//                   if (u==v) t2_t3t4c[D(r,q,t,at,nocc,nvir)] -= tmp3[Dtmp34(p,u,at,v,nvir)] * t3 * scale; 
//                   if (u==v) t2_t3t4c[D(r,p,t,at,nocc,nvir)] += tmp3[Dtmp34(q,u,at,v,nvir)] * t3 * scale; 
//
//                            t2_t3t4c[D(q,r,at,v,nocc,nvir)] -= tmp4[Dtmp34(p,u,at,t,nvir)] * t3; 
//                            t2_t3t4c[D(p,r,at,v,nocc,nvir)] += tmp4[Dtmp34(q,u,at,t,nvir)] * t3; 
//                   if (v<t) t2_t3t4c[D(r,q,at,u,nocc,nvir)] += tmp4[Dtmp34(p,t,at,v,nvir)] * t3; 
//                   if (v<t) t2_t3t4c[D(r,p,at,u,nocc,nvir)] -= tmp4[Dtmp34(q,t,at,v,nvir)] * t3; 
//                   if (v<u) t2_t3t4c[D(r,q,at,t,nocc,nvir)] -= tmp4[Dtmp34(p,u,at,v,nvir)] * t3; 
//                   if (v<u) t2_t3t4c[D(r,p,at,t,nocc,nvir)] += tmp4[Dtmp34(q,u,at,v,nvir)] * t3; 
//
//                            t2_t3t4c[D(q,r,v,at,nocc,nvir)] -= tmp4[Dtmp34(p,u,at,t,nvir)] * t3; 
//                            t2_t3t4c[D(p,r,v,at,nocc,nvir)] += tmp4[Dtmp34(q,u,at,t,nvir)] * t3; 
//                   if (v<t) t2_t3t4c[D(r,q,u,at,nocc,nvir)] += tmp4[Dtmp34(p,t,at,v,nvir)] * t3; 
//                   if (v<t) t2_t3t4c[D(r,p,u,at,nocc,nvir)] -= tmp4[Dtmp34(q,t,at,v,nvir)] * t3; 
//                   if (v<u) t2_t3t4c[D(r,q,t,at,nocc,nvir)] -= tmp4[Dtmp34(p,u,at,v,nvir)] * t3; 
//                   if (v<u) t2_t3t4c[D(r,p,t,at,nocc,nvir)] += tmp4[Dtmp34(q,u,at,v,nvir)] * t3; 
//
//                   scale = 0.5;
//                   if (v==t) t2_t3t4c[D(r,q,at,u,nocc,nvir)] += tmp4[Dtmp34(p,t,at,v,nvir)] * t3 * scale; 
//                   if (v==t) t2_t3t4c[D(r,p,at,u,nocc,nvir)] -= tmp4[Dtmp34(q,t,at,v,nvir)] * t3 * scale; 
//                   if (v==u) t2_t3t4c[D(r,q,at,t,nocc,nvir)] -= tmp4[Dtmp34(p,u,at,v,nvir)] * t3 * scale; 
//                   if (v==u) t2_t3t4c[D(r,p,at,t,nocc,nvir)] += tmp4[Dtmp34(q,u,at,v,nvir)] * t3 * scale; 
//                   if (v==t) t2_t3t4c[D(r,q,u,at,nocc,nvir)] += tmp4[Dtmp34(p,t,at,v,nvir)] * t3 * scale; 
//                   if (v==t) t2_t3t4c[D(r,p,u,at,nocc,nvir)] -= tmp4[Dtmp34(q,t,at,v,nvir)] * t3 * scale; 
//                   if (v==u) t2_t3t4c[D(r,q,t,at,nocc,nvir)] -= tmp4[Dtmp34(p,u,at,v,nvir)] * t3 * scale; 
//                   if (v==u) t2_t3t4c[D(r,p,t,at,nocc,nvir)] += tmp4[Dtmp34(q,u,at,v,nvir)] * t3 * scale; 
//               }
//
////               t2_t3t4c[D(q,r,u,v,nocc,nvir)] += tmp5[S(p,t,nvir)] * t3; 
////               t2_t3t4c[D(p,r,u,v,nocc,nvir)] -= tmp5[S(q,t,nvir)] * t3; 
////               t2_t3t4c[D(q,r,t,v,nocc,nvir)] -= tmp5[S(p,u,nvir)] * t3; 
////               t2_t3t4c[D(p,r,t,v,nocc,nvir)] += tmp5[S(q,u,nvir)] * t3; 
////               t2_t3t4c[D(r,q,v,u,nocc,nvir)] += tmp5[S(p,t,nvir)] * t3; 
////               t2_t3t4c[D(r,p,v,u,nocc,nvir)] -= tmp5[S(q,t,nvir)] * t3; 
////               t2_t3t4c[D(r,q,v,t,nocc,nvir)] -= tmp5[S(p,u,nvir)] * t3; 
////               t2_t3t4c[D(r,p,v,t,nocc,nvir)] += tmp5[S(q,u,nvir)] * t3; 
//
//           }
//       }
//       fclose(fp);
//    }
//    else
//    {
//       // error message
//    }
//
//
////    for (it=0; it< nocc; it++)
////    for (jt=0; jt< nocc; jt++)
////    for (at=0; at< nvir; at++)
////    for (bt=0; bt< nvir; bt++)
////        tmp[it][jt][at][bt] = t2_t3t4c[D(it,jt,at,bt,nocc,nvir)] + t2_t3t4c[D(jt,it,at,bt,nocc,nvir)];
////
////    for (it=0; it< nocc; it++)
////    for (jt=0; jt< nocc; jt++)
////    for (at=0; at< nvir; at++)
////    for (bt=0; bt< nvir; bt++)
////        t2_t3t4c[D(it,jt,at,bt,nocc,nvir)] = tmp[it][jt][at][bt];
//
//    for (it=0; it< nocc; it++){
//        for (jt=0; jt< nocc; jt++){
//            for (at=0; at< nvir; at++){
//                free(tmp[it][jt][at]);
//            }
//            free(tmp[it][jt]);
//        }
//        free(tmp[it]);
//    }   
//    free(tmp);
//
//}

void c1_to_t1_mem(double *t1, double *c1, int nocc_cas, int nvir_cas, int nvir_corr, int nocc_iact) 
{
    int i, a, ia_c, ia_t;
    ia_c = -1;
    for (a = 0; a < nvir_cas; a++) {
    for (i = nocc_cas-1; i > -1; i--) {
        ia_c += 1;
        ia_t  = (i+nocc_iact) * nvir_corr + a;
        t1[ia_t] = c1[ia_c];
    }
    }
}

void c2_to_t2_mem(double *t2aa, double *t2ab, double *c2aa, double *c2ab, double *t1, int nocc_cas, int nvir_cas, int nocc_corr, int nvir_corr, int nocc_iact, double numzero) 
{
    int i, j, a, b, ijab_c, ijab_t1, ijab_t2, ijab_t3, ijab_t4;
    int ia, jb, iajb_c, ijab_t;
    double tmp;

    // t2aa
    ijab_c = -1;
    for (b = 1; b < nvir_cas; b++) {
    for (a = 0; a < b; a++) {
    for (j = nocc_cas-1; j > 0; j--) {
    for (i = j-1; i > -1; i--) {
        ijab_c += 1;
        ijab_t1 = (((i+nocc_iact)*nocc_corr+j+nocc_iact)*nvir_corr+a)*nvir_corr+b;
        ijab_t2 = (((i+nocc_iact)*nocc_corr+j+nocc_iact)*nvir_corr+b)*nvir_corr+a;
        ijab_t3 = (((j+nocc_iact)*nocc_corr+i+nocc_iact)*nvir_corr+a)*nvir_corr+b;
        ijab_t4 = (((j+nocc_iact)*nocc_corr+i+nocc_iact)*nvir_corr+b)*nvir_corr+a;

        tmp = c2aa[ijab_c]; 
        if(fabs(tmp) > numzero) 
        {
            tmp -= t1xt1aa (i+nocc_iact, j+nocc_iact, a, b, nocc_corr, nvir_corr, t1); 
            t2aa[ijab_t1] =  tmp;
            t2aa[ijab_t2] = -tmp;
            t2aa[ijab_t3] = -tmp;
            t2aa[ijab_t4] =  tmp;
        }
    }
    }
    }
    }

    ia = -1;
    for (a = 0; a < nvir_cas; a++) {
    for (i = nocc_cas-1; i > -1; i--) {
        ia += 1;
        jb  =-1;
        for (b = 0; b < nvir_cas; b++) {
        for (j = nocc_cas-1; j > -1; j--) {
            jb += 1;
            iajb_c = ia * nocc_cas*nvir_cas + jb;
            ijab_t = (((i+nocc_iact)*nocc_corr+j+nocc_iact)*nvir_corr+a)*nvir_corr+b;

            tmp = c2ab[iajb_c]; 
            if(fabs(tmp) > numzero) 
            {
                tmp -= t1xt1ab (i+nocc_iact, j+nocc_iact, a, b, nocc_corr, nvir_corr, t1); 
                t2ab[ijab_t] = tmp;
            } 
        }
        }
    }
    }

}

void c2_to_t2_u_mem(double *t2aa, double *t2ab, double *t2bb, double *c2aa, double *c2ab, double *c2bb, double *t1a, double *t1b, int nocca_cas, int nvira_cas, int noccb_cas, int nvirb_cas, int nocca_corr, int nvira_corr, int noccb_corr, int nvirb_corr, int nocca_iact, int noccb_iact, double numzero) 
{
    int i, j, a, b, ijab_c, ijab_t1, ijab_t2, ijab_t3, ijab_t4;
    int ia, jb, iajb_c, ijab_t;
    double tmp;

    // t2aa
    ijab_c = -1;
    for (b = 1; b < nvira_cas; b++) {
    for (a = 0; a < b; a++) {
    for (j = nocca_cas-1; j > 0; j--) {
    for (i = j-1; i > -1; i--) {
        ijab_c += 1;
        ijab_t1 = (((i+nocca_iact)*nocca_corr+j+nocca_iact)*nvira_corr+a)*nvira_corr+b;
        ijab_t2 = (((i+nocca_iact)*nocca_corr+j+nocca_iact)*nvira_corr+b)*nvira_corr+a;
        ijab_t3 = (((j+nocca_iact)*nocca_corr+i+nocca_iact)*nvira_corr+a)*nvira_corr+b;
        ijab_t4 = (((j+nocca_iact)*nocca_corr+i+nocca_iact)*nvira_corr+b)*nvira_corr+a;

        tmp = c2aa[ijab_c]; 
        if(fabs(tmp) > numzero) 
        {
            tmp -= t1xt1aa (i+nocca_iact, j+nocca_iact, a, b, nocca_corr, nvira_corr, t1a); 
            t2aa[ijab_t1] =  tmp;
            t2aa[ijab_t2] = -tmp;
            t2aa[ijab_t3] = -tmp;
            t2aa[ijab_t4] =  tmp;
        }
    }
    }
    }
    }
    // t2ab
    ia = -1;
    for (a = 0; a < nvira_cas; a++) {
    for (i = nocca_cas-1; i > -1; i--) {
        ia += 1;
        jb  =-1;
        for (b = 0; b < nvirb_cas; b++) {
        for (j = noccb_cas-1; j > -1; j--) {
            jb += 1;
            iajb_c = ia * noccb_cas*nvirb_cas + jb;
            ijab_t = (((i+nocca_iact)*noccb_corr+j+noccb_iact)*nvira_corr+a)*nvirb_corr+b;

            tmp = c2ab[iajb_c]; 
            if(fabs(tmp) > numzero) 
            {
                tmp -= t1xt1ab_u (i+nocca_iact, j+noccb_iact, a, b, nvira_corr, nvirb_corr, t1a, t1b); 
                t2ab[ijab_t] = tmp;
            } 
        }
        }
    }
    }
    // t2bb
    ijab_c = -1;
    for (b = 1; b < nvirb_cas; b++) {
    for (a = 0; a < b; a++) {
    for (j = noccb_cas-1; j > 0; j--) {
    for (i = j-1; i > -1; i--) {
        ijab_c += 1;
        ijab_t1 = (((i+noccb_iact)*noccb_corr+j+noccb_iact)*nvirb_corr+a)*nvirb_corr+b;
        ijab_t2 = (((i+noccb_iact)*noccb_corr+j+noccb_iact)*nvirb_corr+b)*nvirb_corr+a;
        ijab_t3 = (((j+noccb_iact)*noccb_corr+i+noccb_iact)*nvirb_corr+a)*nvirb_corr+b;
        ijab_t4 = (((j+noccb_iact)*noccb_corr+i+noccb_iact)*nvirb_corr+b)*nvirb_corr+a;

        tmp = c2bb[ijab_c]; 
        if(fabs(tmp) > numzero) 
        {
            tmp -= t1xt1aa (i+noccb_iact, j+noccb_iact, a, b, noccb_corr, nvirb_corr, t1b); 
            t2bb[ijab_t1] =  tmp;
            t2bb[ijab_t2] = -tmp;
            t2bb[ijab_t3] = -tmp;
            t2bb[ijab_t4] =  tmp;
        }
    }
    }
    }
    }
}

double c3tot3aab_mem(int i, int j, int k, int a, int b, int c, int nocc_corr, int nvir_corr, int nocc_cas, int nvir_cas, int nocc_iact, int nocc2, double *t1, double *t2aa, double *t2ab, double *c3aab, double c0)
{
    double t3 = 0.0;
    t3 = c3aab[DSc(i-nocc_iact, j-nocc_iact, k-nocc_iact, a, b, c, nocc_cas, nvir_cas, nocc2)] / c0;
    t3-= t1xt2aab(i, j, k, a, b, c, nocc_corr, nvir_corr, t1, t2aa, t2ab); 
    t3-= t1xt1xt1aab(i, j, k, a, b, c, nocc_corr, nvir_corr, t1); 
    return t3;
}

double c3tot3aaa_mem(int i, int j, int k, int a, int b, int c, int nocc_corr, int nvir_corr, int nocc_cas, int nvir_cas, int nocc_iact, int nocc3, double *t1, double *t2aa, double *c3aaa, double c0)
{
    double t3 = 0.0;
    t3 = c3aaa[Tc(i-nocc_iact, j-nocc_iact, k-nocc_iact, a, b, c, nocc3)] / c0;
    t3-= t1xt2aaa (i, j, k, a, b, c, nocc_corr, nvir_corr, t1, t2aa); 
    t3-= t1xt1xt1aaa (i, j, k, a, b, c, nocc_corr, nvir_corr, t1); 

    return t3;
}

double t1xc3aabb_mem(int i, int j, int k, int l, int a, int b, int c, int d, int nocc_corr, int nvir_corr, int nocc_cas, int nvir_cas, int nocc_iact, int nocc2, double *t1, double *t2aa, double *t2ab, double *c3aab, double c0)
{
    double t1xt3 = 0.0;
    t1xt3 += t1[S(i, a, nvir_corr)] * c3tot3aab_mem(k, l, j, c, d, b, nocc_corr, nvir_corr, nocc_cas, nvir_cas, nocc_iact, nocc2, t1, t2aa, t2ab, c3aab, c0);
    t1xt3 -= t1[S(i, b, nvir_corr)] * c3tot3aab_mem(k, l, j, c, d, a, nocc_corr, nvir_corr, nocc_cas, nvir_cas, nocc_iact, nocc2, t1, t2aa, t2ab, c3aab, c0);
    t1xt3 -= t1[S(j, a, nvir_corr)] * c3tot3aab_mem(k, l, i, c, d, b, nocc_corr, nvir_corr, nocc_cas, nvir_cas, nocc_iact, nocc2, t1, t2aa, t2ab, c3aab, c0);
    t1xt3 += t1[S(j, b, nvir_corr)] * c3tot3aab_mem(k, l, i, c, d, a, nocc_corr, nvir_corr, nocc_cas, nvir_cas, nocc_iact, nocc2, t1, t2aa, t2ab, c3aab, c0);
    t1xt3 += t1[S(k, c, nvir_corr)] * c3tot3aab_mem(i, j, l, a, b, d, nocc_corr, nvir_corr, nocc_cas, nvir_cas, nocc_iact, nocc2, t1, t2aa, t2ab, c3aab, c0);
    t1xt3 -= t1[S(k, d, nvir_corr)] * c3tot3aab_mem(i, j, l, a, b, c, nocc_corr, nvir_corr, nocc_cas, nvir_cas, nocc_iact, nocc2, t1, t2aa, t2ab, c3aab, c0);
    t1xt3 -= t1[S(l, c, nvir_corr)] * c3tot3aab_mem(i, j, k, a, b, d, nocc_corr, nvir_corr, nocc_cas, nvir_cas, nocc_iact, nocc2, t1, t2aa, t2ab, c3aab, c0);
    t1xt3 += t1[S(l, d, nvir_corr)] * c3tot3aab_mem(i, j, k, a, b, c, nocc_corr, nvir_corr, nocc_cas, nvir_cas, nocc_iact, nocc2, t1, t2aa, t2ab, c3aab, c0);
    return t1xt3;
}

double t1xc3aaab_mem(int i, int j, int k, int l, int a, int b, int c, int d, int nocc_corr, int nvir_corr, int nocc_cas, int nvir_cas, int nocc_iact, int nocc2, int nocc3, double *t1, double *t2aa, double *t2ab, double *c3aaa, double *c3aab, double c0)
{
    double t1xt3 = 0.0;
    t1xt3 += t1[S(i, a, nvir_corr)] * c3tot3aab_mem(j, k, l, b, c, d, nocc_corr, nvir_corr, nocc_cas, nvir_cas, nocc_iact, nocc2, t1, t2aa, t2ab, c3aab, c0);
    t1xt3 -= t1[S(i, b, nvir_corr)] * c3tot3aab_mem(j, k, l, a, c, d, nocc_corr, nvir_corr, nocc_cas, nvir_cas, nocc_iact, nocc2, t1, t2aa, t2ab, c3aab, c0);
    t1xt3 += t1[S(i, c, nvir_corr)] * c3tot3aab_mem(j, k, l, a, b, d, nocc_corr, nvir_corr, nocc_cas, nvir_cas, nocc_iact, nocc2, t1, t2aa, t2ab, c3aab, c0);
    t1xt3 -= t1[S(j, a, nvir_corr)] * c3tot3aab_mem(i, k, l, b, c, d, nocc_corr, nvir_corr, nocc_cas, nvir_cas, nocc_iact, nocc2, t1, t2aa, t2ab, c3aab, c0);
    t1xt3 += t1[S(j, b, nvir_corr)] * c3tot3aab_mem(i, k, l, a, c, d, nocc_corr, nvir_corr, nocc_cas, nvir_cas, nocc_iact, nocc2, t1, t2aa, t2ab, c3aab, c0);
    t1xt3 -= t1[S(j, c, nvir_corr)] * c3tot3aab_mem(i, k, l, a, b, d, nocc_corr, nvir_corr, nocc_cas, nvir_cas, nocc_iact, nocc2, t1, t2aa, t2ab, c3aab, c0);
    t1xt3 += t1[S(k, a, nvir_corr)] * c3tot3aab_mem(i, j, l, b, c, d, nocc_corr, nvir_corr, nocc_cas, nvir_cas, nocc_iact, nocc2, t1, t2aa, t2ab, c3aab, c0);
    t1xt3 -= t1[S(k, b, nvir_corr)] * c3tot3aab_mem(i, j, l, a, c, d, nocc_corr, nvir_corr, nocc_cas, nvir_cas, nocc_iact, nocc2, t1, t2aa, t2ab, c3aab, c0);
    t1xt3 += t1[S(k, c, nvir_corr)] * c3tot3aab_mem(i, j, l, a, b, d, nocc_corr, nvir_corr, nocc_cas, nvir_cas, nocc_iact, nocc2, t1, t2aa, t2ab, c3aab, c0);
    t1xt3 += t1[S(l, d, nvir_corr)] * c3tot3aaa_mem(i, j, k, a, b, c, nocc_corr, nvir_corr, nocc_cas, nvir_cas, nocc_iact, nocc3, t1, t2aa, c3aaa, c0);
    return t1xt3;
}

void t2t4c_dmrg_omp_otf_mem(double *t2t4c, double *t1, double *t2aa, double *t2ab, double *c3aaa, double *c3aab, double *e2ovov, const int nocc_iact, const int nocc_corr, const int nvir_corr, const int nocc_cas, const int nvir_cas, double numzero, const double c0, double norm) 
{
    //double numzero = 1e-7;
    //numzero = 1e-3;

    const int nocc2 = (int) nocc_cas*(nocc_cas-1)/2;
    const int nocc3 = (int) nocc_cas*(nocc_cas-1)*(nocc_cas-2)/6;
    double norm0SDT = norm;

    const int t2size = nocc_corr*nocc_corr*nvir_corr*nvir_corr;

#pragma omp parallel default(none) \
        shared(t1, t2aa, t2ab, c3aaa, c3aab, e2ovov, t2t4c, norm, numzero)
{
       double t4, parity, scale;
       int p, q, r, s, t, u, v, w, itmp, it, jt, at, bt, ifile;
       char typ[4], line[255];
       uint8_t Refdet[nocc_corr+nvir_corr], det_str[nocc_corr+nvir_corr];
       for (itmp = 0; itmp < nocc_corr+nvir_corr; itmp++){
           if (itmp<nocc_corr) Refdet[itmp] = 3;  
           else                Refdet[itmp] = 0;
       }
//       double ****tmp;
//       tmp = (double ****)malloc(sizeof(double ***) * nocc_corr); 
//       for (it=0; it< nocc_corr; it++){
//           tmp[it] = (double ***)malloc(sizeof(double **) * nocc_corr);
//           for (jt=0; jt< nocc_corr; jt++){
//               tmp[it][jt] = (double **)malloc(sizeof(double *) * nvir_corr);
//               for (at=0; at< nvir_corr; at++){
//                   tmp[it][jt][at] = (double *)malloc(sizeof(double) * nvir_corr);
//               }
//           }
//       }

       double *t2t4c_priv;
       t2t4c_priv = (double *)malloc(sizeof(double) * t2size); 
       for (it=0; it< t2size; it++){
           t2t4c_priv[it] = 0.0;
       }
       //lsh test
       //printf ("num_threads = %d\n",omp_get_num_threads());

       int i;
#pragma omp for reduction(+ : norm)
       for (i=0; i<omp_get_num_threads(); i++){
           char s0[20]="t4.";
           char s1[4];
           sprintf(s1, "%d", i);
           char* filename = strcat(s0,s1);
           FILE *fp = fopen(filename, "r");
           //printf ("filename = %s\n",filename);

           if (fp) {
           while ( !feof(fp) ){

           fscanf(fp, "%c%c%c%c,%s\n", &(typ[0]), &(typ[1]), &(typ[2]), &(typ[3]), line);
           fscanf(fp, "%lf\n", &t4);
           //lsh test
           //printf ("typ=%c%c%c%c line=%s\n",typ[0],typ[1],typ[2],typ[3], line);
           if (strncmp(typ, "aabb", 4) == 0 && fabs(t4) > numzero){
               sscanf(line,"%d,%d,%d,%d,%d,%d,%d,%d",&p,&q,&t,&u,&r,&s,&v,&w);
               p += nocc_iact;
               q += nocc_iact;
               r += nocc_iact;
               s += nocc_iact;
               t += - nocc_cas;
               u += - nocc_cas;
               v += - nocc_cas;
               w += - nocc_cas;

//               if(!(p == 2 && q == 3 && t == 0 && u == 1 && \
                    r == 2 && s == 3 && v == 0 && w == 1)) continue;
//               if(!((p == 2 && q == 3 && t == 0 && u == 4 && \
//                    r == 1 && s == 3 && v == 0 && w == 4) || \
//                   (p == 1 && q == 3 && t == 0 && u == 4 && \
//                    r == 2 && s == 3 && v == 0 && w == 4)) ) continue;

//               if(!(p == 2 && q == 3 && t == 0 && u == 2 && \
//                    r == 2 && s == 3 && v == 0 && w == 2)) continue;
               norm += t4*t4;
               for (itmp = 0; itmp < nocc_corr+nvir_corr; itmp++)
                   det_str[itmp] = Refdet[itmp];  
   
               det_str[p] = 1;
               det_str[q] = 1;
               det_str[t+nocc_corr] = 2;
               det_str[u+nocc_corr] = 2;
   
               if (p != r && q != r) det_str[r] = 2;
               else  det_str[r] = 0;
               if (p != s && q != s) det_str[s] = 2;
               else  det_str[s] = 0;
               if (t != v && u != v) det_str[v+nocc_corr] = 1;
               else  det_str[v+nocc_corr] = 3;
               if (t != w && u != w) det_str[w+nocc_corr] = 1;
               else  det_str[w+nocc_corr] = 3;
   
               //parity  = parity_ab_str(det_str, nocc_corr+nvir_corr);
               parity = parity_ci_to_cc(p+q, 2, nocc_corr);
               parity *= parity_ci_to_cc(r+s, 2, nocc_corr);
   
               // interm norm of c4
               t4 = parity * t4 / c0;
               // lsh test 
   
               // extract t4 
               t4-= t1xc3aabb_mem(p, q, r, s, t, u, v, w, nocc_corr, nvir_corr, nocc_cas, nvir_cas, nocc_iact, nocc2, t1, t2aa, t2ab, c3aab, c0); 
               t4-= t2xt2aabb(p, q, r, s, t, u, v, w, nocc_corr, nvir_corr, t2aa, t2ab); 
               t4-= t1xt1xt2aabb(p, q, r, s, t, u, v, w, nocc_corr, nvir_corr, t1, t2aa, t2ab); 
               t4-= t1xt1xt1xt1aabb(p, q, r, s, t, u, v, w, nocc_corr, nvir_corr, t1);   // may have bug 
   
               t2t4c_priv[D(q,s,u,w,nocc_corr,nvir_corr)] += e2ovov[De(p,t,r,v,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(p,s,u,w,nocc_corr,nvir_corr)] -= e2ovov[De(q,t,r,v,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(q,s,t,w,nocc_corr,nvir_corr)] -= e2ovov[De(p,u,r,v,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(p,s,t,w,nocc_corr,nvir_corr)] += e2ovov[De(q,u,r,v,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(q,r,u,w,nocc_corr,nvir_corr)] -= e2ovov[De(p,t,s,v,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(p,r,u,w,nocc_corr,nvir_corr)] += e2ovov[De(q,t,s,v,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(q,r,t,w,nocc_corr,nvir_corr)] += e2ovov[De(p,u,s,v,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(p,r,t,w,nocc_corr,nvir_corr)] -= e2ovov[De(q,u,s,v,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(q,s,u,v,nocc_corr,nvir_corr)] -= e2ovov[De(p,t,r,w,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(p,s,u,v,nocc_corr,nvir_corr)] += e2ovov[De(q,t,r,w,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(q,s,t,v,nocc_corr,nvir_corr)] += e2ovov[De(p,u,r,w,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(p,s,t,v,nocc_corr,nvir_corr)] -= e2ovov[De(q,u,r,w,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(q,r,u,v,nocc_corr,nvir_corr)] += e2ovov[De(p,t,s,w,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(p,r,u,v,nocc_corr,nvir_corr)] -= e2ovov[De(q,t,s,w,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(q,r,t,v,nocc_corr,nvir_corr)] -= e2ovov[De(p,u,s,w,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(p,r,t,v,nocc_corr,nvir_corr)] += e2ovov[De(q,u,s,w,nocc_corr,nvir_corr)] * t4;

           }
           else if (strncmp(typ, "aaab", 4) == 0 && fabs(t4) > numzero){
               norm += 2.0*t4*t4; 
               //lsh test
               //printf ("typ=%c%c%c%c line=%s c4=%lf\n",typ[0],typ[1],typ[2],typ[3], line, t4);

               sscanf(line,"%d,%d,%d,%d,%d,%d,%d,%d",&p,&q,&r,&t,&u,&v,&s,&w);
               p += nocc_iact;
               q += nocc_iact;
               r += nocc_iact;
               s += nocc_iact;
               t += - nocc_cas;
               u += - nocc_cas;
               v += - nocc_cas;
               w += - nocc_cas;

               //printf ("p=%d, q=%d, r=%d, t=%d, u=%d, v=%d, s=%d, w=%d",p,q,r,t,u,v,s,w);

               for (itmp = 0; itmp < nocc_corr+nvir_corr; itmp++)
                   det_str[itmp] = Refdet[itmp];  
               det_str[p] = 2;
               det_str[q] = 2;
               det_str[r] = 2;
               det_str[t+nocc_corr] = 1;
               det_str[u+nocc_corr] = 1;
               det_str[v+nocc_corr] = 1;
   
               if (p != s && q != s && r != s) det_str[s] = 1;
               else  det_str[s] = 0;
               if (t != w && u != w && v != w) det_str[w+nocc_corr] = 2;
               else  det_str[w+nocc_corr] = 3;
               //parity  = parity_ab_str(det_str, nocc_corr+nvir_corr);
               parity = parity_ci_to_cc(p+q+r, 3, nocc_corr);
               parity *= parity_ci_to_cc(s, 1, nocc_corr);
   
               // interm norm of c4
               t4 = parity * t4 / c0;
   
               // extract t4 
               t4-= t1xc3aaab_mem (p, q, r, s, t, u, v, w, nocc_corr, nvir_corr, nocc_cas, nvir_cas, nocc_iact, nocc2, nocc3, t1, t2aa, t2ab, c3aaa, c3aab, c0);
               t4-= t2xt2aaab (p, q, r, s, t, u, v, w, nocc_corr, nvir_corr, t2aa, t2ab);         // may have 1e-3 bug 
               t4-= t1xt1xt2aaab (p, q, r, s, t, u, v, w, nocc_corr, nvir_corr, t1, t2aa, t2ab);  // may have 1e-5 bug 
               t4-= t1xt1xt1xt1aaab (p, q, r, s, t, u, v, w, nocc_corr, nvir_corr, t1);           // may have 1e-6 bug
   
               t2t4c_priv[D(r,s,v,w,nocc_corr,nvir_corr)] += e2ovov[De(p,t,q,u,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(q,s,v,w,nocc_corr,nvir_corr)] += e2ovov[De(r,t,p,u,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(p,s,v,w,nocc_corr,nvir_corr)] += e2ovov[De(q,t,r,u,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(r,s,v,w,nocc_corr,nvir_corr)] -= e2ovov[De(q,t,p,u,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(q,s,v,w,nocc_corr,nvir_corr)] -= e2ovov[De(p,t,r,u,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(p,s,v,w,nocc_corr,nvir_corr)] -= e2ovov[De(r,t,q,u,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(r,s,u,w,nocc_corr,nvir_corr)] += e2ovov[De(p,v,q,t,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(q,s,u,w,nocc_corr,nvir_corr)] += e2ovov[De(r,v,p,t,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(p,s,u,w,nocc_corr,nvir_corr)] += e2ovov[De(q,v,r,t,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(r,s,u,w,nocc_corr,nvir_corr)] -= e2ovov[De(q,v,p,t,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(q,s,u,w,nocc_corr,nvir_corr)] -= e2ovov[De(p,v,r,t,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(p,s,u,w,nocc_corr,nvir_corr)] -= e2ovov[De(r,v,q,t,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(r,s,t,w,nocc_corr,nvir_corr)] += e2ovov[De(p,u,q,v,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(q,s,t,w,nocc_corr,nvir_corr)] += e2ovov[De(r,u,p,v,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(p,s,t,w,nocc_corr,nvir_corr)] += e2ovov[De(q,u,r,v,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(r,s,t,w,nocc_corr,nvir_corr)] -= e2ovov[De(q,u,p,v,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(q,s,t,w,nocc_corr,nvir_corr)] -= e2ovov[De(p,u,r,v,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(p,s,t,w,nocc_corr,nvir_corr)] -= e2ovov[De(r,u,q,v,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(r,s,u,w,nocc_corr,nvir_corr)] -= e2ovov[De(p,t,q,v,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(q,s,u,w,nocc_corr,nvir_corr)] -= e2ovov[De(r,t,p,v,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(p,s,u,w,nocc_corr,nvir_corr)] -= e2ovov[De(q,t,r,v,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(r,s,u,w,nocc_corr,nvir_corr)] += e2ovov[De(q,t,p,v,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(q,s,u,w,nocc_corr,nvir_corr)] += e2ovov[De(p,t,r,v,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(p,s,u,w,nocc_corr,nvir_corr)] += e2ovov[De(r,t,q,v,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(r,s,t,w,nocc_corr,nvir_corr)] -= e2ovov[De(p,v,q,u,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(q,s,t,w,nocc_corr,nvir_corr)] -= e2ovov[De(r,v,p,u,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(p,s,t,w,nocc_corr,nvir_corr)] -= e2ovov[De(q,v,r,u,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(r,s,t,w,nocc_corr,nvir_corr)] += e2ovov[De(q,v,p,u,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(q,s,t,w,nocc_corr,nvir_corr)] += e2ovov[De(p,v,r,u,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(p,s,t,w,nocc_corr,nvir_corr)] += e2ovov[De(r,v,q,u,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(r,s,v,w,nocc_corr,nvir_corr)] -= e2ovov[De(p,u,q,t,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(q,s,v,w,nocc_corr,nvir_corr)] -= e2ovov[De(r,u,p,t,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(p,s,v,w,nocc_corr,nvir_corr)] -= e2ovov[De(q,u,r,t,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(r,s,v,w,nocc_corr,nvir_corr)] += e2ovov[De(q,u,p,t,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(q,s,v,w,nocc_corr,nvir_corr)] += e2ovov[De(p,u,r,t,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(p,s,v,w,nocc_corr,nvir_corr)] += e2ovov[De(r,u,q,t,nocc_corr,nvir_corr)] * t4;

//               for (it=0; it< nocc_corr; it++){
//               for (jt=0; jt< nocc_corr; jt++){
//               for (at=0; at< nvir_corr; at++){
//               for (bt=0; bt< nvir_corr; bt++){
//                   tmp[it][jt][at][bt] = 0.0;
//               }
//               }
//               }
//               }
//   
//               tmp[r][s][v][w] += (e2ovov[De(p,t,q,u,nocc_corr,nvir_corr)]-e2ovov[De(p,u,q,t,nocc_corr,nvir_corr)]) * t4; 
//               tmp[q][s][v][w] -= (e2ovov[De(p,t,r,u,nocc_corr,nvir_corr)]-e2ovov[De(p,u,r,t,nocc_corr,nvir_corr)]) * t4; 
//               tmp[p][s][v][w] += (e2ovov[De(q,t,s,u,nocc_corr,nvir_corr)]-e2ovov[De(q,u,s,t,nocc_corr,nvir_corr)]) * t4; 
//               tmp[r][s][u][w] -= (e2ovov[De(p,t,q,v,nocc_corr,nvir_corr)]-e2ovov[De(p,v,q,t,nocc_corr,nvir_corr)]) * t4; 
//               tmp[q][s][u][w] += (e2ovov[De(p,t,r,v,nocc_corr,nvir_corr)]-e2ovov[De(p,v,r,t,nocc_corr,nvir_corr)]) * t4; 
//               tmp[p][s][u][w] -= (e2ovov[De(q,t,s,v,nocc_corr,nvir_corr)]-e2ovov[De(q,v,s,t,nocc_corr,nvir_corr)]) * t4; 
//               tmp[r][s][t][w] += (e2ovov[De(p,u,q,v,nocc_corr,nvir_corr)]-e2ovov[De(p,v,q,u,nocc_corr,nvir_corr)]) * t4; 
//               tmp[q][s][t][w] -= (e2ovov[De(p,u,r,v,nocc_corr,nvir_corr)]-e2ovov[De(p,v,r,u,nocc_corr,nvir_corr)]) * t4; 
//               tmp[p][s][t][w] += (e2ovov[De(q,u,s,v,nocc_corr,nvir_corr)]-e2ovov[De(q,v,s,u,nocc_corr,nvir_corr)]) * t4; 
//   
//               for (it = 0; it < nocc_corr; it++)
//               for (jt = 0; jt < nocc_corr; jt++)
//               for (at = 0; at < nvir_corr; at++)
//               for (bt = 0; bt < nvir_corr; bt++)
//                   t2t4c_priv[D(it,jt,at,bt,nocc_corr,nvir_corr)] += tmp[it][jt][at][bt] + tmp[jt][it][bt][at];

           }

           }
           fclose(fp);
           }
       }

#pragma omp critical
       { 
           for (it=0; it< nocc_corr; it++){
               for (jt=0; jt< nocc_corr; jt++){
                   for (at=0; at< nvir_corr; at++){
                       for (bt=0; bt< nvir_corr; bt++){
                           //t2t4c[D(it,jt,at,bt,nocc_corr,nvir_corr)] += t2t4c_priv[D(it,jt,at,bt,nocc_corr,nvir_corr)];
                           t2t4c[D(it,jt,at,bt,nocc_corr,nvir_corr)] += 0.5*(t2t4c_priv[D(it,jt,at,bt,nocc_corr,nvir_corr)]+t2t4c_priv[D(jt,it,bt,at,nocc_corr,nvir_corr)]);
                       }
                   }
               }
           }   
           free(t2t4c_priv);
       } 
}

    printf ("0SDTQ (Q) = %18.16f ( %18.16f )\n", norm, norm-norm0SDT);

}

void t2t4c_shci_omp_otf_mem(double *t2t4c, double *t1, double *t2aa, double *t2ab, double *c3aaa, double *c3aab, double *e2ovov, const int nocc_iact, const int nocc_corr, const int nvir_corr, const int nocc_cas, const int nvir_cas, double numzero, const double c0, double norm) 
{
    //double numzero = 1e-7;
    //numzero = 1e-3;

    const int nocc2 = (int) nocc_cas*(nocc_cas-1)/2;
    const int nocc3 = (int) nocc_cas*(nocc_cas-1)*(nocc_cas-2)/6;
    double norm0SDT = norm;

    const int t2size = nocc_corr*nocc_corr*nvir_corr*nvir_corr;

#pragma omp parallel default(none) \
        shared(t1, t2aa, t2ab, c3aaa, c3aab, e2ovov, t2t4c, norm, numzero)
{
       double t4, parity, scale;
       int p, q, r, s, t, u, v, w, itmp, it, jt, at, bt, ifile;
       char typ[4], line[255];
       uint8_t Refdet[nocc_corr+nvir_corr], det_str[nocc_corr+nvir_corr];
       for (itmp = 0; itmp < nocc_corr+nvir_corr; itmp++){
           if (itmp<nocc_corr) Refdet[itmp] = 3;  
           else                Refdet[itmp] = 0;
       }
//       double ****tmp;
//       tmp = (double ****)malloc(sizeof(double ***) * nocc_corr); 
//       for (it=0; it< nocc_corr; it++){
//           tmp[it] = (double ***)malloc(sizeof(double **) * nocc_corr);
//           for (jt=0; jt< nocc_corr; jt++){
//               tmp[it][jt] = (double **)malloc(sizeof(double *) * nvir_corr);
//               for (at=0; at< nvir_corr; at++){
//                   tmp[it][jt][at] = (double *)malloc(sizeof(double) * nvir_corr);
//               }
//           }
//       }

       double *t2t4c_priv;
       t2t4c_priv = (double *)malloc(sizeof(double) * t2size); 
       for (it=0; it< t2size; it++){
           t2t4c_priv[it] = 0.0;
       }
       //lsh test
       //printf ("num_threads = %d\n",omp_get_num_threads());

       int i;
#pragma omp for reduction(+ : norm)
       for (i=0; i<omp_get_num_threads(); i++){
           char s0[20]="t4.";
           char s1[4];
           sprintf(s1, "%d", i);
           char* filename = strcat(s0,s1);
           FILE *fp = fopen(filename, "r");
           //printf ("filename = %s\n",filename);

           if (fp) {
           while ( !feof(fp) ){

           fscanf(fp, "%c%c%c%c,%s\n", &(typ[0]), &(typ[1]), &(typ[2]), &(typ[3]), line);
           fscanf(fp, "%lf\n", &t4);
           //lsh test
           //printf ("typ=%c%c%c%c line=%s\n",typ[0],typ[1],typ[2],typ[3], line);
           if (strncmp(typ, "aabb", 4) == 0 && fabs(t4) > numzero){
               sscanf(line,"%d,%d,%d,%d,%d,%d,%d,%d",&p,&q,&t,&u,&r,&s,&v,&w);
               p += nocc_iact;
               q += nocc_iact;
               r += nocc_iact;
               s += nocc_iact;
               t += - nocc_cas;
               u += - nocc_cas;
               v += - nocc_cas;
               w += - nocc_cas;

//               if(!(p == 2 && q == 3 && t == 0 && u == 1 && \
                    r == 2 && s == 3 && v == 0 && w == 1)) continue;
//               if(!((p == 2 && q == 3 && t == 0 && u == 4 && \
//                    r == 1 && s == 3 && v == 0 && w == 4) || \
//                   (p == 1 && q == 3 && t == 0 && u == 4 && \
//                    r == 2 && s == 3 && v == 0 && w == 4)) ) continue;

//               if(!(p == 2 && q == 3 && t == 0 && u == 2 && \
//                    r == 2 && s == 3 && v == 0 && w == 2)) continue;
               norm += t4*t4;
               for (itmp = 0; itmp < nocc_corr+nvir_corr; itmp++)
                   det_str[itmp] = Refdet[itmp];  
   
               det_str[p] = 2;
               det_str[q] = 2;
               det_str[t+nocc_corr] = 1;
               det_str[u+nocc_corr] = 1;
   
               if (p != r && q != r) det_str[r] = 1;
               else  det_str[r] = 0;
               if (p != s && q != s) det_str[s] = 1;
               else  det_str[s] = 0;
               if (t != v && u != v) det_str[v+nocc_corr] = 2;
               else  det_str[v+nocc_corr] = 3;
               if (t != w && u != w) det_str[w+nocc_corr] = 2;
               else  det_str[w+nocc_corr] = 3;
   
               //parity  = parity_ab_str(det_str, nocc_corr+nvir_corr);
               parity = parity_ci_to_cc(p+q, 2, nocc_corr);
               parity *= parity_ci_to_cc(r+s, 2, nocc_corr);
   
               // interm norm of c4
               t4 = parity * t4 / c0;
               // lsh test 
   
               // extract t4 
               t4-= t1xc3aabb_mem(p, q, r, s, t, u, v, w, nocc_corr, nvir_corr, nocc_cas, nvir_cas, nocc_iact, nocc2, t1, t2aa, t2ab, c3aab, c0); 
               t4-= t2xt2aabb(p, q, r, s, t, u, v, w, nocc_corr, nvir_corr, t2aa, t2ab); 
               t4-= t1xt1xt2aabb(p, q, r, s, t, u, v, w, nocc_corr, nvir_corr, t1, t2aa, t2ab); 
               t4-= t1xt1xt1xt1aabb(p, q, r, s, t, u, v, w, nocc_corr, nvir_corr, t1);   // may have bug 
   
               t2t4c_priv[D(q,s,u,w,nocc_corr,nvir_corr)] += e2ovov[De(p,t,r,v,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(p,s,u,w,nocc_corr,nvir_corr)] -= e2ovov[De(q,t,r,v,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(q,s,t,w,nocc_corr,nvir_corr)] -= e2ovov[De(p,u,r,v,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(p,s,t,w,nocc_corr,nvir_corr)] += e2ovov[De(q,u,r,v,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(q,r,u,w,nocc_corr,nvir_corr)] -= e2ovov[De(p,t,s,v,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(p,r,u,w,nocc_corr,nvir_corr)] += e2ovov[De(q,t,s,v,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(q,r,t,w,nocc_corr,nvir_corr)] += e2ovov[De(p,u,s,v,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(p,r,t,w,nocc_corr,nvir_corr)] -= e2ovov[De(q,u,s,v,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(q,s,u,v,nocc_corr,nvir_corr)] -= e2ovov[De(p,t,r,w,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(p,s,u,v,nocc_corr,nvir_corr)] += e2ovov[De(q,t,r,w,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(q,s,t,v,nocc_corr,nvir_corr)] += e2ovov[De(p,u,r,w,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(p,s,t,v,nocc_corr,nvir_corr)] -= e2ovov[De(q,u,r,w,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(q,r,u,v,nocc_corr,nvir_corr)] += e2ovov[De(p,t,s,w,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(p,r,u,v,nocc_corr,nvir_corr)] -= e2ovov[De(q,t,s,w,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(q,r,t,v,nocc_corr,nvir_corr)] -= e2ovov[De(p,u,s,w,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(p,r,t,v,nocc_corr,nvir_corr)] += e2ovov[De(q,u,s,w,nocc_corr,nvir_corr)] * t4;

           }
           else if (strncmp(typ, "aaab", 4) == 0 && fabs(t4) > numzero){
               norm += 2.0*t4*t4; 
               //lsh test
               //printf ("typ=%c%c%c%c line=%s c4=%lf\n",typ[0],typ[1],typ[2],typ[3], line, t4);

               sscanf(line,"%d,%d,%d,%d,%d,%d,%d,%d",&p,&q,&r,&t,&u,&v,&s,&w);
               p += nocc_iact;
               q += nocc_iact;
               r += nocc_iact;
               s += nocc_iact;
               t += - nocc_cas;
               u += - nocc_cas;
               v += - nocc_cas;
               w += - nocc_cas;

               //printf ("p=%d, q=%d, r=%d, t=%d, u=%d, v=%d, s=%d, w=%d",p,q,r,t,u,v,s,w);

               for (itmp = 0; itmp < nocc_corr+nvir_corr; itmp++)
                   det_str[itmp] = Refdet[itmp];  
               det_str[p] = 2;
               det_str[q] = 2;
               det_str[r] = 2;
               det_str[t+nocc_corr] = 1;
               det_str[u+nocc_corr] = 1;
               det_str[v+nocc_corr] = 1;
   
               if (p != s && q != s && r != s) det_str[s] = 1;
               else  det_str[s] = 0;
               if (t != w && u != w && v != w) det_str[w+nocc_corr] = 2;
               else  det_str[w+nocc_corr] = 3;
               //parity  = parity_ab_str(det_str, nocc_corr+nvir_corr);
               parity = parity_ci_to_cc(p+q+r, 3, nocc_corr);
               parity *= parity_ci_to_cc(s, 1, nocc_corr);
   
               // interm norm of c4
               t4 = parity * t4 / c0;
   
               // extract t4 
               t4-= t1xc3aaab_mem (p, q, r, s, t, u, v, w, nocc_corr, nvir_corr, nocc_cas, nvir_cas, nocc_iact, nocc2, nocc3, t1, t2aa, t2ab, c3aaa, c3aab, c0);
               t4-= t2xt2aaab (p, q, r, s, t, u, v, w, nocc_corr, nvir_corr, t2aa, t2ab);         // may have 1e-3 bug 
               t4-= t1xt1xt2aaab (p, q, r, s, t, u, v, w, nocc_corr, nvir_corr, t1, t2aa, t2ab);  // may have 1e-5 bug 
               t4-= t1xt1xt1xt1aaab (p, q, r, s, t, u, v, w, nocc_corr, nvir_corr, t1);           // may have 1e-6 bug
   
               t2t4c_priv[D(r,s,v,w,nocc_corr,nvir_corr)] += e2ovov[De(p,t,q,u,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(q,s,v,w,nocc_corr,nvir_corr)] += e2ovov[De(r,t,p,u,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(p,s,v,w,nocc_corr,nvir_corr)] += e2ovov[De(q,t,r,u,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(r,s,v,w,nocc_corr,nvir_corr)] -= e2ovov[De(q,t,p,u,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(q,s,v,w,nocc_corr,nvir_corr)] -= e2ovov[De(p,t,r,u,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(p,s,v,w,nocc_corr,nvir_corr)] -= e2ovov[De(r,t,q,u,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(r,s,u,w,nocc_corr,nvir_corr)] += e2ovov[De(p,v,q,t,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(q,s,u,w,nocc_corr,nvir_corr)] += e2ovov[De(r,v,p,t,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(p,s,u,w,nocc_corr,nvir_corr)] += e2ovov[De(q,v,r,t,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(r,s,u,w,nocc_corr,nvir_corr)] -= e2ovov[De(q,v,p,t,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(q,s,u,w,nocc_corr,nvir_corr)] -= e2ovov[De(p,v,r,t,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(p,s,u,w,nocc_corr,nvir_corr)] -= e2ovov[De(r,v,q,t,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(r,s,t,w,nocc_corr,nvir_corr)] += e2ovov[De(p,u,q,v,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(q,s,t,w,nocc_corr,nvir_corr)] += e2ovov[De(r,u,p,v,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(p,s,t,w,nocc_corr,nvir_corr)] += e2ovov[De(q,u,r,v,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(r,s,t,w,nocc_corr,nvir_corr)] -= e2ovov[De(q,u,p,v,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(q,s,t,w,nocc_corr,nvir_corr)] -= e2ovov[De(p,u,r,v,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(p,s,t,w,nocc_corr,nvir_corr)] -= e2ovov[De(r,u,q,v,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(r,s,u,w,nocc_corr,nvir_corr)] -= e2ovov[De(p,t,q,v,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(q,s,u,w,nocc_corr,nvir_corr)] -= e2ovov[De(r,t,p,v,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(p,s,u,w,nocc_corr,nvir_corr)] -= e2ovov[De(q,t,r,v,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(r,s,u,w,nocc_corr,nvir_corr)] += e2ovov[De(q,t,p,v,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(q,s,u,w,nocc_corr,nvir_corr)] += e2ovov[De(p,t,r,v,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(p,s,u,w,nocc_corr,nvir_corr)] += e2ovov[De(r,t,q,v,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(r,s,t,w,nocc_corr,nvir_corr)] -= e2ovov[De(p,v,q,u,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(q,s,t,w,nocc_corr,nvir_corr)] -= e2ovov[De(r,v,p,u,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(p,s,t,w,nocc_corr,nvir_corr)] -= e2ovov[De(q,v,r,u,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(r,s,t,w,nocc_corr,nvir_corr)] += e2ovov[De(q,v,p,u,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(q,s,t,w,nocc_corr,nvir_corr)] += e2ovov[De(p,v,r,u,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(p,s,t,w,nocc_corr,nvir_corr)] += e2ovov[De(r,v,q,u,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(r,s,v,w,nocc_corr,nvir_corr)] -= e2ovov[De(p,u,q,t,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(q,s,v,w,nocc_corr,nvir_corr)] -= e2ovov[De(r,u,p,t,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(p,s,v,w,nocc_corr,nvir_corr)] -= e2ovov[De(q,u,r,t,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(r,s,v,w,nocc_corr,nvir_corr)] += e2ovov[De(q,u,p,t,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(q,s,v,w,nocc_corr,nvir_corr)] += e2ovov[De(p,u,r,t,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(p,s,v,w,nocc_corr,nvir_corr)] += e2ovov[De(r,u,q,t,nocc_corr,nvir_corr)] * t4;

//               for (it=0; it< nocc_corr; it++){
//               for (jt=0; jt< nocc_corr; jt++){
//               for (at=0; at< nvir_corr; at++){
//               for (bt=0; bt< nvir_corr; bt++){
//                   tmp[it][jt][at][bt] = 0.0;
//               }
//               }
//               }
//               }
//   
//               tmp[r][s][v][w] += (e2ovov[De(p,t,q,u,nocc_corr,nvir_corr)]-e2ovov[De(p,u,q,t,nocc_corr,nvir_corr)]) * t4; 
//               tmp[q][s][v][w] -= (e2ovov[De(p,t,r,u,nocc_corr,nvir_corr)]-e2ovov[De(p,u,r,t,nocc_corr,nvir_corr)]) * t4; 
//               tmp[p][s][v][w] += (e2ovov[De(q,t,s,u,nocc_corr,nvir_corr)]-e2ovov[De(q,u,s,t,nocc_corr,nvir_corr)]) * t4; 
//               tmp[r][s][u][w] -= (e2ovov[De(p,t,q,v,nocc_corr,nvir_corr)]-e2ovov[De(p,v,q,t,nocc_corr,nvir_corr)]) * t4; 
//               tmp[q][s][u][w] += (e2ovov[De(p,t,r,v,nocc_corr,nvir_corr)]-e2ovov[De(p,v,r,t,nocc_corr,nvir_corr)]) * t4; 
//               tmp[p][s][u][w] -= (e2ovov[De(q,t,s,v,nocc_corr,nvir_corr)]-e2ovov[De(q,v,s,t,nocc_corr,nvir_corr)]) * t4; 
//               tmp[r][s][t][w] += (e2ovov[De(p,u,q,v,nocc_corr,nvir_corr)]-e2ovov[De(p,v,q,u,nocc_corr,nvir_corr)]) * t4; 
//               tmp[q][s][t][w] -= (e2ovov[De(p,u,r,v,nocc_corr,nvir_corr)]-e2ovov[De(p,v,r,u,nocc_corr,nvir_corr)]) * t4; 
//               tmp[p][s][t][w] += (e2ovov[De(q,u,s,v,nocc_corr,nvir_corr)]-e2ovov[De(q,v,s,u,nocc_corr,nvir_corr)]) * t4; 
//   
//               for (it = 0; it < nocc_corr; it++)
//               for (jt = 0; jt < nocc_corr; jt++)
//               for (at = 0; at < nvir_corr; at++)
//               for (bt = 0; bt < nvir_corr; bt++)
//                   t2t4c_priv[D(it,jt,at,bt,nocc_corr,nvir_corr)] += tmp[it][jt][at][bt] + tmp[jt][it][bt][at];

           }

           }
           fclose(fp);
           }
       }

#pragma omp critical
       { 
           for (it=0; it< nocc_corr; it++){
               for (jt=0; jt< nocc_corr; jt++){
                   for (at=0; at< nvir_corr; at++){
                       for (bt=0; bt< nvir_corr; bt++){
                           //t2t4c[D(it,jt,at,bt,nocc_corr,nvir_corr)] += t2t4c_priv[D(it,jt,at,bt,nocc_corr,nvir_corr)];
                           t2t4c[D(it,jt,at,bt,nocc_corr,nvir_corr)] += 0.5*(t2t4c_priv[D(it,jt,at,bt,nocc_corr,nvir_corr)]+t2t4c_priv[D(jt,it,bt,at,nocc_corr,nvir_corr)]);
                       }
                   }
               }
           }   
           free(t2t4c_priv);
       } 
}

    printf ("0SDTQ (Q) = %18.16f ( %18.16f )\n", norm, norm-norm0SDT);

}

void t2t4c_shci_otf_mem(double *t2t4c, double *t1, double *t2aa, double *t2ab, double *c3aaa, double *c3aab, double *e2ovov, const int nocc_iact, const int nocc_corr, const int nvir_corr, const int nocc_cas, const int nvir_cas, const double numzero, const double c0, double norm) 
{
    //double numzero = 1e-7;

    const int nocc2 = (int) nocc_cas*(nocc_cas-1)/2;
    const int nocc3 = (int) nocc_cas*(nocc_cas-1)*(nocc_cas-2)/6;
    double norm0SDT = norm;

    const int t2size = nocc_corr*nocc_corr*nvir_corr*nvir_corr;

       double t4, parity, scale;
       int p, q, r, s, t, u, v, w, itmp, it, jt, at, bt, ifile;
       char typ[4], line[255];
       uint8_t Refdet[nocc_corr+nvir_corr], det_str[nocc_corr+nvir_corr];
       for (itmp = 0; itmp < nocc_corr+nvir_corr; itmp++){
           if (itmp<nocc_corr) Refdet[itmp] = 3;  
           else                Refdet[itmp] = 0;
       }

       double *t2t4c_priv;
       t2t4c_priv = (double *)malloc(sizeof(double) * t2size); 
       for (it=0; it< t2size; it++){
           t2t4c_priv[it] = 0.0;
       }
       //lsh test
       //printf ("num_threads = %d\n",omp_get_num_threads());

       FILE *fp;
       fp = fopen("CIcoeff_shci.out", "r");
       fscanf(fp, "%s\n", line);

           if (fp) {
           while ( !feof(fp) ){

           fscanf(fp, "%c%c%c%c,%s\n", &(typ[0]), &(typ[1]), &(typ[2]), &(typ[3]), line);
           fscanf(fp, "%lf\n", &t4);
           //lsh test
           //printf ("typ=%c%c%c%c line=%s\n",typ[0],typ[1],typ[2],typ[3], line);
           if (strncmp(typ, "aabb", 4) == 0 && fabs(t4) > numzero){
               norm += t4*t4;
               sscanf(line,"%d,%d,%d,%d,%d,%d,%d,%d",&p,&q,&t,&u,&r,&s,&v,&w);
               p += nocc_iact;
               q += nocc_iact;
               r += nocc_iact;
               s += nocc_iact;
               t += - nocc_cas;
               u += - nocc_cas;
               v += - nocc_cas;
               w += - nocc_cas;

//               if(!(p == 2 && q == 3 && t == 0 && u == 1 && \
//                    r == 2 && s == 3 && v == 0 && w == 1)) continue;
   
               for (itmp = 0; itmp < nocc_corr+nvir_corr; itmp++)
                   det_str[itmp] = Refdet[itmp];  
   
               det_str[p] = 2;
               det_str[q] = 2;
               det_str[t+nocc_corr] = 1;
               det_str[u+nocc_corr] = 1;
   
               if (p != r && q != r) det_str[r] = 1;
               else  det_str[r] = 0;
               if (p != s && q != s) det_str[s] = 1;
               else  det_str[s] = 0;
               if (t != v && u != v) det_str[v+nocc_corr] = 2;
               else  det_str[v+nocc_corr] = 3;
               if (t != w && u != w) det_str[w+nocc_corr] = 2;
               else  det_str[w+nocc_corr] = 3;
   
               //parity  = parity_ab_str(det_str, nocc_corr+nvir_corr);
               parity = parity_ci_to_cc(p+q, 2, nocc_corr);
               parity *= parity_ci_to_cc(r+s, 2, nocc_corr);
   
               // interm norm of c4
               t4 = parity * t4 / c0;
               // lsh test 
//               printf("c4 mem %20.10f \n",t4);
   
               // extract t4 
               t4-= t1xc3aabb_mem(p, q, r, s, t, u, v, w, nocc_corr, nvir_corr, nocc_cas, nvir_cas, nocc_iact, nocc2, t1, t2aa, t2ab, c3aab, c0); 
               t4-= t2xt2aabb(p, q, r, s, t, u, v, w, nocc_corr, nvir_corr, t2aa, t2ab); 
               t4-= t1xt1xt2aabb(p, q, r, s, t, u, v, w, nocc_corr, nvir_corr, t1, t2aa, t2ab); 
               t4-= t1xt1xt1xt1aabb(p, q, r, s, t, u, v, w, nocc_corr, nvir_corr, t1);   // may have bug 
               //printf("t4 fast %d %d %d %d %d %d %d %d %20.10f \n",p,q,r,s,t,u,v,w,t4);
   
               t2t4c_priv[D(q,s,u,w,nocc_corr,nvir_corr)] += e2ovov[De(p,t,r,v,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(p,s,u,w,nocc_corr,nvir_corr)] -= e2ovov[De(q,t,r,v,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(q,s,t,w,nocc_corr,nvir_corr)] -= e2ovov[De(p,u,r,v,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(p,s,t,w,nocc_corr,nvir_corr)] += e2ovov[De(q,u,r,v,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(q,r,u,w,nocc_corr,nvir_corr)] -= e2ovov[De(p,t,s,v,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(p,r,u,w,nocc_corr,nvir_corr)] += e2ovov[De(q,t,s,v,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(q,r,t,w,nocc_corr,nvir_corr)] += e2ovov[De(p,u,s,v,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(p,r,t,w,nocc_corr,nvir_corr)] -= e2ovov[De(q,u,s,v,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(q,s,u,v,nocc_corr,nvir_corr)] -= e2ovov[De(p,t,r,w,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(p,s,u,v,nocc_corr,nvir_corr)] += e2ovov[De(q,t,r,w,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(q,s,t,v,nocc_corr,nvir_corr)] += e2ovov[De(p,u,r,w,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(p,s,t,v,nocc_corr,nvir_corr)] -= e2ovov[De(q,u,r,w,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(q,r,u,v,nocc_corr,nvir_corr)] += e2ovov[De(p,t,s,w,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(p,r,u,v,nocc_corr,nvir_corr)] -= e2ovov[De(q,t,s,w,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(q,r,t,v,nocc_corr,nvir_corr)] -= e2ovov[De(p,u,s,w,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(p,r,t,v,nocc_corr,nvir_corr)] += e2ovov[De(q,u,s,w,nocc_corr,nvir_corr)] * t4;

           }
           else if (strncmp(typ, "aaab", 4) == 0 && fabs(t4) > numzero){
               norm += 2.0*t4*t4; 
               //lsh test
               //printf ("typ=%c%c%c%c line=%s c4=%lf\n",typ[0],typ[1],typ[2],typ[3], line, t4);

               sscanf(line,"%d,%d,%d,%d,%d,%d,%d,%d",&p,&q,&r,&t,&u,&v,&s,&w);
               p += nocc_iact;
               q += nocc_iact;
               r += nocc_iact;
               s += nocc_iact;
               t += - nocc_cas;
               u += - nocc_cas;
               v += - nocc_cas;
               w += - nocc_cas;

               //printf ("p=%d, q=%d, r=%d, t=%d, u=%d, v=%d, s=%d, w=%d",p,q,r,t,u,v,s,w);

               for (itmp = 0; itmp < nocc_corr+nvir_corr; itmp++)
                   det_str[itmp] = Refdet[itmp];  
               det_str[p] = 2;
               det_str[q] = 2;
               det_str[r] = 2;
               det_str[t+nocc_corr] = 1;
               det_str[u+nocc_corr] = 1;
               det_str[v+nocc_corr] = 1;
   
               if (p != s && q != s && r != s) det_str[s] = 1;
               else  det_str[s] = 0;
               if (t != w && u != w && v != w) det_str[w+nocc_corr] = 2;
               else  det_str[w+nocc_corr] = 3;
               //parity  = parity_ab_str(det_str, nocc_corr+nvir_corr);
               parity = parity_ci_to_cc(p+q+r, 3, nocc_corr);
               parity *= parity_ci_to_cc(s, 1, nocc_corr);
   
               // interm norm of c4
               t4 = parity * t4 / c0;
   
               // extract t4 
               t4-= t1xc3aaab_mem (p, q, r, s, t, u, v, w, nocc_corr, nvir_corr, nocc_cas, nvir_cas, nocc_iact, nocc2, nocc3, t1, t2aa, t2ab, c3aaa, c3aab, c0);
               t4-= t2xt2aaab (p, q, r, s, t, u, v, w, nocc_corr, nvir_corr, t2aa, t2ab);         // may have 1e-3 bug 
               t4-= t1xt1xt2aaab (p, q, r, s, t, u, v, w, nocc_corr, nvir_corr, t1, t2aa, t2ab);  // may have 1e-5 bug 
               t4-= t1xt1xt1xt1aaab (p, q, r, s, t, u, v, w, nocc_corr, nvir_corr, t1);           // may have 1e-6 bug
   
               t2t4c_priv[D(r,s,v,w,nocc_corr,nvir_corr)] += e2ovov[De(p,t,q,u,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(q,s,v,w,nocc_corr,nvir_corr)] += e2ovov[De(r,t,p,u,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(p,s,v,w,nocc_corr,nvir_corr)] += e2ovov[De(q,t,r,u,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(r,s,v,w,nocc_corr,nvir_corr)] -= e2ovov[De(q,t,p,u,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(q,s,v,w,nocc_corr,nvir_corr)] -= e2ovov[De(p,t,r,u,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(p,s,v,w,nocc_corr,nvir_corr)] -= e2ovov[De(r,t,q,u,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(r,s,u,w,nocc_corr,nvir_corr)] += e2ovov[De(p,v,q,t,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(q,s,u,w,nocc_corr,nvir_corr)] += e2ovov[De(r,v,p,t,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(p,s,u,w,nocc_corr,nvir_corr)] += e2ovov[De(q,v,r,t,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(r,s,u,w,nocc_corr,nvir_corr)] -= e2ovov[De(q,v,p,t,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(q,s,u,w,nocc_corr,nvir_corr)] -= e2ovov[De(p,v,r,t,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(p,s,u,w,nocc_corr,nvir_corr)] -= e2ovov[De(r,v,q,t,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(r,s,t,w,nocc_corr,nvir_corr)] += e2ovov[De(p,u,q,v,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(q,s,t,w,nocc_corr,nvir_corr)] += e2ovov[De(r,u,p,v,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(p,s,t,w,nocc_corr,nvir_corr)] += e2ovov[De(q,u,r,v,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(r,s,t,w,nocc_corr,nvir_corr)] -= e2ovov[De(q,u,p,v,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(q,s,t,w,nocc_corr,nvir_corr)] -= e2ovov[De(p,u,r,v,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(p,s,t,w,nocc_corr,nvir_corr)] -= e2ovov[De(r,u,q,v,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(r,s,u,w,nocc_corr,nvir_corr)] -= e2ovov[De(p,t,q,v,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(q,s,u,w,nocc_corr,nvir_corr)] -= e2ovov[De(r,t,p,v,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(p,s,u,w,nocc_corr,nvir_corr)] -= e2ovov[De(q,t,r,v,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(r,s,u,w,nocc_corr,nvir_corr)] += e2ovov[De(q,t,p,v,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(q,s,u,w,nocc_corr,nvir_corr)] += e2ovov[De(p,t,r,v,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(p,s,u,w,nocc_corr,nvir_corr)] += e2ovov[De(r,t,q,v,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(r,s,t,w,nocc_corr,nvir_corr)] -= e2ovov[De(p,v,q,u,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(q,s,t,w,nocc_corr,nvir_corr)] -= e2ovov[De(r,v,p,u,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(p,s,t,w,nocc_corr,nvir_corr)] -= e2ovov[De(q,v,r,u,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(r,s,t,w,nocc_corr,nvir_corr)] += e2ovov[De(q,v,p,u,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(q,s,t,w,nocc_corr,nvir_corr)] += e2ovov[De(p,v,r,u,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(p,s,t,w,nocc_corr,nvir_corr)] += e2ovov[De(r,v,q,u,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(r,s,v,w,nocc_corr,nvir_corr)] -= e2ovov[De(p,u,q,t,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(q,s,v,w,nocc_corr,nvir_corr)] -= e2ovov[De(r,u,p,t,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(p,s,v,w,nocc_corr,nvir_corr)] -= e2ovov[De(q,u,r,t,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(r,s,v,w,nocc_corr,nvir_corr)] += e2ovov[De(q,u,p,t,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(q,s,v,w,nocc_corr,nvir_corr)] += e2ovov[De(p,u,r,t,nocc_corr,nvir_corr)] * t4;
               t2t4c_priv[D(p,s,v,w,nocc_corr,nvir_corr)] += e2ovov[De(r,u,q,t,nocc_corr,nvir_corr)] * t4;
           }

           }
           fclose(fp);
           }

           for (it=0; it< nocc_corr; it++){
               for (jt=0; jt< nocc_corr; jt++){
                   for (at=0; at< nvir_corr; at++){
                       for (bt=0; bt< nvir_corr; bt++){
                           t2t4c[D(it,jt,at,bt,nocc_corr,nvir_corr)] += 0.5*(t2t4c_priv[D(it,jt,at,bt,nocc_corr,nvir_corr)]+t2t4c_priv[D(jt,it,bt,at,nocc_corr,nvir_corr)]);
                       }
                   }
               }
           }   
           free(t2t4c_priv);

    printf ("0SDTQ (Q) =    %f   ( %f )\n", norm, norm-norm0SDT);

}

double permut_value_xaaa(int i, int j, int k, int a, int b, int c, int nocc, int nvir, double t3, double *t1, double *t2aa)
{
    double t3p   = 2.0 * t3; 
    double t1t2p = 0.0, t1t1t1p = 0.0;

    t1t2p += t1[S(i, a, nvir)] * t2aa[D(j, k, b, c, nocc, nvir)];
    t1t2p += t1[S(j, a, nvir)] * t2aa[D(k, i, b, c, nocc, nvir)];
    t1t2p += t1[S(k, a, nvir)] * t2aa[D(i, j, b, c, nocc, nvir)];
    t1t2p -= t1[S(j, a, nvir)] * t2aa[D(i, k, b, c, nocc, nvir)];
    t1t2p -= t1[S(i, a, nvir)] * t2aa[D(k, j, b, c, nocc, nvir)];
    t1t2p -= t1[S(k, a, nvir)] * t2aa[D(j, i, b, c, nocc, nvir)];
    t1t2p += t1[S(i, b, nvir)] * t2aa[D(j, k, c, a, nocc, nvir)];
    t1t2p += t1[S(j, b, nvir)] * t2aa[D(k, i, c, a, nocc, nvir)];
    t1t2p += t1[S(k, b, nvir)] * t2aa[D(i, j, c, a, nocc, nvir)];
    t1t2p -= t1[S(j, b, nvir)] * t2aa[D(i, k, c, a, nocc, nvir)];
    t1t2p -= t1[S(i, b, nvir)] * t2aa[D(k, j, c, a, nocc, nvir)];
    t1t2p -= t1[S(k, b, nvir)] * t2aa[D(j, i, c, a, nocc, nvir)];
    t1t2p += t1[S(i, c, nvir)] * t2aa[D(j, k, a, b, nocc, nvir)];
    t1t2p += t1[S(j, c, nvir)] * t2aa[D(k, i, a, b, nocc, nvir)];
    t1t2p += t1[S(k, c, nvir)] * t2aa[D(i, j, a, b, nocc, nvir)];
    t1t2p -= t1[S(j, c, nvir)] * t2aa[D(i, k, a, b, nocc, nvir)];
    t1t2p -= t1[S(i, c, nvir)] * t2aa[D(k, j, a, b, nocc, nvir)];
    t1t2p -= t1[S(k, c, nvir)] * t2aa[D(j, i, a, b, nocc, nvir)];
    t1t2p -= t1[S(i, b, nvir)] * t2aa[D(j, k, a, c, nocc, nvir)];
    t1t2p -= t1[S(j, b, nvir)] * t2aa[D(k, i, a, c, nocc, nvir)];
    t1t2p -= t1[S(k, b, nvir)] * t2aa[D(i, j, a, c, nocc, nvir)];
    t1t2p += t1[S(j, b, nvir)] * t2aa[D(i, k, a, c, nocc, nvir)];
    t1t2p += t1[S(i, b, nvir)] * t2aa[D(k, j, a, c, nocc, nvir)];
    t1t2p += t1[S(k, b, nvir)] * t2aa[D(j, i, a, c, nocc, nvir)];
    t1t2p -= t1[S(i, a, nvir)] * t2aa[D(j, k, c, b, nocc, nvir)];
    t1t2p -= t1[S(j, a, nvir)] * t2aa[D(k, i, c, b, nocc, nvir)];
    t1t2p -= t1[S(k, a, nvir)] * t2aa[D(i, j, c, b, nocc, nvir)];
    t1t2p += t1[S(j, a, nvir)] * t2aa[D(i, k, c, b, nocc, nvir)];
    t1t2p += t1[S(i, a, nvir)] * t2aa[D(k, j, c, b, nocc, nvir)];
    t1t2p += t1[S(k, a, nvir)] * t2aa[D(j, i, c, b, nocc, nvir)];
    t1t2p -= t1[S(i, c, nvir)] * t2aa[D(j, k, b, a, nocc, nvir)];
    t1t2p -= t1[S(j, c, nvir)] * t2aa[D(k, i, b, a, nocc, nvir)];
    t1t2p -= t1[S(k, c, nvir)] * t2aa[D(i, j, b, a, nocc, nvir)];
    t1t2p += t1[S(j, c, nvir)] * t2aa[D(i, k, b, a, nocc, nvir)];
    t1t2p += t1[S(i, c, nvir)] * t2aa[D(k, j, b, a, nocc, nvir)];
    t1t2p += t1[S(k, c, nvir)] * t2aa[D(j, i, b, a, nocc, nvir)];

    t1t1t1p += t1[S(i, a, nvir)] * t1[S(j, b, nvir)] * t1[S(k, c, nvir)];
    t1t1t1p += t1[S(j, a, nvir)] * t1[S(k, b, nvir)] * t1[S(i, c, nvir)];
    t1t1t1p += t1[S(k, a, nvir)] * t1[S(i, b, nvir)] * t1[S(j, c, nvir)];
    t1t1t1p -= t1[S(j, a, nvir)] * t1[S(i, b, nvir)] * t1[S(k, c, nvir)];
    t1t1t1p -= t1[S(i, a, nvir)] * t1[S(k, b, nvir)] * t1[S(j, c, nvir)];
    t1t1t1p -= t1[S(k, a, nvir)] * t1[S(j, b, nvir)] * t1[S(i, c, nvir)];
    t1t1t1p += t1[S(i, b, nvir)] * t1[S(j, c, nvir)] * t1[S(k, a, nvir)];
    t1t1t1p += t1[S(j, b, nvir)] * t1[S(k, c, nvir)] * t1[S(i, a, nvir)];
    t1t1t1p += t1[S(k, b, nvir)] * t1[S(i, c, nvir)] * t1[S(j, a, nvir)];
    t1t1t1p -= t1[S(j, b, nvir)] * t1[S(i, c, nvir)] * t1[S(k, a, nvir)];
    t1t1t1p -= t1[S(i, b, nvir)] * t1[S(k, c, nvir)] * t1[S(j, a, nvir)];
    t1t1t1p -= t1[S(k, b, nvir)] * t1[S(j, c, nvir)] * t1[S(i, a, nvir)];
    t1t1t1p += t1[S(i, c, nvir)] * t1[S(j, a, nvir)] * t1[S(k, b, nvir)];
    t1t1t1p += t1[S(j, c, nvir)] * t1[S(k, a, nvir)] * t1[S(i, b, nvir)];
    t1t1t1p += t1[S(k, c, nvir)] * t1[S(i, a, nvir)] * t1[S(j, b, nvir)];
    t1t1t1p -= t1[S(j, c, nvir)] * t1[S(i, a, nvir)] * t1[S(k, b, nvir)];
    t1t1t1p -= t1[S(i, c, nvir)] * t1[S(k, a, nvir)] * t1[S(j, b, nvir)];
    t1t1t1p -= t1[S(k, c, nvir)] * t1[S(j, a, nvir)] * t1[S(i, b, nvir)];
    t1t1t1p -= t1[S(i, b, nvir)] * t1[S(j, a, nvir)] * t1[S(k, c, nvir)];
    t1t1t1p -= t1[S(j, b, nvir)] * t1[S(k, a, nvir)] * t1[S(i, c, nvir)];
    t1t1t1p -= t1[S(k, b, nvir)] * t1[S(i, a, nvir)] * t1[S(j, c, nvir)];
    t1t1t1p += t1[S(j, b, nvir)] * t1[S(i, a, nvir)] * t1[S(k, c, nvir)];
    t1t1t1p += t1[S(i, b, nvir)] * t1[S(k, a, nvir)] * t1[S(j, c, nvir)];
    t1t1t1p += t1[S(k, b, nvir)] * t1[S(j, a, nvir)] * t1[S(i, c, nvir)];
    t1t1t1p -= t1[S(i, a, nvir)] * t1[S(j, c, nvir)] * t1[S(k, b, nvir)];
    t1t1t1p -= t1[S(j, a, nvir)] * t1[S(k, c, nvir)] * t1[S(i, b, nvir)];
    t1t1t1p -= t1[S(k, a, nvir)] * t1[S(i, c, nvir)] * t1[S(j, b, nvir)];
    t1t1t1p += t1[S(j, a, nvir)] * t1[S(i, c, nvir)] * t1[S(k, b, nvir)];
    t1t1t1p += t1[S(i, a, nvir)] * t1[S(k, c, nvir)] * t1[S(j, b, nvir)];
    t1t1t1p += t1[S(k, a, nvir)] * t1[S(j, c, nvir)] * t1[S(i, b, nvir)];
    t1t1t1p -= t1[S(i, c, nvir)] * t1[S(j, b, nvir)] * t1[S(k, a, nvir)];
    t1t1t1p -= t1[S(j, c, nvir)] * t1[S(k, b, nvir)] * t1[S(i, a, nvir)];
    t1t1t1p -= t1[S(k, c, nvir)] * t1[S(i, b, nvir)] * t1[S(j, a, nvir)];
    t1t1t1p += t1[S(j, c, nvir)] * t1[S(i, b, nvir)] * t1[S(k, a, nvir)];
    t1t1t1p += t1[S(i, c, nvir)] * t1[S(k, b, nvir)] * t1[S(j, a, nvir)];
    t1t1t1p += t1[S(k, c, nvir)] * t1[S(j, b, nvir)] * t1[S(i, a, nvir)];

    return t3p + t1t2p/2.0 + t1t1t1p/3.0;
}

double permut_value_xaab(int i, int j, int k, int a, int b, int c, int nocc, int nvir, double t3, double *t1, double *t2aa, double *t2ab)
{
    double t3p   = 2.0 * t3; 
    double t1t2aap = 0.0, t1t2abp = 0.0, t1t1t1p = 0.0;

    t1t2aap += t2aa[D(i, j, a, b, nocc, nvir)];
    t1t2aap -= t2aa[D(j, i, a, b, nocc, nvir)];
    t1t2aap -= t2aa[D(i, j, b, a, nocc, nvir)];
    t1t2aap += t2aa[D(j, i, b, a, nocc, nvir)];
    t1t2aap *= t1[S(k, c, nvir)];

    t1t2abp += t1[S(i, a, nvir)] * t2ab[D(j, k, b, c, nocc, nvir)];
    t1t2abp -= t1[S(j, a, nvir)] * t2ab[D(i, k, b, c, nocc, nvir)];
    t1t2abp -= t1[S(i, b, nvir)] * t2ab[D(j, k, a, c, nocc, nvir)];
    t1t2abp += t1[S(j, b, nvir)] * t2ab[D(i, k, a, c, nocc, nvir)];

    t1t1t1p += t1[S(i, a, nvir)] * t1[S(j, b, nvir)] * t1[S(k, c, nvir)];
    t1t1t1p -= t1[S(j, a, nvir)] * t1[S(i, b, nvir)] * t1[S(k, c, nvir)];
    t1t1t1p -= t1[S(i, b, nvir)] * t1[S(j, a, nvir)] * t1[S(k, c, nvir)];
    t1t1t1p += t1[S(j, b, nvir)] * t1[S(i, a, nvir)] * t1[S(k, c, nvir)];

    return t3p + t1t2aap/2.0 + 2.0*t1t2abp + t1t1t1p;
}

double permut_value_xaabb(const int i, const int j, const int k, const int l,
    const int a, const int b, const int c, const int d, const int nocc_corr,
    const int nvir_corr, const int nocc_cas, const int nvir_cas,
    const int nocc_iact, const int nocc2, const double t4, double *t1,
    double *t2aa, double *t2ab, double *c3aab, double *paab, const double c0)
{
    double t4p   = 2.0 * t4; 
    double t1t3aab = 0.0, t2t2aa = 0.0, t2t2ab = 0.0;
    double t2aat1t1 = 0.0, t2abt1t1 = 0.0, t1t1t1t1 = 0.0;

    //lsh test
    //printf ("p=%d, q=%d, r=%d, s=%d, t=%d, u=%d, v=%d, w=%d",i,j,k,l,a,b,c,d);

    //t1t3aab
    double t3_ijk_abc, t3_ijl_abc, t3_ijk_abd, t3_ijl_abd;

    t3_ijk_abc = c3tot3aab_mem(i, j, k, a, b, c, nocc_corr, nvir_corr, 
                 nocc_cas, nvir_cas, nocc_iact, nocc2, t1, t2aa, t2ab, c3aab, c0);
    t3_ijl_abc = c3tot3aab_mem(i, j, l, a, b, c, nocc_corr, nvir_corr, 
                 nocc_cas, nvir_cas, nocc_iact, nocc2, t1, t2aa, t2ab, c3aab, c0);
    t3_ijk_abd = c3tot3aab_mem(i, j, k, a, b, d, nocc_corr, nvir_corr, 
                 nocc_cas, nvir_cas, nocc_iact, nocc2, t1, t2aa, t2ab, c3aab, c0);
    t3_ijl_abd = c3tot3aab_mem(i, j, l, a, b, d, nocc_corr, nvir_corr, 
                 nocc_cas, nvir_cas, nocc_iact, nocc2, t1, t2aa, t2ab, c3aab, c0);

    t1t3aab += t1[S(l, d, nvir_corr)] * t3_ijk_abc;
    t1t3aab -= t1[S(k, d, nvir_corr)] * t3_ijl_abc;
    t1t3aab -= t1[S(l, c, nvir_corr)] * t3_ijk_abd;
    t1t3aab += t1[S(k, c, nvir_corr)] * t3_ijl_abd;
    t1t3aab *= 4.0;

    //t2t2aa
    t2t2aa += t2aa[D(i, j, a, b, nocc_corr, nvir_corr)] * t2aa[D(k, l, c, d, nocc_corr, nvir_corr)];
    t2t2aa *= 16.0;

    //t2t2ab
    t2t2ab += t2ab[D(i, k, a, c, nocc_corr, nvir_corr)] * t2ab[D(j, l, b, d, nocc_corr, nvir_corr)];
    t2t2ab -= t2ab[D(j, k, a, c, nocc_corr, nvir_corr)] * t2ab[D(i, l, b, d, nocc_corr, nvir_corr)];
    t2t2ab -= t2ab[D(i, k, b, c, nocc_corr, nvir_corr)] * t2ab[D(j, l, a, d, nocc_corr, nvir_corr)];
    t2t2ab += t2ab[D(j, k, b, c, nocc_corr, nvir_corr)] * t2ab[D(i, l, a, d, nocc_corr, nvir_corr)];

    t2t2ab -= t2ab[D(i, l, a, c, nocc_corr, nvir_corr)] * t2ab[D(j, k, b, d, nocc_corr, nvir_corr)];
    t2t2ab += t2ab[D(j, l, a, c, nocc_corr, nvir_corr)] * t2ab[D(i, k, b, d, nocc_corr, nvir_corr)];
    t2t2ab += t2ab[D(i, l, b, c, nocc_corr, nvir_corr)] * t2ab[D(j, k, a, d, nocc_corr, nvir_corr)];
    t2t2ab -= t2ab[D(j, l, b, c, nocc_corr, nvir_corr)] * t2ab[D(i, k, a, d, nocc_corr, nvir_corr)];

    t2t2ab -= t2ab[D(i, k, a, d, nocc_corr, nvir_corr)] * t2ab[D(j, l, b, c, nocc_corr, nvir_corr)];
    t2t2ab += t2ab[D(j, k, a, d, nocc_corr, nvir_corr)] * t2ab[D(i, l, b, c, nocc_corr, nvir_corr)];
    t2t2ab += t2ab[D(i, k, b, d, nocc_corr, nvir_corr)] * t2ab[D(j, l, a, c, nocc_corr, nvir_corr)];
    t2t2ab -= t2ab[D(j, k, b, d, nocc_corr, nvir_corr)] * t2ab[D(i, l, a, c, nocc_corr, nvir_corr)];

    t2t2ab += t2ab[D(i, l, a, d, nocc_corr, nvir_corr)] * t2ab[D(j, k, b, c, nocc_corr, nvir_corr)];
    t2t2ab -= t2ab[D(j, l, a, d, nocc_corr, nvir_corr)] * t2ab[D(i, k, b, c, nocc_corr, nvir_corr)];
    t2t2ab -= t2ab[D(i, l, b, d, nocc_corr, nvir_corr)] * t2ab[D(j, k, a, c, nocc_corr, nvir_corr)];
    t2t2ab += t2ab[D(j, l, b, d, nocc_corr, nvir_corr)] * t2ab[D(i, k, a, c, nocc_corr, nvir_corr)];

    // t2aat1t1
    t2aat1t1 += t2aa[D(i, j, a, b, nocc_corr, nvir_corr)] * t1[S(k, c, nvir_corr)] * t1[S(l, d, nvir_corr)];
    t2aat1t1 -= t2aa[D(i, j, a, b, nocc_corr, nvir_corr)] * t1[S(l, c, nvir_corr)] * t1[S(k, d, nvir_corr)];
    t2aat1t1 *= 8.0;

    // t2abt1t1
    t2abt1t1 += t2ab[D(i, k, a, c, nocc_corr, nvir_corr)] * t1[S(j, b, nvir_corr)] * t1[S(l, d, nvir_corr)];
    t2abt1t1 -= t2ab[D(j, k, a, c, nocc_corr, nvir_corr)] * t1[S(i, b, nvir_corr)] * t1[S(l, d, nvir_corr)];
    t2abt1t1 -= t2ab[D(i, k, b, c, nocc_corr, nvir_corr)] * t1[S(j, a, nvir_corr)] * t1[S(l, d, nvir_corr)];
    t2abt1t1 += t2ab[D(j, k, b, c, nocc_corr, nvir_corr)] * t1[S(i, a, nvir_corr)] * t1[S(l, d, nvir_corr)];

    t2abt1t1 -= t2ab[D(i, l, a, c, nocc_corr, nvir_corr)] * t1[S(j, b, nvir_corr)] * t1[S(k, d, nvir_corr)];
    t2abt1t1 += t2ab[D(j, l, a, c, nocc_corr, nvir_corr)] * t1[S(i, b, nvir_corr)] * t1[S(k, d, nvir_corr)];
    t2abt1t1 += t2ab[D(i, l, b, c, nocc_corr, nvir_corr)] * t1[S(j, a, nvir_corr)] * t1[S(k, d, nvir_corr)];
    t2abt1t1 -= t2ab[D(j, l, b, c, nocc_corr, nvir_corr)] * t1[S(i, a, nvir_corr)] * t1[S(k, d, nvir_corr)];

    t2abt1t1 -= t2ab[D(i, k, a, d, nocc_corr, nvir_corr)] * t1[S(j, b, nvir_corr)] * t1[S(l, c, nvir_corr)];
    t2abt1t1 += t2ab[D(j, k, a, d, nocc_corr, nvir_corr)] * t1[S(i, b, nvir_corr)] * t1[S(l, c, nvir_corr)];
    t2abt1t1 += t2ab[D(i, k, b, d, nocc_corr, nvir_corr)] * t1[S(j, a, nvir_corr)] * t1[S(l, c, nvir_corr)];
    t2abt1t1 -= t2ab[D(j, k, b, d, nocc_corr, nvir_corr)] * t1[S(i, a, nvir_corr)] * t1[S(l, c, nvir_corr)];

    t2abt1t1 += t2ab[D(i, l, a, d, nocc_corr, nvir_corr)] * t1[S(j, b, nvir_corr)] * t1[S(k, c, nvir_corr)];
    t2abt1t1 -= t2ab[D(j, l, a, d, nocc_corr, nvir_corr)] * t1[S(i, b, nvir_corr)] * t1[S(k, c, nvir_corr)];
    t2abt1t1 -= t2ab[D(i, l, b, d, nocc_corr, nvir_corr)] * t1[S(j, a, nvir_corr)] * t1[S(k, c, nvir_corr)];
    t2abt1t1 += t2ab[D(j, l, b, d, nocc_corr, nvir_corr)] * t1[S(i, a, nvir_corr)] * t1[S(k, c, nvir_corr)];

    //t1t1t1t1
    t1t1t1t1 += t1[S(i, a, nvir_corr)] * t1[S(j, b, nvir_corr)] * t1[S(k, c, nvir_corr)] * t1[S(l, d, nvir_corr)];
    t1t1t1t1 -= t1[S(j, a, nvir_corr)] * t1[S(i, b, nvir_corr)] * t1[S(k, c, nvir_corr)] * t1[S(l, d, nvir_corr)];
    t1t1t1t1 -= t1[S(i, a, nvir_corr)] * t1[S(j, b, nvir_corr)] * t1[S(l, c, nvir_corr)] * t1[S(k, d, nvir_corr)];
    t1t1t1t1 += t1[S(j, a, nvir_corr)] * t1[S(i, b, nvir_corr)] * t1[S(l, c, nvir_corr)] * t1[S(k, d, nvir_corr)];
    t1t1t1t1 *= 4.0;

    return t4p + t1t3aab/2.0 + t2t2aa/16.0 + t2t2ab/2.0 + t2aat1t1/4.0 + t2abt1t1 + t1t1t1t1/4.0;
}
double permut_value_xaaab(const int i, const int j, const int k, const int l,
    const int a, const int b, const int c, const int d, const int nocc_corr,
    const int nvir_corr, const int nocc_cas, const int nvir_cas,
    const int nocc_iact, const int nocc2, const int nocc3, const double t4, double *t1,
    double *t2aa, double *t2ab, double *c3aaa, double *c3aab,
    double *paaa, double *paab, const double c0)
{
    double t4p   = 2.0 * t4; 
    double t1t3aaa = 0.0, t1t3aab = 0.0, t2t2 = 0.0;
    double t2aat1t1 = 0.0, t2abt1t1 = 0.0, t1t1t1t1 = 0.0;

    //lsh test
    //printf ("p=%d, q=%d, r=%d, s=%d, t=%d, u=%d, v=%d, w=%d",i,j,k,l,a,b,c,d);

    //t1t3aaa
    double t3aaa;
    t3aaa = c3tot3aaa_mem(i, j, k, a, b, c, nocc_corr, nvir_corr, 
            nocc_cas, nvir_cas, nocc_iact, nocc3, t1, t2aa, c3aaa, c0);
    t1t3aaa = 2.0 * t1[S(l, d, nvir_corr)] * t3aaa;

    //t1t3aab
    double t3_jkl_bcd, t3_ikl_bcd, t3_ijl_bcd;
    double t3_jkl_acd, t3_ikl_acd, t3_ijl_acd;
    double t3_jkl_abd, t3_ikl_abd, t3_ijl_abd;
    t3_jkl_bcd = c3tot3aab_mem(j, k, l, b, c, d, nocc_corr, nvir_corr, 
                 nocc_cas, nvir_cas, nocc_iact, nocc2, t1, t2aa, t2ab, c3aab, c0);
    t3_ikl_bcd = c3tot3aab_mem(i, k, l, b, c, d, nocc_corr, nvir_corr, 
                 nocc_cas, nvir_cas, nocc_iact, nocc2, t1, t2aa, t2ab, c3aab, c0);
    t3_ijl_bcd = c3tot3aab_mem(i, j, l, b, c, d, nocc_corr, nvir_corr, 
                 nocc_cas, nvir_cas, nocc_iact, nocc2, t1, t2aa, t2ab, c3aab, c0);
    t3_jkl_acd = c3tot3aab_mem(j, k, l, a, c, d, nocc_corr, nvir_corr, 
                 nocc_cas, nvir_cas, nocc_iact, nocc2, t1, t2aa, t2ab, c3aab, c0);
    t3_ikl_acd = c3tot3aab_mem(i, k, l, a, c, d, nocc_corr, nvir_corr, 
                 nocc_cas, nvir_cas, nocc_iact, nocc2, t1, t2aa, t2ab, c3aab, c0);
    t3_ijl_acd = c3tot3aab_mem(i, j, l, a, c, d, nocc_corr, nvir_corr, 
                 nocc_cas, nvir_cas, nocc_iact, nocc2, t1, t2aa, t2ab, c3aab, c0);
    t3_jkl_abd = c3tot3aab_mem(j, k, l, a, b, d, nocc_corr, nvir_corr, 
                 nocc_cas, nvir_cas, nocc_iact, nocc2, t1, t2aa, t2ab, c3aab, c0);
    t3_ikl_abd = c3tot3aab_mem(i, k, l, a, b, d, nocc_corr, nvir_corr, 
                 nocc_cas, nvir_cas, nocc_iact, nocc2, t1, t2aa, t2ab, c3aab, c0);
    t3_ijl_abd = c3tot3aab_mem(i, j, l, a, b, d, nocc_corr, nvir_corr, 
                 nocc_cas, nvir_cas, nocc_iact, nocc2, t1, t2aa, t2ab, c3aab, c0);

    t1t3aab += t1[S(i, a, nvir_corr)] * t3_jkl_bcd;
    t1t3aab -= t1[S(j, a, nvir_corr)] * t3_ikl_bcd;
    t1t3aab += t1[S(k, a, nvir_corr)] * t3_ijl_bcd;
    t1t3aab -= t1[S(j, a, nvir_corr)] * t3_ikl_bcd;
    t1t3aab += t1[S(i, a, nvir_corr)] * t3_jkl_bcd;
    t1t3aab += t1[S(k, a, nvir_corr)] * t3_ijl_bcd;

    t1t3aab -= t1[S(i, b, nvir_corr)] * t3_jkl_acd;
    t1t3aab += t1[S(j, b, nvir_corr)] * t3_ikl_acd;
    t1t3aab -= t1[S(k, b, nvir_corr)] * t3_ijl_acd;
    t1t3aab += t1[S(j, b, nvir_corr)] * t3_ikl_acd;
    t1t3aab -= t1[S(i, b, nvir_corr)] * t3_jkl_acd;
    t1t3aab -= t1[S(k, b, nvir_corr)] * t3_ijl_acd;

    t1t3aab += t1[S(i, c, nvir_corr)] * t3_jkl_abd;
    t1t3aab -= t1[S(j, c, nvir_corr)] * t3_ikl_abd;
    t1t3aab += t1[S(k, c, nvir_corr)] * t3_ijl_abd;
    t1t3aab -= t1[S(j, c, nvir_corr)] * t3_ikl_abd;
    t1t3aab += t1[S(i, c, nvir_corr)] * t3_jkl_abd;
    t1t3aab += t1[S(k, c, nvir_corr)] * t3_ijl_abd;

    t1t3aab -= t1[S(i, b, nvir_corr)] * t3_jkl_acd;
    t1t3aab += t1[S(j, b, nvir_corr)] * t3_ikl_acd;
    t1t3aab -= t1[S(k, b, nvir_corr)] * t3_ijl_acd;
    t1t3aab += t1[S(j, b, nvir_corr)] * t3_ikl_acd;
    t1t3aab -= t1[S(i, b, nvir_corr)] * t3_jkl_acd;
    t1t3aab -= t1[S(k, b, nvir_corr)] * t3_ijl_acd;

    t1t3aab += t1[S(i, a, nvir_corr)] * t3_jkl_bcd;
    t1t3aab -= t1[S(j, a, nvir_corr)] * t3_ikl_bcd;
    t1t3aab += t1[S(k, a, nvir_corr)] * t3_ijl_bcd;
    t1t3aab -= t1[S(j, a, nvir_corr)] * t3_ikl_bcd;
    t1t3aab += t1[S(i, a, nvir_corr)] * t3_jkl_bcd;
    t1t3aab += t1[S(k, a, nvir_corr)] * t3_ijl_bcd;

    t1t3aab += t1[S(i, c, nvir_corr)] * t3_jkl_abd;
    t1t3aab -= t1[S(j, c, nvir_corr)] * t3_ikl_abd;
    t1t3aab += t1[S(k, c, nvir_corr)] * t3_ijl_abd;
    t1t3aab -= t1[S(j, c, nvir_corr)] * t3_ikl_abd;
    t1t3aab += t1[S(i, c, nvir_corr)] * t3_jkl_abd;
    t1t3aab += t1[S(k, c, nvir_corr)] * t3_ijl_abd;

    //t2t2
    t2t2 += t2aa[D(i, j, a, b, nocc_corr, nvir_corr)] * t2ab[D(k, l, c, d, nocc_corr, nvir_corr)];
    t2t2 += t2aa[D(j, k, a, b, nocc_corr, nvir_corr)] * t2ab[D(i, l, c, d, nocc_corr, nvir_corr)];
    t2t2 += t2aa[D(k, i, a, b, nocc_corr, nvir_corr)] * t2ab[D(j, l, c, d, nocc_corr, nvir_corr)];
    t2t2 -= t2aa[D(j, i, a, b, nocc_corr, nvir_corr)] * t2ab[D(k, l, c, d, nocc_corr, nvir_corr)];
    t2t2 -= t2aa[D(i, k, a, b, nocc_corr, nvir_corr)] * t2ab[D(j, l, c, d, nocc_corr, nvir_corr)];
    t2t2 -= t2aa[D(k, j, a, b, nocc_corr, nvir_corr)] * t2ab[D(i, l, c, d, nocc_corr, nvir_corr)];

    t2t2 += t2aa[D(i, j, b, c, nocc_corr, nvir_corr)] * t2ab[D(k, l, a, d, nocc_corr, nvir_corr)];
    t2t2 += t2aa[D(j, k, b, c, nocc_corr, nvir_corr)] * t2ab[D(i, l, a, d, nocc_corr, nvir_corr)];
    t2t2 += t2aa[D(k, i, b, c, nocc_corr, nvir_corr)] * t2ab[D(j, l, a, d, nocc_corr, nvir_corr)];
    t2t2 -= t2aa[D(j, i, b, c, nocc_corr, nvir_corr)] * t2ab[D(k, l, a, d, nocc_corr, nvir_corr)];
    t2t2 -= t2aa[D(i, k, b, c, nocc_corr, nvir_corr)] * t2ab[D(j, l, a, d, nocc_corr, nvir_corr)];
    t2t2 -= t2aa[D(k, j, b, c, nocc_corr, nvir_corr)] * t2ab[D(i, l, a, d, nocc_corr, nvir_corr)];

    t2t2 += t2aa[D(i, j, c, a, nocc_corr, nvir_corr)] * t2ab[D(k, l, b, d, nocc_corr, nvir_corr)];
    t2t2 += t2aa[D(j, k, c, a, nocc_corr, nvir_corr)] * t2ab[D(i, l, b, d, nocc_corr, nvir_corr)];
    t2t2 += t2aa[D(k, i, c, a, nocc_corr, nvir_corr)] * t2ab[D(j, l, b, d, nocc_corr, nvir_corr)];
    t2t2 -= t2aa[D(j, i, c, a, nocc_corr, nvir_corr)] * t2ab[D(k, l, b, d, nocc_corr, nvir_corr)];
    t2t2 -= t2aa[D(i, k, c, a, nocc_corr, nvir_corr)] * t2ab[D(j, l, b, d, nocc_corr, nvir_corr)];
    t2t2 -= t2aa[D(k, j, c, a, nocc_corr, nvir_corr)] * t2ab[D(i, l, b, d, nocc_corr, nvir_corr)];

    t2t2 -= t2aa[D(i, j, b, a, nocc_corr, nvir_corr)] * t2ab[D(k, l, c, d, nocc_corr, nvir_corr)];
    t2t2 -= t2aa[D(j, k, b, a, nocc_corr, nvir_corr)] * t2ab[D(i, l, c, d, nocc_corr, nvir_corr)];
    t2t2 -= t2aa[D(k, i, b, a, nocc_corr, nvir_corr)] * t2ab[D(j, l, c, d, nocc_corr, nvir_corr)];
    t2t2 += t2aa[D(j, i, b, a, nocc_corr, nvir_corr)] * t2ab[D(k, l, c, d, nocc_corr, nvir_corr)];
    t2t2 += t2aa[D(i, k, b, a, nocc_corr, nvir_corr)] * t2ab[D(j, l, c, d, nocc_corr, nvir_corr)];
    t2t2 += t2aa[D(k, j, b, a, nocc_corr, nvir_corr)] * t2ab[D(i, l, c, d, nocc_corr, nvir_corr)];

    t2t2 -= t2aa[D(i, j, a, c, nocc_corr, nvir_corr)] * t2ab[D(k, l, b, d, nocc_corr, nvir_corr)];
    t2t2 -= t2aa[D(j, k, a, c, nocc_corr, nvir_corr)] * t2ab[D(i, l, b, d, nocc_corr, nvir_corr)];
    t2t2 -= t2aa[D(k, i, a, c, nocc_corr, nvir_corr)] * t2ab[D(j, l, b, d, nocc_corr, nvir_corr)];
    t2t2 += t2aa[D(j, i, a, c, nocc_corr, nvir_corr)] * t2ab[D(k, l, b, d, nocc_corr, nvir_corr)];
    t2t2 += t2aa[D(i, k, a, c, nocc_corr, nvir_corr)] * t2ab[D(j, l, b, d, nocc_corr, nvir_corr)];
    t2t2 += t2aa[D(k, j, a, c, nocc_corr, nvir_corr)] * t2ab[D(i, l, b, d, nocc_corr, nvir_corr)];

    t2t2 -= t2aa[D(i, j, c, b, nocc_corr, nvir_corr)] * t2ab[D(k, l, a, d, nocc_corr, nvir_corr)];
    t2t2 -= t2aa[D(j, k, c, b, nocc_corr, nvir_corr)] * t2ab[D(i, l, a, d, nocc_corr, nvir_corr)];
    t2t2 -= t2aa[D(k, i, c, b, nocc_corr, nvir_corr)] * t2ab[D(j, l, a, d, nocc_corr, nvir_corr)];
    t2t2 += t2aa[D(j, i, c, b, nocc_corr, nvir_corr)] * t2ab[D(k, l, a, d, nocc_corr, nvir_corr)];
    t2t2 += t2aa[D(i, k, c, b, nocc_corr, nvir_corr)] * t2ab[D(j, l, a, d, nocc_corr, nvir_corr)];
    t2t2 += t2aa[D(k, j, c, b, nocc_corr, nvir_corr)] * t2ab[D(i, l, a, d, nocc_corr, nvir_corr)];

    // t2aat1t1
    t2aat1t1 += t2aa[D(i, j, a, b, nocc_corr, nvir_corr)] * t1[S(k, c, nvir_corr)] * t1[S(l, d, nvir_corr)];
    t2aat1t1 += t2aa[D(j, k, a, b, nocc_corr, nvir_corr)] * t1[S(i, c, nvir_corr)] * t1[S(l, d, nvir_corr)];
    t2aat1t1 += t2aa[D(i, j, a, b, nocc_corr, nvir_corr)] * t1[S(j, c, nvir_corr)] * t1[S(l, d, nvir_corr)];
    t2aat1t1 -= t2aa[D(j, i, a, b, nocc_corr, nvir_corr)] * t1[S(k, c, nvir_corr)] * t1[S(l, d, nvir_corr)];
    t2aat1t1 -= t2aa[D(i, k, a, b, nocc_corr, nvir_corr)] * t1[S(j, c, nvir_corr)] * t1[S(l, d, nvir_corr)];
    t2aat1t1 -= t2aa[D(k, j, a, b, nocc_corr, nvir_corr)] * t1[S(i, c, nvir_corr)] * t1[S(l, d, nvir_corr)];

    t2aat1t1 += t2aa[D(i, j, b, c, nocc_corr, nvir_corr)] * t1[S(k, a, nvir_corr)] * t1[S(l, d, nvir_corr)];
    t2aat1t1 += t2aa[D(j, k, b, c, nocc_corr, nvir_corr)] * t1[S(i, a, nvir_corr)] * t1[S(l, d, nvir_corr)];
    t2aat1t1 += t2aa[D(i, j, b, c, nocc_corr, nvir_corr)] * t1[S(j, a, nvir_corr)] * t1[S(l, d, nvir_corr)];
    t2aat1t1 -= t2aa[D(j, i, b, c, nocc_corr, nvir_corr)] * t1[S(k, a, nvir_corr)] * t1[S(l, d, nvir_corr)];
    t2aat1t1 -= t2aa[D(i, k, b, c, nocc_corr, nvir_corr)] * t1[S(j, a, nvir_corr)] * t1[S(l, d, nvir_corr)];
    t2aat1t1 -= t2aa[D(k, j, b, c, nocc_corr, nvir_corr)] * t1[S(i, a, nvir_corr)] * t1[S(l, d, nvir_corr)];

    t2aat1t1 += t2aa[D(i, j, c, a, nocc_corr, nvir_corr)] * t1[S(k, b, nvir_corr)] * t1[S(l, d, nvir_corr)];
    t2aat1t1 += t2aa[D(j, k, c, a, nocc_corr, nvir_corr)] * t1[S(i, b, nvir_corr)] * t1[S(l, d, nvir_corr)];
    t2aat1t1 += t2aa[D(i, j, c, a, nocc_corr, nvir_corr)] * t1[S(j, b, nvir_corr)] * t1[S(l, d, nvir_corr)];
    t2aat1t1 -= t2aa[D(j, i, c, a, nocc_corr, nvir_corr)] * t1[S(k, b, nvir_corr)] * t1[S(l, d, nvir_corr)];
    t2aat1t1 -= t2aa[D(i, k, c, a, nocc_corr, nvir_corr)] * t1[S(j, b, nvir_corr)] * t1[S(l, d, nvir_corr)];
    t2aat1t1 -= t2aa[D(k, j, c, a, nocc_corr, nvir_corr)] * t1[S(i, b, nvir_corr)] * t1[S(l, d, nvir_corr)];

    t2aat1t1 -= t2aa[D(i, j, b, a, nocc_corr, nvir_corr)] * t1[S(k, c, nvir_corr)] * t1[S(l, d, nvir_corr)];
    t2aat1t1 -= t2aa[D(j, k, b, a, nocc_corr, nvir_corr)] * t1[S(i, c, nvir_corr)] * t1[S(l, d, nvir_corr)];
    t2aat1t1 -= t2aa[D(i, j, b, a, nocc_corr, nvir_corr)] * t1[S(j, c, nvir_corr)] * t1[S(l, d, nvir_corr)];
    t2aat1t1 += t2aa[D(j, i, b, a, nocc_corr, nvir_corr)] * t1[S(k, c, nvir_corr)] * t1[S(l, d, nvir_corr)];
    t2aat1t1 += t2aa[D(i, k, b, a, nocc_corr, nvir_corr)] * t1[S(j, c, nvir_corr)] * t1[S(l, d, nvir_corr)];
    t2aat1t1 += t2aa[D(k, j, b, a, nocc_corr, nvir_corr)] * t1[S(i, c, nvir_corr)] * t1[S(l, d, nvir_corr)];

    t2aat1t1 -= t2aa[D(i, j, a, c, nocc_corr, nvir_corr)] * t1[S(k, b, nvir_corr)] * t1[S(l, d, nvir_corr)];
    t2aat1t1 -= t2aa[D(j, k, a, c, nocc_corr, nvir_corr)] * t1[S(i, b, nvir_corr)] * t1[S(l, d, nvir_corr)];
    t2aat1t1 -= t2aa[D(i, j, a, c, nocc_corr, nvir_corr)] * t1[S(j, b, nvir_corr)] * t1[S(l, d, nvir_corr)];
    t2aat1t1 += t2aa[D(j, i, a, c, nocc_corr, nvir_corr)] * t1[S(k, b, nvir_corr)] * t1[S(l, d, nvir_corr)];
    t2aat1t1 += t2aa[D(i, k, a, c, nocc_corr, nvir_corr)] * t1[S(j, b, nvir_corr)] * t1[S(l, d, nvir_corr)];
    t2aat1t1 += t2aa[D(k, j, a, c, nocc_corr, nvir_corr)] * t1[S(i, b, nvir_corr)] * t1[S(l, d, nvir_corr)];

    t2aat1t1 -= t2aa[D(i, j, c, b, nocc_corr, nvir_corr)] * t1[S(k, a, nvir_corr)] * t1[S(l, d, nvir_corr)];
    t2aat1t1 -= t2aa[D(j, k, c, b, nocc_corr, nvir_corr)] * t1[S(i, a, nvir_corr)] * t1[S(l, d, nvir_corr)];
    t2aat1t1 -= t2aa[D(i, j, c, b, nocc_corr, nvir_corr)] * t1[S(j, a, nvir_corr)] * t1[S(l, d, nvir_corr)];
    t2aat1t1 += t2aa[D(j, i, c, b, nocc_corr, nvir_corr)] * t1[S(k, a, nvir_corr)] * t1[S(l, d, nvir_corr)];
    t2aat1t1 += t2aa[D(i, k, c, b, nocc_corr, nvir_corr)] * t1[S(j, a, nvir_corr)] * t1[S(l, d, nvir_corr)];
    t2aat1t1 += t2aa[D(k, j, c, b, nocc_corr, nvir_corr)] * t1[S(i, a, nvir_corr)] * t1[S(l, d, nvir_corr)];

    // t2abt1t1
    t2abt1t1 += t2ab[D(i, l, a, d, nocc_corr, nvir_corr)] * t1[S(j, b, nvir_corr)] * t1[S(k, c, nvir_corr)];
    t2abt1t1 += t2ab[D(j, l, a, d, nocc_corr, nvir_corr)] * t1[S(k, b, nvir_corr)] * t1[S(i, c, nvir_corr)];
    t2abt1t1 += t2ab[D(i, l, a, d, nocc_corr, nvir_corr)] * t1[S(j, b, nvir_corr)] * t1[S(j, c, nvir_corr)];
    t2abt1t1 -= t2ab[D(j, l, a, d, nocc_corr, nvir_corr)] * t1[S(i, b, nvir_corr)] * t1[S(k, c, nvir_corr)];
    t2abt1t1 -= t2ab[D(i, l, a, d, nocc_corr, nvir_corr)] * t1[S(k, b, nvir_corr)] * t1[S(j, c, nvir_corr)];
    t2abt1t1 -= t2ab[D(k, l, a, d, nocc_corr, nvir_corr)] * t1[S(j, b, nvir_corr)] * t1[S(i, c, nvir_corr)];
    t2abt1t1 += t2ab[D(i, l, b, d, nocc_corr, nvir_corr)] * t1[S(j, c, nvir_corr)] * t1[S(k, a, nvir_corr)];
    t2abt1t1 += t2ab[D(j, l, b, d, nocc_corr, nvir_corr)] * t1[S(k, c, nvir_corr)] * t1[S(i, a, nvir_corr)];
    t2abt1t1 += t2ab[D(i, l, b, d, nocc_corr, nvir_corr)] * t1[S(j, c, nvir_corr)] * t1[S(j, a, nvir_corr)];
    t2abt1t1 -= t2ab[D(j, l, b, d, nocc_corr, nvir_corr)] * t1[S(i, c, nvir_corr)] * t1[S(k, a, nvir_corr)];
    t2abt1t1 -= t2ab[D(i, l, b, d, nocc_corr, nvir_corr)] * t1[S(k, c, nvir_corr)] * t1[S(j, a, nvir_corr)];
    t2abt1t1 -= t2ab[D(k, l, b, d, nocc_corr, nvir_corr)] * t1[S(j, c, nvir_corr)] * t1[S(i, a, nvir_corr)];
    t2abt1t1 += t2ab[D(i, l, c, d, nocc_corr, nvir_corr)] * t1[S(j, a, nvir_corr)] * t1[S(k, b, nvir_corr)];
    t2abt1t1 += t2ab[D(j, l, c, d, nocc_corr, nvir_corr)] * t1[S(k, a, nvir_corr)] * t1[S(i, b, nvir_corr)];
    t2abt1t1 += t2ab[D(i, l, c, d, nocc_corr, nvir_corr)] * t1[S(j, a, nvir_corr)] * t1[S(j, b, nvir_corr)];
    t2abt1t1 -= t2ab[D(j, l, c, d, nocc_corr, nvir_corr)] * t1[S(i, a, nvir_corr)] * t1[S(k, b, nvir_corr)];
    t2abt1t1 -= t2ab[D(i, l, c, d, nocc_corr, nvir_corr)] * t1[S(k, a, nvir_corr)] * t1[S(j, b, nvir_corr)];
    t2abt1t1 -= t2ab[D(k, l, c, d, nocc_corr, nvir_corr)] * t1[S(j, a, nvir_corr)] * t1[S(i, b, nvir_corr)];
    t2abt1t1 -= t2ab[D(i, l, b, d, nocc_corr, nvir_corr)] * t1[S(j, a, nvir_corr)] * t1[S(k, c, nvir_corr)];
    t2abt1t1 -= t2ab[D(j, l, b, d, nocc_corr, nvir_corr)] * t1[S(k, a, nvir_corr)] * t1[S(i, c, nvir_corr)];
    t2abt1t1 -= t2ab[D(i, l, b, d, nocc_corr, nvir_corr)] * t1[S(j, a, nvir_corr)] * t1[S(j, c, nvir_corr)];
    t2abt1t1 += t2ab[D(j, l, b, d, nocc_corr, nvir_corr)] * t1[S(i, a, nvir_corr)] * t1[S(k, c, nvir_corr)];
    t2abt1t1 += t2ab[D(i, l, b, d, nocc_corr, nvir_corr)] * t1[S(k, a, nvir_corr)] * t1[S(j, c, nvir_corr)];
    t2abt1t1 += t2ab[D(k, l, b, d, nocc_corr, nvir_corr)] * t1[S(j, a, nvir_corr)] * t1[S(i, c, nvir_corr)];
    t2abt1t1 -= t2ab[D(i, l, a, d, nocc_corr, nvir_corr)] * t1[S(j, c, nvir_corr)] * t1[S(k, b, nvir_corr)];
    t2abt1t1 -= t2ab[D(j, l, a, d, nocc_corr, nvir_corr)] * t1[S(k, c, nvir_corr)] * t1[S(i, b, nvir_corr)];
    t2abt1t1 -= t2ab[D(i, l, a, d, nocc_corr, nvir_corr)] * t1[S(j, c, nvir_corr)] * t1[S(j, b, nvir_corr)];
    t2abt1t1 += t2ab[D(j, l, a, d, nocc_corr, nvir_corr)] * t1[S(i, c, nvir_corr)] * t1[S(k, b, nvir_corr)];
    t2abt1t1 += t2ab[D(i, l, a, d, nocc_corr, nvir_corr)] * t1[S(k, c, nvir_corr)] * t1[S(j, b, nvir_corr)];
    t2abt1t1 += t2ab[D(k, l, a, d, nocc_corr, nvir_corr)] * t1[S(j, c, nvir_corr)] * t1[S(i, b, nvir_corr)];
    t2abt1t1 -= t2ab[D(i, l, c, d, nocc_corr, nvir_corr)] * t1[S(j, b, nvir_corr)] * t1[S(k, a, nvir_corr)];
    t2abt1t1 -= t2ab[D(j, l, c, d, nocc_corr, nvir_corr)] * t1[S(k, b, nvir_corr)] * t1[S(i, a, nvir_corr)];
    t2abt1t1 -= t2ab[D(i, l, c, d, nocc_corr, nvir_corr)] * t1[S(j, b, nvir_corr)] * t1[S(j, a, nvir_corr)];
    t2abt1t1 += t2ab[D(j, l, c, d, nocc_corr, nvir_corr)] * t1[S(i, b, nvir_corr)] * t1[S(k, a, nvir_corr)];
    t2abt1t1 += t2ab[D(i, l, c, d, nocc_corr, nvir_corr)] * t1[S(k, b, nvir_corr)] * t1[S(j, a, nvir_corr)];
    t2abt1t1 += t2ab[D(k, l, c, d, nocc_corr, nvir_corr)] * t1[S(j, b, nvir_corr)] * t1[S(i, a, nvir_corr)];

    //t1t1t1t1
    t1t1t1t1 += t1[S(i, a, nvir_corr)] * t1[S(j, b, nvir_corr)] * t1[S(k, c, nvir_corr)] * t1[S(l, d, nvir_corr)];
    t1t1t1t1 += t1[S(j, a, nvir_corr)] * t1[S(k, b, nvir_corr)] * t1[S(i, c, nvir_corr)] * t1[S(l, d, nvir_corr)];
    t1t1t1t1 += t1[S(k, a, nvir_corr)] * t1[S(i, b, nvir_corr)] * t1[S(j, c, nvir_corr)] * t1[S(l, d, nvir_corr)];
    t1t1t1t1 -= t1[S(j, a, nvir_corr)] * t1[S(i, b, nvir_corr)] * t1[S(k, c, nvir_corr)] * t1[S(l, d, nvir_corr)];
    t1t1t1t1 -= t1[S(i, a, nvir_corr)] * t1[S(k, b, nvir_corr)] * t1[S(j, c, nvir_corr)] * t1[S(l, d, nvir_corr)];
    t1t1t1t1 -= t1[S(k, a, nvir_corr)] * t1[S(j, b, nvir_corr)] * t1[S(i, c, nvir_corr)] * t1[S(l, d, nvir_corr)];

    t1t1t1t1 += t1[S(i, b, nvir_corr)] * t1[S(j, c, nvir_corr)] * t1[S(k, a, nvir_corr)] * t1[S(l, d, nvir_corr)];
    t1t1t1t1 += t1[S(j, b, nvir_corr)] * t1[S(k, c, nvir_corr)] * t1[S(i, a, nvir_corr)] * t1[S(l, d, nvir_corr)];
    t1t1t1t1 += t1[S(k, b, nvir_corr)] * t1[S(i, c, nvir_corr)] * t1[S(j, a, nvir_corr)] * t1[S(l, d, nvir_corr)];
    t1t1t1t1 -= t1[S(j, b, nvir_corr)] * t1[S(i, c, nvir_corr)] * t1[S(k, a, nvir_corr)] * t1[S(l, d, nvir_corr)];
    t1t1t1t1 -= t1[S(i, b, nvir_corr)] * t1[S(k, c, nvir_corr)] * t1[S(j, a, nvir_corr)] * t1[S(l, d, nvir_corr)];
    t1t1t1t1 -= t1[S(k, b, nvir_corr)] * t1[S(j, c, nvir_corr)] * t1[S(i, a, nvir_corr)] * t1[S(l, d, nvir_corr)];

    t1t1t1t1 += t1[S(i, c, nvir_corr)] * t1[S(j, a, nvir_corr)] * t1[S(k, b, nvir_corr)] * t1[S(l, d, nvir_corr)];
    t1t1t1t1 += t1[S(j, c, nvir_corr)] * t1[S(k, a, nvir_corr)] * t1[S(i, b, nvir_corr)] * t1[S(l, d, nvir_corr)];
    t1t1t1t1 += t1[S(k, c, nvir_corr)] * t1[S(i, a, nvir_corr)] * t1[S(j, b, nvir_corr)] * t1[S(l, d, nvir_corr)];
    t1t1t1t1 -= t1[S(j, c, nvir_corr)] * t1[S(i, a, nvir_corr)] * t1[S(k, b, nvir_corr)] * t1[S(l, d, nvir_corr)];
    t1t1t1t1 -= t1[S(i, c, nvir_corr)] * t1[S(k, a, nvir_corr)] * t1[S(j, b, nvir_corr)] * t1[S(l, d, nvir_corr)];
    t1t1t1t1 -= t1[S(k, c, nvir_corr)] * t1[S(j, a, nvir_corr)] * t1[S(i, b, nvir_corr)] * t1[S(l, d, nvir_corr)];

    t1t1t1t1 -= t1[S(i, b, nvir_corr)] * t1[S(j, a, nvir_corr)] * t1[S(k, c, nvir_corr)] * t1[S(l, d, nvir_corr)];
    t1t1t1t1 -= t1[S(j, b, nvir_corr)] * t1[S(k, a, nvir_corr)] * t1[S(i, c, nvir_corr)] * t1[S(l, d, nvir_corr)];
    t1t1t1t1 -= t1[S(k, b, nvir_corr)] * t1[S(i, a, nvir_corr)] * t1[S(j, c, nvir_corr)] * t1[S(l, d, nvir_corr)];
    t1t1t1t1 += t1[S(j, b, nvir_corr)] * t1[S(i, a, nvir_corr)] * t1[S(k, c, nvir_corr)] * t1[S(l, d, nvir_corr)];
    t1t1t1t1 += t1[S(i, b, nvir_corr)] * t1[S(k, a, nvir_corr)] * t1[S(j, c, nvir_corr)] * t1[S(l, d, nvir_corr)];
    t1t1t1t1 += t1[S(k, b, nvir_corr)] * t1[S(j, a, nvir_corr)] * t1[S(i, c, nvir_corr)] * t1[S(l, d, nvir_corr)];

    t1t1t1t1 -= t1[S(i, a, nvir_corr)] * t1[S(j, c, nvir_corr)] * t1[S(k, b, nvir_corr)] * t1[S(l, d, nvir_corr)];
    t1t1t1t1 -= t1[S(j, a, nvir_corr)] * t1[S(k, c, nvir_corr)] * t1[S(i, b, nvir_corr)] * t1[S(l, d, nvir_corr)];
    t1t1t1t1 -= t1[S(k, a, nvir_corr)] * t1[S(i, c, nvir_corr)] * t1[S(j, b, nvir_corr)] * t1[S(l, d, nvir_corr)];
    t1t1t1t1 += t1[S(j, a, nvir_corr)] * t1[S(i, c, nvir_corr)] * t1[S(k, b, nvir_corr)] * t1[S(l, d, nvir_corr)];
    t1t1t1t1 += t1[S(i, a, nvir_corr)] * t1[S(k, c, nvir_corr)] * t1[S(j, b, nvir_corr)] * t1[S(l, d, nvir_corr)];
    t1t1t1t1 += t1[S(k, a, nvir_corr)] * t1[S(j, c, nvir_corr)] * t1[S(i, b, nvir_corr)] * t1[S(l, d, nvir_corr)];

    t1t1t1t1 -= t1[S(i, c, nvir_corr)] * t1[S(j, b, nvir_corr)] * t1[S(k, a, nvir_corr)] * t1[S(l, d, nvir_corr)];
    t1t1t1t1 -= t1[S(j, c, nvir_corr)] * t1[S(k, b, nvir_corr)] * t1[S(i, a, nvir_corr)] * t1[S(l, d, nvir_corr)];
    t1t1t1t1 -= t1[S(k, c, nvir_corr)] * t1[S(i, b, nvir_corr)] * t1[S(j, a, nvir_corr)] * t1[S(l, d, nvir_corr)];
    t1t1t1t1 += t1[S(j, c, nvir_corr)] * t1[S(i, b, nvir_corr)] * t1[S(k, a, nvir_corr)] * t1[S(l, d, nvir_corr)];
    t1t1t1t1 += t1[S(i, c, nvir_corr)] * t1[S(k, b, nvir_corr)] * t1[S(j, a, nvir_corr)] * t1[S(l, d, nvir_corr)];
    t1t1t1t1 += t1[S(k, c, nvir_corr)] * t1[S(j, b, nvir_corr)] * t1[S(i, a, nvir_corr)] * t1[S(l, d, nvir_corr)];

    return t4p + t1t3aaa + t1t3aab/2.0 + t2t2 + t2aat1t1/2.0 + t2abt1t1 + t1t1t1t1/3.0;
}

void denom_t3_shci(double *t1, double *t2aa, double *t2ab, const int nc, const int nocc, const int nvir, const double numzero, const double c0, double denom) 
{

    const int t1size = nocc*nvir;
#pragma omp parallel default(none) \
        shared(t1, t2aa, t2ab, denom)
{
    int p, q, r, t, u, v, itmp, it, at;
    double t3, parity, scale;

    int i;
#pragma omp for reduction(+ : denom)
    for (i=0; i<omp_get_num_threads(); i++){

        char line[255], typ[4];
        //char *ptr;
        char s0[20]="t3.";
        char s1[4];
        double xaaa, xaab;

        sprintf(s1, "%d", i);
        char* filename = strcat(s0,s1);
        FILE *fp = fopen(filename, "r");
        //printf ("filename = %s\n",filename);

        fp = fopen(filename, "r");
    
        if (fp) {
           while ( !feof(fp) ){

               fscanf(fp, "%c%c%c,%s\n", &(typ[0]), &(typ[1]), &(typ[2]), line);
               fscanf(fp, "%lf\n", &t3);
               if (strncmp(typ, "aaa", 3) == 0 && fabs(t3) > numzero){
                   sscanf(line,"%d,%d,%d,%d,%d,%d",&p,&q,&r,&t,&u,&v);
                   p += nc;
                   q += nc;
                   r += nc;
                   t += - nocc + nc;
                   u += - nocc + nc;
                   v += - nocc + nc;
    
                   parity = parity_ci_to_cc(p+q+r, 3, nocc);
       
                   // interm norm of c3
                   t3 = parity * t3 / c0;
                   // extract t3
                   t3-= t1xt2aaa (p, q, r, t, u, v, nocc, nvir, t1, t2aa); 
                   t3-= t1xt1xt1aaa (p, q, r, t, u, v, nocc, nvir, t1); 

                   xaaa = permut_value_xaaa (p, q, r, t, u, v, nocc, nvir, t3, t1, t2aa);
                   denom += t3 * xaaa;    
    
               }
               else if (strncmp(typ, "aab", 3) == 0 && fabs(t3) > numzero){
                   sscanf(line,"%d,%d,%d,%d,%d,%d",&p,&q,&t,&u,&r,&v);
                   //printf("'%s, %d, %d, %d, %d, %d, %d, %15.8f'\n", typ, p,q,t,u,r,v,t3);
    
                   p += nc;
                   q += nc;
                   r += nc;
                   t += - nocc + nc;
                   u += - nocc + nc;
                   v += - nocc + nc;
    
                   //lsh test
                   //if(!(p == 2 && q == 3 && r == 3 && \
                   //     t == 0 && u == 1 && v == 1)) continue;
                   //printf("c3 in OTF: %15.8f\n",t3);
    
                   parity = parity_ci_to_cc(p+q, 2, nocc);
                   parity *= parity_ci_to_cc(r, 1, nocc);
       
                   // interm norm of c3
                   t3 = parity * t3 / c0;
       
                   // extract t3 
                   t3-= t1xt2aab(p, q, r, t, u, v, nocc, nvir, t1, t2aa, t2ab); 
                   t3-= t1xt1xt1aab(p, q, r, t, u, v, nocc, nvir, t1); 

                   xaab = permut_value_xaab (p, q, r, t, u, v, nocc, nvir, t3, t1, t2aa, t2ab);
                   denom += t3 * xaab;       
    
               }
           }
           fclose(fp);
        }
        else
        {
           // error message
        }

    }

}

}

void denom_t4_shci(double *t1, double *t2aa, double *t2ab, double *c3aaa, double *c3aab, double *paaa, double *paab, const int nocc_iact, const int nocc_corr, const int nvir_corr, const int nocc_cas, const int nvir_cas, const double numzero, const double c0, double denom) 
{
    //double numzero = 1e-7;
    //numzero = 1e-3;

    const int nocc2 = (int) nocc_cas*(nocc_cas-1)/2;
    const int nocc3 = (int) nocc_cas*(nocc_cas-1)*(nocc_cas-2)/6;

    const int t2size = nocc_corr*nocc_corr*nvir_corr*nvir_corr;

#pragma omp parallel default(none) \
        shared(t1, t2aa, t2ab, c3aaa, c3aab, paaa, paab, denom)
{
       double t4, parity, scale, xaaab, xaabb;
       int p, q, r, s, t, u, v, w, itmp, it, jt, at, bt, ifile;
       char typ[4], line[255];

       int i;
#pragma omp for reduction(+ : denom)
       for (i=0; i<omp_get_num_threads(); i++){
           char s0[20]="t4.";
           char s1[4];
           sprintf(s1, "%d", i);
           char* filename = strcat(s0,s1);
           FILE *fp = fopen(filename, "r");
           //printf ("filename = %s\n",filename);

           if (fp) {
           while ( !feof(fp) ){

           fscanf(fp, "%c%c%c%c,%s\n", &(typ[0]), &(typ[1]), &(typ[2]), &(typ[3]), line);
           fscanf(fp, "%lf\n", &t4);
           //lsh test
           //printf ("typ=%c%c%c%c line=%s\n",typ[0],typ[1],typ[2],typ[3], line);
           if (strncmp(typ, "aabb", 4) == 0 && fabs(t4) > numzero){
               sscanf(line,"%d,%d,%d,%d,%d,%d,%d,%d",&p,&q,&t,&u,&r,&s,&v,&w);
               p += nocc_iact;
               q += nocc_iact;
               r += nocc_iact;
               s += nocc_iact;
               t += - nocc_cas;
               u += - nocc_cas;
               v += - nocc_cas;
               w += - nocc_cas;

//               if(!(p == 2 && q == 3 && t == 0 && u == 1 && \
                    r == 2 && s == 3 && v == 0 && w == 1)) continue;
//               if(!((p == 2 && q == 3 && t == 0 && u == 4 && \
//                    r == 1 && s == 3 && v == 0 && w == 4) || \
//                   (p == 1 && q == 3 && t == 0 && u == 4 && \
//                    r == 2 && s == 3 && v == 0 && w == 4)) ) continue;

//               if(!(p == 2 && q == 3 && t == 0 && u == 2 && \
//                    r == 2 && s == 3 && v == 0 && w == 2)) continue;
   
               parity = parity_ci_to_cc(p+q, 2, nocc_corr);
               parity *= parity_ci_to_cc(r+s, 2, nocc_corr);
   
               // interm norm of c4
               t4 = parity * t4 / c0;
               // lsh test 
   
               // extract t4 
               t4-= t1xc3aabb_mem(p, q, r, s, t, u, v, w, nocc_corr, nvir_corr, nocc_cas, nvir_cas, nocc_iact, nocc2, t1, t2aa, t2ab, c3aab, c0); 
               t4-= t2xt2aabb(p, q, r, s, t, u, v, w, nocc_corr, nvir_corr, t2aa, t2ab); 
               t4-= t1xt1xt2aabb(p, q, r, s, t, u, v, w, nocc_corr, nvir_corr, t1, t2aa, t2ab); 
               t4-= t1xt1xt1xt1aabb(p, q, r, s, t, u, v, w, nocc_corr, nvir_corr, t1);   // may have bug 

               xaabb = permut_value_xaabb (p, q, r, s, t, u, v, w, nocc_corr, nvir_corr, nocc_cas, nvir_cas, nocc_iact, nocc2, t4, t1, t2aa, t2ab, c3aab, paab, c0);
               denom += t4 * xaabb;    
   
           }
           else if (strncmp(typ, "aaab", 4) == 0 && fabs(t4) > numzero){
               //lsh test
               //printf ("typ=%c%c%c%c line=%s c4=%lf\n",typ[0],typ[1],typ[2],typ[3], line, t4);

               sscanf(line,"%d,%d,%d,%d,%d,%d,%d,%d",&p,&q,&r,&t,&u,&v,&s,&w);
               p += nocc_iact;
               q += nocc_iact;
               r += nocc_iact;
               s += nocc_iact;
               t += - nocc_cas;
               u += - nocc_cas;
               v += - nocc_cas;
               w += - nocc_cas;

               //printf ("p=%d, q=%d, r=%d, t=%d, u=%d, v=%d, s=%d, w=%d",p,q,r,t,u,v,s,w);

               parity = parity_ci_to_cc(p+q+r, 3, nocc_corr);
               parity *= parity_ci_to_cc(s, 1, nocc_corr);
   
               // interm norm of c4
               t4 = parity * t4 / c0;
   
               // extract t4 
               t4-= t1xc3aaab_mem (p, q, r, s, t, u, v, w, nocc_corr, nvir_corr, nocc_cas, nvir_cas, nocc_iact, nocc2, nocc3, t1, t2aa, t2ab, c3aaa, c3aab, c0);
               t4-= t2xt2aaab (p, q, r, s, t, u, v, w, nocc_corr, nvir_corr, t2aa, t2ab);         // may have 1e-3 bug 
               t4-= t1xt1xt2aaab (p, q, r, s, t, u, v, w, nocc_corr, nvir_corr, t1, t2aa, t2ab);  // may have 1e-5 bug 
               t4-= t1xt1xt1xt1aaab (p, q, r, s, t, u, v, w, nocc_corr, nvir_corr, t1);           // may have 1e-6 bug

               xaaab = permut_value_xaaab (p, q, r, s, t, u, v, w, nocc_corr, nvir_corr, nocc_cas, nvir_cas, nocc_iact, nocc2, nocc3, t4, t1, t2aa, t2ab, c3aaa, c3aab, paaa, paab, c0);
               denom += t4 * xaaab;      

           }

           }
           fclose(fp);
           }
       }

}

}
