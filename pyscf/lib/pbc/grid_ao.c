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

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include "config.h"
#include "cint.h"
#include "vhf/fblas.h"
#include "gto/grid_ao_drv.h"

#define MIN(X,Y)        ((X)<(Y)?(X):(Y))
#define MAX(X,Y)        ((X)>(Y)?(X):(Y))
#define ALL_IMAGES      255
#define IMGBLK          40
#define OF_CMPLX        2

double CINTcommon_fac_sp(int l);
void GTOshell_eval_grid_cart(double *gto, double *ri, double *exps,
                             double *coord, double *alpha, double *coeff, double *env,
                             int l, int np, int nc, size_t nao, size_t ngrids, size_t bgrids);
void GTOshell_eval_grid_cart_deriv1(double *gto, double *ri, double *exps,
                                    double *coord, double *alpha, double *coeff, double *env,
                                    int l, int np, int nc, size_t nao, size_t ngrids, size_t bgrids);
void GTOshell_eval_grid_cart_deriv2(double *cgto, double *ri, double *exps,
                                    double *coord, double *alpha, double *coeff, double *env,
                                    int l, int np, int nc, size_t nao, size_t ngrids, size_t bgrids);
void GTOshell_eval_grid_cart_deriv3(double *cgto, double *ri, double *exps,
                                    double *coord, double *alpha, double *coeff, double *env,
                                    int l, int np, int nc, size_t nao, size_t ngrids, size_t bgrids);
void GTOshell_eval_grid_cart_deriv4(double *cgto, double *ri, double *exps,
                                    double *coord, double *alpha, double *coeff, double *env,
                                    int l, int np, int nc, size_t nao, size_t ngrids, size_t bgrids);
void GTOshell_eval_grid_cart(double *gto, double *ri, double *exps,
                             double *coord, double *alpha, double *coeff,
                             double *env, int l, int np, int nc,
                             size_t nao, size_t ngrids, size_t bgrids);
void GTOshell_eval_grid_ip_cart(double *gto, double *ri, double *exps,
                                double *coord, double *alpha, double *coeff,
                                double *env, int l, int np, int nc,
                                size_t nao, size_t ngrids, size_t bgrids);

/*
 * Extend the meaning of non0table:  given shell ID and block ID,
 * non0table is the number of images in Ls that does not vanish.
 * Ls should be sorted based on the distance to center cell.
 */
void PBCnr_ao_screen(unsigned char *non0table, double *coords, int ngrids,
                     double *Ls, int nimgs,
                     int *atm, int natm, int *bas, int nbas, double *env)
{
        const int nblk = (ngrids+BLKSIZE-1) / BLKSIZE;

#pragma omp parallel
{
        int i, j, m;
        int np, nc, atm_id;
        size_t bas_id, ib;
        double rr, arr, maxc;
        double logcoeff[NPRIMAX];
        double dr[3];
        double rL[3];
        double *p_exp, *pcoeff, *ratm;
#pragma omp for nowait schedule(dynamic)
        for (bas_id = 0; bas_id < nbas; bas_id++) {
                np = bas[NPRIM_OF+bas_id*BAS_SLOTS];
                nc = bas[NCTR_OF +bas_id*BAS_SLOTS];
                p_exp = env + bas[PTR_EXP+bas_id*BAS_SLOTS];
                pcoeff = env + bas[PTR_COEFF+bas_id*BAS_SLOTS];
                atm_id = bas[ATOM_OF+bas_id*BAS_SLOTS];
                ratm = env + atm[atm_id*ATM_SLOTS+PTR_COORD];

                for (j = 0; j < np; j++) {
                        maxc = 0;
                        for (i = 0; i < nc; i++) {
                                maxc = MAX(maxc, fabs(pcoeff[i*np+j]));
                        }
                        logcoeff[j] = log(maxc);
                }

                for (ib = 0; ib < nblk; ib++) {
                for (m = nimgs-1; m >= 0; m--) {
                        rL[0] = ratm[0] + Ls[m*3+0];
                        rL[1] = ratm[1] + Ls[m*3+1];
                        rL[2] = ratm[2] + Ls[m*3+2];
                        for (i = ib*BLKSIZE; i < MIN(ngrids, (ib+1)*BLKSIZE); i++) {
                                dr[0] = coords[0*ngrids+i] - rL[0];
                                dr[1] = coords[1*ngrids+i] - rL[1];
                                dr[2] = coords[2*ngrids+i] - rL[2];
                                rr = dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2];
                                for (j = 0; j < np; j++) {
                                        arr = p_exp[j] * rr;
                                        if (arr-logcoeff[j] < EXPCUTOFF) {
                                                non0table[ib*nbas+bas_id] = MIN(ALL_IMAGES, m+1);
                                                goto next_blk;
                                        }
                                }
                        }
                }
                non0table[ib*nbas+bas_id] = 0;
next_blk:;
                }
        }
}
}

static void _copy(double complex *out, double *ao_k,
                  size_t ngrids, size_t bgrids,
                  int nkpts, int ncomp, int nao, int ncol)
{
        int i, j, k, ic;
        double complex *pout;
        double *ao_r, *ao_i;
        int blksize = ncomp * ncol * bgrids;
        for (k = 0; k < nkpts; k++) {
                ao_r = ao_k + k*2 * blksize;
                ao_i = ao_k +(k*2+1) * blksize;
                for (ic = 0; ic < ncomp; ic++) {
                        pout = out + (k * ncomp + ic) * nao * ngrids;
                        for (j = 0; j < ncol; j++) {
                        for (i = 0; i < bgrids; i++) {
                                pout[j*ngrids+i] = (ao_r[j*bgrids+i] +
                                                    ao_i[j*bgrids+i]*_Complex_I);
                        } }
                        ao_r += ncol * bgrids;
                        ao_i += ncol * bgrids;
                }
        }
}

// grid2atm[nimgs,xyz,grid_id]
static void _fill_grid2atm(double *grid2atm, double *min_grid2atm,
                           double *coord, double *Ls, double *r_atm,
                           int atm_imag_max, size_t bgrids, size_t ngrids, int nimgs)
{
        int ig, m;
        double rL[3];
        double dist;
        double dist_min;
        for (m = 0; m < nimgs; m++) {
                if ((m < atm_imag_max || atm_imag_max == ALL_IMAGES)) {
                        rL[0] = r_atm[0] + Ls[m*3+0];
                        rL[1] = r_atm[1] + Ls[m*3+1];
                        rL[2] = r_atm[2] + Ls[m*3+2];
                        dist_min = 1e9;
                        for (ig = 0; ig < bgrids; ig++) {
                                grid2atm[0*BLKSIZE+ig] = coord[0*ngrids+ig] - rL[0];
                                grid2atm[1*BLKSIZE+ig] = coord[1*ngrids+ig] - rL[1];
                                grid2atm[2*BLKSIZE+ig] = coord[2*ngrids+ig] - rL[2];

                                dist = (grid2atm[0*BLKSIZE+ig]*grid2atm[0*BLKSIZE+ig] +
                                        grid2atm[1*BLKSIZE+ig]*grid2atm[1*BLKSIZE+ig] +
                                        grid2atm[2*BLKSIZE+ig]*grid2atm[2*BLKSIZE+ig]);
                                dist_min = MIN(dist, dist_min);
                        }
                        min_grid2atm[m] = sqrt(dist_min);
                }
                grid2atm += 3*BLKSIZE;
        }
}


void PBCeval_cart_iter(FPtr_eval feval,  FPtr_exp fexp,
                       size_t nao, size_t ngrids, size_t bgrids, size_t offao,
                       int param[], int *shls_slice, int *ao_loc, double *buf,
                       double *Ls, double complex *expLk,
                       int nimgs, int nkpts, int di_max, double complex *ao,
                       double *coord, double *rcut, unsigned char *non0table,
                       int *atm, int natm, int *bas, int nbas, double *env)
{
        const int ncomp = param[TENSOR];
        const int sh0 = shls_slice[0];
        const int sh1 = shls_slice[1];

        const char TRANS_N = 'N';
        const char TRANS_T = 'T';
        const double D1 = 1;
        const int nkpts2 = nkpts * OF_CMPLX;

        int i, j, k, l, np, nc, atm_id, bas_id, deg, ao_id;
        int iL, iL0, iLcount, dimc;
        int grid2atm_atm_id, count;
        double fac;
        double *p_exp, *pcoeff, *pcoord, *pao, *ri;
        double *grid2atm = buf; // shape [nimgs,3,bgrids]
        double *eprim = grid2atm + nimgs*3*BLKSIZE;
        double *aobuf = eprim + NPRIMAX*BLKSIZE*2;
        double *aobufk = aobuf + IMGBLK*ncomp*di_max*bgrids;
        double *Lk_buf = aobufk + nkpts*ncomp*di_max*bgrids * OF_CMPLX;
        double complex *zLk_buf = (double complex *)Lk_buf;
        double *min_grid2atm = Lk_buf + IMGBLK * nkpts * OF_CMPLX;
        double *pexpLk;
        int img_idx[nimgs];
        int atm_imag_max[natm];

        for (i = 0; i < natm; i++) {
                atm_imag_max[i] = 0;
        }
        for (bas_id = sh0; bas_id < sh1; bas_id++) {
                atm_id = bas[bas_id*BAS_SLOTS+ATOM_OF];
                atm_imag_max[atm_id] = MAX(atm_imag_max[atm_id], non0table[bas_id]);
        }

        grid2atm_atm_id = -1;
        for (bas_id = sh0; bas_id < sh1; bas_id++) {
                np = bas[bas_id*BAS_SLOTS+NPRIM_OF];
                nc = bas[bas_id*BAS_SLOTS+NCTR_OF ];
                l  = bas[bas_id*BAS_SLOTS+ANG_OF  ];
                deg = (l+1)*(l+2)/2;
                dimc = nc*deg * ncomp * bgrids;
                fac = CINTcommon_fac_sp(l);
                p_exp  = env + bas[bas_id*BAS_SLOTS+PTR_EXP];
                pcoeff = env + bas[bas_id*BAS_SLOTS+PTR_COEFF];
                atm_id = bas[bas_id*BAS_SLOTS+ATOM_OF];
                ri = env + atm[PTR_COORD+atm_id*ATM_SLOTS];
                ao_id = ao_loc[bas_id] - ao_loc[sh0];

                if (grid2atm_atm_id != atm_id) {
                        _fill_grid2atm(grid2atm, min_grid2atm, coord, Ls, ri,
                                       atm_imag_max[atm_id], bgrids, ngrids, nimgs);
                        grid2atm_atm_id = atm_id;
                }

                for (i = 0; i < nkpts2*dimc; i++) {
                        aobufk[i] = 0;
                }
                for (iL0 = 0; iL0 < nimgs; iL0+=IMGBLK) {
                        iLcount = MIN(IMGBLK, nimgs - iL0);

                        count = 0;
                        for (iL = iL0; iL < iL0+iLcount; iL++) {

        pcoord = grid2atm + iL * 3*BLKSIZE;
        if ((iL < non0table[bas_id] || non0table[bas_id] == ALL_IMAGES) &&
            (min_grid2atm[iL] < rcut[bas_id]) &&
            (*fexp)(eprim, pcoord, p_exp, pcoeff, l, np, nc, bgrids, fac)) {
                pao = aobuf + count * dimc;
                (*feval)(pao, ri, eprim, pcoord, p_exp, pcoeff, env,
                         l, np, nc, nc*deg, bgrids, bgrids);
                img_idx[count] = iL;
                count += 1;
        }
                        }

                        if (count > 0) {
        if (img_idx[count-1] != iL0 + count-1) {
                // some images are skipped
                for (i = 0; i < count; i++) {
                        j = img_idx[i];
                        for (k = 0; k < nkpts; k++) {
                                zLk_buf[i*nkpts+k] = expLk[j*nkpts+k];
                        }
                }
                pexpLk = Lk_buf;
        } else {
                pexpLk = (double *)(expLk + nkpts * iL0);
        }
        dgemm_(&TRANS_N, &TRANS_T, &dimc, &nkpts2, &count,
               &D1, aobuf, &dimc, pexpLk, &nkpts2, &D1, aobufk, &dimc);
                        }
                }

                _copy(ao+ao_id*ngrids+offao, aobufk,
                      ngrids, bgrids, nkpts, ncomp, nao, nc*deg);
        }
}


void PBCeval_sph_iter(FPtr_eval feval,  FPtr_exp fexp,
                      size_t nao, size_t ngrids, size_t bgrids, size_t offao,
                      int param[], int *shls_slice, int *ao_loc, double *buf,
                      double *Ls, double complex *expLk,
                      int nimgs, int nkpts, int di_max, double complex *ao,
                      double *coord, double *rcut, unsigned char *non0table,
                      int *atm, int natm, int *bas, int nbas, double *env)
{
        const int ncomp = param[TENSOR];
        const int sh0 = shls_slice[0];
        const int sh1 = shls_slice[1];

        const char TRANS_N = 'N';
        const char TRANS_T = 'T';
        const double D1 = 1;
        const int nkpts2 = nkpts * OF_CMPLX;

        int i, j, k, l, np, nc, atm_id, bas_id, deg, dcart, ao_id;
        int iL, iL0, iLcount, dimc;
        int grid2atm_atm_id, count;
        double fac;
        double *p_exp, *pcoeff, *pcoord, *pcart, *pao, *ri;
        double *grid2atm = buf; // shape [nimgs,3,bgrids]
        double *eprim = grid2atm + nimgs*3*BLKSIZE;
        double *aobuf = eprim + NPRIMAX*BLKSIZE*2;
        double *aobufk = aobuf + IMGBLK*ncomp*di_max*bgrids;
        double *Lk_buf = aobufk + nkpts*ncomp*di_max*bgrids * OF_CMPLX;
        double complex *zLk_buf = (double complex *)Lk_buf;
        double *cart_gto = Lk_buf + IMGBLK * nkpts * OF_CMPLX;
        double *min_grid2atm = cart_gto + ncomp*NCTR_CART*bgrids;
        double *pexpLk;
        int img_idx[nimgs];
        int atm_imag_max[natm];

        for (i = 0; i < natm; i++) {
                atm_imag_max[i] = 0;
        }
        for (bas_id = sh0; bas_id < sh1; bas_id++) {
                atm_id = bas[bas_id*BAS_SLOTS+ATOM_OF];
                atm_imag_max[atm_id] = MAX(atm_imag_max[atm_id], non0table[bas_id]);
        }

        grid2atm_atm_id = -1;
        for (bas_id = sh0; bas_id < sh1; bas_id++) {
                np = bas[bas_id*BAS_SLOTS+NPRIM_OF];
                nc = bas[bas_id*BAS_SLOTS+NCTR_OF ];
                l  = bas[bas_id*BAS_SLOTS+ANG_OF  ];
                deg = l * 2 + 1;
                dcart = (l+1)*(l+2)/2;
                dimc = nc*deg * ncomp * bgrids;
                fac = CINTcommon_fac_sp(l);
                p_exp  = env + bas[bas_id*BAS_SLOTS+PTR_EXP];
                pcoeff = env + bas[bas_id*BAS_SLOTS+PTR_COEFF];
                atm_id = bas[bas_id*BAS_SLOTS+ATOM_OF];
                ri = env + atm[PTR_COORD+atm_id*ATM_SLOTS];
                ao_id = ao_loc[bas_id] - ao_loc[sh0];

                if (grid2atm_atm_id != atm_id) {
                        _fill_grid2atm(grid2atm, min_grid2atm, coord, Ls, ri,
                                       atm_imag_max[atm_id], bgrids, ngrids, nimgs);
                        grid2atm_atm_id = atm_id;
                }

                for (i = 0; i < nkpts2*dimc; i++) {
                        aobufk[i] = 0;
                }
                for (iL0 = 0; iL0 < nimgs; iL0+=IMGBLK) {
                        iLcount = MIN(IMGBLK, nimgs - iL0);

                        count = 0;
                        for (iL = iL0; iL < iL0+iLcount; iL++) {

        pcoord = grid2atm + iL * 3*BLKSIZE;
        if ((iL < non0table[bas_id] || non0table[bas_id] == ALL_IMAGES) &&
            (min_grid2atm[iL] < rcut[bas_id]) &&
            (*fexp)(eprim, pcoord, p_exp, pcoeff, l, np, nc, bgrids, fac)) {
                pao = aobuf + count * dimc;
                if (l <= 1) { // s, p functions
                        (*feval)(pao, ri, eprim, pcoord, p_exp, pcoeff, env,
                                 l, np, nc, nc*dcart, bgrids, bgrids);
                } else {
                        (*feval)(cart_gto, ri, eprim, pcoord, p_exp, pcoeff, env,
                                 l, np, nc, nc*dcart, bgrids, bgrids);
                        pcart = cart_gto;
                        for (i = 0; i < ncomp * nc; i++) {
                                CINTc2s_ket_sph1(pao, pcart, bgrids, bgrids, l);
                                pao += deg * bgrids;
                                pcart += dcart * bgrids;
                        }
                }

                img_idx[count] = iL;
                count++;
        }
                        }

                        if (count > 0) {
        if (img_idx[count-1] != iL0 + count-1) {
                // some images are skipped
                for (i = 0; i < count; i++) {
                        j = img_idx[i];
                        for (k = 0; k < nkpts; k++) {
                                zLk_buf[i*nkpts+k] = expLk[j*nkpts+k];
                        }
                }
                pexpLk = Lk_buf;
        } else {
                pexpLk = (double *)(expLk + nkpts * iL0);
        }
        dgemm_(&TRANS_N, &TRANS_T, &dimc, &nkpts2, &count,
               &D1, aobuf, &dimc, pexpLk, &nkpts2, &D1, aobufk, &dimc);
                        }
                }

                _copy(ao+ao_id*ngrids+offao, aobufk,
                      ngrids, bgrids, nkpts, ncomp, nao, nc*deg);
        }
}


int GTOshloc_by_atom(int *shloc, int *shls_slice, int *ao_loc, int *atm, int *bas);
/*
 * blksize <= 1024 to avoid stack overflow
 *
 * non0table[ngrids/blksize,natm] is the T/F table for ao values to
 * screen the ao evaluation for each shell
 */
void PBCeval_loop(void (*fiter)(), FPtr_eval feval, FPtr_exp fexp,
                  int ngrids, int param[], int *shls_slice, int *ao_loc,
                  double *Ls, int nimgs, double complex *expLk, int nkpts,
                  double complex *ao, double *coord,
                  double *rcut, unsigned char *non0table,
                  int *atm, int natm, int *bas, int nbas, double *env)
{
        int shloc[shls_slice[1]-shls_slice[0]+1];
        const int nshblk = GTOshloc_by_atom(shloc, shls_slice, ao_loc, atm, bas);
        const int nblk = (ngrids+BLKSIZE-1) / BLKSIZE;
        const size_t Ngrids = ngrids;

        int i;
        int di_max = 0;
        for (i = shls_slice[0]; i < shls_slice[1]; i++) {
                di_max = MAX(di_max, ao_loc[i+1] - ao_loc[i]);
        }

#pragma omp parallel
{
        const int sh0 = shls_slice[0];
        const int sh1 = shls_slice[1];
        const size_t nao = ao_loc[sh1] - ao_loc[sh0];
        int ip, ib, k, iloc, ish;
        size_t aoff, bgrids;
        size_t bufsize =((nimgs*3 + NPRIMAX*2 +
                          nkpts *param[POS_E1]*param[TENSOR]*di_max * OF_CMPLX +
                          IMGBLK*param[POS_E1]*param[TENSOR]*di_max +
                          param[POS_E1]*param[TENSOR]*NCTR_CART) * BLKSIZE
                         + nkpts * IMGBLK * OF_CMPLX + nimgs);
        double *buf = malloc(sizeof(double) * bufsize);
#pragma omp for nowait schedule(dynamic, 1)
        for (k = 0; k < nblk*nshblk; k++) {
                iloc = k / nblk;
                ish = shloc[iloc];
                ib = k - iloc * nblk;
                ip = ib * BLKSIZE;
                aoff = (ao_loc[ish] - ao_loc[sh0]) * Ngrids + ip;
                bgrids = MIN(ngrids-ip, BLKSIZE);
                (*fiter)(feval, fexp, nao, Ngrids, bgrids, aoff,
                         param, shloc+iloc, ao_loc, buf,
                         Ls, expLk, nimgs, nkpts, di_max,
                         ao, coord+ip, rcut, non0table+ib*nbas,
                         atm, natm, bas, nbas, env);
        }
        free(buf);
}
}

void PBCeval_cart_drv(FPtr_eval feval, FPtr_exp fexp,
                      int ngrids, int param[], int *shls_slice, int *ao_loc,
                      double *Ls, int nimgs, double complex *expLk, int nkpts,
                      double complex *ao, double *coord,
                      double *rcut, unsigned char *non0table,
                      int *atm, int natm, int *bas, int nbas, double *env)
{
        PBCeval_loop(PBCeval_cart_iter, feval, fexp,
                     ngrids, param, shls_slice, ao_loc, Ls, nimgs, expLk, nkpts,
                     ao, coord, rcut, non0table, atm, natm, bas, nbas, env);
}

void PBCeval_sph_drv(FPtr_eval feval, FPtr_exp fexp,
                     int ngrids, int param[], int *shls_slice, int *ao_loc,
                     double *Ls, int nimgs, double complex *expLk, int nkpts,
                     double complex *ao, double *coord,
                     double *rcut, unsigned char *non0table,
                     int *atm, int natm, int *bas, int nbas, double *env)
{
        PBCeval_loop(PBCeval_sph_iter, feval, fexp,
                     ngrids, param, shls_slice, ao_loc, Ls, nimgs, expLk, nkpts,
                     ao, coord, rcut, non0table, atm, natm, bas, nbas, env);
}

void PBCGTOval_cart_deriv0(int ngrids, int *shls_slice, int *ao_loc,
                           double *Ls, int nimgs, double complex *expLk, int nkpts,
                           double complex *ao, double *coord,
                           double *rcut, unsigned char *non0table,
                           int *atm, int natm, int *bas, int nbas, double *env)
{
        int param[] = {1, 1};
        PBCeval_cart_drv(GTOshell_eval_grid_cart, GTOcontract_exp0,
                         ngrids, param, shls_slice, ao_loc, Ls, nimgs, expLk, nkpts,
                         ao, coord, rcut, non0table, atm, natm, bas, nbas, env);
}

void PBCGTOval_sph_deriv0(int ngrids, int *shls_slice, int *ao_loc,
                          double *Ls, int nimgs, double complex *expLk, int nkpts,
                          double complex *ao, double *coord,
                          double *rcut, unsigned char *non0table,
                          int *atm, int natm, int *bas, int nbas, double *env)
{
        int param[] = {1, 1};
        PBCeval_sph_drv(GTOshell_eval_grid_cart, GTOcontract_exp0,
                        ngrids, param, shls_slice, ao_loc, Ls, nimgs, expLk, nkpts,
                        ao, coord, rcut, non0table, atm, natm, bas, nbas, env);
}

void PBCGTOval_cart_deriv1(int ngrids, int *shls_slice, int *ao_loc,
                           double *Ls, int nimgs, double complex *expLk, int nkpts,
                           double complex *ao, double *coord,
                           double *rcut, unsigned char *non0table,
                           int *atm, int natm, int *bas, int nbas, double *env)
{
        int param[] = {1, 4};
        PBCeval_cart_drv(GTOshell_eval_grid_cart_deriv1, GTOcontract_exp1,
                         ngrids, param, shls_slice, ao_loc, Ls, nimgs, expLk, nkpts,
                         ao, coord, rcut, non0table, atm, natm, bas, nbas, env);
}

void PBCGTOval_sph_deriv1(int ngrids, int *shls_slice, int *ao_loc,
                          double *Ls, int nimgs, double complex *expLk, int nkpts,
                          double complex *ao, double *coord,
                          double *rcut, unsigned char *non0table,
                          int *atm, int natm, int *bas, int nbas, double *env)
{
        int param[] = {1, 4};
        PBCeval_sph_drv(GTOshell_eval_grid_cart_deriv1, GTOcontract_exp1,
                        ngrids, param, shls_slice, ao_loc, Ls, nimgs, expLk, nkpts,
                        ao, coord, rcut, non0table, atm, natm, bas, nbas, env);
}

void PBCGTOval_cart_deriv2(int ngrids, int *shls_slice, int *ao_loc,
                           double *Ls, int nimgs, double complex *expLk, int nkpts,
                           double complex *ao, double *coord,
                           double *rcut, unsigned char *non0table,
                           int *atm, int natm, int *bas, int nbas, double *env)
{
        int param[] = {1, 10};
        PBCeval_cart_drv(GTOshell_eval_grid_cart_deriv2, GTOprim_exp,
                         ngrids, param, shls_slice, ao_loc, Ls, nimgs, expLk, nkpts,
                         ao, coord, rcut, non0table, atm, natm, bas, nbas, env);
}

void PBCGTOval_sph_deriv2(int ngrids, int *shls_slice, int *ao_loc,
                          double *Ls, int nimgs, double complex *expLk, int nkpts,
                          double complex *ao, double *coord,
                          double *rcut, unsigned char *non0table,
                          int *atm, int natm, int *bas, int nbas, double *env)
{
        int param[] = {1, 10};
        PBCeval_sph_drv(GTOshell_eval_grid_cart_deriv2, GTOprim_exp,
                        ngrids, param, shls_slice, ao_loc, Ls, nimgs, expLk, nkpts,
                        ao, coord, rcut, non0table, atm, natm, bas, nbas, env);
}

void PBCGTOval_cart_deriv3(int ngrids, int *shls_slice, int *ao_loc,
                           double *Ls, int nimgs, double complex *expLk, int nkpts,
                           double complex *ao, double *coord,
                           double *rcut, unsigned char *non0table,
                           int *atm, int natm, int *bas, int nbas, double *env)
{
        int param[] = {1, 20};
        PBCeval_cart_drv(GTOshell_eval_grid_cart_deriv3, GTOprim_exp,
                         ngrids, param, shls_slice, ao_loc, Ls, nimgs, expLk, nkpts,
                         ao, coord, rcut, non0table, atm, natm, bas, nbas, env);
}

void PBCGTOval_sph_deriv3(int ngrids, int *shls_slice, int *ao_loc,
                          double *Ls, int nimgs, double complex *expLk, int nkpts,
                          double complex *ao, double *coord,
                          double *rcut, unsigned char *non0table,
                          int *atm, int natm, int *bas, int nbas, double *env)
{
        int param[] = {1, 20};
        PBCeval_sph_drv(GTOshell_eval_grid_cart_deriv3, GTOprim_exp,
                        ngrids, param, shls_slice, ao_loc, Ls, nimgs, expLk, nkpts,
                        ao, coord, rcut, non0table, atm, natm, bas, nbas, env);
}

void PBCGTOval_cart_deriv4(int ngrids, int *shls_slice, int *ao_loc,
                           double *Ls, int nimgs, double complex *expLk, int nkpts,
                           double complex *ao, double *coord,
                           double *rcut, unsigned char *non0table,
                           int *atm, int natm, int *bas, int nbas, double *env)
{
        int param[] = {1, 35};
        PBCeval_cart_drv(GTOshell_eval_grid_cart_deriv4, GTOprim_exp,
                         ngrids, param, shls_slice, ao_loc, Ls, nimgs, expLk, nkpts,
                         ao, coord, rcut, non0table, atm, natm, bas, nbas, env);
}

void PBCGTOval_sph_deriv4(int ngrids, int *shls_slice, int *ao_loc,
                          double *Ls, int nimgs, double complex *expLk, int nkpts,
                          double complex *ao, double *coord,
                          double *rcut, unsigned char *non0table,
                          int *atm, int natm, int *bas, int nbas, double *env)
{
        int param[] = {1, 35};
        PBCeval_sph_drv(GTOshell_eval_grid_cart_deriv4, GTOprim_exp,
                        ngrids, param, shls_slice, ao_loc, Ls, nimgs, expLk, nkpts,
                        ao, coord, rcut, non0table, atm, natm, bas, nbas, env);
}

void PBCGTOval_cart(int ngrids, int *shls_slice, int *ao_loc,
                    double *Ls, int nimgs, double complex *expLk, int nkpts,
                    double complex *ao, double *coord,
                    double *rcut, unsigned char *non0table,
                    int *atm, int natm, int *bas, int nbas, double *env)
{
//        int param[] = {1, 1};
//        PBCeval_cart_drv(GTOshell_eval_grid_cart, GTOcontract_exp0,
//                         ngrids, param, shls_slice, ao_loc, Ls, nimgs, expLk, nkpts,
//                         ao, coord, rcut, non0table, atm, natm, bas, nbas, env);
        PBCGTOval_cart_deriv0(ngrids, shls_slice, ao_loc, Ls, nimgs, expLk, nkpts,
                              ao, coord, rcut, non0table, atm, natm, bas, nbas, env);
}
void PBCGTOval_sph(int ngrids, int *shls_slice, int *ao_loc,
                   double *Ls, int nimgs, double complex *expLk, int nkpts,
                   double complex *ao, double *coord,
                   double *rcut, unsigned char *non0table,
                   int *atm, int natm, int *bas, int nbas, double *env)
{
//        int param[] = {1, 1};
//        PBCeval_sph_drv(GTOshell_eval_grid_cart, GTOcontract_exp0,
//                        ngrids, param, shls_slice, ao_loc, Ls, nimgs, expLk, nkpts,
//                        ao, coord, rcut, non0table, atm, natm, bas, nbas, env);
        PBCGTOval_sph_deriv0(ngrids, shls_slice, ao_loc, Ls, nimgs, expLk, nkpts,
                             ao, coord, rcut, non0table, atm, natm, bas, nbas, env);
}

void PBCGTOval_ip_cart(int ngrids, int *shls_slice, int *ao_loc,
                       double *Ls, int nimgs, double complex *expLk, int nkpts,
                       double complex *ao, double *coord,
                       double *rcut, unsigned char *non0table,
                       int *atm, int natm, int *bas, int nbas, double *env)
{
        int param[] = {1, 3};
        PBCeval_cart_drv(GTOshell_eval_grid_ip_cart, GTOcontract_exp1,
                         ngrids, param, shls_slice, ao_loc, Ls, nimgs, expLk, nkpts,
                         ao, coord, rcut, non0table, atm, natm, bas, nbas, env);
}
void PBCGTOval_ip_sph(int ngrids, int *shls_slice, int *ao_loc,
                      double *Ls, int nimgs, double complex *expLk, int nkpts,
                      double complex *ao, double *coord,
                      double *rcut, unsigned char *non0table,
                      int *atm, int natm, int *bas, int nbas, double *env)
{
        int param[] = {1, 3};
        PBCeval_sph_drv(GTOshell_eval_grid_ip_cart, GTOcontract_exp1,
                        ngrids, param, shls_slice, ao_loc, Ls, nimgs, expLk, nkpts,
                        ao, coord, rcut, non0table, atm, natm, bas, nbas, env);
}
