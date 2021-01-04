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

import unittest
import numpy
from pyscf.pbc import gto as pgto
from pyscf.pbc import dft as pdft
from pyscf.pbc.df import ft_ao
from pyscf.pbc import tools
from pyscf import lib

cell = pgto.Cell()
cell.atom = '''
He1   1.3    .2       .3
He2    .1    .1      1.1 '''
cell.basis = {'He1': 'sto3g', 'He2': 'ccpvdz'}
cell.mesh = (31,)*3
cell.a = numpy.diag([2.2, 1.9, 2.])
cell.build()

cell1 = pgto.Cell()
cell1.atom = '''
He   1.3    .2       .3
He    .1    .1      1.1 '''
cell1.basis = {'He': [[0, [0.8, 1]],
                      [1, [0.6, 1]]
                     ]}
cell1.mesh = [17]*3
cell1.a = numpy.array(([2.0,  .9, 0. ],
                       [0.1, 1.9, 0.4],
                       [0.8, 0  , 2.1]))
cell1.build()

def finger(a):
    w = numpy.cos(numpy.arange(a.size))
    return numpy.dot(w, a.ravel())

class KnowValues(unittest.TestCase):
    def test_ft_ao(self):
        coords = pdft.gen_grid.gen_uniform_grids(cell)
        aoR = pdft.numint.eval_ao(cell, coords)
        ngrids, nao = aoR.shape
        ref = numpy.asarray([tools.fft(aoR[:,i], cell.mesh) for i in range(nao)])
        ref = ref.T * (cell.vol/ngrids)
        dat = ft_ao.ft_ao(cell, cell.Gv)
        self.assertAlmostEqual(numpy.linalg.norm(ref[:,0]-dat[:,0])  , 8.4358614794095722e-11, 9)
        self.assertAlmostEqual(numpy.linalg.norm(ref[:,1]-dat[:,1])  , 0.0041669297531642616 , 4)
        self.assertAlmostEqual(numpy.linalg.norm(ref[:,2:]-dat[:,2:]), 5.8677286005879366e-14, 9)

        coords = pdft.gen_grid.gen_uniform_grids(cell1)
        aoR = pdft.numint.eval_ao(cell1, coords)
        ngrids, nao = aoR.shape
        ref = numpy.asarray([tools.fft(aoR[:,i], cell1.mesh) for i in range(nao)])
        ref = ref.T * (cell1.vol/ngrids)
        dat = ft_ao.ft_ao(cell1, cell1.Gv)
        self.assertAlmostEqual(numpy.linalg.norm(ref[:,0]-dat[:,0])  , 0, 5)
        self.assertAlmostEqual(numpy.linalg.norm(ref[:,1]-dat[:,1])  , 0, 3)
        self.assertAlmostEqual(numpy.linalg.norm(ref[:,2:]-dat[:,2:]), 0, 3)

    def test_ft_ao_with_kpts(self):
        numpy.random.seed(1)
        kpt = numpy.random.random(3)
        coords = pdft.gen_grid.gen_uniform_grids(cell)
        aoR = pdft.numint.eval_ao(cell, coords, kpt=kpt)
        ngrids, nao = aoR.shape
        expmikr = numpy.exp(-1j*numpy.dot(coords,kpt))
        ref = numpy.asarray([tools.fftk(aoR[:,i], cell.mesh, expmikr) for i in range(nao)])
        ref = ref.T * (cell.vol/ngrids)
        dat = ft_ao.ft_ao(cell, cell.Gv, kpt=kpt)
        self.assertAlmostEqual(numpy.linalg.norm(ref[:,0]-dat[:,0])  , 1.3359899490499813e-10, 9)
        self.assertAlmostEqual(numpy.linalg.norm(ref[:,1]-dat[:,1])  , 0.0042404556036939756 , 4)
        self.assertAlmostEqual(numpy.linalg.norm(ref[:,2:]-dat[:,2:]), 4.8856357999633564e-14, 9)

        coords = pdft.gen_grid.gen_uniform_grids(cell1)
        aoR = pdft.numint.eval_ao(cell1, coords, kpt=kpt)
        ngrids, nao = aoR.shape
        expmikr = numpy.exp(-1j*numpy.dot(coords,kpt))
        ref = numpy.asarray([tools.fftk(aoR[:,i], cell1.mesh, expmikr) for i in range(nao)])
        ref = ref.T * (cell1.vol/ngrids)
        dat = ft_ao.ft_ao(cell1, cell1.Gv, kpt=kpt)
        self.assertAlmostEqual(numpy.linalg.norm(ref[:,0]-dat[:,0])  , 0, 5)
        self.assertAlmostEqual(numpy.linalg.norm(ref[:,1]-dat[:,1])  , 0, 3)
        self.assertAlmostEqual(numpy.linalg.norm(ref[:,2:]-dat[:,2:]), 0, 3)

    def test_ft_aoao(self):
        #coords = pdft.gen_grid.gen_uniform_grids(cell)
        #aoR = pdft.numint.eval_ao(cell, coords)
        #ngrids, nao = aoR.shape
        #ref = numpy.asarray([tools.fft(aoR[:,i].conj()*aoR[:,j], cell.mesh)
        #                     for i in range(nao) for j in range(nao)])
        #ref = ref.reshape(nao,nao,-1).transpose(2,0,1) * (cell.vol/ngrids)
        #dat = ft_ao.ft_aopair(cell, cell.Gv, aosym='s1hermi')
        #self.assertAlmostEqual(numpy.linalg.norm(ref[:,0,0]-dat[:,0,0])    , 0, 5)
        #self.assertAlmostEqual(numpy.linalg.norm(ref[:,1,1]-dat[:,1,1])    , 0.02315483195832373, 4)
        #self.assertAlmostEqual(numpy.linalg.norm(ref[:,2:,2:]-dat[:,2:,2:]), 0, 9)
        #self.assertAlmostEqual(numpy.linalg.norm(ref[:,0,2:]-dat[:,0,2:])  , 0, 9)
        #self.assertAlmostEqual(numpy.linalg.norm(ref[:,2:,0]-dat[:,2:,0])  , 0, 9)
        #idx = numpy.tril_indices(nao)
        #ref = dat[:,idx[0],idx[1]]
        #dat = ft_ao.ft_aopair(cell, cell.Gv, aosym='s2')
        #self.assertAlmostEqual(abs(dat-ref).sum(), 0, 9)

        coords = pdft.gen_grid.gen_uniform_grids(cell1)
        Gv, Gvbase, kws = cell1.get_Gv_weights(cell1.mesh)
        b = cell1.reciprocal_vectors()
        gxyz = lib.cartesian_prod([numpy.arange(len(x)) for x in Gvbase])
        dat = ft_ao.ft_aopair(cell1, cell1.Gv, aosym='s1', b=b,
                              gxyz=gxyz, Gvbase=Gvbase)
        self.assertAlmostEqual(finger(dat), 1.5666516306798806+1.953555017583245j, 9)
        dat = ft_ao.ft_aopair(cell1, cell1.Gv, aosym='s2', b=b,
                              gxyz=gxyz, Gvbase=Gvbase)
        self.assertAlmostEqual(finger(dat), -0.85276967757297917+1.0378751267506394j, 9)
        dat = ft_ao.ft_aopair(cell1, cell1.Gv, aosym='s1hermi', b=b,
                              gxyz=gxyz, Gvbase=Gvbase)
        self.assertAlmostEqual(finger(dat), 1.5666516306798806+1.953555017583245j, 9)
        aoR = pdft.numint.eval_ao(cell1, coords)
        ngrids, nao = aoR.shape
        aoaoR = numpy.einsum('pi,pj->ijp', aoR, aoR)
        ref = tools.fft(aoaoR.reshape(nao*nao,-1), cell1.mesh)
        ref = ref.reshape(nao,nao,-1).transpose(2,0,1) * (cell1.vol/ngrids)
        self.assertAlmostEqual(numpy.linalg.norm(ref[:,0,0]-dat[:,0,0])    , 0, 7)
        self.assertAlmostEqual(numpy.linalg.norm(ref[:,1,1]-dat[:,1,1])    , 0, 7)
        self.assertAlmostEqual(numpy.linalg.norm(ref[:,2:,2:]-dat[:,2:,2:]), 0, 7)
        self.assertAlmostEqual(numpy.linalg.norm(ref[:,0,2:]-dat[:,0,2:])  , 0, 7)
        self.assertAlmostEqual(numpy.linalg.norm(ref[:,2:,0]-dat[:,2:,0])  , 0, 7)
        idx = numpy.tril_indices(nao)
        ref = dat[:,idx[0],idx[1]]
        dat = ft_ao.ft_aopair(cell1, cell1.Gv, aosym='s2')
        self.assertAlmostEqual(abs(dat-ref).sum(), 0, 9)

    def test_ft_aoao_pdotp(self):
        coords = pdft.gen_grid.gen_uniform_grids(cell1)
        Gv, Gvbase, kws = cell1.get_Gv_weights(cell1.mesh)
        dat = ft_ao.ft_aopair(cell1, cell1.Gv, aosym='s1', intor='GTO_ft_pdotp_sph')
        self.assertAlmostEqual(finger(dat), 5.7858606710458078-8.654809509773056j, 9)
        aoR = pdft.numint.eval_ao(cell1, coords, deriv=1)
        ngrids, nao = aoR.shape[1:]
        aoaoR = numpy.einsum('xpi,xpj->ijp', aoR[1:4], aoR[1:4])
        ref = tools.fft(aoaoR.reshape(nao*nao,-1), cell1.mesh)
        ref = ref.reshape(nao,nao,-1).transpose(2,0,1) * (cell1.vol/ngrids)
        self.assertAlmostEqual(abs(ref-dat).max(), 0, 7)

    def test_ft_aoao_pxp(self):
        coords = pdft.gen_grid.gen_uniform_grids(cell1)
        Gv, Gvbase, kws = cell1.get_Gv_weights(cell1.mesh)
        dat = ft_ao.ft_aopair(cell1, cell1.Gv, aosym='s1', intor='GTO_ft_pxp_sph', comp=3)
        self.assertAlmostEqual(finger(dat), (6.4124798727215779-10.673712733378771j), 9)
        aoR = pdft.numint.eval_ao(cell1, coords, deriv=1)
        ngrids, nao = aoR.shape[1:]
        aox, aoy, aoz = aoR[1:]
        aoaoR =(numpy.einsum('pi,pj->ijp', aoy, aoz) - numpy.einsum('pi,pj->ijp', aoz, aoy),
                numpy.einsum('pi,pj->ijp', aoz, aox) - numpy.einsum('pi,pj->ijp', aox, aoz),
                numpy.einsum('pi,pj->ijp', aox, aoy) - numpy.einsum('pi,pj->ijp', aoy, aox))
        ref = tools.fft(numpy.array(aoaoR).reshape(3*nao*nao,-1), cell1.mesh)
        ref = ref.reshape(3,nao,nao,-1).transpose(0,3,1,2) * (cell1.vol/ngrids)
        self.assertAlmostEqual(abs(ref-dat).max(), 0, 7)

    def test_ft_aoao_with_kpts_high_cost(self):
        numpy.random.seed(1)
        kpti, kptj = numpy.random.random((2,3))
        dat = ft_ao.ft_aopair(cell, cell.Gv, kpti_kptj=(kpti,kptj))
        self.assertAlmostEqual(finger(dat), -0.80184732435570638+2.4078835207597176j, 9)
        coords = pdft.gen_grid.gen_uniform_grids(cell)
        aoi = pdft.numint.eval_ao(cell, coords, kpt=kpti)
        aoj = pdft.numint.eval_ao(cell, coords, kpt=kptj)
        ngrids, nao = aoj.shape
        q = kptj - kpti
        expmikr = numpy.exp(-1j*numpy.dot(coords,q))
        ref = numpy.asarray([tools.fftk(aoi[:,i].conj()*aoj[:,j], cell.mesh, expmikr)
                             for i in range(nao) for j in range(nao)])
        ref = ref.reshape(nao,nao,-1).transpose(2,0,1) * (cell.vol/ngrids)
        self.assertAlmostEqual(numpy.linalg.norm(ref[:,0,0]-dat[:,0,0])    , 0, 5)
        self.assertAlmostEqual(numpy.linalg.norm(ref[:,1,1]-dat[:,1,1])    , 0.023225471785938184  , 4)
        self.assertAlmostEqual(numpy.linalg.norm(ref[:,2:,2:]-dat[:,2:,2:]), 0, 9)
        self.assertAlmostEqual(numpy.linalg.norm(ref[:,0,2:]-dat[:,0,2:])  , 0, 9)
        self.assertAlmostEqual(numpy.linalg.norm(ref[:,2:,0]-dat[:,2:,0])  , 0, 9)

    def test_ft_aoao_pair_vs_fft(self):
        numpy.random.seed(1)
        kpti, kptj = numpy.random.random((2,3))
        coords = pdft.gen_grid.gen_uniform_grids(cell1)
        aoi = pdft.numint.eval_ao(cell1, coords, kpt=kpti)
        aoj = pdft.numint.eval_ao(cell1, coords, kpt=kptj)
        ngrids, nao = aoj.shape
        q = kptj - kpti
        dat = ft_ao.ft_aopair(cell1, cell1.Gv, kpti_kptj=(kpti,kptj), q=q)
        self.assertAlmostEqual(finger(dat), 0.72664436503332241+3.2542145296611373j, 9)
        expmikr = numpy.exp(-1j*numpy.dot(coords,q))
        ref = numpy.asarray([tools.fftk(aoi[:,i].conj()*aoj[:,j], cell1.mesh, expmikr)
                             for i in range(nao) for j in range(nao)])
        ref = ref.reshape(nao,nao,-1).transpose(2,0,1) * (cell1.vol/ngrids)
        self.assertAlmostEqual(numpy.linalg.norm(ref[:,0,0]-dat[:,0,0])    , 0, 7)
        self.assertAlmostEqual(numpy.linalg.norm(ref[:,1,1]-dat[:,1,1])    , 0, 7)
        self.assertAlmostEqual(numpy.linalg.norm(ref[:,2:,2:]-dat[:,2:,2:]), 0, 7)
        self.assertAlmostEqual(numpy.linalg.norm(ref[:,0,2:]-dat[:,0,2:])  , 0, 7)
        self.assertAlmostEqual(numpy.linalg.norm(ref[:,2:,0]-dat[:,2:,0])  , 0, 7)

    def test_ft_aoao_with_kpts1(self):
        numpy.random.seed(1)
        kpti, kptj = kpts = numpy.random.random((2,3))
        Gv = cell.get_Gv([11]*3)
        q = numpy.random.random(3)
        dat = ft_ao._ft_aopair_kpts(cell, Gv, q=q, kptjs=kpts)
        self.assertAlmostEqual(finger(dat[0]), (2.3753953914129382-2.5365192689115088j), 9)
        self.assertAlmostEqual(finger(dat[1]), (2.4951510097641840-3.1990956672116355j), 9)
        dat = ft_ao.ft_aopair(cell, Gv)
        self.assertAlmostEqual(finger(dat), (1.2534723618134684+1.830086071817564j), 9)


if __name__ == '__main__':
    print('Full Tests for ft_ao')
    unittest.main()
