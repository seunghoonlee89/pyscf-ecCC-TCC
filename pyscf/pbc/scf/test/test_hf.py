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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import unittest
import numpy
from pyscf import lib
from pyscf.pbc import gto as pbcgto
from pyscf.pbc.scf import hf as pbchf
import pyscf.pbc.scf as pscf
from pyscf.pbc import df as pdf

L = 4
n = 21
cell = pbcgto.Cell()
cell.build(unit = 'B',
           verbose = 7,
           output = '/dev/null',
           a = ((L,0,0),(0,L,0),(0,0,L)),
           mesh = [n,n,n],
           atom = [['He', (L/2.-.5,L/2.,L/2.-.5)],
                   ['He', (L/2.   ,L/2.,L/2.+.5)]],
           basis = { 'He': [[0, (0.8, 1.0)],
                            [0, (1.0, 1.0)],
                            [0, (1.2, 1.0)]]})

mf = pbchf.RHF(cell, exxdiv='ewald').run()
kmf = pscf.KRHF(cell, [[0,0,0]], exxdiv='ewald').run()

def tearDownModule():
    global cell, mf, kmf
    cell.stdout.close()
    del cell, mf, kmf

class KnownValues(unittest.TestCase):
    def test_hcore(self):
        h1ref = pbchf.get_hcore(cell)
        h1 = pbchf.RHF(cell).get_hcore()
        self.assertAlmostEqual(abs(h1-h1ref).max(), 0, 9)
        self.assertAlmostEqual(lib.finger(h1), 0.14116483012673137, 9)

        cell1 = cell.copy()
        cell1.ecp = {'He': (2, ((-1, (((7.2, .3),),)),))}
        cell1.build(0, 0)
        kpt = numpy.ones(3) * .5
        h1ref = pbchf.get_hcore(cell1, kpt)
        h1 = pbchf.RHF(cell1).get_hcore(kpt=kpt)
        self.assertAlmostEqual(abs(h1-h1ref).max(), 0, 9)
        self.assertAlmostEqual(lib.finger(h1), -2.708431894877279-0.395390980665125j, 9)

        h1 = pscf.KRHF(cell1).get_hcore(kpts=[kpt])
        self.assertEqual(h1.ndim, 3)
        self.assertAlmostEqual(abs(h1[0]-h1ref).max(), 0, 9)

    def test_rhf_vcut_sph(self):
        mf = pbchf.RHF(cell, exxdiv='vcut_sph')
        e1 = mf.kernel()
        self.assertAlmostEqual(e1, -4.29190260870812, 8)
        self.assertTrue(mf.mo_coeff.dtype == numpy.double)

        mf = pscf.KRHF(cell, [[0,0,0]], exxdiv='vcut_sph')
        e0 = mf.kernel()
        self.assertTrue(numpy.allclose(e0,e1))

        numpy.random.seed(1)
        k = numpy.random.random(3)
        mf = pbchf.RHF(cell, k, exxdiv='vcut_sph')
        e1 = mf.kernel()
        self.assertAlmostEqual(e1, -4.1379172088570595, 8)
        self.assertTrue(mf.mo_coeff.dtype == numpy.complex128)

        mf = pscf.KRHF(cell, k, exxdiv='vcut_sph')
        e0 = mf.kernel()
        self.assertTrue(numpy.allclose(e0,e1))

    def test_rhf_exx_ewald(self):
        self.assertAlmostEqual(mf.e_tot, -4.3511582284698633, 8)
        self.assertTrue(mf.mo_coeff.dtype == numpy.double)
        self.assertAlmostEqual(mf.e_tot, kmf.e_tot, 8)

        # test bands
        numpy.random.seed(1)
        kpts_band = numpy.random.random((2,3))
        e1, c1 = mf.get_bands(kpts_band)
        e0, c0 = kmf.get_bands(kpts_band)
        self.assertAlmostEqual(abs(e0[0]-e1[0]).max(), 0, 7)
        self.assertAlmostEqual(abs(e0[1]-e1[1]).max(), 0, 7)
        self.assertAlmostEqual(lib.finger(e1[0]), -6.2986775452228283, 7)
        self.assertAlmostEqual(lib.finger(e1[1]), -7.6616273746782362, 7)

    def test_rhf_exx_ewald_with_kpt(self):
        numpy.random.seed(1)
        k = numpy.random.random(3)
        mf = pbchf.RHF(cell, k, exxdiv='ewald')
        e1 = mf.kernel()
        self.assertAlmostEqual(e1, -4.2048655827967139, 8)
        self.assertTrue(mf.mo_coeff.dtype == numpy.complex128)

        kmf = pscf.KRHF(cell, k, exxdiv='ewald')
        e0 = kmf.kernel()
        self.assertTrue(numpy.allclose(e0,e1))

        # test bands
        numpy.random.seed(1)
        kpt_band = numpy.random.random(3)
        e1, c1 = mf.get_bands(kpt_band)
        e0, c0 = kmf.get_bands(kpt_band)
        self.assertAlmostEqual(abs(e0-e1).max(), 0, 7)
        self.assertAlmostEqual(lib.finger(e1), -6.8312867098806249, 7)

    def test_rhf_exx_None(self):
        mf = pbchf.RHF(cell, exxdiv=None)
        e1 = mf.kernel()
        self.assertAlmostEqual(e1, -2.9325094887283196, 8)
        self.assertTrue(mf.mo_coeff.dtype == numpy.double)

        mf = pscf.KRHF(cell, [[0,0,0]], exxdiv=None)
        e0 = mf.kernel()
        self.assertTrue(numpy.allclose(e0,e1))

        numpy.random.seed(1)
        k = numpy.random.random(3)
        mf = pbchf.RHF(cell, k, exxdiv=None)
        mf.init_guess = 'hcore'
        e1 = mf.kernel()
        self.assertAlmostEqual(e1, -2.7862168430230341, 8)
        self.assertTrue(mf.mo_coeff.dtype == numpy.complex128)

        mf = pscf.KRHF(cell, k, exxdiv=None)
        mf.init_guess = 'hcore'
        e0 = mf.kernel()
        self.assertTrue(numpy.allclose(e0,e1))

    def test_init_guess_by_chkfile(self):
        numpy.random.seed(1)
        k = numpy.random.random(3)
        mf = pbchf.RHF(cell, k, exxdiv='vcut_sph')
        mf.max_cycle = 1
        mf.diis = None
        e1 = mf.kernel()
        self.assertAlmostEqual(e1, -4.132445328608581, 9)

        mf1 = pbchf.RHF(cell, exxdiv='vcut_sph')
        mf1.chkfile = mf.chkfile
        mf1.init_guess = 'chkfile'
        mf1.diis = None
        mf1.max_cycle = 1
        e1 = mf1.kernel()
        self.assertAlmostEqual(e1, -4.291854736401251, 9)
        self.assertTrue(mf1.mo_coeff.dtype == numpy.double)

    def test_uhf_exx_ewald(self):
        mf = pscf.UHF(cell, exxdiv='ewald')
        mf.init_guess = 'hcore'
        e1 = mf.kernel()
        self.assertAlmostEqual(e1, -4.3511582287379111, 8)
        self.assertTrue(mf.mo_coeff[0].dtype == numpy.double)

        kmf = pscf.KUHF(cell, [[0,0,0]], exxdiv='ewald')
        kmf.init_guess = 'hcore'
        e0 = kmf.kernel()
        self.assertTrue(numpy.allclose(e0,e1))

        # test bands
        numpy.random.seed(1)
        kpts_band = numpy.random.random((2,3))
        e1a, e1b = mf.get_bands(kpts_band)[0]
        e0a, e0b = kmf.get_bands(kpts_band)[0]
        self.assertAlmostEqual(abs(e0a[0]-e1a[0]).max(), 0, 5)
        self.assertAlmostEqual(abs(e0a[1]-e1a[1]).max(), 0, 5)
        self.assertAlmostEqual(abs(e0b[0]-e1b[0]).max(), 0, 5)
        self.assertAlmostEqual(abs(e0b[1]-e1b[1]).max(), 0, 5)
        self.assertAlmostEqual(lib.finger(e1a[0]), -6.2986775452228283, 5)
        self.assertAlmostEqual(lib.finger(e1a[1]), -7.6616273746782362, 5)

        numpy.random.seed(1)
        k = numpy.random.random(3)
        mf = pscf.UHF(cell, k, exxdiv='ewald')
        e1 = mf.kernel()
        self.assertAlmostEqual(e1, -4.2048655827967139, 8)
        self.assertTrue(mf.mo_coeff[0].dtype == numpy.complex128)

        kmf = pscf.KUHF(cell, k, exxdiv='ewald')
        e0 = kmf.kernel()
        self.assertTrue(numpy.allclose(e0,e1))

        # test bands
        numpy.random.seed(1)
        kpts_band = numpy.random.random((2,3))
        e1a, e1b = mf.get_bands(kpts_band)[0]
        e0a, e0b = kmf.get_bands(kpts_band)[0]
        self.assertAlmostEqual(abs(e0a[0]-e1a[0]).max(), 0, 5)
        self.assertAlmostEqual(abs(e0a[1]-e1a[1]).max(), 0, 5)
        self.assertAlmostEqual(abs(e0b[0]-e1b[0]).max(), 0, 5)
        self.assertAlmostEqual(abs(e0b[1]-e1b[1]).max(), 0, 5)
        self.assertAlmostEqual(lib.finger(e1a[0]), -6.8312867098806249, 5)
        self.assertAlmostEqual(lib.finger(e1a[1]), -6.1120214505413086, 5)

    def test_ghf_exx_ewald(self):
        mf = pscf.GHF(cell, exxdiv='ewald')
        mf.init_guess = 'hcore'
        e1 = mf.kernel()
        self.assertAlmostEqual(e1, -4.3511582287379111, 8)
        self.assertTrue(mf.mo_coeff.dtype == numpy.double)

        kmf = pscf.KGHF(cell, [[0,0,0]], exxdiv='ewald')
        kmf.init_guess = 'hcore'
        e0 = kmf.kernel()
        self.assertTrue(numpy.allclose(e0,e1))

#        # test bands
#        numpy.random.seed(1)
#        kpts_band = numpy.random.random((2,3))
#        e1, c1 = mf.get_bands(kpts_band)
#        e0, c0 = kmf.get_bands(kpts_band)
#        self.assertAlmostEqual(abs(e0[0]-e1[0]).max(), 0, 7)
#        self.assertAlmostEqual(abs(e0[1]-e1[1]).max(), 0, 7)
#        self.assertAlmostEqual(lib.finger(e1[0]), -6.2986775452228283, 7)
#        self.assertAlmostEqual(lib.finger(e1[1]), -7.6616273746782362, 7)

        numpy.random.seed(1)
        k = numpy.random.random(3)
        mf = pscf.GHF(cell, k, exxdiv='ewald')
        e1 = mf.kernel()
        self.assertAlmostEqual(e1, -4.2048655827967139, 8)
        self.assertTrue(mf.mo_coeff.dtype == numpy.complex128)

        kmf = pscf.KGHF(cell, k, exxdiv='ewald')
        e0 = kmf.kernel()
        self.assertTrue(numpy.allclose(e0,e1))

#        # test bands
#        numpy.random.seed(1)
#        kpts_band = numpy.random.random((2,3))
#        e1, c1 = mf.get_bands(kpts_band)
#        e0, c0 = kmf.get_bands(kpts_band)
#        self.assertAlmostEqual(abs(e0[0]-e1[0]).max(), 0, 7)
#        self.assertAlmostEqual(abs(e0[1]-e1[1]).max(), 0, 7)
#        self.assertAlmostEqual(lib.finger(e1[0]), -6.8312867098806249, 7)
#        self.assertAlmostEqual(lib.finger(e1[1]), -6.1120214505413086, 7)

#    def test_rhf_0d(self):
#        from pyscf.df import mdf_jk
#        from pyscf.scf import hf
#        L = 4
#        cell = pbcgto.Cell()
#        cell.build(unit = 'B',
#                   a = numpy.eye(3)*L*5,
#                   mesh = [21]*3,
#                   atom = '''He 2 2 2; He 2 2 3''',
#                   dimension = 0,
#                   verbose = 0,
#                   basis = { 'He': [[0, (0.8, 1.0)],
#                                    [0, (1.0, 1.0)],
#                                    [0, (1.2, 1.0)]]})
#        mol = cell.to_mol()
#        mf = mdf_jk.density_fit(hf.RHF(mol))
#        mf.with_df.mesh = [21]*3
#        mf.with_df.auxbasis = {'He':[[0, (1e6, 1)]]}
#        mf.with_df.charge_constraint = False
#        mf.with_df.metric = 'S'
#        eref = mf.kernel()
#
#        mf = pbchf.RHF(cell)
#        mf.with_df = pdf.AFTDF(cell)
#        mf.exxdiv = None
#        mf.get_hcore = lambda *args: hf.get_hcore(mol)
#        mf.energy_nuc = lambda *args: mol.energy_nuc()
#        e1 = mf.kernel()
#        self.assertAlmostEqual(e1, eref, 8)

    def test_rhf_1d(self):
        L = 4
        cell = pbcgto.Cell()
        cell.build(unit = 'B',
                   a = [[L,0,0],[0,L*5,0],[0,0,L*5]],
                   mesh = [11,20,20],
                   atom = '''He 2 0 0; He 3 0 0''',
                   dimension = 1,
                   low_dim_ft_type = 'inf_vacuum',
                   verbose = 0,
                   basis = { 'He': [[0, (0.8, 1.0)],
                                    #[0, (1.0, 1.0)],
                                    [0, (1.2, 1.0)]
                                   ]})
        mf = pbchf.RHF(cell)
        mf.with_df = pdf.AFTDF(cell)
        mf.with_df.eta = 0.3
        mf.with_df.mesh = cell.mesh
        mf.init_guess = 'hcore'
        e1 = mf.kernel()
        self.assertAlmostEqual(e1, -3.24497234871167, 5)

    def test_rhf_2d(self):
        L = 4
        cell = pbcgto.Cell()
        cell.build(unit = 'B',
                   a = [[L,0,0],[0,L,0],[0,0,L*5]],
                   mesh = [11,11,20],
                   atom = '''He 2 0 0; He 3 0 0''',
                   dimension = 2,
                   low_dim_ft_type = 'inf_vacuum',
                   verbose = 0,
                   basis = { 'He': [[0, (0.8, 1.0)],
                                    #[0, (1.0, 1.0)],
                                    [0, (1.2, 1.0)]
                                   ]})
        mf = pbchf.RHF(cell)
        mf.with_df = pdf.AFTDF(cell)
        mf.with_df.eta = 0.3
        mf.with_df.mesh = cell.mesh
        e1 = mf.kernel()
        self.assertAlmostEqual(e1, -3.2681555164454039, 5)

    def test_rhf_2d_fft(self):
        L = 4
        cell = pbcgto.Cell()
        cell.build(unit = 'B',
                   a = [[L,0,0],[0,L,0],[0,0,L*5]],
                   mesh = [11,11,20],
                   atom = '''He 2 0 0; He 3 0 0''',
                   dimension = 2,
                   verbose = 0,
                   basis = { 'He': [[0, (0.8, 1.0)],
                                    [0, (1.2, 1.0)]
                                   ]})
        mf = pbchf.RHF(cell, exxdiv='ewald')
        mf.with_df = pdf.FFTDF(cell)
        mf.with_df.mesh = cell.mesh
        e1 = mf.kernel()
        self.assertAlmostEqual(e1, -3.5797041803667593, 5)

        mf1 = pbchf.RHF(cell, exxdiv='ewald')
        mf1.with_df = pdf.FFTDF(cell)
        mf1.with_df.mesh = cell.mesh
        mf1.direct_scf = True
        e1 = mf1.kernel()
        self.assertAlmostEqual(e1, -3.5797041803667593, 5)

        mf2 = pbchf.RHF(cell, exxdiv=None)
        mf2.with_df = pdf.FFTDF(cell)
        mf2.with_df.mesh = cell.mesh
        mf2.direct_scf = True
        e2 = mf2.kernel()
        self.assertAlmostEqual(e2, -1.629571720365774, 5)

    def test_uhf_1d(self):
        L = 4
        cell = pbcgto.Cell()
        cell.build(unit = 'B',
                   a = numpy.eye(3)*4,
                   mesh = [10,20,20],
                   atom = '''He 2 0 0; He 3 0 0''',
                   dimension = 1,
                   low_dim_ft_type = 'inf_vacuum',
                   verbose = 0,
                   basis = { 'He': [[0, (0.8, 1.0)],
                                    #[0, (1.0, 1.0)],
                                    [0, (1.2, 1.0)]
                                   ]})
        mf = pscf.UHF(cell)
        mf.with_df = pdf.AFTDF(cell)
        mf.with_df.eta = 0.3
        mf.with_df.mesh = cell.mesh
        mf.init_guess = 'hcore'
        e1 = mf.kernel()
        self.assertAlmostEqual(e1, -3.24497234871167, 5)

    def test_ghf_1d(self):
        L = 4
        cell = pbcgto.Cell()
        cell.build(unit = 'B',
                   a = numpy.eye(3)*4,
                   mesh = [10,20,20],
                   atom = '''He 2 0 0; He 3 0 0''',
                   dimension = 1,
                   low_dim_ft_type = 'inf_vacuum',
                   verbose = 0,
                   basis = { 'He': [[0, (0.8, 1.0)],
                                    #[0, (1.0, 1.0)],
                                    [0, (1.2, 1.0)]
                                   ]})
        mf = pscf.GHF(cell)
        mf.with_df = pdf.AFTDF(cell)
        mf.with_df.eta = 0.3
        mf.with_df.mesh = cell.mesh
        mf.init_guess = 'hcore'
        e1 = mf.kernel()
        self.assertAlmostEqual(e1, -3.24497234871167, 5)

    def test_get_veff(self):
        mf = pscf.RHF(cell)
        numpy.random.seed(1)
        nao = cell.nao_nr()
        dm = numpy.random.random((nao,nao)) + numpy.random.random((nao,nao))*1j
        dm = dm + dm.conj().T
        v11 = mf.get_veff(cell, dm, kpt=cell.get_abs_kpts([.25,.25,.25]))
        v12 = mf.get_veff(cell, dm, kpts_band=cell.get_abs_kpts([.25,.25,.25]))
        v13 = mf.get_veff(cell, dm, kpt=cell.get_abs_kpts([-1./3,1./3,.25]),
                          kpts_band=cell.get_abs_kpts([.25,.25,.25]))
        v14 = mf.get_veff(cell, dm, kpt=cell.get_abs_kpts([-1./3,1./3,.25]),
                          kpts_band=cell.make_kpts([2,1,1]))
        self.assertTrue(v11.dtype == numpy.complex128)
        self.assertTrue(v12.dtype == numpy.complex128)

        mf = pscf.UHF(cell)
        v21 = mf.get_veff(cell, dm, kpt=cell.get_abs_kpts([.25,.25,.25]))
        dm = [dm*.5,dm*.5]
        v22 = mf.get_veff(cell, dm, kpts_band=cell.get_abs_kpts([.25,.25,.25]))
        v23 = mf.get_veff(cell, dm, kpt=cell.get_abs_kpts([-1./3,1./3,.25]),
                          kpts_band=cell.get_abs_kpts([.25,.25,.25]))
        v24 = mf.get_veff(cell, dm, kpt=cell.get_abs_kpts([-1./3,1./3,.25]),
                          kpts_band=cell.make_kpts([2,1,1]))
        self.assertAlmostEqual(abs(v11-v21).max(), 0, 9)
        self.assertAlmostEqual(abs(v12-v22).max(), 0, 9)
        self.assertAlmostEqual(abs(v13-v23).max(), 0, 9)
        self.assertAlmostEqual(abs(v14-v24).max(), 0, 9)
        self.assertAlmostEqual(lib.finger(v11), -0.30110964334164825+0.81409418199767414j, 9)
        self.assertAlmostEqual(lib.finger(v12), -2.1601376488983997-9.4070613374115908j, 9)

    def test_init(self):
        from pyscf.pbc import dft
        cell_u = cell.copy()
        cell_u.spin = 2
        self.assertTrue(isinstance(pscf.RKS  (cell  ), dft.rks.RKS    ))
        self.assertTrue(isinstance(pscf.RKS  (cell_u), dft.roks.ROKS  ))
        self.assertTrue(isinstance(pscf.UKS  (cell  ), dft.uks.UKS    ))
        self.assertTrue(isinstance(pscf.ROKS (cell  ), dft.roks.ROKS  ))
        self.assertTrue(isinstance(pscf.KS   (cell  ), dft.rks.RKS    ))
        self.assertTrue(isinstance(pscf.KS   (cell_u), dft.uks.UKS    ))
        self.assertTrue(isinstance(pscf.KRKS (cell  ), dft.krks.KRKS  ))
        self.assertTrue(isinstance(pscf.KRKS (cell_u), dft.krks.KRKS  ))
        self.assertTrue(isinstance(pscf.KUKS (cell  ), dft.kuks.KUKS  ))
        self.assertTrue(isinstance(pscf.KROKS(cell  ), dft.kroks.KROKS))
        self.assertTrue(isinstance(pscf.KKS  (cell  ), dft.krks.KRKS  ))
        self.assertTrue(isinstance(pscf.KKS  (cell_u), dft.kuks.KUKS  ))

        self.assertTrue(isinstance(pscf.RHF  (cell  ), pscf.hf.RHF     ))
        self.assertTrue(isinstance(pscf.RHF  (cell_u), pscf.rohf.ROHF  ))
        self.assertTrue(isinstance(pscf.KRHF (cell  ), pscf.khf.KRHF   ))
        self.assertTrue(isinstance(pscf.KRHF (cell_u), pscf.khf.KRHF   ))
        self.assertTrue(isinstance(pscf.UHF  (cell  ), pscf.uhf.UHF    ))
        self.assertTrue(isinstance(pscf.KUHF (cell_u), pscf.kuhf.KUHF  ))
        self.assertTrue(isinstance(pscf.GHF  (cell  ), pscf.ghf.GHF    ))
        self.assertTrue(isinstance(pscf.KGHF (cell_u), pscf.kghf.KGHF  ))
        self.assertTrue(isinstance(pscf.ROHF (cell  ), pscf.rohf.ROHF  ))
        self.assertTrue(isinstance(pscf.ROHF (cell_u), pscf.rohf.ROHF  ))
        self.assertTrue(isinstance(pscf.KROHF(cell  ), pscf.krohf.KROHF))
        self.assertTrue(isinstance(pscf.KROHF(cell_u), pscf.krohf.KROHF))
        self.assertTrue(isinstance(pscf.HF   (cell  ), pscf.hf.RHF     ))
        self.assertTrue(isinstance(pscf.HF   (cell_u), pscf.uhf.UHF    ))
        self.assertTrue(isinstance(pscf.KHF  (cell  ), pscf.khf.KRHF   ))
        self.assertTrue(isinstance(pscf.KHF  (cell_u), pscf.kuhf.KUHF  ))

    def test_dipole_moment(self):
        dip = mf.dip_moment()
        self.assertAlmostEqual(lib.finger(dip), 0.03847620192010277, 8)

        # For test cover only. Results for low-dimesion system are not
        # implemented.
        with lib.temporary_env(cell, dimension=1):
            kdm = kmf.get_init_guess(key='minao')
            dip = kmf.dip_moment(cell, kdm)
        #self.assertAlmostEqual(lib.finger(dip), 0, 9)

    def test_makov_payne_correction(self):
        de = pbchf.makov_payne_correction(mf)
        self.assertAlmostEqual(de[0], -0.1490687416177664, 7)
        self.assertAlmostEqual(de[0], de[1], 7)
        self.assertAlmostEqual(de[0], de[2], 7)

    def test_init_guess_by_1e(self):
        dm = mf.get_init_guess(key='1e')
        self.assertAlmostEqual(lib.finger(dm), 0.025922864381755062, 9)

        dm = kmf.get_init_guess(key='1e')
        self.assertEqual(dm.ndim, 3)
        self.assertAlmostEqual(lib.finger(dm), 0.025922864381755062, 9)

    def test_init_guess_by_atom(self):
        with lib.temporary_env(cell, dimension=1):
            dm = mf.get_init_guess(key='minao')
            kdm = kmf.get_init_guess(key='minao')

        self.assertAlmostEqual(lib.finger(dm), -1.714952331211208, 8)

        self.assertEqual(kdm.ndim, 3)
        self.assertAlmostEqual(lib.finger(dm), -1.714952331211208, 8)

    def test_jk(self):
        nao = cell.nao
        numpy.random.seed(2)
        dm = numpy.random.random((2,nao,nao)) + .5j*numpy.random.random((2,nao,nao))
        dm = dm + dm.conj().transpose(0,2,1)
        ref = pbchf.get_jk(mf, cell, dm)
        vj, vk = mf.get_jk_incore(cell, dm)
        self.assertAlmostEqual(abs(vj - ref[0]).max(), 0, 9)
        self.assertAlmostEqual(abs(vk - ref[1]).max(), 0, 9)


if __name__ == '__main__':
    print("Full Tests for pbc.scf.hf")
    unittest.main()
