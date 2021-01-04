#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
A simple example to run FCI
'''

from pyscf import gto, scf, fci

mol = gto.Mole()
mol.build(
    atom = 'O       -0.26677564    -0.27872083     0.00000000;\
          H       -0.26677564     0.82127917     0.00000000;\
          H       -0.26677564    -0.64538753     1.03708994',
    basis = 'ccpvtz',
    symmetry = True,
)

myhf = scf.RHF(mol)
myhf.kernel()

#
# Function fci.FCI creates an FCI solver based on the given orbitals and the
# num. electrons and spin of the given mol object
#
cisolver = fci.FCI(mol, myhf.mo_coeff)
print('E(FCI) = %.12f' % cisolver.kernel()[0])

