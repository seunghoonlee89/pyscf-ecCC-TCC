#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Block (DMRG) program has two branches.  The OpenMP/MPI hybrid implementation
Block-1.5 (stackblock) code is more efficient than the old pure MPI
implementation Block-1.1 in both the computing time and memory footprint.
This example shows how to input new defined keywords for stackblock program.

Block-1.5 (stackblock) defines two new keywords memory and num_thrds.  The
rest keywords are all compatible to the old Block program.
'''

from pyscf import lib
from pyscf import gto
from pyscf import scf
from pyscf import mcscf
from pyscf import dmrgscf

import os
from pyscf.dmrgscf import settings
settings.MPIPREFIX = 'mpirun -n 10'

b = 1.2
mol = gto.M(
    verbose = 4,
    atom = 'O       -0.26677564    -0.27872083     0.00000000;\
          H       -0.26677564     0.82127917     0.00000000;\
          H       -0.26677564    -0.64538753     1.03708994',
    basis = 'ccpvtz',
    symmetry = True,
)
mf = scf.RHF(mol)
mf.kernel()

#
# Pass stackblock keywords memory and num_thrds to fcisolver attributes
#
mc = dmrgscf.DMRGSCF(mf, 58, 10, maxM=5)
mc.fcisolver.memory = 4  # in GB
mc.fcisolver.num_thrds = 8
emc = mc.kernel()[0]
print(emc)

#mc = dmrgscf.DMRGSCF(mf, 8, 8)
#mc.state_average_([0.5, 0.5])
#mc.fcisolver.memory = 4  # in GB
#mc.fcisolver.num_thrds = 8
#mc.kernel()
#print(mc.e_tot)


