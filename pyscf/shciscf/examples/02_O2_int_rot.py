#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
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
# Author: James Smith <james.smith9113@gmail.com>
#
# All diatomic bond lengths taken from:
# http://cccbdb.nist.gov/diatomicexpbondx.asp
"""
Comparing determining the effect of acitve-active orbital rotations. All output
is deleted after the run to keep the directory neat. Comment out the cleanup
section to view output.
"""
from pyscf import gto, scf, mcscf, dmrgscf
from pyscf.shciscf import shci

#
# Mean Field
#
mol = gto.M(verbose=4, atom="O 0 0 0; O 0 0 1.208", basis="ccpvdz")
mf = scf.RHF(mol).run()


#
# Multireference WF
#
ncas = 12  # Number of orbitals included in the active space.
nelecas = 12  # Number of electrons included in the active space.
nfrozen = 2  # Number of orbitals that won't be rotated/optimized.

# Calculate energy of the molecules with frozen core.
mc = shci.SHCISCF(mf, ncas, nelecas)
mc.frozen = nfrozen  # Freezes the innermost 2 orbitals.
mc.fcisolver.sweep_iter = [0]
mc.fcisolver.sweep_epsilon = [1e-3]  # Loose variational tolerances.
mc.fcisolver.nPTiter = 0
e_noaa = mc.mc1step()[0]

# Calculate energy of the molecule with frozen core and active-active rotations
mc = shci.SHCISCF(mf, ncas, nelecas)
mc.frozen = nfrozen  # Freezes the innermost 2 orbitals.
mc.internal_rotation = True  # Do active-active orbital rotations.
mc.fcisolver.sweep_iter = [0]
mc.fcisolver.sweep_epsilon = [1e-3]
mc.fcisolver.nPTiter = 0
mc.max_cycle_macro = 20
e_aa = mc.mc1step()[0]

# Comparison Calculations
del_aa = e_aa - e_noaa

print("\n\nEnergies for O2 give in E_h.")
print("=====================================")
print("SHCI w/o Act.-Act. Rotations: %6.12f" % e_noaa)
print("SHCI w/ Act.-Act. Rotations: %6.12f" % e_aa)
print("Change w/ Act.-Act. Rotations: %6.12f" % del_aa)

# File cleanup. Comment out to help debugging.
mc.fcisolver.cleanup_dice_files()
