from pyscf import gto, scf, mcscf ,dmrgscf
from pyscf.dmrgscf import DMRGCI
#from pyscf.geomopt.berny_solver import optimize
from pyscf.dmrgscf import settings
from pyscf.geomopt.geometric_solver import optimize
#settings.BLOCKSCRATCHDIR = '/home/u3/weiting07/scratch'
dmrgscf.settings.MPIPREFIX = 'mpirun -n 1' # Because it is on PBS

mol = gto.Mole()
mol.atom = '''
  N      0.0 0.0 0.0 
  N      0.0 0.0 1.2 
'''
mol.basis = 'ccpvdz'  
mol.charge = 0
mol.spin = 0 ## or triplet state=2
mol.symmetry = False # in paper no symmetry is used
mol.build()

#
# geometry optimization for CASSCF
#
mf = scf.RHF(mol)
mol.build(verbose=7, output = '4Pas_dmrg.out')
mc = mcscf.CASSCF(mf, 4, 4)
mc.fcisolver = dmrgscf.DMRGCI(mol,maxM=100, tol=1e-8)
conv_params = {
    'convergence_energy': 1e-4,  # Eh
    'convergence_grms': 3e-3,    # Eh/Bohr
    'convergence_gmax': 4.5e-3,  # Eh/Bohr
    'convergence_drms': 1.2e-2,  # Angstrom
    'convergence_dmax': 1.8e-2,  # Angstrom
}
mc.sorting_mo_energy = True
#mc.fix_spin_()

#
# Tune DMRG parameters.  It's not necessary in most scenario.
#
#mc.fcisolver.outputlevel = 3
#mc.fcisolver.scheduleSweeps = [0, 4, 8, 12, 16, 20, 24, 28, 30, 34]
#mc.fcisolver.scheduleMaxMs  = [200, 400, 800, 1200, 2000, 4000, 3000, 2000, 1000, 500]
#mc.fcisolver.scheduleTols   = [0.0001, 0.0001, 0.0001, 0.0001, 1e-5, 1e-6, 1e-7, 1e-7, 1e-7, 1e-7 ]
#mc.fcisolver.scheduleNoises = [0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0, 0.0, 0.0, 0.0]
#mc.fcisolver.twodot_to_onedot = 38
#mc.fcisolver.maxIter = 50


# method 1
mol_eq = optimize(mc, **conv_params)
print(mol_eq.atom_coords())

