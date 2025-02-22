#!/usr/bin/env python3
import os, sys
import numpy as np
from ase.db import connect
from ase.io import read, write
from ase.constraints import FixAtoms, FixBondLengths

# This is a starting point for relaxations
# Afterwards constraints should be tightened

C = sys.argv[1]
DB = sys.argv[2]

db = connect(DB)
images = [db.get_atoms(id=i+1) for i in range(len(db))]
assert len(images)>0, 'no images found'
print(len(images), 'images')

magmoms = {'H': 1.0, 'C': 1.0, 'O': 1.0, 'Ag': 0.4, 'Au': 0.4, 'Cu': 0.5, 'Ni': 2.0, 'Pd': 0.6, 'Pt': 0.6}

INCAR = """SYSTEM = AgAuCuNiPdPt
ISTART = 0
ICHARG = 2
NWRITE = 1
LCHARG = TRUE
LWAVE  = TRUE
LREAL = .FALSE.

ENCUT = 300

NCORE = 4
PREC = Normal
ISMEAR = 1

ISPIN = 2
MAGMOM = ZZZ

IBRION = 2
NELM = 60
NSW = 200
EDIFF = 1E-3
EDIFFG = 0.01
SIGMA = 0.1

ALGO = ov

#LORBIT = 11
#LMAXTAU = 6
#LDIPOL = .TRUE.
#DIPOL = 0.5 0.5 0.5
#IDIPOL = 4

METAGGA = R2SCAN
LUSE_VDW = .TRUE.
LASPH = .TRUE.
BPARAM = 11.95

"""

kpoints = """K-Points
0
Gamma
1 1 1
0 0 0
"""

runscript = """#!/usr/bin/env bash
#SBATCH --account=akara
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=8
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=8000
#SBATCH --exclude=ec[1-50]
#SBATCH --error=%J.err
#SBATCH --output=%J.out
#SBATCH --job-name=NN

ml vasp
#ml vasp/vasp-6.4.2-oneapi-2023.1.0
#module load vasp/vasp-5.4.4-oneapi-2023.1.0-WITH-PATCHES
srun vasp_gam > job_output

##
"""

#fixes n nearest neighbor atoms or atoms >= n agnstroms
def fixnn(image, n, dist=False):
		
	test = image.copy()
	d = [d[:55] for d in test.get_all_distances()[55:]]
	dmean = np.mean(d, axis=0)
	
	if dist: index = np.where([d>=n for d in dmean])[0]
	else: index = np.argsort(dmean)[::-1][:-n]
	
	test.set_constraint(FixAtoms(index))
	print(len(index), 'indices fixed')
	
	return test

for i in range(len(db)):
        
	test = images[i]
	test.set_constraint(None)
	#test = fixnn(test, 5.0, dist=True)
	stoi = test.get_chemical_symbols()
	z, c = np.unique(stoi, return_counts=True)

	line = [str(c[i]) + '*' + str(magmoms[z[i]]) for i in range(len(z))]
	magmom_line = "MAGMOM = " + ' '.join(line)

	NEW = INCAR.replace('ZZZ', ' '.join(line))

	newpath = 'DFT'+str(C)+'/'+str(i)+'/'
	if not os.path.exists(newpath): os.makedirs(newpath)

	with open(os.path.join(newpath, 'INCAR'), 'w') as f:
		f.write(NEW)
	
	with open(os.path.join(newpath, 'KPOINTS'), 'w') as f:
		f.write(kpoints)

	with open(os.path.join(newpath, 'run.sh'), 'w') as f:
		f.write(runscript)

	index = np.sort(np.unique(stoi, return_index=True)[1])
	elements = [stoi[i] for i in index]

	with open(os.path.join(newpath, 'POTCAR'), 'w') as potcar:
		for e in elements:
			with open('POTS/'+e+'.POTCAR', 'r') as f:
				potcar.write(f.read())

	with open(os.path.join(newpath, 'POSCAR'), 'w', newline='\n') as f:
			write(f, test, format="vasp")
			
print('>DB2DFT DONE')

###
