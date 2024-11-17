#!/usr/bin/env python3
import os, sys
import numpy as np
from ase.db import connect
from ase.io import read, write

DB = sys.argv[1]

db = connect(DB)
images = [db.get_atoms(id=i+1) for i in range(len(db))]
assert len(images)>0, 'no images found'
print(len(images), 'images')

magmoms = {'H': 1.0, 'C': 1.0, 'O': 1.0, 'Ag': 0.4, 'Au': 0.4, 'Cu': 0.5, 'Ni': 2.0, 'Pd': 0.6, 'Pt': 0.6}

INCAR = """SYSTEM = AgAuCuNiPdPt
ISTART = 0
ICHARG = 2
NWRITE = 1
LCHARG = .FALSE.
LWAVE  = .FALSE.
LREAL = Auto

NCORE= 4
PREC = Accurate
ENCUT = 500
ISMEAR = 0

ISPIN = 2
MAGMOM = ZZZ

IBRION = -1
NSW = 0
NELM = 100
EDIFF = 1E-6
SIGMA = 0.01

ISYM = 0
ALGO = Normal

METAGGA = R2SCAN
LUSE_VDW = .TRUE.
BPARAM = 11.95 #recommendation
CPARAM = 0.0093
LASPH = .TRUE.

LMAXTAU = 6
LDIPOL = .TRUE.
DIPOL = 0.5 0.5 0.5
IDIPOL = 4
"""

kpoints = """K-Points
0
Gamma
1 1 1
0 0 0
"""

for i in range(len(db)):
	
	cluster = images[i]
	stoi = cluster.get_chemical_symbols()
	z, c = np.unique(stoi, return_counts=True)

	line = [str(c[i]) + '*' + str(magmoms[z[i]]) for i in range(len(z))]
	magmom_line = "MAGMOM = " + ' '.join(line)

	NEW = INCAR.replace('ZZZ', ' '.join(line))

	newpath = 'DFTs/'+str(i+1)+'/'
	if not os.path.exists(newpath): os.makedirs(newpath)

	with open(os.path.join(newpath, 'INCAR'), 'w') as incar:
		incar.write(NEW)
	
	with open(os.path.join(newpath, 'KPOINTS'), 'w') as incar:
		incar.write(kpoints)

	index = np.sort(np.unique(stoi, return_index=True)[1])
	elements = [stoi[i] for i in index]

	with open(os.path.join(newpath, 'POTCAR'), 'w') as potcar:
		for e in elements:
			with open('POTS/'+e+'.POTCAR', 'r') as f:
				potcar.write(f.read())
	write(os.path.join(newpath, 'POSCAR'), cluster)

print('>DB2DFT DONE')
