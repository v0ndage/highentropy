#!/usr/bin/env python3
# This script generates input files for VASP calculations from an ASE database.
# It processes each structure in the database, applies constraints, sets up magnetic moments,
# and prepares all necessary input files for VASP.

import os
import sys
import numpy as np
from ase.db import connect
from ase.io import read, write
from ase.constraints import FixAtoms

# Get the database filename from the command-line arguments.
DB = sys.argv[1]

# Connect to the ASE database specified by the user.
db = connect(DB)

# Retrieve all atomic structures (images) from the database.
images = [db.get_atoms(id=i+1) for i in range(len(db))]

# Ensure that the database is not empty.
assert len(images) > 0, 'no images found'

# Print the number of images found in the database.
print(len(images), 'images')

# Define initial magnetic moments for different elements.
# These values are used to set up the initial spin configuration in VASP calculations.
magmoms = {
    'H': 1.0,  # Hydrogen
    'C': 1.0,  # Carbon
    'O': 1.0,  # Oxygen
    'Ag': 0.4, # Silver
    'Au': 0.4, # Gold
    'Cu': 0.5, # Copper
    'Ni': 2.0, # Nickel
    'Pd': 0.6, # Palladium
    'Pt': 0.6  # Platinum
}

# Template for the INCAR file.
# 'ZZZ' is a placeholder for the MAGMOM line, which will be replaced later.
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

IBRION = 3
NSW = 100
NELM = 100
EDIFF = 1E-4
EDIFFG = -1E-3
SIGMA = 0.01

ISYM = 0
ALGO = Normal

METAGGA = R2SCAN
LASPH = .TRUE.
LMAXTAU = 6
LDIPOL = .TRUE.
DIPOL = 0.5 0.5 0.5
IDIPOL = 4
"""

# Template for the KPOINTS file.
# Specifies a Gamma-point calculation with a 1x1x1 k-point grid.
kpoints = """K-Points
0
Gamma
1 1 1
0 0 0
"""

# Template for the job submission script (e.g., for SLURM).
# Adjust the directives and module loading as per your computing environment.
runscript = """#!/usr/bin/env bash
#SBATCH --account=akara
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=8000
#SBATCH --error=%J.err
#SBATCH --output=%J.out
#SBATCH --job-name=NN

# Load the VASP module (adjust as necessary for your environment).
module load vasp/vasp-5.4.4-oneapi-2023.1.0-WITH-PATCHES

# Run VASP and redirect the output to 'job_output'.
mpirun vasp_std > job_output
"""

# Loop over each image (atomic structure) in the database.
for i in range(len(db)):
    # Get the current atomic structure from the list.
    cluster = images[i]

    # Apply a constraint to fix the positions of the first 55 atoms.
    # This is useful when you want to keep certain atoms stationary during relaxation.
    cluster.set_constraint([FixAtoms(np.arange(55))])

    # Get the list of chemical symbols (elements) present in the cluster.
    stoi = cluster.get_chemical_symbols()

    # Find unique elements and their counts in the cluster.
    # 'z' contains the unique elements.
    # 'c' contains the counts of each element.
    z, c = np.unique(stoi, return_counts=True)

    # Construct the MAGMOM line for the INCAR file.
    # Format: 'number_of_atoms*magnetic_moment', e.g., '2*1.0'.
    line = [str(c[j]) + '*' + str(magmoms[z[j]]) for j in range(len(z))]

    # Create the complete MAGMOM line.
    magmom_line = "MAGMOM = " + ' '.join(line)

    # Replace the 'ZZZ' placeholder in the INCAR template with the actual MAGMOM line.
    NEW = INCAR.replace('ZZZ', ' '.join(line))

    # Define the directory path for the current calculation.
    # Each calculation is stored in 'DFTs/<index>/'.
    newpath = 'DFTs/' + str(i+1) + '/'

    # Create the directory if it doesn't already exist.
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    # Write the INCAR file to the calculation directory.
    with open(os.path.join(newpath, 'INCAR'), 'w') as f:
        f.write(NEW)

    # Write the KPOINTS file to the calculation directory.
    with open(os.path.join(newpath, 'KPOINTS'), 'w') as f:
        f.write(kpoints)

    # Write the job submission script to the calculation directory.
    with open(os.path.join(newpath, 'run.sh'), 'w') as f:
        f.write(runscript)

    # Find the indices of unique elements in the order they appear in 'stoi'.
    index = np.sort(np.unique(stoi, return_index=True)[1])

    # Get the list of unique elements in the order they appear in the cluster.
    elements = [stoi[j] for j in index]

    # Write the POTCAR file by concatenating the POTCAR files for each element.
    # The POTCAR files are expected to be located in the 'POTS/' directory.
    with open(os.path.join(newpath, 'POTCAR'), 'w') as potcar:
        for e in elements:
            # Open the POTCAR file for the element and append its content.
            with open('POTS/' + e + '.POTCAR', 'r') as f:
                potcar.write(f.read())

    # Write the POSCAR file (contains atomic positions and lattice vectors).
    # Uses ASE's write function to format the data appropriately.
    write(os.path.join(newpath, 'POSCAR'), cluster)

# Print a message indicating that the script has completed successfully.
print('>DB2DFT DONE')
