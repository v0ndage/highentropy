#!/bin/bash
#SBATCH --account=akara
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --error=%J.err
#SBATCH --output=%J.out
#SBATCH --job-name=clean

# Saves only converged outcars
# and removes their directory.

C=$1
START=$2
STOP=$3

cd DFT$C/
mkdir SAVE$C

for ((i=$START; i<=$STOP; i++));
do
	if grep -q "required" "$i/OUTCAR"; then
		cp $i/OUTCAR SAVE$C/$i.OUTCAR
		rm -r $i
	fi
done
cd ../

###
