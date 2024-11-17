#!/bin/bash
#SBATCH --account=akara
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --error=%J.err
#SBATCH --output=%J.out
#SBATCH --job-name=clean

C=$1
START=$2
STOP=$3

cd DFTs/
mkdir SAVE$C

for ((i=$START; i<=$STOP; i++));
do
	cp $i/OUTCAR SAVE$C/$i.OUTCAR
	#cp $i/DOSCAR SAVE$C/$i.DOSCAR
	rm -r $i
done
cd ../

rm *err
rm *out
rm waitlist
rm outfiles/*
exit 1


