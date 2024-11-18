#!/bin/bash
#SBATCH --account=akara
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --error=%J.err
#SBATCH --output=%J.out
#SBATCH --job-name=DFT

START=$1
STOP=$2

cd DFTs/

for ((i=$START; i<=$STOP; i++));
do
    #make and enter n DFT dirs
    if cd "$i"; then
	#cp -r ../../run.sh .
    	sed -i "/--job-name/,//s/NN/DFT-$i/" run.sh
    	#sbatch run.sh
    	sbatch --partition=preemptable --qos=preemptable run.sh
    	sleep 2
    	cd -
    fi
done
cd ../

###
