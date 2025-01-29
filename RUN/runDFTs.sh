#!/bin/bash
#SBATCH --account=akara
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --error=%J.err
#SBATCH --output=%J.out
#SBATCH --job-name=DFT

# Runs set of jobs iff no outcar present

ITER=$1
START=$2
STOP=$3

cd DFT$1/

for ((i=$START; i<=$STOP; i++));
do
	# enter only if dir exist
	if [[ -d "$i" ]]; then
		cd $i/
		
		if [ ! -f "OUTCAR" ]; then
			
			# run dfts 
			sed -i "/--job-name/,//s/NN/DFT-$i/" run.sh
			#sbatch run.sh
			sbatch --partition=preemptable --qos=preemptable run.sh
			sleep 2
		fi

		cd ../
	fi
done
cd ../

###
