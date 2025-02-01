#!/usr/bin/env bash
#SBATCH --account=akara
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=8000
#SBATCH --error=%J.err
#SBATCH --output=%J.out
#SBATCH --job-name=rerun

# Checks a set of outcars for completion
# ignores those without outcars,
# fixes Zbrent / frozen error, and re-submits.

ITER=$1
START=$2
STOP=$3

cd DFT$ITER/
mkdir SAVE$ITER

for ((i=$START; i<=$STOP; i++));
do
	# enter only if dir exists
	if [[ -d "$i" ]]; then

		cd $i/

		# check job
		if [[ -f "OUTCAR" ]]; then

			# at least one iteration (otherwise presumed stuck)
			if grep -q "LOOP" "OUTCAR"; then

				# check completion
				if grep -q "required" "OUTCAR"; then
					echo "$i done"
					cd ..
					continue
				fi

				# check error
				if grep -q "ZBRENT" "OUTCAR"; then
					echo "$i error"
					#improve resolution
					sed -i 's/^EDIFF *=.*/EDIFF = 2E-4/' "INCAR"
				fi

				# update job iff contcar exists (as it should here)
				if [ -e "CONTCAR" ]; then
					mv "POSCAR" "POS1"
					mv "OUTCAR" "OUT1"
					mv "CONTCAR" "POSCAR"
					sed -i 's/^ISTART *=.*/ISTART = 1/' "INCAR"
					sed -i 's/^ICHARG *=.*/ICHARG = 0/' "INCAR"
				fi
			else
				# find and cancel jobs that are presumed stuck
				for file in *.err; do
					if [[ -f "$file" ]]; then
						jobID=$(basename "$file" .err)
						scancel "$jobID"
					fi
				done
				echo "$i canceled"
				rm OUTCAR
			fi

			# re-run job
			echo "running $i"
			#sbatch run.sh
			sbatch --partition=preemptable --qos=preemptable run.sh
			sleep 2

		else
			echo "$i no OUTCAR"
		fi

		cd ..
	fi
done
cd ../

###
