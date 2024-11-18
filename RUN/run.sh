#!/usr/bin/env bash
#SBATCH --account=akara
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=8000
#SBATCH --error=%J.err
#SBATCH --output=%J.out
#SBATCH --job-name=NN

module load vasp/vasp-5.4.4-oneapi-2023.1.0-WITH-PATCHES
mpirun vasp_std > job_output

##
