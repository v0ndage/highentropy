#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --error=%J.err
#SBATCH --output=%J.out
#SBATCH --job-name=schnetpk
#SBATCH --gres=gpu:2

name=$1
db=$2

module load cuda
srun python3 train.py $name $db
