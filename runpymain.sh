#!/bin/bash

#SBATCH --partition=main       # Partition (job queue)

#SBATCH --requeue                 # Return job to the queue if preempted

#SBATCH --nodes=1                 # Number of nodes you require

#SBATCH --ntasks=1                # Total # of tasks across all nodes

#SBATCH --cpus-per-task=1         # Cores per task (>1 if multithread tasks)

#SBATCH --mem=64000                # Real memory (RAM) required (MB)

#SBATCH --time=3-00:00:00           # Total run time limit (D-HH:MM:SS)

#SBATCH --output=slurm/out/%j.out  # STDOUT output file

#SBATCH --error=slurm/err/%j.err   # STDERR output file (optional)

echo ${day} $SLURM_JOBID "node_list" $SLURM_NODELIST $@  "\n" >> jobs.txt
module purge
eval "$(conda shell.bash hook)"
conda activate math
srun python3 $@
conda deactivate