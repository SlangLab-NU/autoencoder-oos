#!/bin/bash
#SBATCH --job-name=clinc150_0.10
#SBATCH --partition=a100.40gb
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --gres=gpu:ampere:1

# Load the Miniconda environment
source /nlu/projects/tianyi_zhang/miniconda3/bin/activate /nlu/projects/tianyi_zhang/miniconda3/envs/ood

# Run the Python script
python /nlu/projects/tianyi_zhang/clinc150-ood/improved_ce_fine_clinc150_010.py



