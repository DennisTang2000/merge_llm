#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH --mem=750G
#SBATCH --job-name="bash"
#SBATCH -p compsci-gpu

source ../../MergeLM/unc2/bin/activate
python3 merge_test.py
        