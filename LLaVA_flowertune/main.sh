#!/bin/bash


#SBATCH --job-name=temp # job name
#SBATCH --output=job_submissions_out/train%j%x.out # output log file
#SBATCH --error=job_submissions_out/train%j%x.err  # error file
#SBATCH --time=00:10:00   # 12 hour of wall time
#SBATCH --nodes=1        # 1 GPU node
#SBATCH --partition=gpu  # gpu partition
#SBATCH --ntasks=2      # 1 CPU core to drive GPU
#SBATCH --gres=gpu:1     # Request 1 GPU


echo "This is federated training"
# Add lines here to run your GPU-based computations.
cd /scratch/nshaheen/project/may_federated/LLaVA_flowertune/
python main.py
