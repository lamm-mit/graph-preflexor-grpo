#!/bin/bash
#SBATCH --job-name=exp2_redo
#SBATCH --account=bfir-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=/projects/bfir/ssourav/logs/exp2_redo_%j.out
#SBATCH --error=/projects/bfir/ssourav/logs/exp2_redo_%j.err

echo "=========================================="
echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $SLURMD_NODENAME"
echo "Start     : $(date)"
echo "=========================================="

mkdir -p /projects/bfir/ssourav/logs
nvidia-smi

export HF_HOME=/work/nvme/bfir/ssourav/cache/huggingface
export SENTENCE_TRANSFORMERS_HOME=/work/nvme/bfir/ssourav/cache/sentence_transformers

cd /projects/bfir/ssourav
python circuit_exp2_redo.py

echo "Finished  : $(date)"
echo "=========================================="
