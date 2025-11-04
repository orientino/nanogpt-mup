#!/bin/bash -l
#SBATCH -J shake
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=chenxiang.zhang@uni.lu
#SBATCH --account=p200535
#SBATCH --qos=default
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=0-08:00:00
#SBATCH --output=slurm-%x-%j.out

echo -e "--------------------------------"
echo -e "Start:\t $(date)"
echo -e "JobID:\t ${SLURM_JOBID}"
echo -e "Node:\t ${SLURM_NODELIST}"
echo -e "--------------------------------\n"

micromamba activate mup

# mup_examples/mutransfer_lr_shakespeare_char/mup/run.sh
# mup_examples/mutransfer_lr_shakespeare_char/sp/run.sh
python _merge.py --init sp

