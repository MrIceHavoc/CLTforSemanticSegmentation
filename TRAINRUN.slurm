#!/bin/bash
#SBATCH -J cv_train # job name
#SBATCH -o cv_train.out # standard output and error log
#SBATCH -p titanxp # queue name or partiton name 2080tti*
#SBATCH -t 24:00:00 # Run time (hh:mm:ss)
#SBATCH  --gres=gpu:2 # Number of GPUs
#SBATCH  --nodes=1
#SBATCH  --ntasks=1
#SBATCH  --tasks-per-node=1
#SBATCH  --cpus-per-task=2

# echo node name (ex: n1~n35)
srun -l /bin/hostname
# echo current directory
srun -l /bin/pwd
# echo current time
srun -l /bin/date

# Remove all Linux Modules (recommended)
module purge

# For debug purpose
pip freeze

# Check GPUs
echo $CUDA_VISIBLE_DEVICES

sh mmselfsup/tools/dist_train.sh mmselfsup/configs/selfsup/mocov3/mocov3_vit-small-p16_32xb128-fp16-coslr-300e_in1k-224.py 2

#python -u mmselfsup/tools/train.py mmselfsup/configs/selfsup/mocov3/mocov3_vit-small-p16_32xb128-fp16-coslr-300e_in1k-224.py \
#        --work-dir=./work_dir \
#        --gpus=2 \
