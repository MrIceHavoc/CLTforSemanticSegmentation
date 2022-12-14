#!/bin/bash
#SBATCH -J cv_baseline # job name
#SBATCH -o cv_baseline_ann.out # standard output and error log
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

#sh mmselfsup/tools/dist_train.sh mmselfsup/configs/benchmarks/mmsegmentation/voc12aug/fcn_r50-d8_512x512_20k_voc12aug.py 2
#python mmselfsup/tools/model_converters/extract_backbone_weights.py mmselfsup/configs/benchmarks/mmsegmentation/voc12aug/ann_r50-d8_512x512_40k_voc12aug_20200613_231314-b5dac322.pth mmselfsup/configs/benchmarks/mmsegmentation/voc12aug/ann_r50-d8_backbone.pth

bash mmselfsup/tools/benchmarks/mmsegmentation/mim_dist_train.sh mmselfsup/configs/benchmarks/mmsegmentation/voc12aug/ann_r50-d8_512x512_40k_voc12aug.py mmselfsup/configs/benchmarks/mmsegmentation/voc12aug/ann_r50-d8_backbone.pth 2

