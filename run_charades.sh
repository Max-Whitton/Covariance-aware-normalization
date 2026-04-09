#!/bin/bash -l
#$ -P cs585
#$ -N charades_ego
#$ -j y
#$ -o /projectnb/cs585/students/sanjiv/Covariance-aware-normalization/logs/charades_ego_$JOB_ID.out
#$ -m ae
#$ -l h_rt=24:00:00
#$ -pe omp 8
#$ -l gpus=2
#$ -l gpu_type=L40S

export OMP_NUM_THREADS=$NSLOTS
export NCCL_P2P_DISABLE=1

mkdir -p /projectnb/cs585/students/sanjiv/Covariance-aware-normalization/logs
mkdir -p /projectnb/cs585/students/sanjiv/Covariance-aware-normalization/results

cd /projectnb/cs585/students/sanjiv/Covariance-aware-normalization

module load miniconda
module load academic-ml/spring-2026
conda activate spring-2026-pyt

python -m torch.distributed.launch \
  --nproc_per_node=2 \
  --master_port=8081 \
  run/train_charades.py --config configs/ft/charades.json
