#!/bin/bash -l
#$ -P cs585
#$ -N setup_env
#$ -j y
#$ -o /projectnb/cs585/students/sanjiv/Covariance-aware-normalization/logs/setup_env_$JOB_ID.out
#$ -m ae
#$ -l h_rt=1:00:00

mkdir -p /projectnb/cs585/students/sanjiv/Covariance-aware-normalization/logs

module load miniconda
module load academic-ml/spring-2026
conda activate spring-2026-pyt

# Install dependencies (skips already-installed packages)
conda install -y -c conda-forge -c pytorch \
  transformers \
  timm \
  pandas \
  numpy \
  opencv \
  pillow \
  av \
  decord \
  ffmpeg-python \
  sacred \
  scipy \
  scikit-learn \
  tqdm \
  einops \
  tensorboardx \
  humanize \
  psutil \
  dominate \
  ipdb

echo "Environment setup complete."
