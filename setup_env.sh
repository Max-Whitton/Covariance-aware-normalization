#!/bin/bash -l

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
