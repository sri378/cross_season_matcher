#!/usr/bin/env bash
set -e

python src/run_matching.py \
  --pairs_csv data/pairs.csv \
  --out_dir output/reproduced \
  --top_k 1200 \
  --max_side 512 \
  --ransac_thresh 5.0
