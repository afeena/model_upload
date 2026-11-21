#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  printf 'Usage: bash %s CDGM_TEXTGEN_DIR MODEL_DIR MEDICAL_DATA_DIR [l2|ip]\n' "$0" >&2
  exit 2
fi

repo_dir=$1
model_dir=$2
data_dir=$3
out="$model_dir/datastore-medical"
capacity=5700000
dim=1024
ncentroids=2048
coarse_index=${4:-ip}
if [[ "$coarse_index" != l2 && "$coarse_index" != ip ]]; then
  printf 'Expected l2 or ip for the coarse quantizer.\n' >&2
  exit 2
fi
index_path="$out/medical.deen.faiss_${coarse_index}_IVFPQindex_d${dim}_nc${ncentroids}"

if [[ ! -f "$model_dir/checkpoint_best.pt" || ! -d "$data_dir" ]]; then
  printf 'Expected checkpoint_best.pt and a binarized medical dataset directory.\n' >&2
  exit 2
fi

mkdir "$out"
python3 "$repo_dir/extract_key_values_knnmt.py" "$data_dir" \
  --path "$model_dir/checkpoint_best.pt" \
  --task translation --source-lang de --target-lang en \
  --valid-subset train \
  --results-path "$out" \
  --ntotal "$capacity" \
  --batch-size 8 \
  --distributed-world-size 1

ntotal=$(cat "$out/train.ntotal.txt")
if [[ ! $ntotal =~ ^[1-9][0-9]*$ ]] || (( ntotal > capacity )); then
  printf 'Invalid extracted row count: %s\n' "$ntotal" >&2
  exit 1
fi

python3 "$repo_dir/build_datastore.py" \
  --keys_path "$out/train.features.npy" \
  --values_path "$out/train.targets.npy" \
  --dstore_size "$ntotal" \
  --faiss_index "$index_path" \
  --coarse_idx "$coarse_index" \
  --dim "$dim" \
  --ncentroids "$ncentroids" \
  --code_size 64 \
  --pca 0

ln -s train.targets.npy "${index_path}_vals.npy"
