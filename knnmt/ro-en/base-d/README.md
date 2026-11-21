# Model Card for RoEn-Base-D

## Model Details

- **Developed by:** Evgeniia Tokarchuk, Sergey Troshin, and Vlad Niculae
- **Model type:** Dispersion-fine-tuned Transformer for neural machine translation
- **Languages:** Romanian (`ro`) to English (`en`)
- **Paper:** [Angular Dispersion Accelerates k-Nearest Neighbors Machine Translation](https://aclanthology.org/2025.findings-emnlp.759/)
- **Implementation:** [cdgm_textgen](https://github.com/ltl-uva/cdgm_textgen)

RoEn-Base-D fine-tunes a Romanian–English Transformer with a sliced angular dispersion regularizer on its 128-dimensional decoder representations. Use the checkpoint for standalone translation or add k-NN interpolation at inference time by building a datastore from the WMT16 training split.

## Uses and Limitations

The checkpoint is intended for research on Romanian-to-English news translation and approximate nearest-neighbor retrieval. The published results cover WMT16 news and may vary with domain, preprocessing, and retrieval settings.

## Standalone Decoding

Run these Bash examples from the repository root. Set the paths to your local files and place the released checkpoint at `$model_dir/checkpoint_best.pt`. Use the matching Fairseq dictionaries and SentencePiece model. The input file contains one preprocessed Romanian sentence per line.

```bash
repo_dir="$HOME/develop/cdgm_textgen"
model_dir=ro_en_transformer_dispersion_ft
data_dir=data/ro-en/fairseq-bin-spm-disjoint
input_file=data/ro-en/newstest2016.tok.tc.spm.ro
checkpoint="$model_dir/checkpoint_best.pt"

python3 "$repo_dir/decode.py" "$data_dir" \
  --path "$checkpoint" --input "$input_file" \
  --task translation --source-lang ro --target-lang en \
  --beam 5 --decoding-measure cosine
```

## Training Details

The [paper (§§4.2–4.3 and Appendix B)](https://aclanthology.org/2025.findings-emnlp.759.pdf) describes fine-tuning on the WMT16 Romanian–English news corpus of about 612,000 training pairs with SentencePiece preprocessing. The fine-tuning objective combines label-smoothed cross-entropy (smoothing 0.1) and a sliced dispersion term with weight 1.0. The trainable layers are concentrated in the last decoder block and output projection.

The archived checkpoint's configuration enables validation BLEU with beam size 1 and selects the checkpoint with the highest BLEU. It specifies Adam, 500 warm-up steps, a learning rate of `5e-4`, and a limit of 4,000 updates. The paper reports `5e-5` and 5,000 updates for Ro–En.

## Decoding with k-NN

### Build the Datastore and Index

Build the datastore and FAISS index locally from the checkpoint and the matching binarized `train` split. The index builder requires CUDA-enabled PyTorch and GPU FAISS. The archived training split yielded 21,507,934 target rows. Set the allocation capacity to at least the row count of your preprocessed split. The feature memmap alone uses about 11 GB at the capacity shown below.

The example builds a 128-dimensional L2 IVFPQ index with 2,048 IVF centroids and 64-byte PQ codes. FAISS trains on up to one million sampled vectors. The extractor records the number of rows written, which the builder uses to exclude unused capacity.

```bash
out="$model_dir/datastore-ro-en"
index_path="$out/roen.faiss_l2_IVFPQindex_d128_nc2048"
capacity=21507934

(
  set -e
  mkdir "$out"
  python3 "$repo_dir/extract_key_values_knnmt.py" "$data_dir" \
    --path "$checkpoint" --task translation --source-lang ro --target-lang en \
    --valid-subset train --results-path "$out" --ntotal "$capacity" \
    --batch-size 8 --distributed-world-size 1

  ntotal=$(cat "$out/train.ntotal.txt")
  if [[ ! $ntotal =~ ^[1-9][0-9]*$ ]] || (( ntotal > capacity )); then
    printf 'Invalid extracted row count: %s\n' "$ntotal" >&2
    exit 1
  fi

  python3 "$repo_dir/build_datastore.py" \
    --keys_path "$out/train.features.npy" --values_path "$out/train.targets.npy" \
    --dstore_size "$ntotal" --faiss_index "$index_path" \
    --coarse_idx l2 --dim 128 --ncentroids 2048 --code_size 64 --pca 0

  ln -s train.targets.npy "${index_path}_vals.npy"
)
```

Use a fresh output directory for each build. The decoder reads the index and its `_vals.npy` sidecar together.

### Decode with the Rebuilt Index

The saved Ro–En launcher uses eight neighbors, temperature 100, interpolation weight 0.3, and eight IVF probes. The paper also reports a 32-probe setting.

```bash
python3 "$repo_dir/decode.py" "$data_dir" \
  --path "$checkpoint" --input "$input_file" \
  --task translation_knn --source-lang ro --target-lang en \
  --beam 5 --decoding-measure cosine \
  --knn-index-file "$index_path" --knn-metric l2 --knn-nprobe 8 \
  --knn-k 8 --knn-temperature 100.0 --knn-lambda 0.3
```

## Evaluation

The following values are from the standalone Ro–En `base-D` row in [Table 1](https://aclanthology.org/2025.findings-emnlp.759.pdf#page=6). The paper evaluates on `newstest2016` using SacreBLEU and `Unbabel/wmt22-comet-da`. Throughput is the median over three runs.

| Test set | BLEU | COMET | tok/s |
|---|---:|---:|---:|
| newstest2016 | 31.7 | 78.96 | 79 |

## Citation

```bibtex
@inproceedings{tokarchuk-etal-2025-angular,
  title = {Angular Dispersion Accelerates $k$-Nearest Neighbors Machine Translation},
  author = {Tokarchuk, Evgeniia and Troshin, Sergey and Niculae, Vlad},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2025},
  year = {2025},
  pages = {14120--14132},
  doi = {10.18653/v1/2025.findings-emnlp.759},
  url = {https://aclanthology.org/2025.findings-emnlp.759/}
}
```
