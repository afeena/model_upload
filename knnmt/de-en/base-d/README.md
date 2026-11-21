# Model Card for DeEn-Base-D

## Model Details

### Model Description

- **Developed by:** Evgeniia Tokarchuk, Sergey Troshin, Vlad Niculae
- **Shared by:** Evgeniia Tokarchuk
- **Model type:** Dispersion-fine-tuned neural machine translation checkpoint
- **Language(s) (NLP):** German (`de`), English (`en`)
- **License:** See the licenses of the pretrained checkpoint, datasets, and linked implementations

This model fine-tunes the German-to-English baseline so that decoder context representations are more angularly dispersed while their norms remain intact. It does not require a datastore for ordinary NMT decoding. The system is evaluated for domain adaptation in the Medical, Law, IT, and Koran domains.

### Model Sources

- **Paper:** [Angular Dispersion Accelerates $k$-Nearest Neighbors Machine Translation](https://aclanthology.org/2025.findings-emnlp.759/)
- **Base checkpoint:** [Facebook FAIR WMT19 German-English model](https://github.com/facebookresearch/fairseq/blob/main/examples/wmt19/README.md)
- **kNN-MT implementation:** https://github.com/urvashik/knnmt
- **Dispersion library:** https://github.com/ltl-uva/ledoh-torch
- **Repository:** - https://github.com/ltl-uva/cdgm_textgen

## Uses

### Direct Use

German-to-English translation for the four evaluated domains. The system is primarily intended for research into domain adaptation, approximate nearest-neighbor retrieval, and angular dispersion of model representations.

### Bias, Risks, and Limitations

Performance is domain- and datastore-dependent. The system may reproduce errors and biases in the WMT and domain corpora, and results may not generalize to other domains, language varieties, hardware, FAISS settings, or datastore sizes. k-NN configurations require substantial extra storage and retrieval infrastructure. Automatic BLEU and COMET scores do not establish factuality, fairness, or suitability for high-stakes use.

## How to Get Started with the Model

Run the examples from the repository root. Set the paths below to your local files. Place the released checkpoint at `$model_dir/checkpoint_best.pt`. Use the checkpoint's Fairseq dictionaries and apply its BPE preprocessing to the German input.

```bash
repo_dir="$HOME/develop/cdgm_textgen"
model_dir=de_en_transformer_dispersion_ft
dict_dir=wmt19.en-de.joined-dict.ensemble/data/data-bin/wmt19ende
medical_data_dir=data/multi-domain/medical/fairseq-bin
finetune_data_dir=wmt19-deen/data/hf/fairseq-bin
input_file=wmt19.en-de.joined-dict.ensemble/data/multi_domain/medical/test.bpe.de
checkpoint="$model_dir/checkpoint_best.pt"

python3 "$repo_dir/decode.py" "$dict_dir" \
  --path "$checkpoint" --input "$input_file" \
  --task translation --source-lang de --target-lang en \
  --beam 5 --decoding-measure cosine
```

## Training Details

The starting point is the pretrained German-to-English Transformer of Ng et al. (2019). Dispersion fine-tuning uses the WMT19 German-English news corpus (34 million filtered sentence pairs), cross-entropy with label smoothing 0.1, and a sliced dispersion regularizer with weight $\gamma=1$ and one great circle per batch. Only the last decoder block's two feed-forward layers, final layer normalization, and output projection are trainable. Fine-tuning runs for 5,000 steps with Adam, learning rate $7\times10^{-5}$, 500 warmup steps, and an effective batch size of approximately 65,500 tokens. The model has 356M parameters, of which 51M are trained; the reported run took approximately 20 hours on one NVIDIA GeForce GTX TITAN X.

We fine-tune on the [WMT19 German–English training set](https://huggingface.co/datasets/wmt/wmt19), which the paper describes as 34 million filtered news sentence pairs. The domain-specific corpora are used for datastore construction and evaluation.

The YAML below contains the fine-tuning settings. It disables BLEU computation during training. The paper selects checkpoints by validation BLEU, so reproducing that selection requires scoring the candidate checkpoints separately or enabling validation BLEU with the appropriate preprocessing.

```yaml
# @package _group_
task:
  _name: translation
  data: wmt19-deen/data/hf/fairseq-bin
  source_lang: de
  target_lang: en
  eval_bleu: false
criterion:
  _name: ls_ce_output_dispersion
  dispersed_weight: 1.0
  dispersion_type: sliced
  label_smoothing: 0.1
model:
  _name: transformer_dispersion_ft
  train_layers: ["decoder.layers.5.fc1", "decoder.layers.5.fc2","decoder.layers.5.final_layer_norm","decoder.project_out_dim","decoder.output_projection"]
  features_to_disperse: decoder_out
  dispersion_dim: 1024
  decoder:
    embed_dim: 1024
    output_dim: 1024
    input_dim: 1024
    attention_heads: 16
    learned_pos: false
    ffn_embed_dim: 4096
    normalize_before: false
  encoder:
    embed_dim: 1024
    attention_heads: 16
    learned_pos: false
    ffn_embed_dim: 8192
    normalize_before: false
  dropout: 0.2
  attention_dropout: 0.1
  share_decoder_input_output_embed: false
  share_all_embeddings: false
optimizer:
  _name: adam
  adam_betas: (0.9,0.98)
lr_scheduler:
  _name: inverse_sqrt
  warmup_updates: 500
  warmup_init_lr: 1e-7
dataset:
  max_tokens: 4096
  validate_interval_updates: 1000
optimization:
  lr: [0.00007]
  update_freq: [16]
  max_update: 5000
  stop_min_lr: 1e-09
checkpoint:
  finetune_from_model: wmt19-deen/checkpoint_last.pt
  no_epoch_checkpoints: false
  save_interval_updates: 500
common:
  wandb_project: knn-disp-emb
  fp16: false
  fp16_init_scale: 128
  log_format: simple
  log_interval: 200
  ```

## Domain adaptation with k-NN

### Building k-NN datastore

For the out-of-domain datastores, we apply BPE and sentence-length filtering to the multi-domain German–English corpus (Aharoni and Goldberg, 2020), following the [kNN-MT preprocessing recipe](https://github.com/urvashik/knnmt/blob/master/examples/translation/prepare-domadapt.sh). Binarize the Medical training split with the checkpoint's vocabulary.
```
@inproceedings{aharoni2020unsupervised,
  title={Unsupervised domain clusters in pretrained language models},
  author={Aharoni, Roee and Goldberg, Yoav},
  booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
  year={2020},
  url={https://arxiv.org/abs/2004.02105},
  publisher = "Association for Computational Linguistics"
}
@inproceedings{khandelwal2021nearest,
  title={Nearest Neighbor Machine Translation},
  author={Khandelwal, Urvashi and Fan, Angela and Jurafsky, Dan and Zettlemoyer, Luke and Lewis, Mike},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2021}
}
```

The release includes the checkpoint. Build the datastore and FAISS index locally before running k-NN decoding. With CUDA-enabled PyTorch and GPU FAISS installed, run the wrapper using the paths defined above. It builds a 1,024-dimensional FAISS IVFPQ index with 2,048 IVF centroids and 64-byte PQ codes. The Medical evaluation recipe uses an `IndexFlatL2` coarse quantizer and squared Euclidean retrieval. See [§4.4](https://aclanthology.org/2025.findings-emnlp.759.pdf#page=6).

```bash
coarse_index=l2
bash knnmt/build_medical_datastore.sh "$repo_dir" "$model_dir" "$medical_data_dir" "$coarse_index"
```

### Decoding with the datastore

The wrapper creates the values sidecar beside the index. Decode with four neighbors, temperature 10, interpolation weight 0.8, and 32 probes, as in the saved Medical L2 evaluation run. See [Appendix B, Table 6](https://aclanthology.org/2025.findings-emnlp.759.pdf#page=13).

```bash
index_path="$model_dir/datastore-medical/medical.deen.faiss_${coarse_index}_IVFPQindex_d1024_nc2048"

python3 "$repo_dir/decode.py" "$finetune_data_dir" \
  --path "$checkpoint" --input "$input_file" \
  --task translation_knn --source-lang de --target-lang en \
  --beam 5 --decoding-measure cosine \
  --knn-index-file "$index_path" --knn-metric l2 --knn-nprobe 32 \
  --knn-k 4 --knn-temperature 10.0 --knn-lambda 0.8
```

To run the Medical IP variant, set `coarse_index=ip` before building and use `--knn-k 8` when decoding. The wrapper requires a fresh `datastore-medical` output directory, so use a separate model directory containing the same checkpoint if you want to keep both builds.

## Evaluation

Table 2 of the paper reports case-sensitive SacreBLEU 2.3.1 (`nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.3.1`), COMET with `Unbabel/wmt22-comet-da`, and median decoding throughput in tokens per second. Throughput was measured on an NVIDIA GeForce GTX TITAN X over ten random 200-sentence subsets sampled with replacement.

The results are from the De–En `base-D` row in [Table 2](https://aclanthology.org/2025.findings-emnlp.759.pdf#page=8) and use standalone decoding. The paper reports SacreBLEU and COMET (`Unbabel/wmt22-comet-da`) on the four domain test sets. Throughput is the median across ten 200-sentence samples drawn with replacement, measured on an NVIDIA GeForce GTX TITAN X. See [§5.3](https://aclanthology.org/2025.findings-emnlp.759.pdf#page=7) and [Appendix B](https://aclanthology.org/2025.findings-emnlp.759.pdf#page=13).

| Domain | BLEU | COMET | tok/s |
|---|---:|---:|---:|
| Medical | 40.5 | 83.30 | 86 |
| Law | 46.0 | 85.37 | 92 |
| IT | 38.3 | 82.45 | 82 |
| Koran | 17.1 | 72.58 | 76 |

These values reproduce the base-d row of Table 2. The Table 2 datastore sizes are Medical 5.7M, Law 18.4M, IT 3.1M, and Koran 0.5M keys.

## Citation

```bibtex
@inproceedings{tokarchuk-etal-2025-angular,
  title = "Angular Dispersion Accelerates $k$-Nearest Neighbors Machine Translation",
  author = "Tokarchuk, Evgeniia and Troshin, Sergey and Niculae, Vlad",
  booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2025",
  month = nov,
  year = "2025",
  address = "Suzhou, China",
  publisher = "Association for Computational Linguistics",
  url = "https://aclanthology.org/2025.findings-emnlp.759/",
  doi = "10.18653/v1/2025.findings-emnlp.759",
  pages = "14120--14132"
}
```

