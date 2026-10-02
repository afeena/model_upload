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

## Uses

### Direct Use

German-to-English translation for the four evaluated domains. The system is primarily intended for research into domain adaptation, approximate nearest-neighbor retrieval, and angular dispersion of model representations.

### Bias, Risks, and Limitations

Performance is domain- and datastore-dependent. The system may reproduce errors and biases in the WMT and domain corpora, and results may not generalize to other domains, language varieties, hardware, FAISS settings, or datastore sizes. k-NN configurations require substantial extra storage and retrieval infrastructure. Automatic BLEU and COMET scores do not establish factuality, fairness, or suitability for high-stakes use.

## How to Get Started with the Model

Load the checkpoint with [fairseq](https://github.com/facebookresearch/fairseq) and use standard German-to-English generation. The checkpoint expects the preprocessing and vocabulary of the pretrained WMT19 German-English model described in the paper.

## Training Details

The starting point is the pretrained German-to-English Transformer of Ng et al. (2019). Dispersion fine-tuning uses the WMT19 German-English news corpus (34 million filtered sentence pairs), cross-entropy with label smoothing 0.1, and a sliced dispersion regularizer with weight $\gamma=1$ and one great circle per batch. Only the last decoder block's two feed-forward layers, final layer normalization, and output projection are trainable. Fine-tuning runs for 5,000 steps with Adam, learning rate $7\times10^{-5}$, 500 warmup steps, and an effective batch size of approximately 65,500 tokens. The model has 356M parameters, of which 51M are trained; the reported run took approximately 20 hours on one NVIDIA GeForce GTX TITAN X.

## Evaluation

Table 2 of the paper reports case-sensitive SacreBLEU 2.3.1 (`nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.3.1`), COMET with `Unbabel/wmt22-comet-da`, and median decoding throughput in tokens per second. Throughput was measured on an NVIDIA GeForce GTX TITAN X over ten random 200-sentence subsets sampled with replacement.

- **Number of IVFPQ probes:** —

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

