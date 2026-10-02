# Model Card for EnDe-Spherical-Sliced

## Model Details

### Model Description

- **Developed by:** Evgeniia Tokarchuk, Hua Chang Bakker, Vlad Niculae
- **Funded by:** Dutch Research Council (NWO), grant VI.Veni.212.228
- **Shared by:** Evgeniia Tokarchuk
- **Model type:** Discrete-output Transformer machine translation model with spherical embeddings
- **Language(s) (NLP):** English (`en`), German (`de`)
- **License:** MIT

This model translates English news text into German. Its tied decoder input/output embeddings are 128-dimensional manifold parameters constrained to the unit hypersphere and trained with Riemannian Adam. It adds the paper's sliced dispersion regularizer, using one randomly sampled great circle per translation minibatch with no additional vocabulary subsampling.

### Model Sources

- **Dispersion library:** https://github.com/ltl-uva/ledoh-torch
- **Training framework:** https://github.com/afeena/cdgm_textgen 
- **Paper:** [Keep your distance: learning dispersed embeddings on $\mathbb{S}_m$](https://arxiv.org/abs/2502.08231)

## Uses

### Direct Use

English-to-German translation in the news domain. The model is intended for research on spherical optimization, representation dispersion, and machine translation.

### Bias, Risks, and Limitations

The training data consists of news and may reproduce biases or errors present in that data. Quality may degrade outside the news domain, on dialectal or specialized text, or for language varieties underrepresented in the corpus. Automatic scores do not establish factuality, fairness, or suitability for high-stakes use; translations should be reviewed by a qualified speaker when errors could cause harm.

## How to Get Started with the Model

The checkpoint uses the `fairseq` preprocessing and SentencePiece setup described in the paper. After preparing compatible binarized data, run:

2. Download library with `git`

    ```bash
    git clone https://github.com/afeena/cdgm_textgen
    cd cdgm_textgen
    ```

3. Translate test data with CLI:
    ```bash
    #run decoding with decode.py from cdgm_textgen lib
    python decode.py </path/to/data> --beam 1 --task translation --source-lang en --target-lang de --print-step  --decoding-measure cosine  --path checkpoint_best.pt --input </path/to/testfile> > hyp.txt

    #extract and postprocess hypothesis
    cat hyp.txt | grep -P "^H" | sort -V | cut -f3- | ~/develop/sentencepiece/build/src/spm_decode --model=mbart.cc25.v2/sentence.bpe.model  | sed 's/▁//g' | sacremoses -l de detokenize | sacremoses -l de detruecase > hyp_eval_ready.txt

    #score hypothesis
    cat hyp_eval_ready.txt | sacrebleu <data/en-de/newstest2016.en>
    comet-score -s  <data/en-de/newstest2016.en> -t hyp_eval_ready.txt -r  <data/en-de/newstest2016.de> --only_system 

    
    ```

## Training Details

### Training Data

[WMT 2019 English-German News Translation Task](https://www.statmt.org/wmt19/translation-task.html), containing approximately 9.1 million training sentence pairs including back-translated data.

#### Preprocessing 

**Install libraries**: [mosesdecoder](https://github.com/moses-smt/mosesdecoder), [sentencepiece](https://github.com/google/sentencepiece)

**Download additional files**

1. Pre-trained UnigramLM model from [mBART 25](https://huggingface.co/docs/transformers/main/model_doc/mbart) 

    [sentencepiece.bpe.model](https://huggingface.co/facebook/mbart-large-cc25/blob/e84f32f3b320dcc2ee3d0c0d257c978137d10c25/sentencepiece.bpe.model) 

```bash
#!/bin/bash


SPM_ENCODE=sentencepiece/build/src/spm_encode
SRC="en"
TRG="de"


for prefix in train newsdev2016 newstest2016
 do
   cat $prefix.$SRC | \
   sacremoses -l $TRG normalize | \
   sacremoses -l $TRG tokenize > data-pp/$prefix.tok.$TRG

   cat $prefix.$TRG | \
   sacremoses -l $TRG normalize | \
   sacremoses -l $TRG tokenize > data-pp/$prefix.tok.$TRG

 done

for l in $SRC $TRG
  do
    cat data-pp/train.tok.$l | sacremoses train-truecase -m data-pp/truecase-model.$l
  done

for prefix in train newstest2016 newstest2017
  do
    cat data-pp/$prefix.tok.$SRC | sacremoses truecase -m data-pp/truecase-model.$SRC > data-pp/$prefix.tok.tc.$SRC
    cat data-pp/$prefix.tok.$TRG | sacremoses truecase -m data-pp/truecase-model.$TGT > data-pp/$prefix.tok.tc.$TRG
  done

for prefix in train newsdev2016 newstest2016
  do
    SPM_ENCODE --model=sentencepiece.bpe.model --output_format=piece < data-pp/$prefix.tok.tc.$SRC > data-pp/$prefix.tok.tc.spm.$SRC
    SPM_ENCODE --model=sentencepiece.bpe.model --output_format=piece < data-pp/$prefix.tok.tc.$TGT > data-pp/$prefix.tok.tc.spm.$TGT
  done
```

### Training Procedure

The system uses a standard Transformer objective with cross-entropy loss and label smoothing of 0.1. Decoder embeddings are tied between the input and output layers. All non-manifold parameters use Adam with learning rate $5\times10^{-4}$; the spherical embedding parameter uses Riemannian Adam with learning rate $5\times10^{-3}$. Training uses a 10,000-step warmup, an effective batch size of approximately 65,500 tokens, and 50,000 updates. Subword tokenization uses the SentencePiece model from mBART-25.

#### Training config

```yaml
# @package _group_
task:
  _name: translation
  data: en-de/fairseq-bin/
  source_lang: en
  target_lang: de
  eval_bleu: true
  eval_bleu_detok: moses
  eval_bleu_remove_bpe: sentencepiece
  eval_bleu_print_samples: false
  eval_bleu_args: '{"beam":1}'
criterion:
  _name: ls_cross_entropy_sliced_disp
  label_smoothing: 0.1
  dispersed_weight: 10.0
  report_dispersed: false
model:
  _name: riemannian_emb_transformer
  decoder:
    learned_pos: true
    embed_dim: 128
    output_dim: 128
  encoder:
    learned_pos: true
  dropout: 0.1
  share_decoder_input_output_embed: true
  manifold: "sphere"
optimizer:
  _name: composite
  groups:
    default:
      optimizer:
        _name: adam
        lr: [0.0005]
        adam_betas: (0.9,0.98)
      lr_scheduler:
        _name: inverse_sqrt
        warmup_updates: 10000
        warmup_init_lr: 1e-07
    riemannian:
      optimizer:
        lr: [0.005]
        _name: riemannian_adam
      lr_scheduler:
        _name: inverse_sqrt
        warmup_updates: 10000
        warmup_init_lr: 1e-07
lr_scheduler: pass_through
dataset:
  max_tokens: 8000
  validate_interval_updates: 1000
optimization:
  lr: [0.0005]
  stop_min_lr: 1e-09
  update_freq: [8]
  max_update: 100000
checkpoint:
  no_epoch_checkpoints: true
  best_checkpoint_metric: bleu
  maximize_best_checkpoint_metric: true
common:
  wandb_project: ende-spm
  log_format: simple
  log_interval: 100
```

## Evaluation

The model was evaluated on WMT `newstest2016` with beam size 5. BLEU uses SacreBLEU 2.3.1 with signature `nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.3.1`; COMET uses `unbabel-comet` 2.2.25 with `Unbabel/wmt22-comet-da`. Dispersion is reported as minimum geodesic distance ($d_{min}$) and spherical variance (`svar`).

| BLEU | COMET | $d_{min}$ | svar |
|---:|---:|---:|---:|
| 33.5* | 0.821* | 0.222 | 0.999 |

`*` indicates a statistically significant difference ($p<0.05$) from the Euclidean baseline in Table 3.

## Citation

```bibtex
@article{tokarchuk2025keep,
  title   = {Keep your distance: learning dispersed embeddings on $\mathbb{S}_m$},
  author  = {Tokarchuk, Evgeniia and Bakker, Hua Chang and Niculae, Vlad},
  journal = {Transactions on Machine Learning Research},
  year    = {2025},
  url     = {https://openreview.net/forum?id=5JIQE6HcTd}
}
```
