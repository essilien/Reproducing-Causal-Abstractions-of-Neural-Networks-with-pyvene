# Running this on Kaggle

Recommended scale (see chat for rationale): 40K/3K/3K examples, BERT 3
epochs, interchange search with M=150 examples over 6 key nodes. This
fits comfortably in one Kaggle GPU session and is enough to reproduce the
paper's core contrasts (BERT >> BiLSTM accuracy; BERT shows real
compositional structure at a handful of nodes, BiLSTM doesn't).

## 1. Upload the code as a Kaggle Dataset

- Kaggle -> "Create" -> "New Dataset" -> upload `mqnli_causal_abstraction.zip`
  (or drag the extracted `proj/` folder). Name it e.g. `mqnli-causal-abstraction`.
- This only needs to be done once; the dataset shows up under
  `/kaggle/input/mqnli-causal-abstraction/` in any notebook you attach it to.

## 2. New Notebook

- "Create" -> "New Notebook".
- Right sidebar -> **Settings**:
  - Accelerator: **GPU T4 x2** (or P100) -- BERT fine-tuning needs a GPU.
  - **Internet: ON** -- required to download `bert-base-uncased` from
    HuggingFace and to `pip install` pyvene/networkx.
- Sidebar -> "+ Add Input" -> attach the `mqnli-causal-abstraction` dataset
  you just uploaded.

## 3. Notebook cells

**Cell 1 -- setup**
```python
!pip install -q pyvene networkx
!cp -r /kaggle/input/mqnli-causal-abstraction/proj /kaggle/working/proj
%cd /kaggle/working/proj
```

**Cell 2 -- sanity checks (run these first, ~1 minute total)**
```python
!python3 tests/tests_smoke_causal_model.py
!python3 tests/tests_smoke_pyvene_lstm.py
!python3 tests/tests_smoke_pyvene_bert.py
```

**Cell 3 -- generate the scaled-up dataset (~a few minutes, CPU only)**
```python
!python3 data/generate_dataset.py \
    --n_train 40000 --n_dev 3000 --n_test 3000 \
    --out_dir datasets/scaled
```

**Cell 4 -- train BiLSTM baseline (~10-20 min on GPU)**
```python
!python3 train.py --data_dir datasets/scaled --out_dir checkpoints/bilstm_scaled \
    --epochs 20 --batch_size 128 --emb_dim 256 --h_dim 256 --n_layer 3
```

**Cell 5 -- fine-tune BERT (~1-3 hours on a T4, depending on augmentation)**
```python
!python3 train_bert.py --data_dir datasets/scaled --out_dir checkpoints/bert_scaled \
    --epochs 3 --batch_size 32 --lr 3e-5
# add --use_augmentation if dev_acc plateaus noticeably below what you expect
```
This prints `dev_acc` after every epoch and checkpoints after every epoch
(`model_epoch0.pt`, `model_epoch1.pt`, ...) specifically so a Kaggle
session timeout doesn't lose all progress -- if it gets cut off, restart
the notebook, re-run cells 1-3, then point at the last saved
`model_epochN.pt` instead of re-running cell 5 from scratch.

**Cell 6 -- interchange-intervention analysis (start small, then scale)**
```python
# quick check, ~a few minutes: does the pipeline find *any* structure?
!python3 analysis/interchange_experiments.py \
    --checkpoint_dir checkpoints/bert_scaled --model_type bert \
    --data_dir datasets/scaled --split test --n_examples 30 \
    --nodes NPObj NObj --layers 2 5 8 \
    --out results_bert_quick.json

# the real run: M=150, 6 key nodes, all layers -- budget 1-2+ hours
!python3 analysis/interchange_experiments.py \
    --checkpoint_dir checkpoints/bert_scaled --model_type bert \
    --data_dir datasets/scaled --split test --n_examples 150 \
    --nodes NObj AdjObj NPObj VP NegP QObj \
    --out results_bert.json

# same, for the BiLSTM baseline -- expect much smaller clique sizes
!python3 analysis/interchange_experiments.py \
    --checkpoint_dir checkpoints/bilstm_scaled --model_type bilstm \
    --data_dir datasets/scaled --split test --n_examples 150 \
    --nodes NObj AdjObj NPObj VP NegP QObj \
    --out results_bilstm.json
```

**Cell 7 -- probing baseline (for the probe-vs-intervention contrast, Fig. 4)**
```python
!python3 analysis/probing.py --checkpoint_dir checkpoints/bilstm_scaled \
    --data_dir datasets/scaled --node NPObj --direction fw --layer 2 \
    --side hypothesis --pos 0 --n_examples 300
```
(The probing script currently only wires up the BiLSTM path -- extending
`get_hidden_states` in `analysis/probing.py` to BERT is a small addition,
same pattern as `bert_encode_batch` in `interchange_experiments.py`, if
you want the BERT-side probing comparison too.)

## 4. Getting results out

Everything written under `/kaggle/working/` (checkpoints, `results_*.json`)
appears in the notebook's "Output" tab and can be downloaded, or saved
directly as a new Kaggle Dataset via "Save Version" -> "Save & Run All"
so later notebooks can attach it as input instead of retraining.

## Time/quota budget

Kaggle gives ~30 GPU-hours/week (subject to change) and caps a single
session at several hours before auto-restarting. At this scale:
- dataset generation: a few minutes (CPU)
- BiLSTM training: ~15-20 minutes
- BERT training: the long pole, ~1-3 hours depending on whether you turn
  on `--use_augmentation`
- interchange analysis: the other long pole -- it's O(M^2) per candidate
  location; M=150 x 6 nodes x ~12 BERT layers x ~2 candidate word
  positions/node is still a lot of forward passes. If a session times out
  midway, rerun with `--nodes` restricted to whichever ones haven't
  finished yet (each node's result is independent).

If even M=150 proves too slow in practice, drop to M=80-100 and/or search
only every other BERT layer (`--layers 0 2 4 6 8 10`) -- clique-size
*trends* across nodes/layers are the thing you're checking for, not exact
match to the paper's numbers.
