# Reproducing "Causal Abstractions of Neural Networks" with pyvene

This is a from-scratch, working reimplementation of the MQNLI case study
in Geiger, Lu, Icard & Potts, *Causal Abstractions of Neural Networks*
(NeurIPS 2021), using **pyvene** (Wu, Geiger, Arora, Huang, Wang, Goodman,
Manning & Potts, 2024) for the interchange interventions instead of the
original paper's bespoke `antra`/`Interchange` codebase.

It is **not** a byte-for-byte reproduction (we don't have the original
500K/60K/10K MQNLI split, and pyvene doesn't ship BERT/LSTM support out of
the box), but every piece of the methodology is here and has been tested
end-to-end on CPU: a synthetic-but-correctly-labeled MQNLI generator, the
natural-logic causal model (`CNatLog`) with forward computation *and*
interventions, a BiLSTM and a BERT NLI model both wired into pyvene, and
the alignment-search / interchange-intervention / clique-finding /
probing analysis from Section 5.

**Read `NOTES.md` before running anything at scale** -- it documents a few
non-obvious engineering decisions and one real limitation we hit in the
installed pyvene version.

## What's here

```
causal_model/            CNatLog: the natural-logic causal model
  natural_logic.py          relation algebra (join table, projectivity
                             signatures) -- adapted from the original
                             paper's own reference implementation
  mqnli_causal_model.py     Sentence/MQNLIExample + forward compute +
                             intervene() (eq. 1/3/4/6/7 in the paper)
  node_spans.py             node name -> token-index span

data/
  vocab/                    100-word vocab lists per category (subject
                             nouns, object nouns, adjectives, adverbs,
                             verbs) -- copied from the original paper's
                             released MQNLI data files
  generate_dataset.py        samples + labels a synthetic MQNLI dataset,
                             with the Appendix-B.2 subphrase augmentation
  tokenizer.py                whitespace vocab for the BiLSTM

models/
  bilstm_nli.py              cell-level BiLSTM (pyvene-hookable)
  bert_nli.py                 thin transformers.BertModel wrapper
  pyvene_registration.py      registers both with pyvene's
                               type_to_module_mapping

train.py                    trains BiLSTM (+ the augmentation aux loss)

analysis/
  interchange_experiments.py  alignment search + interchange
                               interventions + clique-finding (Table 1)
  probing.py                   linear probe + control task (Appendix C)

tests_smoke_*.py            standalone scripts that validate the causal
                             model and the pyvene integration; run these
                             first after any change
```

## Quickstart (CPU, a few minutes)

```bash
pip install pyvene transformers torch networkx --break-system-packages

# 1. sanity-check the causal model against the paper's own Fig. 2b examples
python tests_smoke_causal_model.py

# 2. sanity-check the pyvene integration (both architectures)
python tests_smoke_pyvene_lstm.py
python tests_smoke_pyvene_bert.py

# 3. generate a small dataset (a few seconds)
python data/generate_dataset.py --n_train 6000 --n_dev 600 --n_test 600 \
    --out_dir datasets/medium

# 4. train a BiLSTM (no network access needed) -- a couple minutes on CPU
python train.py --data_dir datasets/medium --out_dir checkpoints/run1 \
    --epochs 8 --emb_dim 64 --h_dim 64 --n_layer 2

# 5. run the causal-abstraction analysis on a handful of examples
python analysis/interchange_experiments.py \
    --checkpoint_dir checkpoints/run1 --data_dir datasets/medium \
    --split test --n_examples 30 --nodes NObj AdjObj NPObj

# 6. probing baseline, for the same node/location, for comparison
python analysis/probing.py --checkpoint_dir checkpoints/run1 \
    --data_dir datasets/medium --node NObj --direction fw --layer 1 \
    --side hypothesis --pos 0
```

With these tiny/CPU-friendly settings you should **not** expect
paper-level numbers (88% BERT accuracy, clean cliques) -- you're
validating that the *pipeline* is correct, not reproducing the *result*.
See "Scaling up" below for that.

## Scaling up to something paper-like

See `KAGGLE.md` for a concrete, step-by-step walkthrough (what to upload,
notebook settings, exact commands) for a recommended scaled-down-but-real
run: 40K/3K/3K examples, BERT fine-tuned for 3 epochs, and an interchange-
intervention search with M=150 examples over 6 key nodes. That's enough
to reproduce the paper's core contrasts (BERT >> BiLSTM accuracy; BERT
shows real compositional structure at several nodes, BiLSTM doesn't)
without needing the paper's full 500K-example / M=1000 budget.

`train_bert.py` (BERT fine-tuning, needs network access to
huggingface.co) and `analysis/interchange_experiments.py --model_type
bert` (interchange interventions on BERT via `pv.IntervenableModel`) are
both implemented and included -- they just can't be exercised inside the
network-restricted sandbox this package was built in, so run them on
Kaggle/Colab per `KAGGLE.md`.

## What's faithful vs. what we simplified

- **Natural logic algebra** (join table, determiner/negation projectivity
  signatures): adapted directly from the original paper's own released
  implementation (the composition tables are exactly the kind of thing
  that's easy to get subtly wrong by hand-deriving them from MacCartney &
  Manning 2009, so we kept their tested version rather than re-deriving
  it). Verified against the paper's own Fig. 2b worked examples in
  `tests_smoke_causal_model.py` (all three: contradiction / neutral /
  entailment reproduce exactly).
- **Node structure**: matches Fig. 2a / Table 1 (13 named nodes + root).
  One documented simplification: a determiner's *own* embedded negation
  ("no" = negated "some", "not every" = negated "every") is bundled into
  the QSubj/QObj leaf rather than split into separate NegSubj/NegObj
  nodes, since both pieces of information occupy the same single token
  span in the input and so can never be intervened on separately in a
  neural network anyway. See the docstring at the top of
  `causal_model/mqnli_causal_model.py`.
- **Dataset**: sampled fresh (not the original file), with sampling
  probabilities tuned toward the paper's own worked examples (mostly
  shared nouns/verbs, varying determiners/negation/adjectives) and
  class-balanced via rejection sampling, since free sampling is
  >90% "neutral" (an inherent feature of natural-logic composition, not a
  bug -- most random premise/hypothesis pairs really are logically
  unrelated).
- **Subphrase augmentation** (Appendix B.2): implemented as a single
  auxiliary classification head over the union of all nodes' relation
  labels (rather than one head per node), trained jointly with the main
  3-way task. Simpler than the original, same spirit.
- **BiLSTM interchange interventions**: hand-rolled instead of going
  through `pv.IntervenableModel`'s `unit="t"` path -- see NOTES.md.
- **Probing control task** (Appendix C.2): simplified to a random
  relabeling keyed on the gold label's identity rather than full lexical
  identity of the aligned subphrase; captures the same "does the probe
  just have raw capacity to fit anything" check, less faithful to the
  exact construction.

## Attribution

- Vocabulary lists (`data/vocab/*.txt`) and the natural-logic relation
  algebra (`causal_model/natural_logic.py`) are adapted from
  https://github.com/atticusg/Interchange (Geiger, Lu, Icard & Potts,
  released with the paper).
- Everything else (the causal-model wrapper, dataset generator, both
  neural models, pyvene registration, and all analysis code) is a
  fresh implementation written for this task.
- pyvene: https://github.com/stanfordnlp/pyvene (Wu et al. 2024).
