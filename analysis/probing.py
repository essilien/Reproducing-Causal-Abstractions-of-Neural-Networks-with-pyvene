"""
Probing baseline (Appendix C): for a given causal-model node N and neural
location L, train a small linear probe to predict N's relation from the
neural representation at L, plus a "control task" (Hewitt & Liang 2019)
that predicts a *random* fixed relabeling of the same inputs, so that
selectivity = probe_accuracy - control_accuracy factors out probe
capacity. The paper's headline point (confirmed in Fig. 4 / Appendix D) is
that probes find information nearly everywhere (high accuracy, but often
low selectivity), while interchange interventions are far more
discriminating -- this module exists so you can reproduce that contrast
directly against `analysis/interchange_experiments.py`'s clique sizes.
"""
import argparse
import json
import os
import random
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from causal_model import Sentence, MQNLIExample, compute_leaves, compose, LEAF_NODES, PHRASAL_NODES, leaf_signature_id
from causal_model.node_spans import NODE_TOKEN_SLICE
from data.tokenizer import WordVocab, encode_example, PAD
from models.bilstm_nli import BiLSTMConfig, BiLSTMForNLI
from analysis.interchange_experiments import load_examples, encode_batch, PREMISE_OFFSET, HYP_OFFSET


def node_gold_label(node, ex: MQNLIExample):
    if node in ("QSubj", "QObj", "Neg"):
        return str(leaf_signature_id(node, ex.premise, ex.hypothesis))
    leaves = compute_leaves(ex.premise, ex.hypothesis)
    if node in leaves:
        return leaves[node]
    return compose(leaves)[node]


def build_control_task(gold_labels, seed=0):
    """Random mapping surface-identity -> random label (Hewitt & Liang
    2019 control task), keyed by the *gold label's own identity* here for
    simplicity (a stand-in that still tests probe memorization capacity
    without needing full lexical-identity keys)."""
    rng = random.Random(seed)
    uniq = sorted(set(gold_labels))
    random_targets = list(range(len(uniq)))
    rng.shuffle(random_targets)
    mapping = {u: t for u, t in zip(uniq, random_targets)}
    return [mapping[g] for g in gold_labels]


def get_hidden_states(model, ids, direction, layer, abs_pos):
    """Returns (N, h_dim) hidden states for a batch, one example at a
    time (reuses BiLSTMEncoder.forward_capture)."""
    reps = []
    with torch.no_grad():
        for i in range(ids.shape[0]):
            fw_states, bw_states = model.lstm.forward_capture(ids[i:i + 1])
            states = fw_states if direction == "fw" else bw_states
            reps.append(states[abs_pos][layer].squeeze(0))
    return torch.stack(reps, dim=0)


def train_probe(features, labels, n_classes, epochs=200, lr=0.05, weight_decay=0.01):
    probe = nn.Linear(features.shape[1], n_classes)
    opt = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)
    n = features.shape[0]
    n_train = max(int(n * 0.8), 1)
    idx = list(range(n))
    random.Random(0).shuffle(idx)
    train_idx, dev_idx = idx[:n_train], idx[n_train:] or idx[:1]
    for _ in range(epochs):
        opt.zero_grad()
        logits = probe(features[train_idx])
        loss = nn.functional.cross_entropy(logits, labels[train_idx])
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = probe(features[dev_idx]).argmax(-1)
        acc = (pred == labels[dev_idx]).float().mean().item()
    return acc


def probe_node(model, vocab, examples, node, direction, layer, local_pos, side):
    abs_pos = (PREMISE_OFFSET if side == "premise" else HYP_OFFSET) + local_pos
    ids = encode_batch(vocab, examples)
    features = get_hidden_states(model, ids, direction, layer, abs_pos)

    gold = [node_gold_label(node, ex) for ex in examples]
    uniq = sorted(set(gold))
    if len(uniq) < 2:
        return None
    label2id = {g: i for i, g in enumerate(uniq)}
    gold_ids = torch.tensor([label2id[g] for g in gold])
    real_acc = train_probe(features, gold_ids, len(uniq))

    control = build_control_task(gold)
    control_ids = torch.tensor(control)
    n_control_classes = len(set(control))
    control_acc = train_probe(features, control_ids, n_control_classes)

    return {"accuracy": real_acc, "control_accuracy": control_acc, "selectivity": real_acc - control_acc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n_examples", type=int, default=200)
    ap.add_argument("--node", required=True)
    ap.add_argument("--direction", choices=["fw", "bw"], default="fw")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--side", choices=["premise", "hypothesis"], default="premise")
    ap.add_argument("--pos", type=int, default=0)
    args = ap.parse_args()

    vocab = WordVocab.load(os.path.join(args.checkpoint_dir, "vocab.json"))
    config = BiLSTMConfig.from_pretrained(args.checkpoint_dir)
    model = BiLSTMForNLI(config)
    model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, "model.pt"), map_location="cpu"))
    model.eval()

    examples = load_examples(os.path.join(args.data_dir, f"{args.split}.jsonl"), n=args.n_examples)
    result = probe_node(model, vocab, examples, args.node, args.direction, args.layer, args.pos, args.side)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
