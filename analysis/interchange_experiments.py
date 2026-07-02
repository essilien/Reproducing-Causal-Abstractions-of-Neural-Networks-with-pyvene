"""
Reproduces the paper's Section 5.1 methodology: for each causal-model node
N, search over candidate neural locations L, run interchange interventions
for every ordered pair of examples, and report the size of the largest
"clique" of examples (Section 5, "Quantifying Partial Success") on which
C^N_NatLog is a constructive abstraction of the neural model at L.

pyvene usage note: for the BERT model, this uses `pv.IntervenableModel`
directly (the standard, well-supported "pos"-unit Transformer path -- see
`models/pyvene_registration.py::register_bert_nli`). For the BiLSTM, we
found that this installed version of pyvene's `unit="t"` handling for
*custom* (non-built-in-GRU) recurrent modules does not reliably respect
the requested timestep (empirically, interventions always landed on the
2nd invocation of the hooked module regardless of the configured
position -- verified by a self-intervention no-op sweep across positions,
see `tests_smoke_pyvene_lstm.py` / project README for details). Since a
correct interchange intervention is the entire point of this analysis, we
therefore use `BiLSTMEncoder.forward_with_intervention` (a small,
directly-verified, hand-rolled implementation of exactly the same
operation) for the BiLSTM case, and reserve `pv.IntervenableModel` for
BERT, where it is fully reliable.
"""
import argparse
import itertools
import json
import os
import random
import sys
from collections import defaultdict

import networkx as nx
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from causal_model import Sentence, MQNLIExample, compute_label, intervene, LEAF_NODES, PHRASAL_NODES
from causal_model.node_spans import NODE_TOKEN_SLICE, SENTENCE_LEN
from data.tokenizer import WordVocab, encode_example, PAD, CLS, SEP
from models.bilstm_nli import BiLSTMConfig, BiLSTMForNLI

MAIN_LABELS = ["entailment", "neutral", "contradiction"]

# Offsets of the premise / hypothesis segments within the full
# [CLS] premise [SEP] hypothesis [SEP] sequence.
PREMISE_OFFSET = 1
HYP_OFFSET = 1 + SENTENCE_LEN + 1


def load_examples(path, n=None, seed=0):
    recs = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("is_augmented"):
                continue
            recs.append(rec)
    if n is not None and n < len(recs):
        random.Random(seed).shuffle(recs)
        recs = recs[:n]
    examples = []
    for rec in recs:
        premise = Sentence(**rec["premise_fields"])
        hypothesis = Sentence(**rec["hypothesis_fields"])
        examples.append(MQNLIExample(premise=premise, hypothesis=hypothesis))
    return examples


def candidate_positions(node):
    """Local (within-sentence, 0..11) token positions spanned by `node`,
    used as alignment-search candidates on both the premise and
    hypothesis side (mirroring Section 5.1's "hidden representations
    above the ... descendant leaf tokens")."""
    start, end = NODE_TOKEN_SLICE[node]
    return list(range(start, end))


def encode_batch(vocab, examples):
    id_lists = []
    for ex in examples:
        ids, _ = encode_example(vocab, ex.premise.tokens(), ex.hypothesis.tokens())
        id_lists.append(ids)
    max_len = max(len(x) for x in id_lists)
    pad_id = vocab.tok2id[PAD]
    out = torch.full((len(id_lists), max_len), pad_id, dtype=torch.long)
    for i, ids in enumerate(id_lists):
        out[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
    return out


def predict_labels(model, vocab, examples, batch_size=64):
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch = examples[start:start + batch_size]
            ids = encode_batch(vocab, batch)
            logits = model(input_ids=ids)["logits"]
            preds.extend(logits.argmax(-1).tolist())
    return [MAIN_LABELS[p] for p in preds]


def build_success_graph(base_labels_by_pair, impactful_pairs):
    """`base_labels_by_pair[(i,j)]` = True if intervening node N on base
    i with source j succeeds (eq. 6). Builds the undirected graph used for
    the clique search (Section 5, "Quantifying Partial Success"): an edge
    (i,j) iff BOTH (i,j) and (j,i) succeed."""
    G = nx.Graph()
    for (i, j), ok in base_labels_by_pair.items():
        if i == j:
            continue
        rev_ok = base_labels_by_pair.get((j, i), False)
        if ok and rev_ok:
            G.add_edge(i, j)
    for i in {i for i, _ in base_labels_by_pair}:
        G.add_node(i)
    return G


def largest_clique_with_impactful_edge(G, impactful_pairs):
    best = 0
    for clique in nx.find_cliques(G):
        if len(clique) < 2:
            continue
        has_impactful = any(
            (a, b) in impactful_pairs or (b, a) in impactful_pairs
            for a, b in itertools.combinations(clique, 2)
        )
        if has_impactful and len(clique) > best:
            best = len(clique)
    return best


# --------------------------------------------------------------------------
# BiLSTM path (hand-rolled intervention, see module docstring)
# --------------------------------------------------------------------------

def bilstm_node_clique_size(model, vocab, examples, node, direction, layer, local_pos, side):
    """Runs the full O(n^2) interchange-intervention search for one
    candidate location (direction, layer, local_pos, side) and returns the
    resulting clique size (as a fraction of len(examples))."""
    abs_pos = (PREMISE_OFFSET if side == "premise" else HYP_OFFSET) + local_pos
    ids = encode_batch(vocab, examples)
    pred_labels = predict_labels(model, vocab, examples)
    true_labels = [compute_label(ex) for ex in examples]
    correct_idx = [i for i in range(len(examples)) if pred_labels[i] == true_labels[i]]
    if len(correct_idx) < 4:
        return 0.0

    # capture every example's hidden state at (direction, layer) once
    fw_by_example, bw_by_example = [], []
    with torch.no_grad():
        for i in range(len(examples)):
            fw_states, bw_states = model.lstm.forward_capture(ids[i:i + 1])
            fw_by_example.append(fw_states)
            bw_by_example.append(bw_states)

    def source_value(idx):
        states = fw_by_example[idx] if direction == "fw" else bw_by_example[idx]
        return states[abs_pos][layer]

    base_label = {}
    with torch.no_grad():
        for i in correct_idx:
            logits = model(input_ids=ids[i:i + 1])["logits"]
            base_label[i] = MAIN_LABELS[logits.argmax(-1).item()]

    success, impactful = {}, set()
    with torch.no_grad():
        for i in correct_idx:
            for j in correct_idx:
                if i == j:
                    continue
                neural_out = model.logits_with_intervention(
                    ids[i:i + 1], direction, layer, abs_pos, source_value(j)
                )
                neural_label = MAIN_LABELS[neural_out.argmax(-1).item()]
                causal_label = intervene(examples[i], examples[j], node)
                success[(i, j)] = (neural_label == causal_label)
                if causal_label != base_label[i]:
                    impactful.add((i, j))

    G = build_success_graph(success, impactful)
    clique = largest_clique_with_impactful_edge(G, impactful)
    return clique / len(correct_idx)


def alignment_search_bilstm(model, vocab, examples, nodes=None, layers=None, directions=("fw", "bw")):
    nodes = nodes or (list(LEAF_NODES) + list(PHRASAL_NODES))
    layers = layers if layers is not None else range(model.config.n_layer)
    results = {}
    for node in nodes:
        best = (0.0, None)
        for direction in directions:
            for layer in layers:
                for side in ("premise", "hypothesis"):
                    for local_pos in candidate_positions(node):
                        size = bilstm_node_clique_size(model, vocab, examples, node, direction, layer, local_pos, side)
                        if size > best[0]:
                            best = (size, {"direction": direction, "layer": layer, "side": side, "pos": local_pos})
        results[node] = best
        print(f"  {node}: best clique={best[0]:.3f} at {best[1]}")
    return results


# --------------------------------------------------------------------------
# BERT path (pv.IntervenableModel, "pos" unit -- reliable & batchable, see
# NOTES.md and tests/tests_smoke_pyvene_bert.py)
# --------------------------------------------------------------------------

def bert_encode_batch(tokenizer, examples):
    from data.bert_tokenize import encode_pair_for_bert, pad_encoded_batch
    encoded, p_spans_all, h_spans_all = [], [], []
    for ex in examples:
        ids, ttids, p_spans, h_spans = encode_pair_for_bert(tokenizer, ex.premise.tokens(), ex.hypothesis.tokens())
        encoded.append((ids, ttids))
        p_spans_all.append(p_spans)
        h_spans_all.append(h_spans)
    input_ids, token_type_ids, attn = pad_encoded_batch(encoded, tokenizer.pad_token_id)
    return input_ids, token_type_ids, attn, p_spans_all, h_spans_all


def bert_predict_labels(model, input_ids, token_type_ids, attn, batch_size=32):
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, input_ids.shape[0], batch_size):
            sl = slice(start, start + batch_size)
            logits = model(input_ids=input_ids[sl], attention_mask=attn[sl], token_type_ids=token_type_ids[sl])["logits"]
            preds.extend(logits.argmax(-1).tolist())
    return [MAIN_LABELS[p] for p in preds]


def bert_node_clique_size(iv_model, model, tokenizer, examples, node, layer, side, local_pos, batch_size=16):
    """Same eq. (6)/(7) methodology as `bilstm_node_clique_size`, but using
    `pv.IntervenableModel` (the "pos" unit, verified reliable for BERT --
    see tests/tests_smoke_pyvene_bert.py), and batched across all ordered
    pairs at once since a single candidate location is fixed for the whole
    search step. `local_pos` here is a *word* index (0..11); since a word
    may span multiple subwords, we additionally search every subword
    position within it and keep the best (mirrors the paper's own
    per-token alignment search over a node's descendant leaf tokens)."""
    input_ids, token_type_ids, attn, p_spans_all, h_spans_all = bert_encode_batch(tokenizer, examples)
    pred_labels = bert_predict_labels(model, input_ids, token_type_ids, attn, batch_size)
    true_labels = [compute_label(ex) for ex in examples]
    correct_idx = [i for i in range(len(examples)) if pred_labels[i] == true_labels[i]]
    if len(correct_idx) < 4:
        return 0.0, None

    with torch.no_grad():
        base_logits = model(input_ids=input_ids, attention_mask=attn, token_type_ids=token_type_ids)["logits"]
    base_label = {i: MAIN_LABELS[base_logits[i].argmax(-1).item()] for i in correct_idx}

    # every example's own word->subword span may differ slightly if
    # tokenization is context-sensitive; here it isn't (each word is
    # encoded independently), so all examples share the same *word*
    # position -> use example 0's span to pick which subword offsets to try.
    spans = p_spans_all[0] if side == "premise" else h_spans_all[0]
    subword_positions = list(range(*spans[local_pos]))

    best = (0.0, None)
    for abs_pos in subword_positions:
        success, impactful = {}, set()
        pairs = [(i, j) for i in correct_idx for j in correct_idx if i != j]
        for start in range(0, len(pairs), batch_size):
            chunk = pairs[start:start + batch_size]
            base_idx = torch.tensor([p[0] for p in chunk])
            src_idx = torch.tensor([p[1] for p in chunk])
            with torch.no_grad():
                _, out = iv_model(
                    base={"input_ids": input_ids[base_idx], "attention_mask": attn[base_idx],
                          "token_type_ids": token_type_ids[base_idx]},
                    sources=[{"input_ids": input_ids[src_idx], "attention_mask": attn[src_idx],
                              "token_type_ids": token_type_ids[src_idx]}],
                    unit_locations={"sources->base": abs_pos},
                )
            neural_labels = [MAIN_LABELS[l] for l in out["logits"].argmax(-1).tolist()]
            for (i, j), neural_label in zip(chunk, neural_labels):
                causal_label = intervene(examples[i], examples[j], node)
                success[(i, j)] = (neural_label == causal_label)
                if causal_label != base_label[i]:
                    impactful.add((i, j))
        G = build_success_graph(success, impactful)
        clique = largest_clique_with_impactful_edge(G, impactful)
        size = clique / len(correct_idx)
        if size > best[0]:
            best = (size, abs_pos)
    return best


def alignment_search_bert(model, tokenizer, examples, nodes=None, layers=None):
    """See module docstring: register_bert_nli() must already have been
    called. Builds ONE `pv.IntervenableModel` per layer (component =
    "block_output") and reuses it across nodes/positions -- only the
    `unit_locations` passed at call time changes."""
    import pyvene as pv
    nodes = nodes or (list(LEAF_NODES) + list(PHRASAL_NODES))
    layers = layers if layers is not None else range(model.config.num_hidden_layers - 1)
    iv_models = {}
    for layer in layers:
        iv_config = pv.IntervenableConfig(
            {"layer": layer, "component": "block_output", "intervention_type": pv.VanillaIntervention},
            model=model,
        )
        iv_models[layer] = pv.IntervenableModel(iv_config, model=model)

    results = {}
    for node in nodes:
        best = (0.0, None)
        for layer in layers:
            for side in ("premise", "hypothesis"):
                for local_pos in candidate_positions(node):
                    size, abs_pos = bert_node_clique_size(iv_models[layer], model, tokenizer, examples, node, layer, side, local_pos)
                    if size > best[0]:
                        best = (size, {"layer": layer, "side": side, "word_pos": local_pos, "subword_pos": abs_pos})
        results[node] = best
        print(f"  {node}: best clique={best[0]:.3f} at {best[1]}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--model_type", choices=["bilstm", "bert"], default="bilstm")
    ap.add_argument("--n_examples", type=int, default=40,
                     help="Sample size M (the paper used M=1000, i.e. up to 1e6 "
                          "ordered pairs per candidate location -- this is O(M^2) "
                          "per candidate, so keep M small for a CPU smoke run.")
    ap.add_argument("--nodes", nargs="*", default=None)
    ap.add_argument("--layers", type=int, nargs="*", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    examples = load_examples(os.path.join(args.data_dir, f"{args.split}.jsonl"), n=args.n_examples)
    print(f"Loaded {len(examples)} examples from {args.split}.jsonl")

    if args.model_type == "bilstm":
        vocab = WordVocab.load(os.path.join(args.checkpoint_dir, "vocab.json"))
        config = BiLSTMConfig.from_pretrained(args.checkpoint_dir)
        model = BiLSTMForNLI(config)
        model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, "model.pt"), map_location="cpu"))
        model.eval()
        results = alignment_search_bilstm(model, vocab, examples, nodes=args.nodes,
                                           layers=args.layers)
    else:
        from transformers import AutoTokenizer
        from models.bert_nli import BertForNLI
        from models.pyvene_registration import register_bert_nli
        register_bert_nli()
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir)
        model = BertForNLI(args.checkpoint_dir)
        model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, "model.pt"), map_location="cpu"))
        model.eval()
        results = alignment_search_bert(model, tokenizer, examples, nodes=args.nodes, layers=args.layers)

    if args.out:
        with open(args.out, "w") as f:
            json.dump({k: v for k, v in results.items()}, f, indent=2, default=str)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
