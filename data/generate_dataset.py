"""
Generate a synthetic MQNLI-style dataset.

This is *not* a byte-for-byte reproduction of the original 500K/60K/10K
MQNLI splits (those were released as a fixed static file we don't have
access to here) -- it is a from-scratch generator that samples sentence
pairs from the same vocabulary categories described in the paper
(Appendix A.1: 100 subject nouns, 100 object nouns, 100 subject
adjectives, 100 object adjectives, 100 adverbs, 100 verbs, determiners
{some, every, no, not every}) and labels every example -- and every
"augmented" subphrase example -- using the *exact* CNatLog causal model
in `causal_model/`, so the resulting dataset has the correct compositional
structure needed for the causal-abstraction analysis to be meaningful.

Usage:
    python generate_dataset.py --n_train 20000 --n_dev 2000 --n_test 2000 \
        --out_dir ../datasets/small
"""
import argparse
import json
import os
import random
import sys
from dataclasses import asdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from causal_model import (
    Sentence, MQNLIExample, compute_label, compute_leaves, compose,
    LEAF_NODES, PHRASAL_NODES, leaf_signature_id,
)

VOCAB_DIR = os.path.join(os.path.dirname(__file__), "vocab")
DETERMINERS = ("some", "every", "no", "not every")

# (leaf/phrasal node -> which raw fields of Sentence it spans, for
# subphrase augmentation). Each entry is (premise_field_getter, hyp_field_getter)
# expressed as attribute name lists to pull off a Sentence.
NODE_SPAN_FIELDS = {
    "QSubj": ["subject_determiner"],
    "AdjSubj": ["subject_adjective"],
    "NSubj": ["subject_noun"],
    "Neg": ["negated"],
    "Adv": ["adverb"],
    "V": ["verb", "verb_3ps"],
    "QObj": ["object_determiner"],
    "AdjObj": ["object_adjective"],
    "NObj": ["object_noun"],
    "NPSubj": ["subject_adjective", "subject_noun"],
    "NPObj": ["object_adjective", "object_noun"],
    "VP": ["adverb", "verb", "verb_3ps"],
    "QPObj": ["object_determiner", "object_adjective", "object_noun", "verb", "verb_3ps", "adverb"],
    "NegP": ["negated", "object_determiner", "object_adjective", "object_noun", "verb", "verb_3ps", "adverb"],
}


def load_vocab(fname, n_fields=1):
    path = os.path.join(VOCAB_DIR, fname)
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(line.split())
    return items


class Vocabulary:
    def __init__(self):
        self.subject_nouns = [x[0] for x in load_vocab("subject_nouns.txt")]
        self.object_nouns = [x[0] for x in load_vocab("object_nouns.txt")]
        self.subject_adjs = [x[0] for x in load_vocab("subject_adjectives.txt")]
        self.object_adjs = [x[0] for x in load_vocab("object_adjectives.txt")]
        self.adverbs = [x[0] for x in load_vocab("adverbs.txt")]
        verbs = load_vocab("verbs.txt")  # "eats eaten eat" per line: 3ps, past-participle, base
        self.verbs_3ps = [v[0] for v in verbs]
        self.verbs_base = [v[2] for v in verbs]
        assert len(self.verbs_3ps) == len(self.verbs_base)


def _sample_pair(vocab_list, rng, p_same=0.35, p_empty=0.0):
    """Sample a (premise_value, hypothesis_value) pair from vocab_list.
    With prob p_same they're forced identical (-> 'equivalence' leaf
    relation); with prob p_empty either side is set to "" (only used for
    optional adjective/adverb slots); otherwise independently sampled
    (typically -> 'independence', since the vocab items are constructed to
    be mutually unrelated)."""
    if p_empty > 0 and rng.random() < p_empty:
        p_val = ""
    else:
        p_val = rng.choice(vocab_list)
    if rng.random() < p_same:
        h_val = p_val
    elif p_empty > 0 and rng.random() < p_empty:
        h_val = ""
    else:
        h_val = rng.choice(vocab_list)
    return p_val, h_val


def _sample_determiner_pair(rng, p_same=0.3):
    p = rng.choice(DETERMINERS)
    h = p if rng.random() < p_same else rng.choice(DETERMINERS)
    return p, h


def sample_example(vocab: Vocabulary, rng: random.Random) -> MQNLIExample:
    # Content words (nouns/verbs) mostly match across premise/hypothesis --
    # MQNLI's interesting logical relations come from varying determiners,
    # negation, and (to a lesser extent) adjectives/adverbs, exactly as in
    # the paper's own examples (Fig. 2b: nouns and verbs are shared, only
    # the determiners/adjectives/negation differ).
    sd_p, sd_h = _sample_determiner_pair(rng, p_same=0.35)
    od_p, od_h = _sample_determiner_pair(rng, p_same=0.35)
    sadj_p, sadj_h = _sample_pair(vocab.subject_adjs, rng, p_same=0.55, p_empty=0.35)
    oadj_p, oadj_h = _sample_pair(vocab.object_adjs, rng, p_same=0.55, p_empty=0.35)
    adv_p, adv_h = _sample_pair(vocab.adverbs, rng, p_same=0.55, p_empty=0.35)
    sn_p, sn_h = _sample_pair(vocab.subject_nouns, rng, p_same=0.85)
    on_p, on_h = _sample_pair(vocab.object_nouns, rng, p_same=0.85)
    v_idx_p = rng.randrange(len(vocab.verbs_base))
    if rng.random() < 0.9:
        v_idx_h = v_idx_p
    else:
        v_idx_h = rng.randrange(len(vocab.verbs_base))
    neg_p = rng.random() < 0.25
    neg_h = rng.random() < 0.25

    premise = Sentence(
        subject_determiner=sd_p, subject_adjective=sadj_p, subject_noun=sn_p,
        negated=neg_p, adverb=adv_p, verb=vocab.verbs_base[v_idx_p], verb_3ps=vocab.verbs_3ps[v_idx_p],
        object_determiner=od_p, object_adjective=oadj_p, object_noun=on_p,
    )
    hypothesis = Sentence(
        subject_determiner=sd_h, subject_adjective=sadj_h, subject_noun=sn_h,
        negated=neg_h, adverb=adv_h, verb=vocab.verbs_base[v_idx_h], verb_3ps=vocab.verbs_3ps[v_idx_h],
        object_determiner=od_h, object_adjective=oadj_h, object_noun=on_h,
    )
    return MQNLIExample(premise=premise, hypothesis=hypothesis)


def example_to_json(ex: MQNLIExample, idx: int) -> dict:
    label = compute_label(ex)
    leaves = compute_leaves(ex.premise, ex.hypothesis)
    nodes = compose(leaves)
    node_relations = {}
    for n in PHRASAL_NODES:
        node_relations[n] = nodes[n]
    for n in ("AdjSubj", "NSubj", "Adv", "V", "AdjObj", "NObj"):
        node_relations[n] = leaves[n]
    for n in ("QSubj", "QObj", "Neg"):
        node_relations[n] = "|".join(str(x) for x in leaf_signature_id(n, ex.premise, ex.hypothesis))
    return {
        "idx": idx,
        "premise_tokens": ex.premise.tokens(),
        "hypothesis_tokens": ex.hypothesis.tokens(),
        "premise_str": ex.premise.surface_string(),
        "hypothesis_str": ex.hypothesis.surface_string(),
        "premise_fields": asdict(ex.premise),
        "hypothesis_fields": asdict(ex.hypothesis),
        "label": label,
        "node_relations": node_relations,
    }


def masked_subphrase_tokens(ex: MQNLIExample, node: str, mask_token="[PAD]"):
    """Build a subphrase-augmented example (Appendix B.2): keep both full
    sentences' *positions* (so the location of the node in the sequence is
    unchanged -- important since the neural models later get intervened on
    at fixed token positions) but mask out every token that doesn't belong
    to the aligned span under `node`."""
    fields = set(NODE_SPAN_FIELDS[node])
    # crude field->token-index map mirroring Sentence.tokens()
    field_to_slice = {
        "subject_determiner": slice(0, 2), "subject_adjective": slice(2, 3), "subject_noun": slice(3, 4),
        "negated": slice(4, 6), "adverb": slice(6, 7), "verb": slice(7, 8), "verb_3ps": slice(7, 8),
        "object_determiner": slice(8, 10), "object_adjective": slice(10, 11), "object_noun": slice(11, 12),
    }
    keep_idx = set()
    for f in fields:
        s = field_to_slice[f]
        keep_idx.update(range(s.start, s.stop))

    def mask(tokens):
        return [t if i in keep_idx else mask_token for i, t in enumerate(tokens)]

    return mask(ex.premise.tokens()), mask(ex.hypothesis.tokens())


def build_augmented_examples(ex: MQNLIExample, idx: int):
    """One example per intermediate node (Appendix B.2), each labeled with
    that node's own relation (kept in a *separate* per-node label
    namespace, disjoint from the 3-way sentence labels, exactly as
    described in the paper)."""
    leaves = compute_leaves(ex.premise, ex.hypothesis)
    nodes = compose(leaves)
    out = []
    for node in list(LEAF_NODES) + list(PHRASAL_NODES):
        if node in ("QSubj", "QObj", "Neg"):
            node_label = f"{node}={leaf_signature_id(node, ex.premise, ex.hypothesis)}"
        elif node in leaves:
            node_label = f"{node}={leaves[node]}"
        else:
            node_label = f"{node}={nodes[node]}"
        p_tok, h_tok = masked_subphrase_tokens(ex, node)
        out.append({
            "idx": f"{idx}-{node}",
            "premise_tokens": p_tok,
            "hypothesis_tokens": h_tok,
            "label": node_label,
            "is_augmented": True,
            "node": node,
        })
    return out


def generate_split(vocab, rng, n, augment=False, balance=True, oversample_factor=8):
    """Generate `n` base examples (plus, if `augment`, their subphrase
    augmentations). If `balance`, we rejection-sample so the three labels
    are as close to evenly split as possible: under free sampling,
    'neutral' dominates (as in natural logic generally -- most random
    premise/hypothesis pairs are logically unrelated), so we oversample a
    pool and bucket by label."""
    seen_labels = {"entailment": 0, "neutral": 0, "contradiction": 0}
    examples = []
    if not balance:
        for base_idx in range(n):
            ex = sample_example(vocab, rng)
            rec = example_to_json(ex, base_idx)
            rec["is_augmented"] = False
            examples.append(rec)
            seen_labels[rec["label"]] += 1
            if augment:
                examples.extend(build_augmented_examples(ex, rec["idx"]))
        return examples, seen_labels

    target_per_class = n // 3
    buckets = {"entailment": [], "neutral": [], "contradiction": []}
    pool_size = n * oversample_factor
    tries = 0
    max_tries = pool_size * 50
    while min(len(v) for v in buckets.values()) < target_per_class and tries < max_tries:
        tries += 1
        ex = sample_example(vocab, rng)
        label = compute_label(ex)
        if len(buckets[label]) < target_per_class:
            buckets[label].append(ex)
    base_idx = 0
    for label, exs in buckets.items():
        for ex in exs:
            rec = example_to_json(ex, base_idx)
            rec["is_augmented"] = False
            examples.append(rec)
            seen_labels[rec["label"]] += 1
            if augment:
                examples.extend(build_augmented_examples(ex, rec["idx"]))
            base_idx += 1
    rng.shuffle(examples)
    return examples, seen_labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=20000)
    ap.add_argument("--n_dev", type=int, default=2000)
    ap.add_argument("--n_test", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--augment_train", action="store_true", default=True)
    ap.add_argument("--no_augment_train", dest="augment_train", action="store_false")
    ap.add_argument("--balance", action="store_true", default=True)
    ap.add_argument("--no_balance", dest="balance", action="store_false")
    ap.add_argument("--out_dir", type=str, default=os.path.join(os.path.dirname(__file__), "..", "datasets", "small"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    vocab = Vocabulary()
    rng = random.Random(args.seed)

    for split, n, augment in [
        ("train", args.n_train, args.augment_train),
        ("dev", args.n_dev, False),
        ("test", args.n_test, False),
    ]:
        examples, label_counts = generate_split(vocab, rng, n, augment=augment, balance=args.balance)
        out_path = os.path.join(args.out_dir, f"{split}.jsonl")
        with open(out_path, "w") as f:
            for rec in examples:
                f.write(json.dumps(rec) + "\n")
        print(f"{split}: wrote {len(examples)} examples ({n} base + augmentation) to {out_path}")
        print(f"  label distribution (base examples): {label_counts}")


if __name__ == "__main__":
    main()
