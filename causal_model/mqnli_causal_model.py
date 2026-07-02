"""
CNatLog: the natural-logic causal model that generates MQNLI labels
(Geiger, Lu, Icard & Potts 2021, Fig. 2a / Appendix H.1), reimplemented
as a clean, self-contained, forward-computable + intervenable causal
model, on top of the relation algebra in `natural_logic.py`.

Sentence template (Appendix A.1), 9 "slots":
    QSubj AdjSubj NSubj Neg Adv V QObj AdjObj NObj
Node structure (Fig. 2a): each slot is a *leaf* of the tree; leaves are
composed bottom-up into 5 *phrasal* nodes, and finally into the sentence
label:

    NPSubj = phrase(AdjSubj, NSubj)
    NPObj  = phrase(AdjObj, NObj)
    VP     = phrase(Adv, V)
    QPObj  = det(QObj, NPObj, VP)         # object DP, after object's own
                                           # quantifier *and* any negation
                                           # baked into "no"/"not every"
    NegP   = neg(Neg, QPObj)              # after main-clause "does not"
    root   = det(QSubj, NPSubj, NegP)     # after subject's own quantifier
                                           # and any "no"/"not every"

Design note on negation: MQNLI sentences carry negation in up to two
independent places -- inside a determiner ("no" = negated "some", "not
every" = negated "every") and as main-clause negation ("does not"). Since
a determiner and its embedded negation live in the *same* input token
span, a neural network intervention at that span necessarily moves both
pieces of information together, so we bundle them into a single QSubj /
QObj leaf (each leaf's value is a *signature*, i.e. a function of the two
relations it will combine, rather than a bare relation -- see
`natural_logic.py`). Main-clause negation ("Neg") is a separate leaf at
its own token span.

This module reuses the relation algebra released with the original paper
(see natural_logic.py docstring for provenance) but is otherwise an
independent, from-scratch implementation.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, Any

from . import natural_logic as nl

DETERMINERS = ("some", "every", "no", "not every")
LEAF_NODES = ("QSubj", "AdjSubj", "NSubj", "Neg", "Adv", "V", "QObj", "AdjObj", "NObj")
PHRASAL_NODES = ("NPSubj", "NPObj", "VP", "QPObj", "NegP")
ALL_NODES = LEAF_NODES + PHRASAL_NODES  # "root" is the sentence-level output, handled separately


def _split_determiner(det: str) -> Tuple[str, bool]:
    """"no" -> ("some", True); "not every" -> ("every", True);
    "some"/"every" -> (det, False)."""
    if det == "no":
        return "some", True
    if det == "not every":
        return "every", True
    if det in ("some", "every"):
        return det, False
    raise ValueError(f"unknown determiner {det!r}")


@dataclass(frozen=True)
class Sentence:
    """One simple sentence: `QDet AdjSubj NSubj [does not] Adv V QDet AdjObj NObj`."""
    subject_determiner: str
    subject_adjective: str  # "" if absent
    subject_noun: str
    negated: bool            # main-clause "does not"
    adverb: str               # "" if absent
    verb: str                 # base form, e.g. "eat"
    verb_3ps: str              # 3rd person singular present, e.g. "eats"
    object_determiner: str
    object_adjective: str
    object_noun: str

    def tokens(self) -> list:
        """The fixed-width, ε-padded token sequence (Appendix A.1/B.1).
        QSubj(2) AdjSubj(1) NSubj(1) Neg(2) Adv(1) V(1) QObj(2) AdjObj(1) NObj(1)
        = 12 tokens, identical length/alignment for every sentence."""
        eps = "\u03b5"  # ε
        qs_tokens = ["not", "every"] if self.subject_determiner == "not every" \
            else [eps, self.subject_determiner]
        qo_tokens = ["not", "every"] if self.object_determiner == "not every" \
            else [eps, self.object_determiner]
        neg_tokens = ["does", "not"] if self.negated else [eps, eps]
        return (
            qs_tokens
            + [self.subject_adjective or eps, self.subject_noun]
            + neg_tokens
            + [self.adverb or eps, self.verb_3ps if not self.negated else self.verb]
            + qo_tokens
            + [self.object_adjective or eps, self.object_noun]
        )

    def surface_string(self) -> str:
        return " ".join(t for t in self.tokens())


@dataclass(frozen=True)
class MQNLIExample:
    premise: Sentence
    hypothesis: Sentence


# --------------------------------------------------------------------------
# Leaf computation
# --------------------------------------------------------------------------

def _combined_determiner_signature(det_p: str, det_h: str) -> Dict[Tuple[str, str], str]:
    """QSubj / QObj leaf value: a signature dict (rel_NP, rel_pred) -> rel_out
    that bundles the bare quantifier composition with the composition of
    whatever negation is embedded in "no" / "not every"."""
    bare_p, neg_p = _split_determiner(det_p)
    bare_h, neg_h = _split_determiner(det_h)
    bare_sig = nl.determiner_merge(bare_p, bare_h)
    neg_sig = nl.negation_merge(neg_p, neg_h)
    return {key: neg_sig[val] for key, val in bare_sig.items()}


def compute_leaves(premise: Sentence, hypothesis: Sentence) -> Dict[str, Any]:
    """Returns a dict with the 9 leaf-node values. AdjSubj/NSubj/Adv/V/
    AdjObj/NObj are plain relations (one of the 7 MacCartney relations,
    though open-class lexical comparison only ever yields
    {equivalence, entails, reverse entails, independence}). QSubj/QObj/Neg
    are *signatures* (callables via dict lookup), since they need to be
    composed with something else further up the tree."""
    p, h = premise, hypothesis
    return {
        "QSubj": _combined_determiner_signature(p.subject_determiner, h.subject_determiner),
        "AdjSubj": nl.standard_lexical_merge(p.subject_adjective, h.subject_adjective),
        "NSubj": nl.standard_lexical_merge(p.subject_noun, h.subject_noun),
        "Neg": nl.negation_merge(p.negated, h.negated),
        "Adv": nl.standard_lexical_merge(p.adverb, h.adverb),
        "V": nl.standard_lexical_merge(p.verb, h.verb),
        "QObj": _combined_determiner_signature(p.object_determiner, h.object_determiner),
        "AdjObj": nl.standard_lexical_merge(p.object_adjective, h.object_adjective),
        "NObj": nl.standard_lexical_merge(p.object_noun, h.object_noun),
    }


def compose(leaves: Dict[str, Any], override: Optional[Tuple[str, Any]] = None) -> Dict[str, str]:
    """Compose the 9 leaves into the 5 phrasal nodes + the sentence-level
    `root` relation. If `override` = (node_name, value) is given, that
    node's value is forced to `value` instead of being computed from
    `leaves` -- this is exactly an *interchange intervention* on the
    high-level causal model (eq. 4/6 in the paper): everything upstream of
    `node_name` is irrelevant, everything downstream is recomputed using
    the overridden value, exactly as it would be in the real tree."""
    def node(name, compute_fn):
        if override is not None and override[0] == name:
            return override[1]
        return compute_fn()

    NPSubj = node("NPSubj", lambda: nl.standard_phrase(leaves["AdjSubj"], leaves["NSubj"]))
    NPObj = node("NPObj", lambda: nl.standard_phrase(leaves["AdjObj"], leaves["NObj"]))
    VP = node("VP", lambda: nl.standard_phrase(leaves["Adv"], leaves["V"]))
    QPObj = node("QPObj", lambda: nl.determiner_phrase(leaves["QObj"], NPObj, VP))
    NegP = node("NegP", lambda: nl.negation_phrase(leaves["Neg"], QPObj))
    root = node("root", lambda: nl.determiner_phrase(leaves["QSubj"], NPSubj, NegP))
    return {"NPSubj": NPSubj, "NPObj": NPObj, "VP": VP, "QPObj": QPObj, "NegP": NegP, "root": root}


def compute_label(example: MQNLIExample) -> str:
    leaves = compute_leaves(example.premise, example.hypothesis)
    root = compose(leaves)["root"]
    return nl.relation_to_label(root)


def intervene(base: MQNLIExample, source: MQNLIExample, node: str) -> str:
    """Implements C^{node <- source}_NatLog(base) from eq. (4)/(6): compute
    `base`'s leaves, but with `node`'s value swapped in from `source`, then
    recompute everything downstream. Returns the resulting 3-way NLI label.
    `node` may be any entry of LEAF_NODES or PHRASAL_NODES."""
    base_leaves = compute_leaves(base.premise, base.hypothesis)
    src_leaves = compute_leaves(source.premise, source.hypothesis)
    if node in LEAF_NODES:
        merged = dict(base_leaves)
        merged[node] = src_leaves[node]
        root = compose(merged)["root"]
    elif node in PHRASAL_NODES:
        src_value = compose(src_leaves)[node]
        root = compose(base_leaves, override=(node, src_value))["root"]
    else:
        raise ValueError(f"unknown node {node!r}")
    return nl.relation_to_label(root)


def leaf_signature_id(node: str, premise: Sentence, hypothesis: Sentence) -> Any:
    """A small discrete id for the leaf's *identity* (used as a probing /
    control-task target for QSubj, QObj, Neg, whose "value" is a function
    rather than a bare relation -- see Appendix C.2)."""
    if node == "Neg":
        return (premise.negated, hypothesis.negated)
    if node == "QSubj":
        bp, np_ = _split_determiner(premise.subject_determiner)
        bh, nh = _split_determiner(hypothesis.subject_determiner)
        return (bp, np_, bh, nh)
    if node == "QObj":
        bp, np_ = _split_determiner(premise.object_determiner)
        bh, nh = _split_determiner(hypothesis.object_determiner)
        return (bp, np_, bh, nh)
    raise ValueError(f"{node} is not a signature-valued leaf")
