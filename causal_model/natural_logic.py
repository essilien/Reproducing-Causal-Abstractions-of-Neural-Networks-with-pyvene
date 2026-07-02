"""
Natural logic relation algebra (MacCartney & Manning 2009).

This module implements the seven basic natural-logic relations and the
"join"/composition table, plus the determiner and negation "projectivity
signatures" needed to compositionally compute the relation between a
premise and a hypothesis sentence in MQNLI.

The algebra here is adapted (cleaned up, type-annotated, and re-organized,
but mathematically identical) from the implementation released by the
authors of "Causal Abstractions of Neural Networks" (Geiger, Lu, Icard &
Potts, NeurIPS 2021) at https://github.com/atticusg/Interchange
(mqnli/natural_logic_model.py). We keep it because re-deriving the join
table and the projectivity signatures for "every"/"some"/"not every"/"no"
by hand is exactly the kind of thing that is easy to get subtly wrong, and
the original implementation has already been validated against the MQNLI
dataset used in the paper.

The seven relations, using MacCartney's symbols (see paper Fig. 2a caption
and Appendix H.1):
    equivalence      (=)   e.g. "banker" / "banker"
    entails          (<)   e.g. "some banker" entails "some person"
    reverse entails   (>)
    contradiction    (^)
    cover            (v)
    alternation      (|)
    independence     (#)
"""
from __future__ import annotations
from typing import Dict, Tuple

RELATIONS = [
    "equivalence", "entails", "reverse entails",
    "contradiction", "cover", "alternation", "independence",
]

# --- The MacCartney "join" table: composing two relations along a chain ---
_relation_composition: Dict[Tuple[str, str], str] = {
    (r, r2): "independence" for r in RELATIONS for r2 in RELATIONS
}
for r in RELATIONS:
    _relation_composition[("equivalence", r)] = r
    _relation_composition[(r, "equivalence")] = r
_relation_composition[("entails", "entails")] = "entails"
_relation_composition[("entails", "contradiction")] = "alternation"
_relation_composition[("entails", "alternation")] = "alternation"
_relation_composition[("reverse entails", "reverse entails")] = "reverse entails"
_relation_composition[("reverse entails", "contradiction")] = "cover"
_relation_composition[("reverse entails", "cover")] = "cover"
_relation_composition[("contradiction", "entails")] = "cover"
_relation_composition[("contradiction", "reverse entails")] = "alternation"
_relation_composition[("contradiction", "contradiction")] = "equivalence"
_relation_composition[("contradiction", "cover")] = "reverse entails"
_relation_composition[("contradiction", "alternation")] = "entails"
_relation_composition[("alternation", "reverse entails")] = "alternation"
_relation_composition[("alternation", "contradiction")] = "entails"
_relation_composition[("alternation", "cover")] = "entails"
_relation_composition[("cover", "entails")] = "cover"
_relation_composition[("cover", "contradiction")] = "reverse entails"
_relation_composition[("cover", "alternation")] = "reverse entails"


def _strong_composition(sig1, sig2, r1, r2):
    c1 = _relation_composition[(sig1[r1], sig2[r2])]
    c2 = _relation_composition[(sig2[r2], sig1[r1])]
    if c1 == "independence":
        return c2
    return c1


# --- Negation / empty-string projectivity signatures ---
NEGATION_SIGNATURE = {
    "equivalence": "equivalence", "entails": "reverse entails",
    "reverse entails": "entails", "contradiction": "contradiction",
    "cover": "alternation", "alternation": "cover", "independence": "independence",
}
EMPTYSTRING_SIGNATURE = {r: r for r in RELATIONS}
_COMPOSE_WITH_CONTRADICTION = {r: _relation_composition[(r, "contradiction")] for r in RELATIONS}
SYMMETRIC_RELATION = {
    "equivalence": "equivalence", "entails": "reverse entails",
    "reverse entails": "entails", "contradiction": "contradiction",
    "cover": "cover", "alternation": "alternation", "independence": "independence",
}


def _compose_signatures(f, g):
    return {r: g[f[r]] for r in f}


# --- Determiner projectivity signatures: (det1, det2) -> {(rel_arg1, rel_arg2): rel_out} ---
_determiner_signatures: Dict[Tuple[str, str], Dict[Tuple[str, str], str]] = {}

_sig_some_1 = {"equivalence": "equivalence", "entails": "entails",
               "reverse entails": "reverse entails", "independence": "independence"}
_sig_some_2 = {"equivalence": "equivalence", "entails": "entails",
               "reverse entails": "reverse entails", "contradiction": "cover",
               "cover": "cover", "alternation": "independence", "independence": "independence"}
_determiner_signatures[("some", "some")] = (_sig_some_1, _sig_some_2)

_sig_every_1 = {"equivalence": "equivalence", "entails": "reverse entails",
                "reverse entails": "entails", "independence": "independence"}
_sig_every_2 = {"equivalence": "equivalence", "entails": "entails",
                "reverse entails": "reverse entails", "contradiction": "alternation",
                "cover": "independence", "alternation": "alternation", "independence": "independence"}
_determiner_signatures[("every", "every")] = (_sig_every_1, _sig_every_2)

for _key in [("some", "some"), ("every", "every")]:
    _s1, _s2 = _determiner_signatures[_key]
    _new = {}
    for _k1 in _s1:
        for _k2 in _s2:
            _new[(_k1, _k2)] = _strong_composition(_s1, _s2, _k1, _k2)
    _determiner_signatures[_key] = _new

_new_some_every = {}
for r1 in ["equivalence", "entails", "reverse entails", "independence"]:
    for r2 in RELATIONS:
        if (r2 in ("equivalence", "reverse entails")) and r1 != "independence":
            _new_some_every[(r1, r2)] = "reverse entails"
        else:
            _new_some_every[(r1, r2)] = "independence"
_new_some_every[("entails", "contradiction")] = "alternation"
_new_some_every[("entails", "alternation")] = "alternation"
_new_some_every[("equivalence", "alternation")] = "alternation"
_new_some_every[("equivalence", "contradiction")] = "contradiction"
_new_some_every[("equivalence", "cover")] = "cover"
_new_some_every[("reverse entails", "cover")] = "cover"
_new_some_every[("reverse entails", "contradiction")] = "cover"
_determiner_signatures[("some", "every")] = _new_some_every

_new_every_some = {}
for _key2, _val in _determiner_signatures[("some", "every")].items():
    _new_every_some[(SYMMETRIC_RELATION[_key2[0]], SYMMETRIC_RELATION[_key2[1]])] = SYMMETRIC_RELATION[_val]
_determiner_signatures[("every", "some")] = _new_every_some


def determiner_merge(det1: str, det2: str) -> Dict[Tuple[str, str], str]:
    """`det1`/`det2` in {"some", "every"} (negation is handled separately;
    "no" = negated "some", "not every" = negated "every"). Returns the
    projectivity signature (a dict mapping (rel_NP, rel_VP) -> rel_out)."""
    return _determiner_signatures[(det1, det2)]


def negation_merge(neg1: bool, neg2: bool) -> Dict[str, str]:
    if neg1 == neg2 and not neg2:
        return EMPTYSTRING_SIGNATURE
    if neg1 == neg2 and neg2:
        return NEGATION_SIGNATURE
    if not neg1:
        return _COMPOSE_WITH_CONTRADICTION
    return _compose_signatures(NEGATION_SIGNATURE, _COMPOSE_WITH_CONTRADICTION)


def standard_lexical_merge(x: str, y: str) -> str:
    """Relation between two aligned open-class lexical items (nouns,
    adjectives, verbs, adverbs). "" denotes an absent modifier."""
    if x == y:
        return "equivalence"
    if x == "":
        return "reverse entails"
    if y == "":
        return "entails"
    return "independence"


def standard_phrase(mod_relation: str, head_relation: str) -> str:
    """Combine a modifier relation (adjective/adverb) with a head relation
    (noun/verb) into a phrase relation, under intersective composition
    [Mod Head] = {x : Mod(x) & Head(x)}. Since MQNLI's nouns/verbs are
    drawn from disjoint, mutually "unrelated" vocabularies (Appendix A.1),
    the *head* relation dominates: if the heads match exactly
    (equivalence), the phrase relation is determined by the modifier
    (which may itself be empty on one side, giving reverse-entails /
    entails); if the heads differ, the phrase relation collapses to
    independence regardless of the modifier."""
    if head_relation == "equivalence":
        return mod_relation
    return "independence"


def determiner_phrase(signature: Dict[Tuple[str, str], str], np_relation: str, vp_relation: str) -> str:
    return signature[(np_relation, vp_relation)]


def negation_phrase(signature: Dict[str, str], relation: str) -> str:
    return signature[relation]


def relation_to_label(relation: str) -> str:
    """Collapse MacCartney's 7 relations to the 3 NLI classes used by
    MQNLI (see Appendix H.1 / Fig. 2b)."""
    if relation in ("cover", "independence", "reverse entails"):
        return "neutral"
    if relation in ("entails", "equivalence"):
        return "entailment"
    return "contradiction"  # alternation, contradiction
