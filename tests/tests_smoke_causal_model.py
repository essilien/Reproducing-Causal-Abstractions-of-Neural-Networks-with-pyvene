import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from causal_model import Sentence, MQNLIExample, compute_label, intervene, LEAF_NODES, PHRASAL_NODES


def S(qd, adj, n, neg, adv, v, qo, adjo, no):
    return Sentence(
        subject_determiner=qd, subject_adjective=adj, subject_noun=n,
        negated=neg, adverb=adv, verb=v, verb_3ps=v + "s",
        object_determiner=qo, object_adjective=adjo, object_noun=no,
    )


# Figure 2b, example 1 (contradiction):
# P: "every baker eats no bread"   H: "no angry baker eats no bread"
ex1 = MQNLIExample(
    premise=S("every", "", "baker", False, "", "eat", "no", "", "bread"),
    hypothesis=S("no", "angry", "baker", False, "", "eat", "no", "", "bread"),
)
print("ex1 (expect contradiction):", compute_label(ex1))

# Figure 2b, example 2 (neutral):
# P: "every silly professor sells not every book"
# H: "every silly professor sells not every chair"
ex2 = MQNLIExample(
    premise=S("every", "silly", "professor", False, "", "sell", "not every", "", "book"),
    hypothesis=S("every", "silly", "professor", False, "", "sell", "not every", "", "chair"),
)
print("ex2 (expect neutral):", compute_label(ex2))

# Figure 2b, example 3 (entailment):
# P: "not every sad baker fairly admits not every odd idea"
# H: "some baker does not admit no idea"
ex3 = MQNLIExample(
    premise=S("not every", "sad", "baker", False, "fairly", "admit", "not every", "odd", "idea"),
    hypothesis=S("some", "", "baker", True, "", "admit", "no", "", "idea"),
)
print("ex3 (expect entailment):", compute_label(ex3))

print()
print("premise tokens:", ex1.premise.tokens(), "len=", len(ex1.premise.tokens()))
print("hyp     tokens:", ex1.hypothesis.tokens())

print()
print("--- interventions on ex1 using ex2's premise/hyp as source ---")
for node in LEAF_NODES + PHRASAL_NODES:
    print(f"  intervene(ex1, ex2, {node!r}) =", intervene(ex1, ex2, node))

print()
print("--- sanity: intervening on 'root'-feeding leaves with itself changes nothing ---")
for node in LEAF_NODES + PHRASAL_NODES:
    assert intervene(ex1, ex1, node) == compute_label(ex1), f"self-intervention changed label at {node}"
print("OK: self-intervention is always a no-op")
