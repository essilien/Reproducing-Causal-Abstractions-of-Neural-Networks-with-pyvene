"""Token-index spans, within one 12-token sentence
(`Sentence.tokens()`), for each causal-model node. Position indices are
local to a single sentence; `analysis/interchange_experiments.py` offsets
them into the full `[CLS] premise [SEP] hypothesis [SEP]` sequence."""

NODE_TOKEN_SLICE = {
    "QSubj": (0, 2), "AdjSubj": (2, 3), "NSubj": (3, 4),
    "Neg": (4, 6), "Adv": (6, 7), "V": (7, 8),
    "QObj": (8, 10), "AdjObj": (10, 11), "NObj": (11, 12),
    # Phrasal nodes: span of all leaf tokens composed into them.
    "NPSubj": (2, 4), "NPObj": (10, 12), "VP": (6, 8),
    "QPObj": (8, 12),          # QObj + NPObj + VP -- but VP is a different
                                 # segment (6,8); we use the *object DP*
                                 # span (determiner..object noun) as the
                                 # primary candidate region, consistent
                                 # with the paper's Fig. 3 alignment (which
                                 # searches over the object-DP tokens).
    "NegP": (4, 6),             # aligned at the negation token span, per
                                 # Appendix 5.1 ("NegP: ... above NegP and NegH")
}

SENTENCE_LEN = 12
