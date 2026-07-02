import json
import os
from collections import Counter

PAD, UNK, CLS, SEP = "[PAD]", "[UNK]", "[CLS]", "[SEP]"


class WordVocab:
    def __init__(self, tokens_iterable=None):
        self.tok2id = {}
        self.id2tok = []
        for special in (PAD, UNK, CLS, SEP):
            self._add(special)
        if tokens_iterable:
            for tok in tokens_iterable:
                self._add(tok)

    def _add(self, tok):
        if tok not in self.tok2id:
            self.tok2id[tok] = len(self.id2tok)
            self.id2tok.append(tok)
        return self.tok2id[tok]

    def __len__(self):
        return len(self.id2tok)

    def encode(self, tokens):
        return [self.tok2id.get(t, self.tok2id[UNK]) for t in tokens]

    @classmethod
    def build_from_jsonl(cls, path):
        counter = Counter()
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                counter.update(rec["premise_tokens"])
                counter.update(rec["hypothesis_tokens"])
        vocab = cls()
        for tok, _ in counter.most_common():
            vocab._add(tok)
        return vocab

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.id2tok, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            id2tok = json.load(f)
        v = cls.__new__(cls)
        v.id2tok = id2tok
        v.tok2id = {t: i for i, t in enumerate(id2tok)}
        return v


def encode_example(vocab: WordVocab, premise_tokens, hypothesis_tokens):
    """[CLS] premise [SEP] hypothesis [SEP] -- single sequence, matching
    Appendix B ("we concatenate the premise and hypothesis into one string
    with special separator tokens")."""
    tokens = [CLS] + premise_tokens + [SEP] + hypothesis_tokens + [SEP]
    return vocab.encode(tokens), tokens
