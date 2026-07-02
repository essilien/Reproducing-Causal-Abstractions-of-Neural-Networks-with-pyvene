"""
Our causal-model nodes are defined over fixed *word* positions (see
`causal_model/node_spans.py`: 12 words per sentence). BERT's tokenizer
splits some of those words into multiple WordPiece subwords, so a node's
"token span" is not fixed -- it has to be computed per-example from the
tokenizer's output. This module does that bookkeeping once so the rest of
the analysis code can just ask "which subword positions does node N sit
at, in this specific example's encoded sequence".
"""
from typing import List, Tuple


def encode_pair_for_bert(tokenizer, premise_words: List[str], hypothesis_words: List[str]):
    """[CLS] premise-subwords [SEP] hypothesis-subwords [SEP], plus, for
    each of the 12 premise words and 12 hypothesis words, the (start, end)
    range of subword positions it landed on in the final sequence."""
    def encode_words(words):
        return [tokenizer.encode(w, add_special_tokens=False) for w in words]

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    input_ids = [cls_id]
    p_spans: List[Tuple[int, int]] = []
    for ids in encode_words(premise_words):
        start = len(input_ids)
        input_ids.extend(ids or [tokenizer.unk_token_id])
        p_spans.append((start, len(input_ids)))
    input_ids.append(sep_id)
    h_spans: List[Tuple[int, int]] = []
    for ids in encode_words(hypothesis_words):
        start = len(input_ids)
        input_ids.extend(ids or [tokenizer.unk_token_id])
        h_spans.append((start, len(input_ids)))
    input_ids.append(sep_id)
    token_type_ids = [0] * (p_spans[-1][1] + 1) + [1] * (len(input_ids) - p_spans[-1][1] - 1)
    return input_ids, token_type_ids, p_spans, h_spans


def node_subword_positions(node_word_slice, word_spans: List[Tuple[int, int]]) -> List[int]:
    """`node_word_slice` = (start_word, end_word) from NODE_TOKEN_SLICE.
    Returns every subword position covered by that word range, for one
    side (premise or hypothesis) of one specific encoded example."""
    start_word, end_word = node_word_slice
    positions = []
    for w in range(start_word, end_word):
        s, e = word_spans[w]
        positions.extend(range(s, e))
    return positions


def pad_encoded_batch(encoded_list, pad_id):
    max_len = max(len(ids) for ids, _ in encoded_list)
    import torch
    input_ids = torch.full((len(encoded_list), max_len), pad_id, dtype=torch.long)
    token_type_ids = torch.zeros((len(encoded_list), max_len), dtype=torch.long)
    attention_mask = torch.zeros((len(encoded_list), max_len), dtype=torch.long)
    for i, (ids, ttids) in enumerate(encoded_list):
        input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        token_type_ids[i, :len(ttids)] = torch.tensor(ttids, dtype=torch.long)
        attention_mask[i, :len(ids)] = 1
    return input_ids, token_type_ids, attention_mask
