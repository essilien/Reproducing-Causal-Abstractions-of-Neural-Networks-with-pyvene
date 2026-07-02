"""
Train a BiLSTM (default; no network access needed) or BERT NLI model on
the generated MQNLI-style dataset, with the auxiliary subphrase-relation
loss from Appendix B.2 ("Dataset Augmentation with Labeled Subphrases"),
which the paper found essential (dev accuracy 88.25% -> 55.42% without
it).

Usage:
    python train.py --data_dir datasets/small --model_type bilstm \
        --epochs 5 --out_dir checkpoints/bilstm_run1
"""
import argparse
import json
import os
import random
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from data.tokenizer import WordVocab, encode_example, PAD
from models.bilstm_nli import BiLSTMConfig, BiLSTMForNLI

MAIN_LABELS = ["entailment", "neutral", "contradiction"]
MAIN_LABEL2ID = {l: i for i, l in enumerate(MAIN_LABELS)}


def load_jsonl(path):
    recs = []
    with open(path) as f:
        for line in f:
            recs.append(json.loads(line))
    return recs


def build_aux_vocab(train_recs):
    labels = sorted({r["label"] for r in train_recs if r.get("is_augmented")})
    return {l: i for i, l in enumerate(labels)}


def pad_batch(id_lists, pad_id):
    max_len = max(len(x) for x in id_lists)
    out = torch.full((len(id_lists), max_len), pad_id, dtype=torch.long)
    mask = torch.zeros((len(id_lists), max_len), dtype=torch.long)
    for i, ids in enumerate(id_lists):
        out[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        mask[i, :len(ids)] = 1
    return out, mask


def make_batches(recs, vocab, batch_size, rng, label2id, label_key="label"):
    idxs = list(range(len(recs)))
    rng.shuffle(idxs)
    for start in range(0, len(idxs), batch_size):
        batch = [recs[i] for i in idxs[start:start + batch_size]]
        id_lists = []
        labels = []
        for r in batch:
            ids, _ = encode_example(vocab, r["premise_tokens"], r["hypothesis_tokens"])
            id_lists.append(ids)
            labels.append(label2id[r[label_key]])
        input_ids, attn = pad_batch(id_lists, vocab.tok2id[PAD])
        yield input_ids, attn, torch.tensor(labels, dtype=torch.long)


@torch.no_grad()
def evaluate(model, recs, vocab, device, batch_size=64):
    model.eval()
    rng = random.Random(0)
    correct, total = 0, 0
    for input_ids, attn, labels in make_batches(recs, vocab, batch_size, rng, MAIN_LABEL2ID):
        input_ids, labels = input_ids.to(device), labels.to(device)
        logits = model(input_ids=input_ids)["logits"]
        pred = logits.argmax(-1)
        correct += (pred == labels).sum().item()
        total += len(labels)
    model.train()
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model_type", choices=["bilstm"], default="bilstm",
                     help="BERT training requires network access to download pretrained "
                          "weights; see models/bert_nli.py and README for the BERT path.")
    ap.add_argument("--emb_dim", type=int, default=128)
    ap.add_argument("--h_dim", type=int, default=128)
    ap.add_argument("--n_layer", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--aux_weight", type=float, default=0.5)
    ap.add_argument("--use_augmentation", action="store_true", default=True)
    ap.add_argument("--no_augmentation", dest="use_augmentation", action="store_false")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_recs_all = load_jsonl(os.path.join(args.data_dir, "train.jsonl"))
    dev_recs = load_jsonl(os.path.join(args.data_dir, "dev.jsonl"))
    test_recs = load_jsonl(os.path.join(args.data_dir, "test.jsonl"))

    main_recs = [r for r in train_recs_all if not r.get("is_augmented")]
    aux_recs = [r for r in train_recs_all if r.get("is_augmented")] if args.use_augmentation else []

    vocab = WordVocab.build_from_jsonl(os.path.join(args.data_dir, "train.jsonl"))
    vocab.save(os.path.join(args.out_dir, "vocab.json"))
    aux_label2id = build_aux_vocab(train_recs_all)
    with open(os.path.join(args.out_dir, "aux_labels.json"), "w") as f:
        json.dump(aux_label2id, f)

    config = BiLSTMConfig(vocab_size=len(vocab), emb_dim=args.emb_dim, h_dim=args.h_dim,
                           n_layer=args.n_layer, n_labels=len(MAIN_LABELS))
    model = BiLSTMForNLI(config).to(device)
    aux_classifier = nn.Linear(2 * args.h_dim, max(len(aux_label2id), 1)).to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(aux_classifier.parameters()), lr=args.lr
    )

    print(f"train(main)={len(main_recs)} train(aux)={len(aux_recs)} dev={len(dev_recs)} test={len(test_recs)} "
          f"vocab={len(vocab)} aux_labels={len(aux_label2id)}")

    aux_iter = None
    for epoch in range(args.epochs):
        model.train()
        total_loss, n_batches = 0.0, 0
        main_batches = list(make_batches(main_recs, vocab, args.batch_size, rng, MAIN_LABEL2ID))
        if aux_recs:
            aux_batches = list(make_batches(aux_recs, vocab, args.batch_size, rng, aux_label2id))
        else:
            aux_batches = []
        for i, (input_ids, attn, labels) in enumerate(main_batches):
            input_ids, labels = input_ids.to(device), labels.to(device)
            optimizer.zero_grad()
            pooled, _, _ = model.lstm(input_ids=input_ids)
            main_logits = model.classifier(pooled)
            loss = nn.functional.cross_entropy(main_logits, labels)

            if aux_batches:
                a_input_ids, a_attn, a_labels = aux_batches[i % len(aux_batches)]
                a_input_ids, a_labels = a_input_ids.to(device), a_labels.to(device)
                a_pooled, _, _ = model.lstm(input_ids=a_input_ids)
                aux_logits = aux_classifier(a_pooled)
                loss = loss + args.aux_weight * nn.functional.cross_entropy(aux_logits, a_labels)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        dev_acc = evaluate(model, dev_recs, vocab, device, batch_size=args.batch_size)
        print(f"epoch {epoch}: train_loss={total_loss / max(n_batches,1):.4f} dev_acc={dev_acc:.4f}")

    test_acc = evaluate(model, test_recs, vocab, device, batch_size=args.batch_size)
    print(f"final test_acc={test_acc:.4f}")

    torch.save(model.state_dict(), os.path.join(args.out_dir, "model.pt"))
    config.save_pretrained(args.out_dir)
    print(f"saved checkpoint to {args.out_dir}")


if __name__ == "__main__":
    main()
