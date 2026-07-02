"""
Fine-tune BERT on the generated MQNLI-style dataset. This is the
BERT-side counterpart to `train.py` (which only handles the BiLSTM,
since it needs no network access). Requires internet access to download
`bert-base-uncased` from the HuggingFace Hub -- run this on Kaggle/Colab
with internet enabled, not in a network-restricted sandbox.

Usage:
    python train_bert.py --data_dir datasets/scaled --out_dir checkpoints/bert_run1 \
        --epochs 3 --batch_size 32 --lr 3e-5
"""
import argparse
import json
import os
import random
import sys

import torch
import torch.nn as nn
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from data.bert_tokenize import encode_pair_for_bert, pad_encoded_batch
from models.bert_nli import BertForNLI

MAIN_LABELS = ["entailment", "neutral", "contradiction"]
MAIN_LABEL2ID = {l: i for i, l in enumerate(MAIN_LABELS)}


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def build_aux_vocab(train_recs):
    labels = sorted({r["label"] for r in train_recs if r.get("is_augmented")})
    return {l: i for i, l in enumerate(labels)}


def make_batches(recs, tokenizer, batch_size, rng, label2id, label_key="label"):
    idxs = list(range(len(recs)))
    rng.shuffle(idxs)
    for start in range(0, len(idxs), batch_size):
        batch = [recs[i] for i in idxs[start:start + batch_size]]
        encoded = []
        labels = []
        for r in batch:
            input_ids, token_type_ids, _, _ = encode_pair_for_bert(
                tokenizer, r["premise_tokens"], r["hypothesis_tokens"]
            )
            encoded.append((input_ids, token_type_ids))
            labels.append(label2id[r[label_key]])
        input_ids, token_type_ids, attention_mask = pad_encoded_batch(encoded, tokenizer.pad_token_id)
        yield input_ids, token_type_ids, attention_mask, torch.tensor(labels, dtype=torch.long)


@torch.no_grad()
def evaluate(model, recs, tokenizer, device, batch_size=32):
    model.eval()
    rng = random.Random(0)
    correct, total = 0, 0
    for input_ids, token_type_ids, attn, labels in make_batches(recs, tokenizer, batch_size, rng, MAIN_LABEL2ID):
        input_ids, token_type_ids, attn, labels = (
            input_ids.to(device), token_type_ids.to(device), attn.to(device), labels.to(device)
        )
        logits = model(input_ids=input_ids, attention_mask=attn, token_type_ids=token_type_ids)["logits"]
        pred = logits.argmax(-1)
        correct += (pred == labels).sum().item()
        total += len(labels)
    model.train()
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model_name", default="bert-base-uncased")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--aux_weight", type=float, default=0.3)
    ap.add_argument("--use_augmentation", action="store_true", default=False,
                     help="Off by default for BERT: the aux pass roughly doubles "
                          "step count and BERT reaches high accuracy on the "
                          "balanced/moderate-difficulty scaled-down dataset without "
                          "it. Turn on if dev accuracy plateaus low.")
    ap.add_argument("--max_aux_per_epoch", type=int, default=20000,
                     help="Cap on how many augmented rows to use per epoch (they "
                          "outnumber main rows ~14:1; using all of them makes each "
                          "epoch far more expensive for limited extra benefit).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.save_pretrained(args.out_dir)

    train_recs_all = load_jsonl(os.path.join(args.data_dir, "train.jsonl"))
    dev_recs = load_jsonl(os.path.join(args.data_dir, "dev.jsonl"))
    test_recs = load_jsonl(os.path.join(args.data_dir, "test.jsonl"))

    main_recs = [r for r in train_recs_all if not r.get("is_augmented")]
    aux_recs_all = [r for r in train_recs_all if r.get("is_augmented")] if args.use_augmentation else []

    aux_label2id = build_aux_vocab(train_recs_all)
    with open(os.path.join(args.out_dir, "aux_labels.json"), "w") as f:
        json.dump(aux_label2id, f)

    model = BertForNLI(args.model_name, n_labels=len(MAIN_LABELS)).to(device)
    aux_classifier = nn.Linear(model.config.hidden_size, max(len(aux_label2id), 1)).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(aux_classifier.parameters()), lr=args.lr
    )

    print(f"train(main)={len(main_recs)} train(aux available)={len(aux_recs_all)} "
          f"dev={len(dev_recs)} test={len(test_recs)}")

    for epoch in range(args.epochs):
        model.train()
        aux_recs = rng.sample(aux_recs_all, min(args.max_aux_per_epoch, len(aux_recs_all))) if aux_recs_all else []
        main_batches = list(make_batches(main_recs, tokenizer, args.batch_size, rng, MAIN_LABEL2ID))
        aux_batches = list(make_batches(aux_recs, tokenizer, args.batch_size, rng, aux_label2id)) if aux_recs else []

        total_loss, n_batches = 0.0, 0
        for i, (input_ids, token_type_ids, attn, labels) in enumerate(main_batches):
            input_ids, token_type_ids, attn, labels = (
                input_ids.to(device), token_type_ids.to(device), attn.to(device), labels.to(device)
            )
            optimizer.zero_grad()
            out = model.bert(input_ids=input_ids, attention_mask=attn, token_type_ids=token_type_ids)
            cls = out.last_hidden_state[:, 0, :]
            main_logits = model.classifier(model.dropout(cls))
            loss = nn.functional.cross_entropy(main_logits, labels)

            if aux_batches:
                a_ids, a_tt, a_attn, a_labels = aux_batches[i % len(aux_batches)]
                a_ids, a_tt, a_attn, a_labels = (
                    a_ids.to(device), a_tt.to(device), a_attn.to(device), a_labels.to(device)
                )
                a_out = model.bert(input_ids=a_ids, attention_mask=a_attn, token_type_ids=a_tt)
                a_cls = a_out.last_hidden_state[:, 0, :]
                aux_logits = aux_classifier(a_cls)
                loss = loss + args.aux_weight * nn.functional.cross_entropy(aux_logits, a_labels)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
            if n_batches % 200 == 0:
                print(f"  epoch {epoch} step {n_batches}/{len(main_batches)} "
                      f"running_loss={total_loss / n_batches:.4f}")

        dev_acc = evaluate(model, dev_recs, tokenizer, device, batch_size=args.batch_size)
        print(f"epoch {epoch}: train_loss={total_loss / max(n_batches,1):.4f} dev_acc={dev_acc:.4f}")
        torch.save(model.state_dict(), os.path.join(args.out_dir, f"model_epoch{epoch}.pt"))

    test_acc = evaluate(model, test_recs, tokenizer, device, batch_size=args.batch_size)
    print(f"final test_acc={test_acc:.4f}")
    torch.save(model.state_dict(), os.path.join(args.out_dir, "model.pt"))
    model.bert.save_pretrained(args.out_dir)  # HF format (config.json + weights), so
                                                # BertForNLI(checkpoint_dir) can rebuild
                                                # the architecture later without re-
                                                # downloading bert-base-uncased fresh;
                                                # model.pt above still holds the exact
                                                # fine-tuned state (incl. classifier head).
    print(f"saved checkpoint to {args.out_dir}")


if __name__ == "__main__":
    main()
