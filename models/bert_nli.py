"""
BERT-based NLI classifier, matching Appendix B of the paper: fine-tune
`bert-base-uncased`, apply one linear layer to the final layer's [CLS]
representation to get 3-way logits.

NOTE: instantiating this model requires downloading `bert-base-uncased`
from the HuggingFace Hub, which needs outbound network access to
huggingface.co. If you're running this in a network-restricted sandbox,
use `models/bilstm_nli.py` instead -- the causal-abstraction analysis
code in `analysis/` is architecture-agnostic and works identically for
either model once it is registered with pyvene (see
`pyvene_registration.py`).
"""
import torch
import torch.nn as nn
from transformers import BertModel, BertConfig


class BertForNLI(nn.Module):
    def __init__(self, model_name_or_path="bert-base-uncased", n_labels=3, dropout=0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name_or_path, add_pooling_layer=False)
        self.config = self.bert.config
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, n_labels)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls = out.last_hidden_state[:, 0, :]  # [CLS]
        logits = self.classifier(self.dropout(cls))
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        return {"loss": loss, "logits": logits}

    @classmethod
    def from_scratch(cls, vocab_size, n_labels=3, hidden_size=192, num_layers=4, num_heads=4):
        """A randomly-initialized, small BERT (useful for smoke-testing the
        pyvene interchange-intervention pipeline without any network
        access / pretrained-weight download)."""
        config = BertConfig(
            vocab_size=vocab_size, hidden_size=hidden_size, num_hidden_layers=num_layers,
            num_attention_heads=num_heads, intermediate_size=hidden_size * 4,
            max_position_embeddings=64, type_vocab_size=2,
        )
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj.bert = BertModel(config, add_pooling_layer=False)
        obj.config = config
        obj.dropout = nn.Dropout(0.1)
        obj.classifier = nn.Linear(config.hidden_size, n_labels)
        return obj
