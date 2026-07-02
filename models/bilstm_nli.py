"""
A BiLSTM NLI classifier built out of explicit per-timestep `nn.LSTMCell`
calls (rather than `nn.LSTM`), so that pyvene can hook individual
(layer, direction, timestep) cell outputs -- exactly mirroring the design
of pyvene's own built-in `GRUModel` (pyvene/models/gru/modelings_gru.py),
which is the officially-supported way to make a *recurrent* model
intervenable in pyvene (plain `nn.LSTM` only exposes one call per whole
sequence, so there is nothing for pyvene to hook at a given timestep).

Architecture follows Appendix B.1 of Geiger et al. 2021: forward and
backward LSTM directions, hidden states at the last token position (which,
after our ε-padded tokenization, always ends with the object noun -- the
paper concatenates hidden states above "the last [SEP] and the [CLS]",
i.e. the two sequence-boundary positions; we concatenate the last
timestep of both directions) are concatenated and passed through 3 linear
layers down to a 3-way softmax.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel


class BiLSTMConfig(PretrainedConfig):
    model_type = "bilstm_nli"

    def __init__(
        self,
        vocab_size=1000,
        emb_dim=256,
        h_dim=128,
        n_layer=2,
        n_labels=3,
        pdrop=0.1,
        pad_token_id=0,
        initializer_range=0.02,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.h_dim = h_dim
        self.n_layer = n_layer
        self.n_labels = n_labels
        self.pdrop = pdrop
        self.pad_token_id = pad_token_id
        self.initializer_range = initializer_range
        super().__init__(**kwargs)


class BiLSTMPreTrainedModel(PreTrainedModel):
    config_class = BiLSTMConfig

    def _init_weights(self, module):
        # Only re-initialize embeddings with the configured (small) std;
        # leave nn.Linear / nn.LSTMCell at PyTorch's own default init
        # (Kaiming-uniform, scaled by fan-in). Forcing a BERT-style
        # initializer_range=0.02 onto a from-scratch, non-residual 3-layer
        # MLP classifier head causes severe signal attenuation (each layer
        # shrinks activations further, since 0.02 is tuned for much wider
        # Transformer layers) and effectively vanishing gradients.
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class _HookableLSTMCell(nn.Module):
    """Wraps `nn.LSTMCell` so the module's forward() returns a single
    tensor (the new hidden state `h`) rather than the `(h, c)` tuple
    `nn.LSTMCell` normally returns. pyvene's generic hooking logic expects
    a module's output to be directly interveneable as a tensor; returning
    a tuple would require extra `subspaces`/slicing configuration we don't
    need. The cell state `c` is kept as a plain (non-hooked) side-channel
    attribute, so an interchange intervention on this module's output only
    ever swaps the hidden state -- the piece of the LSTM's state that
    actually determines everything computed downstream of this timestep,
    which is the analogue of swapping a Transformer's hidden-state vector."""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.cell = nn.LSTMCell(input_size, hidden_size)
        self.last_c = None

    def forward(self, x, h, c):
        h_new, c_new = self.cell(x, (h, c))
        self.last_c = c_new
        return h_new


class BiLSTMEncoder(BiLSTMPreTrainedModel):
    """Holds `fw_cells` / `bw_cells`: ModuleList[_HookableLSTMCell] per
    layer per direction. These are the pyvene-hookable intervention sites
    (see pyvene_registration.py, components "fw_cell_output" /
    "bw_cell_output")."""

    def __init__(self, config: BiLSTMConfig):
        super().__init__(config)
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.emb_dim, padding_idx=config.pad_token_id)
        self.dropout = nn.Dropout(config.pdrop)
        self.fw_cells = nn.ModuleList([
            _HookableLSTMCell(config.emb_dim if i == 0 else config.h_dim, config.h_dim)
            for i in range(config.n_layer)
        ])
        self.bw_cells = nn.ModuleList([
            _HookableLSTMCell(config.emb_dim if i == 0 else config.h_dim, config.h_dim)
            for i in range(config.n_layer)
        ])
        self.post_init()

    def _run_direction(self, cells, embeds, order, intervention=None):
        """`order`: a list of timestep indices to iterate the sequence in
        (0..T-1 for forward, T-1..0 for backward). Returns a list, indexed
        by *original* position, of the top layer's hidden state emitted
        when that position was consumed.

        `intervention`, if given, is a dict {"layer": L, "t": T,
        "value": tensor} -- after computing layer L's hidden state at
        timestep T normally, it is *overwritten* with `value` before being
        used for anything downstream (next layer / next timestep / pooled
        output). This is a direct (non-pyvene) implementation of an
        interchange intervention; see the module docstring in
        `analysis/interchange_experiments.py` for why we hand-roll this
        instead of using `pv.IntervenableModel`'s `unit="t"` path for this
        architecture."""
        batch_size, seq_len, _ = embeds.shape
        h = [torch.zeros(batch_size, self.config.h_dim, device=embeds.device) for _ in cells]
        c = [torch.zeros(batch_size, self.config.h_dim, device=embeds.device) for _ in cells]
        outputs_by_pos = [None] * seq_len
        for t in order:
            x = embeds[:, t, :]
            for layer, cell in enumerate(cells):
                h[layer] = cell(x, h[layer], c[layer])
                c[layer] = cell.last_c
                if intervention is not None and intervention["layer"] == layer and intervention["t"] == t:
                    h[layer] = intervention["value"]
                x = h[layer]
            outputs_by_pos[t] = h[-1]
        return outputs_by_pos, h[-1]

    def forward_capture(self, input_ids):
        """Plain forward pass that also returns every (direction, layer,
        t) hidden state, for use as intervention *sources*."""
        inputs_embeds = self.dropout(self.wte(input_ids))
        seq_len = inputs_embeds.shape[1]
        fw_states, bw_states = [], []
        h_fw = [torch.zeros(inputs_embeds.shape[0], self.config.h_dim) for _ in self.fw_cells]
        c_fw = [torch.zeros(inputs_embeds.shape[0], self.config.h_dim) for _ in self.fw_cells]
        for t in range(seq_len):
            x = inputs_embeds[:, t, :]
            row = []
            for layer, cell in enumerate(self.fw_cells):
                h_fw[layer] = cell(x, h_fw[layer], c_fw[layer]); c_fw[layer] = cell.last_c
                x = h_fw[layer]
                row.append(h_fw[layer])
            fw_states.append(row)
        h_bw = [torch.zeros(inputs_embeds.shape[0], self.config.h_dim) for _ in self.bw_cells]
        c_bw = [torch.zeros(inputs_embeds.shape[0], self.config.h_dim) for _ in self.bw_cells]
        bw_states_by_pos = [None] * seq_len
        for t in reversed(range(seq_len)):
            x = inputs_embeds[:, t, :]
            row = []
            for layer, cell in enumerate(self.bw_cells):
                h_bw[layer] = cell(x, h_bw[layer], c_bw[layer]); c_bw[layer] = cell.last_c
                x = h_bw[layer]
                row.append(h_bw[layer])
            bw_states_by_pos[t] = row
        # fw_states[t][layer], bw_states_by_pos[t][layer]
        return fw_states, bw_states_by_pos

    def forward_with_intervention(self, input_ids, direction, layer, t, source_value):
        """Run a full forward pass identical to `forward()`, except that
        the hidden state at (`direction`, `layer`, timestep `t`) is
        overwritten with `source_value` (a (batch, h_dim) tensor, usually
        taken from `forward_capture` on a different input) before it
        propagates further. Returns the same `(pooled, fw_by_pos,
        bw_by_pos)` tuple as `forward()`."""
        inputs_embeds = self.dropout(self.wte(input_ids))
        seq_len = inputs_embeds.shape[1]
        intervention = {"layer": layer, "t": t, "value": source_value}
        if direction == "fw":
            fw_by_pos, fw_last = self._run_direction(self.fw_cells, inputs_embeds, range(seq_len), intervention)
            bw_by_pos, bw_last = self._run_direction(self.bw_cells, inputs_embeds, reversed(range(seq_len)))
        else:
            fw_by_pos, fw_last = self._run_direction(self.fw_cells, inputs_embeds, range(seq_len))
            bw_by_pos, bw_last = self._run_direction(self.bw_cells, inputs_embeds, reversed(range(seq_len)), intervention)
        pooled = torch.cat([fw_last, bw_last], dim=-1)
        return pooled, fw_by_pos, bw_by_pos

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None):
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        inputs_embeds = self.dropout(inputs_embeds)
        seq_len = inputs_embeds.shape[1]
        fw_by_pos, fw_last = self._run_direction(self.fw_cells, inputs_embeds, range(seq_len))
        bw_by_pos, bw_last = self._run_direction(self.bw_cells, inputs_embeds, reversed(range(seq_len)))
        # sequence representation: concat final fw state (after last real
        # token) and final bw state (after first real token) -- analogous
        # to concatenating hidden states "above [CLS]" and "above [SEP]".
        pooled = torch.cat([fw_last, bw_last], dim=-1)
        return pooled, fw_by_pos, bw_by_pos


class BiLSTMForNLI(BiLSTMPreTrainedModel):
    def __init__(self, config: BiLSTMConfig):
        super().__init__(config)
        self.lstm = BiLSTMEncoder(config)
        self.classifier = nn.Sequential(
            nn.Linear(2 * config.h_dim, config.h_dim),
            nn.ReLU(),
            nn.Linear(config.h_dim, config.h_dim),
            nn.ReLU(),
            nn.Linear(config.h_dim, config.n_labels),
        )
        self.post_init()

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, labels=None):
        pooled, _, _ = self.lstm(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        logits = self.classifier(pooled)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        return {"loss": loss, "logits": logits}

    @torch.no_grad()
    def logits_with_intervention(self, base_ids, direction, layer, t, source_value):
        pooled, _, _ = self.lstm.forward_with_intervention(base_ids, direction, layer, t, source_value)
        return self.classifier(pooled)
