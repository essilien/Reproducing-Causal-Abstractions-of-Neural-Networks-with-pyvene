"""
pyvene does not ship built-in support for a plain `transformers.BertModel`
or for our custom `BiLSTMEncoder` (see pyvene's own
`models/intervenable_modelcard.py`: only GPT-2/Llama/Gemma/... family
architectures are pre-registered). pyvene is explicitly designed to be
*extensible* to new architectures by adding entries to two module-level
dicts, `type_to_module_mapping` and `type_to_dimension_mapping`, that map
a python `type` to a small dict describing where each named "component"
lives (as a dotted/`%s`-templated attribute path) and how big it is. This
module adds the two entries we need, following exactly the pattern
pyvene uses internally for e.g. GPT-2 (`models/gpt2/modelings_intervenable_gpt2.py`)
and its own GRU model (`models/gru/modelings_intervenable_gru.py`).

Call `register_bert_nli()` / `register_bilstm_nli()` once, before
constructing any `pv.IntervenableModel` around our models.
"""
import pyvene as pv
from pyvene.models.constants import CONST_INPUT_HOOK, CONST_OUTPUT_HOOK
from pyvene.models.intervenable_modelcard import type_to_module_mapping, type_to_dimension_mapping
import pyvene.models.modeling_utils as _pv_modeling_utils

from transformers import BertModel
from models.bert_nli import BertForNLI
from models.bilstm_nli import BiLSTMEncoder, BiLSTMForNLI

# pyvene's `is_stateless(model)` (used to decide whether a hooked module
# is allowed to fire more than once per forward pass, which recurrent
# models inherently do) hardcodes a check for its own built-in GRU
# classes (`models/modeling_utils.py::is_gru`). Our BiLSTM is exactly the
# same kind of "stateful" (multi-invocation) model, so we patch that
# check to also recognize it. This must run before any
# `pv.IntervenableModel(...)` is constructed around a BiLSTMForNLI.
_orig_is_gru = _pv_modeling_utils.is_gru


def _is_gru_or_bilstm(model):
    if type(model) in (BiLSTMForNLI, BiLSTMEncoder):
        return True
    return _orig_is_gru(model)


_pv_modeling_utils.is_gru = _is_gru_or_bilstm


def register_bert_nli():
    """Registers component names for a `BertForNLI` wrapper (attribute
    `.bert` holds the underlying `transformers.BertModel`). Components are
    indexed by `%s` = layer index (0-based) and, like pyvene's other
    Transformer registrations, operate on the standard `(batch, seq, hidden)`
    tensor at that layer -- `unit_locations` then selects the token
    position(s) within that tensor (pyvene's default "h.pos" unit)."""
    mapping = {
        "block_input": ("bert.encoder.layer[%s]", CONST_INPUT_HOOK),
        "block_output": ("bert.encoder.layer[%s]", CONST_OUTPUT_HOOK),
        "attention_output": ("bert.encoder.layer[%s].attention.output", CONST_OUTPUT_HOOK),
        "mlp_output": ("bert.encoder.layer[%s].output", CONST_OUTPUT_HOOK),
        "embedding_output": ("bert.embeddings", CONST_OUTPUT_HOOK),
    }
    dims = {k: ("hidden_size",) for k in mapping}
    type_to_module_mapping[BertForNLI] = mapping
    type_to_dimension_mapping[BertForNLI] = dims


def register_bilstm_nli():
    """Registers component names for our `BiLSTMForNLI` wrapper (attribute
    `.lstm` holds the `BiLSTMEncoder`, which owns `fw_cells` / `bw_cells`
    -- ModuleLists of `nn.LSTMCell`, one per layer, called once per
    timestep). `%s` = layer index; the time dimension is handled by
    pyvene's `unit="t"` mechanism (see pyvene's own GRU registration),
    which counts *invocations* of the hooked module within one forward
    pass. IMPORTANT: because `bw_cells` is iterated in *reverse* temporal
    order (see `BiLSTMEncoder._run_direction`), invocation index `i` for
    the backward direction corresponds to raw token position
    `seq_len - 1 - i`, not position `i`. `analysis/interchange_experiments.py`
    accounts for this when building `unit_locations`."""
    mapping = {
        "fw_cell_output": ("lstm.fw_cells[%s]", CONST_OUTPUT_HOOK),
        "bw_cell_output": ("lstm.bw_cells[%s]", CONST_OUTPUT_HOOK),
        "fw_cell_input": ("lstm.fw_cells[%s]", CONST_INPUT_HOOK),
        "bw_cell_input": ("lstm.bw_cells[%s]", CONST_INPUT_HOOK),
    }
    dims = {k: ("h_dim",) for k in mapping}
    type_to_module_mapping[BiLSTMForNLI] = mapping
    type_to_dimension_mapping[BiLSTMForNLI] = dims
