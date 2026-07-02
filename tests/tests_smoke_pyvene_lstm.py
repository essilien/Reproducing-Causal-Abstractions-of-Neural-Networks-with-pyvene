import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
import pyvene as pv

from models.bilstm_nli import BiLSTMConfig, BiLSTMForNLI
from models.pyvene_registration import register_bilstm_nli

register_bilstm_nli()

vocab_size = 40
seq_len = 12
config = BiLSTMConfig(vocab_size=vocab_size, emb_dim=16, h_dim=8, n_layer=2, n_labels=3)
model = BiLSTMForNLI(config)
model.eval()

torch.manual_seed(0)
base_ids = torch.randint(0, vocab_size, (1, seq_len))
source_ids = torch.randint(0, vocab_size, (1, seq_len))

with torch.no_grad():
    base_out = model(input_ids=base_ids)["logits"]
print("base logits:", base_out)

# --- Part 1: pv.IntervenableModel, unit="t" -----------------------------
# This runs without crashing (the registration + is_stateless patch in
# pyvene_registration.py work), but see NOTES.md: in this pyvene version
# we found the requested timestep is *not* reliably honored for a custom
# (non-built-in-GRU) recurrent model, so this path is NOT used by
# analysis/interchange_experiments.py. We still smoke-test it here so a
# future pyvene upgrade that fixes this is easy to notice (the printed
# check below would start reporting True).
iv_config = pv.IntervenableConfig(
    {"layer": 1, "component": "fw_cell_output", "unit": "t", "intervention_type": pv.VanillaIntervention},
    model=model,
)
iv_model = pv.IntervenableModel(iv_config, model=model)
with torch.no_grad():
    _, intervened = iv_model(
        base={"input_ids": base_ids},
        sources=[{"input_ids": source_ids}],
        unit_locations={"sources->base": (3, 3)},
    )
print("pv.IntervenableModel intervened logits:", intervened["logits"])
with torch.no_grad():
    _, self_intervened = iv_model(
        base={"input_ids": base_ids},
        sources=[{"input_ids": base_ids}],
        unit_locations={"sources->base": (3, 3)},
    )
pv_self_ok = torch.allclose(self_intervened["logits"], base_out, atol=1e-5)
print(f"[known limitation, see NOTES.md] pv.IntervenableModel self-intervention "
      f"is an exact no-op: {pv_self_ok} (expected False in this pyvene version)")

# --- Part 2: hand-rolled forward_with_intervention -----------------------
# This is what analysis/interchange_experiments.py actually uses for the
# BiLSTM. Must be an exact no-op under self-intervention.
fw_states_self, _ = model.lstm.forward_capture(base_ids)
self_logits = model.logits_with_intervention(base_ids, "fw", 1, 3, fw_states_self[3][1])
assert torch.allclose(self_logits, base_out, atol=1e-7), "hand-rolled self-intervention should be an EXACT no-op!"
print("OK: hand-rolled forward_with_intervention self-intervention is an exact no-op.")

fw_states_src, _ = model.lstm.forward_capture(source_ids)
real_logits = model.logits_with_intervention(base_ids, "fw", 1, 3, fw_states_src[3][1])
print("hand-rolled real-intervention logits:", real_logits)
print("OK: hand-rolled intervention pipeline is the one used by analysis/interchange_experiments.py.")
