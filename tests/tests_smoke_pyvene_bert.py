import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
import pyvene as pv

from models.bert_nli import BertForNLI
from models.pyvene_registration import register_bert_nli

register_bert_nli()

vocab_size = 40
seq_len = 12
model = BertForNLI.from_scratch(vocab_size=vocab_size, n_labels=3, hidden_size=32, num_layers=3, num_heads=2)
model.eval()

torch.manual_seed(0)
base_ids = torch.randint(0, vocab_size, (1, seq_len))
source_ids = torch.randint(0, vocab_size, (1, seq_len))

with torch.no_grad():
    base_out = model(input_ids=base_ids)["logits"]
print("base logits:", base_out)

iv_config = pv.IntervenableConfig(
    {"layer": 1, "component": "block_output", "intervention_type": pv.VanillaIntervention},
    model=model,
)
iv_model = pv.IntervenableModel(iv_config, model=model)

with torch.no_grad():
    _, intervened = iv_model(
        base={"input_ids": base_ids},
        sources=[{"input_ids": source_ids}],
        unit_locations={"sources->base": 3},
    )
print("intervened logits:", intervened["logits"])

with torch.no_grad():
    _, self_intervened = iv_model(
        base={"input_ids": base_ids},
        sources=[{"input_ids": base_ids}],
        unit_locations={"sources->base": 3},
    )
assert torch.allclose(self_intervened["logits"], base_out, atol=1e-5)
print("OK: BERT registration + interchange intervention works; self-intervention is a no-op.")
