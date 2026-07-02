# Engineering notes

## The pyvene `unit="t"` limitation for custom recurrent models

pyvene ships one built-in recurrent architecture, `GRUModel`
(`pyvene/models/gru/`), intervened on via `unit="t"` (pyvene counts
*invocations* of the hooked module within a forward pass to figure out
"which timestep" you mean, since a recurrent cell is called once per
timestep rather than once per forward pass like a Transformer layer).

We built `BiLSTMEncoder` (`models/bilstm_nli.py`) in exactly this style
(`nn.LSTMCell`-based, one call per timestep, registered via
`type_to_module_mapping`/`type_to_dimension_mapping` -- see
`models/pyvene_registration.py`), and pyvene's `is_stateless()` check
needed a monkeypatch to recognize our model as "stateful" (multi-
invocation) the same way it recognizes its own `GRUModel` -- that part
works fine and is done in `pyvene_registration.py`.

However, empirically (see the investigation embedded in the git history
of this file / rerun `tests_smoke_pyvene_lstm.py`), we found that
requesting a specific timestep via `unit_locations={"sources->base": (t,
t)}` does **not** reliably intervene at timestep `t` for our
custom-registered model in this pyvene version: sweeping `t` over
`[0,1,2,3]` against a fixed model, the *first* diverging call (traced
directly via a monkeypatched `forward`) was always the module's *2nd*
invocation, regardless of the requested `t`. A true self-intervention
(source == base) is also *not* an exact no-op through this path (small
but real numerical drift, not just floating-point noise) -- which is the
simplest possible correctness check an interchange intervention must
pass, and it fails it here.

We did not track this down to a specific line in pyvene's source within
the time available (candidates: the getter/setter invocation-counter not
resetting correctly between the source-gather sub-pass and the
base-intervene sub-pass for anything other than pyvene's own hardcoded
`GRUModel` type; or `unit_locations` parsing for `"t"` having an
undocumented convention we didn't match). It's plausible this is simply
undertested upstream, since pyvene's own GRU example in the paper only
ever demonstrates batch size 1 with a single fixed timestep.

**What we did instead**: `BiLSTMEncoder.forward_with_intervention` /
`forward_capture` (`models/bilstm_nli.py`) directly implement "run the
encoder, but at (direction, layer, timestep) substitute in a hidden state
captured from a different input" as a plain Python loop -- no pyvene
hooking involved. This is the exact same *operation* an interchange
intervention performs, just without going through
`pv.IntervenableModel`. We verified it directly:

- self-intervention (source state == base's own state at that point) is
  an *exact* (bit-level, `atol=1e-7`) no-op,
- intervening at different `t` values produces genuinely different
  outputs (position sensitivity),
- it works at batch size > 1.

`analysis/interchange_experiments.py` uses this path for the BiLSTM and
`pv.IntervenableModel` (the standard, thoroughly-verified "pos"-unit
Transformer path) for BERT.

**If you want to fix the root cause**: instrument
`pyvene/models/intervenable_base.py`'s getter/setter hook installation
and the invocation counter it maintains per intervention (search for
where `unit in {"t"}` is handled during hook registration, not just
during `gather_neurons`/`scatter_neurons`), and check whether it's keyed
correctly per-module-instance across the two sub-passes pyvene runs
(source-gather, then base-intervene) for an externally-registered type.
If you fix it, `bilstm_node_clique_size` in
`analysis/interchange_experiments.py` can be simplified back to a
`pv.IntervenableModel` call, and importantly could then be **batched**
(right now it's one example at a time, which is the slow part of scaling
up -- see README "Scaling up", point 4).

## Why `standard_phrase`'s argument order matters

`causal_model/natural_logic.py::standard_phrase(mod_relation,
head_relation)` returns `mod_relation` when `head_relation ==
"equivalence"`, and `"independence"` otherwise -- i.e. the *head* (noun/
verb) gates the result, not the modifier. Getting this backwards (which
we did on the first pass, then caught via the Fig. 2b smoke test) silently
produces plausible-looking but wrong relations for every phrase with a
present-on-one-side-only adjective, so if you touch this function, rerun
`tests_smoke_causal_model.py` and check all three worked examples still
match.

## Label imbalance in free sampling

Free (unconstrained) sampling from the vocab produces >90% "neutral"
labels -- not a bug, just a consequence of natural-logic composition:
composing five levels of `standard_phrase`/`determiner_phrase`, each of
which collapses to "independence" (-> neutral) whenever nouns/verbs
differ, makes independence extremely likely to propagate to the root.
`data/generate_dataset.py::generate_split(..., balance=True)` (the
default) rejection-samples into class-balanced buckets instead.
