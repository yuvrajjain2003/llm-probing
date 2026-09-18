# `src/` layout

Grouped by **pipeline stage**, not by goal — Goal 1 and Goal 2 share
`generation/run_eval.py` (prompt rendering, `parse_output`, answer
normalization, entropy, JSONL resume), so a goal-shaped split would have
duplicated it.

```
generation/   talks to the model (vLLM) + maintains the raw records it writes
  run_eval        base CoTs; ALSO the shared parse/normalize/prompt helpers
  checkpoint      split a CoT into sentences, cut prefixes at 10/30/50/70/90%
  continuations   Goal 1 step 3: K continuations from each prefix
  gpu_pool        dynamic (num_shards, shard_index) work queue across GPUs
  merge_shards    concatenate <name>.shardNN.jsonl -> <name>.jsonl
  reparse         re-apply parse_output to stored text; no regeneration

analysis/     pure CPU reductions over those records — no GPU, no model
  labels          Goal 1 step 4: checkpoint_labels + commitment tables
  robustness      K-sufficiency, parser-failure catalog, surface-form leakage
  k_sufficiency   standalone K-subsampling study (bias / precision / decision)
  year_effect     AIME contamination probe: do labels differ across years?

probing/      Goal 2. Run in this order:
  verify_capture       correctness gate: EasySteer capture vs stock HF
  extract_activations  re-feed Goal 1 CoTs, record hidden states
  probe_smoke          is there ANY decodable signal (pipeline check)
  probe_linear         layer x depth-bin linear probes, with a depth control
  probe_goal1_labels   probe the Goal 1 behavioural labels (the real question)

viz/          figures, tables, PDFs — reads run outputs, never a model
  aggregate_viz      per-dataset tables -> goal1_viz.json (what plot_goal1 reads)
  plot_goal1         headline entropy-vs-checkpoint + small multiples
  make_figures       publication figures
  report             summary tables/plots for a whole eval sweep
  make_timing_report docs/goal1_timing_estimate.pdf

setup/        download_dataset — one-time fetch into data/
```

## Running things

From the **repo root**, either form works:

```bash
python src/analysis/labels.py --continuations <run>/raw/continuations.jsonl
python -m analysis.labels --continuations ...        # with src/ on PYTHONPATH
```

Modules that import across packages prepend `src/` to `sys.path` themselves, so
the plain script form does not need `PYTHONPATH` set. Paths in the code are
resolved relative to the repo root (`Path(__file__).parents[2]`), and the
launcher scripts under `runs/` expect to be run from the repo root too.

## Import convention

- Same package → bare import (`from checkpoint import extract_cot` inside
  `generation/`).
- Across packages → package-qualified (`from generation.run_eval import grade`),
  preceded by the `sys.path` shim.
