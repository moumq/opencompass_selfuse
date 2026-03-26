# Flexible Eval Dataset Mapping

This directory is the runtime source of truth for flexible dataset resolution in
[`examples/eval_flexible.py`](/home/qinghua/liumq/opencompass_selfuse/examples/eval_flexible.py).

Resolution order is:

1. Treat `datasets[].ref` as an exact dataset ref first.
2. If no exact ref matches, treat it as a benchmark family name.
3. Resolve that family through `benchmark_default_refs.json`.

That means:

- New users can write a benchmark family such as `aime2024`.
- Advanced users can write an exact ref such as
  `aime2024_llmjudge_gen_5e9f4f`.
- Exact refs always override family default mapping.

Files in this directory:

- `benchmark_default_refs.json`
  Benchmark family to default exact ref used at runtime.
- `benchmark_variant_refs.json`
  Benchmark family to all known exact refs for discovery and tooling.

Any helper tables under `/home/qinghua/liumq/test/cfgs` are only exported
indexes for humans. They are not the runtime source of truth.
