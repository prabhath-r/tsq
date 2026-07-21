# Large-bank candidate benchmark

This is a query-shape and latency benchmark, not a corpus-quality or concurrency certification. It creates synthetic, trivially worded rows solely to exercise the same sealed-release candidate path used by the policy.

Run the benchmark explicitly; it is intentionally excluded from normal tests:

```shell
python3 benchmarks/benchmark_candidate_retrieval.py
```

The default run creates a deterministic 100,000-question SQLite bank in a
temporary directory and reports warm-cache medians and p95 values for the
candidate-ID query, 600-question hydration, candidate-scoped exposure lookup,
and complete adaptive policy selection. Families alternate conceptual and
application kinds so the synthetic bank preserves the live policy's generic
repair/verification capacity invariant. It also prints SQLite's query plan so a
regression from indexed lookups to table scans is visible.

For a more stable comparison run several rounds and retain the environment and query plan with the timing values:

```shell
python3 benchmarks/benchmark_candidate_retrieval.py \
  --questions 100000 --rounds 7 --json
```

Use `--json` for machine-readable output. Pass `--db PATH` to retain the generated database; an existing path is never overwritten. For a quick smoke run, use `--questions 2000 --rounds 2`.

Results vary with CPU, storage, Python, SQLite, cache state, and candidate distribution. Compare like-for-like runs; there is no encoded SLA. The benchmark does not exercise strict JSON import, corpus auditing, full-database integrity verification, authoring quality, or multiple concurrent writers.
