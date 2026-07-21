# 100,000-question retrieval baseline

Measured 2026-07-21 with Python 3.13.0 and SQLite 3.45.3 on arm64 macOS.
The command was:

```shell
.venv/bin/python benchmarks/benchmark_candidate_retrieval.py \
  --questions 100000 --rounds 5 --json
```

The schema-v7-safe deterministic synthetic bank occupied 132,050,944 bytes and
seeded in 14.63 seconds. Five warm-cache rounds with a 600-item candidate bound
produced:

| Stage | Median | p95 |
| --- | ---: | ---: |
| candidate IDs | 83.96 ms | 85.47 ms |
| candidate retrieval and hydration | 100.29 ms | 100.99 ms |
| candidate-scoped exposure summary | 0.99 ms | 1.06 ms |
| complete policy selection | 120.47 ms | 145.65 ms |

This is a reproducibility baseline, not a cross-machine latency guarantee. It
uses a cold learner with no attempts, one concept, one family per item, and four
options per question. Conceptual and application families alternate so generic
repair/verification capacity remains serviceable. It therefore measures the family-distinct fast path; the
family-diverse fallback is certified separately by the prolific-family regression.
The full-policy measurement includes durable decision and event writes.

Compared with the original 283.8 ms full-policy baseline, the current median is
57.6% faster. The stronger v7 schema, indexes, release triggers, and safe staging
make this synthetic seed 6.9% larger and materially slower to construct; that is
an offline benchmark-fixture cost, not a steady-state selection regression.

SQLite's plan now enters the bank through the primary-concept covering index,
then uses indexed release, question, revocation, exposure, and focused-option
lookups. The small temporary requested-scope table is scanned intentionally, and
the final score ordering still uses a temporary B-tree:

```text
MATERIALIZE personal
SEARCH presented USING COVERING INDEX idx_decisions_learner_question (learner_id=?)
SEARCH presented_question USING INDEX sqlite_autoindex_questions_1 (id=?)
USE TEMP B-TREE FOR GROUP BY
SCAN scope
SEARCH qc USING COVERING INDEX idx_question_concepts_primary_scope (concept_id=?)
SEARCH rq USING INDEX idx_release_questions_question_release (question_id=? AND release_id=?)
SEARCH q USING INDEX sqlite_autoindex_questions_1 (id=?)
CORRELATED SCALAR SUBQUERY 3
SEARCH revoked USING COVERING INDEX sqlite_autoindex_question_revocations_1 (question_id=?)
SEARCH personal USING AUTOMATIC COVERING INDEX (family_id=?) LEFT-JOIN
LIST SUBQUERY 1
SEARCH focused USING COVERING INDEX idx_options_misconception_question (misconception_id=?)
USE TEMP B-TREE FOR ORDER BY
```

The final ordering sort and hydration tail are the visible next optimization
targets if the online latency budget tightens. Re-run the script and compare
machine-readable output before changing retrieval SQL or indexes.
