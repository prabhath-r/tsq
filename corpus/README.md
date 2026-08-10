# TSQ curriculum source

This directory is the editable source of the bundled TSQ curriculum. It is
split by learner-facing topic so an author can review one area without opening
one very large JSON document.

## Layout

- `manifest.json` has the exact keys `format`, `format_version`,
  `schema_version`, `title`, `shared_file`, and `topic_files`. Topic entries are
  `{ "topic_id": "...", "path": "topics/<slug>.json" }` in depth-first
  catalog order.
- `shared.json` has the exact arrays `domains`, `edges`, `objective_edges`, and
  `sources`.
- `topics/*.json` has the exact keys `topic`, `concepts`,
  `learning_objectives`, `misconceptions`, and `questions` for one topic.
- `AGENTS.md` is the required authoring and review protocol. Read it completely
  before changing corpus content.
- `src/tsq/data/curriculum/` is the synchronized package copy. It is an output,
  not an authoring tree; never edit it by hand.

A question belongs in the shard that owns its primary concept. Cross-topic
questions stay in that one shard and use typed concept mappings and related
topic references for real overlap. Definitions and questions must not be
copied between shards.

The loader validates the assembled corpus as one release. A topic shard is not
an independent corpus and must not be audited as one.

## Normal workflow

Start by checking that the shards assemble deterministically and still match
the packaged resource:

```bash
python3 scripts/sync_bundled_corpus.py
```

After editing a shard, validate the complete canonical source first:

```bash
PYTHONPATH=src python3 -m tsq audit corpus --strict
PYTHONPATH=src python3 -m tsq capacity corpus --topic t_transformers --json
PYTHONPATH=src python3 experiments/family_independence_lab.py --stdout
```

Replace `t_transformers` with the topic being changed. The family laboratory
nominates cases for semantic review; it does not prove that two families are
dependent or independent. Only after the source passes should the trusted
packaged resource be synchronized:

```bash
python3 scripts/sync_bundled_corpus.py --write
python3 scripts/sync_bundled_corpus.py
```

Inspect the assembled behavior in a disposable database:

```bash
work_dir="$(mktemp -d)"
PYTHONPATH=src python3 -m tsq --db "$work_dir/tsq.db" init --corpus corpus
PYTHONPATH=src python3 -m tsq --db "$work_dir/tsq.db" topics --json
PYTHONPATH=src python3 -m tsq --db "$work_dir/tsq.db" graph t_transformers --json
PYTHONPATH=src python3 -m tsq --db "$work_dir/tsq.db" coverage \
  --topic t_transformers --json
```

Then run the focused corpus, authoring, capacity, packaging, and behavioral
tests followed by the complete suite:

```bash
PYTHONPATH=src python3 -m unittest \
  tests.test_corpus \
  tests.test_corpus_misconception_routes \
  tests.test_objectives \
  tests.test_authoring \
  tests.test_capacity \
  tests.test_family_independence_lab \
  tests.test_packaging \
  tests.test_store_integrity
PYTHONPATH=src python3 -W error::ResourceWarning -m unittest discover -s tests
```

Draft questions and taxonomy proposals stay outside `corpus/`, the manifest,
and the packaged resource until the independent review sequence in `AGENTS.md`
passes. Accepted questions use `approved`. Keep `generated` and `human_review`
truthful descriptive provenance fields, omit all vendor/model identity from
public question provenance, and use `calibrated` only after relevant empirical
item evidence exists.

Question status, release membership, and the current top-level review status
are the authoritative lifecycle state. Nested attestations, repair notes,
activation strings, and source-record notes describe review context recorded
when an immutable identity was published; they never impose a runtime gate.
Historical provenance can therefore mention an earlier authoring workflow
without changing an approved question's eligibility.

Emergency question revocation remains a separate runtime safety control for a
released item. Productive-performance tasks also have their own review and
lifecycle contract; neither mechanism changes the bundled curriculum authoring
rules here.
