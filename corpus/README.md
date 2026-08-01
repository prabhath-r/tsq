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
PYTHONPATH=src python3 -m tsq --db "$work_dir/tsq.db" quarantine list \
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
  tests.test_generated_activation \
  tests.test_packaging
PYTHONPATH=src python3 -W error::ResourceWarning -m unittest discover -s tests
```

Generated questions are candidates only. They stay quarantined, do not count
toward live coverage or capacity, and cannot be selected for a learner. Use
`capacity --quarantine-impact` only to prioritize review; never report its
counterfactual result as live capacity. The current schema cannot quarantine
topics, concepts, objectives, misconceptions, sources, or graph edges, so
AI-proposed definitions must remain non-runtime prose outside `corpus/` until
they receive independent human semantic and source review.
