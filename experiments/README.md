<!-- SPDX-License-Identifier: MPL-2.0 -->

# Adaptive behavior laboratory

This directory is an executable research laboratory, not a unit-test suite. It
drives TSQ's real `AdaptiveEngine`, learner model, release-pinned graph, policy,
event ledger, and projection tables against deterministic adversarial action
patterns. It does not implement an alternate selector or learner model.

The laboratory currently probes:

- deliberate, implausibly fast, and hinted correct answers;
- correct answers paired with explicitly low confidence;
- confident named-misconception choices and persistent failure;
- explicit abstention with low confidence;
- a repeated fixed-option response bias;
- successful repair followed by independent verification;
- a learner who repairs but repeatedly fails verification;
- oscillating answer behavior;
- a targeted causal-masking and attention-scaling weakness; and
- a heterogeneous probabilistic skill profile;
- cross-session family reuse only after the review interval, with a durable
  delayed-retrieval certificate; and
- induction and later reversal of a named misconception across independent
  families and spaced sessions.

Each scenario runs in its own disposable database. By default it is replayed in
a second fresh database and the stable behavior signature and learner projection
are compared. The full JSON artifact contains the curriculum subgraph, every
adaptive step, remediation episodes, transition reasons, boundary decisions,
final learner projections, diagnostic findings, integrity results, cross-profile
comparisons, and any contradicted behavioral hypotheses. When a run reaches a
corpus gap, the artifact also shows remaining per-concept family capacity and
the exact quarantined authoring blueprint emitted by the engine.

The artifact also states the present observation boundary. TSQ can observe the
final option or abstention, confidence, latency, and hint count. Its immutable
semantic-action ledger can additionally record digest-only revisions,
check-result summaries, artifact checkpoints, and allowlisted tool-use purpose
codes. Raw code and free-form reasoning are deliberately excluded. These traces
remain observational: no action or model-scored artifact silently changes the
learner projection. Deterministic or human rubric evaluation is a separate
evidence boundary that is not yet connected to the online mastery updater.

Run the default Transformers laboratory from the repository root:

```sh
python experiments/adaptive_behavior_lab.py
```

The full artifact is written to `experiments/results/adaptive_lab.json`, which is
ignored by Git. The command prints a compact JSON receipt. To emit the entire
artifact to standard output instead:

```sh
python experiments/adaptive_behavior_lab.py --stdout
```

Useful focused runs:

```sh
python experiments/adaptive_behavior_lab.py --list-scenarios
python experiments/adaptive_behavior_lab.py \
  --scenario deliberate_correct \
  --scenario targeted_attention_gap \
  --steps 40
python experiments/adaptive_behavior_lab.py \
  --database-dir experiments/results/databases \
  --fail-on-hypothesis
```

`--database-dir` intentionally preserves synthetic databases for manual SQL
inspection. Each invocation creates a separate `run-*` directory. Never point it
at a real learner database. Generated databases and JSON artifacts must remain
uncommitted.

Exit status `2` means a hard invariant failed, such as event-ledger corruption,
a nondeterministic replay, or a repeated item family inside remediation. Exit
status `3` is used only with `--fail-on-hypothesis` when observed behavior
contradicts an explicit cross-profile hypothesis. Corpus exhaustion is recorded
as evidence in the artifact rather than hidden by a laboratory-only fallback.

The order-invariance laboratory compares the historical sequential Gaussian
approximation with TSQ's exact grid posterior. It permutes the same evidence,
tests same-family budgets, retention boundaries, numerical tail error, and
records a runtime benchmark:

```sh
python experiments/order_invariant_evidence_lab.py
```

Its deterministic artifact is written to
`experiments/results/order_invariant_evidence_lab.json`. The benchmark timing is
kept outside the artifact digest because wall-clock duration is not replayable.

The performance laboratory builds disposable cold and evidence-rich databases,
measures selection, answer submission, objective-state loading, and projection
hashing, and reports profiler evidence rather than hiding regressions behind a
single aggregate time:

```sh
python experiments/learner_runtime_lab.py \
  --output experiments/results/learner_runtime_lab.json
```

It fingerprints the configured protected database before and after the run and
fails if that database changes. Temporary benchmark copies are destroyed.

The objective-discovery laboratory asks the production policy to identify one
stark, objective-localized weakness over several 45-day-spaced sessions. It
checks localization, unrelated-objective separation, exact projection replay,
a later weak-to-strong recovery schedule, and the bounded two-family
persistent-gap episode contract:

```sh
python experiments/objective_discovery_lab.py --fail-on-hypothesis
```

Its ignored artifact includes every selected family, phase transition, objective
snapshot, detection rank, recovery path, durable episode spend, interleaving
audit, and independent-family certificate check. This is a deterministic
identifiability falsification probe, not human calibration or evidence of a
causal teaching effect.

The multimodal evidence laboratory exercises the pure task, rubric, action,
evaluation, dependence-cap, and evidence-reduction contracts without executing
learner artifacts or writing a database:

```sh
python experiments/multimodal_evidence_lab.py
```

It writes an ignored canonical artifact to
`experiments/results/multimodal_evidence_lab.json` and verifies a deterministic
rerun. The laboratory includes implementation, debugging, explanation, and
design scenarios; assisted and post-feedback work; missing evidence; restricted
tool use; and shadow-only model/imported scores.
