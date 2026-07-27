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
- a broad-curriculum fixed-position probe that crosses the exact family-wise
  diagnostic threshold while remaining observational only;
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

The cold-start laboratory measures what fresh learners are actually served
across broad LLM, Transformers, and LLM Agents topics. It uses deterministic
credible-correct, credible-wrong, and explicit-abstention profiles, records
authored difficulty, predicted success, objective depth, prerequisite descent,
deliberate exploration, and categorized gap termination, then repeats the
complete run on a second disposable database. Replication compares both the
full stable behavior trace and learner-projection semantics:

```sh
python experiments/cold_start_lab.py
```

Its ignored artifact juxtaposes static inventory with routed traces to help
distinguish corpus scarcity from routing behavior without retuning the policy.
Each served step includes the production decision's final scored candidate
count and ranked top-ten prefix. A complete prefix has its durable ordered
rank-and-quantized-coverage digest verified; a truncated prefix explicitly
reports how many candidates remain unobserved. The durable digest covers
ordered question IDs, total scores rendered to 8 decimal places, exact integer
coverage counts, and diagnostic-information values rendered to 12 decimal
places—not exact binary floats or every component score. The artifact
separately hashes the complete stored prefix for deterministic replication.
These are post-eligibility traces, not raw release inventory, and they cannot
attribute an exclusion to one specific filter. Difficulty and predicted
success remain uncalibrated diagnostics; the laboratory is not human ability
or efficacy evidence.

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
causal teaching effect. The dual-gap profile deliberately uses a longer spaced
horizon so both weak targets and the strong controls have at least two
independent observed families before localization is judged.

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

The productive-probe laboratory drives the complete mixed path through two
fresh databases: a real wrong MCQ answer localizes an objective and named
misconception, reviewed release-pinned debugging and explanation tasks are
ranked, one productive attempt records assisted semantic actions, an imported
evaluation is reduced into shadow evidence, and a fresh family is recommended
next. It hashes a real temporary learner file at both the artifact and
submission boundaries, confirms both commitments match the fixture bytes, and
scans every persisted table for distinctive private sentinels. It also proves
that MCQ and productive attempts cannot overlap, the selected-response
projection does not change, and the performance projection replays exactly:

```sh
python experiments/productive_probe_lab.py --stdout
```

The ignored artifact is written to
`experiments/results/productive_probe_lab.json`. The fixture tasks exist only
inside disposable laboratory databases; TSQ does not ship them as reviewed
curriculum content. The file is never executed. Its digest proves byte equality
only: it is not skill evidence, encryption, or protection against guessing
low-entropy content.

The artifact-runner laboratory exercises the fixed data-only causal-mask
checker and its durable admission/receipt ledger across two fresh databases:

```sh
python experiments/artifact_runner_lab.py --stdout
```

The valid, semantic-failure, and malformed cases invoke the real child process.
Deterministic receipt injections cover timeout and worker-start failure, and a
controlled post-admission exception covers the unresolved-crash path. The lab
verifies at-most-once behavior under varied caller keys, projection neutrality,
absence of each full artifact byte sequence and path from its database scan,
integrity, and exact projection-copy replay. The checker process boundary is
not an operating-system, filesystem, or network sandbox. Results remain
shadow-only and cannot apply mastery, skill authority, evaluation, or
certification.

The scoring-reconciliation laboratory exercises provider failure after a
durable callback admission. Across two fresh databases it keeps ambiguous
lookups unknown, recovers one authority-free result after `SessionEnded`, and
closes a second operation as definitely absent only through an adapter with an
explicit non-acceptance guarantee:

```sh
python experiments/scoring_reconciliation_lab.py --stdout
```

The laboratory retries both scoring commands adversarially and proves that the
provider callback count remains one. Reconciliation events stay outside the
ended session envelope, recovered evidence remains zero-weight shadow data,
learner projections do not move, and integrity plus copy replay remain exact.
The bundled adapter is synthetic; the result does not validate any production
provider receipt authority.

The family-independence laboratory nominates active cross-family pairs through
documented token-overlap signals, then counterfactually collapses each review
cluster before running the exact capacity analyzer:

```sh
python experiments/family_independence_lab.py --fail-on-critical
```

Its deterministic artifact distinguishes lexical review candidates from proven
dependence. Exit status `3` means a candidate would reduce exact repair or
verification capacity if semantic review confirmed that its families collapse;
it is not a dependence verdict and the laboratory never edits the corpus. The
artifact also tests the quarantined Transformer capacity-repair batch by making
its frozen question dataclasses eligible in memory only. It reports baseline,
declared-family collapse, candidate expansion, and collapsed-plus-candidate
capacity for the exact objective and misconception signature. This is a
counterfactual authoring check: generated candidates remain quarantined,
human-unreviewed, manual-activation-only, and semantically unverified.

The same artifact declares the three active teacher-forced causal-visibility
families as a semantic-review cluster, including the pair that falls below the
lexical nomination threshold. It then evaluates the complete sixteen-subset
power set of four quarantined causal-reserve families under declared-family,
batch/training-pair-collapse, and three-family-collapse assumptions. Every
subset is measured at both the whole causal-masking concept and the exact
three-misconception route. Only one frozen representative of the
cross-attention family is made eligible, so its `_001`/`_002` revision pair
still contributes one family. Missing legacy generation, human-review, or
manual-activation provenance is reported explicitly; quarantine remains the
activation ceiling and no capacity result establishes semantic independence.

The policy-shadow comparison laboratory gives many fresh synthetic learners
exactly one production-selected question each, then evaluates the event-backed
safe frontiers without modifying the live policy or learner state:

```sh
python experiments/policy_shadow_comparison_lab.py
```

It compares reported live, uniform-frontier, and frozen-greedy one-step
estimates with the response generator's declared probability for every logged
frontier action. Smooth, threshold, ability-only, and localized-weakness
profiles span learn, diagnose, and review phases while exercising model
misspecification. The replicated ignored artifact
checks independent IPS arithmetic, overlap/ESS guards, projection replay, and
source-database non-mutation. It reports aggregate, response-profile, and
session-phase overlap separately so a healthy aggregate cannot hide an
underpowered subgroup. Low overlap is inconclusive. Synthetic oracle recovery
is estimator evidence only—not human calibration, an alternate adaptive
trajectory, teaching benefit, retention, or a reason to promote the greedy
challenger.
