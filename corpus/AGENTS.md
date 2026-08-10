# Rules for TSQ corpus work

These instructions apply to every file under `corpus/`. They are the local
authoring contract for people and coding agents. Read this file completely
before adding a topic, concept, objective, misconception, source, family, or
question.

TSQ does not need more quiz text for its own sake. It needs source-grounded
observations that let the adaptive policy distinguish what a learner probably
understands, which specific error they may be making, what repair is available,
and whether that repair transfers to an independent task. A small batch of
good questions is more useful than a large batch of plausible-looking filler.

## Non-negotiable boundaries

1. Draft questions and taxonomy proposals stay outside `corpus/`, its manifest,
   and the synchronized package. A question accepted into the canonical corpus
   uses `approved`; use `calibrated` only after relevant empirical item evidence
   exists. Do not put unlisted JSON proposals under `corpus/` because the
   manifest is a closed inventory.
2. Before a new domain, topic, concept, objective, misconception, source, edge,
   family, or question enters the corpus, it must pass independent semantic,
   source, family, and keyed review appropriate to the change. Automated checks
   are necessary but do not establish factual correctness or independence.
3. Provenance is descriptive, not an eligibility switch. AI-authored or
   AI-revised work records `provenance.generated` as the JSON boolean `true`.
   `provenance.human_review` is `true` only if a person actually reviewed the
   item and is otherwise `false`. Never invent a human identity, review event,
   or empirical result. Public question provenance must not name or encode a
   vendor, model, provider, or generator identity, including through aliases or
   nested metadata.
4. Never change content behind a registry-stable concept, objective,
   misconception, source, or question ID. A changed question gets a new
   question ID, a higher version, a `revision_of` pointer, and the same family
   as its parent. Domains, topics, and graph edges are release-scoped catalog
   snapshots; reviewed changes to them still require a newly sealed immutable
   release, preserve old pinned sessions, and must not silently change the
   meaning of an existing topic.
5. Never weaken a validator, coverage target, family rule, audit, or test to
   make content pass.
6. Never add low-quality items to raise a count. Every distractor must encode a
   credible named misconception. Joke answers, random falsehoods, and generic
   “does not know the topic” labels are forbidden.
7. Never describe authored item parameters as calibrated. Only empirical item
   evidence can support `calibrated`. Never describe an
   MCQ result as proof that a learner can implement, explain, debug, design, or
   complete a real project.
8. Never hand-edit `src/tsq/data/curriculum/`. It is synchronized from this
   directory and packaged for the runtime.

If one of these rules conflicts with a request to produce more content, stop
the content work and report the conflict.

This document governs bundled selected-response curriculum questions. Emergency
question revocation remains a separate runtime safety control and can disable a
released item without rewriting a sealed release. Productive-performance tasks
also retain their separate proposal, review, and lifecycle rules; do not infer
their status or readiness from this curriculum workflow.

## Where data belongs

The source tree has three layers:

- `manifest.json` has exactly `format`, `format_version`, `schema_version`,
  `title`, `shared_file`, and `topic_files`. `format` is
  `tsq-curriculum-shards`, `format_version` is `1`, `schema_version` is `3`,
  and `shared_file` is `shared.json`. Each topic entry has exactly `topic_id`
  and `path`, uses a normalized `topics/<clear_lowercase_slug>.json` path, and
  appears in depth-first catalog order.
- `shared.json` has exactly four arrays: `domains`, `edges`,
  `objective_edges`, and `sources`.
- Each `topics/*.json` file has exactly `topic`, `concepts`,
  `learning_objectives`, `misconceptions`, and `questions`.

A question is stored with the topic that owns its primary concept. Do not copy
a question into every topic named by its tags or supporting concepts. Express
real overlap through typed concept mappings and symmetric `related_topic_ids`.
Cross-topic graph references are allowed and resolved only after every shard
has been assembled.

Ownership is exact: a topic shard owns the concepts listed by its topic;
learning objectives follow their `primary_concept_id`; misconceptions follow
their `concept_id`; and questions follow their exactly-one `primary` concept
mapping. Sources and both graph-edge arrays stay in `shared.json`. Supporting
concepts, diagnostic misconceptions, sources, families, and edges may refer
across topics, but their definitions remain single-copy.

The manifest is a closed inventory. Unlisted JSON files, duplicate or
traversing paths, mismatched topic IDs, and content placed in the wrong owner
shard are errors. Assemble and validate the whole release; never audit one
topic shard as though it were a complete corpus.

Every entity definition must occur once across the assembled source. References
to those IDs in mappings, edges, options, and questions may repeat. A
`family_id` is an implicit dependency-group reference rather than a standalone
definition: it **must** repeat whenever questions share a stimulus, template,
derivation, or solution path. Keep stable IDs—and the raw concept-edge labels
noted below—plain, descriptive, and lowercase snake case:

- `d_...` for domains;
- `t_...` for topics;
- `c_...` for concepts;
- `lo_...` for learning objectives;
- `e_...` for editorial concept-edge labels; the current runtime identifies a
  concept edge by its endpoints, relation, and weight rather than this label;
- `oe_...` for objective edges;
- `m_...` for misconceptions;
- `f_...` for shared question-family dependency groups;
- `q_..._001` for questions; and
- `src_...` for sources.

Use readable topic filenames such as `llm_agents.json` or
`retrieval_augmented_generation.json`. Do not put migration numbers, dates, or
meaningless version labels in filenames.

## Build the taxonomy before the questions

Use these layers for different jobs:

- A **domain** is the broad ownership root.
- A **topic** is a learner-facing navigation bucket. It is not itself evidence
  of mastery and does not imply a prerequisite.
- A **concept** is a stable assessable unit. Each concept has exactly one owner
  topic in the catalog.
- A **learning objective** is a fine content boundary joined to an observable
  operation.
- A **misconception** is a falsifiable error model owned by one concept.
- A **family** is the unit of response dependence: items sharing a stimulus,
  template, derivation, or solution path belong together.
- A **question** is one selected-response observation within that structure.

Parent topics may be empty containers. Do not create a vague scored concept
solely to mirror a topic name. Use child topics and atomic concepts instead.

`related_topic_ids` is for bounded exploration, not hidden prerequisite
behavior, and every related-topic declaration must be symmetric. Concept graph
edges have typed meanings. For `prerequisite` and `requires`, the source is the
prerequisite and the target is the dependent concept. For `part_of`, the source
is the part and the target is the whole. Strict readiness edges, containment
edges, and their mixed closure must remain acyclic. Other relation types are
descriptive and must not be used to smuggle in readiness semantics.

Learning objectives must:

- name one primary concept and only genuinely needed supporting concepts;
- use one supported operation: `distinguish`, `explain`, `predict`, `trace`,
  `diagnose`, or `apply`;
- use `selected_response` as the current evidence type;
- state the exact distinction or operation that the response measures; and
- be narrow enough that different failure hypotheses can be tested directly.

If an eligible question's primary concept is covered by the objective catalog,
the question must declare the exact `learning_objective_id`. An objective edge
is a reviewed claim, not decoration: it needs an ID, a source prerequisite, a
target dependent, a weight in `(0, 1]`, and a concrete rationale.

## Write misconceptions before distractors

A useful misconception predicts a learner's choice across more than one
wording. It names the wrong rule, causal model, boundary, or inference.

Good examples:

- “Schema-valid tool arguments are already an authorized execution.”
- “Attention values determine the query-key softmax weights.”
- “A causal decoder mask also truncates the encoded source memory.”

Bad examples:

- “Does not understand agents.”
- “Is confused about attention.”
- “Chooses option B.”
- A silly statement included only to make the key obvious.

Before adding a misconception, inspect the full assembled registry. Reuse an
existing ID only when it predicts the same error. Do not stretch an old
description to cover a different error. Every misconception has one owning
concept and a falsifiable description.

Each TSQ question must expose three distinct misconception IDs, one for each
distractor. The misconception owner's concept must appear in that question's
typed concept mappings. The distractor's rationale must explain the local
logical or causal failure; “this is incorrect” is not a rationale.

For objective-aware questions, every distractor normally names the question's
direct objective in `diagnostic_objective_id`. A cross-objective diagnosis is
allowed only when it is deliberate, the misconception owner belongs to the
diagnostic objective, and the active bank contains the full direct repair and
verification route for that exact objective/misconception pair. The correct
option must not have a misconception or diagnostic objective.

## Families are evidence boundaries

Different `family_id` values are treated as independent evidence by the
runtime, so family assignment is a safety decision.

These changes do **not** make a new family:

- swapping names, numbers, tokens, or domains in the same calculation;
- paraphrasing the same rule;
- changing option order;
- asking the same derivation in forward and reverse wording;
- adding detail to the same stimulus; or
- correcting an existing question.

Those items remain in the same family. A revision always preserves its
parent's family.

A new family needs a genuinely different keyed solution operation. Useful
contrasts include tracing dataflow, diagnosing code or wiring, predicting an
intervention, evaluating a causal or measured result, calculating a resource
bound, and applying the same objective in a materially different
representation. Even these labels do not prove independence; compare the
actual steps needed to solve the item.

Before writing, make a family-operation matrix with these columns:

| Question | Objective | Named errors | Kind | Difficulty | Keyed operation | Closest family | Why separate or same |
|---|---|---|---|---:|---|---|---|

If the independence argument is uncertain, reuse the existing family or keep
the draft outside the corpus with an explicit `independence_note`. Never create
a fresh family merely because the desired coverage count is low.

`transfer` has a strict meaning: the learner must use the target knowledge in a
materially changed context, representation, or operation. A new tag, renamed
actor, or synonym is not transfer.

The family-independence laboratory uses lexical and structural signals only to
nominate human review. It cannot establish dependence or independence. Review
answer-redacted items by inferring and comparing their solution paths.

## Coverage required by the adaptive path

Raw question count is not the target. The active bank must preserve another
route after a question is used.

For every assessable primary concept, the initial live gate requires at least
three safely serviceable active families. Any main item must leave:

1. a different family that can repair each named misconception; and
2. another, distinct verification-kind family in the same learning scope.

For every direct learning objective and every objective/misconception pair
exposed by an active distractor, plan at least three direct families, including
at least two verification-capable families. Verification-capable kinds are
`application`, `calculation`, `comparison`, `counterfactual`, `debugging`, and
`transfer`. This permits a trigger family, a repair family, and a different
verification family.

Three families are only the initial path. Run exact sustained-capacity analysis
and author against its blocker witnesses. Do not let a healthy topic aggregate
hide a missing, thin, or order-sensitive owned concept or objective. The
coverage planner's concept-kind targets and authored difficulty sequence are
planning aids, not permission to manufacture redundant families.

Drafts do not satisfy live coverage or capacity because they remain outside the
assembled corpus. Capacity reports describe only the accepted release being
audited; do not add unreviewed drafts temporarily to make a report look healthy.

## Question contract

Solve the problem before drafting options. State the assumptions required to
make one answer defensible. Avoid facts that can become stale unless the source
and relevant date are explicit.

Every question must have:

- a stable ID, version, family, lifecycle status, kind, and optional revision
  parent;
- a precise stem long enough to establish a real reasoning task;
- exactly four substantive, normalized-distinct options and exactly one key;
- stable option IDs, normally `a`, `b`, `c`, and `d`;
- no “all of the above,” “none of the above,” or combined meta-option;
- three plausible distractors mapped to three distinct named misconceptions;
- a specific local rationale for all four options;
- exactly one primary concept mapping;
- positive concept weights summing to `1.0` within `0.02`;
- at least one source ID that supports the material claim; and
- provenance and useful retrieval tags.

Options must be parallel in grammar, specificity, abstraction, plausibility,
and approximate length. The answer must not be discoverable from length,
qualifiers, grammar, punctuation, vocabulary, or source position. Keep key
positions balanced in the assembled bank even though runtime presentation
shuffles them.

The stem should require reasoning rather than keyword recognition. A difficult
item should require a deeper operation, closer competing hypotheses, or a real
transfer—not longer prose or rarer jargon. Try to build a counterexample to the
key and rewrite or reject the item if one survives the stated assumptions.

Concept roles are closed: `primary`, `secondary`, `supporting`, `prerequisite`,
`context`, `contrast`, and `transfer`. Weights express evidence attribution,
not keyword frequency. A supporting or context mapping must not be presented as
proof of mastery of that concept.

Question kinds are closed: `diagnostic`, `conceptual`, `application`,
`debugging`, `counterfactual`, `transfer`, `prerequisite_probe`, `calculation`,
and `comparison`.

Item-model fields are authored priors:

- `difficulty`: finite and in `[-4, 4]`;
- `discrimination`: finite and in `[0.25, 3]`;
- `guess_rate`: use `0.25` for the four-option forced-response format; and
- `slip_rate`: finite and in `[0, 0.25]`.

Do not tune these numbers to make a simulated learner look better. Prefer the
target difficulty supplied by a release-pinned coverage blueprint. Do not mark
an item `calibrated` until real pilot evidence supports its parameter and
distractor behavior.

Tags help retrieval and reporting. They do not establish topic ownership,
source support, family independence, transfer, or review status.

## Sources and factual review

Prefer primary papers, standards, and official documentation. A source record
has a stable ID and title plus URI and license information when available.
Do not change metadata behind a published source ID; add a new ID if the
identity has materially changed.

Every factual claim needed to choose the key or reject a distractor must be
supported by the cited source set. A broad citation to a field or an
`expert_synthesis` record is not, by itself, claim-level support. Record in
provenance which source claims support the item and which details are original
synthesis. If a source is ambiguous, inaccessible, obsolete for the claim, or
has unclear usage rights, the item cannot be accepted.

The current schema records document-level sources, not complete claim excerpts
or signatures. Do not overstate what a source ID proves.

## Provenance and acceptance

For an accepted AI-authored item, use truthful provenance in this shape. Use a
stable unique batch ID and do not invent facts to fill placeholders. Public
question provenance deliberately omits vendor and model identity; operational
generation-job records may retain those details outside the curriculum item.
This prohibition is recursive: do not hide an identity in nested metadata or
under aliases such as `modelName`, `vendor_id`, or `generator_identity`.
Opaque output/review commitments and counts are allowed because they reveal no
vendor or model value.

```json
{
  "method": "ai_assisted_source_scoped",
  "generated": true,
  "batch_id": "<stable unique batch id>",
  "human_review": false,
  "review_status": "accepted_after_independent_review",
  "psychometrics": "uncalibrated_author_prior",
  "source_scope": "<source claims used and original synthesis added>",
  "independence_note": "<keyed operation and comparison with nearby families>"
}
```

Do not imitate old questions that omit `provenance.generated`. That exception
is bound by hash to an exact legacy cohort and is unavailable to new or changed
content.

Draft artifacts produced by an offline authoring pipeline stay outside the
canonical shards and packaged resource. A release maintainer may add an item as
`approved` only after its deterministic checks, blind solve, family review,
keyed critique, source review, and route review all pass. Keep
`generated=true` for generated content and keep `human_review=false` unless a
person actually performed a documented review; neither field grants or removes
runtime eligibility. Acceptance creates a newly sealed immutable release and
does not mutate sessions pinned to an older release.

## Shape templates

The topic-shard template below describes canonical accepted data. Draft
taxonomy work stays outside `corpus/` until independent semantic and source
review is complete; after acceptance it may be added to the manifest and
synchronized package in a new immutable release.

Use the existing shard's formatting and field order. A topic shard has this
closed outer shape; do not add shared edges or sources to it:

```json
{
  "topic": {
    "id": "t_example_topic",
    "domain_id": "d_artificial_intelligence",
    "parent_id": "t_parent_topic",
    "name": "Example Topic",
    "description": "A precise learner-facing scope statement.",
    "concept_ids": ["c_example_mechanism"],
    "related_topic_ids": ["t_related_topic"],
    "sort_order": 50
  },
  "concepts": [
    {
      "id": "c_example_mechanism",
      "name": "Example mechanism",
      "description": "The stable assessable unit owned by this topic.",
      "domain": "ai",
      "prior_mastery": 0.2
    }
  ],
  "learning_objectives": [
    {
      "id": "lo_example_trace",
      "name": "Trace the example mechanism",
      "description": "Trace the exact state transition under stated inputs and boundaries.",
      "primary_concept_id": "c_example_mechanism",
      "supporting_concept_ids": [],
      "operation": "trace",
      "evidence_type": "selected_response",
      "prior_mastery": 0.2
    }
  ],
  "misconceptions": [
    {
      "id": "m_example_state_is_intent",
      "concept_id": "c_example_mechanism",
      "name": "Intent is treated as observed state",
      "description": "The learner treats a planned change as authoritative state without observing the external result."
    },
    {
      "id": "m_example_observation_is_complete",
      "concept_id": "c_example_mechanism",
      "name": "Partial observation is treated as completion",
      "description": "The learner treats a partial result as proof that the complete terminal condition holds."
    },
    {
      "id": "m_example_retry_is_always_safe",
      "concept_id": "c_example_mechanism",
      "name": "Uncertain actions are blindly repeated",
      "description": "The learner assumes an action with an unknown result can be repeated without reconciliation or idempotency protection."
    }
  ],
  "questions": []
}
```

The set of `topic.concept_ids` must exactly match the concepts owned and
defined by the shard. An objective is stored with the owner of its
`primary_concept_id`; a misconception is stored with the owner of its
`concept_id`.

A fresh objective-aware candidate follows this shape. Replace the examples
with reviewed content; do not copy the wording as filler:

```json
{
  "id": "q_example_state_reconciliation_001",
  "version": 1,
  "family_id": "f_example_state_reconciliation",
  "learning_objective_id": "lo_example_trace",
  "status": "approved",
  "stem": "<precise scenario and question with all assumptions needed for one key>",
  "kind": "debugging",
  "difficulty": 0.25,
  "discrimination": 1.5,
  "guess_rate": 0.25,
  "slip_rate": 0.07,
  "concepts": [
    {
      "concept_id": "c_example_mechanism",
      "weight": 1.0,
      "role": "primary"
    }
  ],
  "options": [
    {
      "id": "a",
      "text": "<defensible best answer>",
      "correct": true,
      "misconception_id": null,
      "rationale": "<why this follows from the stated mechanism and evidence>"
    },
    {
      "id": "b",
      "text": "<plausible response predicted by named misconception one>",
      "correct": false,
      "misconception_id": "m_example_state_is_intent",
      "diagnostic_objective_id": "lo_example_trace",
      "rationale": "<why that specific model fails in this scenario>"
    },
    {
      "id": "c",
      "text": "<plausible response predicted by named misconception two>",
      "correct": false,
      "misconception_id": "m_example_observation_is_complete",
      "diagnostic_objective_id": "lo_example_trace",
      "rationale": "<why that specific model fails in this scenario>"
    },
    {
      "id": "d",
      "text": "<plausible response predicted by named misconception three>",
      "correct": false,
      "misconception_id": "m_example_retry_is_always_safe",
      "diagnostic_objective_id": "lo_example_trace",
      "rationale": "<why that specific model fails in this scenario>"
    }
  ],
  "source_ids": ["src_primary_source"],
  "provenance": {
    "method": "ai_assisted_source_scoped",
    "generated": true,
    "batch_id": "<stable unique batch id>",
    "human_review": false,
    "review_status": "accepted_after_independent_review",
    "psychometrics": "uncalibrated_author_prior",
    "source_scope": "<source claims used and original synthesis added>",
    "independence_note": "<keyed operation and comparison with nearby families>"
  },
  "tags": ["example_topic", "state_reconciliation"]
}
```

Add a supporting mapping only when it is needed, reduce the primary weight
accordingly, and keep the total at `1.0`. Do not add a supporting concept merely
to make an item look broader. For a revision, add `revision_of`, increase
`version`, use a new question ID, and preserve the parent family.

## Required review sequence

Use separate review views and do not let one stage inherit hidden answers from
another:

1. **Deterministic check:** JSON shape, references, one key, misconception
   ownership, revisions, clue checks, graph integrity, coverage, and package
   assembly.
2. **Blind solve:** remove the key, rationales, misconception IDs, status, and
   provenance. Solve from the cited source context and report ambiguity.
3. **Family review:** additionally remove family IDs and source-selection
   metadata. Infer the keyed operation and cluster candidates that share a
   solution path.
4. **Keyed critique:** inspect the answer, all rationales, diagnostic routes,
   source fit, counterexamples, and the proposed family assignment.
5. **Release acceptance:** a maintainer checks the independent review evidence,
   resolves every finding, and accepts, rejects, or revises the draft. The
   acceptor must not rely on the generator's assertion that its own work is
   correct.
6. **Pilot and calibration:** inspect discrimination, distractor use,
   ambiguity reports, timing, drift, and differential behavior before any
   calibrated claim.

Generator and independent-reviewer roles must be distinct. Keep operational
review evidence outside public question provenance when it would reveal a
vendor or model identity. Deterministic checks cannot prove factual
correctness, a unique best answer, semantic family independence, or
psychometric quality. Record `human_review=true` only when a person actually
performed a substantive review; acceptance itself does not change that fact.

## Workflow for a new or expanded topic

### 1. Establish the baseline

Read `corpus/README.md`, this file, the manifest, `shared.json`, the target
shard, related shards, and representative accepted questions.
Assemble the full bundle. Run the strict audit, topic capacity, coverage plan,
and family-independence report before editing. Record existing gaps and review
risks; do not assume a count is a gap.

### 2. Design the map

Define or verify:

- domain and parent topic;
- one clear owner for each concept;
- objective names, operations, concept scope, and prerequisite edges;
- named misconceptions and their owner concepts;
- source records and exact claims;
- independent keyed operations and family assignments; and
- a small matrix of kinds, difficulty priors, repair paths, and verification
  reserves.

Do not write the question batch until this map is coherent and acyclic.
If any proposed domain, topic, concept, objective, misconception, source, or
edge lacks completed independent semantic and source review, do not add it to
the canonical shards. Keep the draft outside `corpus/`, report the proposed map
for review, and stop that part of the expansion.

### 3. Author a small batch

Write only what the measured gap needs, within a reviewable maximum. Preserve
existing IDs. Draft outside the canonical shards, then put only independently
reviewed and accepted items in the primary concept's topic shard with
`status=approved`.

### 4. Review item behavior

For each item:

- solve the stem without the key;
- run a clue-only review of the options;
- search for a defensible counterexample or second key;
- test whether each distractor follows from its named misconception;
- compare its answer-redacted solution steps with neighboring families; and
- verify that using the family would still leave a distinct repair and
  verification route.

### 5. Assemble and validate

Run:

```bash
PYTHONPATH=src python3 -m tsq audit corpus --strict
PYTHONPATH=src python3 -m tsq capacity corpus --topic <TOPIC_ID> --json
PYTHONPATH=src python3 experiments/family_independence_lab.py --stdout
python3 scripts/sync_bundled_corpus.py --write
python3 scripts/sync_bundled_corpus.py
```

The exact capacity search may fail at an explicit state or evaluation bound.
That is a visible limit, not permission to substitute a heuristic. Report the
bound and reduce the requested analysis scope without weakening the release
gate.

Create a disposable database and inspect the topic catalog, graph scope,
coverage blueprints, and accepted release membership. Never use a learner's
real database for corpus experiments.

Run focused tests for corpus loading/assembly, quality, objectives,
misconception routes, authoring, capacity, family independence, packaging, and
release eligibility. Then run the complete suite with `ResourceWarning` treated
as an error.

### 6. Inspect adaptive behavior at the correct boundary

Drafts remain outside the release and therefore cannot change normal selection
or live capacity. After acceptance and creation of a new active release, run
learner simulations against the proposed route and inspect, rather than merely
count:

- a confident wrong answer selecting the named misconception and a related
  repair;
- an unsure response avoiding false certainty;
- repair followed by a different-family verification;
- repeated answers not manufacturing independent evidence;
- correct, incorrect, fast, slow, hinted, and missing-confidence behavior;
- a strong learner broadening within the requested topic;
- a weak prerequisite causing bounded descent and later parent resumption; and
- exhausted capacity producing an explicit corpus gap rather than a weak
  fallback.

Review question order, phase, focus objective, misconception hypotheses,
family reuse, projection changes, session report, and event integrity.

### 7. Report honestly

Report:

- files and taxonomy changed;
- questions and unique families by status, kind, objective, and primary concept;
- objective × misconception × family coverage;
- strict audit results;
- exact live-capacity results and any explicit search bounds;
- family-dependence risks and unresolved source claims;
- tests and behavior checks run; and
- work still requiring independent review or empirical calibration.

Do not describe a green audit, deterministic test, model review, lexical
similarity screen, or simulation as proof of item correctness, family
independence, human efficacy, or calibration.

## Stop and fail visibly when

Stop the batch, preserve the evidence, and report the exact blocker if any of
these conditions holds:

- an ID or definition is duplicated, ambiguous, or would need in-place
  mutation;
- a source does not support a material claim or its rights are unclear;
- the stem admits a second defensible answer or depends on an unstated
  assumption;
- a distractor is implausible, unnamed, or does not follow from its assigned
  misconception;
- a proposed family is only a paraphrase or its independence is unresolved;
- an objective/misconception route cannot preserve distinct repair and
  verification families;
- graph or objective edges create a cycle;
- split assembly is nondeterministic or the packaged resource differs;
- strict audit emits an error or warning;
- exact capacity exceeds a configured search bound;
- a draft was added to a canonical shard before independent review finished;
- a review stage would have to invent a human identity or calibration claim;
- focused or full tests fail; or
- the only way to meet a count is to lower the quality bar.

Do not hide a blocker by deleting the difficult concept, broadening the topic,
renaming a duplicate family, accepting unreviewed content, or reducing a gate.

## Copy-ready prompt for adding or expanding a topic

Copy the block below and replace every angle-bracket field. If required source
material or topic ownership is unknown, gather that evidence before running the
prompt. The prompt deliberately separates drafting, independent review, and
release acceptance.

```text
Extend TSQ's adaptive-learning corpus with one carefully designed topic or one
bounded expansion of an existing topic. Work as a corpus engineer and
assessment author, not as a quiz-volume generator. Quality and diagnostic value
take priority over the requested count.

Topic name: <TOPIC_NAME>
Existing topic ID, or NEW_TOPIC_PROPOSAL: <TOPIC_ID_OR_NEW_TOPIC_PROPOSAL>
Parent topic ID: <PARENT_TOPIC_ID>
Domain ID: <DOMAIN_ID>
Learner scope, including explicit exclusions: <IN_SCOPE_AND_OUT_OF_SCOPE>
Requested concepts or objectives: <REQUESTED_SCOPE_OR_NONE>
Primary sources or existing source IDs to inspect: <SOURCES>
Maximum accepted questions: <SMALL_REVIEWABLE_LIMIT>

Read corpus/README.md and every line of corpus/AGENTS.md before acting. Inspect
manifest.json, shared.json, the target and related topic shards, and the fully
assembled corpus. Inventory existing topics, concepts, learning objectives,
misconceptions, sources, question families, revisions, kinds, difficulty priors,
key-position balance, diagnostic routes, strict-audit findings, exact capacity
blockers, coverage blueprints, and family-independence nominations. Verify each
reused ID from repository data. Do not infer a content gap from raw count.

Keep every draft outside corpus/, the manifest, and the packaged curriculum
until all review stages below pass. Never put an unlisted JSON proposal under
corpus/. Design the change in this order:

1. domain and parent topic;
2. learner-facing topic boundary;
3. atomic assessable concepts with exactly one owner topic each;
4. fine learning objectives with one observable supported operation;
5. prerequisite and objective edges with direction and rationale;
6. falsifiable named misconceptions;
7. bounded source claims and original synthesis;
8. genuinely distinct keyed solution operations and family groups; and
9. questions, repair routes, and different-family verification reserves.

Use typed concept mappings and symmetric related-topic links for genuine
overlap. Do not duplicate definitions or questions across shards. Keep strict
readiness, containment, and mixed closures acyclic. A topic is a navigation
bucket, not evidence of mastery. A selected-response answer is evidence only
for its named objective and mappings; it does not certify implementation,
explanation, debugging, design, or project skill.

Before drafting options, produce a family-operation matrix with one row per
proposed question and these columns: question purpose; primary concept; direct
objective and operation; kind; authored-prior difficulty; three exact
distractor misconceptions; keyed solution steps; proposed family; closest
existing family and why the operation is the same or genuinely different;
source claims; source_scope; and the distinct repair and verification families
left after the item is used. Reject rows that are paraphrases, number or name
swaps, keyword recognition, trivia, implausible foils, or hidden duplicates.

Draft only the rows justified by that matrix. Each question needs one
defensible best answer under explicit assumptions; four substantive,
parallel, normalized-distinct options; exactly one key; three distinct credible
named misconceptions; one specific rationale per option; correct concept and
objective mappings; sources supporting every material claim; and no key clue
from length, qualifiers, grammar, punctuation, vocabulary, or position. Try to
construct a counterexample to the key. A transfer item must materially change
context, representation, or operation. family_id records shared solution-path
dependence, so reuse a family for paraphrases and revisions. A revision uses a
new ID, increments version, points to revision_of, and preserves its parent's
family.

For public provenance, record truthful descriptive facts. AI-authored or
AI-revised work has generated=true. Set human_review=true only if a person
actually performs a substantive review; otherwise use false. Include a stable
batch_id, psychometrics=uncalibrated_author_prior, a claim-level source_scope,
and a concrete independence_note. Do not include vendor, provider, model, or
generator identity anywhere in public provenance, including nested fields and
aliases. Do not invent a person, review event, timestamp, attestation,
calibration result, or source claim. Keep any operational generation identities
outside the curriculum item.

Use independent review roles that do not inherit the generator's hidden answer:

1. Run deterministic shape, reference, graph, revision, route, clue, and source
   checks on the draft.
2. Give a blind reviewer the stem, options, and necessary source context, but
   remove the key, rationales, misconception IDs, family ID, status, and
   provenance. Require an answer, derivation, ambiguity report, and any second
   defensible key.
3. Give a family reviewer answer-redacted questions without family IDs or
   source-selection metadata. Require inferred solution steps, dependence
   clusters, and challenges to every proposed independent family.
4. Give a keyed critic the complete draft and sources. Require checks of the
   key, every rationale, every misconception route, source fit, counterexamples,
   option clues, difficulty prior, and family assignment.
5. Resolve every finding by revising and repeating affected reviews. Reject the
   item if ambiguity, weak support, false independence, or a broken repair path
   remains.

Only after those stages pass may a release maintainer add the accepted records
to canonical shards. Accepted questions use status=approved. Generated and
human_review remain the truthful descriptive values established above; they do
not decide eligibility. Never use status=calibrated without relevant empirical
item evidence. Preserve stable IDs, place each definition in its one owning
shard, update the closed manifest only for an accepted topic shard, and create a
new immutable release without changing old pinned sessions.

Validate the assembled result, not an isolated shard. Run, at minimum:

PYTHONPATH=src python3 -m tsq audit corpus --strict
PYTHONPATH=src python3 -m tsq capacity corpus --topic <TOPIC_ID> --json
PYTHONPATH=src python3 experiments/family_independence_lab.py --stdout
python3 scripts/sync_bundled_corpus.py --write
python3 scripts/sync_bundled_corpus.py
PYTHONPATH=src python3 -W error::ResourceWarning -m unittest discover -s tests

Also run the focused corpus, objective, misconception-route, authoring,
capacity, family, packaging, database-upgrade, and release-eligibility tests
present in the repository. Treat every strict warning, source mismatch, package
parity failure, unresolved family nomination, route failure, and exact-search
bound as a finding to resolve or report, not as permission to lower a gate. Use
a disposable database to inspect topic resolution, graph scope, coverage demand,
release membership, and emergency revocation behavior. Never experiment on a
learner's real database.

After acceptance in a new active release, run adversarial learner sessions for
correct, incorrect, unsure, fast, slow, hinted, missing-confidence, and repeated
answers, plus remediation, different-family verification, prerequisite descent
and resumption, exploration, and corpus exhaustion. Inspect question order,
phase, focus objective, misconception hypotheses, family reuse, projection
changes, reports, and event integrity. These simulations validate control-flow
behavior; they do not establish semantic correctness or calibration.

Report the exact files changed; accepted and rejected draft counts; taxonomy
added and reused; questions and unique families by status, kind, concept, and
objective; objective x misconception x family coverage; source_scope and
unresolved source risks; family-review decisions; strict audit and exact
capacity results; package parity; focused and full tests; behavioral checks;
and any remaining independent-review or empirical-calibration work. Stop and
report a blocker instead of padding counts, weakening checks, inventing
provenance, mutating stable content in place, or accepting a weak item.
```
