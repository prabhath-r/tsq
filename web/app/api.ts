export type JsonPrimitive = string | number | boolean | null;
export type JsonValue = JsonPrimitive | JsonObject | JsonValue[];
export type JsonObject = { [key: string]: JsonValue };

export type SessionMode = "learn" | "diagnose" | "review";
export type SessionPhase =
  | "diagnose"
  | "learn"
  | "remediate"
  | "verify"
  | "review";
export type SessionStatus = "active" | "completed" | "abandoned";

export interface CorpusCounts {
  topics: number;
  concepts: number;
  learning_objectives: number;
  questions: number;
  active_questions: number;
  retired_questions: number;
  misconceptions: number;
  sources: number;
  active_families: number;
}

export interface Health {
  status: "ok";
  api_version: string;
  schema_version: number;
  corpus_release_id: string;
  corpus: CorpusCounts;
}

export interface Domain {
  id: string;
  name: string;
  description: string;
  sort_order: number;
}

export interface TopicConcept {
  id: string;
  name: string;
}

export interface CatalogTopic {
  id: string;
  domain_id: string;
  parent_id: string | null;
  name: string;
  description: string;
  related_topic_ids: string[];
  sort_order: number;
  concepts: TopicConcept[];
  direct_primary_questions: number;
  cross_topic_questions: number;
  depth: number;
  path: string[];
  direct_concepts: number;
  direct_learning_objectives: number;
  scope_primary_questions: number;
  scope_concepts: number;
  scope_learning_objectives: number;
}

export interface Catalog {
  release_id: string;
  counts: CorpusCounts;
  domains: Domain[];
  topics: CatalogTopic[];
}

export interface Topics {
  release_id: string;
  counts: CorpusCounts;
  topics: CatalogTopic[];
}

export interface LearningObjective {
  id: string;
  name: string;
  description: string;
  primary_concept_id?: string;
  supporting_concept_ids?: string[];
  operation: string;
  evidence_type: string;
}

export interface TopicDetail {
  release_id: string;
  topic: Omit<
    CatalogTopic,
    | "depth"
    | "path"
    | "direct_concepts"
    | "direct_learning_objectives"
    | "scope_primary_questions"
    | "scope_concepts"
    | "scope_learning_objectives"
  >;
  scope_concepts: Array<{
    id: string;
    name: string;
    description: string;
    domain: string;
  }>;
  learning_objectives: Array<
    LearningObjective & {
      primary_concept_id: string;
      supporting_concept_ids: string[];
    }
  >;
}

export interface Learner {
  id: string;
  display_name: string;
  revision: number;
  created_at: string;
}

export interface Session {
  id: string;
  learner_id: string;
  root_concept_id: string;
  corpus_release_id: string;
  mode: SessionMode;
  phase: SessionPhase;
  focus_concept_id: string | null;
  focus_misconception_id: string | null;
  focus_objective_id: string | null;
  remediation_depth: number;
  remediation_path: JsonValue[];
  revision: number;
  rng_seed: number;
  step: number;
  recent_families: string[];
  status: SessionStatus;
  created_at: string;
  updated_at: string;
  topic_id: string | null;
  exploration_mode: "adaptive" | "off" | string;
}

export interface SessionSummary {
  id: string;
  learner_id: string;
  learner_name: string;
  corpus_release_id: string;
  topic_id: string | null;
  root_concept_id: string;
  target_name: string;
  mode: SessionMode;
  phase: SessionPhase;
  status: SessionStatus;
  step: number;
  created_at: string;
  updated_at: string;
  questions_answered: number;
  correct: number;
  abstained: number;
  accuracy: number | null;
  selected_answers: number;
  selected_incorrect: number;
  selected_accuracy: number | null;
}

export interface SessionList {
  sessions: SessionSummary[];
  limit: number;
}

export interface SelectionScore {
  total: number;
  predicted_correct: number;
  information_gain: number;
  learning_fit: number;
  concept_need: number;
  misconception_value: number;
  prerequisite_value: number;
  review_value: number;
  novelty: number;
  kind_fit: number;
  continuity: number;
  boundary_fit: number;
  coverage_raw_exposures: number;
  coverage_diagnostic_information: number;
  coverage_successful_retrieval_families: number;
}

export interface Presentation {
  decision_id: string;
  session_id: string;
  phase: SessionPhase;
  question_id: string;
  family_id: string;
  kind: string;
  pedagogical_role: string;
  stem: string;
  options: Array<{ id: string; text: string }>;
  selection: {
    rationale: string;
    propensity: number;
    score: SelectionScore;
  };
  learning_objective?: Omit<
    LearningObjective,
    "primary_concept_id" | "supporting_concept_ids"
  >;
}

export interface Submission {
  interaction_id: string;
  correct: boolean;
  outcome: "correct" | "incorrect" | "abstained";
  selected_option_id: string | null;
  correct_option_id: string;
  selected_rationale: string | null;
  correct_rationale: string;
  next_phase: SessionPhase;
  focus_concept_id: string | null;
  focus_misconception_id: string | null;
  focus_objective_id: string | null;
  transition_reason: string;
  boundary_decision: JsonObject | null;
  state_changes: JsonObject[];
  idempotent_replay: boolean;
  learning_objective?: {
    id: string;
    name: string;
    state_change: JsonObject;
  };
  focus_learning_objective?: { id: string; name: string };
}

export interface ProfileSkill {
  concept_id: string;
  name: string;
  mastery: number;
  expected_competence: number;
  uncertainty: number;
  stability_hours: number;
  evidence_mass: number;
  projection_kind: string;
  derived_from_objective_ids: string[];
  objective_floor_source_id: string | null;
  independent_families: number;
  successful_retrieval_families: number;
  observed_response_families: number;
  delayed_retrievals: number;
  operation_kinds: number;
  prerequisites_ready: boolean;
  intrinsic_readiness: number;
  prerequisite_support: number;
  effective_readiness: number;
  bottleneck_concept_id: string | null;
  bottleneck_name: string | null;
  state: string;
  state_qualification: string;
  next_review_at: string | null;
}

export interface ProfileMisconception {
  misconception_id: string;
  name: string;
  probability: number;
  evidence_count: number;
  status: "active" | "monitoring";
}

export interface Profile {
  learner_id: string;
  corpus_release_id: string;
  target: { type: "topic" | "concept"; id: string; name: string } | null;
  boundary_algorithm_version: string;
  skills: ProfileSkill[];
  misconception_thresholds: { monitoring: number; active_routing: number };
  misconception_hypotheses: ProfileMisconception[];
  active_misconceptions: ProfileMisconception[];
  selected_response_inference: JsonObject;
  learning_objectives?: JsonObject[];
}

export type SessionReport = JsonObject & {
  session_id: string;
  learner_id: string;
  status: SessionStatus;
  mode: SessionMode;
  topic: { id: string; name: string } | null;
  root_concept_id: string;
  corpus_release_id: string;
  questions_presented: number;
  questions_answered: number;
  correct: number;
  accuracy: number | null;
  abstained: number;
  selected_answers: number;
  selected_incorrect: number;
  selected_accuracy: number | null;
};

export type DecisionTrace = JsonObject & {
  id: string;
  session_id: string;
  learner_id: string;
  question_id: string;
  question_objective_id: string | null;
  question_version: number;
  corpus_release_id: string;
  phase: SessionPhase;
  focus_concept_id: string | null;
  focus_misconception_id: string | null;
  focus_objective_id: string | null;
  pedagogical_role: string;
  policy_version: string;
  propensity: number;
  rationale: string;
  selected_score: JsonObject;
  top_candidates: JsonValue[];
  option_order: string[];
  created_at: string;
  consumed_at: string | null;
  invalidated_at: string | null;
  invalidation_reason: string | null;
};

export interface CreateLearnerInput {
  learner_id: string;
  display_name?: string;
}

export interface StartSessionInput {
  learner_id: string;
  topic_id?: string;
  root_concept_id?: string;
  explore_related?: boolean;
  mode?: SessionMode;
  seed?: number;
}

export interface AnswerInput {
  option_id: string | null;
  confidence?: number;
  response_ms?: number;
  hint_count?: number;
}

export interface EndSessionInput {
  status?: SessionStatus;
  completed?: boolean;
  reason?: string;
}

export interface SessionFilters {
  learner_id?: string;
  status?: SessionStatus;
  limit?: number;
}

export interface ApiErrorPayload {
  error: {
    code: string;
    message: string;
  };
}

export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly payload: unknown;

  constructor(
    status: number,
    code: string,
    message: string,
    payload?: unknown,
    options?: ErrorOptions,
  ) {
    super(message, options);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.payload = payload;
  }
}

export interface ApiClientOptions {
  baseUrl?: string;
  fetch?: typeof fetch;
}

interface RequestOptions {
  method?: "GET" | "POST";
  body?: object;
  idempotencyKey?: string;
  signal?: AbortSignal;
}

function configuredBaseUrl(): string {
  const viteUrl = (
    import.meta as ImportMeta & {
      env?: { VITE_TSQ_API_URL?: string };
    }
  ).env?.VITE_TSQ_API_URL;
  const nextUrl =
    typeof process === "undefined"
      ? undefined
      : process.env.NEXT_PUBLIC_TSQ_API_URL;
  return viteUrl || nextUrl || "/api/v1";
}

function normalizeBaseUrl(value: string): string {
  const trimmed = value.trim().replace(/\/+$/, "");
  if (!trimmed) {
    throw new Error("TSQ API base URL cannot be empty.");
  }
  return trimmed;
}

function isApiErrorPayload(value: unknown): value is ApiErrorPayload {
  if (typeof value !== "object" || value === null || !("error" in value)) {
    return false;
  }
  const error = value.error;
  return (
    typeof error === "object" &&
    error !== null &&
    "code" in error &&
    typeof error.code === "string" &&
    "message" in error &&
    typeof error.message === "string"
  );
}

function pathSegment(value: string): string {
  return encodeURIComponent(value);
}

function withQuery(path: string, values: Record<string, string | number | undefined>): string {
  const query = new URLSearchParams();
  for (const [key, value] of Object.entries(values)) {
    if (value !== undefined) {
      query.set(key, String(value));
    }
  }
  const suffix = query.toString();
  return suffix ? `${path}?${suffix}` : path;
}

export class ApiClient {
  readonly baseUrl: string;
  private readonly fetcher: typeof fetch;

  constructor(options: ApiClientOptions = {}) {
    this.baseUrl = normalizeBaseUrl(options.baseUrl ?? configuredBaseUrl());
    this.fetcher = options.fetch ?? globalThis.fetch.bind(globalThis);
  }

  private async request<T>(path: string, options: RequestOptions = {}): Promise<T> {
    const headers = new Headers({ Accept: "application/json" });
    if (options.body !== undefined) {
      headers.set("Content-Type", "application/json");
    }
    if (options.idempotencyKey !== undefined) {
      headers.set("Idempotency-Key", options.idempotencyKey);
    }

    let response: Response;
    try {
      response = await this.fetcher(`${this.baseUrl}${path}`, {
        method: options.method ?? "GET",
        headers,
        body: options.body === undefined ? undefined : JSON.stringify(options.body),
        cache: "no-store",
        credentials: "same-origin",
        signal: options.signal,
      });
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") {
        throw error;
      }
      throw new ApiError(
        0,
        "service_unavailable",
        "The local TSQ service is unavailable.",
        undefined,
        { cause: error },
      );
    }

    const text = await response.text();
    let payload: unknown = undefined;
    if (text) {
      try {
        payload = JSON.parse(text) as unknown;
      } catch (error) {
        throw new ApiError(
          response.status,
          "invalid_response",
          "The TSQ service returned an invalid JSON response.",
          text,
          { cause: error },
        );
      }
    }

    if (!response.ok) {
      if (isApiErrorPayload(payload)) {
        throw new ApiError(
          response.status,
          payload.error.code,
          payload.error.message,
          payload,
        );
      }
      throw new ApiError(
        response.status,
        "request_failed",
        `The TSQ service rejected the request (${response.status}).`,
        payload,
      );
    }
    return payload as T;
  }

  health(signal?: AbortSignal): Promise<Health> {
    return this.request<Health>("/health", { signal });
  }

  catalog(signal?: AbortSignal): Promise<Catalog> {
    return this.request<Catalog>("/catalog", { signal });
  }

  topics(signal?: AbortSignal): Promise<Topics> {
    return this.request<Topics>("/topics", { signal });
  }

  topic(reference: string, signal?: AbortSignal): Promise<TopicDetail> {
    return this.request<TopicDetail>(`/topics/${pathSegment(reference)}`, { signal });
  }

  createLearner(input: CreateLearnerInput, signal?: AbortSignal): Promise<Learner> {
    return this.request<Learner>("/learners", {
      method: "POST",
      body: input,
      signal,
    });
  }

  learner(learnerId: string, signal?: AbortSignal): Promise<Learner> {
    return this.request<Learner>(`/learners/${pathSegment(learnerId)}`, { signal });
  }

  profile(
    learnerId: string,
    topicId?: string,
    signal?: AbortSignal,
  ): Promise<Profile> {
    const path = withQuery(`/learners/${pathSegment(learnerId)}/profile`, {
      topic_id: topicId,
    });
    return this.request<Profile>(path, { signal });
  }

  sessions(filters: SessionFilters = {}, signal?: AbortSignal): Promise<SessionList> {
    const path = withQuery("/sessions", {
      learner_id: filters.learner_id,
      status: filters.status,
      limit: filters.limit,
    });
    return this.request<SessionList>(path, { signal });
  }

  session(sessionId: string, signal?: AbortSignal): Promise<Session> {
    return this.request<Session>(`/sessions/${pathSegment(sessionId)}`, { signal });
  }

  startSession(
    input: StartSessionInput,
    idempotencyKey: string,
    signal?: AbortSignal,
  ): Promise<Session> {
    return this.request<Session>("/sessions", {
      method: "POST",
      body: input,
      idempotencyKey,
      signal,
    });
  }

  nextQuestion(
    sessionId: string,
    idempotencyKey?: string,
    signal?: AbortSignal,
  ): Promise<Presentation> {
    return this.request<Presentation>(
      `/sessions/${pathSegment(sessionId)}/next`,
      {
        method: "POST",
        idempotencyKey,
        signal,
      },
    );
  }

  answer(
    decisionId: string,
    input: AnswerInput,
    idempotencyKey: string,
    signal?: AbortSignal,
  ): Promise<Submission> {
    return this.request<Submission>(
      `/decisions/${pathSegment(decisionId)}/answers`,
      {
        method: "POST",
        body: input,
        idempotencyKey,
        signal,
      },
    );
  }

  feedbackShown(
    decisionId: string,
    idempotencyKey: string,
    signal?: AbortSignal,
  ): Promise<JsonObject> {
    return this.request<JsonObject>(
      `/decisions/${pathSegment(decisionId)}/feedback`,
      {
        method: "POST",
        idempotencyKey,
        signal,
      },
    );
  }

  endSession(
    sessionId: string,
    input: EndSessionInput,
    idempotencyKey: string,
    signal?: AbortSignal,
  ): Promise<Session> {
    return this.request<Session>(`/sessions/${pathSegment(sessionId)}/end`, {
      method: "POST",
      body: input,
      idempotencyKey,
      signal,
    });
  }

  report(sessionId: string, signal?: AbortSignal): Promise<SessionReport> {
    return this.request<SessionReport>(
      `/sessions/${pathSegment(sessionId)}/report`,
      { signal },
    );
  }

  trace(sessionId: string, signal?: AbortSignal): Promise<DecisionTrace[]> {
    return this.request<DecisionTrace[]>(
      `/sessions/${pathSegment(sessionId)}/trace`,
      { signal },
    );
  }
}

export function createIdempotencyKey(scope: string): string {
  const uuid = globalThis.crypto?.randomUUID?.();
  return `${scope}:${uuid ?? `${Date.now()}:${Math.random().toString(36).slice(2)}`}`;
}

export const api = new ApiClient();
