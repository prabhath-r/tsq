"use client";

import {
  Activity,
  AlertCircle,
  ArrowRight,
  BarChart3,
  BookOpen,
  CheckCircle2,
  ChevronRight,
  CircleHelp,
  Database,
  FlaskConical,
  Gauge,
  History,
  Info,
  Layers3,
  Library,
  LoaderCircle,
  Menu,
  MessageSquare,
  Moon,
  Play,
  Plus,
  RefreshCw,
  Search,
  Settings,
  ShieldCheck,
  Sun,
  Target,
  TerminalSquare,
  X,
  Zap,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ApiError,
  api,
  createIdempotencyKey,
  type AnswerInput,
  type Catalog,
  type CatalogTopic,
  type Health,
  type Learner,
  type Presentation,
  type Profile,
  type Session,
  type SessionMode,
  type SessionReport,
  type SessionSummary,
  type Submission,
  type TopicDetail,
} from "./api";
import type { ViewId } from "./data";

const LEARNER_ID = "me";
const topicTones = ["teal", "blue", "violet", "amber", "rose"] as const;
const confidenceChoices = [
  { value: 0.2, label: "Guessing" },
  { value: 0.4, label: "Unsure" },
  { value: 0.6, label: "Somewhat sure" },
  { value: 0.8, label: "Confident" },
  { value: 0.95, label: "Very confident" },
  { value: "omit", label: "Prefer not to report" },
];

type SessionContext = { id: string; name: string };
type ConfidenceChoice = number | "omit";

type StudyState =
  | { kind: "idle" }
  | { kind: "loading"; message: string }
  | { kind: "question"; session: Session; topic: SessionContext; presentation: Presentation }
  | { kind: "feedback"; session: Session; topic: SessionContext; presentation: Presentation; submission: Submission }
  | { kind: "complete"; session: Session; report: SessionReport }
  | { kind: "exhausted"; session: Session; topic: SessionContext; message: string };

type Theme = "light" | "dark";

function errorMessage(error: unknown) {
  if (error instanceof ApiError) return error.message;
  if (error instanceof Error) return error.message;
  return "The request could not be completed.";
}

function percent(value: number | null | undefined) {
  return value == null ? "—" : `${Math.round(value * 100)}%`;
}

function readable(value: string) {
  return value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function initials(name: string) {
  return name.split(/\s+/).filter(Boolean).slice(0, 2).map((part) => part[0]).join("").toUpperCase() || "T";
}

function formatWhen(value: string) {
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? value
    : new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" }).format(date);
}

function Badge({ children, tone = "neutral" }: { children: React.ReactNode; tone?: "neutral" | "teal" | "green" | "amber" | "blue" | "red" | "violet" }) {
  return <span className={`badge badge-${tone}`}>{children}</span>;
}

function BoundaryNote({ children, compact = false }: { children: React.ReactNode; compact?: boolean }) {
  return <div className={`boundary-note ${compact ? "boundary-note-compact" : ""}`}><Info size={16} aria-hidden="true" /><span>{children}</span></div>;
}

function EmptyAction({ icon: Icon, title, copy, action, onAction }: { icon: typeof History; title: string; copy: string; action: string; onAction: () => void }) {
  return <section className="empty-action"><span className="empty-icon"><Icon size={22} /></span><h3>{title}</h3><p>{copy}</p><button className="button button-secondary" type="button" onClick={onAction}>{action}</button></section>;
}

function PageHeader({ eyebrow, title, copy, actions }: { eyebrow: string; title: string; copy: string; actions?: React.ReactNode }) {
  return <div className="page-header"><div><p className="eyebrow">{eyebrow}</p><h1>{title}</h1><p>{copy}</p></div>{actions && <div className="page-actions">{actions}</div>}</div>;
}

function MetricCard({ label, value, note, icon: Icon, tone }: { label: string; value: string; note: string; icon: typeof Activity; tone: "teal" | "blue" | "violet" | "amber" }) {
  return <div className="metric-card"><span className={`metric-icon metric-${tone}`}><Icon size={18} /></span><div><p>{label}</p><strong>{value}</strong><small>{note}</small></div></div>;
}

export default function Home() {
  const [view, setView] = useState<ViewId>("study");
  const [theme, setTheme] = useState<Theme>("light");
  const [mobileNavOpen, setMobileNavOpen] = useState(false);
  const [health, setHealth] = useState<Health | null>(null);
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [learner, setLearner] = useState<Learner | null>(null);
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [profile, setProfile] = useState<Profile | null>(null);
  const [bootstrapLoading, setBootstrapLoading] = useState(true);
  const [connectionError, setConnectionError] = useState("");
  const [study, setStudy] = useState<StudyState>({ kind: "idle" });
  const [mode, setMode] = useState<SessionMode>("learn");
  const [selectedTopicId, setSelectedTopicId] = useState<string | null>(null);
  const [toast, setToast] = useState("");
  const answerCommands = useRef(new Map<string, { input: AnswerInput; key: string }>());
  const startKeys = useRef(new Map<string, string>());
  const feedbackKeys = useRef(new Map<string, string>());
  const endKeys = useRef(new Map<string, string>());

  const showToast = useCallback((message: string) => {
    setToast(message);
    window.setTimeout(() => setToast(""), 2800);
  }, []);

  const refreshLearnerData = useCallback(async (knownLearner?: Learner | null) => {
    let resolved = knownLearner;
    if (resolved === undefined) {
      try {
        resolved = await api.learner(LEARNER_ID);
      } catch (error) {
        if (!(error instanceof ApiError && error.status === 404)) throw error;
        resolved = null;
      }
    }
    setLearner(resolved ?? null);
    const list = await api.sessions({ learner_id: LEARNER_ID, limit: 100 });
    setSessions(list.sessions);
    if (resolved) {
      setProfile(await api.profile(LEARNER_ID));
    } else {
      setProfile(null);
    }
  }, []);

  const bootstrap = useCallback(async () => {
    setBootstrapLoading(true);
    setConnectionError("");
    try {
      const [nextHealth, nextCatalog] = await Promise.all([api.health(), api.catalog()]);
      if (nextHealth.corpus_release_id !== nextCatalog.release_id) {
        throw new Error("The API health and catalog release identifiers do not match.");
      }
      setHealth(nextHealth);
      setCatalog(nextCatalog);
      setSelectedTopicId((current) => current && nextCatalog.topics.some((topic) => topic.id === current)
        ? current
        : nextCatalog.topics.find((topic) => topic.scope_primary_questions > 0)?.id ?? null);
      await refreshLearnerData();
    } catch (error) {
      setConnectionError(errorMessage(error));
    } finally {
      setBootstrapLoading(false);
    }
  }, [refreshLearnerData]);

  useEffect(() => {
    const timer = window.setTimeout(() => void bootstrap(), 0);
    return () => window.clearTimeout(timer);
  }, [bootstrap]);
  useEffect(() => {
    const frame = window.requestAnimationFrame(() => {
      const stored = window.localStorage.getItem("tsq-theme");
      const next = stored === "dark" ? "dark" : "light";
      setTheme(next);
      document.documentElement.dataset.theme = next;
    });
    return () => window.cancelAnimationFrame(frame);
  }, []);

  const changeTheme = () => {
    const next = theme === "light" ? "dark" : "light";
    setTheme(next);
    document.documentElement.dataset.theme = next;
    window.localStorage.setItem("tsq-theme", next);
  };

  const navigate = (next: ViewId) => {
    setView(next);
    setMobileNavOpen(false);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const startTopic = useCallback(async (topic: CatalogTopic) => {
    setView("study");
    setStudy({ kind: "loading", message: `Starting ${topic.name}…` });
    setConnectionError("");
    try {
      const ensuredLearner = await api.createLearner({ learner_id: LEARNER_ID });
      setLearner(ensuredLearner);
      const startScope = `${topic.id}:${mode}`;
      let startKey = startKeys.current.get(startScope);
      if (!startKey) {
        startKey = createIdempotencyKey(`web:session:${startScope}`);
        startKeys.current.set(startScope, startKey);
      }
      const startedSession = await api.startSession(
        { learner_id: LEARNER_ID, topic_id: topic.id, mode, explore_related: true },
        startKey,
      );
      const presentation = await api.nextQuestion(startedSession.id, createIdempotencyKey(`web:next:${startedSession.id}`));
      const session = await api.session(startedSession.id);
      setStudy({ kind: "question", session, topic, presentation });
      await refreshLearnerData(ensuredLearner);
      startKeys.current.delete(startScope);
    } catch (error) {
      setStudy({ kind: "idle" });
      setConnectionError(errorMessage(error));
    }
  }, [mode, refreshLearnerData]);

  const resumeSession = useCallback(async (summary: SessionSummary) => {
    if (summary.status !== "active") {
      setStudy({ kind: "loading", message: "Loading session report…" });
      setView("study");
      try {
        const [session, report] = await Promise.all([api.session(summary.id), api.report(summary.id)]);
        setStudy({ kind: "complete", session, report });
      } catch (error) {
        setConnectionError(errorMessage(error));
        setStudy({ kind: "idle" });
      }
      return;
    }
    const topic: SessionContext = {
      id: summary.topic_id ?? summary.root_concept_id,
      name: summary.target_name,
    };
    setView("study");
    setStudy({ kind: "loading", message: `Resuming ${topic.name}…` });
    try {
      const pendingSession = await api.session(summary.id);
      const presentation = await api.nextQuestion(pendingSession.id, createIdempotencyKey(`web:next:${pendingSession.id}`));
      const session = await api.session(pendingSession.id);
      setStudy({ kind: "question", session, topic, presentation });
    } catch (error) {
      if (error instanceof ApiError && error.code === "corpus_exhausted") {
        const session = await api.session(summary.id);
        setStudy({ kind: "exhausted", session, topic, message: error.message });
      } else {
        setConnectionError(errorMessage(error));
        setStudy({ kind: "idle" });
      }
    }
  }, []);

  const answerCurrent = useCallback(async (optionId: string | null, confidence: number | null, responseMs: number) => {
    if (study.kind !== "question") throw new Error("There is no pending question to answer.");
    const decisionId = study.presentation.decision_id;
    let command = answerCommands.current.get(decisionId);
    if (!command) {
      command = {
        input: {
          option_id: optionId,
          confidence: optionId === null ? undefined : confidence ?? undefined,
          response_ms: responseMs,
          hint_count: 0,
        },
        key: createIdempotencyKey(`web:answer:${decisionId}`),
      };
      answerCommands.current.set(decisionId, command);
    }
    const submission = await api.answer(decisionId, command.input, command.key);
    setStudy({ ...study, kind: "feedback", submission });
    void refreshLearnerData(learner).catch((error) => setConnectionError(errorMessage(error)));
  }, [learner, refreshLearnerData, study]);

  const acknowledgeFeedback = useCallback(async (decisionId: string) => {
    let key = feedbackKeys.current.get(decisionId);
    if (!key) {
      key = createIdempotencyKey(`web:feedback:${decisionId}`);
      feedbackKeys.current.set(decisionId, key);
    }
    try {
      await api.feedbackShown(decisionId, key);
    } catch (error) {
      showToast(`Feedback is visible, but its activity receipt was not saved: ${errorMessage(error)}`);
      throw error;
    }
  }, [showToast]);

  const nextQuestion = useCallback(async () => {
    if (study.kind !== "feedback") return;
    const { session, topic } = study;
    setStudy({ kind: "loading", message: "Choosing the next safe question…" });
    try {
      const presentation = await api.nextQuestion(session.id, createIdempotencyKey(`web:next:${session.id}`));
      const currentSession = await api.session(session.id);
      setStudy({ kind: "question", session: currentSession, topic, presentation });
    } catch (error) {
      if (error instanceof ApiError && error.code === "corpus_exhausted") {
        setStudy({ kind: "exhausted", session, topic, message: error.message });
      } else {
        setConnectionError(errorMessage(error));
        setStudy({ kind: "feedback", session, topic, presentation: study.presentation, submission: study.submission });
      }
    }
  }, [study]);

  const finishSession = useCallback(async (status: "completed" | "abandoned" = "completed") => {
    if (!("session" in study)) return;
    const priorStudy = study;
    const session = study.session;
    setStudy({ kind: "loading", message: status === "completed" ? "Finishing session…" : "Saving and closing session…" });
    let key = endKeys.current.get(`${session.id}:${status}`);
    if (!key) {
      key = createIdempotencyKey(`web:end:${session.id}:${status}`);
      endKeys.current.set(`${session.id}:${status}`, key);
    }
    try {
      const ended = await api.endSession(session.id, { status, reason: status === "completed" ? "web_session_complete" : "web_session_abandoned" }, key);
      const report = await api.report(session.id);
      setStudy({ kind: "complete", session: ended, report });
      void refreshLearnerData(learner).catch((refreshError) => setConnectionError(errorMessage(refreshError)));
    } catch (error) {
      let reconciled: Session;
      try {
        reconciled = await api.session(session.id);
      } catch (reconcileError) {
        setConnectionError(`${errorMessage(error)} The terminal outcome could not be confirmed: ${errorMessage(reconcileError)}. Your in-memory turn is preserved and retry will use the same key.`);
        setStudy(priorStudy);
        return;
      }
      if (reconciled.status === "active") {
        setConnectionError(`${errorMessage(error)} The session is still active; retry will use the same end-session command.`);
        setStudy(priorStudy);
        return;
      }
      try {
        const report = await api.report(session.id);
        setStudy({ kind: "complete", session: reconciled, report });
        void refreshLearnerData(learner).catch((refreshError) => setConnectionError(errorMessage(refreshError)));
      } catch (reportError) {
        setConnectionError(`The session is ${reconciled.status}, but its report could not be loaded: ${errorMessage(reportError)}. Open it again from Sessions to retry the report.`);
        setStudy({ kind: "idle" });
      }
    }
  }, [learner, refreshLearnerData, study]);

  const startableTopics = catalog?.topics.filter((topic) => topic.scope_primary_questions > 0) ?? [];
  const activeSessions = sessions.filter((session) => session.status === "active");

  return <div className="app-shell">
    <a className="skip-link" href="#main-content">Skip to main content</a>
    <aside className={`sidebar ${mobileNavOpen ? "sidebar-open" : ""}`} aria-label="Primary navigation">
      <div className="sidebar-top"><button className="brand" type="button" onClick={() => navigate("study")}><span className="brand-mark"><span>T</span></span><span className="brand-copy"><strong>TSQ</strong><small>The Second Question</small></span></button><button className="icon-button sidebar-close" type="button" aria-label="Close navigation" onClick={() => setMobileNavOpen(false)}><X size={20} /></button></div>
      <button className="new-session-button" type="button" onClick={() => navigate("topics")}><Plus size={17} /><span>New session</span></button>
      <nav className="nav-list"><p className="nav-label">Workspace</p>{[
        ["study", "Study", MessageSquare], ["topics", "Topics", Library], ["progress", "Progress", BarChart3], ["sessions", "Sessions", History], ["labs", "Labs", FlaskConical],
      ].map(([id, label, Icon]) => <button key={String(id)} type="button" className={`nav-item ${view === id ? "nav-item-active" : ""}`} aria-current={view === id ? "page" : undefined} onClick={() => navigate(id as ViewId)}><Icon size={18} /><span>{String(label)}</span>{id === "sessions" && activeSessions.length > 0 && <Badge tone="teal">{activeSessions.length}</Badge>}</button>)}</nav>
      <div className="sidebar-recents"><div className="sidebar-section-head"><p className="nav-label">Recent</p><button type="button" onClick={() => navigate("sessions")}>View all</button></div>{sessions.length ? sessions.slice(0, 3).map((session) => <button className="sidebar-empty" type="button" key={session.id} onClick={() => void resumeSession(session)}><History size={15} /><span><strong>{session.target_name}</strong><small>{session.status === "active" ? "Resume" : `${session.questions_answered} responses`}</small></span></button>) : <div className="sidebar-empty"><History size={15} /><span><strong>No sessions yet</strong><small>Start from any topic</small></span></div>}</div>
      <div className="sidebar-bottom"><button className={`nav-item ${view === "operations" ? "nav-item-active" : ""}`} type="button" onClick={() => navigate("operations")}><TerminalSquare size={18} /><span>Operations</span></button><button className={`nav-item ${view === "settings" ? "nav-item-active" : ""}`} type="button" onClick={() => navigate("settings")}><Settings size={18} /><span>Settings</span></button><div className="profile-chip"><span className="avatar">{learner ? "ME" : "G"}</span><span><strong>{learner?.display_name || "Guest"}</strong><small>{learner ? "Local learner · tsq.db" : "Created on first session"}</small></span><span className={`status-dot ${health ? "status-dot-active" : ""}`} /></div></div>
    </aside>
    {mobileNavOpen && <button className="nav-scrim" aria-label="Close navigation" onClick={() => setMobileNavOpen(false)} />}
    <div className="app-main">
      <header className="topbar"><div className="topbar-left"><button className="icon-button mobile-menu" type="button" aria-label="Open navigation" onClick={() => setMobileNavOpen(true)}><Menu size={20} /></button><div className="mobile-brand"><span className="brand-mark brand-mark-small">T</span><strong>TSQ</strong></div></div><button className="command-trigger" type="button" onClick={() => navigate("topics")}><Search size={16} /><span>Search the curriculum…</span></button><div className="topbar-actions">{health ? <Badge tone="green"><CheckCircle2 size={13} />Live · same tsq.db</Badge> : <Badge tone="amber">Connecting</Badge>}<button className="icon-button" type="button" aria-label="Toggle theme" onClick={changeTheme}>{theme === "light" ? <Moon size={18} /> : <Sun size={18} />}</button></div></header>
      <main className="content" id="main-content" tabIndex={-1}>
        {connectionError && <div className="boundary-note boundary-error"><AlertCircle size={17} /><span><strong>TSQ service:</strong> {connectionError}</span><button className="text-button" type="button" onClick={() => void bootstrap()}><RefreshCw size={14} />Retry</button></div>}
        {bootstrapLoading && !catalog ? <LoadingScreen message="Opening the shared TSQ database and exact corpus…" /> : <>
          {view === "study" && <StudyWorkspace study={study} catalog={catalog} sessions={sessions} mode={mode} setMode={setMode} startableTopics={startableTopics} resumeSession={resumeSession} answerCurrent={answerCurrent} acknowledgeFeedback={acknowledgeFeedback} nextQuestion={nextQuestion} finishSession={finishSession} browseTopics={() => navigate("topics")} />}
          {view === "topics" && <TopicsWorkspace catalog={catalog} selectedTopicId={selectedTopicId} setSelectedTopicId={setSelectedTopicId} mode={mode} setMode={setMode} startTopic={startTopic} />}
          {view === "progress" && <ProgressWorkspace learner={learner} profile={profile} objectiveCount={health?.corpus.learning_objectives ?? 0} start={() => navigate("topics")} />}
          {view === "sessions" && <SessionsWorkspace sessions={sessions} resumeSession={resumeSession} start={() => navigate("topics")} />}
          {view === "labs" && <LabsWorkspace />}
          {view === "operations" && <OperationsWorkspace health={health} catalog={catalog} learner={learner} sessions={sessions} />}
          {view === "settings" && <SettingsWorkspace health={health} learner={learner} theme={theme} changeTheme={changeTheme} refresh={() => void bootstrap()} />}
        </>}
      </main>
    </div>
    <div className="toast-region" aria-live="polite">{toast && <div className="toast"><CheckCircle2 size={17} />{toast}</div>}</div>
  </div>;
}

function LoadingScreen({ message }: { message: string }) {
  return <div className="fresh-study"><section className="fresh-hero"><div className="fresh-hero-copy"><span className="fresh-release-icon"><LoaderCircle className="spin" size={24} /></span><p className="eyebrow">Local engine</p><h1>{message}</h1><p>TSQ is preserving the same release, learner projections, pending decisions, and event ledger used by the CLI.</p></div></section></div>;
}

function StudyWorkspace({ study, catalog, sessions, mode, setMode, startableTopics, resumeSession, answerCurrent, acknowledgeFeedback, nextQuestion, finishSession, browseTopics }: {
  study: StudyState; catalog: Catalog | null; sessions: SessionSummary[]; mode: SessionMode; setMode: (mode: SessionMode) => void; startableTopics: CatalogTopic[]; resumeSession: (session: SessionSummary) => Promise<void>; answerCurrent: (option: string | null, confidence: number | null, responseMs: number) => Promise<void>; acknowledgeFeedback: (decisionId: string) => Promise<void>; nextQuestion: () => Promise<void>; finishSession: (status?: "completed" | "abandoned") => Promise<void>; browseTopics: () => void;
}) {
  if (study.kind === "loading") return <LoadingScreen message={study.message} />;
  if (study.kind === "question") return <QuestionWorkspace state={study} answerCurrent={answerCurrent} finishSession={finishSession} />;
  if (study.kind === "feedback") return <FeedbackWorkspace state={study} acknowledgeFeedback={acknowledgeFeedback} nextQuestion={nextQuestion} finishSession={finishSession} />;
  if (study.kind === "complete") return <SessionReportWorkspace report={study.report} browseTopics={browseTopics} />;
  if (study.kind === "exhausted") return <div className="page-shell"><PageHeader eyebrow="Safe boundary" title="This session has reached its current serviceable boundary." copy={study.message} /><BoundaryNote><strong>No fallback question was guessed.</strong> TSQ persisted the corpus gap and stopped at the same boundary as the CLI.</BoundaryNote><button className="button button-primary" type="button" onClick={() => void finishSession("completed")}>Finish and view report</button></div>;
  const active = sessions.find((session) => session.status === "active");
  return <div className="fresh-study"><section className="fresh-hero"><div className="fresh-hero-copy"><Badge tone="green">Connected to the real engine</Badge><p className="eyebrow">The Second Question</p><h1>Study the exact released curriculum.</h1><p>Every selection, answer, skip, transition, and learner update is executed by the same Python engine and written to the same <code>tsq.db</code> as the CLI.</p><div className="mode-selector" role="group" aria-label="Session mode">{(["learn", "diagnose", "review"] as const).map((item) => <button key={item} type="button" className={mode === item ? "mode-selected" : ""} aria-pressed={mode === item} onClick={() => setMode(item)}>{readable(item)}</button>)}</div><div className="fresh-actions">{active ? <button className="button button-primary" type="button" onClick={() => void resumeSession(active)}><Play size={17} />Resume {active.target_name}</button> : <button className="button button-primary" type="button" onClick={browseTopics}><Library size={17} />Choose a topic</button>}<button className="button button-secondary" type="button" onClick={browseTopics}>Browse all {catalog?.topics.length ?? 0} topics</button></div><p className="fresh-boundary"><ShieldCheck size={15} />Correct answers and rationales stay server-side until you submit. Retired questions remain in release lineage but are never selected.</p></div><aside className="fresh-release-card"><div className="fresh-release-icon"><BookOpen size={22} /></div><p className="eyebrow">Active corpus</p><h2>{catalog?.counts.active_questions ?? 0} live questions</h2><p>Across {catalog?.topics.length ?? 0} topics. Root topics include every approved descendant question in their sealed scope.</p><div className="fresh-release-stats"><span><strong>{catalog?.counts.questions ?? 0}</strong><small>Release records</small></span><span><strong>{startableTopics.length}</strong><small>Startable topics</small></span></div></aside></section></div>;
}

function QuestionWorkspace({ state, answerCurrent, finishSession }: { state: Extract<StudyState, { kind: "question" }>; answerCurrent: (option: string | null, confidence: number | null, responseMs: number) => Promise<void>; finishSession: (status?: "completed" | "abandoned") => Promise<void> }) {
  const [selected, setSelected] = useState<string | null | undefined>(undefined);
  const [confidence, setConfidence] = useState<ConfidenceChoice | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [commandLocked, setCommandLocked] = useState(false);
  const [error, setError] = useState("");
  const startedAt = useRef(0);
  const titleRef = useRef<HTMLHeadingElement>(null);
  useEffect(() => { startedAt.current = performance.now(); titleRef.current?.focus(); }, [state.presentation.decision_id]);
  const ready = selected === null || (typeof selected === "string" && confidence !== null);
  const submit = useCallback(async () => {
    if (!ready || submitting) return;
    setCommandLocked(true); setSubmitting(true); setError("");
    try {
      await answerCurrent(selected ?? null, typeof confidence === "number" ? confidence : null, Math.max(0, Math.floor(performance.now() - startedAt.current)));
    } catch (cause) {
      setError(errorMessage(cause)); setSubmitting(false);
    }
  }, [answerCurrent, confidence, ready, selected, submitting]);
  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement) return;
      if (commandLocked) return;
      if (["1", "2", "3", "4"].includes(event.key)) { setSelected(state.presentation.options[Number(event.key) - 1]?.id); setConfidence(null); }
      if (event.key.toLowerCase() === "i") { setSelected(null); setConfidence(null); }
      if ((event.metaKey || event.ctrlKey) && event.key === "Enter") { event.preventDefault(); void submit(); }
    };
    window.addEventListener("keydown", handler); return () => window.removeEventListener("keydown", handler);
  }, [commandLocked, state.presentation.options, submit]);
  const objective = state.presentation.learning_objective;
  return <div className="study-layout"><section className="study-column"><div className="study-header"><div><div className="eyebrow-row"><span>{state.topic.name}</span><ChevronRight size={13} /><span>{readable(state.presentation.phase)}</span></div><h1>{objective?.name || readable(state.presentation.pedagogical_role)}</h1></div><button className="button button-quiet" type="button" onClick={() => void finishSession("abandoned")}>Save & exit</button></div><BoundaryNote compact><strong>Live session.</strong> Decision {state.presentation.decision_id.slice(0, 10)}… is pinned to release {state.session.corpus_release_id.slice(0, 12)}…</BoundaryNote><article className="question-card"><div className="question-meta"><div><Badge tone="teal">{readable(state.presentation.pedagogical_role)}</Badge><Badge>{readable(state.presentation.kind)}</Badge></div><span>Question {state.session.step}</span></div>{objective && <p className="objective-label"><Target size={15} />Objective · {objective.name}</p>}<h2 ref={titleRef} tabIndex={-1}>{state.presentation.stem}</h2><fieldset className="option-fieldset" disabled={submitting || commandLocked}><legend className="sr-only">Choose one answer</legend>{state.presentation.options.map((option, index) => <label className={`option-row ${selected === option.id ? "option-selected" : ""}`} key={option.id}><input type="radio" name={state.presentation.decision_id} checked={selected === option.id} onChange={() => { setSelected(option.id); setConfidence(null); }} /><span className="option-key">{String.fromCharCode(65 + index)}</span><span className="option-text">{option.text}</span></label>)}<label className={`option-row skip-row ${selected === null ? "option-selected" : ""}`}><input type="radio" name={state.presentation.decision_id} checked={selected === null} onChange={() => { setSelected(null); setConfidence(null); }} /><span className="option-key"><CircleHelp size={16} /></span><span className="option-text"><strong>I don’t know yet</strong><small>Submit an explicit abstention, not an incorrect selected answer</small></span></label></fieldset>{typeof selected === "string" && <fieldset className="confidence-fieldset" disabled={submitting || commandLocked}><legend>How sure are you?</legend><p>Confidence is optional. Choose a value or explicitly prefer not to report it.</p><div className="confidence-choices">{confidenceChoices.map((choice) => <label key={choice.value} className={confidence === choice.value ? "confidence-selected" : ""}><input type="radio" name="confidence" checked={confidence === choice.value} onChange={() => setConfidence(choice.value as ConfidenceChoice)} /><span>{choice.label}</span><small>{typeof choice.value === "number" ? `${Math.round(choice.value * 100)}%` : "Optional"}</small></label>)}</div></fieldset>}{error && <div className="boundary-note boundary-error"><AlertCircle size={16} /><span>{error} Your submitted option, confidence choice, response time, and idempotency key are locked; retry resends that exact command.</span></div>}<details className="technical-disclosure"><summary>Why this question?</summary><p>{state.presentation.selection.rationale}</p>{objective && <p>{objective.description}</p>}</details><div className="question-actions sticky-action"><span>{selected === undefined ? "Choose an answer or I don’t know." : typeof selected === "string" && confidence === null ? "Choose confidence or prefer not to report it." : error ? "Retry will resend your exact original response." : "Your response is ready."}</span><button className="button button-primary" type="button" disabled={!ready || submitting} onClick={() => void submit()}>{submitting ? <><LoaderCircle className="spin" size={17} />Saving…</> : error ? <>Retry exact answer <RefreshCw size={17} /></> : <>Submit answer <ArrowRight size={17} /></>}</button></div></article></section><aside className="context-rail"><p className="eyebrow">Session context</p><h2>{state.topic.name}</h2><div className="detail-stat-row"><span>Mode</span><strong>{readable(state.session.mode)}</strong></div><div className="detail-stat-row"><span>Phase</span><strong>{readable(state.presentation.phase)}</strong></div><div className="detail-stat-row"><span>Family</span><code>{state.presentation.family_id}</code></div><div className="detail-stat-row"><span>Objective</span><strong>{objective?.name || "Concept-level"}</strong></div><BoundaryNote compact>Family labels are independence units. Repeated aliases do not become new evidence.</BoundaryNote></aside></div>;
}

function FeedbackWorkspace({ state, acknowledgeFeedback, nextQuestion, finishSession }: { state: Extract<StudyState, { kind: "feedback" }>; acknowledgeFeedback: (decisionId: string) => Promise<void>; nextQuestion: () => Promise<void>; finishSession: (status?: "completed" | "abandoned") => Promise<void> }) {
  const [receiptStatus, setReceiptStatus] = useState<"saving" | "saved" | "error">("saving");
  const receiptStarted = useRef(false);
  const persistReceipt = useCallback(async () => {
    setReceiptStatus("saving");
    try {
      await acknowledgeFeedback(state.presentation.decision_id);
      setReceiptStatus("saved");
      return true;
    } catch {
      setReceiptStatus("error");
      return false;
    }
  }, [acknowledgeFeedback, state.presentation.decision_id]);
  useEffect(() => {
    if (!receiptStarted.current) {
      receiptStarted.current = true;
      void persistReceipt();
    }
  }, [persistReceipt]);
  const afterReceipt = async (action: () => Promise<void>) => {
    const saved = receiptStatus === "saved" || await persistReceipt();
    if (saved) await action();
  };
  const correctText = state.presentation.options.find((option) => option.id === state.submission.correct_option_id)?.text;
  const selectedText = state.presentation.options.find((option) => option.id === state.submission.selected_option_id)?.text;
  const skipped = state.submission.outcome === "abstained";
  return <div className="study-layout"><section className="study-column"><div className="study-header"><div><div className="eyebrow-row"><span>{state.topic.name}</span><ChevronRight size={13} /><span>Feedback</span></div><h1>{state.presentation.learning_objective?.name || "Response feedback"}</h1></div></div><article className="question-card"><p className="objective-label"><Target size={15} />{state.presentation.stem}</p><div className={`feedback ${skipped ? "feedback-skip" : state.submission.correct ? "feedback-correct" : "feedback-incorrect"}`} aria-live="polite"><div className="feedback-heading"><span>{skipped ? <CircleHelp size={21} /> : state.submission.correct ? <CheckCircle2 size={21} /> : <AlertCircle size={21} />}</span><div><h2>{skipped ? "Skipped — no answer selected." : state.submission.correct ? "Correct." : "Not quite."}</h2><p>{skipped ? "The abstention was saved separately from an incorrect selected answer." : state.submission.selected_rationale}</p></div></div>{selectedText && !state.submission.correct && <div className="your-answer"><span>Your answer</span><p>{selectedText}</p></div>}<div className="explanation-block"><p className="mini-label">Best answer · {state.submission.correct_option_id.toUpperCase()}</p><p className="best-answer">{correctText}</p><p>{state.submission.correct_rationale}</p></div><div className="transition-card"><span className="transition-icon"><Zap size={16} /></span><div><strong>Next phase · {readable(state.submission.next_phase)}</strong><p>{state.submission.transition_reason}</p></div></div>{receiptStatus === "error" && <div className="boundary-note boundary-error"><AlertCircle size={16} /><span>Feedback remains visible, but its activity receipt was not saved. Retry either action below; the same idempotency key will be used.</span></div>}<div className="feedback-actions"><button className="button button-secondary" type="button" disabled={receiptStatus === "saving"} onClick={() => void afterReceipt(() => finishSession("completed"))}>{receiptStatus === "saving" ? "Saving receipt…" : receiptStatus === "error" ? "Retry & finish" : "Finish session"}</button><button className="button button-primary" type="button" disabled={receiptStatus === "saving"} onClick={() => void afterReceipt(nextQuestion)}>{receiptStatus === "saving" ? "Saving receipt…" : receiptStatus === "error" ? "Retry & continue" : "Continue"} {receiptStatus !== "saving" && <ArrowRight size={17} />}</button></div></div></article></section><aside className="context-rail"><p className="eyebrow">Evidence update</p><h2>{state.submission.outcome === "abstained" ? "Abstention recorded" : "Selected response recorded"}</h2><div className="detail-stat-row"><span>Outcome</span><strong>{readable(state.submission.outcome)}</strong></div><div className="detail-stat-row"><span>Next phase</span><strong>{readable(state.submission.next_phase)}</strong></div><div className="detail-stat-row"><span>State changes</span><strong>{state.submission.state_changes.length}</strong></div><BoundaryNote compact>Selected-response estimates are provisional. They do not certify productive skill or real-world performance.</BoundaryNote></aside></div>;
}

function SessionReportWorkspace({ report, browseTopics }: { report: SessionReport; browseTopics: () => void }) {
  return <div className="page-shell"><PageHeader eyebrow="Persisted session report" title={report.topic?.name || "Session complete"} copy="This summary is reconstructed from the same durable ledger used by the CLI." actions={<Badge tone="green"><CheckCircle2 size={13} />{readable(report.status)}</Badge>} /><div className="metric-grid"><MetricCard label="Responses" value={String(report.questions_answered)} note={`${report.questions_presented} presented`} icon={MessageSquare} tone="teal" /><MetricCard label="Selected answers" value={String(report.selected_answers)} note={`${report.selected_incorrect} selected incorrect`} icon={Target} tone="blue" /><MetricCard label="Selected accuracy" value={percent(report.selected_accuracy)} note="Skips excluded" icon={Gauge} tone="violet" /><MetricCard label="Abstained" value={String(report.abstained)} note="Explicitly separated" icon={CircleHelp} tone="amber" /></div><BoundaryNote><strong>Immutable release pin:</strong> {report.corpus_release_id}. This report never merges skipped responses into selected-answer accuracy.</BoundaryNote><button className="button button-primary" type="button" onClick={browseTopics}>Start another topic <ArrowRight size={16} /></button></div>;
}

function TopicsWorkspace({ catalog, selectedTopicId, setSelectedTopicId, mode, setMode, startTopic }: { catalog: Catalog | null; selectedTopicId: string | null; setSelectedTopicId: (id: string) => void; mode: SessionMode; setMode: (mode: SessionMode) => void; startTopic: (topic: CatalogTopic) => Promise<void> }) {
  const [query, setQuery] = useState("");
  const [rootFilter, setRootFilter] = useState<string>("all");
  const [detail, setDetail] = useState<TopicDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const topics = useMemo(() => catalog?.topics ?? [], [catalog]);
  const roots = topics.filter((topic) => topic.parent_id === null);
  const parentNames = useMemo(() => new Map(topics.map((topic) => [topic.id, topic.name])), [topics]);
  const underRoot = useCallback((topic: CatalogTopic, rootId: string) => {
    let current: CatalogTopic | undefined = topic;
    while (current) { if (current.id === rootId) return true; current = current.parent_id ? topics.find((item) => item.id === current?.parent_id) : undefined; }
    return false;
  }, [topics]);
  const visible = topics.filter((topic) => {
    const matches = `${topic.name} ${topic.description} ${topic.path.join(" ")}`.toLowerCase().includes(query.toLowerCase());
    return matches && (rootFilter === "all" || underRoot(topic, rootFilter));
  });
  const selected = visible.find((topic) => topic.id === selectedTopicId) ?? visible[0] ?? null;
  const selectedId = selected?.id;
  useEffect(() => {
    if (!selectedId) return;
    const controller = new AbortController();
    Promise.resolve().then(() => {
      setDetailLoading(true);
      return api.topic(selectedId, controller.signal);
    }).then(setDetail).catch(() => setDetail(null)).finally(() => setDetailLoading(false));
    return () => controller.abort();
  }, [selectedId]);
  return <div className="page-shell"><PageHeader eyebrow="Exact active catalog" title="Choose any curriculum topic" copy="Counts, hierarchy, concepts, and objectives are loaded from the active corpus release—not copied into the frontend." /><div className="page-toolbar"><label className="search-field"><Search size={17} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder={`Search ${topics.length} topics`} /></label><div className="filter-pills"><button className={`filter-pill ${rootFilter === "all" ? "filter-pill-active" : ""}`} type="button" onClick={() => setRootFilter("all")}>All topics</button>{roots.map((root) => <button className={`filter-pill ${rootFilter === root.id ? "filter-pill-active" : ""}`} type="button" key={root.id} onClick={() => setRootFilter(root.id)}>{root.name}</button>)}</div></div><div className="topics-layout"><section className="topic-grid">{visible.map((topic, index) => { const tone = topicTones[index % topicTones.length]; return <button className={`topic-card ${selected?.id === topic.id ? "topic-card-selected" : ""}`} type="button" key={topic.id} onClick={() => setSelectedTopicId(topic.id)}><span className={`topic-icon topic-${tone}`}>{initials(topic.name)}</span><span className="topic-card-top"><span>{topic.parent_id ? parentNames.get(topic.parent_id) : "Curriculum root"}</span><ChevronRight size={16} /></span><strong>{topic.name}</strong><span className="topic-description">{topic.description}</span><span className="topic-stats"><span>{topic.scope_primary_questions} live questions</span>{topic.scope_learning_objectives > 0 && <span>{topic.scope_learning_objectives} objectives</span>}</span></button>; })}{!visible.length && <EmptyAction icon={Search} title="No topics found" copy="Try a broader name or clear the root filter." action="Clear filters" onAction={() => { setQuery(""); setRootFilter("all"); }} />}</section>{selected && <aside className="topic-detail"><div className="topic-detail-heading"><span className="topic-icon topic-teal">{initials(selected.name)}</span><div><p className="eyebrow">{selected.path.slice(0, -1).join(" · ")}</p><h2>{selected.name}</h2></div></div><p>{selected.description}</p><div className="topic-detail-stats"><div><strong>{selected.scope_primary_questions}</strong><span>Approved scope questions</span></div><div><strong>{selected.scope_learning_objectives || "—"}</strong><span>Scope objectives</span></div></div><div className="mode-selector" role="group" aria-label="New session mode">{(["learn", "diagnose", "review"] as const).map((item) => <button key={item} className={mode === item ? "mode-selected" : ""} type="button" onClick={() => setMode(item)}>{readable(item)}</button>)}</div><div className="topic-outline"><p className="mini-label">Released scope</p>{detailLoading ? <p>Loading exact graph scope…</p> : <>{detail?.scope_concepts.slice(0, 5).map((concept, index) => <div key={concept.id}><span>{index + 1}</span><strong>{concept.name}</strong><Badge>Concept</Badge></div>)}{detail && detail.scope_concepts.length > 5 && <p className="muted">+ {detail.scope_concepts.length - 5} more concepts · {detail.learning_objectives.length} learning objectives</p>}</>}</div><button className="button button-primary button-full" type="button" disabled={selected.scope_primary_questions === 0} onClick={() => void startTopic(selected)}><Play size={16} />Start {selected.name}</button><BoundaryNote compact>Session policy can explore related concepts while remaining pinned to release {catalog?.release_id.slice(0, 12)}…</BoundaryNote></aside>}</div></div>;
}

function ProgressWorkspace({ learner, profile, objectiveCount, start }: { learner: Learner | null; profile: Profile | null; objectiveCount: number; start: () => void }) {
  if (!learner || !profile) return <div className="page-shell"><PageHeader eyebrow="Your workspace" title="Progress evidence" copy="Selected-response evidence appears after your first persisted session." /><EmptyAction icon={BarChart3} title="No learner evidence yet" copy="Start a real topic session. The Python engine will write your responses and provisional estimates to tsq.db." action="Choose a topic" onAction={start} /></div>;
  const objectives = (profile.learning_objectives ?? []) as Array<Record<string, unknown>>;
  const observed = objectives.filter((item) => Number(item.observed_response_families ?? 0) > 0);
  const families = objectives.reduce((sum, item) => sum + Number(item.independent_families ?? 0), 0);
  return <div className="page-shell"><PageHeader eyebrow="Live learner projection" title="Progress evidence" copy="Conservative, provisional selected-response estimates reconstructed from the shared event ledger." /><BoundaryNote><strong>Inference boundary.</strong> These are model estimates with uncertainty—not mastery certificates or productive-skill claims.</BoundaryNote><div className="metric-grid"><MetricCard label="Objectives observed" value={`${observed.length} / ${objectiveCount}`} note="At least one response family" icon={Target} tone="teal" /><MetricCard label="Objective-family units" value={String(families)} note="Canonicalized per objective" icon={Layers3} tone="blue" /><MetricCard label="Active hypotheses" value={String(profile.active_misconceptions.length)} note="Routing threshold only" icon={Activity} tone="violet" /><MetricCard label="Learner revision" value={String(learner.revision)} note="Durable projection revision" icon={Database} tone="amber" /></div><div className="panel"><div className="panel-heading"><div><p className="eyebrow">Learning objectives</p><h2>Evidence by operation</h2></div><Badge>{profile.boundary_algorithm_version}</Badge></div><div className="objective-list">{objectives.map((item) => <div className="objective-row" key={String(item.objective_id)}><span className="metric-icon metric-teal"><Target size={16} /></span><span><strong>{String(item.name)}</strong><small>{String(item.description)}</small></span><span><Badge tone={String(item.state) === "unassessed" ? "neutral" : "teal"}>{readable(String(item.state))}</Badge><small>{Number(item.observed_response_families ?? 0)} observed · {Number(item.successful_retrieval_families ?? 0)} retrieved</small></span></div>)}</div></div></div>;
}

function SessionsWorkspace({ sessions, resumeSession, start }: { sessions: SessionSummary[]; resumeSession: (session: SessionSummary) => Promise<void>; start: () => void }) {
  const [status, setStatus] = useState<"all" | "active" | "completed" | "abandoned">("all");
  const visible = sessions.filter((session) => status === "all" || session.status === status);
  return <div className="page-shell"><PageHeader eyebrow="Shared SQLite history" title="Sessions" copy="Sessions started in the CLI or browser appear together because both use the same database and engine." actions={<button className="button button-primary" type="button" onClick={start}><Plus size={16} />New session</button>} />{sessions.length ? <><div className="filter-pills">{(["all", "active", "completed", "abandoned"] as const).map((item) => <button className={`filter-pill ${status === item ? "filter-pill-active" : ""}`} key={item} type="button" onClick={() => setStatus(item)}>{readable(item)}</button>)}</div><div className="panel session-list">{visible.map((session) => <button className="session-row" type="button" key={session.id} onClick={() => void resumeSession(session)}><span className={`metric-icon ${session.status === "active" ? "metric-teal" : "metric-blue"}`}>{session.status === "active" ? <Play size={17} /> : <History size={17} />}</span><span><strong>{session.target_name}</strong><small>{readable(session.mode)} · {formatWhen(session.updated_at)}</small></span><span><strong>{session.questions_answered} responses</strong><small>{session.selected_answers} selected · {session.abstained} skipped</small></span><Badge tone={session.status === "active" ? "teal" : session.status === "completed" ? "green" : "neutral"}>{readable(session.status)}</Badge><ChevronRight size={17} /></button>)}{!visible.length && <p className="muted">No {status} sessions.</p>}</div></> : <EmptyAction icon={History} title="No study sessions yet" copy="Start from any released topic. This history is persisted in tsq.db and is immediately visible to the CLI." action="Choose a topic" onAction={start} />}</div>;
}

function LabsWorkspace() {
  return <div className="page-shell"><PageHeader eyebrow="Separate evidence boundary" title="Productive skill probes" copy="No productive-task release is currently connected to this learner workspace." actions={<Badge tone="violet">Shadow-only</Badge>} /><BoundaryNote>Productive task observations never alter selected-response progress, routing, certification, or learner mastery.</BoundaryNote><EmptyAction icon={FlaskConical} title="No productive tasks available" copy="Import and review a corpus-pinned productive-task release through the CLI before recommendations can appear." action="View operations" onAction={() => window.scrollTo({ top: 0 })} /></div>;
}

function OperationsWorkspace({ health, catalog, learner, sessions }: { health: Health | null; catalog: Catalog | null; learner: Learner | null; sessions: SessionSummary[] }) {
  const counts = health?.corpus;
  return <div className="operations-shell"><PageHeader eyebrow="Local operator status" title="TSQ operations" copy="The browser is attached to the same active release and SQLite database used by every CLI command." actions={<Badge tone={health ? "green" : "amber"}>{health ? "Engine connected" : "Unavailable"}</Badge>} /><div className="release-hero"><div><Badge tone="green"><CheckCircle2 size={13} />Active release</Badge><h2>Schema {health?.schema_version ?? "—"}</h2><p className="mono">{health?.corpus_release_id ?? "No active release"}</p></div><div className="release-counts"><div><strong>{counts?.active_questions ?? "—"}</strong><span>Live questions</span></div><div><strong>{counts?.retired_questions ?? "—"}</strong><span>Retired lineage</span></div><div><strong>{counts?.active_families ?? "—"}</strong><span>Evidence families</span></div></div></div><div className="metric-grid"><MetricCard label="Topics" value={String(counts?.topics ?? "—")} note={`${catalog?.domains.length ?? 0} domain`} icon={Library} tone="teal" /><MetricCard label="Concepts" value={String(counts?.concepts ?? "—")} note="Active release membership" icon={Layers3} tone="blue" /><MetricCard label="Objectives" value={String(counts?.learning_objectives ?? "—")} note="Selected-response operations" icon={Target} tone="violet" /><MetricCard label="Sessions" value={String(sessions.length)} note={learner ? `Learner ${learner.id}` : "No learner created"} icon={History} tone="amber" /></div><div className="panel"><div className="panel-heading"><div><p className="eyebrow">Parity contract</p><h2>One engine, two interfaces</h2></div><Badge tone="green">No duplicated policy</Badge></div><div className="developer-lab-list"><div><span className="metric-icon metric-teal"><TerminalSquare size={17} /></span><span><strong>CLI</strong><small>Direct commands against tsq.db</small></span><Badge>Available</Badge></div><div><span className="metric-icon metric-blue"><MessageSquare size={17} /></span><span><strong>Web interface</strong><small>Thin loopback API calling AdaptiveEngine</small></span><Badge tone="green">Connected</Badge></div><div><span className="metric-icon metric-violet"><Database size={17} /></span><span><strong>Durable state</strong><small>Learners, sessions, decisions, attempts, projections, and events share one file</small></span><Badge tone="green">tsq.db</Badge></div></div></div><BoundaryNote><strong>Fail-closed boundary.</strong> The web interface does not reimplement selection, scoring, evidence updates, reports, or release rules in TypeScript.</BoundaryNote></div>;
}

function SettingsWorkspace({ health, learner, theme, changeTheme, refresh }: { health: Health | null; learner: Learner | null; theme: Theme; changeTheme: () => void; refresh: () => void }) {
  return <div className="page-shell settings-shell"><PageHeader eyebrow="Local workspace" title="Settings" copy="Interface preferences stay in this browser; learning data stays in the shared TSQ database." /><section className="settings-section"><div className="settings-copy"><h2>Appearance</h2><p>Use a high-contrast light or dark workspace.</p></div><div className="settings-card"><button className="button button-secondary" type="button" onClick={changeTheme}>{theme === "light" ? <Moon size={16} /> : <Sun size={16} />}Switch to {theme === "light" ? "dark" : "light"}</button></div></section><section className="settings-section"><div className="settings-copy"><h2>Learner</h2><p>The local single-user identity is shared with the CLI.</p></div><div className="settings-card"><div className="profile-chip profile-chip-large"><span className="avatar">{learner ? "ME" : "G"}</span><span><strong>{learner?.display_name || "Not created yet"}</strong><small>{learner ? `${learner.id} · revision ${learner.revision}` : "Created on first real session"}</small></span></div></div></section><section className="settings-section"><div className="settings-copy"><h2>Data & release</h2><p>Exact local service boundary.</p></div><div className="settings-card data-settings"><div className="detail-stat-row"><span>API</span><Badge tone={health ? "green" : "amber"}>{health ? `Connected · ${health.api_version}` : "Unavailable"}</Badge></div><div className="detail-stat-row"><span>Schema</span><strong>{health?.schema_version ?? "—"}</strong></div><div className="detail-stat-row"><span>Corpus release</span><code>{health?.corpus_release_id ?? "—"}</code></div><div className="detail-stat-row"><span>Database</span><strong>Shared local tsq.db</strong></div><button className="button button-secondary" type="button" onClick={refresh}><RefreshCw size={15} />Refresh connection</button></div></section></div>;
}
