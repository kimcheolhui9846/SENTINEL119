import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, api } from "./compareApi";
import type { CompareScenario as ScenarioSummary, LlmResponse, ScenarioDetail, StreamResponse } from "./compareApi";
import { advanceHarness, emptyZoro, harnessThreshold, SUBTYPE_OF, suspendHarness } from "./demoHarness";
import type { Commit, ZoroId, ZoroState } from "./demoHarness";

const H4_DEV_PRECISION = [
  { task: "출동유형", sft: [0.986, 0.867], base: [0.723, 1.0] },
  { task: "구급 세부유형", sft: [0.8, 0.492], base: [0.179, 0.75] },
  { task: "긴급도", sft: [0.889, 0.217], base: [0.333, 0.723] },
  { task: "주증상", sft: [0.765, 0.279], base: [0.4, 0.962] },
];
const pct = (value: number) => Math.round(value * 100) + "%";
const TYPE_MS = 420;
const NEXT_DELAY_MS = 900;
export type Labels = StreamResponse["labels"];
export interface LlmState { answer: LlmResponse["answer"] | null; raw: string; latencies: number[]; pending: boolean }
interface TurnLog {
  speaker: 0 | 1;
  text: string;
  notes: Partial<Record<ZoroId | "llm", string>>;
  ms: Partial<Record<ZoroId | "llm", number>>;
}

export const emptyLlm = (): LlmState => ({ answer: null, raw: "", latencies: [], pending: false });
const top = (dist?: Record<string, number>) =>
  dist ? (Object.entries(dist).sort((a, b) => b[1] - a[1])[0] as [string, number]) : undefined;
const avg = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0);

export function llmNote(answer: LlmResponse["answer"], labels: Labels) {
  const parts = [
    answer.dispatch && labels.dispatch[answer.dispatch],
    answer.subtype && answer.subtype_question && labels[answer.subtype_question]?.[answer.subtype],
    answer.urgency && labels.urgency[answer.urgency],
  ].filter(Boolean);
  const bad = answer.invalid.length ? ` (선택지 밖: ${answer.invalid.join(", ")})` : "";
  return (parts.length ? `생성 · ${parts.join(" · ")}` : "판단 보류(null)") + bad;
}

function Chips({ dist, labels, commit, threshold, locked, picked }: {
  dist?: Record<string, number>; labels: Record<string, string>; commit?: Commit; threshold?: number;
  locked?: boolean; picked?: string | null;
}) {
  const best = top(dist);
  const leader = best?.[0];
  return (
    <div className="chips" role="list">
      {Object.entries(labels).map(([key, label]) => {
        const p = dist?.[key] ?? 0;
        const state = picked !== undefined ? (picked === key ? "generated" : picked ? "dim" : "")
          : commit ? (commit.key === key ? "committed" : "dim") : key === leader && !locked ? "leading" : "";
        return (
          <span key={key} role="listitem" className={`chip-opt ${state}`} style={{ ["--p" as string]: p.toFixed(3) }}>
            <span className="chip-name">{commit?.key === key ? "✓ " : ""}{label}</span>
            {commit?.key === key ? <span className="chip-p tabular">{pct(commit.p)} · {commit.turn}턴 확정</span>
              : dist ? <span className="chip-p tabular">{Math.round(p * 100)}</span> : null}
          </span>
        );
      })}
      {!commit && !locked && dist && threshold ? <span className="chip-rule tabular">확정 {Math.round(threshold * 100)}%↑</span> : null}
    </div>
  );
}

// 16 symptom options do not fit one screen: folded by default, the summary shows the commit or the leader.
function SymptomFold({ dist, labels, commit, threshold }: {
  dist: Record<string, number>; labels: Record<string, string>; commit?: Commit; threshold?: number;
}) {
  const best = top(dist);
  return (
    <details className="sym-fold">
      <summary>
        {commit ? <strong className="sym-commit">✓ {commit.label} · {pct(commit.p)} · {commit.turn}턴 확정</strong>
          : best ? <span className="tabular">모델 추정 {labels[best[0]] ?? best[0]} {pct(best[1])}
            {threshold === undefined ? null : " · 정책 기준 " + pct(threshold)}</span>
          : null}
      </summary>
      <Chips dist={dist} labels={labels} commit={commit} threshold={threshold} />
    </details>
  );
}

function Latency({ last, list, busy }: { last?: number; list: number[]; busy: boolean }) {
  return (
    <div className="latency">
      <span className="muted">응답 시간</span>
      <strong className="tabular">{busy ? "…" : last !== undefined ? Math.round(last).toLocaleString() : "—"}<small>ms</small></strong>
      <span className="muted tabular">판단 {list.length}회 · 평균 {Math.round(avg(list)).toLocaleString()}ms</span>
    </div>
  );
}

export function ZoroPanel({ id, name, note, model, labels, busy }: {
  id: ZoroId; name: string; note: string; model: ZoroState; labels: Labels | null; busy: boolean;
}) {
  const probs = model.probs ?? undefined;
  const d = model.commits.dispatch;
  const subQ = d ? SUBTYPE_OF[d.key] : undefined;
  const threshold = (question: string, key?: string) => harnessThreshold(model.profile, question, key);
  const overrides = Object.entries(model.profile?.dispatch_overrides ?? {})
    .map(([key, value]) => ` · ${labels?.dispatch[key] ?? key} ${pct(value!)}`).join("");
  const subtypeOverrides = Object.entries(model.profile?.subtype_overrides ?? {})
    .map(([key, value]) => ` · ${key === "ems_subtype" ? "구급" : "구조"} ${pct(value)}`).join("");
  return (
    <section className={`model-panel variant-${id}`} aria-label={name}>
      <div className="model-id">
        <span className="model-kind">Decision 모델</span>
        <h2>{name}</h2>
        <p className="muted">{note}</p>
        {model.reviewTurn !== null ? <p className="chip-flag">재검토 · {model.reviewTurn}턴</p> : null}
        {model.profile ? <>
          <p className="muted">출동유형 기준 {pct(model.profile.thresholds.dispatch)}{overrides}</p>
          <p className="muted">세부유형 기준 {pct(model.profile.thresholds.subtype)}{subtypeOverrides}
            {" · 긴급도 "}{pct(model.profile.thresholds.urgency)}{" · 주증상 "}{pct(model.profile.thresholds.symptom)}</p>
        </> : null}
      </div>
      <div className="decision-rows">
        <div className="drow"><span className="dlabel">출동유형</span>
          {labels ? <Chips dist={probs?.dispatch} labels={labels.dispatch} commit={d} threshold={threshold("dispatch", top(probs?.dispatch)?.[0])} /> : null}
        </div>
        <div className="drow"><span className="dlabel">세부유형</span>
          {!d ? <span className="pending">출동유형 확정 후 열림</span>
            : subQ && labels?.[subQ] ? <Chips dist={probs?.[subQ]} labels={labels[subQ]} commit={model.commits.subtype} threshold={threshold(subQ)} />
            : <span className="pending">{d.label}: 대분류에서 판단 종료</span>}
        </div>
        <div className="drow"><span className="dlabel">긴급도</span>
          {labels ? <Chips dist={probs?.urgency} labels={labels.urgency} commit={model.commits.urgency}
                           threshold={threshold("urgency")} locked={!d} /> : null}
        </div>
        <div className="drow"><span className="dlabel">주증상</span>
          {!d ? <span className="pending">출동유형 확정 후 열림</span>
            : subQ !== "ems_subtype" ? <span className="pending">{d.label} · 구급 출동에서만 판단</span>
            : probs?.ems_symptom && labels?.ems_symptom
              ? <SymptomFold dist={probs.ems_symptom} labels={labels.ems_symptom}
                  commit={model.commits.symptom} threshold={threshold("ems_symptom")} />
              : <span className="pending">주증상 응답 없음</span>}
        </div>
      </div>
      <Latency last={model.latencies[model.latencies.length - 1]} list={model.latencies} busy={busy} />
    </section>
  );
}

export function LlmPanel({ state, labels }: { state: LlmState; labels: Labels | null }) {
  const a = state.answer;
  const subQ = a?.dispatch ? SUBTYPE_OF[a.dispatch] : undefined;
  return (
    <section className="model-panel variant-llm" aria-label="Qwen3.5-4B LLM">
      <div className="model-id">
        <span className="model-kind">LLM</span>
        <h2>Qwen3.5-4B</h2>
        <p className="muted">Zoro의 바탕 모델 · reasoning off · JSON 생성</p>
      </div>
      <div className="decision-rows">
        {state.pending && !a ? <span className="pending">답변 생성 중…</span> : null}
        <div className="drow"><span className="dlabel">출동유형</span>
          {labels ? <Chips labels={labels.dispatch} picked={a ? a.dispatch : undefined} /> : null}
        </div>
        <div className="drow"><span className="dlabel">세부유형</span>
          {subQ && labels ? <Chips labels={labels[subQ]} picked={a?.subtype ?? null} />
            : <span className="pending">{a ? (a.dispatch ? "해당 없음" : "판단 보류") : "—"}</span>}
        </div>
        <div className="drow"><span className="dlabel">긴급도</span>
          {labels ? <Chips labels={labels.urgency} picked={a ? a.urgency : undefined} /> : null}
        </div>
        <div className="drow"><span className="dlabel">주증상</span>
          <span className="pending">생성 항목 아님 (출동유형·세부유형·긴급도만 생성)</span>
        </div>
        {state.raw ? <p className="llm-raw tabular" title="생성된 원문">{state.raw}</p> : null}
      </div>
      <Latency last={state.latencies[state.latencies.length - 1]} list={state.latencies} busy={state.pending} />
    </section>
  );
}

export default function CompareView({ scenarios }: { scenarios: ScenarioSummary[] }) {
  const [scenarioId, setScenarioId] = useState("");
  const [scenario, setScenario] = useState<ScenarioDetail | null>(null);
  const [shown, setShown] = useState<TurnLog[]>([]);
  const [typing, setTyping] = useState("");
  const [zoro, setZoro] = useState<Record<ZoroId, ZoroState>>({ sft: emptyZoro(), base: emptyZoro() });
  const [llm, setLlm] = useState<LlmState>(emptyLlm());
  const [labels, setLabels] = useState<Labels | null>(null);
  
  const [busy, setBusy] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const zoroRef = useRef(zoro);
  const runId = useRef(0);
  const inFlight = useRef(false);
  useEffect(() => () => { runId.current += 1; }, []);

  useEffect(() => {
    if (!scenarioId && scenarios.length) {
      setScenarioId((scenarios.find((s) => s.private) ?? scenarios.find((s) => s.id.startsWith("s0")) ?? scenarios[0]).id);
    }
  }, [scenarios, scenarioId]);

  const reset = useCallback(() => {
    runId.current += 1;
    inFlight.current = false;
    setPlaying(false); setShown([]); setTyping(""); setError(null); setBusy(false);
    zoroRef.current = { sft: emptyZoro(), base: emptyZoro() };
    setZoro(zoroRef.current); setLlm(emptyLlm()); setLabels(null);
  }, []);

  useEffect(() => {
    if (!scenarioId) return;
    reset();
    setScenario(null);
    let current = true;
    api.scenario(scenarioId).then((data) => { if (current) setScenario(data); })
      .catch((e) => { if (current) setError(e instanceof ApiError ? e.message : "시나리오를 불러오지 못했습니다."); });
    return () => { current = false; };
  }, [scenarioId, reset]);

  const step = useCallback(async () => {
    if (!scenario || inFlight.current || shown.length !== zoroRef.current.sft.turn || shown.length >= scenario.turns.length) return;
    const myRun = runId.current;
    const turn = scenario.turns[shown.length];
    inFlight.current = true;
    setBusy(true); setError(null);
    const chunk = Math.max(1, Math.ceil(turn.text.length / 14));
    for (let i = chunk; i < turn.text.length + chunk; i += chunk) {
      if (myRun !== runId.current) return;
      setTyping(turn.text.slice(0, i));
      await new Promise((r) => setTimeout(r, TYPE_MS / 14));
    }
    if (myRun !== runId.current) return;
    setTyping("");
    const turns = [...shown.map(({ speaker, text }) => ({ speaker, text })), turn];
    const turnNo = turns.length;
    const entry: TurnLog = { ...turn, notes: {}, ms: {} };
    try {
      const res = await api.streamDecide(turns);
      if (myRun !== runId.current) return;
      const next = { ...zoroRef.current };
      for (const id of ["sft", "base"] as ZoroId[]) {
        const r = res.results[id];
        const { next: updated, note, accepted } = advanceHarness(next[id], r.probabilities, { number: turnNo, speaker: turn.speaker }, res.harness, id);
        if (!accepted) throw new ApiError(note);
        next[id] = { ...updated, latencies: [...next[id].latencies, r.latency_ms] };
        entry.notes[id] = note; entry.ms[id] = r.latency_ms;
      }
      zoroRef.current = next;
      setZoro(next);
      setLabels(res.labels);
      setShown([...shown, { ...entry }]);
      setLlm((s) => ({ ...s, pending: true }));
      // Ask the LLM only after Zoro answered, so the two models never share the GPU during a timing.
      try {
        const out = await api.streamLlm(turns);
        if (myRun !== runId.current) return;
        setLlm((s) => ({ answer: out.answer, raw: out.text, latencies: [...s.latencies, out.latency_ms], pending: false }));
        entry.notes.llm = llmNote(out.answer, res.labels); entry.ms.llm = out.latency_ms;
      } catch {
        if (myRun !== runId.current) return;
        setLlm((s) => ({ ...s, pending: false }));
        entry.notes.llm = "생성 응답을 확인할 수 없습니다.";
      }
      setShown((list) => list.map((t, i) => (i === turnNo - 1 ? { ...entry } : t)));
    } catch (e) {
      if (myRun !== runId.current) return;
      zoroRef.current = { sft: suspendHarness(zoroRef.current.sft), base: suspendHarness(zoroRef.current.base) };
      setZoro(zoroRef.current);
      const message = e instanceof ApiError ? e.message : "판단 요청에 실패했습니다.";
      setError(message.startsWith("WAIT") ? message : `WAIT · ${turnNo}턴 ${message} 같은 턴을 다시 시도할 수 있습니다.`);
      setPlaying(false);
    } finally {
      if (myRun === runId.current) { inFlight.current = false; setBusy(false); }
    }
  }, [scenario, shown]);

  const done = !!scenario && shown.length >= scenario.turns.length && !busy;
  useEffect(() => {
    if (!playing || busy) return;
    if (done) { setPlaying(false); return; }
    const id = window.setTimeout(step, shown.length ? NEXT_DELAY_MS : 200);
    return () => window.clearTimeout(id);
  }, [playing, busy, done, step, shown.length]);

  const finalOf = (id: ZoroId) => {
    if (zoro[id].reviewTurn !== null) return `재검토 (${zoro[id].reviewTurn}턴)`;
    const c = zoro[id].commits;
    return [c.dispatch?.label, c.subtype?.label, c.urgency?.label].map((x) => x ?? "미확정").join(" · ");
  };
  const llmFinal = llm.answer && labels ? llmNote(llm.answer, labels).replace(/^생성 · /, "") : "—";

  return (
    <div className="stream compare">
      <div className="stream-bar">
        <label>
          통화
          <select value={scenarioId} onChange={(e) => { reset(); setScenario(null); setScenarioId(e.target.value); }}>
            {scenarios.map((s) => <option key={s.id} value={s.id}>{s.private ? "★ " : ""}{s.title}</option>)}
          </select>
        </label>
        <button type="button" onClick={() => setPlaying((p) => !p)} disabled={!scenario || done}>
          {playing ? "멈춤" : "재생"}
        </button>
        <button type="button" className="ghost" onClick={step} disabled={!scenario || busy || playing || done}>다음 턴</button>
        <button type="button" className="ghost" onClick={reset}>처음부터</button>
        {scenario?.private ? <span className="prov prov-live">실제 통화 (AI Hub 유보 평가셋, 로컬 시연 전용)</span>
          : <span className="prov prov-virtual">가상 대화</span>}
      </div>
      {error ? <div className="banner" role="alert">{error}</div> : null}
      <div className="stream-layout">
        <section className="panel call" aria-labelledby="call-title">
          <header className="panel-head"><h2 id="call-title">신고 통화</h2>
            <span className="muted tabular">{shown.length} / {scenario?.turns.length ?? 0} 턴</span></header>
          <ol className="call-turns">
            {shown.map((t, i) => (
              <li key={i} className={`call-turn speaker-${t.speaker}`}>
                <span className="who">{t.speaker === 0 ? "수보요원" : "신고자"}</span>
                <p>{t.text}</p>
                {t.notes.sft ? (
                  <p className={`zoro-note ${t.notes.sft.startsWith("COMMIT") ? "is-commit" : "is-wait"}`}>
                    <span className="zoro-tag">SENTINEL_ZORO</span>{t.notes.sft}
                    <span className="muted tabular"> {Math.round(t.ms.sft ?? 0)}ms</span>
                  </p>
                ) : <p className="base-note">판단 중…</p>}
                {t.notes.base ? <p className="base-note">base Zoro: {t.notes.base} <span className="tabular">{Math.round(t.ms.base ?? 0)}ms</span></p> : null}
                {t.notes.llm ? <p className="base-note">Qwen: {t.notes.llm} <span className="tabular">{Math.round(t.ms.llm ?? 0).toLocaleString()}ms</span></p>
                  : t.notes.sft ? <p className="base-note">Qwen: 생성 중…</p> : null}
              </li>
            ))}
            {typing ? (
              <li className={`call-turn speaker-${scenario?.turns[shown.length]?.speaker} is-typing`}>
                <span className="who">{scenario?.turns[shown.length]?.speaker === 0 ? "수보요원" : "신고자"}</span>
                <p>{typing}<span className="caret" aria-hidden="true" /></p>
              </li>
            ) : null}
            {!shown.length && !typing ? <li className="empty">재생을 누르면 통화가 한 턴씩 들어오고, 턴마다 세 모델이 판단합니다.</li> : null}
          </ol>
        </section>
        <div className="models">
          <LlmPanel state={llm} labels={labels} />
          <ZoroPanel id="sft" name="SENTINEL_ZORO" note="119 신고 통화로 학습한 Zoro" model={zoro.sft} labels={labels} busy={busy && !zoro.sft.latencies.length} />
          <ZoroPanel id="base" name="Zoro · base" note="공개 Zoro-1 그대로" model={zoro.base} labels={labels} busy={busy && !zoro.base.latencies.length} />
          {done && scenario?.gold ? (
            <section className="gold" aria-label="정답 비교">
              <h3>정답 (AI Hub 라벨): {[scenario.gold.dispatch, scenario.gold.ems_subtype ?? scenario.gold.rescue_subtype, scenario.gold.urgency].filter(Boolean).join(" · ")}</h3>
              <ul>
                <li>Qwen3.5-4B 마지막 답: {llmFinal}</li>
                <li>SENTINEL_ZORO 확정: {finalOf("sft")}</li>
                <li>Zoro · base 확정: {finalOf("base")}</li>
              </ul>
            </section>
          ) : null}
          <section className="dev-precision" aria-label="H4 개발셋 과거 측정">
            <h3>H4 보정 당시 측정 · 개발셋 83통화</h3>
            <table>
              <thead><tr><th scope="col">태스크</th><th scope="col">SENTINEL_ZORO 정밀도 (확정률)</th><th scope="col">Zoro · base 정밀도 (확정률)</th></tr></thead>
              <tbody>
                {H4_DEV_PRECISION.map((row) => (
                  <tr key={row.task}>
                    <th scope="row">{row.task}</th>
                    <td className="tabular"><strong>{pct(row.sft[0])}</strong> ({pct(row.sft[1])})</td>
                    <td className="tabular">{pct(row.base[0])} ({pct(row.base[1])})</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="muted">표는 H4 보정 당시의 정밀도와 확정률입니다. 현재 통화에 적용된 서버 기준은 모델 카드에서 확인할 수 있습니다.</p>
          </section>
          <p className="muted stream-foot">
            Zoro 칩의 밝기는 모델 확률입니다. 확정 후보와 적용 기준은 서버의 판단 정책에서 가져옵니다.
            세부유형·긴급도는 출동유형 확정 후에만 확정합니다. Qwen은 같은 대화를 보고 JSON을 생성하며, 매 턴의 답을 그대로 표시합니다.
            응답 시간은 GPU 서버의 모델 계산 시간입니다.
          </p>
        </div>
      </div>
    </div>
  );
}
