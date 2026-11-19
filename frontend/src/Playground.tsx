import { useRef, useState } from 'react';
import type { FormEvent } from 'react';
import { ApiError, api } from './compareApi';
import { advance, emptyLlm, emptyZoro, llmNote, LlmPanel, ZoroPanel } from './CompareView';
import type { Labels, LlmState, ZoroId, ZoroState } from './CompareView';

type Speaker = 0 | 1;
interface Line { speaker: Speaker; text: string; note?: string; ms?: number }

/** Free input: each submitted utterance is one more turn; both Zoro heads judge the whole prefix. */
export default function Playground() {
  const [lines, setLines] = useState<Line[]>([]);
  const [draft, setDraft] = useState('');
  const [speaker, setSpeaker] = useState<Speaker>(1);
  const [withLlm, setWithLlm] = useState(false);
  const [zoro, setZoro] = useState<Record<ZoroId, ZoroState>>({ sft: emptyZoro(), base: emptyZoro() });
  const [llm, setLlm] = useState<LlmState>(emptyLlm());
  const [labels, setLabels] = useState<Labels | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const zoroRef = useRef(zoro);
  zoroRef.current = zoro;

  function reset() {
    setLines([]); setZoro({ sft: emptyZoro(), base: emptyZoro() }); setLlm(emptyLlm()); setError(null);
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const text = draft.trim();
    if (!text || busy) return;
    const turns = [...lines.map(({ speaker: s, text: t }) => ({ speaker: s, text: t })), { speaker, text }];
    setLines([...lines, { speaker, text }]); setDraft(''); setBusy(true); setError(null);
    try {
      const res = await api.streamDecide(turns);
      setLabels(res.labels);
      const next = { ...zoroRef.current };
      let note = '';
      for (const id of ['sft', 'base'] as ZoroId[]) {
        const { next: updated, note: n } = advance(next[id], res.results[id].probabilities, turns.length, res.labels);
        next[id] = { ...updated, latencies: [...next[id].latencies, res.results[id].latency_ms] };
        if (id === 'sft') note = n;
      }
      setZoro(next);
      setLines((list) => list.map((line, i) => (i === turns.length - 1 ? { ...line, note, ms: res.results.sft.latency_ms } : line)));
      if (withLlm) {
        setLlm((s) => ({ ...s, pending: true }));
        const out = await api.streamLlm(turns);
        setLlm((s) => ({ answer: out.answer, raw: out.text, latencies: [...s.latencies, out.latency_ms], pending: false }));
        void llmNote;
      }
    } catch (e) {
      setLlm((s) => ({ ...s, pending: false }));
      setError(e instanceof ApiError ? e.message : '판단 요청에 실패했습니다.');
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="stream compare playground">
      {error ? <div className="banner" role="alert">{error}</div> : null}
      <div className="stream-layout">
        <section className="panel call" aria-labelledby="play-title">
          <header className="panel-head"><h2 id="play-title">직접 입력</h2>
            <span className="muted tabular">{lines.length}턴</span></header>
          <ol className="call-turns">
            {lines.map((line, i) => (
              <li key={i} className={`call-turn speaker-${line.speaker}`}>
                <span className="who">{line.speaker === 0 ? '수보요원' : '신고자'}</span>
                <p>{line.text}</p>
                {line.note ? <p className={`zoro-note ${line.note === 'WAIT' ? 'is-wait' : 'is-commit'}`}>
                  <span className="zoro-tag">SENTINEL_ZORO</span>{line.note}
                  <span className="muted tabular"> {Math.round(line.ms ?? 0)}ms</span></p> : null}
              </li>
            ))}
            {!lines.length ? <li className="empty">신고 내용을 한 문장씩 입력하면, 턴마다 SENTINEL_ZORO와 base Zoro가 판단합니다.</li> : null}
          </ol>
          <form className="play-form" onSubmit={(e) => void submit(e)}>
            <div className="play-speaker" role="radiogroup" aria-label="화자">
              {([1, 0] as Speaker[]).map((s) => (
                <button type="button" key={s} role="radio" aria-checked={speaker === s}
                  className={speaker === s ? 'is-active' : ''} onClick={() => setSpeaker(s)}>{s === 1 ? '신고자' : '수보요원'}</button>
              ))}
            </div>
            <textarea rows={3} maxLength={2000} value={draft} placeholder="예: 아버지가 갑자기 쓰러지셨어요. 숨을 안 쉬는 것 같아요."
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); e.currentTarget.form?.requestSubmit(); } }} />
            <div className="play-actions">
              <label><input type="checkbox" checked={withLlm} onChange={(e) => setWithLlm(e.target.checked)} /> Qwen 답변도 생성</label>
              <button type="button" onClick={reset} disabled={busy}>처음부터</button>
              <button type="submit" disabled={busy || !draft.trim()}>{busy ? '판단 중…' : '보내기'}</button>
            </div>
          </form>
        </section>
        <div className="models">
          <ZoroPanel id="sft" name="SENTINEL_ZORO" note="119 신고 통화로 학습한 Zoro" model={zoro.sft} labels={labels} busy={busy} />
          <ZoroPanel id="base" name="Zoro · base" note="공개 Zoro-1 그대로" model={zoro.base} labels={labels} busy={busy} />
          {withLlm ? <LlmPanel state={llm} labels={labels} /> : null}
        </div>
      </div>
    </div>
  );
}
