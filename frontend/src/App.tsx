import { useState } from 'react';
import type { FormEvent, KeyboardEvent } from 'react';
import type { Mode, Session } from './api';
import { CandidatePanel, candidateName, DecisionsPanel, HistoryPanel } from './panels';
import { useConsole } from './useConsole';
import './styles.css';

type Page = 'workspace' | 'history';

function Transcript({ session }: { session: Session }) {
  if (!session.turns.length) {
    return <p className="empty-state">다음 발화를 누르거나 신고 내용을 직접 입력하세요.</p>;
  }
  return (
    <ol className="transcript" aria-label="신고 발화">
      {session.turns.map((turn) => (
        <li key={turn.id} className="turn" aria-current={turn.id === session.turn_id ? 'step' : undefined}
          data-speaker={turn.speaker === '신고자' ? 'caller' : turn.speaker === '수보요원' ? 'dispatcher' : undefined}>
          <span className="turn-speaker">{turn.speaker} · {turn.id}턴</span>
          <p className="turn-text">{turn.text}</p>
        </li>
      ))}
    </ol>
  );
}

function DispatchPanel({ session, busy, onDispatch }: { session: Session; busy: boolean; onDispatch: () => void }) {
  const applied = session.applied_vehicle;
  const decision = session.decisions.vehicle;
  const eligible = session.vehicles.some((item) => item.id === decision.choice && item.availability === 'available');
  const canDispatch = decision.status === 'COMMIT' && eligible && !applied;
  let reason = '현재 확정된 차량에 모의 출동을 적용합니다.';
  if (applied) reason = `${candidateName(session.vehicles, applied)}에 모의 출동을 적용했습니다.`;
  else if (!canDispatch) reason = '확정된 가용 차량을 기다리고 있습니다.';
  return (
    <section className="panel dispatch-panel" aria-labelledby="dispatch-heading">
      <div>
        <h2 id="dispatch-heading" className="section-title">모의 출동</h2>
        <p id="dispatch-reason" className="reason" aria-live="polite">{reason}</p>
      </div>
      <button type="button" className="button primary" disabled={busy || !canDispatch}
        aria-describedby="dispatch-reason" onClick={onDispatch}>모의 출동 적용</button>
    </section>
  );
}

export default function App() {
  const consoleState = useConsole();
  const { scenarios, scenarioId, mode, session, busy, autoPlay, problem, canNext, canRetry,
    open, retry, send, setAutoPlay } = consoleState;
  const [page, setPage] = useState<Page>('workspace');
  const [draft, setDraft] = useState('');
  const selected = scenarios.find((item) => item.id === scenarioId);
  const modeLabel = mode === 'live' ? '실시간 Zoro' : '시뮬레이션';
  const showRetry = !!problem?.recovery || canRetry;
  const notice = problem?.message ?? session?.error ??
    (session?.external_status === 'timeout' ? '병원 정보 조회 시간이 초과되었습니다. 저장 자료를 표시합니다.' : null);

  function changeSelection(id: string, nextMode: Mode) {
    setDraft('');
    void open(id, nextMode);
  }

  async function submitText(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const message = draft.trim();
    if (!message) return;
    setAutoPlay(false);
    if (await send('text', message)) setDraft((value) => value === draft ? '' : value);
  }

  function switchTab(event: KeyboardEvent<HTMLButtonElement>) {
    const keys = ['ArrowLeft', 'ArrowRight', 'Home', 'End'];
    if (!keys.includes(event.key)) return;
    event.preventDefault();
    const next = event.key === 'Home' ? 'workspace' : event.key === 'End' ? 'history'
      : page === 'workspace' ? 'history' : 'workspace';
    setPage(next);
    document.getElementById(`tab-${next}`)?.focus();
  }

  return (
    <div className="app-shell">
      <header className="masthead">
        <div className="brand"><span className="brand-mark" aria-hidden="true">+</span><span className="brand-name">SENTINEL 119</span></div>
        <nav className="nav-tabs" aria-label="관제 메뉴">
          <div role="tablist" aria-label="관제 보기">
            {(['workspace', 'history'] as const).map((tab) => (
              <button type="button" key={tab} id={`tab-${tab}`} role="tab"
                className={`nav-tab${page === tab ? ' active' : ''}`} aria-selected={page === tab}
                aria-controls={`page-${tab}`} tabIndex={page === tab ? 0 : -1} onKeyDown={switchTab}
                onClick={() => setPage(tab)}>{tab === 'workspace' ? '관제 화면' : '판단 이력'}</button>
            ))}
          </div>
        </nav>
        <span className="mode-indicator" data-mode={mode}>{modeLabel}</span>
      </header>

      <main className="main">
        <header className="page-heading">
          <h1 className="workspace-title">신고 관제</h1>
          {session?.location && <span className="meta">확인된 위치 · {session.location}</span>}
        </header>

        <section className="scenario-bar" aria-label="시나리오 설정">
          <div className="scenario-info">
            <strong>{selected?.title ?? '신고 시나리오'}</strong>
            {selected && <p>{selected.summary}</p>}
            <span className="meta">합성 신고 · 가상 차량 및 병원</span>
          </div>
          <div className="toolbar">
            <label>시나리오
              <select aria-label="시나리오" value={scenarioId} disabled={!scenarios.length}
                onChange={(event) => changeSelection(event.target.value, mode)}>
                {!scenarios.length && <option value="">선택할 시나리오 없음</option>}
                {scenarios.map((scenario) => <option key={scenario.id} value={scenario.id}>{scenario.title}</option>)}
              </select>
            </label>
            <label>실행 모드
              <select aria-label="실행 모드" value={mode} disabled={!scenarioId}
                onChange={(event) => changeSelection(scenarioId, event.target.value as Mode)}>
                <option value="simulation">시뮬레이션</option>
                <option value="live">실시간 Zoro</option>
              </select>
            </label>
            <button type="button" className="button secondary" disabled={!scenarioId}
              onClick={() => changeSelection(scenarioId, mode)}>초기화</button>
          </div>
        </section>

        {(notice || showRetry) && (
          <div className="alert">
            {notice && <span role="alert">{notice}</span>}
            {showRetry && <button type="button" className="button secondary" disabled={busy}
              onClick={() => void retry()}>재시도</button>}
          </div>
        )}

        <div id="page-workspace" role="tabpanel" aria-labelledby="tab-workspace" hidden={page !== 'workspace'}>
          {!session ? <p className="empty-state">{busy ? '신고 정보를 불러오는 중입니다.' : '시나리오를 선택해 신고를 시작하세요.'}</p> : (
            <div className="workspace-grid">
              <section className="panel transcript-panel" aria-labelledby="transcript-heading">
                <header className="panel-header">
                  <h2 id="transcript-heading" className="section-title">신고 대화</h2>
                  <div className="turn-progress">
                    <span className="turn-counter" aria-live="polite">
                      {session.manual ? `직접 입력 · ${session.turn_id}턴` : `${session.step} / ${session.total_steps}턴`}
                    </span>
                    {!session.manual && session.total_steps > 0 && (
                      <span className="turn-ticks" aria-hidden="true">
                        {Array.from({ length: session.total_steps }, (_, index) => (
                          <i key={index} data-filled={index < Math.min(session.step, session.total_steps) ? 'true' : 'false'} />
                        ))}
                      </span>
                    )}
                  </div>
                </header>
                <Transcript session={session} />
                <div className="controls">
                  <button type="button" className="button primary" disabled={busy || !canNext}
                    onClick={() => { setAutoPlay(false); void send('next'); }}>다음 발화</button>
                  <button type="button" className={`button secondary${autoPlay ? ' active' : ''}`}
                    aria-pressed={autoPlay} disabled={!canNext} onClick={() => setAutoPlay(!autoPlay)}>자동 재생</button>
                </div>
                <form className="composer" onSubmit={(event) => void submitText(event)}>
                  <label htmlFor="manual-input">직접 입력</label>
                  <div className="composer-fields">
                    <textarea id="manual-input" placeholder="신고 내용을 입력하세요" rows={3} maxLength={2000}
                      value={draft} onChange={(event) => setDraft(event.target.value)} />
                    <button type="submit" className="button secondary" disabled={busy || !draft.trim()}>전송</button>
                  </div>
                  {session.manual && <p className="meta">직접 입력은 검토된 근거를 기다리고 있습니다. 시나리오 재생은 초기화 후 다시 시작할 수 있습니다.</p>}
                </form>
              </section>

              <div className="workspace-detail">
                <DecisionsPanel session={session} />
                <div className="resources-grid">
                  <CandidatePanel kind="vehicle" session={session} />
                  <CandidatePanel kind="hospital" session={session} />
                </div>
                <DispatchPanel session={session} busy={busy} onDispatch={() => void send('dispatch')} />
              </div>
            </div>
          )}
        </div>
        <div id="page-history" role="tabpanel" aria-labelledby="tab-history" hidden={page !== 'history'}>
          {page === 'history' && (session ? <HistoryPanel session={session} /> : <p className="empty-state">신고를 열면 판단 이력을 확인할 수 있습니다.</p>)}
        </div>
      </main>
    </div>
  );
}
