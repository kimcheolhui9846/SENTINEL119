import { useEffect, useState } from 'react';
import type { FormEvent, KeyboardEvent, MouseEvent } from 'react';
import type { DispatchEvent, Mode, Session } from './api';
import { HospitalMap } from './HospitalMap';
import { DrillPanel } from './DrillPanel';
import { CandidatePanel, candidateName, DecisionsPanel, HistoryPanel } from './panels';
import { useConsole } from './useConsole';
import CompareView from './CompareView';
import { api as compareApi } from './compareApi';
import type { CompareScenario } from './compareApi';
import './styles.css';

const pages = ['workspace', 'history', 'drill', 'compare'] as const;
type Page = typeof pages[number];
const pageNames = { workspace: '관제 화면', history: '판단 이력', drill: '자동 모의훈련', compare: '모델 비교' };

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

function DispatchPanel({ session, disabled, onDispatch }: {
  session: Session; disabled: boolean; onDispatch: (event: DispatchEvent) => void;
}) {
  const applied = session.applied_vehicle;
  const { status, choice, recommendation_revision: revision } = session.decisions.vehicle;
  const vehicle = session.vehicles.find((item) => item.id === choice && item.availability === 'available');
  const canDispatch = status === 'COMMIT' && !!vehicle && typeof revision === 'number' && !applied;
  let reason = '확정된 가용 차량을 기다리고 있습니다.';
  if (applied) reason = `${candidateName(session.vehicles, applied)}에 모의 출동을 적용했습니다.`;
  else if (canDispatch) reason = `${vehicle.name}에 모의 출동을 적용합니다.`;
  return (
    <section className="panel dispatch-panel" aria-labelledby="dispatch-heading">
      <div>
        <h2 id="dispatch-heading" className="section-title">모의 출동</h2>
        <p id="dispatch-reason" className="reason" aria-live="polite">{reason}</p>
      </div>
      <button key={`${session.id}:${session.version}:${choice}:${revision}`} type="button" className="button primary"
        disabled={disabled || !canDispatch} aria-describedby="dispatch-reason"
        onClick={() => {
          if (canDispatch) onDispatch({ action: 'dispatch', expected_vehicle_id: vehicle.id, expected_vehicle_revision: revision });
        }}>모의 출동 적용</button>
    </section>
  );
}

export default function App() {
  const consoleState = useConsole();
  const { scenarios, scenarioId, mode, session, busy, autoPlay, problem, canNext, canRetry,
    open, retry, send, setAutoPlay } = consoleState;
  const [page, setPage] = useState<Page>(window.location.hash === '#drill' ? 'drill' : 'workspace');
  const [drillOpened, setDrillOpened] = useState(page === 'drill');
  const [draft, setDraft] = useState('');
  const [compareCalls, setCompareCalls] = useState<CompareScenario[]>([]);
  useEffect(() => { compareApi.scenarios().then(setCompareCalls).catch(() => setCompareCalls([])); }, []);
  const selected = scenarios.find((item) => item.id === scenarioId);
  const modeLabel = mode === 'live' ? '실시간 Zoro' : '시뮬레이션';
  const showRetry = !!problem?.recovery || canRetry;
  const notice = problem?.message ?? session?.error ??
    (session?.external_status === 'timeout' ? '병원 정보 조회 시간이 초과되었습니다. 최신 수용 정보를 기다리고 있습니다.' : null);

  function changeSelection(id: string, nextMode: Mode) {
    setDraft('');
    // Call files have no reviewed fixture; they run on the live model only.
    const liveOnly = scenarios.find((item) => item.id === id)?.live_only;
    void open(id, liveOnly ? 'live' : nextMode);
  }

  async function submitText(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const message = draft.trim();
    if (!message) return;
    setAutoPlay(false);
    if (await send({ action: 'text', text: message })) setDraft((value) => value === draft ? '' : value);
  }

  function goToMap(event: MouseEvent<HTMLAnchorElement>) {
    event.preventDefault();
    setPage('workspace');
    window.setTimeout(() => {
      const target = document.getElementById('nmc-map');
      target?.scrollIntoView({ block: 'start' });
      target?.focus({ preventScroll: true });
    }, 0);
  }

  function selectPage(next: Page) {
    if (next === 'drill') {
      setDrillOpened(true);
      setAutoPlay(false);
    }
    setPage(next);
  }

  function switchTab(event: KeyboardEvent<HTMLButtonElement>) {
    const keys = ['ArrowLeft', 'ArrowRight', 'Home', 'End'];
    if (!keys.includes(event.key)) return;
    event.preventDefault();
    const index = pages.indexOf(page);
    const next = event.key === 'Home' ? pages[0] : event.key === 'End' ? pages[pages.length - 1]
      : pages[(index + (event.key === 'ArrowLeft' ? -1 : 1) + pages.length) % pages.length];
    selectPage(next);
    document.getElementById(`tab-${next}`)?.focus();
  }

  return (
    <div className="app-shell">
      <header className="masthead">
        <div className="brand"><span className="brand-mark" aria-hidden="true">+</span><span className="brand-name">SENTINEL 119</span></div>
        <nav className="nav-tabs" aria-label="관제 메뉴">
          <div role="tablist" aria-label="관제 보기">
            {pages.map((tab) => (
              <button type="button" key={tab} id={`tab-${tab}`} role="tab"
                className={`nav-tab${page === tab ? ' active' : ''}`} aria-selected={page === tab}
                aria-controls={`page-${tab}`} tabIndex={page === tab ? 0 : -1} onKeyDown={switchTab}
                onClick={() => selectPage(tab)}>{pageNames[tab]}</button>
            ))}
          </div>
        </nav>
        {page !== 'drill' && page !== 'compare' && <span className="mode-indicator" data-mode={mode}>{modeLabel}</span>}
      </header>

      <main className="main">
        {page !== 'drill' && page !== 'compare' && <>
        <header className="page-heading">
          <h1 className="workspace-title">신고 관제</h1>
          <p className="location-field" data-known={session?.location ? 'true' : 'false'}>
            <span className="field-label">신고 위치</span>
            <strong className="field-value" aria-live="polite">{session?.location || '확인 전'}</strong>
          </p>
          <a className="map-link" href="#nmc-map" onClick={goToMap}>병원 지도</a>
        </header>

        <section className="scenario-bar" aria-label="시나리오 설정">
          <div className="scenario-info">
            <strong>{selected?.title ?? '신고 시나리오'}</strong>
            {selected && <p>{selected.summary}</p>}
            <span className="meta">{selected?.private ? '실제 통화 (AI Hub 유보 평가셋, 로컬 시연 전용) · 가상 차량 및 병원'
              : selected?.live_only ? '가상 통화 · 모델 확률로 확정 · 가상 차량 및 병원' : '합성 신고 · 가상 차량 및 병원'}</span>
          </div>
          <div className="toolbar">
            <label>시나리오
              <select aria-label="시나리오" value={scenarioId} disabled={!scenarios.length}
                onChange={(event) => changeSelection(event.target.value, mode)}>
                {!scenarios.length && <option value="">선택할 시나리오 없음</option>}
                {scenarios.map((scenario) => <option key={scenario.id} value={scenario.id}>{scenario.private ? '★ ' : ''}{scenario.title}</option>)}
              </select>
            </label>
            <label>실행 모드
              <select aria-label="실행 모드" value={mode} disabled={!scenarioId}
                onChange={(event) => changeSelection(scenarioId, event.target.value as Mode)}>
                <option value="simulation" disabled={!!selected?.live_only}>시뮬레이션</option>
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
        </>}

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
                    onClick={() => { setAutoPlay(false); void send({ action: 'next' }); }}>다음 발화</button>
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
                <div className="resources-block">
                  <p className="resources-note">가상 출동 후보입니다. 아래 지도에서 공공데이터에 등록된 병·의원 위치를 확인할 수 있습니다.</p>
                  <div className="resources-grid">
                    <CandidatePanel kind="vehicle" session={session} />
                    <CandidatePanel kind="hospital" session={session} />
                  </div>
                </div>
                <DispatchPanel session={session} disabled={busy || !!problem?.recovery}
                  onDispatch={(event) => void send(event, session)} />
              </div>
            </div>
          )}
          {page === 'workspace' && (
            <div id="nmc-map" className="map-region" tabIndex={-1}>
              <p className="map-caption">국립중앙의료원 병·의원 위치 자료</p>
              <HospitalMap />
            </div>
          )}
        </div>
        <div id="page-compare" role="tabpanel" aria-labelledby="tab-compare" hidden={page !== 'compare'}>
          {page === 'compare' && (compareCalls.length ? <CompareView scenarios={compareCalls} />
            : <p className="empty-state">비교할 통화가 없습니다. SENTINEL_SCENARIO_DIRS를 설정하세요.</p>)}
        </div>
        <div id="page-history" role="tabpanel" aria-labelledby="tab-history" hidden={page !== 'history'}>
          {page === 'history' && (session ? <HistoryPanel session={session} /> : <p className="empty-state">신고를 열면 판단 이력을 확인할 수 있습니다.</p>)}
        </div>
        <div id="page-drill" role="tabpanel" aria-labelledby="tab-drill" hidden={page !== 'drill'}>
          {drillOpened && <DrillPanel />}
        </div>
      </main>
    </div>
  );
}
