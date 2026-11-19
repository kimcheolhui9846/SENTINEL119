import { useEffect, useState } from 'react';
import type { KeyboardEvent } from 'react';
import CompareView from './CompareView';
import DispatchView from './DispatchView';
import { api as compareApi } from './compareApi';
import type { CompareScenario } from './compareApi';
import './styles.css';

// Three screens: dispatch map for sample calls, model speed/accuracy comparison, free input.
const pages = ['dispatch', 'compare', 'play'] as const;
type Page = typeof pages[number];
const pageNames: Record<Page, string> = { dispatch: '출동 관제', compare: '모델 비교', play: '직접 입력' };

export default function App() {
  const [page, setPage] = useState<Page>(() => {
    const hash = window.location.hash.slice(1) as Page;
    return pages.includes(hash) ? hash : 'dispatch';
  });
  const [calls, setCalls] = useState<CompareScenario[] | null>(null);
  useEffect(() => { compareApi.scenarios().then(setCalls).catch(() => setCalls([])); }, []);

  function select(next: Page) {
    setPage(next);
    window.history.replaceState(null, '', '#' + next);
  }
  function switchTab(event: KeyboardEvent<HTMLButtonElement>) {
    if (!['ArrowLeft', 'ArrowRight'].includes(event.key)) return;
    event.preventDefault();
    const next = pages[(pages.indexOf(page) + (event.key === 'ArrowLeft' ? pages.length - 1 : 1)) % pages.length];
    select(next);
    document.getElementById(`tab-${next}`)?.focus();
  }
  const mapCalls = (calls ?? []).filter((call) => call.area);

  return (
    <div className="app-shell">
      <header className="masthead">
        <div className="brand"><span className="brand-mark" aria-hidden="true">+</span><span className="brand-name">SENTINEL 119</span></div>
        <nav className="nav-tabs" aria-label="화면">
          <div role="tablist" aria-label="화면 선택">
            {pages.map((tab) => (
              <button type="button" key={tab} id={`tab-${tab}`} role="tab" className={`nav-tab${page === tab ? ' active' : ''}`}
                aria-selected={page === tab} aria-controls={`page-${tab}`} tabIndex={page === tab ? 0 : -1}
                onKeyDown={switchTab} onClick={() => select(tab)}>{pageNames[tab]}</button>
            ))}
          </div>
        </nav>
        <span className="mode-indicator" data-mode="live">실시간 SENTINEL_ZORO</span>
      </header>
      <main className="main">
        {calls === null ? <p className="empty-state">시연 통화를 불러오는 중입니다.</p> : null}
        <div id="page-dispatch" role="tabpanel" aria-labelledby="tab-dispatch" hidden={page !== 'dispatch'}>
          {page === 'dispatch' && calls !== null && (mapCalls.length ? <DispatchView calls={mapCalls} />
            : <p className="empty-state">위치가 있는 시연 통화가 없습니다. SENTINEL_SCENARIO_DIRS를 설정하세요.</p>)}
        </div>
        <div id="page-compare" role="tabpanel" aria-labelledby="tab-compare" hidden={page !== 'compare'}>
          {page === 'compare' && calls !== null && (calls.length ? <CompareView scenarios={calls} />
            : <p className="empty-state">비교할 통화가 없습니다. SENTINEL_SCENARIO_DIRS를 설정하세요.</p>)}
        </div>
        <div id="page-play" role="tabpanel" aria-labelledby="tab-play" hidden={page !== 'play'}>
          {page === 'play' && <DispatchView calls={[]} live />}
        </div>
      </main>
    </div>
  );
}
