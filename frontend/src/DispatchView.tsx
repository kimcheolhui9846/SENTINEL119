import type React from 'react';
import { useCallback, useEffect, useRef, useState } from 'react';
import * as L from 'leaflet';
import { api } from './compareApi';
import type { CompareScenario, ScenarioDetail } from './compareApi';
import { advance, emptyZoro, THRESHOLD } from './CompareView';
import type { ZoroState } from './CompareView';
import { dispatchApi } from './dispatchApi';
import type { Assignment, FleetUnit, Patrol, Point } from './dispatchApi';
import 'leaflet/dist/leaflet.css';
import './drill.css';
import './dispatch.css';

const PICTOGRAMS = {
  caller: '<path d="M5 4h14a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2h-7l-5 4v-4H5a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2Z"/><path d="M8 11h8"/>',
  ambulance: '<path d="M2 6h12v11H2zM14 9h4l4 4v4h-8M6 9v5m-2.5-2.5h5"/><circle cx="6" cy="18" r="2" fill="currentColor"/><circle cx="18" cy="18" r="2" fill="currentColor"/>',
  pump: '<path d="M2 9h13v8H2zM15 10h4l3 4v3h-7M3 5h13m-11 0v4m4-4v4m4-4v4"/><circle cx="6" cy="18" r="2" fill="currentColor"/><circle cx="18" cy="18" r="2" fill="currentColor"/>',
  rescue: '<path d="M2 9h13v8H2zM15 10h4l3 4v3h-7M5 9V5l8 4"/><circle cx="6" cy="18" r="2" fill="currentColor"/><circle cx="18" cy="18" r="2" fill="currentColor"/>',
} as const;
const UNIT_LABEL = { ambulance: '구급차', pump: '펌프차', rescue: '구조공작차' } as const;
const ROUTE_COLOR = { ambulance: '#007f77', pump: '#b23a1c', rescue: '#8a5a00' } as const;
const ROUTE_SOURCE = { tmap: 'TMAP 도로 경로', osrm: 'OSM 도로 경로', straight_line: '직선 추정' } as const;
// Real ETAs are minutes; the map replays them compressed so a dispatch is visible during the call.
const TIME_SCALE = 12;
const TURN_DELAY_MS = 1400;

function icon(kind: keyof typeof PICTOGRAMS, extra = '', size = 30) {
  return L.divIcon({
    className: `drill-marker dispatch-marker dispatch-marker--${kind} ${extra}`, iconSize: [size, size],
    html: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round" '
      + 'stroke-linecap="round" aria-hidden="true" focusable="false">' + PICTOGRAMS[kind] + '</svg>',
  });
}
const latLng = (p: Point) => L.latLng(p.lat, p.lon);

interface Moving { id: string; path: L.LatLng[]; cumulative: number[]; total: number; start: number; ms: number }
function moving(id: string, path: Point[], durationS: number, start: number): Moving {
  const pts = path.map(latLng);
  const cumulative = [0];
  for (let i = 1; i < pts.length; i += 1) cumulative.push(cumulative[i - 1] + pts[i - 1].distanceTo(pts[i]));
  return { id, path: pts, cumulative, total: cumulative.at(-1) ?? 0, start, ms: Math.max(5000, (durationS * 1000) / TIME_SCALE) };
}
function positionAt(m: Moving, now: number, loop = false) {
  const raw = (now - m.start) / m.ms;
  const t = loop ? raw - Math.floor(raw) : Math.min(1, raw);
  const d = m.total * t;
  const i = Math.max(1, m.cumulative.findIndex((c) => c >= d));
  const [a, b] = [m.path[i - 1], m.path[i] ?? m.path[i - 1]];
  const span = m.cumulative[i] - m.cumulative[i - 1] || 1;
  const f = Math.min(1, (d - m.cumulative[i - 1]) / span);
  return { at: L.latLng(a.lat + (b.lat - a.lat) * f, a.lng + (b.lng - a.lng) * f), done: t >= 1 };
}

interface Line { speaker: 0 | 1; text: string; note?: string; base?: string; ms?: number }
interface Order extends Assignment { turn: number; dispatchedAt: number }

function DispatchMap({ fleet, area, orders, patrols, positions }: {
  fleet: FleetUnit[]; area: ScenarioDetail['area'] | null; orders: Order[]; patrols: Patrol[];
  positions: React.MutableRefObject<Record<string, Point>>;
}) {
  const element = useRef<HTMLDivElement>(null);
  const state = useRef<{ map: L.Map; markers: Map<string, L.Marker>; incident?: L.LayerGroup; routes: L.LayerGroup;
    moving: Map<string, Moving>; patrol: Map<string, Moving> } | null>(null);
  const [tileError, setTileError] = useState(false);

  useEffect(() => {
    if (!element.current) return;
    const map = L.map(element.current, { zoomSnap: 0.5, minZoom: 10, maxZoom: 18, scrollWheelZoom: true })
      .setView([37.5665, 126.978], 11);
    map.attributionControl.setPrefix(false);
    L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19, attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> · 소방관서 위치 OSM',
    }).on('tileerror', () => setTileError(true)).addTo(map);
    void fetch(new URL('./seoul-districts.geojson', import.meta.url).href).then(async (r) => {
      const geo = r.ok ? await r.json() : null;
      if (geo?.type === 'FeatureCollection') {
        L.geoJSON(geo, { interactive: false, style: { color: '#6f8797', weight: 1, fillOpacity: 0 } }).addTo(map).bringToBack();
      }
    }).catch(() => {});
    const routes = L.layerGroup().addTo(map);
    state.current = { map, markers: new Map(), routes, moving: new Map(), patrol: new Map() };
    let frame = 0;
    const render = (now: number) => {
      const s = state.current;
      if (s) {
        for (const m of s.moving.values()) s.markers.get(m.id)?.setLatLng(positionAt(m, now).at);
        for (const m of s.patrol.values()) {
          if (s.moving.has(m.id)) continue;
          const at = positionAt(m, now, true).at;
          s.markers.get(m.id)?.setLatLng(at);
          positions.current[m.id] = { lat: at.lat, lon: at.lng };
        }
      }
      frame = requestAnimationFrame(render);
    };
    frame = requestAnimationFrame(render);
    const observer = new ResizeObserver(() => map.invalidateSize({ debounceMoveend: true }));
    observer.observe(element.current);
    return () => { cancelAnimationFrame(frame); observer.disconnect(); state.current = null; map.remove(); };
  }, []);

  // Units wait at their stations.
  useEffect(() => {
    const s = state.current;
    if (!s) return;
    for (const marker of s.markers.values()) marker.remove();
    s.markers.clear();
    for (const unit of fleet) {
      const offset = unit.kind === 'ambulance' ? 0 : unit.kind === 'pump' ? 0.00035 : -0.00035;
      s.markers.set(unit.id, L.marker([unit.lat, unit.lon + offset], { icon: icon(unit.kind, 'is-idle', 22), keyboard: false })
        .bindTooltip(unit.name).addTo(s.map));
    }
  }, [fleet]);

  // Nearby units drive road loops (demo effect) until one is dispatched.
  useEffect(() => {
    const s = state.current;
    if (!s) return;
    for (const id of s.patrol.keys()) {
      const unit = fleet.find((u) => u.id === id);
      if (unit && !s.moving.has(id)) { s.markers.get(id)?.setLatLng([unit.lat, unit.lon]).setIcon(icon(unit.kind, 'is-idle', 22)); }
    }
    s.patrol.clear();
    positions.current = {};
    const start = performance.now();
    for (const p of patrols) {
      const unit = fleet.find((u) => u.id === p.id);
      if (!unit || p.path.length < 2) continue;
      s.patrol.set(p.id, { ...moving(p.id, p.path, p.duration_s, start - Math.random() * 20000), ms: Math.max(20000, (p.duration_s * 1000) / TIME_SCALE) });
      if (!s.moving.has(p.id)) s.markers.get(p.id)?.setIcon(icon(unit.kind, 'is-patrol', 26));
    }
  }, [patrols, fleet, positions]);

  // Incident area: the call's 법정동 and its centre.
  useEffect(() => {
    const s = state.current;
    if (!s) return;
    s.incident?.remove();
    s.incident = undefined;
    if (!area) return;
    const group = L.layerGroup().addTo(s.map);
    L.geoJSON({ type: 'Feature', properties: {}, geometry: area.geometry } as GeoJSON.Feature, {
      interactive: false, style: { color: '#613ca4', weight: 2.5, dashArray: '6 4', fillColor: '#8b6cc6', fillOpacity: 0.18 },
    }).addTo(group);
    L.marker(latLng(area.center), { icon: icon('caller', 'is-caller', 40), zIndexOffset: 2000, keyboard: false })
      .bindTooltip(`신고 위치 · ${area.district} ${area.dong} 일대`, { permanent: true, direction: 'top', offset: [0, -20] })
      .addTo(group);
    s.incident = group;
    s.map.flyTo(latLng(area.center), 14.5, { duration: 1.2 });
  }, [area]);

  // Each new order draws its road route and drives the unit along it.
  useEffect(() => {
    const s = state.current;
    if (!s || !area) return;
    s.routes.clearLayers();
    const bounds = L.latLngBounds([latLng(area.center)]);
    for (const order of orders) {
      if (!order.unit || !order.route) continue;
      const path = order.route.path.map(latLng);
      L.polyline(path, { color: '#ffffff', weight: 8, opacity: 0.8, interactive: false }).addTo(s.routes);
      L.polyline(path, { color: ROUTE_COLOR[order.kind], weight: 4.5, opacity: 0.95, interactive: false }).addTo(s.routes);
      path.forEach((p) => bounds.extend(p));
      const marker = s.markers.get(order.unit.id);
      if (marker && !s.moving.has(order.unit.id)) {
        delete positions.current[order.unit.id];
        marker.setIcon(icon(order.kind, 'is-dispatched', 34));
        marker.setZIndexOffset(1500);
        s.moving.set(order.unit.id, moving(order.unit.id, order.route.path, order.route.duration_s, order.dispatchedAt));
      }
    }
    for (const [id, m] of s.moving) {
      if (!orders.some((o) => o.unit?.id === id)) {
        s.moving.delete(id);
        const unit = fleet.find((u) => u.id === id);
        const marker = s.markers.get(id);
        if (unit && marker) {
          marker.setLatLng(s.patrol.has(id) ? m.path[0] : [unit.lat, unit.lon]);
          marker.setIcon(icon(unit.kind, s.patrol.has(id) ? 'is-patrol' : 'is-idle', s.patrol.has(id) ? 26 : 22));
          marker.setZIndexOffset(0);
        }
      }
    }
    if (orders.length) s.map.flyToBounds(bounds, { padding: [50, 50], maxZoom: 15.5, duration: 1.0 });
  }, [orders, area, fleet, positions]);

  return (
    <div className="dispatch-map-wrap">
      <div ref={element} className="drill-map dispatch-map" role="region" aria-label="출동 관제 지도" />
      {tileError ? <p className="dispatch-map-alert">지도 배경을 불러오지 못해 경계와 위치만 표시합니다.</p> : null}
      <ul className="dispatch-legend" aria-label="범례">
        {(['ambulance', 'pump', 'rescue'] as const).map((k) => <li key={k} data-kind={k}>{UNIT_LABEL[k]}</li>)}
        <li data-kind="caller">신고 위치(법정동)</li>
        {patrols.length ? <li className="legend-note">차량 순찰은 시연용 연출 · 서울 {patrols.length}대</li> : null}
      </ul>
    </div>
  );
}

function Countdown({ order, now }: { order: Order; now: number }) {
  if (!order.route) return null;
  const ms = Math.max(5000, (order.route.duration_s * 1000) / TIME_SCALE);
  const left = Math.max(0, 1 - (now - order.dispatchedAt) / ms) * order.route.duration_s;
  return <span className="tabular">{left > 0 ? `도착까지 ${Math.floor(left / 60)}분 ${Math.round(left % 60)}초` : '현장 도착'}</span>;
}

export default function DispatchView({ calls }: { calls: CompareScenario[] }) {
  const [callId, setCallId] = useState(calls.find((c) => c.area)?.id ?? calls[0]?.id ?? '');
  const [call, setCall] = useState<ScenarioDetail | null>(null);
  const [fleet, setFleet] = useState<FleetUnit[]>([]);
  const [lines, setLines] = useState<Line[]>([]);
  const [sft, setSft] = useState<ZoroState>(emptyZoro());
  const [base, setBase] = useState<ZoroState>(emptyZoro());
  const [orders, setOrders] = useState<Order[]>([]);
  const [playing, setPlaying] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showGold, setShowGold] = useState(false);
  const [now, setNow] = useState(performance.now());
  const [patrols, setPatrols] = useState<Patrol[]>([]);
  const positions = useRef<Record<string, Point>>({});
  const run = useRef(0);
  const sentKey = useRef('');
  const zoro = useRef({ sft, base });
  zoro.current = { sft, base };
  const ordersRef = useRef(orders);
  ordersRef.current = orders;

  useEffect(() => {
    dispatchApi.fleet().then((f) => setFleet(f.units)).catch(() => setError('소방관서 자료를 불러오지 못했습니다.'));
    dispatchApi.patrols().then((p) => setPatrols(p.patrols)).catch(() => setPatrols([]));
  }, []);
  useEffect(() => { const id = window.setInterval(() => setNow(performance.now()), 500); return () => window.clearInterval(id); }, []);

  const reset = useCallback(() => {
    run.current += 1; sentKey.current = '';
    setPlaying(false); setBusy(false); setLines([]); setSft(emptyZoro()); setBase(emptyZoro()); setOrders([]);
    setError(null); setShowGold(false);
  }, []);
  useEffect(() => {
    if (!callId) return;
    reset();
    api.scenario(callId).then(setCall).catch(() => setError('통화를 불러오지 못했습니다.'));
  }, [callId, reset]);

  // Dispatch when the type commits; subtype / urgency / symptom commits may add units (e.g. 심정지 → 펌뷸런스).
  const syncDispatch = useCallback(async (state: ZoroState, turn: number) => {
    const c = state.commits;
    if (!c.dispatch || !call?.area) return;
    const key = [c.dispatch.label, c.subtype?.label, c.symptom?.label, c.urgency?.label].join('|');
    if (key === sentKey.current) return;
    sentKey.current = key;
    const myRun = run.current;
    const current = ordersRef.current;
    const res = await dispatchApi.dispatch({
      ...call.area.center, dispatch: c.dispatch.label, subtype: c.subtype?.label, symptom: c.symptom?.label,
      urgency: c.urgency?.label, busy: current.flatMap((o) => (o.unit ? [o.unit.id] : [])), skip: current.length,
      positions: { ...positions.current },
    });
    if (myRun !== run.current) return;
    const at = performance.now();
    setOrders((prev) => [
      ...prev.map((o, i) => ({ ...o, ...(res.plan[i] ? { role: res.plan[i].role, reason: res.plan[i].reason, source: res.plan[i].source } : {}) })),
      ...res.assignments.map((a) => ({ ...a, turn, dispatchedAt: at })),
    ]);
  }, [call]);

  const step = useCallback(async () => {
    if (!call || busy || lines.length >= call.turns.length) return;
    const myRun = run.current;
    const turns = [...lines.map(({ speaker, text }) => ({ speaker, text })), call.turns[lines.length]];
    setLines((l) => [...l, { ...call.turns[lines.length] }]);
    setBusy(true); setError(null);
    try {
      const res = await api.streamDecide(turns);
      if (myRun !== run.current) return;
      const s = advance(zoro.current.sft, res.results.sft.probabilities, turns.length, res.labels);
      const b = advance(zoro.current.base, res.results.base.probabilities, turns.length, res.labels);
      const nextSft = { ...s.next, latencies: [...zoro.current.sft.latencies, res.results.sft.latency_ms] };
      setSft(nextSft);
      setBase({ ...b.next, latencies: [...zoro.current.base.latencies, res.results.base.latency_ms] });
      setLines((l) => l.map((line, i) => (i === turns.length - 1
        ? { ...line, note: s.note, base: b.note, ms: res.results.sft.latency_ms } : line)));
      await syncDispatch(nextSft, turns.length);
    } catch {
      if (myRun === run.current) { setError('모델 서버 응답이 없습니다. 연결을 확인하세요.'); setPlaying(false); }
    } finally {
      if (myRun === run.current) setBusy(false);
    }
  }, [call, busy, lines, syncDispatch]);

  const done = !!call && lines.length >= call.turns.length && !busy;
  useEffect(() => {
    if (!playing || busy) return;
    if (done) { setPlaying(false); setShowGold(true); return; }
    const id = window.setTimeout(step, lines.length ? TURN_DELAY_MS : 300);
    return () => window.clearTimeout(id);
  }, [playing, busy, done, step, lines.length]);

  const transcript = useRef<HTMLOListElement>(null);
  useEffect(() => { transcript.current?.lastElementChild?.scrollIntoView({ block: 'nearest' }); }, [lines.length]);

  const c = sft.commits;
  const chip = (label: string, commit?: { label: string; turn: number }, threshold?: number) => (
    <div className="dispatch-decision" data-state={commit ? 'commit' : 'wait'}>
      <span className="dd-name">{label}</span>
      <strong>{commit ? commit.label : 'WAIT'}</strong>
      <span className="dd-meta tabular">{commit ? `${commit.turn}턴 확정` : threshold ? `확정 기준 ${Math.round(threshold * 100)}%` : ''}</span>
    </div>
  );
  const subQ = c.dispatch ? ({ ems: 'ems_subtype', rescue: 'rescue_subtype' } as Record<string, string>)[c.dispatch.key] : undefined;
  const baseFinal = [base.commits.dispatch?.label, base.commits.subtype?.label, base.commits.urgency?.label].map((x) => x ?? '—').join(' · ');
  const gold = call?.gold;

  return (
    <div className="dispatch">
      <div className="dispatch-bar">
        <label>시연 통화
          <select value={callId} onChange={(e) => setCallId(e.target.value)}>
            {calls.map((s) => <option key={s.id} value={s.id}>{s.private ? '★ ' : ''}{s.title}</option>)}
          </select>
        </label>
        <button type="button" className="button primary" disabled={!call || done} onClick={() => setPlaying((p) => !p)}>{playing ? '멈춤' : '재생'}</button>
        <button type="button" className="button secondary" disabled={!call || busy || playing || done} onClick={() => void step()}>다음 발화</button>
        <button type="button" className="button secondary" onClick={reset}>처음부터</button>
        <span className="meta">{call?.private ? '실제 통화 · AI Hub Validation(학습 미사용) · 로컬 시연 전용' : '가상 통화'}</span>
      </div>
      {error ? <div className="alert" role="alert">{error}</div> : null}
      <div className="dispatch-grid">
        <DispatchMap fleet={fleet} area={lines.length ? call?.area ?? null : null} orders={orders} patrols={patrols} positions={positions} />
        <aside className="dispatch-side">
          <section className="panel dispatch-card">
            <header className="panel-header"><h2 className="section-title">SENTINEL_ZORO 판단</h2>
              <span className="meta tabular">{sft.latencies.length ? `${Math.round(sft.latencies.at(-1) ?? 0)}ms / 턴` : '대기'}</span></header>
            <div className="dispatch-decisions">
              {chip('출동유형', c.dispatch, THRESHOLD.dispatch)}
              {subQ ? chip('세부유형', c.subtype, THRESHOLD.subtype) : null}
              {c.dispatch ? chip('긴급도', c.urgency, THRESHOLD.urgency) : null}
              {subQ === 'ems_subtype' ? chip('주증상', c.symptom, THRESHOLD.symptom) : null}
            </div>
            <p className="meta">base Zoro 확정: {baseFinal}</p>
          </section>
          <section className="panel dispatch-card">
            <header className="panel-header"><h2 className="section-title">출동 지령</h2>
              <span className="meta">{call?.area ? `${call.area.district} ${call.area.dong} 일대` : ''}</span></header>
            {!orders.length ? <p className="meta">{c.dispatch ? (c.dispatch.label === '기타' ? '기타: 출동 없이 상담·이첩' : '배차 계산 중…') : '출동유형이 확정되면 가장 가까운 차량을 보냅니다.'}</p> : (
              <ol className="dispatch-orders">
                {orders.map((o, i) => (
                  <li key={i} data-kind={o.kind}>
                    <div className="do-head"><strong>{o.unit ? o.unit.name : `${UNIT_LABEL[o.kind]} 없음`}</strong><Countdown order={o} now={now} /></div>
                    <p>{o.role} · {o.reason}</p>
                    <p className="meta">{o.turn}턴 지령 · {o.route ? `${(o.route.distance_m / 1000).toFixed(1)}km · ${ROUTE_SOURCE[o.route.source]}` : ''}</p>
                    <p className="do-source">근거: {o.source}</p>
                  </li>
                ))}
              </ol>
            )}
          </section>
          <section className="panel dispatch-card dispatch-transcript">
            <header className="panel-header"><h2 className="section-title">신고 통화</h2>
              <span className="meta tabular">{lines.length} / {call?.turns.length ?? 0}턴</span></header>
            <ol ref={transcript}>
              {lines.map((line, i) => (
                <li key={i} data-speaker={line.speaker}>
                  <span className="who">{line.speaker === 0 ? '수보요원' : '신고자'}</span> {line.text}
                  {line.note ? <span className={`dt-note ${line.note === 'WAIT' ? 'is-wait' : 'is-commit'}`}>{line.note}</span> : null}
                </li>
              ))}
              {!lines.length ? <li className="meta">재생을 누르면 통화가 한 턴씩 들어오고, 턴마다 SENTINEL_ZORO가 판단합니다.</li> : null}
            </ol>
          </section>
          {gold ? (
            <section className="panel dispatch-card">
              <button type="button" className="button secondary" onClick={() => setShowGold((v) => !v)}>{showGold ? '정답 숨기기' : '정답 라벨 보기'}</button>
              {showGold ? <p className="dispatch-gold">AI Hub 라벨: {[gold.dispatch, gold.ems_subtype ?? gold.rescue_subtype, `긴급도 ${gold.urgency}`].filter(Boolean).join(' · ')}</p> : null}
            </section>
          ) : null}
        </aside>
      </div>
    </div>
  );
}
