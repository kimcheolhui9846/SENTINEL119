import type React from 'react';
import { useCallback, useEffect, useRef, useState } from 'react';
import * as L from 'leaflet';
import { api } from './compareApi';
import type { CompareScenario, ScenarioDetail } from './compareApi';
import { advance, DISPATCH_THRESHOLD, emptyZoro, THRESHOLD } from './CompareView';
import type { Labels, ZoroState } from './CompareView';
import { dispatchApi } from './dispatchApi';
import type { Assignment, FleetUnit, Patrol, Point, TransferResponse } from './dispatchApi';
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

const HOSPITAL_ICON = '<path d="M4 21V5h16v16M2 21h20m-10-13v6m-3-3h6m-5 10v-4h4v4"/>';
function hospitalIcon(chosen: boolean) {
  return L.divIcon({
    className: `drill-marker dispatch-marker dispatch-marker--hospital ${chosen ? 'is-chosen' : ''}`, iconSize: chosen ? [38, 38] : [28, 28],
    html: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round" '
      + 'stroke-linecap="round" aria-hidden="true" focusable="false">' + HOSPITAL_ICON + '</svg>',
  });
}

function DispatchMap({ fleet, area, orders, patrols, positions, overlay, transfer }: {
  fleet: FleetUnit[]; area: ScenarioDetail['area'] | null; orders: Order[]; patrols: Patrol[];
  positions: React.MutableRefObject<Record<string, Point>>; overlay?: React.ReactNode; transfer: TransferResponse | null;
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

  // Transfer: candidate hospitals and the road route to Zoro's choice.
  const hospitalLayer = useRef<L.LayerGroup | null>(null);
  useEffect(() => {
    const s = state.current;
    if (!s) return;
    hospitalLayer.current?.remove();
    hospitalLayer.current = null;
    if (!transfer || transfer.status !== 'ok' || !area) return;
    const group = L.layerGroup().addTo(s.map);
    const chosen = transfer.zoro?.choice;
    for (const h of transfer.hospitals) {
      const p = transfer.zoro?.probabilities[h.hpid];
      L.marker([h.lat, h.lon], { icon: hospitalIcon(h.hpid === chosen), zIndexOffset: h.hpid === chosen ? 1800 : 900, keyboard: false })
        .bindTooltip(`${h.name} · ${h.level}${p !== undefined ? ` · Zoro ${Math.round(p * 100)}%` : ''}`,
          { permanent: h.hpid === chosen, direction: 'top', offset: [0, -18] }).addTo(group);
      if (h.hpid === chosen) {
        const path = h.route.path.map(latLng);
        L.polyline(path, { color: '#ffffff', weight: 8, opacity: 0.8, interactive: false }).addTo(group);
        L.polyline(path, { color: '#185c91', weight: 4.5, dashArray: '10 6', interactive: false }).addTo(group);
      }
    }
    hospitalLayer.current = group;
  }, [transfer, area]);

  // Incident area: the call's 법정동 and its centre.
  useEffect(() => {
    const s = state.current;
    if (!s) return;
    s.incident?.remove();
    s.incident = undefined;
    if (!area) return;
    const group = L.layerGroup().addTo(s.map);
    L.marker(latLng(area.center), { icon: icon('caller', 'is-caller', 40), zIndexOffset: 2000, keyboard: false })
      .bindTooltip(`신고 위치 · ${area.district} ${area.dong}`, { permanent: true, direction: 'top', offset: [0, -20] })
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
      {overlay}
      <ul className="dispatch-legend" aria-label="범례">
        {(['ambulance', 'pump', 'rescue'] as const).map((k) => <li key={k} data-kind={k}>{UNIT_LABEL[k]}</li>)}
        <li data-kind="caller">신고 위치 (동 안의 시연용 임의 지점)</li>
        {transfer?.status === 'ok' ? <li data-kind="hospital">이송 병원 (Zoro 선택)</li> : null}
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
  const [labels, setLabels] = useState<Labels | null>(null);
  const [transfer, setTransfer] = useState<TransferResponse | null>(null);
  const transferKey = useRef('');
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
    run.current += 1; sentKey.current = ''; transferKey.current = ''; setTransfer(null);
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

  // EMS only: once urgency is committed, pick the transfer hospital (KB rule + NMC live beds + Zoro choice).
  const syncTransfer = useCallback(async (state: ZoroState, turns: { speaker: 0 | 1; text: string }[]) => {
    const c = state.commits;
    if (c.dispatch?.label !== '구급' || !c.urgency || !call?.area) return;
    const key = [c.subtype?.label, c.symptom?.label, c.urgency.label].join('|');
    if (key === transferKey.current) return;
    transferKey.current = key;
    const myRun = run.current;
    try {
      const res = await dispatchApi.transfer({ ...call.area.center, turns, subtype: c.subtype?.label,
        symptom: c.symptom?.label, urgency: c.urgency.label });
      if (myRun === run.current) setTransfer(res);
    } catch { /* the dispatch view keeps working without hospital data */ }
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
      setLabels(res.labels);
      const s = advance(zoro.current.sft, res.results.sft.probabilities, turns.length, res.labels);
      const b = advance(zoro.current.base, res.results.base.probabilities, turns.length, res.labels);
      const nextSft = { ...s.next, latencies: [...zoro.current.sft.latencies, res.results.sft.latency_ms] };
      setSft(nextSft);
      setBase({ ...b.next, latencies: [...zoro.current.base.latencies, res.results.base.latency_ms] });
      setLines((l) => l.map((line, i) => (i === turns.length - 1
        ? { ...line, note: s.note, base: b.note, ms: res.results.sft.latency_ms } : line)));
      await syncDispatch(nextSft, turns.length);
      void syncTransfer(nextSft, turns);
    } catch {
      if (myRun === run.current) { setError('모델 서버 응답이 없습니다. 연결을 확인하세요.'); setPlaying(false); }
    } finally {
      if (myRun === run.current) setBusy(false);
    }
  }, [call, busy, lines, syncDispatch, syncTransfer]);

  const done = !!call && lines.length >= call.turns.length && !busy;
  useEffect(() => {
    if (!playing || busy) return;
    if (done) { setPlaying(false); setShowGold(true); return; }
    const id = window.setTimeout(step, lines.length ? TURN_DELAY_MS : 300);
    return () => window.clearTimeout(id);
  }, [playing, busy, done, step, lines.length]);

  const transcript = useRef<HTMLOListElement>(null);
  // Scroll only inside the transcript box; scrollIntoView would also move the page to the box.
  useEffect(() => {
    const box = transcript.current;
    if (box) box.scrollTo({ top: box.scrollHeight, behavior: 'smooth' });
  }, [lines.length]);

  const c = sft.commits;
  const probs = sft.probs;
  const subQ = c.dispatch ? ({ ems: 'ems_subtype', rescue: 'rescue_subtype' } as Record<string, string>)[c.dispatch.key] : undefined;
  const baseFinal = [base.commits.dispatch?.label, base.commits.subtype?.label, base.commits.urgency?.label].map((x) => x ?? '—').join(' · ');
  const gold = call?.gold;
  const last = lines.at(-1);
  const pct = (x: number) => `${Math.round(x * 100)}%`;
  const dist = (qid: string) => probs?.[qid as keyof NonNullable<typeof probs>] as Record<string, number> | undefined;
  const need = (qid: string, key: string) => qid === 'dispatch' ? DISPATCH_THRESHOLD[key] ?? THRESHOLD.dispatch
    : qid === 'urgency' ? THRESHOLD.urgency : qid === 'ems_symptom' ? THRESHOLD.symptom : THRESHOLD.subtype;

  // Banner: what the model decided on the latest turn, in one line.
  let banner: { state: 'wait' | 'commit' | 'idle'; title: string; detail: string };
  if (!last) banner = { state: 'idle', title: '대기', detail: '재생을 누르면 신고 통화가 한 턴씩 들어옵니다.' };
  else if (!last.note) banner = { state: 'idle', title: '판단 중…', detail: `${lines.length}턴까지의 대화를 보고 있습니다.` };
  else if (last.note !== 'WAIT') {
    const sent = orders.filter((o) => o.turn === lines.length && o.unit);
    banner = { state: 'commit', title: last.note.replace('COMMIT · ', 'COMMIT  '),
      detail: sent.length ? `→ ${sent.map((o) => o.unit?.name).join(', ')} 출동`
        : c.dispatch?.label === '기타' ? '→ 출동 없이 상담·이첩' : '' };
  } else {
    const open: [string, string] | null = !c.dispatch ? ['dispatch', '출동유형'] : subQ && !c.subtype ? [subQ, '세부유형']
      : !c.urgency ? ['urgency', '긴급도'] : null;
    const d = open ? dist(open[0]) : undefined;
    const best = d ? Object.entries(d).sort((a, b) => b[1] - a[1])[0] : undefined;
    banner = { state: 'wait', title: 'WAIT  더 듣기',
      detail: open && best && labels ? `${open[1]} 1순위 ${labels[open[0]]?.[best[0]] ?? best[0]} ${pct(best[1])} · 확정 기준 ${pct(need(open[0], best[0]))} 미만`
        : '추가로 확정할 항목 없이 다음 발화를 기다립니다.' };
  }

  const bars = (qid: string, title: string, commit?: { key: string; turn: number }) => {
    const d = dist(qid);
    if (!d || !labels?.[qid]) return null;
    const rows = Object.entries(d).sort((a, b) => b[1] - a[1]).slice(0, 4);
    return (
      <div className="prob-block" data-state={commit ? 'commit' : 'wait'}>
        <div className="prob-head"><span>{title}</span>
          <strong>{commit ? `${labels[qid][commit.key] ?? commit.key} · ${commit.turn}턴 확정` : 'WAIT'}</strong></div>
        {rows.map(([key, p]) => (
          <div key={key} className="prob-row" data-picked={commit?.key === key}>
            <span className="prob-label">{labels[qid][key] ?? key}</span>
            <span className="prob-track"><span className="prob-fill" style={{ width: pct(p) }} />
              <span className="prob-need" style={{ left: pct(need(qid, key)) }} aria-hidden="true" /></span>
            <span className="prob-value tabular">{pct(p)}</span>
          </div>
        ))}
      </div>
    );
  };

  const overlay = (
    <>
      <div className="map-banner" data-state={banner.state} key={`b-${lines.length}-${banner.state}`} aria-live="polite">
        <span className="map-banner-tag">SENTINEL_ZORO{last?.ms ? ` · ${Math.round(last.ms)}ms` : ''}</span>
        <strong>{banner.title}</strong>
        {banner.detail ? <span className="map-banner-detail">{banner.detail}</span> : null}
      </div>
      {last ? (
        <div className="map-caption" data-speaker={last.speaker} key={`c-${lines.length}`}>
          <span className="map-caption-who">{last.speaker === 0 ? '수보요원' : '신고자'} · {lines.length}턴</span>
          <p>{last.text}</p>
        </div>
      ) : null}
    </>
  );

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
        <DispatchMap fleet={fleet} area={lines.length ? call?.area ?? null : null} orders={orders} patrols={patrols}
          positions={positions} overlay={overlay} transfer={transfer} />
        <aside className="dispatch-side">
          <section className="panel dispatch-card">
            <header className="panel-header"><h2 className="section-title">SENTINEL_ZORO 판단</h2>
              <span className="meta tabular">{sft.latencies.length
                ? `평균 ${Math.round(sft.latencies.reduce((a, b) => a + b, 0) / sft.latencies.length)}ms / 턴` : '대기'}</span></header>
            {probs ? (
              <>
                {bars('dispatch', '출동유형', c.dispatch)}
                {subQ ? bars(subQ, '세부유형', c.subtype) : null}
                {c.dispatch ? bars('urgency', '긴급도', c.urgency) : null}
                {subQ === 'ems_subtype' ? bars('ems_symptom', '주증상', c.symptom) : null}
              </>
            ) : <p className="meta">첫 발화가 들어오면 확률이 표시됩니다. 막대의 세로선이 확정 기준입니다.</p>}
            <p className="meta">같은 대화에서 base Zoro 확정: {baseFinal}</p>
          </section>
          {gold ? (
            <section className="panel dispatch-card">
              <button type="button" className="button secondary" onClick={() => setShowGold((v) => !v)}>{showGold ? '정답 숨기기' : '정답 라벨 보기'}</button>
              {showGold ? <p className="dispatch-gold">AI Hub 라벨: {[gold.dispatch, gold.ems_subtype ?? gold.rescue_subtype, `긴급도 ${gold.urgency}`].filter(Boolean).join(' · ')}</p> : null}
            </section>
          ) : null}
        </aside>
      </div>
      <div className="dispatch-bottom">
          <section className="panel dispatch-card dispatch-transcript">
            <header className="panel-header"><h2 className="section-title">신고 통화</h2>
              <span className="meta tabular">{lines.length} / {call?.turns.length ?? 0}턴</span></header>
            <ol ref={transcript}>
              {lines.map((line, i) => (
                <li key={i} data-speaker={line.speaker}>
                  <span className="who">{line.speaker === 0 ? '수보요원' : '신고자'} · {i + 1}턴</span>
                  <p>{line.text}</p>
                  {line.note ? <span className={`dt-note ${line.note === 'WAIT' ? 'is-wait' : 'is-commit'}`}>{line.note}</span> : null}
                </li>
              ))}
              {!lines.length ? <li className="meta">재생을 누르면 통화가 한 턴씩 들어오고, 턴마다 SENTINEL_ZORO가 판단합니다.</li> : null}
            </ol>
          </section>
          <section className="panel dispatch-card dispatch-transfer">
            <header className="panel-header"><h2 className="section-title">이송 병원</h2>
              <span className="meta">{transfer?.zoro?.latency_ms ? `Zoro ${Math.round(transfer.zoro.latency_ms)}ms` : ''}</span></header>
            {!transfer ? <p className="meta">구급 출동에서 긴급도가 확정되면, 지침 기준과 응급실 실시간 병상으로 후보를 고르고 Zoro가 이송 병원을 선택합니다.</p>
              : transfer.status === 'unconfigured' ? <p className="meta">국립중앙의료원 API 키(NMC_SERVICE_KEY)가 설정되지 않아 병원 정보를 가져오지 못했습니다.</p>
              : transfer.status === 'unavailable' ? <p className="meta">국립중앙의료원 API 응답이 없습니다. 잠시 뒤 다시 시도합니다.</p> : (
              <>
                <p className="transfer-rule"><strong>{transfer.rule.category}</strong> · {transfer.rule.text}</p>
                <p className="do-source">근거: {transfer.rule.source}</p>
                {!transfer.hospitals.length ? <p className="meta">기준에 맞고 응급실 병상이 있는 병원이 주변에 없습니다.</p> : (
                  <ol className="transfer-list">
                    {transfer.hospitals.map((h) => {
                      const p = transfer.zoro?.probabilities[h.hpid] ?? 0;
                      const chosen = transfer.zoro?.choice === h.hpid;
                      return (
                        <li key={h.hpid} data-chosen={chosen}>
                          <div className="do-head"><strong>{chosen ? '✓ ' : ''}{h.name}</strong><span className="tabular">{Math.round(p * 100)}%</span></div>
                          <span className="prob-track"><span className="prob-fill" style={{ width: `${Math.round(p * 100)}%` }} /></span>
                          <p className="meta">{h.context.slice(h.name.length + 3)}</p>
                        </li>
                      );
                    })}
                  </ol>
                )}
                <p className="do-source">{transfer.source} · 위 병원별 정보가 Zoro 선택지 문구로 들어가며, 병원 선택은 공개 Zoro 결정 헤드(학습하지 않은 과제)입니다.</p>
              </>
            )}
          </section>
          <section className="panel dispatch-card">
            <header className="panel-header"><h2 className="section-title">출동 지령</h2>
              <span className="meta">{call?.area ? `${call.area.district} ${call.area.dong}` : ''}</span></header>
            {!orders.length ? <p className="meta">{c.dispatch ? (c.dispatch.label === '기타' ? '기타: 출동 없이 상담·이첩' : '배차 계산 중…')
              : '출동유형이 확정되면 가장 빨리 도착하는 차량을 보냅니다.'}</p> : (
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
      </div>
    </div>
  );
}
