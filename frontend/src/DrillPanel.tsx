import { useEffect, useRef, useState } from 'react';
import * as L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import './drill.css';

type Mode = 'live' | 'simulation';
type Scenario = 'ems' | 'fire';
type Point = { lat: number; lon: number };
type Unit = {
  id: string; name: string; kind: 'ambulance' | 'fire'; crew: string[]; equipment: string[];
  route: Point[]; status: 'patrol' | 'dispatched';
};
type Hospital = Point & {
  id: string; name: string; address: string; capabilities: string[]; available: boolean;
  location_source: 'nmc'; operational_source: 'simulation';
};
type LocationBasis = {
  dataset_id: string; level: 'collection_city' | 'label_dong'; city: string;
  district: string | null; dong: string | null; count: number;
  zone: { city: string; district: string; dong: string };
};
type Drill = {
  id: string; mode: Mode; scenario: Scenario; turn: number; total_turns: number;
  caller: Point & { dong: string; source: 'synthetic_in_dong'; location_basis: LocationBasis };
  turns: { speaker: string; text: string }[];
  units: Unit[]; hospitals: Hospital[];
  hospital_source: { provider: 'nmc'; status: 'live' | 'snapshot' | 'unavailable'; fetched_at: string | null };
  dispatch: { state: 'wait' | 'committed'; category: Scenario | null; unit_id: string | null; evidence_turn: number | null };
  subtype: { state: 'wait' | 'committed'; code: 'cardiac_arrest' | null };
  link: { state: 'wait' | 'committed'; hospital_id: string | null };
  model_id: string | null; model_status: 'pending' | 'live' | 'simulation';
};
type Problem = { kind: 'model' | 'turn' | 'start' | 'expired'; message: string };
type Selection = { id: string; revealMap?: boolean };
type MapFrame = 'scene' | 'nearby' | 'all';
const MODE_LABELS = { live: '실시간 Zoro', simulation: '오프라인 시뮬레이션' };
const CATEGORY_LABELS = { ems: '구급', fire: '화재' };
const UNIT_LABELS = { ambulance: '구급차', fire: '소방차' };
const CAPABILITY_LABELS: Record<string, string> = { cardiac_arrest: '심정지 대응', injury: '외상 대응' };
const toLatLng = ({ lat, lon }: Point) => L.latLng(lat, lon);
const CALLER_ACTION = '신고자 마커로 다음 발화 공개';
const HOSPITAL_SOURCE = 'https://www.data.go.kr/data/15000736/openapi.do';
const STATION_PHOTO = new URL('./assets/drill-station.webp', import.meta.url).href;
const ZONE_SOURCE = new URL('../../backend/data/drill-dongs.geojson', import.meta.url).href;
const DATE_FORMAT = new Intl.DateTimeFormat('ko-KR', {
  dateStyle: 'medium', timeStyle: 'short', timeZone: 'Asia/Seoul',
});
const PICTOGRAMS = {
  caller: '<path d="M5 4h14a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2h-7l-5 4v-4H5a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2Z"/><path d="M8 11h8"/>',
  ambulance: '<path d="M2 6h12v11H2zM14 9h4l4 4v4h-8M6 9v5m-2.5-2.5h5"/><circle cx="6" cy="18" r="2" fill="currentColor"/><circle cx="18" cy="18" r="2" fill="currentColor"/>',
  fire: '<path d="M2 9h13v8H2zM15 10h4l3 4v3h-7M3 5h13m-11 0v4m4-4v4m4-4v4"/><circle cx="6" cy="18" r="2" fill="currentColor"/><circle cx="18" cy="18" r="2" fill="currentColor"/>',
  hospital: '<path d="M4 21V5h16v16M2 21h20m-10-13v6m-3-3h6m-5 10v-4h4v4"/>',
};

function actualHospitals(drill: Drill) {
  return drill.hospital_source?.provider === 'nmc' && drill.hospital_source.status !== 'unavailable'
    ? drill.hospitals.filter((item) => item.location_source === 'nmc')
    : [];
}

function trainingSetting(facility: Hospital) {
  if (facility.operational_source !== 'simulation') return '확인된 훈련 조건 없음';
  const capabilities = facility.capabilities.map((key) => CAPABILITY_LABELS[key] ?? key);
  return [facility.available ? '모의 연결 가능' : '모의 연결 불가', ...capabilities].join(' · ');
}

class DrillError extends Error {
  constructor(readonly code: string) { super(code); }
}

async function requestDrill(path: string, body: object | undefined, signal: AbortSignal) {
  const response = await fetch('/api/drills' + path, {
    signal, method: body ? 'POST' : 'GET',
    headers: body ? { 'Content-Type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new DrillError(error.detail ?? 'request_failed');
  }
  return await response.json() as Drill;
}

function routeSamples(route: L.LatLng[]) {
  let distance = 0;
  return route.map((point, index) => {
    if (index) distance += route[index - 1].distanceTo(point);
    return { point, distance };
  });
}

function interpolate(from: L.LatLng, to: L.LatLng, ratio: number) {
  return L.latLng(from.lat + (to.lat - from.lat) * ratio, from.lng + (to.lng - from.lng) * ratio);
}

function patrolAt(samples: ReturnType<typeof routeSamples>, elapsed: number, period: number) {
  const fraction = 1 - Math.abs(1 - (elapsed / period) % 2);
  const distance = (samples.at(-1)?.distance ?? 0) * fraction;
  const index = samples.findIndex((sample) => sample.distance >= distance);
  if (index <= 0) return samples[0].point;
  const [before, next] = [samples[index - 1], samples[index]];
  return interpolate(before.point, next.point, (distance - before.distance) / (next.distance - before.distance));
}

function markerIcon(kind: keyof typeof PICTOGRAMS, size = 36) {
  return L.divIcon({
    className: `drill-marker drill-marker--${kind}`, iconSize: [size, size],
    html: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round" stroke-linecap="round" aria-hidden="true" focusable="false">' + PICTOGRAMS[kind] + '</svg>',
  });
}

function tooltipText(text: string) {
  const label = document.createElement('span');
  label.textContent = text;
  return label;
}

function hospitalPopup(facility: Hospital) {
  const content = document.createElement('div');
  content.className = 'drill-hospital-popup';
  content.appendChild(document.createElement('strong')).textContent = facility.name;
  content.appendChild(document.createElement('p')).textContent = facility.address || '등록 주소 없음';
  content.appendChild(document.createElement('small')).textContent = '국립중앙의료원 공개 위치 정보';
  content.appendChild(document.createElement('p')).textContent = '훈련용 수용 설정 · ' + trainingSetting(facility);
  return content;
}

function DrillMap({ drill, hospital, selected, onSelect, enabled, onAdvance }: {
  drill: Drill; hospital: Hospital | undefined; selected: Selection | null;
  onSelect: (id: string) => void; enabled: boolean; onAdvance: () => void;
}) {
  const container = useRef<HTMLDivElement>(null);
  const init = useRef(drill);
  const input = useRef({ enabled, onAdvance, onSelect });
  const controls = useRef<{ frame: (mode: MapFrame) => void; open: (selection: Selection) => void } | null>(null);
  const callerIcon = useRef<HTMLElement | undefined>(undefined);
  const [tileError, setTileError] = useState(false);
  const unitId = drill.dispatch.state === 'committed' ? drill.dispatch.unit_id : null;
  const assignment = useRef({ unitId, hospital });

  useEffect(() => {
    input.current = { enabled, onAdvance, onSelect };
    callerIcon.current?.setAttribute('aria-disabled', String(!enabled));
  }, [enabled, onAdvance, onSelect]);
  useEffect(() => {
    assignment.current = { unitId, hospital };
  }, [unitId, hospital]);
  useEffect(() => {
    if (selected) controls.current?.open(selected);
  }, [selected]);

  useEffect(() => {
    const element = container.current!;
    const scene = init.current;
    const callerPoint = toLatLng(scene.caller);
    const facilities = actualHospitals(scene);
    const vehicles = [callerPoint, ...scene.units.flatMap((unit) => unit.route.map(toLatLng))];
    const points = facilities.map(toLatLng);
    const nearest = points.map((point) => ({ point, distance: callerPoint.distanceTo(point) }))
      .sort((a, b) => a.distance - b.distance).slice(0, 3);
    const nearby = nearest.filter(({ distance }, index) => index === 0 || distance <= 1600).map(({ point }) => point);
    const reduced = matchMedia('(prefers-reduced-motion: reduce)');
    const map = L.map(element, {
      scrollWheelZoom: false, minZoom: 10, maxZoom: 19, zoomSnap: 0.5,
      zoomAnimation: !reduced.matches, fadeAnimation: !reduced.matches,
    });
    const frameMap = (mode: MapFrame) => {
      map.closePopup();
      const visibleFacilities = mode === 'all' ? points : mode === 'nearby' ? nearby : [];
      map.fitBounds(L.latLngBounds([...vehicles, ...visibleFacilities]),
        { padding: [32, 32], maxZoom: 17, animate: false });
    };
    frameMap('nearby');
    map.attributionControl.setPrefix(false);
    const tiles = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19, attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
    }).on('tileerror', () => setTileError(true)).addTo(map);
    const boundaryRequest = new AbortController();
    void fetch(new URL('./seoul-districts.geojson', import.meta.url).href, { signal: boundaryRequest.signal })
      .then(async (response) => {
        if (!response.ok) return;
        const geo = await response.json();
        if (!boundaryRequest.signal.aborted && geo?.type === 'FeatureCollection') {
          L.geoJSON(geo, {
            interactive: false, style: { color: '#718ca0', weight: 1, fillColor: '#d8e7e6', fillOpacity: 0.28 },
            attribution: '<a href="https://github.com/statgarten/maps/tree/d5f8ea3208f19a73a01f865847d20cc195ae91ba">SGIS 2020 · StatGarten</a>',
          }).addTo(map).bringToBack();
        }
      }).catch(() => {});
    void fetch(ZONE_SOURCE, { signal: boundaryRequest.signal }).then(async (response) => {
      if (!response.ok) return;
      const geo = await response.json();
      if (!boundaryRequest.signal.aborted && geo?.type === 'FeatureCollection') {
        const zonePane = map.createPane('training-zone');
        zonePane.style.zIndex = '350';
        L.geoJSON(geo, {
          pane: 'training-zone', interactive: false,
          style: { color: '#30786e', weight: 2, dashArray: '5 4', fillColor: '#83b6a8', fillOpacity: 0.14 },
        }).addTo(map);
      }
    }).catch(() => {});

    const caller = L.marker(callerPoint, {
      icon: markerIcon('caller', 44), title: CALLER_ACTION, zIndexOffset: 1400,
    }).on('click', (event: L.LeafletMouseEvent) => {
      if (event.originalEvent.detail <= 1 && input.current.enabled) input.current.onAdvance();
    }).addTo(map);
    const icon = caller.getElement();
    callerIcon.current = icon ?? undefined;
    icon?.setAttribute('aria-label', CALLER_ACTION);
    icon?.setAttribute('aria-disabled', String(!input.current.enabled));
    icon?.addEventListener('keydown', (event) => {
      if (event.key !== 'Enter' && event.key !== ' ') return;
      event.preventDefault();
      if (input.current.enabled && !event.repeat) input.current.onAdvance();
    });

    const pins = new Map<string, L.Marker>();
    for (const facility of facilities) {
      const select = () => input.current.onSelect(facility.id);
      const marker = L.marker(toLatLng(facility), { icon: markerIcon('hospital', 34), title: facility.name })
        .bindPopup(() => hospitalPopup(facility), { maxWidth: 260, minWidth: 155 }).on('click', select).addTo(map);
      const pin = marker.getElement();
      pin?.setAttribute('aria-label', facility.name);
      pin?.addEventListener('keydown', (event) => {
        if (event.key !== 'Enter' && event.key !== ' ') return;
        event.preventDefault();
        if (!event.repeat) select();
      });
      pins.set(facility.id, marker);
    }
    controls.current = {
      frame: frameMap,
      open: ({ id, revealMap }) => {
        const marker = pins.get(id);
        if (!marker) return;
        if (!map.getBounds().pad(-0.12).contains(marker.getLatLng())) {
          map.panTo(marker.getLatLng(), { animate: false });
        }
        marker.openPopup();
        if (revealMap) {
          marker.getElement()?.focus({ preventScroll: true });
          element.closest('.drill-map-card')?.scrollIntoView({ block: 'nearest' });
        }
      },
    };
    const movements = scene.units.flatMap((unit, index) => {
      const route = unit.route.map(toLatLng);
      if (!route.length) return [];
      const line = L.polyline(route, {
        color: unit.kind === 'ambulance' ? '#007f77' : '#ac411e', dashArray: '5 8',
        weight: 3, opacity: 0.8, interactive: false,
      }).addTo(map);
      const marker = L.marker(route[0], {
        icon: markerIcon(unit.kind), keyboard: false, title: unit.name, zIndexOffset: 400,
      }).bindTooltip(tooltipText(unit.name)).addTo(map);
      return [{
        id: unit.id, marker, line, samples: routeSamples(route), period: 18_000 + index * 3_000,
        committed: null as { from: L.LatLng; at: number } | null,
      }];
    });

    const started = performance.now();
    let frame: number;
    let link: L.Polyline | undefined;
    let linkedId: string | undefined;
    const render = (now: number) => {
      for (const unit of movements) {
        if (unit.id === assignment.current.unitId && !unit.committed) {
          unit.committed = { from: unit.marker.getLatLng(), at: now };
          unit.line.setLatLngs([unit.committed.from, callerPoint]);
          unit.line.setStyle({ dashArray: '', weight: 4, opacity: 1 });
          unit.marker.getElement()?.classList.add('dispatched');
        }
        const destination = unit.committed
          ? interpolate(unit.committed.from, callerPoint, reduced.matches ? 1 : Math.min(1, (now - unit.committed.at) / 4800))
          : patrolAt(unit.samples, reduced.matches ? 0 : now - started, unit.period);
        if (!destination.equals(unit.marker.getLatLng(), 1e-8)) unit.marker.setLatLng(destination);
      }
      const target = assignment.current.hospital;
      if (target?.id !== linkedId) {
        link?.remove();
        linkedId = target?.id;
        if (target) {
          link = L.polyline([callerPoint, toLatLng(target)], {
            className: 'drill-hospital-link', color: '#145d92', dashArray: '11 6',
            opacity: 1, weight: 4, interactive: false,
          }).addTo(map);
        }
      }
      frame = requestAnimationFrame(render);
    };
    frame = requestAnimationFrame(render);
    const observer = new ResizeObserver(() => {
      if (element.clientWidth && element.clientHeight) map.invalidateSize({ debounceMoveend: true });
    });
    observer.observe(element);
    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
      boundaryRequest.abort();
      tiles.off();
      map.remove();
      callerIcon.current = undefined;
      controls.current = null;
    };
  }, []);

  return (
    <section className="drill-map-card" aria-labelledby="drill-map-title">
      <header className="drill-map-heading">
        <h3 id="drill-map-title">현장 지도</h3>
        <div>
          <button type="button" className="drill-button subtle" onClick={() => controls.current?.frame('scene')}>현장 중심</button>
          {!!actualHospitals(drill).length && <button type="button" className="drill-button subtle"
            onClick={() => controls.current?.frame('all')}>병원 전체 보기</button>}
        </div>
      </header>
      <div className="drill-map-wrap">
        <div className="drill-map" ref={container} role="region" aria-label="훈련 차량 및 신고 위치 지도" />
        {tileError && <p className="drill-map-notice" role="status">도로 배경을 불러오지 못했습니다.</p>}
      </div>
      <ul className="drill-legend" aria-label="지도 범례">
        <li data-kind="caller">가상 신고 위치</li><li data-kind="ambulance">가상 구급차</li>
        <li data-kind="fire">가상 소방차</li><li data-kind="hospital">실제 의료기관 위치</li>
      </ul>
    </section>
  );
}

export function DrillPanel() {
  const [mode, setMode] = useState<Mode>('live');
  const [scenario, setScenario] = useState<Scenario>('ems');
  const [drill, setDrill] = useState<Drill | null>(null);
  const [selected, setSelected] = useState<Selection | null>(null);
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState<Problem | null>(null);
  const pending = useRef<AbortController | null>(null);
  useEffect(() => () => {
    pending.current?.abort();
    pending.current = null;
  }, []);

  async function run(current?: Drill) {
    if (pending.current) return;
    const controller = new AbortController();
    pending.current = controller;
    setBusy(true);
    setProblem(null);
    const signal = (timeout: number) => AbortSignal.any([controller.signal, AbortSignal.timeout(timeout)]);
    const path = current ? '/' + encodeURIComponent(current.id) : '';
    try {
      const result = await requestDrill(current ? path + '/turns' : path,
        current ? { expected_turn: current.turn } : { mode, scenario }, signal(30_000));
      if (pending.current === controller) {
        if (!current) setSelected(null);
        setDrill(result);
      }
    } catch (error) {
      if (pending.current !== controller) return;
      const code = error instanceof DrillError ? error.code : 'request_failed';
      if (current && code === 'live_model_unavailable') {
        setProblem({ kind: 'model', message: 'Zoro에 연결할 수 없습니다. 같은 발화에서 다시 시도할 수 있습니다.' });
      } else if (code === 'drill_not_found') {
        setProblem({ kind: 'expired', message: '훈련 세션이 만료됐습니다. 새 훈련을 시작하세요.' });
      } else if (current) {
        try {
          const updated = await requestDrill(path, undefined, signal(6000));
          if (pending.current !== controller) return;
          setDrill(updated);
          if (updated.turn === current.turn && updated.turn < updated.total_turns) {
            setProblem({ kind: 'turn', message: '발화를 진행하지 못했습니다. 다시 시도할 수 있습니다.' });
          }
        } catch {
          if (pending.current === controller) {
            setProblem({ kind: 'turn', message: '진행 상태를 확인하지 못했습니다. 지도에 마지막 확인 상태를 표시합니다.' });
          }
        }
      } else {
        setProblem({ kind: 'start', message: '훈련을 시작하지 못했습니다. 다시 시도해 주세요.' });
      }
    } finally {
      if (pending.current === controller) {
        pending.current = null;
        setBusy(false);
      }
    }
  }

  const finished = !!drill && drill.turn >= drill.total_turns;
  const canAdvance = !!drill && !finished && !busy && (!problem || problem.kind === 'start');
  const unit = drill?.units.find((item) => item.id === drill.dispatch.unit_id);
  const facilities = drill ? actualHospitals(drill) : [];
  const capability = drill?.subtype.state === 'committed' ? drill.subtype.code : null;
  const hospital = drill?.dispatch.category === 'ems' && drill.link.state === 'committed' && capability
    ? facilities.find((item) => item.id === drill.link.hospital_id && item.available
      && item.operational_source === 'simulation' && item.capabilities.includes(capability))
    : undefined;
  const location = drill?.caller.location_basis;
  const geographicSource = location?.level === 'label_dong'
    ? `AI Hub 라벨 주소 · ${location.dong ?? location.zone.dong}`
    : `${location?.level === 'collection_city' ? 'AI Hub 서울 수집분 · ' : ''}훈련 구역 ${location?.zone.dong ?? drill?.caller.dong ?? ''}`;
  const fetched = drill?.hospital_source?.fetched_at ? new Date(drill.hospital_source.fetched_at) : null;
  const recent = drill?.turns.at(-1);
  const modelStatus = drill?.model_status === 'live' ? 'Zoro 응답 적용'
    : drill?.model_status === 'simulation' ? '사전 작성된 판단' : 'Zoro 응답 대기';
  const selectFacility = (id: string) => setSelected({ id });
  const settings = (
    <form className="drill-controls" onSubmit={(event) => { event.preventDefault(); void run(); }}>
      <label>신고 유형<select value={scenario} onChange={(event) => setScenario(event.target.value as Scenario)}>
        <option value="ems">구급</option><option value="fire">화재</option>
      </select></label>
      <label>실행 모드<select value={mode} onChange={(event) => setMode(event.target.value as Mode)}>
        <option value="live">{MODE_LABELS.live}</option>
        <option value="simulation">{MODE_LABELS.simulation}</option>
      </select></label>
      <button type="submit" className="drill-button primary" disabled={busy}>
        {busy ? '훈련 요청 중' : drill ? '새 훈련 시작' : '훈련 시작'}
      </button>
    </form>
  );

  return (
    <section className="drill" aria-labelledby="drill-title">
      {drill ? <header className="drill-header">
        <div>
          <h2 id="drill-title">{CATEGORY_LABELS[drill.scenario]} 모의훈련</h2>
          <p>{geographicSource}</p>
          <p>신고 위치와 차량 경로는 훈련용으로 가상 생성합니다.</p>
        </div>
        <div className="drill-session-tools">
          <span>{MODE_LABELS[drill.mode]} · {modelStatus}</span>
          <details className="drill-settings"><summary>훈련 설정</summary>{settings}</details>
        </div>
      </header> : <header className="drill-hero">
        <div>
          <h2 id="drill-title">자동 모의훈련</h2>
          <p>신고 발화를 한 차례씩 공개하고 지도에서 차량 출동과 병원 연결 판단을 확인합니다.</p>
          {settings}
        </div>
        <img src={STATION_PHOTO} width={1100} height={733} alt="" />
      </header>}
      {problem && <div className="drill-problem" role="alert">
        <span>{problem.message}</span>
        {drill && (problem.kind === 'turn' || problem.kind === 'model') && <button type="button" className="drill-button subtle"
          onClick={() => void run(drill)}>{problem.kind === 'model' ? '모델 다시 시도' : '발화 다시 시도'}</button>}
      </div>}

      {drill ? (
        <>
          <div className="drill-grid">
            <div className="drill-overview">
              <section className="drill-latest drill-card" aria-labelledby="drill-transcript-title">
                <header>
                  <h3 id="drill-transcript-title">현재 발화</h3>
                  <span role="status">{drill.turn} / {drill.total_turns}턴</span>
                </header>
                {recent ? <div className="drill-latest-turn" aria-live="polite">
                  <strong>{recent.speaker}</strong><p>{recent.text}</p>
                </div> : <p className="drill-latest-turn">신고자 마커나 아래 버튼으로 첫 발화를 공개하세요.</p>}
                <button type="button" className="drill-button primary" disabled={!canAdvance} onClick={(event) => {
                  if (event.detail <= 1 && canAdvance) void run(drill);
                }}>{finished ? '훈련 완료' : busy ? '발화 처리 중' : '다음 발화 공개'}</button>
              </section>
              <section className="drill-card drill-decisions" aria-labelledby="drill-decisions-title">
                <h3 id="drill-decisions-title">모의 판단</h3>
                <dl aria-live="polite">
                  <div data-ready={drill.dispatch.state === 'committed'}>
                    <dt>{drill.dispatch.state === 'committed' ? '출동 확정' : '출동 보류'}</dt>
                    <dd>{drill.dispatch.category && unit
                      ? `${unit.name} · 근거 ${drill.dispatch.evidence_turn}턴` : '신고 유형 확인 중'}</dd>
                  </div>
                  <div data-ready={drill.subtype.state === 'committed'}>
                    <dt>세부 구급 판단</dt>
                    <dd>{drill.scenario === 'fire' ? '화재 단계 적용 없음' : capability ? '심정지 의심' : '추가 발화 대기'}</dd>
                  </div>
                  <div data-ready={!!hospital}>
                    <dt>모의 병원 연결</dt>
                    <dd>{drill.scenario === 'fire' ? '화재 단계 적용 없음' : hospital ? `${hospital.name} 연결`
                      : facilities.length ? '훈련 연결 조건 확인 중' : '확인된 의료기관 위치 없음'}</dd>
                  </div>
                </dl>
              </section>
            </div>
            <DrillMap key={drill.id} drill={drill} hospital={hospital} selected={selected} onSelect={selectFacility}
              enabled={canAdvance} onAdvance={() => { if (canAdvance) void run(drill); }} />
          </div>

          <section className="drill-card drill-facilities" aria-labelledby="drill-hospitals-title">
            <header>
              <h3 id="drill-hospitals-title">지도 속 의료기관</h3>
              <p><a href={HOSPITAL_SOURCE}>국립중앙의료원 공개 위치</a>
                {drill.hospital_source?.status === 'snapshot' && ' · 저장된 조회 결과'}
                {fetched && Number.isFinite(fetched.getTime()) && <> · <time dateTime={fetched.toISOString()}>{DATE_FORMAT.format(fetched)} KST</time></>}
              </p>
            </header>
            {!!facilities.length ? <ul className="drill-facility-list">
              {facilities.map((item) => <li key={item.id}>
                <button type="button" aria-pressed={selected?.id === item.id}
                  onClick={() => setSelected({ id: item.id, revealMap: true })}>
                  <strong>{item.name}</strong><span>{item.address || '등록 주소 없음'}</span>
                </button>
                <p><b>훈련용 수용 설정</b> {trainingSetting(item)}</p>
              </li>)}
            </ul> : <p role="status">표시할 공개 의료기관 위치를 확보하지 못했습니다.</p>}
          </section>

          <details className="drill-trace drill-card">
            <summary>훈련 세부 정보</summary>
            <div className="drill-details">
              <section aria-labelledby="drill-log-title">
                <h3 id="drill-log-title">발화 기록</h3>
                {drill.turns.length ? <ol aria-label="공개된 발화">
                  {drill.turns.map((turn, index) => (
                    <li key={index}><strong>{index + 1}턴 · {turn.speaker}</strong><p>{turn.text}</p></li>
                  ))}
                </ol> : <p>공개된 발화가 없습니다.</p>}
              </section>
              <section aria-labelledby="drill-units-title">
                <h3 id="drill-units-title">가상 차량 · 대원 및 장비</h3>
                <ul className="drill-resources">
                  {drill.units.map((item) => <li key={item.id}>
                    <strong>{item.name}</strong><span>{UNIT_LABELS[item.kind]} · {item.status === 'dispatched' ? '모의 출동' : '순찰 중'}</span>
                    <p><b>대원</b> {item.crew.join(', ')}</p><p><b>장비</b> {item.equipment.join(', ')}</p>
                  </li>)}
                </ul>
              </section>
              <section aria-labelledby="drill-origin-title">
                <h3 id="drill-origin-title">판단 및 지역 근거</h3>
                <p>{MODE_LABELS[drill.mode]} · {modelStatus}</p>
                {drill.model_id && <p className="drill-model-id">{drill.model_id}</p>}
                {location && <p>AI Hub {location.dataset_id} · 지역 표본 {location.count.toLocaleString('ko-KR')}건</p>}
                <p>법정동 경계 <a href="https://github.com/southkorea/seoul-maps/tree/dbb1dcde2e3c9937684c5a44ab9301888ad63ee6/juso/2015">JUSO 2015 · seoul-maps</a></p>
              </section>
            </div>
          </details>
        </>
      ) : null}
    </section>
  );
}
