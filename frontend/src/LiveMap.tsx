import { useEffect, useRef, useState } from 'react';
import * as L from 'leaflet';
import type { Session } from './api';
import 'leaflet/dist/leaflet.css';
import './drill.css';
import './live-map.css';

// Same pictograms as the dispatch drill; rescue reuses the truck outline.
const PICTOGRAMS = {
  caller: '<path d="M5 4h14a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2h-7l-5 4v-4H5a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2Z"/><path d="M8 11h8"/>',
  ambulance: '<path d="M2 6h12v11H2zM14 9h4l4 4v4h-8M6 9v5m-2.5-2.5h5"/><circle cx="6" cy="18" r="2" fill="currentColor"/><circle cx="18" cy="18" r="2" fill="currentColor"/>',
  fire: '<path d="M2 9h13v8H2zM15 10h4l3 4v3h-7M3 5h13m-11 0v4m4-4v4m4-4v4"/><circle cx="6" cy="18" r="2" fill="currentColor"/><circle cx="18" cy="18" r="2" fill="currentColor"/>',
  rescue: '<path d="M2 9h13v8H2zM15 10h4l3 4v3h-7M5 9V5l8 4"/><circle cx="6" cy="18" r="2" fill="currentColor"/><circle cx="18" cy="18" r="2" fill="currentColor"/>',
  hospital: '<path d="M4 21V5h16v16M2 21h20m-10-13v6m-3-3h6m-5 10v-4h4v4"/>',
} as const;
type Kind = keyof typeof PICTOGRAMS;
const UNIT_FOR: Record<string, Kind> = { 구급: 'ambulance', 화재: 'fire', 구조: 'rescue' };
const MOVE_MS = 4800;

function icon(kind: Kind, extra = '', size = 34) {
  return L.divIcon({
    className: `drill-marker drill-marker--${kind === 'rescue' ? 'fire' : kind} live-marker--${kind} ${extra}`,
    iconSize: [size, size],
    html: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round" '
      + 'stroke-linecap="round" aria-hidden="true" focusable="false">' + PICTOGRAMS[kind] + '</svg>',
  });
}

function label(text: string) {
  const span = document.createElement('span');
  span.textContent = text;
  return span;
}

/** Unit the map sends: the committed vehicle recommendation for 구급, else the nearest unit of that type. */
function dispatchedUnit(session: Session) {
  const dispatch = session.decisions.dispatch;
  if (dispatch.status !== 'COMMIT' || !dispatch.choice) return null;
  const units = session.units ?? [];
  if (dispatch.choice === '구급') {
    const vehicle = session.decisions.vehicle;
    return vehicle.status === 'COMMIT' ? units.find((unit) => unit.id === vehicle.choice) ?? null : null;
  }
  const kind = UNIT_FOR[dispatch.choice];
  return units.filter((unit) => unit.kind === kind).sort((a, b) => a.distance_km - b.distance_km)[0] ?? null;
}

export function LiveMap({ session }: { session: Session }) {
  const element = useRef<HTMLDivElement>(null);
  const layers = useRef<{
    map: L.Map; units: Map<string, L.Marker>; hospitals: Map<string, L.Marker>;
    center: L.LatLng; route?: L.Polyline; link?: L.Polyline; moving?: { id: string; from: L.LatLng; at: number };
  } | null>(null);
  const [tileError, setTileError] = useState(false);
  const area = session.area;

  // One map per call area; markers are rebuilt only when the area changes.
  useEffect(() => {
    if (!element.current || !area) return;
    const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
    const center = L.latLng(area.center.lat, area.center.lon);
    const map = L.map(element.current, { scrollWheelZoom: false, minZoom: 10, maxZoom: 18, zoomSnap: 0.5,
      zoomAnimation: !reduced, fadeAnimation: !reduced });
    map.attributionControl.setPrefix(false);
    L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19, attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
    }).on('tileerror', () => setTileError(true)).addTo(map);
    const dong = L.geoJSON({ type: 'Feature', properties: {}, geometry: area.geometry } as GeoJSON.Feature, {
      interactive: false,
      style: { color: '#613ca4', weight: 2.5, dashArray: '6 4', fillColor: '#8b6cc6', fillOpacity: 0.16 },
      attribution: 'JUSO 2015 · <a href="https://github.com/southkorea/seoul-maps">southkorea/seoul-maps</a>',
    }).addTo(map);
    void fetch(new URL('./seoul-districts.geojson', import.meta.url).href).then(async (response) => {
      if (!response.ok) return;
      const geo = await response.json();
      if (geo?.type === 'FeatureCollection' && layers.current?.map === map) {
        L.geoJSON(geo, { interactive: false, style: { color: '#718ca0', weight: 1, fillOpacity: 0 } }).addTo(map).bringToBack();
      }
    }).catch(() => {});
    L.marker(center, { icon: icon('caller', '', 42), keyboard: false, zIndexOffset: 1200 })
      .bindTooltip(label(`신고 위치 · ${area.district} ${area.dong} 일대`), { direction: 'top', offset: [0, -18] })
      .addTo(map);
    const units = new Map<string, L.Marker>();
    for (const unit of session.units ?? []) {
      units.set(unit.id, L.marker([unit.lat, unit.lon], { icon: icon(unit.kind as Kind), keyboard: false, zIndexOffset: 400 })
        .bindTooltip(label(`${unit.name} · ${unit.distance_km}km · 약 ${unit.eta_min}분`)).addTo(map));
    }
    const hospitals = new Map<string, L.Marker>();
    for (const hospital of session.hospitals) {
      if (hospital.lat === undefined || hospital.lon === undefined) continue;
      hospitals.set(hospital.id, L.marker([hospital.lat, hospital.lon], { icon: icon('hospital'), keyboard: false })
        .bindTooltip(label(hospital.name)).addTo(map));
    }
    const points = [center, ...[...units.values(), ...hospitals.values()].map((marker) => marker.getLatLng())];
    map.fitBounds(L.latLngBounds(points).extend(dong.getBounds()), { padding: [28, 28], maxZoom: 15, animate: false });
    layers.current = { map, units, hospitals, center };
    let frame = 0;
    const render = (now: number) => {
      const moving = layers.current?.moving;
      const marker = moving && layers.current?.units.get(moving.id);
      if (moving && marker) {
        const t = reduced ? 1 : Math.min(1, (now - moving.at) / MOVE_MS);
        marker.setLatLng(L.latLng(moving.from.lat + (center.lat - moving.from.lat) * t,
          moving.from.lng + (center.lng - moving.from.lng) * t));
      }
      frame = requestAnimationFrame(render);
    };
    frame = requestAnimationFrame(render);
    const observer = new ResizeObserver(() => map.invalidateSize({ debounceMoveend: true }));
    observer.observe(element.current);
    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
      layers.current = null;
      map.remove();
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [session.id, area?.code]);

  // Dispatch and hospital decisions drive the route and the hospital link.
  const unit = dispatchedUnit(session);
  const hospitalId = session.decisions.hospital.status === 'COMMIT' ? session.decisions.hospital.choice : null;
  useEffect(() => {
    const current = layers.current;
    if (!current) return;
    if (unit && current.moving?.id !== unit.id) {
      current.route?.remove();
      for (const [id, marker] of current.units) marker.getElement()?.classList.toggle('dispatched', id === unit.id);
      const from = L.latLng(unit.lat, unit.lon);
      current.units.get(unit.id)?.setLatLng(from);
      current.route = L.polyline([from, current.center], {
        color: unit.kind === 'ambulance' ? '#007f77' : '#ac411e', weight: 4, opacity: 0.95, interactive: false,
      }).addTo(current.map);
      current.moving = { id: unit.id, from, at: performance.now() };
    } else if (!unit && current.moving) {
      current.route?.remove();
      const previous = (session.units ?? []).find((item) => item.id === current.moving?.id);
      if (previous) current.units.get(previous.id)?.setLatLng([previous.lat, previous.lon]);
      for (const marker of current.units.values()) marker.getElement()?.classList.remove('dispatched');
      current.moving = undefined;
    }
    current.link?.remove();
    current.link = undefined;
    const target = hospitalId ? current.hospitals.get(hospitalId) : undefined;
    if (target) {
      current.link = L.polyline([current.center, target.getLatLng()], {
        color: '#145d92', dashArray: '11 6', weight: 4, interactive: false,
      }).addTo(current.map);
    }
    for (const [id, marker] of current.hospitals) marker.getElement()?.classList.toggle('dispatched', id === hospitalId);
  }, [unit?.id, hospitalId, session.units]);

  if (!area) {
    return (
      <section className="panel live-map-panel" aria-labelledby="live-map-heading">
        <h2 id="live-map-heading" className="section-title">신고 위치</h2>
        <p className="meta">이 신고에는 지도에 표시할 위치 정보가 없습니다.</p>
      </section>
    );
  }
  const dispatch = session.decisions.dispatch;
  return (
    <section className="panel live-map-panel" aria-labelledby="live-map-heading">
      <header className="panel-header">
        <h2 id="live-map-heading" className="section-title">신고 위치 · {area.district} {area.dong} 일대</h2>
        <span className="meta">
          {dispatch.status === 'COMMIT' ? (unit ? `${unit.name} 출동 · 약 ${unit.eta_min}분` : `${dispatch.choice} 확정 · 차량 판단 대기`)
            : '출동유형 판단 대기'}
        </span>
      </header>
      <div className="drill-map-wrap live-map-wrap">
        <div ref={element} className="drill-map live-map" role="region" aria-label={`${area.district} ${area.dong} 일대 지도`} />
      </div>
      <p className="meta live-map-note">
        위치는 AI Hub 기록의 법정동 수준(정확한 주소 아님)이며, 차량과 병원은 시연용 가상 자원입니다.
        {tileError ? ' 지도 배경을 불러오지 못해 경계와 위치만 표시합니다.' : ''}
      </p>
    </section>
  );
}
