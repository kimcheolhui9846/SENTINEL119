import { useEffect, useMemo, useRef, useState } from 'react';
import * as L from 'leaflet';
import type { FeatureCollection, Geometry } from 'geojson';
import 'leaflet/dist/leaflet.css';
import './hospital-map.css';

const DISTRICTS = [
  '강남구', '강동구', '강북구', '강서구', '관악구', '광진구', '구로구', '금천구', '노원구',
  '도봉구', '동대문구', '동작구', '마포구', '서대문구', '서초구', '성동구', '성북구', '송파구',
  '양천구', '영등포구', '용산구', '은평구', '종로구', '중구', '중랑구',
];
const SEOUL: L.LatLngExpression = [37.5665, 126.978];
const FILTERS = [
  { value: 'hospitals', label: '병원·응급' },
  { value: 'all', label: '전체' },
  { value: 'emergency', label: '응급의료' },
] as const;
const DATE_FORMAT = new Intl.DateTimeFormat('ko-KR', {
  dateStyle: 'medium', timeStyle: 'short', timeZone: 'Asia/Seoul',
});
type Filter = (typeof FILTERS)[number]['value'];
type Hospital = {
  id: string; name: string; address: string; lat: number; lon: number;
  category: string; emergency_code: string;
};
type HospitalData = {
  district: string; source: 'nmc'; status: 'ok' | 'stale' | 'unconfigured' | 'unavailable';
  fetched_at: string | null; items: Hospital[];
};
type Boundaries = FeatureCollection<Geometry, { name: string }>;

let boundariesRequest: Promise<Boundaries> | undefined;

function fetchBoundaries() {
  boundariesRequest ??= fetch(new URL('./seoul-districts.geojson', import.meta.url).href).then(async (response) => {
    if (!response.ok) throw new Error('Boundary request failed');
    const data: Boundaries = await response.json();
    if (data.type !== 'FeatureCollection' || !Array.isArray(data.features)) {
      throw new Error('Invalid boundary collection');
    }
    return data;
  }).catch((error) => {
    boundariesRequest = undefined;
    throw error;
  });
  return boundariesRequest;
}

function isMappedHospital(item: Hospital) {
  return item && typeof item.id === 'string' && typeof item.name === 'string'
    && Number.isFinite(item.lat) && Math.abs(item.lat) <= 90
    && Number.isFinite(item.lon) && Math.abs(item.lon) <= 180
    && (item.lat !== 0 || item.lon !== 0)
    && typeof item.category === 'string' && typeof item.emergency_code === 'string'
    && typeof item.address === 'string';
}

function isEmergency(hospital: Hospital) {
  // NMC uses G099 for facilities outside the emergency classification.
  return /^G00[1-9]$/i.test(hospital.emergency_code.trim());
}

function popupContent(hospital: Hospital) {
  const content = document.createElement('div');
  const title = content.appendChild(document.createElement('strong'));
  title.textContent = hospital.name;
  const detail = content.appendChild(document.createElement('p'));
  detail.textContent = [hospital.category, isEmergency(hospital) ? '응급의료기관' : '', hospital.address]
    .filter(Boolean).join(' · ');
  return content;
}

export function HospitalMap() {
  const [district, setDistrict] = useState('성동구');
  const [filter, setFilter] = useState<Filter>('hospitals');
  const [search, setSearch] = useState('');
  const [selected, setSelected] = useState<string | null>(null);
  const [request, setRequest] = useState(0);
  const [data, setData] = useState<HospitalData | null>(null);
  const [boundaries, setBoundaries] = useState<Boundaries | null>(null);
  const [tileError, setTileError] = useState(false);
  const container = useRef<HTMLDivElement>(null);
  const map = useRef<L.Map | null>(null);
  const markerLayer = useRef<L.LayerGroup | null>(null);
  const markers = useRef(new Map<string, L.Marker>());
  const userMoved = useRef(false);
  const list = useRef<HTMLUListElement>(null);

  useEffect(() => {
    const element = container.current;
    if (!element) return;
    const instance = L.map(element, { scrollWheelZoom: false, minZoom: 9, maxZoom: 19 })
      .setView(SEOUL, 11);
    map.current = instance;
    instance.attributionControl.setPrefix(false);
    markerLayer.current = L.layerGroup().addTo(instance);
    const failedTiles = new Set<HTMLElement>();
    const tiles = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19,
      attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
    });
    tiles.on('tileerror', ({ tile }: L.TileErrorEvent) => {
      failedTiles.add(tile);
      setTileError(true);
    });
    const removeFailure = ({ tile }: L.TileEvent) => {
      if (failedTiles.delete(tile)) setTileError(failedTiles.size > 0);
    };
    tiles.on('tileunload', removeFailure);
    tiles.on('tileload', removeFailure);
    tiles.addTo(instance);
    const noteInteraction = () => { userMoved.current = true; };
    element.addEventListener('pointerdown', noteInteraction);
    element.addEventListener('keydown', noteInteraction);
    const observer = new ResizeObserver(() => {
      if (element.clientWidth && element.clientHeight) instance.invalidateSize({ debounceMoveend: true });
    });
    observer.observe(element);
    return () => {
      element.removeEventListener('pointerdown', noteInteraction);
      element.removeEventListener('keydown', noteInteraction);
      observer.disconnect();
      tiles.off();
      instance.remove();
      map.current = null;
      markerLayer.current = null;
      markers.current.clear();
    };
  }, []);

  useEffect(() => {
    let active = true;
    void fetchBoundaries().then((value) => {
      if (active) setBoundaries(value);
    }).catch(() => {});
    return () => { active = false; };
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    setData(null);
    const query = new URLSearchParams({ district });
    void fetch(`/api/hospital-locations?${query}`, { signal: controller.signal }).then(async (response) => {
      if (!response.ok) throw new Error('Hospital request failed');
      const result: HospitalData = await response.json();
      if (result.district !== district || result.source !== 'nmc'
        || !['ok', 'stale', 'unconfigured', 'unavailable'].includes(result.status) || !Array.isArray(result.items)) {
        throw new Error('Invalid hospital response');
      }
      if (controller.signal.aborted) return;
      const valid = result.status === 'ok' || result.status === 'stale' ? result.items.filter(isMappedHospital) : [];
      const unique = new Map(valid.map((item) => [item.id, item]));
      setData({ ...result, items: [...unique.values()] });
    }).catch(() => {
      if (!controller.signal.aborted) {
        setData({ district, source: 'nmc', status: 'unavailable', fetched_at: null, items: [] });
      }
    });
    return () => controller.abort();
  }, [district, request]);

  const current = data?.district === district ? data : null;
  const visible = useMemo(() => {
    const query = search.trim().toLocaleLowerCase('ko');
    return (current?.items ?? []).filter((item) =>
      item.name.toLocaleLowerCase('ko').includes(query)
      && (filter === 'all' || isEmergency(item) || (filter === 'hospitals' && item.category.includes('병원'))),
    ).sort((a, b) => Number(isEmergency(b)) - Number(isEmergency(a)) || a.name.localeCompare(b.name, 'ko'));
  }, [current, filter, search]);

  useEffect(() => {
    const instance = map.current;
    if (!instance || !boundaries) return;
    let districtBounds: L.LatLngBounds | undefined;
    const outline = L.geoJSON(boundaries, {
      interactive: false,
      filter: (feature) => ['Polygon', 'MultiPolygon'].includes(feature.geometry?.type),
      style: (feature) => feature?.properties?.name === district
        ? { color: '#38d9e2', weight: 2, opacity: 0.85, fillColor: '#219bb8', fillOpacity: 0.18 }
        : { color: '#698aa1', weight: 1, opacity: 0.5, fillColor: '#25485e', fillOpacity: 0.27 },
      onEachFeature: (feature, layer) => {
        if (feature.properties?.name === district && layer instanceof L.Polygon) districtBounds = layer.getBounds();
      },
    }).addTo(instance);
    if (!userMoved.current && districtBounds?.isValid()) {
      instance.fitBounds(districtBounds, { padding: [24, 24], maxZoom: 14, animate: false });
    }
    return () => { outline.remove(); };
  }, [boundaries, district]);

  useEffect(() => {
    const instance = map.current;
    if (!instance || userMoved.current) return;
    if (boundaries?.features.some((feature) => feature.properties?.name === district)) return;
    const points = current?.items.map(({ lat, lon }): L.LatLngTuple => [lat, lon]);
    if (points?.length) {
      instance.fitBounds(points, { padding: [24, 24], maxZoom: 14, animate: false });
    } else {
      instance.setView(SEOUL, 11, { animate: false });
    }
  }, [boundaries, current, district]);

  useEffect(() => {
    const layer = markerLayer.current;
    if (!layer) return;
    const registry = markers.current;
    for (const hospital of visible) {
      const selectMarker = () => {
        userMoved.current = true;
        setSelected(hospital.id);
        const row = list.current?.querySelector('[data-marker-id="' + CSS.escape(hospital.id) + '"]');
        if (row && list.current) {
          const listBounds = list.current.getBoundingClientRect();
          const rowBounds = row.getBoundingClientRect();
          if (rowBounds.top < listBounds.top || rowBounds.bottom > listBounds.bottom) {
            list.current.scrollTop += rowBounds.top - listBounds.top;
          }
        }
      };
      const marker = L.marker([hospital.lat, hospital.lon], {
        title: hospital.name,
        icon: L.divIcon({
          className: `hospital-map-pin${isEmergency(hospital) ? ' emergency' : ''}`,
          iconSize: [20, 20], iconAnchor: [10, 10], popupAnchor: [0, -10],
        }),
      }).bindPopup(() => popupContent(hospital))
        .on('click', selectMarker).addTo(layer);
      const icon = marker.getElement();
      icon?.setAttribute('aria-label', hospital.name);
      icon?.addEventListener('keydown', (event) => {
        if (event.key !== 'Enter' && event.key !== ' ') return;
        event.preventDefault();
        selectMarker();
        marker.openPopup();
      });
      registry.set(hospital.id, marker);
    }
    return () => {
      layer.clearLayers();
      registry.clear();
    };
  }, [visible]);

  useEffect(() => {
    for (const [id, marker] of markers.current) {
      const active = id === selected;
      const element = marker.getElement();
      element?.classList.toggle('selected', active);
      element?.setAttribute('aria-pressed', String(active));
      marker.setZIndexOffset(active ? 1000 : 0);
    }
  }, [selected, visible]);

  function choose(hospital: Hospital) {
    const instance = map.current;
    const marker = markers.current.get(hospital.id);
    if (!instance || !marker) return;
    userMoved.current = true;
    setSelected(hospital.id);
    if (!instance.getBounds().pad(-0.15).contains(marker.getLatLng())) {
      instance.panTo(marker.getLatLng(), { animate: false });
    }
    marker.openPopup();
  }

  let feedback = '';
  if (!current) feedback = '의료기관 위치를 불러오는 중입니다.';
  else if (current.status === 'unconfigured') feedback = '국립중앙의료원 API 인증키가 설정되지 않았습니다.';
  else if (current.status === 'unavailable') feedback = '의료기관 위치를 불러오지 못했습니다.';
  else if (!current.items.length) feedback = '이 자치구에 표시할 의료기관 위치가 없습니다.';
  else if (!visible.length) feedback = '현재 조건에 맞는 의료기관이 없습니다.';
  const fetched = current?.fetched_at ? new Date(current.fetched_at) : null;

  return (
    <section className="hospital-map" aria-labelledby="hospital-map-heading">
      <header className="hospital-map-header">
        <div>
          <h2 id="hospital-map-heading">서울 의료기관 지도</h2>
          <p><a href="https://www.data.go.kr/data/15000736/openapi.do">국립중앙의료원 공개 위치 정보</a> · 환자 수용 여부는 별도 확인이 필요합니다.</p>
        </div>
        <label className="hospital-map-field">
          서울 자치구
          <select value={district} onChange={(event) => {
            userMoved.current = false;
            setSelected(null);
            setSearch('');
            setDistrict(event.target.value);
          }}>
            {DISTRICTS.map((name) => <option key={name}>{name}</option>)}
          </select>
        </label>
      </header>

      <div className="hospital-map-body">
        <div className="hospital-map-canvas">
          <div className="hospital-map-leaflet" ref={container} role="region" aria-label={`${district} 의료기관 위치`} />
          {tileError && <p className="hospital-map-tile-error" role="status">배경 지도의 일부 또는 전체를 불러오지 못했습니다.</p>}
          <div className="hospital-map-legend" aria-hidden="true">
            <span><i /> 의료기관</span><span><i /> 응급의료기관</span>
          </div>
        </div>
        <div className="hospital-map-browser">
          <div className="hospital-map-filters" role="group" aria-label="의료기관 분류">
            {FILTERS.map(({ value, label }) => (
              <button type="button" key={value} aria-pressed={filter === value} onClick={() => {
                setFilter(value);
                setSelected(null);
              }}>{label}</button>
            ))}
          </div>
          <label className="hospital-map-field">
            기관명 검색
            <input type="search" value={search} placeholder="의료기관 이름" onChange={(event) => {
              setSearch(event.target.value);
              setSelected(null);
            }} />
          </label>
          <div className="hospital-map-summary" role="status">
            <strong>지도 {visible.length.toLocaleString('ko-KR')}곳</strong>
            {current?.status === 'stale' && <span className="hospital-map-stale">저장된 위치 정보</span>}
          </div>
          {feedback && <p className="hospital-map-feedback" role="status">{feedback}</p>}
          {current?.status === 'unavailable' && <button className="hospital-map-retry" type="button"
            onClick={() => setRequest((value) => value + 1)}>위치 조회 다시 시도</button>}
          <ul className="hospital-map-list" ref={list} aria-label="지도에 표시된 의료기관">
            {visible.map((hospital) => (
              <li key={hospital.id}>
                <button type="button" data-marker-id={hospital.id} aria-pressed={selected === hospital.id}
                  onClick={() => choose(hospital)}>
                  <strong>{hospital.name}</strong>
                  <span>{[hospital.category, isEmergency(hospital) ? '응급의료' : ''].filter(Boolean).join(' · ')}</span>
                </button>
              </li>
            ))}
          </ul>
        </div>
      </div>
      <footer className="hospital-map-footer">
        {fetched && Number.isFinite(fetched.getTime()) && (
          <p>위치 정보 조회 <time dateTime={fetched.toISOString()}>{DATE_FORMAT.format(fetched)} KST</time></p>
        )}
        <p>자치구 경계 <a href="https://github.com/statgarten/maps/tree/d5f8ea3208f19a73a01f865847d20cc195ae91ba">통계청 SGIS 2020 · StatGarten</a></p>
      </footer>
    </section>
  );
}
