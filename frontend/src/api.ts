export type Task = 'dispatch' | 'urgency' | 'vehicle' | 'hospital';
export type Mode = 'simulation' | 'live';
export type DispatchEvent = { action: 'dispatch'; expected_vehicle_id: string; expected_vehicle_revision: number };
export type SessionEvent =
  | { action: 'next' | 'retry' }
  | { action: 'text'; text: string }
  | DispatchEvent;

export interface Scenario { id: string; title: string; summary: string; turn_count: number; live_only?: boolean; private?: boolean; area?: string | null }
export interface HierarchyItem {
  question: string; parent: string; probabilities: Record<string, number>; threshold: number;
  status: 'WAIT' | 'COMMIT'; choice: string | null; turn: number | null;
}
export interface Candidate {
  id: string; name: string; availability: 'available' | 'unavailable' | 'unknown';
  eta_min: number | null; detail: string; source: string; source_at: string | null;
  last_observation?: { availability: Candidate['availability']; source_at: string } | null;
  lat?: number; lon?: number; kind?: string;
}
export interface CallArea {
  city: string; district: string; dong: string; code: string; precision: string; source: string;
  center: { lat: number; lon: number }; geometry: GeoJSON.Geometry;
}
export interface Unit {
  id: string; name: string; kind: 'ambulance' | 'fire' | 'rescue'; eta_min: number; distance_km: number;
  detail: string; lat: number; lon: number;
}
export interface Decision {
  status: 'WAIT' | 'COMMIT' | 'OUT_OF_SCOPE'; choice: string | null; reason: string;
  evidence_turn_ids: number[]; probabilities: Record<string, number> | null;
  execution: 'idle' | 'running' | 'ok' | 'error'; model_id: string | null;
  missing_information: string[]; guideline_refs: string[]; candidate_snapshot_id: number | null;
  produced_at: string | null; latency_ms: number | null;
  recommendation_revision?: number;
}
export interface Session {
  id: string; scenario_id: string; scenario_title: string; mode: Mode;
  version: number; view_revision: number; turn_id: number; total_steps: number;
  step: number; processing: boolean; manual: boolean;
  turns: { id: number; speaker: string; text: string }[];
  decisions: Record<Task, Decision>; vehicles: Candidate[]; hospitals: Candidate[];
  location: string | null; external_status: 'ok' | 'unknown' | 'timeout' | 'cached';
  history: { id: number; turn_id: number; task: string; status: string; choice: string | null; reason: string }[];
  applied_vehicle: string | null; error: string | null;
  hierarchy?: Partial<Record<'subtype' | 'symptom', HierarchyItem>>;
  area?: CallArea | null; units?: Unit[];
}

const messages: Record<string, string> = {
  session_not_found: '세션이 만료되었습니다. 초기화해 새 신고를 여세요.',
  stale_version: '신고 내용이 바뀌어 최신 상태를 불러옵니다.',
  stale_vehicle_recommendation: '차량 추천이 바뀌어 최신 상태를 불러옵니다.',
  vehicle_not_ready: '확정된 차량을 기다리고 있습니다.',
  dispatch_already_applied: '이 신고에는 이미 모의 출동이 적용됐습니다.',
  retry_unavailable: '현재 재시도할 작업이 없습니다.',
  replay_finished: '신고 재생이 끝났습니다.',
  turn_limit: '이 세션의 입력 한도에 도달했습니다.',
  session_capacity: '현재 세션이 많습니다. 잠시 뒤 다시 시도해 주세요.',
  live_only_scenario: '실제 통화는 실시간 Zoro 모드에서만 재생합니다.',
};

export class ApiError extends Error {
  constructor(public status: number, public code: string | undefined) {
    super((code && messages[code]) || '요청을 처리하지 못했습니다. 다시 시도해 주세요.');
  }
}

async function request<T>(path: string, body?: object): Promise<T> {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), 10000);
  try {
    const response = await fetch('/api' + path, {
      method: body ? 'POST' : 'GET', signal: controller.signal,
      headers: body ? { 'Content-Type': 'application/json' } : {},
      body: body ? JSON.stringify(body) : undefined,
    });
    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new ApiError(response.status, typeof error.detail === 'string' ? error.detail : undefined);
    }
    return await response.json() as T;
  } finally { window.clearTimeout(timer); }
}

export async function listScenarios(): Promise<Scenario[]> {
  return (await request<{ scenarios: Scenario[] }>('/scenarios')).scenarios;
}
export function createSession(scenario_id: string, mode: Mode) {
  return request<Session>('/sessions', { scenario_id, mode });
}
export function getSession(id: string) {
  return request<Session>('/sessions/' + encodeURIComponent(id));
}
export function postEvent(session: Session, event: SessionEvent) {
  return request<Session>('/sessions/' + encodeURIComponent(session.id) + '/events',
    { ...event, expected_version: session.version });
}
