// Dispatch map API: the fleet at real stations and road-routed assignments for a committed call type.
export interface Point { lat: number; lon: number }
export interface FleetUnit extends Point {
  id: string; name: string; kind: 'ambulance' | 'pump' | 'rescue'; station: string; district: string;
}
export interface PlanStep { kind: FleetUnit['kind']; role: string; reason: string; source: string }
export interface Assignment extends PlanStep {
  unit: FleetUnit | null;
  route?: { path: Point[]; distance_m: number; duration_s: number; source: 'tmap' | 'osrm' | 'straight_line' };
}
export interface Patrol { id: string; path: Point[]; duration_s: number; source: string }
export interface TransferHospital extends Point {
  hpid: string; name: string; address: string; level: string; district: string; er_beds: number | null;
  beds_at: string | null; accepts: string[]; beds: Record<string, number>; equipment: string[]; messages: string[];
  data: 'live' | 'directory'; context: string;
  route: { path: Point[]; distance_m: number; duration_s: number; source: 'tmap' | 'osrm' | 'straight_line' };
}
export interface TransferResponse {
  status: 'ok' | 'unconfigured' | 'unavailable'; data?: 'live' | 'directory';
  rule: { category: string; min_level: string; text: string; source: string; prefer?: string[]; symptom?: string | null };
  hospitals: TransferHospital[];
  zoro: { probabilities: Record<string, number>; choice: string; latency_ms?: number } | null;
  source: string;
  reason?: TransferReason | null;
}
export interface TransferReason {
  chosen: string; p: number; runner_up: string | null; runner_up_p: number | null; facts: string[]; method: string;
  effects: { group: string; without: number; delta: number; choice_without: string }[];
  reasoned: { text: string; finished: boolean; tokens: number; ms: number; direct: Record<string, number>;
    reasoned: Record<string, number>; choice: string; choice_name: string } | null;
}
export interface DispatchResponse { incident: Point; plan: PlanStep[]; assignments: Assignment[] }
export interface CallArea {
  district: string; dong: string; code: string; center: Point; geometry: GeoJSON.Geometry; precision: string;
}

async function call<T>(path: string, body?: object): Promise<T> {
  const response = await fetch('/api/map' + path, {
    method: body ? 'POST' : 'GET',
    headers: body ? { 'Content-Type': 'application/json' } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!response.ok) throw new Error('dispatch_failed');
  return await response.json() as T;
}

export const dispatchApi = {
  fleet: () => call<{ units: FleetUnit[]; attribution: string }>('/fleet'),
  dispatch: (body: { lat: number; lon: number; dispatch: string; subtype?: string | null; symptom?: string | null;
    urgency?: string | null; busy: string[]; skip: number; positions?: Record<string, Point> }) => call<DispatchResponse>('/dispatch', body),
  transfer: (body: { lat: number; lon: number; turns: { speaker: 0 | 1; text: string }[]; subtype?: string | null;
    symptom?: string | null; urgency?: string | null; explain?: boolean }) => call<TransferResponse>('/transfer', body),
  locate: (p: Point) => call<CallArea>(`/locate?lat=${p.lat}&lon=${p.lon}`),
  patrols: () => call<{ patrols: Patrol[]; note: string }>('/patrols'),
};
