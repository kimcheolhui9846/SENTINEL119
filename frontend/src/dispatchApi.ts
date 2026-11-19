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
  patrols: (center: Point) => call<{ patrols: Patrol[]; note: string }>(`/patrols?lat=${center.lat}&lon=${center.lon}`),
};
