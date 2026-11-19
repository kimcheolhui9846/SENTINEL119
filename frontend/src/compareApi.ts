// Model comparison tab: one transcript prefix to the plain LLM, SENTINEL_ZORO and base Zoro.
import type { CallArea } from './dispatchApi';
import type { Harness } from './demoHarness';
export interface CompareScenario { id: string; title: string; summary: string; private?: boolean; area?: string | null }
export interface ScenarioDetail extends Omit<CompareScenario, 'area'> {
  gold?: Record<string, string> | null;
  area?: CallArea | null;
  turns: { speaker: 0 | 1; text: string }[];
}
export interface LlmResponse {
  model: { model_id?: string; variant?: string };
  text: string;
  answer: { dispatch: string | null; subtype: string | null; subtype_question: string | null;
            urgency: string | null; invalid: string[] };
  latency_ms: number;
  new_tokens: number;
}
export type StreamQuestion = 'dispatch' | 'urgency' | 'ems_subtype' | 'rescue_subtype' | 'ems_symptom';
export interface StreamVariantResult {
  model: { model_id?: string; variant?: string; checkpoint?: { path?: string } | null };
  probabilities: Partial<Record<StreamQuestion, Record<string, number>>>;
  latency_ms: number;
}
export interface StreamResponse {
  profile: string;
  labels: Record<string, Record<string, string>>;
  results: Record<'sft' | 'base', StreamVariantResult>;
  harness: Harness;
}

const messages: Record<string, string> = {
  compare_unconfigured: 'SENTINEL_ZORO_URL이 설정되지 않아 모델 비교를 실행할 수 없습니다.',
  compare_unavailable: '모델 서버에 연결하지 못했습니다.',
  compare_harness_invalid: '모델 또는 판단 정책을 확인할 수 없습니다.',
  scenario_not_found: '비교할 통화를 찾지 못했습니다.',
};
export class ApiError extends Error {}

async function call<T>(path: string, body?: object): Promise<T> {
  const response = await fetch('/api/compare' + path, {
    method: body ? 'POST' : 'GET',
    headers: body ? { 'Content-Type': 'application/json' } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new ApiError(messages[error.detail] ?? '모델 비교 요청에 실패했습니다.');
  }
  return await response.json() as T;
}

type Turns = { speaker: 0 | 1; text: string }[];
export const api = {
  scenarios: async () => (await call<{ scenarios: CompareScenario[] }>('')).scenarios,
  scenario: (id: string) => call<ScenarioDetail>('/' + encodeURIComponent(id)),
  streamDecide: (turns: Turns) => call<StreamResponse>('/decide', { turns }),
  streamLlm: (turns: Turns) => call<LlmResponse>('/llm', { turns }),
};
