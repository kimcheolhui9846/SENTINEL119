import type { StreamQuestion, StreamVariantResult } from './compareApi';

export type ZoroId = 'sft' | 'base';
type Task = 'dispatch' | 'subtype' | 'urgency' | 'symptom';
type Gate = 'allow' | 'wait' | 'review';
export type Probabilities = StreamVariantResult['probabilities'];
type Choice = { key: string; label: string; p: number };
export type Commit = Choice & { turn: number };
type Thresholds = Record<Task, number>;

export interface Harness {
  model_id: string;
  policy_digest: string;
  thresholds: Thresholds;
  dispatch_overrides: Partial<Record<string, number>>;
  gate: Gate;
  choices: Record<ZoroId, Partial<Record<StreamQuestion, Choice>>>;
}

export interface ZoroState {
  probs: Probabilities | null;
  commits: Partial<Record<Task, Commit>>;
  latencies: number[];
  turn: number;
  gate: Gate;
  reviewTurn: number | null;
  blocked: boolean;
  profile: Pick<Harness, 'model_id' | 'policy_digest' | 'thresholds' | 'dispatch_overrides'> | null;
}

export const SUBTYPE_OF: Partial<Record<string, 'ems_subtype' | 'rescue_subtype'>> = {
  ems: 'ems_subtype', rescue: 'rescue_subtype',
};
const TASK_NAMES: Record<Task, string> = {
  dispatch: '출동유형', subtype: '세부유형', urgency: '긴급도', symptom: '주증상',
};
const QUESTION_IDS = ['dispatch', 'urgency', 'ems_subtype', 'rescue_subtype', 'ems_symptom'];
const DISPATCH_LABELS = new Map([['fire', '화재'], ['rescue', '구조'], ['ems', '구급'], ['other', '기타']]);
const isRecord = (value: unknown): value is Record<string, unknown> =>
  !!value && typeof value === 'object' && !Array.isArray(value);
const nonempty = (value: unknown): value is string => typeof value === 'string' && value.trim().length > 0;
const probability = (value: unknown): value is number => typeof value === 'number' && value >= 0 && value <= 1;

function isHarness(value: unknown): value is Harness {
  if (!isRecord(value)) return false;
  const { model_id, policy_digest, thresholds, dispatch_overrides, gate, choices } = value;
  if (!nonempty(model_id) || !nonempty(policy_digest) || !isRecord(thresholds)
    || !Object.keys(TASK_NAMES).every((task) => probability(thresholds[task]))
    || !isRecord(dispatch_overrides) || !Object.entries(dispatch_overrides).every(
      ([key, override]) => DISPATCH_LABELS.has(key) && probability(override))
    || (gate !== 'allow' && gate !== 'wait' && gate !== 'review') || !isRecord(choices)) return false;
  return [choices.sft, choices.base].every((variant) => isRecord(variant)
    && Object.entries(variant).every(([question, choice]) =>
      QUESTION_IDS.includes(question) && isRecord(choice) && nonempty(choice.key) && nonempty(choice.label)
      && probability(choice.p) && (question !== 'dispatch' || DISPATCH_LABELS.get(choice.key) === choice.label)));
}

export const emptyZoro = (): ZoroState => ({
  probs: null, commits: {}, latencies: [], turn: 0, gate: 'wait', reviewTurn: null, blocked: false, profile: null,
});

export const canAct = (state: ZoroState) => !state.blocked && state.reviewTurn === null && !!state.commits.dispatch;
export const suspendHarness = (state: ZoroState): ZoroState => ({ ...state, blocked: true });

export function transferEvidence(state: ZoroState) {
  const { dispatch, subtype, symptom, urgency } = state.commits;
  if (!canAct(state) || dispatch?.key !== 'ems' || !subtype) return null;
  return { subtype: subtype.label, symptom: symptom?.label, urgency: urgency?.label };
}

export function harnessThreshold(profile: ZoroState['profile'], question: string, key?: string) {
  if (!profile) return undefined;
  if (question === 'dispatch') return (key && Object.hasOwn(profile.dispatch_overrides, key)
    ? profile.dispatch_overrides[key] : undefined) ?? profile.thresholds.dispatch;
  if (question === 'urgency') return profile.thresholds.urgency;
  return question === 'ems_symptom' ? profile.thresholds.symptom : profile.thresholds.subtype;
}

export function advanceHarness(
  state: ZoroState, probs: Probabilities | null, turn: { number: number; speaker: 0 | 1 },
  harness: unknown, variant: ZoroId,
): { next: ZoroState; note: string; accepted: boolean } {
  const rejected = (reason: string) => ({
    next: suspendHarness(state), note: `WAIT · ${turn.number}턴 ${reason}`, accepted: false,
  });
  if (turn.number !== state.turn + 1) return rejected('순서를 확인할 수 없습니다.');
  if (!isHarness(harness)) return rejected('판단 정책을 확인할 수 없습니다.');
  if (state.profile && (harness.model_id !== state.profile.model_id || harness.policy_digest !== state.profile.policy_digest)) {
    return rejected('모델 또는 판단 정책이 바뀌었습니다.');
  }
  const profile = state.profile ?? {
    model_id: harness.model_id, policy_digest: harness.policy_digest,
    thresholds: { ...harness.thresholds }, dispatch_overrides: { ...harness.dispatch_overrides },
  };
  const choices = harness.choices[variant];
  const commits = { ...state.commits };
  const callerAllowed = turn.speaker === 1 && harness.gate === 'allow';
  const conflicting = callerAllowed && commits.dispatch && choices.dispatch && commits.dispatch.key !== choices.dispatch.key;
  const reviewTurn = state.reviewTurn ?? (harness.gate === 'review' || conflicting ? turn.number : null);
  const next: ZoroState = {
    ...state, profile, probs, commits, turn: turn.number, reviewTurn, blocked: false,
    gate: reviewTurn !== null ? 'review' : harness.gate,
  };
  if (reviewTurn !== null) return { next, note: `재검토 · ${reviewTurn}턴부터 출동·이송 판단 보류`, accepted: true };
  const fresh: string[] = [];
  function commit(task: Task, choice?: Choice) {
    if (choice && !commits[task]) {
      commits[task] = { ...choice, turn: turn.number };
      fresh.push(`${TASK_NAMES[task]} ${choice.label}`);
    }
  }
  if (callerAllowed) {
    commit('dispatch', choices.dispatch);
    if (commits.dispatch) {
      const subtype = SUBTYPE_OF[commits.dispatch.key];
      if (subtype) commit('subtype', choices[subtype]);
      commit('urgency', choices.urgency);
      if (subtype === 'ems_subtype' && commits.subtype) commit('symptom', choices.ems_symptom);
    }
  }
  return {
    next, note: fresh.length ? `COMMIT · ${fresh.join(' · ')}` : `WAIT · ${turn.number}턴 추가 근거 대기`, accepted: true,
  };
}
