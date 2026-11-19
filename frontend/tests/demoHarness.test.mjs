import assert from 'node:assert/strict';
import { test } from 'node:test';
import { advanceHarness, canAct, emptyZoro, harnessThreshold, transferEvidence } from '../src/demoHarness.ts';

const thresholds = { dispatch: .9, subtype: .85, urgency: .85, symptom: .8 };
const fire = { key: 'fire', label: '화재', p: .78 };
const ems = { key: 'ems', label: '구급', p: .99 };
const arrest = { key: 'cardiac_arrest', label: '심정지', p: .94 };
const symptom = { key: 'arrest', label: '심정지·호흡정지', p: .92 };
const harness = (sft = {}, changes = {}) => ({
  model_id: 'team-demo/sft/example@sha256:123',
  policy_digest: 'policy-example',
  thresholds,
  dispatch_overrides: { fire: .75 },
  gate: 'allow',
  choices: { sft, base: {} },
  ...changes,
});
const raw = { dispatch: { fire: .99, ems: .01 }, ems_symptom: { arrest: 1 } };
const advance = (state, turn, policy, speaker = 1, variant = 'sft') =>
  advanceHarness(state, raw, { number: turn, speaker }, policy, variant);

test('raw leading scores are display only and server-approved fire can commit', () => {
  const first = advance(emptyZoro(), 1, harness());
  assert.equal(first.accepted, true);
  assert.equal(first.next.probs, raw);
  assert.deepEqual(first.next.commits, {});
  const second = advance(first.next, 2, harness({ dispatch: fire }));
  assert.deepEqual(second.next.commits.dispatch, { ...fire, turn: 2 });
  assert.equal(canAct(second.next), true);
});

test('operator turn, wait gate, and other variant cannot supply a new commitment', () => {
  const operator = advance(emptyZoro(), 1, harness({ dispatch: ems, ems_subtype: arrest }), 0);
  const wait = advance(operator.next, 2, harness({ dispatch: ems }, { gate: 'wait' }));
  const base = advance(wait.next, 3, harness({ dispatch: ems }), 1, 'base');
  assert.deepEqual(operator.next.commits, {});
  assert.deepEqual(wait.next.commits, {});
  assert.deepEqual(base.next.commits, {});
  assert.equal(canAct(wait.next), false);
  const emsCommit = advance(base.next, 4, harness({ dispatch: ems, ems_symptom: symptom }));
  assert.equal(emsCommit.next.commits.subtype, undefined);
  assert.equal(emsCommit.next.commits.symptom, undefined);
  const subtypeCommit = advance(emsCommit.next, 5, harness({ ems_subtype: arrest, ems_symptom: symptom }));
  assert.deepEqual(subtypeCommit.next.commits.subtype, { ...arrest, turn: 5 });
  assert.deepEqual(subtypeCommit.next.commits.symptom, { ...symptom, turn: 5 });
});

test('explicit review and a conflicting approved category latch the first review turn', () => {
  const initial = advance(emptyZoro(), 1, harness({ dispatch: ems })).next;
  for (const conflicted of [harness({}, { gate: 'review' }), harness({ dispatch: fire })]) {
    const review = advance(initial, 2, conflicted);
    assert.equal(review.next.reviewTurn, 2);
    assert.equal(canAct(review.next), false);
    assert.deepEqual(review.next.commits.dispatch, { ...ems, turn: 1 });
    const later = advance(review.next, 3, harness({ dispatch: ems, ems_subtype: arrest }));
    assert.equal(later.next.reviewTurn, 2);
    assert.equal(later.next.commits.subtype, undefined);
    assert.match(later.note, /2턴/);
  }
});

test('missing or malformed harness and identity or digest drift suspend action without advancing', () => {
  const initial = advance(emptyZoro(), 1, harness({ dispatch: ems })).next;
  for (const invalid of [undefined, harness({}, { model_id: 'different' }), harness({}, { policy_digest: 'different' }),
    harness({}, { choices: { sft: {} } }), harness({ dispatch: { ...fire, p: 1.01 } }),
    harness({}, { thresholds: { ...thresholds, dispatch: NaN } })]) {
    const response = advance(initial, 2, invalid);
    assert.equal(response.accepted, false);
    assert.equal(response.next.turn, 1);
    assert.equal(canAct(response.next), false);
    assert.match(response.note, /2턴/);
    assert.deepEqual(response.next.commits, initial.commits);
    assert.equal(advance(response.next, 2, harness()).accepted, true);
  }
  assert.equal(advance(initial, 1, harness()).accepted, false);
  assert.equal(advance(initial, 3, harness()).accepted, false);
  for (const changed of [{ model_id: 'different' }, { policy_digest: 'different' }]) {
    assert.equal(advance(initial, 2, harness({}, changed)).note,
      'WAIT · 2턴 모델 또는 판단 정책이 바뀌었습니다. 이전 모델과 정책을 복원하거나 새 통화를 시작하세요.');
  }
  assert.match(advance(initial, 2, undefined).note, /같은 턴을 다시 시도할 수 있습니다\.$/);
});

test('threshold display uses the server override for any category and handles empty overrides', () => {
  const normal = advance(emptyZoro(), 1, harness({}, { dispatch_overrides: {} }));
  assert.equal(normal.accepted, true);
  assert.equal(harnessThreshold(normal.next.profile, 'dispatch', 'fire'), thresholds.dispatch);
  const alternate = advance(emptyZoro(), 1, harness({}, { dispatch_overrides: { rescue: .87 } }));
  assert.equal(alternate.accepted, true);
  assert.equal(harnessThreshold(alternate.next.profile, 'dispatch', 'rescue'), .87);
  assert.equal(advance(emptyZoro(), 1, harness({}, { dispatch_overrides: { unexpected: .5 } })).accepted, false);
});

test('subtype display accepts an optional server override and rejects malformed subtype policies', () => {
  const legacy = advance(emptyZoro(), 1, harness());
  assert.equal(harnessThreshold(legacy.next.profile, 'rescue_subtype'), thresholds.subtype);
  const alternate = advance(emptyZoro(), 1, harness({}, { subtype_overrides: { rescue_subtype: .68 } }));
  assert.equal(alternate.accepted, true);
  assert.equal(harnessThreshold(alternate.next.profile, 'rescue_subtype'), .68);
  assert.equal(harnessThreshold(alternate.next.profile, 'ems_subtype'), thresholds.subtype);
  for (const subtype_overrides of [null, [], { dispatch: .9 }, { ems_subtype: 2 }, { rescue_subtype: NaN }]) {
    assert.equal(advance(emptyZoro(), 1, harness({}, { subtype_overrides })).accepted, false);
  }
});

test('transfer requires a confirmed EMS subtype and uses only confirmed symptom evidence', () => {
  const dispatch = advance(emptyZoro(), 1, harness({ dispatch: ems })).next;
  assert.equal(transferEvidence(dispatch), null);
  const subtype = advance(dispatch, 2, harness({ ems_subtype: arrest })).next;
  assert.deepEqual(transferEvidence(subtype), { subtype: '심정지', symptom: undefined, urgency: undefined });
  const confirmed = advance(subtype, 3, harness({ ems_symptom: symptom })).next;
  assert.equal(transferEvidence(confirmed).symptom, symptom.label);
  assert.equal(transferEvidence(advance(confirmed, 4, harness({}, { gate: 'review' })).next), null);
  assert.equal(transferEvidence(advance(confirmed, 4, undefined).next), null);
});
