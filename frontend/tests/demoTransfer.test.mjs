import assert from 'node:assert/strict';
import { test } from 'node:test';
import { advanceTransfer, emptyTransfer } from '../src/demoTransfer.ts';

const hospital = (id) => ({ hpid: id, name: id, route: { path: [{ lat: 37.5, lon: 127 }], duration_s: 12 } });
const H1 = hospital('H1');
const H2 = hospital('H2');
const suggestion = (chosen) => ({
  response: { status: 'ok', hospitals: [chosen], zoro: { choice: chosen.hpid } },
  basis: { subtype: '심정지' },
});
const request = (state, key = 'ems-arrest') => advanceTransfer(state, { type: 'request', key });
const receive = (state, generation, value) =>
  advanceTransfer(state, { type: 'receive', generation, suggestion: value });
const issue = (state, authorized, startedAt = 1200) =>
  advanceTransfer(state, { type: 'issue', authorized, unitId: 'ambulance-1', startedAt });
const block = (state) => advanceTransfer(state, { type: 'block' });

test('a temporary model failure or review preserves the issued transport and ignores a second destination', () => {
  const pending = request(emptyTransfer());
  const shown = receive(pending, pending.generation, suggestion(H1));
  assert.equal(issue(shown, false), shown);
  const issued = issue(shown, true);
  assert.deepEqual(issued.issued, { unitId: 'ambulance-1', hospital: H1, startedAt: 1200 });
  assert.equal(issued.suggestion, shown.suggestion);
  const suspended = block(issued);
  assert.equal(suspended, issued);
  assert.equal(request(suspended), issued);
  assert.equal(request(suspended, 'changed-ems-evidence'), issued);
  assert.equal(receive(suspended, pending.generation, suggestion(H2)), issued);
  assert.equal(issue(suspended, true, 5000), issued);
  const nextCall = emptyTransfer();
  assert.equal(nextCall.issued, null);
  assert.notEqual(request(nextCall), nextCall);
});

test('a pending or never-issued selection is invalidated and the same evidence can be retried', () => {
  const pending = request(emptyTransfer());
  assert.equal(request(pending), pending);
  const suspended = block(pending);
  assert.equal(receive(suspended, pending.generation, suggestion(H1)), suspended);
  assert.equal(issue(suspended, false), suspended);
  const retry = request(suspended);
  assert.notEqual(retry.generation, pending.generation);
  const shown = receive(retry, retry.generation, suggestion(H2));
  assert.equal(shown.suggestion.response.zoro.choice, 'H2');
  const reviewed = block(shown);
  assert.equal(reviewed.suggestion, null);
  assert.equal(issue(reviewed, true), reviewed);
});

test('an unavailable or failed hospital lookup can retry the same evidence', () => {
  const pending = request(emptyTransfer());
  const unavailable = receive(pending, pending.generation,
    { response: { status: 'unavailable' }, basis: { subtype: '심정지' } });
  assert.equal(issue(unavailable, true), unavailable);
  const retry = request(unavailable);
  assert.notEqual(retry.generation, pending.generation);
  const failed = receive(retry, retry.generation, null);
  assert.notEqual(request(failed), failed);
});
