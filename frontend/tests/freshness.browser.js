async page => {
  const base = 'http://127.0.0.1:8024';
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto(base);
  await page.getByRole('combobox', { name: '시나리오' }).selectOption('reroute');
  await page.getByText('0 / 3턴', { exact: true }).waitFor();

  const eventResponse = (response, action) => response.url().endsWith('/events') &&
    response.request().method() === 'POST' && response.request().postDataJSON()?.action === action;
  const [next] = await Promise.all([
    page.waitForResponse(response => eventResponse(response, 'next')),
    page.getByRole('button', { name: '다음 발화', exact: true }).click(),
  ]);
  const sessionURL = base + '/api/sessions/' + (await next.json()).id;
  const vehicle = page.getByRole('article', { name: '차량 추천', exact: true });
  await vehicle.locator('.choice').filter({ hasText: '구급차 A' }).waitFor();
  await page.getByRole('region', { name: '역할별 판단' }).getByText('현재 판단', { exact: true }).waitFor();
  const dispatch = page.getByRole('button', { name: '모의 출동 적용', exact: true });
  const before = await (await page.request.get(sessionURL)).json();
  if (before.decisions.vehicle.choice !== 'A' || await dispatch.isDisabled()) throw Error('A must be ready before expiry');
  const expiry = await page.request.post(base + '/_test/expire-vehicle');
  if (!expiry.ok()) throw Error('could not advance fixture clock');

  const [rejected, refreshed] = await Promise.all([
    page.waitForResponse(response => eventResponse(response, 'dispatch')),
    page.waitForResponse(response => response.url() === sessionURL && response.request().method() === 'GET'),
    dispatch.click(),
  ]);
  if (rejected.status() !== 409 || (await rejected.json()).detail !== 'vehicle_not_ready') throw Error('expired dispatch was not rejected');
  const after = await refreshed.json();
  if (!refreshed.ok() || after.view_revision <= before.view_revision || after.version !== before.version ||
      after.decisions.vehicle.status !== 'WAIT' || after.decisions.vehicle.choice !== null ||
      after.applied_vehicle !== null || after.vehicles.find(v => v.id === 'A')?.availability !== 'unknown') {
    throw Error('409 recovery GET must withdraw expired A and advance the view revision');
  }
  const row = page.getByRole('region', { name: '차량 후보' }).getByRole('listitem').filter({ hasText: '구급차 A' });
  await row.getByText('정보 없음', { exact: true }).waitFor();
  await vehicle.locator('.status.wait').waitFor();
  if (!(await dispatch.isDisabled())) throw Error('dispatch still enabled after recovery');

  await page.getByRole('combobox', { name: '시나리오' }).selectOption('outage');
  await page.getByText('0 / 3턴', { exact: true }).waitFor();
  const nextTurn = page.getByRole('button', { name: '다음 발화', exact: true });
  await nextTurn.click();
  const hospital = page.getByRole('table', { name: '병원 후보 목록' }).getByRole('row').filter({ hasText: '성동 시뮬레이션 병원' });
  await hospital.getByText('가용', { exact: true }).waitFor();
  const observedAt = await hospital.locator('time').getAttribute('datetime');
  if (!observedAt) throw Error('initial hospital observation is missing');
  await nextTurn.click();
  for (let attempt = 0; attempt < 2; attempt += 1) {
    const [retry] = await Promise.all([
      page.waitForResponse(response => eventResponse(response, 'retry')),
      page.getByRole('button', { name: '재시도', exact: true }).click(),
    ]);
    const state = await retry.json();
    const h2 = state.hospitals.find(h => h.id === 'H2');
    if (!retry.ok() || state.external_status !== 'timeout' || h2?.availability !== 'unknown' ||
        h2.last_observation?.source_at !== observedAt || h2.last_observation.availability !== 'available') {
      throw Error('a failed retry lost the last hospital observation');
    }
    await hospital.getByText('현재 추천 불가', { exact: true }).waitFor();
    await hospital.getByText('마지막 확인 · 가용', { exact: true }).waitFor();
    if (await hospital.locator('time').getAttribute('datetime') !== observedAt) throw Error('stale observation was re-dated');
  }
  return { http: rejected.status(), vehicle: 'unknown', buttonDisabled: true, failedHospitalRetries: 2, observedAt };
}
