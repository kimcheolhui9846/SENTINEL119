async page => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto('http://127.0.0.1:8000/');
  const scenario = page.getByRole('combobox', { name: '시나리오' });
  const dispatch = page.getByRole('button', { name: '모의 출동 적용', exact: true });
  const next = page.getByRole('button', { name: '다음 발화', exact: true });
  const eventPath = /\/api\/sessions\/[^/]+\/events$/;
  const openScenario = async (id) => {
    await scenario.selectOption(id);
    await page.getByText('0 / 3턴', { exact: true }).waitFor();
  };
  const hospitalDecision = page.getByRole('article', { name: '병원 후보', exact: true });

  await openScenario('outage');
  await next.click();
  const table = page.getByRole('table', { name: '병원 후보 목록' });
  await table.waitFor({ timeout: 5000 });
  const hospitalColumns = await table.getByRole('columnheader').allTextContents();
  if (hospitalColumns.join(',') !== '병원명,수용 상태,정보 시각,ETA') {
    throw Error(`hospital columns: ${hospitalColumns.join(',')}`);
  }
  const hospital = table.getByRole('row').filter({ has: page.getByRole('rowheader', { name: /성동 시뮬레이션 병원/ }) });
  await hospital.getByText('가용', { exact: true }).waitFor();
  const originalTimestamp = await hospital.locator('time').getAttribute('datetime');
  if (!originalTimestamp) throw Error('missing fresh observation timestamp');
  await next.click();
  await page.getByRole('alert').filter({ hasText: '초과' }).waitFor();
  await hospitalDecision.locator('.status.wait').waitFor();
  await hospital.getByText('현재 추천 불가', { exact: true }).waitFor();
  await hospital.getByText('마지막 확인 · 가용', { exact: true }).waitFor();
  if (await hospital.locator('time').getAttribute('datetime') !== originalTimestamp) {
    throw Error('stale timestamp must be the last observation time, not the timeout time');
  }
  const neverObserved = table.getByRole('row').filter({ hasText: '동부 시뮬레이션 병원' });
  await neverObserved.getByText('이전 확인 정보 없음', { exact: true }).waitFor();
  if (await neverObserved.locator('time').count()) throw Error('timestamp shown for an unobserved hospital');
  const sameColumns = await table.getByRole('row').evaluateAll((rows) => {
    const columns = (row) => Array.from(row.cells, (cell) => cell.getBoundingClientRect().left);
    const header = columns(rows[0]);
    return rows.slice(1).every((row) => columns(row).every((left, index) => Math.abs(left - header[index]) < 1));
  });
  if (!sameColumns) throw Error('hospital values are not aligned in desktop columns');
  await page.getByRole('button', { name: '재시도', exact: true }).click();
  await hospitalDecision.locator('.status.commit').waitFor();
  await hospital.getByText('가용', { exact: true }).waitFor();
  if (await hospital.getByText('마지막 확인', { exact: false }).count()) throw Error('stale marker survived recovery');
  if (await hospital.locator('time').getAttribute('datetime') === originalTimestamp) throw Error('recovered observation time did not change');

  await openScenario('reroute');
  const readyResponse = page.waitForResponse((response) => eventPath.test(response.url()) && response.request().postDataJSON()?.action === 'next');
  await next.click();
  const initial = await (await readyResponse).json();
  const vehicle = page.getByRole('article', { name: '차량 추천', exact: true });
  await vehicle.locator('.choice').filter({ hasText: '구급차 A' }).waitFor();
  const viewed = await (await page.request.get(new URL(`/api/sessions/${initial.id}`, page.url()).href)).json();
  const posted = [];
  const race = async (route) => {
    const payload = route.request().postDataJSON();
    if (payload.action === 'dispatch') {
      posted.push(payload);
      if (posted.length === 1) {
        const advance = await page.request.post(route.request().url(), { data: { action: 'next', expected_version: viewed.version } });
        if (!advance.ok()) throw Error(`could not advance the racing session: ${advance.status()}`);
      }
    }
    await route.continue();
  };
  await page.route(eventPath, race);
  try {
    const conflict = page.waitForResponse((response) => eventPath.test(response.url()) && response.request().postDataJSON()?.action === 'dispatch');
    await dispatch.click();
    const rejected = await conflict;
    const first = posted[0];
    if (first?.expected_vehicle_id !== 'A' || !Number.isInteger(first.expected_vehicle_revision) ||
      first.expected_vehicle_revision !== viewed.decisions.vehicle.recommendation_revision || first.expected_version !== viewed.version) {
      throw Error(`dispatch did not send the rendered vehicle and revisions: ${JSON.stringify(first)}`);
    }
    if (rejected.status() !== 409) throw Error(`expected conflict, received ${rejected.status()}`);
    await page.getByRole('alert').filter({ hasText: '다시 확인' }).waitFor();
    await vehicle.locator('.choice').filter({ hasText: '구급차 B' }).waitFor();
    await page.getByRole('region', { name: '역할별 판단' }).getByText('현재 판단', { exact: true }).waitFor();
    const urgency = page.getByRole('article', { name: '긴급도', exact: true });
    await urgency.locator('.status.wait').waitFor();
    await urgency.getByText('현재 발화에서 판단 근거를 더 확인해야 합니다.').waitFor();
    if (posted.length !== 1 || await page.getByText(/모의 출동을 적용했습니다/).count()) {
      throw Error('conflict applied or automatically replayed dispatch');
    }
    await dispatch.click();
    await page.getByText('구급차 B에 모의 출동을 적용했습니다.').waitFor();
    if (posted.length !== 2 || posted[1].expected_vehicle_id !== 'B' ||
      posted[1].expected_version <= first.expected_version || posted[1].expected_vehicle_revision <= first.expected_vehicle_revision) {
      throw Error(`explicit reconfirmation did not dispatch the updated recommendation: ${JSON.stringify(posted)}`);
    }
  } finally {
    await page.unroute(eventPath, race);
  }
  return { hospitalColumns, staleObservation: true, hospitalRecovery: true, dispatchRace: true, clinicalHold: true };
}
