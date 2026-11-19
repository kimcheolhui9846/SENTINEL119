async page => {
  const base = 'http://127.0.0.1:8020';
  const runs = [
    { mode: 'live', scenario: 'ems', width: 1440, height: 900 },
    { mode: 'live', scenario: 'fire', width: 1280, height: 800 },
    { mode: 'simulation', scenario: 'ems', width: 1920, height: 1080 },
  ];
  const summaries = [];

  function check(condition, message) {
    if (!condition) throw Error(message);
  }

  async function postResponse(path, action) {
    const [response] = await Promise.all([
      page.waitForResponse(r => new URL(r.url()).pathname === path && r.request().method() === 'POST'),
      action(),
    ]);
    check(response.ok(), path + ' returned HTTP ' + response.status());
    return response.json();
  }

  for (const { mode, scenario, width, height } of runs) {
    await page.setViewportSize({ width, height });
    await page.goto(base + '/?check=' + mode + '-' + scenario + '#drill');
    const panel = page.getByRole('tabpanel', { name: '자동 모의훈련' });
    await panel.getByLabel('실행 모드').selectOption(mode);
    await panel.getByLabel('신고 유형').selectOption(scenario);
    const initial = await postResponse('/api/drills', () => panel.getByRole('button', { name: '훈련 시작', exact: true }).click());
    const marker = panel.getByRole('button', { name: '신고자 마커로 다음 발화 공개' });
    await marker.waitFor();
    const basis = initial.caller.location_basis;
    check(await marker.count() === 1 && basis.level === 'label_dong' && basis.dong === initial.caller.dong, '라벨 지역과 신고 핀');
    const district = basis.zone.district;
    const directoryResponse = await page.request.get(base + '/api/hospital-locations', { params: { district } });
    const directory = await directoryResponse.json();
    check(directory.status === 'ok' && initial.hospitals.length > 0, '공개 의료기관 조회');
    for (const hospital of initial.hospitals) {
      const ref = directory.items.find(item => item.id === hospital.id);
      check(ref && ref.name === hospital.name && ref.lat === hospital.lat && ref.lon === hospital.lon, '기관 이름과 좌표');
      check(hospital.location_source === 'nmc' && hospital.operational_source === 'simulation', '위치 및 모의 수용 출처');
    }
    check(await panel.locator('.drill-marker--hospital').count() === initial.hospitals.length, '지도에 표시한 기관 수');
    const mapBox = await panel.locator('.drill-map-card').boundingBox();
    const latestBox = await panel.getByRole('region', { name: '현재 발화' }).boundingBox();
    check(mapBox && latestBox && latestBox.x >= mapBox.x + mapBox.width && latestBox.y < height, '지도 및 현재 발화의 데스크톱 배치');

    await panel.getByRole('button', { name: '현장 중심' }).click();
    const vehicleSelector = '.drill-marker--ambulance,.drill-marker--fire';
    const vehicleIcons = panel.locator(vehicleSelector);
    const before = await vehicleIcons.evaluateAll(icons => icons.map(icon => icon.style.transform));
    check(before.length === 3, '구급차 두 대와 소방차 한 대');
    await page.waitForFunction(({ selector, positions }) =>
      [...document.querySelectorAll(selector)].every((icon, i) => icon.style.transform !== positions[i]),
      { selector: vehicleSelector, positions: before }, { timeout: 12_000 });

    const steps = [];
    for (let turn = 1; turn <= 3; turn++) {
      const state = await postResponse('/api/drills/' + initial.id + '/turns', () => marker.click());
      await panel.getByText(turn + ' / 3턴', { exact: true }).waitFor();
      check(state.turn === turn && state.turns.length === turn, '클릭당 발화 하나');
      check(state.caller.lat === initial.caller.lat && state.caller.lon === initial.caller.lon, '세션 내 신고 위치');
      check(state.model_status === (mode === 'live' ? 'live' : 'simulation'), '판정 출처');

      const unit = turn === 1 ? null : (scenario === 'ems' ? 'A2' : 'F1');
      const requiresHospital = turn === 3 && scenario === 'ems';
      check(state.dispatch.unit_id === unit, '턴별 출동 차량');
      check(state.units.filter(v => v.status === 'dispatched').length === (unit ? 1 : 0), '중복 출동');
      check(state.subtype.code === (requiresHospital ? 'cardiac_arrest' : null), '세부 판정 시점');
      if (unit) await panel.getByText('출동 확정', { exact: true }).waitFor();
      if (requiresHospital) {
        const target = state.hospitals.find(h => h.id === state.link.hospital_id);
        check(target && target.available && target.capabilities.includes(state.subtype.code), '병원 훈련 수용 조건');
        check((await panel.innerText()).includes(target.name), '연결 기관 표시');
      } else {
        check(state.link.hospital_id === null, '근거 확정 전 병원 대기');
      }
      steps.push([turn, state.dispatch.unit_id, state.subtype.code, state.link.hospital_id]);
    }
    check(await marker.getAttribute('aria-disabled') === 'true', '종료 후 핀 상태');
    check(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), '가로 넘침');
    summaries.push({ mode, scenario, width, district, source: initial.hospital_source.status, steps });
  }
  return summaries;
}
