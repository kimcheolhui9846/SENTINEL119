async page => {
  await page.route('https://tile.openstreetmap.org/**', (route) => route.abort());
  let attempts = 0;
  let posts = 0;
  let locationsUnavailable = false;
  let sessions = 0;
  const base = {
    id: 'training', mode: 'live', scenario: 'ems', turn: 0, total_turns: 2,
    caller: { lat: 37.552, lon: 127.046, dong: '성수동1가', source: 'synthetic_in_dong',
      location_basis: { dataset_id: '71768', level: 'collection_city', city: '서울특별시',
        district: null, dong: null, count: 5,
        zone: { city: '서울특별시', district: '성동구', dong: '성수동1가' } } },
    turns: [],
    units: [{ id: 'ems-a', name: '구급 A', kind: 'ambulance', crew: ['1급 응급구조사'],
      equipment: ['제세동기'], route: [{ lat: 37.552, lon: 127.037 }, { lat: 37.560, lon: 127.049 }], status: 'patrol' },
    { id: 'fire-a', name: '소방 A', kind: 'fire', crew: ['소방대원'],
      equipment: ['관창'], route: [{ lat: 37.556, lon: 127.044 }, { lat: 37.549, lon: 127.050 }], status: 'patrol' }],
    hospital_source: { provider: 'nmc', status: 'live', fetched_at: '2026-11-19T01:14:00Z' },
    hospitals: [{ id: 'hospital-a', name: '<i>테스트의료센터</i>', address: '서울 성동구 테스트로 10',
      lat: 37.547, lon: 127.054, capabilities: ['cardiac_arrest'], available: true,
      location_source: 'nmc', operational_source: 'simulation' }],
    dispatch: { state: 'wait', category: null, unit_id: null, evidence_turn: null },
    subtype: { state: 'wait', code: null },
    link: { state: 'wait', hospital_id: null }, model_id: null, model_status: 'pending',
  };
  const firstTurn = { ...base, turn: 1, model_status: 'live', model_id: 'zoro-test',
    turns: [{ speaker: '신고자', text: '<b>몸이 아파요</b>' }],
    dispatch: { state: 'committed', category: 'ems', unit_id: 'ems-a', evidence_turn: 1 },
    units: [{ ...base.units[0], status: 'dispatched' }, base.units[1]],
  };
  const finalTurn = { ...firstTurn, turn: 2,
    turns: [...firstTurn.turns, { speaker: '신고자', text: '가슴 통증이 심해요' }],
    subtype: { state: 'committed', code: 'cardiac_arrest' },
    link: { state: 'committed', hospital_id: 'hospital-a' },
  };
  let snapshot = base;
  await page.route('**/api/drills**', async (route) => {
    const request = route.request();
    if (request.method() === 'GET') return route.fulfill({ json: snapshot });
    if (new URL(request.url()).pathname === '/api/drills') {
      if (request.postDataJSON().mode !== 'live') throw Error('Live is not the default');
      const { scenario } = request.postDataJSON();
      const result = { ...base, id: ++sessions === 1 ? base.id : base.id + sessions, scenario };
      if (locationsUnavailable) {
        result.hospital_source = { provider: 'nmc', status: 'unavailable', fetched_at: null };
        result.hospitals = [];
      } else if (scenario === 'fire') {
        result.hospital_source = { ...base.hospital_source, status: 'snapshot' };
        result.caller = { ...base.caller, location_basis: { ...base.caller.location_basis,
          level: 'label_dong', district: '성동구', dong: '성수동1가', count: 17 } };
        result.hospitals = [{ ...base.hospitals[0], lat: 37.535, lon: 127.059 }];
      }
      return route.fulfill({ json: result });
    }
    posts++;
    const expectedTurn = request.postDataJSON().expected_turn;
    if (expectedTurn !== snapshot.turn) throw Error('The request used a stale turn');
    if (++attempts === 1) {
      return route.fulfill({ status: 503, json: { detail: 'live_model_unavailable' } });
    }
    snapshot = expectedTurn === 0 ? firstTurn : finalTurn;
    return route.fulfill({ json: snapshot });
  });

  await page.goto('http://127.0.0.1:8020/');
  await page.getByRole('tab', { name: '자동 모의훈련' }).click();
  const panel = page.getByRole('tabpanel', { name: '자동 모의훈련' });
  await panel.locator('.drill-hero img').waitFor();
  await panel.getByRole('button', { name: '훈련 시작' }).click();
  const map = panel.getByRole('region', { name: '훈련 차량 및 신고 위치 지도' });
  await map.waitFor();
  if (await panel.locator('.drill-hero img').count()) throw Error('Preflight photo consumes map space');
  await panel.getByText(/AI Hub 서울 수집분/).waitFor();
  const facilityMarker = map.getByRole('button', { name: '<i>테스트의료센터</i>', exact: true });
  await facilityMarker.press('Enter');
  const popup = map.locator('.leaflet-popup-content');
  await popup.getByText('<i>테스트의료센터</i>', { exact: true }).waitFor();
  await popup.getByText('서울 성동구 테스트로 10', { exact: true }).waitFor();
  await popup.getByText(/훈련용 수용 설정/).waitFor();
  if (await popup.locator('i').count()) throw Error('Hospital name was interpreted as HTML');
  await map.getByRole('button', { name: /close|닫기/i }).click();
  await panel.getByText('훈련 세부 정보', { exact: true }).click();
  await panel.getByText('1급 응급구조사').waitFor();
  await panel.getByText('관창').waitFor();
  await panel.getByText('훈련 세부 정보', { exact: true }).click();
  const vehiclePositions = () => Array.from(document.querySelectorAll('.drill-marker--ambulance, .drill-marker--fire'))
    .map((marker) => getComputedStyle(marker).transform);
  const startingPositions = await page.evaluate(vehiclePositions);
  if (startingPositions.length !== 2) throw Error('Both patrol vehicles must appear on the map');
  await page.waitForFunction((initial) => {
    const markers = Array.from(document.querySelectorAll('.drill-marker--ambulance, .drill-marker--fire'));
    return markers.length === initial.length && markers.every((marker, index) =>
      getComputedStyle(marker).transform !== initial[index]);
  }, startingPositions);
  const caller = panel.getByRole('button', { name: '신고자 마커로 다음 발화 공개' });
  await caller.press('Enter');
  const retry = panel.getByRole('button', { name: '모델 다시 시도' });
  await retry.waitFor();
  if (posts !== 1 || !await caller.isDisabled()) throw Error('Model outage did not stop automatic progression');
  await retry.click();
  const turn = panel.getByRole('region', { name: '현재 발화' }).getByText('<b>몸이 아파요</b>', { exact: true });
  await turn.waitFor();
  await panel.getByText('출동 확정').waitFor();
  await page.waitForFunction(() => {
    const callerIcon = document.querySelector('.drill-marker--caller');
    const dispatched = document.querySelector('.drill-marker.dispatched');
    return callerIcon && dispatched && getComputedStyle(callerIcon).transform === getComputedStyle(dispatched).transform;
  }, null, { timeout: 8000 });
  await caller.click();
  await panel.getByText('<i>테스트의료센터</i> 연결').waitFor();
  await panel.locator('.drill-hospital-link').waitFor();
  if (posts !== 3) throw Error('Each authorized action must attempt exactly one turn');
  await page.setViewportSize({ width: 1280, height: 800 });
  if (await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1)) {
    throw Error('Drill causes horizontal overflow on a laptop screen');
  }
  await page.reload();
  await page.getByRole('tab', { name: '자동 모의훈련' }).click();
  await panel.getByLabel('신고 유형').selectOption('fire');
  await panel.getByRole('button', { name: '훈련 시작' }).click();
  await facilityMarker.waitFor();
  await map.locator('.drill-marker--fire').waitFor();
  const mapBox = await map.boundingBox();
  const hospitalBox = await facilityMarker.boundingBox();
  if (!mapBox || !hospitalBox || hospitalBox.x < mapBox.x || hospitalBox.y < mapBox.y
    || hospitalBox.x + hospitalBox.width > mapBox.x + mapBox.width
    || hospitalBox.y + hospitalBox.height > mapBox.y + mapBox.height) {
    throw Error('The initial map excludes the nearest published facility');
  }
  await panel.getByText('AI Hub 라벨 주소 · 성수동1가', { exact: true }).waitFor();
  await panel.getByText(/저장된 조회 결과/).waitFor();
  locationsUnavailable = true;
  await panel.getByText('훈련 설정', { exact: true }).click();
  await panel.getByRole('button', { name: '새 훈련 시작' }).click();
  await panel.getByText('표시할 공개 의료기관 위치를 확보하지 못했습니다.').waitFor();
  if (await map.locator('.drill-marker--hospital').count()) throw Error('Training invented a hospital when NMC was unavailable');
  return { liveDefault: true, patrol: true, explicitRetry: true, keyboard: true, safeTranscript: true,
    clickAfterArrival: true, link: true, laptop: true, actualHospitalsInBothScenarios: true, noFabricatedLocations: true };
}
