async page => {
  let retryCalls = 0;
  const payload = (district, status, items = []) => ({
    district, status, source: 'nmc', fetched_at: items.length ? '2026-11-18T10:00:00Z' : null,
    total_count: items.length, items, reason: null,
  });
  await page.route('https://tile.openstreetmap.org/**', (route) => route.abort());
  await page.route('**/api/hospital-locations?*', async (route) => {
    const district = new URL(route.request().url()).searchParams.get('district');
    const item = (id, name, category, emergency_code, lat, lon) =>
      ({ id, name, category, emergency_code, lat, lon, address: '서울특별시 성동구' });
    const response = district === '성동구' ? payload(district, 'ok', [
      item('a', '<b>테스트종합병원</b>', '종합병원', 'G099', 37.561, 127.043),
      item('b', '테스트응급의원', '의원', 'G001', 37.551, 127.043),
      item('c', '일반의원', '의원', 'G099', 37.551, 127.053),
      item('d', '잘못된좌표병원', '병원', '', 999, 127.053),
    ]) : district === '강북구' ? (++retryCalls === 1 ? payload(district, 'unavailable')
      : payload(district, 'stale', [item('e', '저장병원', '병원', '', 37.640, 127.025)]))
      : payload(district, district === '강남구' ? 'unconfigured' : 'ok');
    await route.fulfill({ json: response });
  });

  await page.goto('http://127.0.0.1:8020/');
  const map = page.getByRole('region', { name: '서울 의료기관 지도', exact: true });
  await map.waitFor({ timeout: 3000 });
  const list = map.getByRole('list', { name: '지도에 표시된 의료기관' });
  const name = '<b>테스트종합병원</b>';
  await list.getByRole('button', { name }).waitFor();
  await list.getByRole('button', { name: /테스트응급의원/ }).waitFor();
  if (await list.getByRole('button').count() !== 2) throw Error('Default filter or invalid-coordinate handling failed');
  await map.getByText('지도 2곳', { exact: true }).waitFor();
  await map.getByText(/배경 지도.*불러오지 못했습니다/).waitFor();
  await map.locator('.leaflet-overlay-pane path[stroke="#38d9e2"]').waitFor();

  await list.getByRole('button', { name }).press('Enter');
  const popup = map.locator('.leaflet-popup-content');
  await popup.getByText(name, { exact: true }).waitFor();
  if (await popup.locator('b').count()) throw Error('The popup interpreted a facility name as HTML');
  if (await list.getByRole('button', { name }).getAttribute('aria-pressed') !== 'true') {
    throw Error('Keyboard selection did not synchronize with the list');
  }
  const emergency = map.getByRole('button', { name: '테스트응급의원', exact: true });
  await emergency.press('Enter');
  await popup.getByText('테스트응급의원', { exact: true }).waitFor();
  if (await list.getByRole('button', { name: /테스트응급의원/ }).getAttribute('aria-pressed') !== 'true') {
    throw Error('Marker keyboard activation did not synchronize with the list');
  }

  await map.getByRole('button', { name: '전체', exact: true }).click();
  await list.getByRole('button', { name: /일반의원/ }).waitFor();
  const search = map.getByRole('searchbox', { name: '기관명 검색' });
  await search.fill('일반의원');
  await map.getByText('지도 1곳', { exact: true }).waitFor();
  if (await list.getByRole('button').count() !== 1) throw Error('Search and marker filter disagree');

  const district = map.getByRole('combobox', { name: '서울 자치구' });
  if (await district.locator('option').count() !== 25) throw Error('Seoul boroughs are missing');
  await district.selectOption('강남구');
  await map.getByText(/인증키가 설정되지 않았습니다/).waitFor();
  if (await list.getByRole('button').count()) throw Error('Markers from the previous district leaked');
  await district.selectOption('강동구');
  await map.getByText(/표시할 의료기관 위치가 없습니다/).waitFor();
  await district.selectOption('강북구');
  await map.getByText('의료기관 위치를 불러오지 못했습니다.', { exact: true }).waitFor();
  await map.getByRole('button', { name: '위치 조회 다시 시도' }).click();
  await list.getByRole('button', { name: /저장병원/ }).waitFor();
  await map.getByText('저장된 위치 정보').waitFor();
  const timestamp = map.locator('time[datetime="2026-11-18T10:00:00.000Z"]');
  await timestamp.waitFor();
  if (!(await timestamp.textContent()).includes('KST')) throw Error('The timestamp has no timezone');
  return { defaults: true, offlineBoundary: true, markerCount: true, keyboard: true, safePopup: true,
    search: true, missingKey: true, empty: true, retry: true, stale: true };
}
