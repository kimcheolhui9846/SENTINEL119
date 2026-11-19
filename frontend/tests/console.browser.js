async page => {
  await page.setViewportSize({width:1440,height:900});
  await page.goto('http://127.0.0.1:8000/');
  await page.getByRole('heading',{name:'신고 관제'}).waitFor();
  const card=(name)=>page.getByRole('article',{name, exact:true});
  const click=(name)=>page.getByRole('button',{name,exact:true}).click();
  const waitChoice=async (name,choice)=>{await card(name).locator('.choice').filter({hasText:choice}).waitFor({timeout:8000});};
  const waitStatus=async (name,status)=>{await card(name).locator('.status').first().filter({hasText:status}).waitFor({timeout:8000});};
  const scenario=async (id)=>{await page.getByRole('combobox',{name:'시나리오',exact:true}).selectOption(id);await page.getByText(/^0 \/ \d+턴$/).waitFor();};
  const assertProgress = async (expected) => {
    const actual = await page.locator('.turn-ticks i').evaluateAll((ticks) => ticks.map((tick) => tick.dataset.filled));
    if (actual.join(',') !== expected.join(',')) {
      throw Error(`turn progress expected ${expected.join(',')}, received ${actual.join(',')}`);
    }
  };
  const activeTurn = page.locator('li.turn[aria-current="step"]');
  const assertActiveTurn = async (text) => {
    const actual = await activeTurn.locator('.turn-text').allTextContents();
    if (actual.length !== 1 || actual[0] !== text) {
      throw Error(`expected one active turn with text ${text}, received ${JSON.stringify(actual)}`);
    }
  };
  const assertEtas = async (regionName, expected) => {
    const region = page.getByRole('region', {name: regionName, exact: true});
    const actual = await region.locator('.candidate-eta').allTextContents();
    if (actual.join(',') !== expected.join(',')) {
      throw Error(`${regionName} ETA labels expected ${expected.join(',')}, received ${actual.join(',')}`);
    }
  };
  await page.getByRole('button',{name:'다음 발화'}).waitFor();
  await page.getByText('0 / 4턴', {exact: true}).waitFor();
  await assertProgress(['false', 'false', 'false', 'false']);
  if (await activeTurn.count()) throw Error('turn active before the first utterance');
  await click('다음 발화'); await page.getByText('1 / 4턴',{exact:true}).waitFor();
  await assertProgress(['true', 'false', 'false', 'false']);
  await assertActiveTurn('여기 사람이 모여 있어요. 도와주세요.');
  await click('다음 발화'); await waitChoice('차량 추천','구급차 A');await waitStatus('병원 후보','보류');
  await assertProgress(['true', 'true', 'false', 'false']);
  await assertActiveTurn('성수역 2번 출구 광장이에요. 한 분이 넘어져 구급차가 필요해요.');
  await assertEtas('차량 후보', ['가상 ETA 3분', '가상 ETA 9분']);
  await assertEtas('병원 후보 목록', ['가상 ETA 8분', '가상 ETA 11분', '가상 ETA 16분']);
  await card('차량 추천').locator('summary').click();
  await card('차량 추천').getByText('2번 발화').waitFor();
  await click('다음 발화');await waitChoice('병원 후보','성동 시뮬레이션 병원');
  await click('다음 발화');await waitChoice('차량 추천','구급차 B');
  await page.getByRole('tab',{name:'판단 이력'}).click();await page.getByRole('heading',{name:'판단 변경 이력'}).waitFor();
  await page.getByRole('tab',{name:'관제 화면'}).click();
  const workspaceTab = page.getByRole('tab', {name: '관제 화면', exact: true});
  const historyTab = page.getByRole('tab', {name: '판단 이력', exact: true});
  await workspaceTab.press('ArrowRight');
  await page.getByRole('tabpanel', {name: '판단 이력', exact: true}).waitFor();
  if (!await historyTab.evaluate((tab) => tab === document.activeElement && tab.getAttribute('aria-selected') === 'true')) {
    throw Error('ArrowRight did not focus and select the history tab');
  }
  await historyTab.press('ArrowLeft');
  await page.getByRole('tabpanel', {name: '관제 화면', exact: true}).waitFor();
  if (!await workspaceTab.evaluate((tab) => tab === document.activeElement && tab.getAttribute('aria-selected') === 'true')) {
    throw Error('ArrowLeft did not focus and select the workspace tab');
  }
  await scenario('reroute');await click('다음 발화');await waitChoice('차량 추천','구급차 A');
  await click('다음 발화');await waitChoice('차량 추천','구급차 B');
  await click('모의 출동 적용');await page.getByText('구급차 B에 모의 출동을 적용했습니다.').waitFor();
  if(!await page.getByRole('button',{name:'모의 출동 적용'}).isDisabled())throw Error('duplicate dispatch enabled');
  await scenario('outage');await click('다음 발화');await waitChoice('차량 추천','구급차 A');
  await click('다음 발화');await page.getByRole('alert').filter({hasText:'초과'}).waitFor();
  await waitStatus('병원 후보','보류');await click('재시도');await waitChoice('병원 후보','성동 시뮬레이션 병원');
  await waitChoice('차량 추천','구급차 A');
  await page.getByRole('textbox',{name:'직접 입력'}).fill('위치를 정정합니다.');
  await click('전송');await page.getByText('위치를 정정합니다.',{exact:true}).waitFor();
  await page.getByText('직접 입력 · 3턴', {exact: true}).waitFor();
  if (await page.locator('.turn-ticks').count()) throw Error('turn progress appeared during manual entry');
  for(const n of ['출동유형','긴급도','차량 추천','병원 후보'])await waitStatus(n,'보류');
  await click('초기화');await page.getByText('0 / 3턴',{exact:true}).waitFor();
  await assertProgress(['false', 'false', 'false']);
  if (await activeTurn.count()) throw Error('turn active after reset');
  await click('자동 재생');await page.getByText('3 / 3턴',{exact:true}).waitFor({timeout:12000});
  await assertProgress(['true', 'true', 'true']);
  const mobile = [];
  for (const width of [320, 390]) {
    await page.setViewportSize({width, height: 844});
    const dims = await page.evaluate(() => ({viewport: innerWidth, content: document.documentElement.scrollWidth}));
    if (dims.content > dims.viewport + 1) throw Error('horizontal overflow ' + JSON.stringify(dims));
    mobile.push(dims);
  }
  return {threeScenarios:true,correction:true,evidence:true,history:true,dispatchOnce:true,retry:true,manual:true,autoplay:true,progress:true,syntheticEtas:true,keyboardTabs:true,mobile};
}
