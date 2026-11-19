# 서울 자동 모의훈련

앱에서 **자동 모의훈련 → 구급 또는 화재 → 훈련 시작**을 선택한다. 신고자 핀을 누르거나 키보드로 선택하면 다음 발화 하나가 공개된다. 최신 발화와 출동·세부 유형·병원 연결 판단은 지도 옆에 표시된다. 데모는 가상 자원만 움직인다.

## 로컬 실행

Python 3.11 이상과 Node.js 20.19 이상(또는 22.12 이상)을 사용한다. Windows PowerShell에서 저장소 루트를 기준으로 실행한다.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
npm.cmd --prefix frontend ci
npm.cmd --prefix frontend run build
.\.venv\Scripts\python.exe -m uvicorn backend.drill_console:create_console --factory --host 127.0.0.1 --port 8020
```

[자동 모의훈련 열기](http://127.0.0.1:8020/#drill)에서 **시뮬레이션**을 선택하면 사전 작성한 모델 출력과 저장된 의료기관 위치로 바로 재생할 수 있다. 키와 모델 서버 없이도 해당 모드가 동작한다.

### 실시간 Zoro 모드

팀 모델 서버 주소를 받아 `ZORO_INTAKE_BASE_URL`에 설정한다. 모델 서버 접속에 SSH가 필요하다면 별도 터미널에서 아래 예시의 `MODEL_HOST`와 원격 포트 `8119`를 실제 접속 정보로 바꿔 실행한다.

```powershell
ssh -N -L 18119:127.0.0.1:8119 MODEL_HOST
```

서버를 시작할 PowerShell에서 전달받은 주소를 설정한다. 아래 값은 위 SSH 포트 포워딩의 예시다.

```powershell
$env:ZORO_INTAKE_BASE_URL = 'http://127.0.0.1:18119'
```

기본 체크포인트는 2026년 11월 19일에 확인한 `zoro-hier-h1b`다. 다른 체크포인트로 검증할 때는 팀이 확인한 전체 응답 식별자를 `ZORO_INTAKE_MODEL_ID`에 넣고 두 시나리오를 다시 재생한다. 국립중앙의료원 키는 필요할 때만 서버 환경 변수 `NMC_HOSPITAL_API_KEY`로 설정한다. 키가 없으면 훈련은 저장된 의료기관 위치를 사용하고 별도 기관 지도는 조회 키가 필요한 상태를 표시한다. 환경 변수 설정 후 위 서버 명령을 새로 실행한다. 모델 오류는 현재 턴에서 재시도할 수 있다.

## 위치 자료와 훈련 설정

- AI Hub 71768의 서울 검증 라벨 파일키 `539707`에서 최상위 `address`의 시·구·법정동을 대조했다. 서울특별시 성동구 성수동1가와 일치한 라벨은 17건이다. 앱에는 단일 지역 집계를 포함한다. 신고 대화는 합성이며 신고자 핀과 두 구급차·소방차의 순찰 경로는 이 법정동 안에서 생성한다. [AI Hub 원본 명세](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&dataSetSn=71768&topMenu=100)
- 구 경계는 통계청 SGIS 2020 자료를 WGS84로 변환·단순화했다. 성수동1가의 법정동 경계는 JUSO 2015 자료에서 코드 `11200114`에 해당하는 경계를 추출했다. 화면과 서버가 같은 법정동 파일을 사용한다. [StatGarten 경계 원본](https://github.com/statgarten/maps/tree/d5f8ea3208f19a73a01f865847d20cc195ae91ba) · [southkorea/seoul-maps과 라이선스 표기](https://github.com/southkorea/seoul-maps#copyright-and-license) · [JUSO 경계 원본](https://github.com/southkorea/seoul-maps/tree/dbb1dcde2e3c9937684c5a44ab9301888ad63ee6/juso/2015) · [배포 라이선스](../../licenses/)
- 병원 이름·주소·위경도와 기관 코드는 국립중앙의료원 병·의원 찾기 서비스에서 받는다. 서버에 `NMC_HOSPITAL_API_KEY`를 설정하면 실시간 모드에서 조회할 수 있다. 저장 자료에는 2026년 11월 19일 조회한 성동구 소재 일반병원·종합병원 5곳이 있다. 훈련은 해당 세션의 후보를 최대 세 곳으로 고정하며, 관제 화면의 별도 기관 지도에서는 자치구별 전체 조회가 가능하다. [국립중앙의료원 API](https://www.data.go.kr/data/15000736/openapi.do)
- `drill-v1`의 가상 수용 여부와 처치 조건은 훈련을 위해 부여한 값이다. 현재 구급 시나리오에서는 출동유형과 발화 근거가 일치하면 가상 구급차가 출동하고, 심정지 의심 판정과 훈련 수용 조건까지 일치하면 지도에 병원 연결선이 표시된다. 화재 시나리오는 소방차 출동으로 끝난다.
- 시작 화면 사진은 이 데모를 위해 생성한 이미지다. 도로 배경은 OpenStreetMap을 사용한다. 공개 지도 타일을 이용할 수 없을 때도 법정동 경계와 위치 마커를 확인할 수 있다. [OpenStreetMap 저작권](https://www.openstreetmap.org/copyright)

## 재검증

저장소 루트에서 `python -B -m unittest discover`로 회귀 테스트를 실행한다. `frontend/tests/drill.browser.js`는 가상 응답으로 조회 실패·재시도와 지도 상호작용을 검증한다. `frontend/tests/drill-live.browser.js`는 8020 서버와 실제 팀 모델·병원 API를 연결한 뒤 Playwright 페이지에서 실행하는 통합 테스트다. 같은 턴을 중복 제출했을 때 한 번만 전진하는지, 세부 유형이 보류된 동안 병원 연결이 대기하는지도 확인한다.

실시간 Zoro 모드에서 **구급**과 **화재**를 각각 새 훈련으로 시작하고 신고자 핀을 세 번 누른다. 두 시나리오 모두 1턴은 출동 대기 상태여야 한다. 구급은 2턴에 가상 구급차가 출동하고 3턴에 심정지 의심과 훈련 수용 조건이 일치한 병원에 연결한다. 화재는 2턴에 가상 소방차가 출동하고 3턴까지 병원 연결이 생기지 않는다.

2026년 11월 19일 데스크톱 브라우저에서 H1b 실시간 응답으로 두 흐름을 재생했다. 1440×900 구급 훈련은 2턴에 차량 `A2`, 3턴에 `cardiac_arrest`와 기관 `A1100013` 모의 연결을 표시했다. 1280×800 화재 훈련은 2턴에 `F1`이 출동했고 3턴에도 병원 연결은 대기했다. 두 훈련의 기관 좌표를 성동구 공공 API 응답과 대조했다. 1920×1080에서는 저장된 좌표로 구급 시뮬레이션을 재생했다. 이 검사는 모델·하네스·화면의 연결을 확인하는 합성 시나리오 두 건이다.
