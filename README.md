# SENTINEL119

**응급환자 이송 지연 문제의 해결을 위한 Zoro-1 기반 119 지능형 관제 에이전트**

SENTINEL119는 **응급환자 이송 지연 문제**를 줄이기 위한 프로젝트입니다. 119 신고 단계부터 환자에 관한 정보와 구급 자원, 병원의 수용 정보를 연결해 관제 담당자의 의사결정을 돕는 것을 목표로 합니다.

신고 대화를 한 턴씩 읽고, 지금 판단할 수 있는 내용과 추가 정보가 필요한 내용을 구분합니다. 구급대원의 구성·장비·도착 예상시간과 병원 후보를 함께 비교해 **어떤 자원을 보내고 어느 병원을 검토할지** 근거와 함께 제시하는 관제 보조 에이전트를 개발합니다.

## 풀고자 하는 문제

응급환자를 적절한 병원으로 연결하려면 환자의 상태뿐 아니라 이동 시간, 필요한 처치를 제공할 수 있는 자원, 병원의 수용 정보까지 함께 살펴야 합니다.

SENTINEL119는 이 정보들을 신고 접수 단계부터 연결하는 데 집중합니다. 대화가 끝날 때까지 모든 판단을 미루지 않고, 근거가 충분한 판단부터 진행하며 새 정보와 정정에 따라 추천을 갱신하는 방식을 탐구합니다. 이를 통해 이송 후보를 찾고 비교하는 과정의 지연을 줄이는 것이 목표입니다.

## 목표 기능

| 기능 | 역할 |
| --- | --- |
| 턴 단위 신고 이해 | 현재까지 들어온 대화를 바탕으로 상황을 갱신하고 정정 내용을 반영 |
| 출동유형·긴급도 판단 | 화재·구조·구급·기타와 상·중·하 긴급도를 분류 |
| 태스크별 판단 확정·대기 | 근거가 충분하면 판단을 확정하고, 부족하면 다음 정보를 기다림 |
| 구급 자원 추천 | 지침, 대원 구성, 장비, 가용 여부와 도착 예상시간을 고려해 가상 차량 후보 비교 |
| 병원 후보 추천 | 병상·질환별 수용 정보와 이동 시간을 바탕으로 후보 비교 |
| 근거와 상태 표시 | 판단에 사용한 발화·지침, 정보의 갱신 시각, 누락·오류 상태를 함께 제시 |

예를 들어 출동유형과 차량 추천을 먼저 확정하고 병원 선택은 추가 정보를 기다릴 수 있습니다. 병원 정보가 없거나 조회에 실패한 경우에도 그 상태를 드러내고 보류·재시도·저장 정보 활용으로 이어지는 흐름을 설계합니다.

## 모델과 데이터

- **의사결정 모델:** [Trillion Labs Zoro-1](https://huggingface.co/trillionlabs/Zoro-1)을 활용한 선택형 판단.
- **신고 대화 데이터:** [AI Hub 119 지능형 신고접수 음성 인식 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&dataSetSn=71768&topMenu=100)의 전사와 분류 라벨.
- **지침 근거:** 소방청 「119구급대원 현장응급처치 표준지침(2023년 개정본)」의 필요한 항목을 원문과 대조해 활용.
- **병원 정보:** [국립중앙의료원 응급의료기관 정보 조회 API](https://www.data.go.kr/data/15000563/openapi.do) 연동을 통해 병원 후보의 수용 정보를 비교.

## 로컬 관제 화면 실행

서울 합성 신고 3종을 재생하는 FastAPI·React 데모입니다. 기본 모드는 시뮬레이션입니다. 다음 명령을 저장소 루트에서 PowerShell로 실행한 뒤 브라우저에서 [로컬 관제 화면](http://127.0.0.1:8000/)을 엽니다. 검증 환경은 Python 3.13과 Node.js 26입니다.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
npm.cmd ci --prefix frontend
npm.cmd run build --prefix frontend
.\.venv\Scripts\python.exe -m uvicorn backend.api:app --host 127.0.0.1 --port 8000
```

**다음 발화**로 신고를 진행합니다. 성수역 시나리오의 2턴에서 차량 확정과 병원 보류가 함께 나타납니다. 강변 시나리오는 ETA 역전, 병원 정보 지연 시나리오는 조회 실패와 재시도를 보여줍니다. 직접 입력은 검토된 근거가 생길 때까지 보류됩니다. 초기화하면 새 신고가 시작됩니다.

실시간 모드는 백엔드 실행 전에 `ZORO_BASE_URL`을 지정합니다. 서버가 여러 모델을 제공하면 `ZORO_MODEL_ID`도 지정합니다. 역할별 서버는 `ZORO_INTAKE_BASE_URL`, `ZORO_VEHICLE_BASE_URL`, `ZORO_HOSPITAL_BASE_URL`과 각각의 `_MODEL_ID`로 설정합니다. 서버가 응답해야 [모델 어댑터](backend/inference.py)가 모델 ID를 확인하고 typed decision을 요청합니다. 차량·병원 자료는 두 모드 모두 가상입니다.

모의 출동 클라이언트는 화면에 표시한 세션의 `expected_version`, 차량의 `expected_vehicle_id`, `decisions.vehicle.recommendation_revision`에서 읽은 `expected_vehicle_revision`을 `dispatch` 이벤트에 함께 보냅니다. 추천이 바뀌어 HTTP 409를 받으면 최신 세션을 표시하고 운영자의 새 클릭을 기다립니다. 모델 응답에 `model`이 있으면 요청한 ID와 일치해야 합니다. 이 필드를 생략하는 기존 서버에서는 요청에 선택한 ID만 확인할 수 있습니다.

백엔드 단위·HTTP 테스트는 `.\.venv\Scripts\python.exe -B -m unittest discover -v`로 실행합니다. 프론트 개발 서버는 `npm.cmd run dev --prefix frontend`로 시작하며 `/api`를 로컬 FastAPI에 전달합니다. 서버가 켜진 상태에서 Playwright MCP로 [데스크톱 관제 흐름](frontend/tests/console.browser.js)과 [차량 재확인·병원 정보 복구](frontend/tests/review.browser.js)를 검증할 수 있습니다.

차량 만료와 병원 조회 연속 실패를 재현하려면 프론트 빌드 후 `.\.venv\Scripts\python.exe -m uvicorn tests.freshness_console:create_test_app --factory --host 127.0.0.1 --port 8024`로 테스트 서버를 실행합니다. [데스크톱 장애 회귀 검사](frontend/tests/freshness.browser.js)는 출동 거절 직후의 버튼 상태와 병원의 마지막 관측 시각을 확인합니다.

## 공용 판단 하네스

Python 표준 라이브러리로 구현한 [판단 검증 모듈](decision_guard.py)은 합성 신고 시뮬레이션에서 Zoro 형식의 선택을 검증합니다. 모델 요청 전에 서버가 검토한 판단 근거, 위치 `(값, 근거 턴)`, 차량·병원 후보의 가용 상태를 `advance`에 전달하고 `begin`으로 요청 티켓을 발급합니다. 응답을 받으면 티켓과 함께 `apply`에 전달하고, 현재 판단은 `active`로 조회합니다. 시뮬레이션의 기본 응답 제한은 5초이며 `apply`에서 검사합니다. 네트워크 무응답은 호출부에서 처리합니다.

ETA나 역량만 달라져도 추천을 다시 검토하려면 `candidate_versions`를 사용합니다. 서버에서 순서를 보장한 차량·병원 피드별로 `CandidateVersionTracker`를 하나씩 만들고, 같은 `Guard`를 사용하는 동안 재사용합니다. 매 갱신에서 가용 상태와 ETA를 비롯해 후보 비교에 쓰는 전체 JSON 객체를 `observe`에 전달합니다.

```python
from decision_guard import CandidateVersionTracker, Guard

guard = Guard("demo")
vehicle_versions = CandidateVersionTracker()
hospital_versions = CandidateVersionTracker()

def refresh(turn_id, supports, location, vehicles, hospitals):
    guard.advance(
        turn_id, supports=supports, location=location,
        vehicles={key: candidate["availability"] for key, candidate in vehicles.items()},
        hospitals={key: candidate["availability"] for key, candidate in hospitals.items()},
        candidate_versions={
            "vehicle": vehicle_versions.observe(vehicles),
            "hospital": hospital_versions.observe(hospitals),
        },
    )
```

데이터 공급자가 후보 버전을 제공하면 그 값도 사용할 수 있습니다. 한 번호는 같은 후보 데이터에만 사용합니다. 버전을 제공하기 시작한 뒤 번호를 생략하거나 낮추면 해당 추천을 보류합니다. 같은 번호에 후보 ID나 가용 상태가 달라지면 더 높은 버전이 들어올 때까지 해당 추천을 보류합니다.

실제 팀 체크포인트의 판단 품질과 이송 지연 감소 효과는 별도 검증이 필요합니다.

첫 구현은 서울의 구급 시나리오와 가상 차량 환경을 대상으로 하는 텍스트 턴 재생 데모입니다. 4종 출동유형 분류를 유지하면서 구급 자원·병원 후보 추천에 집중하고, 이후 음성 입력으로 확장할 계획입니다. 데모는 관제 의사결정을 지원하는 시뮬레이션이며 실제 출동·병원 수용 확정을 수행하지 않습니다.

## 문서

데모 첫 화면은 출동 관제(실제 통화 재생, 지도 배차·병원 이송), 모델 비교(Qwen·SENTINEL_ZORO·base), 직접 입력(지도에서 위치 지정 후 대화) 세 탭입니다. 이전 관제 콘솔은 `/#console`, 자동 모의훈련은 `/#drill` 주소로 열 수 있습니다.

서울 지도에서 신고자 핀을 누르며 구급·화재 출동을 재현하는 방법은 [자동 모의훈련 실행 안내](docs/handoffs/seoul-drill.md)에 있습니다.

상세 설계와 개발 문서는 [문서 안내](docs/README.md)에서 확인할 수 있습니다.
