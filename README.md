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

## 평가 결과

최종 모델은 계층형 결정 체크포인트 **H4**입니다(Zoro-1 + LoRA, 출동유형 → 구급·구조 세부유형 → 긴급도 + 구급 주증상). AI Hub 공식 Validation에서 나눈 dev 300 · confirm 1,000 · test 1,000통화 중 test는 모델을 고른 뒤 한 번만 평가했습니다.

**test 1,000통화** (통화 전체를 본 판단)

| 모델 | 출동유형 F1 | 긴급도 F1 | 긴급도 하향 오판 | 구급 세부유형 | 구조 세부유형 | 주증상 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Zoro-1 base | 0.588 | 0.318 | 8.0% | 27.4% | 9.4% | 37.4% |
| **SENTINEL_ZORO (H4)** | **0.720** | **0.544** | 23.3% | **59.9%** | **50.0%** | **45.2%** |

세부유형은 출동유형까지 맞힌 경우만 정답으로 셉니다. base의 낮은 하향 오판은 대부분을 긴급도 상으로 답하는 편향 때문입니다.

**턴 단위 확정** (dev 83통화의 모든 턴 앞부분, 같은 임계값을 두 모델에 적용)

| 태스크 | SENTINEL_ZORO 정밀도 (확정률) | base 정밀도 (확정률) |
| --- | ---: | ---: |
| 출동유형 | 98.6% (87%) | 72.3% (100%) |
| 구급 세부유형 | 80.0% (49%) | 17.9% (75%) |
| 긴급도 | 88.9% (22%) | 33.3% (72%) |
| 주증상 | 76.5% (28%) | 40.0% (96%) |

임계값은 맞는 확정 +1, 틀린 확정 −2, 미확정 0의 합이 최대가 되도록 dev에서 골랐습니다. 한 실행·한 seed의 결과이며 학습 설정과 confirm 비교는 [학습 안내](docs/training.md) 10절에 있습니다. 이송 지연 감소 효과는 측정하지 않았습니다.

## 데모

| 탭 | 동작 |
| --- | --- |
| 출동 관제 | 신고 통화를 한 턴씩 재생합니다. SENTINEL_ZORO가 출동유형을 확정하면 지식베이스 편성 규칙으로 구급차·펌프차·구조대를 배차하고, 차량이 실제 도로 경로(TMAP)를 따라 이동합니다. 구급이면 국립중앙의료원 실시간 병상·수술·장비·메시지와 주증상별 이송 원칙으로 병원 후보를 만들고 Zoro가 이송 병원을 고릅니다. |
| 모델 비교 | 같은 통화를 Qwen3.5-4B(JSON 생성), SENTINEL_ZORO, Zoro-1 base에 동시에 넣고 턴별 판단·확정 시점·응답 시간을 비교합니다. |
| 직접 입력 | 서울 지도에서 신고 위치를 찍고 가상의 수보요원과 대화하면 출동 관제와 같은 흐름으로 동작합니다. |

차량은 서울 소방서 위치(OpenStreetMap)에 배치한 시연용 가상 차량이며, 출동·이송은 실제로 요청하지 않는 시뮬레이션입니다. 이전 관제 콘솔은 `/#console`, 자동 모의훈련은 `/#drill`로 열 수 있습니다.

### 실행

GPU 서버에서 [판단 서버](docs/training.md#11-판단-서버와-데모-연동)를 띄우고(`python tools/decision_server.py --checkpoint runs/zoro-hier-h4 --llm Qwen/Qwen3.5-4B --port 8119`) 데모 PC에서 SSH 터널로 `127.0.0.1:8119`에 연결한 뒤, 저장소 루트에서 실행합니다. 검증 환경은 Python 3.13, Node.js 26입니다.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
npm.cmd ci --prefix frontend
npm.cmd run build --prefix frontend
$env:SENTINEL_ZORO_URL = "http://127.0.0.1:8119"
.\.venv\Scripts\python.exe -m uvicorn backend.drill_console:create_console --factory --host 127.0.0.1 --port 8020
```

도로 경로와 병원 정보는 저장소 루트의 `.env.local`(Git 제외)에 `TMAP_APP_KEY`, `NMC_SERVICE_KEY`를 넣으면 사용합니다. TMAP 호출이 실패하면 경로는 OSRM, 그다음 직선으로 내려갑니다. 병원 실시간 정보 조회가 실패한 지역은 응급의료기관 목록 정보로 대신하고, 키가 없거나 조회 전체가 실패하면 병원 추천을 보류합니다. 화면과 로그에 각 정보의 출처를 표시합니다. 재생할 통화 파일(`{id, title, address, turns}`)은 `SENTINEL_SCENARIO_DIRS`로 지정합니다. AI Hub 실제 통화는 로컬 시연에만 쓰고 저장소에 넣지 않습니다.

백엔드 테스트는 `.\.venv\Scripts\python.exe -B -m unittest discover -v`, 프론트 타입 검사는 `npx tsc --noEmit -p frontend`로 실행합니다. 학습 코드 테스트는 `pip install -e ".[train]"` 환경에서 `pytest`로 실행합니다. 데모 전용 환경의 `unittest discover`에서는 이 테스트 모듈들이 import 오류로 표시됩니다.

## 에이전트 실행 로그

[`tools/scenario_log.py`](tools/scenario_log.py)가 데모 백엔드를 그대로 호출해 시나리오 하나를 처음부터 끝까지 실행하고, 턴별 Zoro 입력·확률·WAIT/COMMIT, 배차·병원 도구 호출, 오류와 복구를 JSONL로 남깁니다. 입력은 공유용으로 작성한 합성 통화이며 모델·도로·병원 정보는 실제 호출입니다.

| 로그 | 시나리오 | 결과 |
| --- | --- | --- |
| [scenario-01](docs/submission/logs/scenario-01.jsonl) | 성북구 장위동 심정지 의심 | 2턴 구급 확정 → 구급차 배차·병원 추천, 6턴 심정지·긴급도 상·주증상 확정 → 펌뷸런스 추가 배차 |
| [scenario-02](docs/submission/logs/scenario-02.jsonl) | 영등포구 당산동3가 빌라 화재 | 2턴 화재·긴급도 상 확정 → 펌프차 2대 + 구급차 배차 |
| [scenario-03](docs/submission/logs/scenario-03.jsonl) | 강남구 삼성동 흉통 + 장애 주입 | 병원 API timeout(주입) → 병원 보류 후 다음 턴 재조회 성공, 4턴 판단 서버 연결 실패(주입) → 이전 확정 유지 후 다음 턴 복구, 8턴 긴급도 상 확정. 주증상(흉통)은 1순위였지만 임계값 미만으로 끝까지 대기 |

기록 규약과 재실행 방법은 [실행 로그 안내](docs/submission/logs/README.md)에 있습니다.

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

## 현재 상태와 개발 범위

119 데이터 준비, Zoro-1 지도학습·계층형 결정 학습, 독립 평가, 판단 서버, 세 탭 데모와 제출용 실행 로그를 구현했습니다. 실제 관제 환경 연동과 이송 지연 감소 효과는 검증하지 않았습니다. 데모는 관제 의사결정을 돕는 시뮬레이션이며 실제 출동·병원 수용 확정을 수행하지 않습니다. 설치·학습·평가 절차는 [학습 안내](docs/training.md)를 참고하세요.

## 문서

서울 지도에서 신고자 핀을 누르며 구급·화재 출동을 재현하는 방법은 [자동 모의훈련 실행 안내](docs/handoffs/seoul-drill.md)에 있습니다. 상세 설계와 개발 문서는 [문서 안내](docs/README.md)에서 확인할 수 있습니다.
