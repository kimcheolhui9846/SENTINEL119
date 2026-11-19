# 제출용 에이전트 실행 로그

이 디렉터리는 **팀·심사위원에게 공유할 실제 실행 기록**을 보관한다. 일반 실험 원본 로그는 Git에서 제외된 `/logs/`, `/runs/`, `/outputs/`에 둔다.

공식 요구사항은 **서로 다른 시나리오 3가지의 에이전트 실행 로그**다. JSONL 형식·아래 필드·파일명은 프로젝트 내부 규약이며 공식 지정 형식이 아니다. [평가·제출 체크리스트](../../track1-submission.md)를 참고한다.

## 제출 로그

2026-11-20에 [`tools/scenario_log.py`](../../../tools/scenario_log.py)로 실행한 기록이다. 입력 통화는 [scenarios](../scenarios/)의 공유용 합성 통화이고, 모델·도로 경로·병원 정보는 실제 호출이다. 각 로그의 `run_start`에 실행 commit, 모드, 임계값, 주입한 장애가 있다.

| 파일 | 시나리오 | 확인할 동작 |
| --- | --- | --- |
| `scenario-01.jsonl` | 성북구 장위동 심정지 의심 | 태스크별 확정 시점이 다름(2턴 출동유형, 6턴 세부유형·긴급도·주증상), 확정 전 1순위 주증상으로 병원 미리 추천, 심정지 확정 후 펌뷸런스 추가 배차 |
| `scenario-02.jsonl` | 영등포구 당산동3가 빌라 화재 | 구급 외 유형: 화재 확정 시 펌프차 2대 + 구급차 동시 배차, 병원 단계 없음 |
| `scenario-03.jsonl` | 강남구 삼성동 흉통 + 장애 주입 | 병원 API timeout(주입) → 보류·재조회, 판단 서버 연결 실패(주입) → 확정 유지·다음 턴 복구. 주증상은 임계값 미만으로 끝까지 WAIT (기대 동작과 다름, 그대로 기록) |

배차 차량과 경로 시간은 실행 시점의 TMAP 경로에 따라 달라질 수 있다.

### 다시 실행

판단 서버(`SENTINEL_ZORO_URL`)와 `.env.local`의 `TMAP_APP_KEY`·`NMC_SERVICE_KEY`가 있는 상태에서 저장소 루트에서 실행한다.

```bash
SENTINEL_ZORO_URL=http://127.0.0.1:8119 python tools/scenario_log.py   docs/submission/scenarios/scenario-01.json docs/submission/logs/scenario-01.jsonl
```

### 이벤트

| `event` | 내용 |
| --- | --- |
| `run_start` | `run_id`, `git_commit`, 모드(`live_model`, `synthetic_script`, `synthetic_world`, `live_api`), 임계값, 신고 지점, 주입 장애 |
| `fleet`, `model` | 가상 차량 수, 모델 ID·revision·체크포인트 파일 sha256 |
| `turn` | 공개된 발화 한 턴 (모델 입력은 1턴부터 현재 턴까지) |
| `model_call` | `request_id`, 입력 턴 수, SENTINEL_ZORO 문항별 확률, base 1순위, 지연, 성공·오류, 주입 여부 |
| `policy` | 태스크별 `COMMIT`/`WAIT`/`LOCKED`/`N/A`, 조회 실패 시 `HOLD` |
| `tool_call` | `dispatch_plan`(지식베이스 편성), `transfer_hospital`(이송 원칙, 후보별 병상·처치·메시지, 데이터 출처) |
| `action` | `dispatch_unit`, `recommend_hospital` (모두 `simulated: true`) |
| `run_end` | 최종 확정, 배차 목록, 기대 동작 대비 결과, 전체 실행 시간 |

## 기록 규약

한 줄에 JSON 이벤트 하나를 기록하고 `run_id`로 실행 전체를 연결한다. 다음 정보가 실행 전체에서 확인돼야 한다.

| 정보 | 최소 내용 |
| --- | --- |
| 실행 식별 | `run_id`, `scenario_id`, 실행 시각, Git commit SHA, 실행 설정 |
| 모델 | `model_id`, `model_revision`, Base/SFT/PPO 변형, 실제 사용한 adapter/head 식별 |
| 현재 상태 | `turn_id`, `state_version`, 현재까지 공개된 입력, 사용한 지침·후보 snapshot |
| 모델 호출 | `request_id`, 호출 입력·출력, 선택지별 확률, 지연, 성공·오류 상태 |
| 정책·행동 | 태스크별 WAIT/COMMIT, 잠정·채택 선택, 정책 출처, 모의 출동 등 실제 적용 이벤트 |
| 도구·외부 정보 | API/도구명, 성공·timeout·정보 없음, 원천/수집 시각, 오류 주입·fallback 여부 |
| 결과 | 최종 상태, 기대 동작 대비 관찰 결과, 실패·한계, 전체 실행 시간 |

1. 실제 실행기로 생성한 기록을 보존한다. 계획한 정답이나 임의의 모델 출력을 실행 결과처럼 채우지 않는다.
2. 공유용 합성 입력을 우선 사용한다. 선별·마스킹이 필요하면 방식과 누락 범위를 기록하며 실패 결과를 삭제해 성공으로 보이게 만들지 않는다.
3. 비밀키·인증 헤더·쿠키·실제 환자 식별정보·개인 경로가 포함된 디버그 덤프를 넣지 않는다.
4. 전체 미래 대화·최종 정답이 현재 모델 입력에 섞이지 않았는지 확인한다.
5. `live_model`, `recorded_replay`, `cached_api`, `synthetic_world` 등 사용 모드를 명시한다. 재생 결과를 새 실시간 추론으로 표시하지 않는다.
6. JSON 파싱, 이벤트 순서, 모델 요청과 결과 연결, 세 시나리오 구분을 확인한 뒤 Git에 추가한다.

이 경로의 JSONL은 Git 추적 대상이다. 전역 `*.log` 제외 규칙은 유지하며 확장자는 `.jsonl`로 통일한다. 공식 안내자료에 별도 제출 형식이 추가되면 그 형식에 맞춰 내보내고 이 문서를 갱신한다.
