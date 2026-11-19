# TMAP 경로 공급자 인수인계

기준 `dev`(`9cfc3f5`)에는 명세와 문서가 있고 로컬 관제 콘솔은 PR #1 병합 대기 중이다. 이 브랜치의 공급자는 Python 표준 라이브러리만으로 두 구간의 후보별 ETA와 경로를 조회한다. 관제 연동은 PR #1 병합 후 진행한다. [병원·ETA 명세](../spec.md#병원eta)

## PowerShell에서 오프라인 실행

Python 3.11 이상, Windows PowerShell 5.1에서 실행한다. 저장소 루트에서 합성 좌표와 후보를 넣는다. 5.1은 기본 `$OutputEncoding`(ASCII)으로 네이티브 명령에 파이프하므로 한글 ID가 `?`로 바뀐다. 그래서 파이프라인 전에 BOM 없는 UTF-8을 지정하고, `finally`에서 원래 값을 복원한다.

```powershell
$previousEncoding = $OutputEncoding
try {
    $OutputEncoding = [System.Text.UTF8Encoding]::new($false)
    $json = @{
        leg = 'vehicle_to_incident'
        incident = @(127.1, 37.5)
        candidates = @(
            @{ id = '동부'; coordinates = @(127.2, 37.6); simulated_eta_min = 3 }
            @{ id = '서부'; coordinates = @(127.3, 37.7) }
        )
    } | ConvertTo-Json -Depth 5 -Compress
    $raw = $json | python -m backend.tmap_routes
    ($raw | ConvertFrom-Json).routes | Select-Object id, source, duration_s, reason
} finally {
    $OutputEncoding = $previousEncoding
}
```

`ConvertFrom-Json`으로 결과를 읽으면 `id` 열에 입력한 `동부`와 `서부`가 그대로 표시된다.

`동부`는 `simulation`으로 `duration_s`가 `180`, `서부`는 `unknown`으로 `null`이다. 두 후보의 `reason`은 `offline`이다.

## 호출 계약

- `backend.tmap_routes`에서 `evaluate_routes(payload, fetch_route=None)`을 가져온다. `fetch_route`를 생략하면 오프라인으로 처리한다. 실시간 조회가 필요한 서버 호출부는 `backend.tmap_http`의 `make_tmap_fetcher(app_key)`가 반환한 함수를 전달한다.
- 실시간 CLI는 `python -m backend.tmap_routes --live`에 같은 JSON을 전달한다. 서버의 `TMAP_APP_KEY`가 비어 있거나 키 형식이 유효하지 않으면 오프라인으로 처리해 외부 요청을 보내지 않는다. 키는 서버에서만 보관하며 브라우저로 전달하지 않는다. 실제 TMAP 호출 검증은 별도 작업이다.
- 입력은 `leg`, `incident`, `candidates`를 담는다. `vehicle_to_incident`는 차량 후보 → 현장, `incident_to_hospital`은 현장 → 병원 후보 방향이다. `incident`와 후보의 `coordinates`는 WGS84 `[경도, 위도]`다.
- 한 요청에 최대 16개 후보를 받는다. 후보마다 고유한 문자열 `id`와 `coordinates`, 선택적으로 0 이상의 정수 `simulated_eta_min`을 제공한다. 후보별 조회에 실패해도 결과에 입력 순서와 ID를 보존한다.
- CLI 표준 입력은 64 KiB, TMAP 응답 본문은 2 MiB까지 받는다. 두 JSON의 중첩 깊이 한도는 64다. 소켓 작업별 제한 시간은 3초다. 후보를 하나씩 순차 조회하며 배치 전체 제한 시간은 따로 두지 않는다. 후보 16개가 각각 3초 제한 시간에 한 번씩 걸리면 약 48초가 걸린다. 이는 소켓 제한 시간의 합일 뿐 전체 상한이 아니다.

## 결과와 실패 처리

결과는 `{leg, routes}`이며 각 행에 `id`, `source`, `distance_m`, `duration_s`, `path_segments`, `observed_at`, `reason`이 있다.

| `source` | 결과 |
| --- | --- |
| `tmap` | TMAP 응답 파싱에 성공하면 거리(m)와 시간(초)을 반환한다. `0`도 유효하다. `observed_at`은 성공 시점의 UTC 시각이며 `reason`은 `null`이다. `path_segments`는 응답에 있는 유효한 선분의 좌표 배열이다. |
| `simulation` | 조회를 건너뛰거나 실패하면 입력에 있는 `simulated_eta_min`을 초로 환산한다. `reason`에 건너뛰거나 실패한 이유를 보존한다. |
| `unknown` | 조회 결과와 가상 ETA가 모두 없으면 `duration_s`를 `null`로 반환한다. `reason`은 그대로 남긴다. |

TMAP 응답에 유효한 선분이 없거나 `source`가 `simulation` 또는 `unknown`이면 `path_segments`는 `[]`이다. 가상·미상 결과의 `distance_m`과 `observed_at`은 `null`이다.

| `reason` | 의미 |
| --- | --- |
| `offline` | 기본 모드이거나 서버 키가 비어 있거나 형식이 유효하지 않음 |
| `invalid_incident_coordinates` | 현장 좌표 누락 또는 오류 |
| `invalid_candidate_coordinates` | 해당 후보 좌표 누락 또는 오류 |
| `timeout` | 조회 시간 초과 |
| `http_error` | HTTP 오류 상태 |
| `network_error` | 연결 또는 통신 오류 |
| `malformed_response` | 응답 구조·값·인코딩 또는 크기 오류 |
| `missing_totals` | 총거리 또는 총시간 누락 |

## PR #1 병합 후 관제 연동

- `backend/sessions.py`의 후보 상태 갱신에서 확인된 현장 좌표와 해당 구간의 후보를 조회에 사용한다. 별도 작업자 스레드에서 동기 조회를 실행하고 폴링은 현재 상태만 반환한다. 응답 시점에 현장과 후보 리비전을 다시 확인해 오래된 응답을 버리고, 반영된 ETA와 출처의 변경은 후보 리비전에 포함한다.
- `frontend/src/panels.tsx`의 기존 `가상 ETA` 표시는 `simulation`에 적용하고 `tmap`은 출처를 `TMAP`으로 표시한다. `unknown`의 ETA는 `null`로 유지한다. 조회 결과는 판단 근거로 제공하며 차량이나 병원을 자동 선택하지 않는다.
- 관제 연동 시에는 도착한 결과를 후보별로 먼저 표시하고, 전체 조회 예산을 별도로 적용해야 한다. 예산 안에 확인하지 못한 후보는 `unknown`으로 표시한다. 이는 앞으로 할 작업이며 현재 구현되어 있지 않다.

참고 자료로 공식 API와 기존 거리 계산을 확인할 수 있다. [TMAP 자동차 경로안내](https://tmap-skopenapi.readme.io/reference/자동차-경로안내) · [2025-ermct-project의 `distance_logic.py`](https://github.com/Jaeuk-Han/2025-ermct-project/blob/8a7f7689a0edf85892d5622a56655e595caacb10/app/distance_logic.py)

회귀 검사는 저장소 루트에서 `python -B -m unittest discover -v`로 실행한다.
