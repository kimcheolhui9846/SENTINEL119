# Design

## Source of truth

- Status: Draft. 사용자 확정 범위는 [프로젝트 명세](docs/spec.md)에 있고, 이 문서의 시각·구현 선택은 기본안이다.
- Last refreshed: 2026-11-19.
- Primary product surfaces: 데스크톱 관제 데모 한 화면, 실행 비교 패널.
- Evidence reviewed: 사용자 답변, 로컬 데이터 표본, Zoro 카드, 기존 ERMCT README·API 소스. 이 저장소에는 기존 UI·스크린샷·브랜드 자산이 없다.

## Brand

- Personality: 침착하고 명료한 관제 보조 도구.
- Trust signals: 근거 발화·지침 쪽·정보 시각·모델 구분·가상 데이터 표시.
- Avoid: 과도한 사이렌·점멸, 검증되지 않은 정확도 홍보, 작은 글씨로 숨긴 오류.

## Product goals

- Goals: 응급환자 이송 지연 완화를 위한 관제 의사결정 보조를 목표로, 신고·구급 자원·병원 정보를 연결하고 무엇을 먼저 결정하며 무엇을 기다리는지 시연한다. 실제 지연 감소 효과는 아직 검증하지 않았다.
- Non-goals: 실출동·실예약, 병원 운영 대시보드 전체 복제, 로그인·권한 시스템.
- Success signals: 세 구급 시나리오 완주, 차량 추천과 병원 대기의 동시 표현, 정정·API 실패 처리.

## Personas and jobs

- Primary personas: 해커톤 심사자·시연자; 수보자의 업무 맥락을 설명하는 시뮬레이션.
- User jobs: 발화 진행, 추천 변화 관찰, 근거 확인, 차량·병원 상태 변경, 모델 비교.
- Key contexts: 발표 화면 공유·현장 노트북. 실제 운영 검증을 거친 제품으로 표현하지 않는다.

## Information architecture

- Primary navigation: 관제 데모 / 실행 비교. 최초 구현은 한 화면과 탭으로 충분하다.
- Core routes/screens: 시나리오 선택·턴 제어, 대화, 의사결정 카드, 차량·병원 후보, 근거·이력.
- Content hierarchy: 현재 발화와 최신 판단을 먼저 표시하고, 세부 근거는 펼쳐서 읽는다.

```text
SENTINEL119   시나리오 · 모델 · 데이터 상태 · 재생/정지/초기화
대화·텍스트 입력 | 유형·긴급도·차량·병원 판단 카드
                | 차량 A/B · 병원 후보 3개 · ETA/상태 시각
현재 턴·경과시간 | 근거 발화·지침 · 판단 변경 이력
```

## Design principles

- 태스크마다 WAIT/COMMIT을 표시한다. 병원 대기가 다른 카드의 완료를 가리지 않는다.
- 색·문구·아이콘을 함께 사용한다. 모델 확률과 채택 상태를 다른 정보로 표시한다.
- 가상 세계·실제 API·저장 응답·기록 재생을 화면 상단과 관련 카드에서 구분한다.
- Tradeoffs: 상세 지도보다 상태 변화의 이해와 안정적인 재생을 우선한다. 지도는 있으면 유용한 보조 영역이다.

## Visual language

- Color: 짙은 중성 배경, 밝은 본문; COMMIT은 청록, WAIT는 황색, 오류·높은 긴급도는 문구와 함께 붉은색. 두 의미를 색만으로 구분하지 않는다.
- Typography: 한국어 시스템 sans-serif, 본문 최소 16px, 수치는 tabular 스타일.
- Spacing/layout rhythm: 8px 간격 기준, 읽기 쉬운 카드와 패널 여백.
- Shape/radius/elevation: 단순한 경계선·작은 모서리 반경, 그림자는 최소화.
- Motion: 값 변경 시 짧은 강조만 사용하고 자동 스크롤은 사용자 읽기를 방해하지 않는다.
- Imagery/iconography: 텍스트 라벨을 동반하는 단순 아이콘. 사진·장식 이미지는 필수 아님.

## Components

- Existing components to reuse: 이 저장소에는 없음. ERMCT의 관련 후보 카드·API 타입은 호환성 확인 후 선별.
- New/changed components: 턴 플레이어, 대화 목록, 태스크 카드, 차량 카드, 병원 후보, 근거 패널, 이벤트 이력.
- Variants and states: 잠정 후보 / WAIT / COMMIT / 범위 밖; 실행 중 / 실패 / 최신 응답 대기.
- Token/component ownership: 프론트엔드 담당이 공통 토큰을 관리하고 상태 의미는 백엔드 계약을 따른다.

## Accessibility

- Target standard: WCAG 2.2 AA를 목표로 구현하되 인증·달성을 주장하지 않는다.
- Keyboard/focus behavior: 다음 발화·재생·초기화·근거 열기를 키보드로 사용하고 포커스를 표시한다.
- Contrast/readability: 본문 대비·상태 라벨·발표 화면의 가독성을 확인한다.
- Screen-reader semantics: 제목·버튼·목록 의미를 보존하고 상태 변경은 간결한 live region으로 알린다.
- Reduced motion and sensory considerations: 점멸 금지, reduced-motion 설정 존중, 소리 자동 재생 없음.

## Responsive behavior

- Supported breakpoints/devices: 노트북 1280×720 이상을 우선, 1440×900을 기본 확인 크기로 사용한다.
- Layout adaptations: 좁은 화면에서는 대화→판단→후보 순으로 한 열 배치한다.
- Touch/hover differences: 핵심 정보와 조작을 hover에만 두지 않는다.

## Interaction states

- Loading: 어떤 태스크가 계산 중인지 표시하고 이전 값에는 이전 턴 표식을 남긴다.
- Empty: 시나리오 선택 또는 첫 발화 입력을 안내한다. 근거 없는 초기 추천을 채우지 않는다.
- Error: 추론 실패·API timeout을 따로 표시하고 재시도·저장 응답 전환을 제공한다.
- Success: 최신 state_version에 대한 결과만 활성 카드에 반영한다.
- Disabled: 같은 모의 출동 이벤트의 중복 적용을 막고 이유를 표시한다.
- Offline/slow network: 저장 정보의 시각과 모드를 표시한다. 최신 정보인 것처럼 보이지 않게 한다.

## Content voice

- Tone: 간결한 한국어, 현재 근거와 미확인 정보를 구체적으로 표현한다.
- Terminology: 판단 확정(COMMIT), 추가 정보 대기(WAIT), 잠정 후보, 모의 출동, 병원 후보, 정보 없음.
- Microcopy rules: `수술 가능 확정` 대신 실제 API 필드·수집 시각과 `후보`를 표현한다. 설명은 근거에 연결하고 임의의 확률을 만들지 않는다.

## Implementation constraints

- Framework/styling system: React/Vite/TypeScript 기본안, 추가 디자인 시스템 구축은 피한다.
- Design-token constraints: 공통 CSS 변수로 색·간격·타이포그래피를 모은다.
- Performance constraints: 지연 목표와 측정 방법은 명세 §7을 따른다. 렌더링이 모델 응답을 기다리며 전체 정지하지 않도록 한다.
- Compatibility constraints: 텍스트 재생 우선, 로그인 없이 단일 시연 세션, 모델/외부 상태는 백엔드 관리.
- Test/screenshot expectations: 대표 시나리오·WAIT/COMMIT 공존·정정·API 실패 화면을 실제 UI로 확인한다. 현재 UI는 구현되지 않았다.

## Open questions

- [ ] GPU 접속·NMC/Tmap 실제 사용 가능 여부 / 환경 담당 / 실시간 연결 또는 저장 모드 결정.
- [ ] 발표 장비의 화면 크기 / 시연 담당 / 최종 레이아웃 확인.
- [ ] 재사용 컴포넌트의 실제 의존성 / 프론트엔드 담당 / 복사 범위 결정.
