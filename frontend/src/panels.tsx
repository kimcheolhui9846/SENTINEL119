import type { Candidate, Decision, Session, Task } from './api';

const tasks: Task[] = ['dispatch', 'urgency', 'vehicle', 'hospital'];
const taskInfo: Record<Task, { name: string; role: string }> = {
  dispatch: { name: '출동유형', role: '수보요원' },
  urgency: { name: '긴급도', role: '수보요원' },
  vehicle: { name: '차량 추천', role: '배차요원' },
  hospital: { name: '병원 후보', role: '병원연락' },
};
const statusLabels: Record<string, string> = {
  WAIT: '보류', COMMIT: '확정', OUT_OF_SCOPE: '범위 밖', APPLIED: '모의 출동 적용',
  ERROR: '실행 오류', STALE: '이전 응답 폐기',
};
const availabilityLabels: Record<Candidate['availability'], string> = {
  available: '가용', unavailable: '불가', unknown: '정보 없음',
};
const sourceLabels: Record<Session['external_status'], string> = {
  ok: '정보 확인', unknown: '정보 없음', timeout: '조회 시간 초과', cached: '저장 자료',
};
const sourceTime = new Intl.DateTimeFormat('ko-KR', {
  timeZone: 'Asia/Seoul', month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit',
  hourCycle: 'h23',
});

export function candidateName(candidates: Candidate[], id: string): string {
  return candidates.find((candidate) => candidate.id === id)?.name ?? id;
}

function choiceName(session: Session, task: string, choice: string): string {
  if (task === 'vehicle') return candidateName(session.vehicles, choice);
  if (task === 'hospital') return candidateName(session.hospitals, choice);
  return choice;
}

function statusClass(status: string): string {
  return status.toLowerCase().replaceAll('_', '-');
}

function DecisionCard({ task, decision, session }: { task: Task; decision: Decision; session: Session }) {
  const { name, role } = taskInfo[task];
  const { status, choice, evidence_turn_ids: evidence, probabilities, execution } = decision;
  return (
    <article className="decision-card" data-status={status} aria-labelledby={`task-${task}`}>
      <div className="meta">{role}</div>
      <div className="card-heading">
        <h3 id={`task-${task}`} className="card-name">{name}</h3>
        <span className={`status ${statusClass(status)}`} title={status}>{statusLabels[status]}</span>
      </div>
      <p className="choice">
        {choice ? choiceName(session, task, choice) : status === 'OUT_OF_SCOPE' ? '해당 없음' : '판단 대기'}
      </p>
      {execution === 'running' && <p className="meta">판단 중</p>}
      {execution === 'error' && <span className="status error">실행 오류</span>}
      <p className="reason">{decision.reason}</p>
      <details className="evidence">
        <summary>근거 및 세부 정보</summary>
        {evidence.length ? (
          <ul>
            {evidence.map((id) => {
              const turn = session.turns.find((item) => item.id === id);
              return <li key={id}><strong>{id}번 발화</strong>{turn ? ` · ${turn.text}` : ''}</li>;
            })}
          </ul>
        ) : <p className="meta">공개된 근거가 없습니다.</p>}
        {decision.guideline_refs.length > 0 &&
          <ul>{decision.guideline_refs.map((reference) => <li key={reference}>{reference}</li>)}</ul>}
        {session.mode === 'live' && decision.model_id && <p className="meta">모델 {decision.model_id}</p>}
        {session.mode === 'live' && probabilities && (
          <ul aria-label="모델 선택 확률">
            {Object.entries(probabilities).map(([key, value]) => (
              <li key={key}>{choiceName(session, task, key)} · {Math.round(value * 100)}%</li>
            ))}
          </ul>
        )}
      </details>
    </article>
  );
}

export function DecisionsPanel({ session }: { session: Session }) {
  return (
    <section className="panel" aria-labelledby="decisions-heading">
      <header className="panel-header">
        <h2 id="decisions-heading" className="section-title">역할별 판단</h2>
        <span className="meta" aria-live="polite">{session.processing ? '판단 중' : '현재 판단'}</span>
      </header>
      <div className="decisions-grid">
        {tasks.map((task) => <DecisionCard key={task} task={task} decision={session.decisions[task]} session={session} />)}
      </div>
    </section>
  );
}

function CandidateTime({ value }: { value: string | null }) {
  if (!value) return null;
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? <span>{value}</span> : <time dateTime={value}>{sourceTime.format(date)}</time>;
}

function HospitalTable({ session }: { session: Session }) {
  const stale = session.external_status === 'timeout' || session.external_status === 'cached';
  return (
    <div className="hospital-table-scroll">
      <table className="hospital-table" aria-labelledby="candidates-hospital">
        <thead>
          <tr>
            <th scope="col">병원명</th>
            <th scope="col">수용 상태</th>
            <th scope="col">정보 시각</th>
            <th scope="col">ETA</th>
          </tr>
        </thead>
        <tbody>
          {session.hospitals.map((candidate) => {
            const observation = stale ? candidate.last_observation : null;
            const time = stale ? observation?.source_at : candidate.source_at;
            return (
              <tr key={candidate.id} data-selected={!stale && session.decisions.hospital.status === 'COMMIT' &&
                session.decisions.hospital.choice === candidate.id ? 'true' : undefined}>
                <th scope="row">
                  <strong className="candidate-name">{candidate.name}</strong>
                  <span className="meta">{candidate.detail}</span>
                </th>
                <td>
                  {stale ? (
                    <>
                      <strong className="hospital-stale">현재 추천 불가</strong>
                      <span className="meta">{observation
                        ? `마지막 확인 · ${availabilityLabels[observation.availability]}` : '이전 확인 정보 없음'}</span>
                    </>
                  ) : <span className="availability" data-status={candidate.availability}>{availabilityLabels[candidate.availability]}</span>}
                </td>
                <td>
                  {time ? <CandidateTime value={time} /> : <span className="meta">확인 시각 없음</span>}
                  {time && <span className="meta">{stale ? '이전 확인 시각' : candidate.source}</span>}
                </td>
                <td>{candidate.eta_min === null ? <span className="meta">정보 없음</span>
                  : <span className="candidate-eta">가상 ETA {candidate.eta_min}분</span>}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export function CandidatePanel({ kind, session }: { kind: 'vehicle' | 'hospital'; session: Session }) {
  const isVehicle = kind === 'vehicle';
  const candidates = isVehicle ? session.vehicles : session.hospitals;
  return (
    <section className="panel" aria-labelledby={`candidates-${kind}`}>
      <header className="panel-header">
        <h2 id={`candidates-${kind}`} className="section-title">{isVehicle ? '차량 후보' : '병원 후보 목록'}</h2>
        <span className={`meta${!isVehicle && session.external_status === 'timeout' ? ' error' : ''}`}>
          {isVehicle ? '가상 차량' : sourceLabels[session.external_status]}
        </span>
      </header>
      {!candidates.length ? <p className="empty-state">현재 표시할 후보가 없습니다.</p> : !isVehicle ? <HospitalTable session={session} /> : (
        <ul className="candidate-list">
          {candidates.map((candidate) => (
            <li key={candidate.id} className="candidate-row" data-availability={candidate.availability}
              data-status={session.decisions[kind].choice === candidate.id
                ? statusClass(session.decisions[kind].status) : undefined}>
              <div className="candidate-heading">
                <strong className="candidate-name">{candidate.name}</strong>
                <span className={`availability ${candidate.availability}`} data-status={candidate.availability}>
                  {session.applied_vehicle === candidate.id
                    ? '모의 출동 적용됨' : availabilityLabels[candidate.availability]}
                </span>
              </div>
              <div className="candidate-detail">
                <span>{candidate.detail}</span>
                {candidate.eta_min !== null && <span className="candidate-eta">가상 ETA {candidate.eta_min}분</span>}
              </div>
              <p className="meta">{candidate.source} · <CandidateTime value={candidate.source_at} /></p>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

export function HistoryPanel({ session }: { session: Session }) {
  return (
    <section className="panel" aria-labelledby="history-heading">
      <header className="panel-header">
        <h2 id="history-heading" className="section-title">판단 변경 이력</h2>
        <span className="meta">{session.history.length}건</span>
      </header>
      {session.history.length ? (
        <ol className="history-list" reversed>
          {session.history.slice().reverse().map((item) => {
            const task = taskInfo[item.task as Task];
            return (
              <li key={item.id}>
                <div className="history-item-heading">
                  <span className="meta">{item.turn_id}턴</span>
                  <strong>{task?.name ?? item.task}</strong>
                  <span className={`status ${statusClass(item.status)}`}>{statusLabels[item.status] ?? item.status}</span>
                  {item.choice && <span>{choiceName(session, item.task, item.choice)}</span>}
                </div>
                <p className="reason">{item.reason}</p>
              </li>
            );
          })}
        </ol>
      ) : <p className="empty-state">발화가 들어오면 판단 변경을 여기에 기록합니다.</p>}
    </section>
  );
}
