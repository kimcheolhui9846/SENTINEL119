import { useCallback, useEffect, useRef, useState } from 'react';
import { ApiError, createSession, getSession, listScenarios, postEvent } from './api';
import type { Action, Mode, Scenario, Session } from './api';

const POLL_DELAY_MS = 250;
const POLL_RETRY_MS = [500, 1000, 2000];
const PLAY_DELAY_MS = 1200;

type Recovery = 'scenarios' | 'open' | 'sync';
type Problem = { message: string; recovery?: Recovery };

function errorMessage(error: unknown): string {
  return error instanceof ApiError ? error.message : '서버에 연결할 수 없습니다. 잠시 뒤 다시 시도해 주세요.';
}

export function useConsole() {
  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [scenarioId, setScenarioId] = useState('');
  const [mode, setMode] = useState<Mode>('simulation');
  const [session, setSession] = useState<Session | null>(null);
  const [busy, setBusy] = useState(true);
  const [autoPlay, setAutoPlay] = useState(false);
  const [problem, setProblem] = useState<Problem | null>(null);
  const [pollCycle, setPollCycle] = useState(0);
  const current = useRef<Session | null>(null);
  const generation = useRef(0);
  const busyRef = useRef(true);

  const changeBusy = useCallback((next: boolean) => {
    busyRef.current = next;
    setBusy(next);
  }, []);

  const publish = useCallback((next: Session, token: number) => {
    if (token !== generation.current) return;
    const previous = current.current;
    if (previous && (previous.id !== next.id || previous.view_revision >= next.view_revision)) return;
    current.current = next;
    setSession(next);
  }, []);

  const open = useCallback(async (id: string, nextMode: Mode) => {
    if (!id) return;
    const token = ++generation.current;
    current.current = null;
    setSession(null);
    setScenarioId(id);
    setMode(nextMode);
    setAutoPlay(false);
    setProblem(null);
    changeBusy(true);
    try {
      publish(await createSession(id, nextMode), token);
    } catch (error) {
      if (token === generation.current) setProblem({ message: errorMessage(error), recovery: 'open' });
    } finally {
      if (token === generation.current) changeBusy(false);
    }
  }, [changeBusy, publish]);

  const load = useCallback(async () => {
    const token = ++generation.current;
    changeBusy(true);
    setProblem(null);
    try {
      const list = await listScenarios();
      if (token !== generation.current) return;
      setScenarios(list);
      if (list.length) {
        await open(list[0].id, 'simulation');
      } else {
        setProblem({ message: '재생할 시나리오를 찾지 못했습니다.', recovery: 'scenarios' });
      }
    } catch (error) {
      if (token === generation.current) setProblem({ message: errorMessage(error), recovery: 'scenarios' });
    } finally {
      if (token === generation.current) changeBusy(false);
    }
  }, [changeBusy, open]);

  useEffect(() => {
    void load();
    return () => { generation.current += 1; };
  }, [load]);

  const sessionId = session?.id;
  const inputVersion = session?.version;
  const processing = session?.processing ?? false;
  useEffect(() => {
    if (!sessionId || !processing) return;
    const token = generation.current;
    let canceled = false;
    let timer: number | undefined;
    let failures = 0;

    const schedule = (delay: number) => { timer = window.setTimeout(poll, delay); };
    const poll = async () => {
      try {
        const updated = await getSession(sessionId);
        if (canceled || token !== generation.current) return;
        failures = 0;
        publish(updated, token);
        if (current.current?.processing) schedule(POLL_DELAY_MS);
      } catch (error) {
        if (canceled || token !== generation.current) return;
        if (!(error instanceof ApiError && error.status === 404) && failures < POLL_RETRY_MS.length) {
          schedule(POLL_RETRY_MS[failures++]);
          return;
        }
        setAutoPlay(false);
        setProblem({
          message: errorMessage(error),
          recovery: error instanceof ApiError && error.status === 404 ? 'open' : 'sync',
        });
      }
    };
    schedule(POLL_DELAY_MS);
    return () => {
      canceled = true;
      window.clearTimeout(timer);
    };
  }, [inputVersion, pollCycle, processing, publish, sessionId]);

  const sync = useCallback(async (id: string, token: number, message?: string) => {
    try {
      const updated = await getSession(id);
      if (token !== generation.current) return;
      publish(updated, token);
      setProblem(message ? { message } : null);
      setPollCycle((cycle) => cycle + 1);
    } catch (error) {
      if (token !== generation.current) return;
      setProblem({
        message: errorMessage(error),
        recovery: error instanceof ApiError && error.status === 404 ? 'open' : 'sync',
      });
    }
  }, [publish]);

  const send = useCallback(async (action: Action, text?: string): Promise<boolean> => {
    const active = current.current;
    if (!active || busyRef.current) return false;
    const token = generation.current;
    changeBusy(true);
    setProblem(null);
    try {
      const updated = await postEvent(active, action, text);
      if (token !== generation.current) return false;
      publish(updated, token);
      return true;
    } catch (error) {
      if (token !== generation.current) return false;
      setAutoPlay(false);
      if (error instanceof ApiError && error.status === 409) {
        await sync(active.id, token, error.message);
      } else if (!(error instanceof ApiError) || error.status >= 500) {
        await sync(active.id, token, errorMessage(error));
      } else {
        setProblem({ message: errorMessage(error), recovery: error.status === 404 ? 'open' : undefined });
      }
      return false;
    } finally {
      if (token === generation.current) changeBusy(false);
    }
  }, [changeBusy, publish, sync]);

  const canNext = !!session && !session.manual && session.step < session.total_steps;
  useEffect(() => {
    if (!autoPlay) return;
    if (!canNext) {
      setAutoPlay(false);
      return;
    }
    if (processing || busy) return;
    const timer = window.setTimeout(() => { void send('next'); }, PLAY_DELAY_MS);
    return () => window.clearTimeout(timer);
  }, [autoPlay, busy, canNext, processing, send, session?.step]);

  const canRetry = !!session && session.turn_id > 0 && !session.manual && !processing &&
    (session.external_status === 'timeout' || session.external_status === 'cached' ||
      Object.values(session.decisions).some((decision) => decision.execution === 'error'));

  async function retry() {
    const active = current.current;
    if (busyRef.current) return;
    if (problem?.recovery === 'scenarios') {
      await load();
    } else if (problem?.recovery === 'open') {
      await open(scenarioId, mode);
    } else if (problem?.recovery === 'sync' && active) {
      const token = generation.current;
      changeBusy(true);
      await sync(active.id, token);
      if (token === generation.current) changeBusy(false);
    } else if (canRetry) {
      await send('retry');
    }
  }

  return { scenarios, scenarioId, mode, session, busy, autoPlay, problem, canNext, canRetry,
    open, retry, send, setAutoPlay };
}
