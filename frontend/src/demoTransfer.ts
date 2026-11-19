import type { TransferHospital, TransferResponse } from './dispatchApi';

type Suggestion = {
  response: TransferResponse;
  basis: { subtype: string; symptom?: string };
};

export type Transport = { unitId: string; hospital: TransferHospital; startedAt: number };

interface TransferState {
  key: string | null;
  generation: number;
  suggestion: Suggestion | null;
  issued: Transport | null;
}

export type TransferEvent =
  | { type: 'request'; key: string }
  | { type: 'receive'; generation: number; suggestion: Suggestion | null }
  | { type: 'issue'; authorized: boolean; unitId: string; startedAt: number }
  | { type: 'block' }
  | { type: 'reset' };

export const emptyTransfer = (): TransferState => ({ key: null, generation: 0, suggestion: null, issued: null });

export function advanceTransfer(state: TransferState, event: TransferEvent): TransferState {
  switch (event.type) {
    case 'reset':
      return { ...emptyTransfer(), generation: state.generation + 1 };
    case 'block':
      // An issued transport belongs to the call history and keeps its original destination.
      return state.issued || (state.key === null && !state.suggestion)
        ? state : { ...emptyTransfer(), generation: state.generation + 1 };
    case 'request':
      if (state.issued || state.key === event.key) return state;
      return { ...state, key: event.key, generation: state.generation + 1, suggestion: null };
    case 'receive': {
      if (state.issued || state.key === null || event.generation !== state.generation) return state;
      const status = event.suggestion?.response.status;
      const key = status === undefined || status === 'unavailable' ? null : state.key;
      return { ...state, key, suggestion: event.suggestion };
    }
    case 'issue': {
      if (!event.authorized || state.issued || !state.suggestion) return state;
      const response = state.suggestion.response;
      if (response.status !== 'ok') return state;
      const hospital = response.hospitals.find((option) => option.hpid === response.zoro?.choice);
      if (!hospital) return state;
      return { ...state, issued: { unitId: event.unitId, hospital, startedAt: event.startedAt } };
    }
  }
}
