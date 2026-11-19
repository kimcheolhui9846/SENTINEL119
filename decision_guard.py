"""Gate Zoro choices against a session's server-reviewed snapshot."""

from collections.abc import Mapping
from dataclasses import dataclass
from json import dumps
from math import fsum
from sys import float_info
from threading import Lock
from time import monotonic
from types import MappingProxyType
from weakref import WeakKeyDictionary


_LABELS = {
    "dispatch": ("화재", "구조", "구급", "기타"),
    "urgency": ("상", "중", "하"),
}
_DEPENDENTS = ("vehicle", "hospital")
_TASKS = (*_LABELS, *_DEPENDENTS)
_AVAILABILITY = ("available", "unknown", "unavailable")


@dataclass(frozen=True, slots=True)
class Decision:
    """A result with the source snapshot revision and reviewed turn IDs."""

    session_id: str
    turn_id: int
    revision: int
    task: str | None
    status: str
    reason: str
    choice: str | None = None
    evidence_turn_ids: tuple[int, ...] = ()
    probabilities: Mapping[str, float] | None = None


class _Ticket:
    __slots__ = ("__weakref__",)


@dataclass(frozen=True, slots=True)
class _Request:
    task: str
    turn_id: int
    revision: int
    issued_at: float


@dataclass(frozen=True, slots=True)
class _CandidateVersion:
    number: int
    candidates: dict[str, str]
    conflicted: bool = False


def _copy_json(value):
    if type(value) is dict:
        result = {}
        for key, child in value.items():
            if type(key) is not str:
                raise ValueError("ranking facts require string keys")
            result[key] = _copy_json(child)
        return result
    if type(value) is list:
        return [_copy_json(child) for child in value]
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise ValueError("ranking facts require JSON values")


class CandidateVersionTracker:
    """Version the complete ranking facts for one session and candidate task."""

    def __init__(self):
        self._lock = Lock()
        self._canonical = None
        self._version = 0

    def observe(self, facts):
        """Advance the version when the canonical JSON facts change."""
        if type(facts) is not dict:
            raise ValueError("ranking facts must be a JSON object")
        with self._lock:
            try:
                canonical = dumps(_copy_json(facts), sort_keys=True, allow_nan=False,
                                  separators=(",", ":"))
            except RecursionError as error:
                raise ValueError("ranking facts must be a finite JSON tree") from error
            if canonical != self._canonical:
                self._version += 1
                self._canonical = canonical
            return self._version


def _nonblank(value):
    return type(value) is str and bool(value.strip())


def _seen(turn, current):
    return type(turn) is int and 1 <= turn <= current


def _copy_supports(supports, turn):
    if not isinstance(supports, Mapping):
        raise ValueError("supports must map classification tasks to reviewed labels")
    supports = dict(supports)
    if not supports.keys() <= _LABELS.keys():
        raise ValueError("supports must map classification tasks to reviewed labels")
    result = {}
    for task, labels in _LABELS.items():
        source = supports.get(task, {})
        if not isinstance(source, Mapping):
            raise ValueError("invalid reviewed labels for " + task)
        source = dict(source)
        if not source.keys() <= set(labels):
            raise ValueError("invalid reviewed label for " + task)
        result[task] = {}
        for label, ids in source.items():
            ids = tuple(ids) if isinstance(ids, (list, tuple)) else ()
            if (not ids or not all(_seen(source_id, turn) for source_id in ids)
                    or len(set(ids)) != len(ids)):
                raise ValueError("a reviewed label needs distinct seen turn IDs")
            result[task][label] = tuple(sorted(ids))
    return result


def _copy_candidates(candidates):
    if not isinstance(candidates, Mapping):
        raise ValueError("candidates must be a mapping of at most 255 IDs")
    owned = dict(candidates)
    if len(owned) > 255:
        raise ValueError("candidates must be a mapping of at most 255 IDs")
    if any(not _nonblank(key) or type(value) is not str or value not in _AVAILABILITY
           for key, value in owned.items()):
        raise ValueError("invalid candidate ID or availability")
    return owned


def _copy_candidate_versions(versions):
    if versions is None:
        return {}
    if not isinstance(versions, Mapping):
        raise ValueError("candidate_versions must be a mapping or None")
    versions = dict(versions)
    if any(type(task) is not str or task not in _DEPENDENTS or type(number) is not int
           or number < 0 for task, number in versions.items()):
        raise ValueError("candidate_versions requires vehicle or hospital keys and nonnegative integers")
    return versions


def _model_choice(envelope, task, labels):
    if not isinstance(envelope, Mapping):
        return None
    answers = envelope.get("answers")
    if not isinstance(answers, Mapping):
        return None
    answer = answers.get(task)
    if not isinstance(answer, Mapping) or answer.get("type") != "choice":
        return None
    choice = answer.get("choice")
    probabilities = answer.get("probabilities")
    if (type(choice) is not str or choice not in labels
            or not isinstance(probabilities, Mapping) or len(probabilities) != len(labels)):
        return None
    probabilities = dict(probabilities)
    if probabilities.keys() != set(labels):
        return None
    values = probabilities.values()
    if any(type(value) not in (int, float) or not 0 <= value <= 1 for value in values):
        return None
    if abs(fsum(values) - 1) > len(labels) * 0.00005 + 1e-12:
        return None
    if probabilities[choice] != max(values):
        return None
    return choice, MappingProxyType(probabilities)


class Guard:
    """Advance reviewed facts, issue one ticket per task, then apply its reply.

    Candidate maps are server-prequalified. Zoro supplies the choice and its
    probabilities. Every active recommendation uses the current reviewed facts.
    """

    def __init__(self, session_id, *, max_response_age_s=5.0, clock=monotonic, thresholds=None):
        """thresholds {dispatch|urgency: p} switches a task from reviewed support to model confidence:
        COMMIT when the top probability reaches p, and an earlier COMMIT stays until another label
        reaches p. Urgency waits for a committed dispatch (the hierarchy opens after the type)."""
        thresholds = dict(thresholds or {})
        if any(task not in _LABELS or type(value) not in (int, float) or not 0 < value <= 1
               for task, value in thresholds.items()):
            raise ValueError("thresholds map dispatch/urgency to a probability in (0, 1]")
        self._thresholds = thresholds
        if not _nonblank(session_id):
            raise ValueError("session_id must be a nonblank string")
        if (type(max_response_age_s) not in (int, float)
                or not 0 < max_response_age_s <= float_info.max):
            raise ValueError("max_response_age_s must be a finite positive number")
        if not callable(clock):
            raise ValueError("clock must be a monotonic time source")
        self._session_id = session_id
        self._max_response_age_s = max_response_age_s
        self._clock = clock
        self._lock = Lock()
        self._turn = self._revision = 0
        self._supports = {task: {} for task in _LABELS}
        self._location = None
        self._candidates = {task: {} for task in _DEPENDENTS}
        self._candidate_versions = {}
        self._candidate_version_issues = {}
        self._active = {task: self._result(task, "WAIT", "no decision yet") for task in _TASKS}
        self._pending = {}
        # Audit late tickets while their callers still hold them.
        self._issued = WeakKeyDictionary()
        self._history = []

    def advance(self, turn_id, *, supports, location, vehicles, hospitals, candidate_versions=None):
        """Replace the full reviewed snapshot for the current or next turn."""
        with self._lock:
            if not _seen(turn_id, self._turn + 1) or turn_id < self._turn:
                raise ValueError("turn_id must be the current or next positive integer")
            supports = _copy_supports(supports, turn_id)
            if location is not None and (type(location) is not tuple or len(location) != 2
                    or not _nonblank(location[0]) or not _seen(location[1], turn_id)):
                raise ValueError("location must be (value, seen turn ID) or None")
            candidates = {"vehicle": _copy_candidates(vehicles),
                          "hospital": _copy_candidates(hospitals)}
            versions = _copy_candidate_versions(candidate_versions)

            location_changed = location != self._location
            previous_candidates = self._candidates
            self._turn, self._supports, self._location, self._candidates = (
                turn_id, supports, location, candidates)
            self._revision += 1
            self._pending.clear()
            for task in _LABELS:
                current = self._active[task]
                if task in self._thresholds:
                    continue
                if current.status == "COMMIT" and supports[task] != {
                        current.choice: current.evidence_turn_ids}:
                    self._set_active(self._result(task, "WAIT", "reviewed support changed"))
            for task in _DEPENDENTS:
                self._refresh_candidate(task, versions.get(task), location_changed,
                                        candidates[task] != previous_candidates[task])

    def _refresh_candidate(self, task, version, location_changed, candidates_changed):
        highest = self._candidate_versions.get(task)
        version_changed = version is not None and (highest is None or version > highest.number)
        issue = None
        if version_changed:
            self._candidate_versions[task] = _CandidateVersion(version, self._candidates[task])
        elif highest is not None:
            if version is None or version < highest.number:
                issue = "candidate version missing or older than the latest version"
            elif highest.conflicted or highest.candidates != self._candidates[task]:
                self._candidate_versions[task] = _CandidateVersion(highest.number, highest.candidates, True)
                issue = "candidate version conflicts with its reviewed candidate map"

        previous_issue = self._candidate_version_issues.pop(task, None)
        if issue is not None:
            self._candidate_version_issues[task] = issue
            self._set_active(self._result(task, self._dependent_status(), issue))
        elif self._active[task].status == "COMMIT":
            if version_changed:
                self._set_active(self._result(task, "WAIT", "reviewed candidate version changed"))
            elif location_changed or candidates_changed:
                self._set_active(self._result(task, "WAIT", "reviewed location or candidates changed"))
        elif previous_issue is not None:
            self._set_active(self._result(task, self._dependent_status(), "candidate version restored"))

    def begin(self, task):
        """Issue an opaque, latest-only ticket for this task and snapshot."""
        with self._lock:
            self._check_task(task)
            ticket = _Ticket()
            self._issued[ticket] = _Request(task, self._turn, self._revision, self._clock())
            self._pending[task] = ticket
            return ticket

    def apply(self, ticket, envelope):
        """Consume a current ticket and validate its typed Zoro choice."""
        with self._lock:
            request = self._issued.get(ticket) if type(ticket) is _Ticket else None
            if request is None:
                return self._result(None, "STALE", "ticket is no longer current")
            task = request.task
            if self._pending.get(task) is not ticket:
                return Decision(self._session_id, request.turn_id, request.revision, task,
                                "STALE", "ticket is no longer current")
            del self._pending[task]
            if self._clock() - request.issued_at >= self._max_response_age_s:
                return self._failed(task, "Zoro response expired")

            downstream = task in _DEPENDENTS
            version_issue = self._candidate_version_issues.get(task)
            if version_issue is not None:
                result = self._result(task, self._dependent_status(), version_issue)
            elif downstream and not self._candidates[task]:
                status = self._dependent_status()
                result = self._result(task, status, "no candidates in the reviewed snapshot")
            else:
                labels = self._candidates[task] if downstream else _LABELS[task]
                model_choice = _model_choice(envelope, task, labels)
                if model_choice is None:
                    return self._failed(task, "invalid Zoro choice response")
                choice, probabilities = model_choice
                if downstream:
                    result = self._candidate_decision(task, choice, probabilities)
                elif task in self._thresholds:
                    result = self._threshold_decision(task, choice, probabilities)
                else:
                    support = self._supports[task]
                    if len(support) == 1 and choice in support:
                        result = self._result(task, "COMMIT", "reviewed support matches Zoro choice",
                                              choice, support[choice], probabilities)
                    else:
                        result = self._result(task, "WAIT", "no unique matching reviewed support",
                                              probabilities=probabilities)
            self._set_active(result)
            return result

    def active(self, task):
        with self._lock:
            self._check_task(task)
            return self._active[task]

    @property
    def history(self):
        """Immutable history of active status, choice, and evidence changes."""
        with self._lock:
            return tuple(self._history)

    def _candidate_decision(self, task, choice, probabilities):
        dispatch = self._active["dispatch"]
        if dispatch.status != "COMMIT" or dispatch.choice != "구급":
            return self._result(task, self._dependent_status(), "requires a 구급 dispatch",
                                probabilities=probabilities)
        if self._location is None or self._candidates[task][choice] != "available":
            return self._result(task, "WAIT", "requires reviewed location and available candidate",
                                probabilities=probabilities)
        evidence = tuple(sorted({*dispatch.evidence_turn_ids, self._location[1]}))
        return self._result(task, "COMMIT", "reviewed dispatch, location and candidate match",
                            choice, evidence, probabilities)

    def _threshold_decision(self, task, choice, probabilities):
        current = self._active[task]
        if task != "dispatch" and self._active["dispatch"].status != "COMMIT":
            return self._result(task, "WAIT", "requires a committed dispatch", probabilities=probabilities)
        if probabilities[choice] >= self._thresholds[task]:
            if current.status == "COMMIT" and current.choice == choice:
                evidence = current.evidence_turn_ids
            else:
                evidence = (self._turn,)
            return self._result(task, "COMMIT", "model probability reached threshold",
                                choice, evidence, probabilities)
        if current.status == "COMMIT":
            return self._result(task, "COMMIT", "model probability reached threshold",
                                current.choice, current.evidence_turn_ids, probabilities)
        return self._result(task, "WAIT", "model probability below threshold", probabilities=probabilities)

    def _dependent_status(self):
        dispatch = self._active["dispatch"]
        if dispatch.status == "COMMIT" and dispatch.choice != "구급":
            return "OUT_OF_SCOPE"
        return "WAIT"

    def _failed(self, task, reason):
        status = self._dependent_status() if task in _DEPENDENTS else "WAIT"
        # A failed call does not withdraw a confidence commit; only a stronger label replaces it.
        if not (task in self._thresholds and self._active[task].status == "COMMIT"):
            self._set_active(self._result(task, status, reason))
        return self._result(task, "ERROR", reason)

    def _set_active(self, result):
        previous = self._active[result.task]
        self._active[result.task] = result
        if (previous.status, previous.choice, previous.evidence_turn_ids) == (
                result.status, result.choice, result.evidence_turn_ids):
            return
        self._history.append(result)

        if result.task == "dispatch":
            for task in _DEPENDENTS:
                self._pending.pop(task, None)
                self._set_active(self._result(
                    task, self._dependent_status(), "upstream dispatch changed"))

    def _result(self, task, status, reason, choice=None, evidence=(), probabilities=None):
        return Decision(self._session_id, self._turn, self._revision, task, status, reason,
                        choice, evidence, probabilities)

    @staticmethod
    def _check_task(task):
        if type(task) is not str or task not in _TASKS:
            raise ValueError("unknown decision task")
