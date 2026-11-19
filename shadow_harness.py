"""Read-only decision policy for reviewed synthetic dispatch transcripts."""

from collections.abc import Mapping
from math import fsum, isclose


POLICY_VERSION = "shadow-intake-v1"
_MIN_CONFIDENCE = .85
_CATEGORIES = {"ems", "fire"}
_DISPATCH_LABELS = _CATEGORIES | {"rescue", "other"}
_EMS_SUBTYPES = {
    "severe_illness", "illness", "cardiac_arrest", "injury",
    "accident", "pregnancy", "poisoning", "other_ems",
}
_CALLER = "신고자"
_SPEAKERS = {_CALLER, "수보요원"}

# Whole reviewed utterances are fixtures for the exercise. Text is preserved verbatim.
_REQUESTS = {
    "성수동에서 사람이 갑자기 쓰러졌어요. 구급차를 보내 주세요.": "ems",
    "성수동에서 사람이 갑자기 쓰러졌어요. 구급차를 보내 주실래요?": "ems",
    "성수동 건물에 불이 났어요. 연기와 불꽃이 보여요. 소방차를 보내 주세요.": "fire",
}
_CANCEL = "정정할게요. 연습 상황이었고 구급차 요청을 취소합니다."
_UPDATES = {
    "의식이 없고 숨을 전혀 쉬지 않아요. 심정지가 의심된다고 해서 옆사람이 심폐소생술을 하고 있어요.": "ems",
    "불길이 번지고 있어요. 건물 밖으로 대피했어요.": "fire",
}


def _valid_probabilities(scores, labels):
    if not isinstance(scores, Mapping) or scores.keys() != labels:
        return False
    values = scores.values()
    return (
        all(type(value) in (int, float) and 0 <= value <= 1 for value in values)
        and isclose(fsum(values), 1.0, rel_tol=0, abs_tol=1e-5)
    )


def valid_dispatch_scores(scores):
    """Check the complete mandatory distribution without choosing a category."""
    return _valid_probabilities(scores, _DISPATCH_LABELS)


def _nonblank(value):
    return isinstance(value, str) and bool(value.strip())


def _validate_inputs(turns, eligible_units, model_id, adapter_version, applied_unit):
    if not _nonblank(model_id) or not _nonblank(adapter_version):
        raise ValueError("model_id and adapter_version must be nonblank strings")
    if not isinstance(turns, (list, tuple)) or not turns:
        raise ValueError("turns must be a nonempty sequence")
    for turn in turns:
        if (
            not isinstance(turn, Mapping) or turn.keys() != {"speaker", "text"}
            or not isinstance(turn["speaker"], str) or turn["speaker"] not in _SPEAKERS
            or not _nonblank(turn["text"])
        ):
            raise ValueError("each turn must contain a supported speaker and nonblank text")
    if not isinstance(eligible_units, Mapping) or not eligible_units.keys() <= _CATEGORIES:
        raise ValueError("eligible_units must map ems or fire to ordered unit IDs")
    for unit_ids in eligible_units.values():
        if (
            not isinstance(unit_ids, (list, tuple)) or any(not _nonblank(unit) for unit in unit_ids)
            or len(set(unit_ids)) != len(unit_ids)
        ):
            raise ValueError("eligible unit IDs must be distinct nonblank strings")
    if applied_unit is not None and (
        not isinstance(applied_unit, Mapping) or applied_unit.keys() != {"category", "unit_id"}
        or not isinstance(applied_unit["category"], str) or applied_unit["category"] not in _CATEGORIES
        or not _nonblank(applied_unit["unit_id"])
    ):
        raise ValueError("applied_unit must contain a supported category and unit ID")


def _review_evidence(turns):
    category = evidence_turn = evidence_span = None
    for turn_number, turn in enumerate(turns, 1):
        if turn["speaker"] != _CALLER:
            continue
        message = turn["text"]
        request = _REQUESTS.get(message)
        update = _UPDATES.get(message)
        if message == _CANCEL:
            return None, None, None, "cancelled_request"
        if category is not None:
            if request is None and update is None:
                return None, None, None, "unreviewed_caller_turn"
            if (request or update) != category:
                return None, None, None, "conflicting_evidence"
        if request is not None:
            category, evidence_turn, evidence_span = request, turn_number, message
    return category, evidence_turn, evidence_span, None


def decide_shadow(
    turns, dispatch_scores, eligible_units, *, model_id, adapter_version,
    applied_unit=None, subtype_scores=None,
):
    """Return an advisory JSON result using one reviewed prefix and one saved response.

    ``evidence_span`` is the complete literal caller turn identified by the 1-based
    ``evidence_turn``. The resource owner provides eligible IDs in its selection
    order. An ambiguous or cancelled request stays in review for the call prefix.
    A valid optional EMS distribution makes its head available for evaluation.
    """
    _validate_inputs(turns, eligible_units, model_id, adapter_version, applied_unit)
    evidence_category, evidence_turn, evidence_span, evidence_error = _review_evidence(turns)
    result = {
        "category": None,
        "evidence_turn": None,
        "evidence_span": None,
        "eligible_units": [],
        "proposed_unit": None,
        "needs_review": evidence_error is not None,
        "reason": evidence_error,
        "subtype_status": "wait",
        "model_id": model_id,
        "adapter_version": adapter_version,
        "policy_version": POLICY_VERSION,
    }
    if evidence_error is not None:
        return result
    if not valid_dispatch_scores(dispatch_scores):
        result["reason"] = "invalid_dispatch_scores"
        return result
    if evidence_category is None:
        result["reason"] = "no_reviewed_evidence"
        return result

    choice = max(dispatch_scores, key=dispatch_scores.__getitem__)
    confidence = dispatch_scores[choice]
    if confidence < _MIN_CONFIDENCE:
        result["reason"] = "uncertain_score"
        return result
    if choice != evidence_category:
        result["reason"] = "model_evidence_disagreement"
        result["needs_review"] = True
        return result

    matching_units = list(eligible_units.get(choice, ()))
    result.update(category=choice, evidence_turn=evidence_turn,
                  evidence_span=evidence_span, eligible_units=matching_units)
    if choice == "ems" and _valid_probabilities(subtype_scores, _EMS_SUBTYPES):
        result["subtype_status"] = "available"
    if applied_unit is not None:
        result["needs_review"] = applied_unit["category"] != choice
        result["reason"] = "applied_category_conflict" if result["needs_review"] else "already_applied"
    elif matching_units:
        result["proposed_unit"] = matching_units[0]
        result["reason"] = "proposed"
    else:
        result["reason"] = "no_eligible_unit"
    return result
