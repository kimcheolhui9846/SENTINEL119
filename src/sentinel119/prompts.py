"""Optional guideline-derived choice text, read at run time from an external knowledge directory.

The structured guideline knowledge is not stored in this repository. A run points ``--guideline-dir``
at the knowledge root (the directory containing ``extracted/chapters``) and the rendered text, the
source file hash and page reference are recorded in ``run.json``. Evaluation of a checkpoint reuses
the recorded text, so the knowledge directory is only required when a new prompt is rendered.

Only the urgency ``high`` option and the urgency instructions change. The dispatch-stage protocol
lists early/special dispatch situations; it does not define AI Hub ``medium``/``low``, so those keep
their bare names. Equating the protocol categories with the AI Hub ``high`` label is an unverified
hypothesis that each run must test against dev.
"""

from copy import deepcopy
import hashlib
import re

CHAPTER_ID = "p2-s2-c02"
VARIANTS = ("bare", "protocol", "protocol-checklist")
EARLY_ANCHOR = "조기 긴급출동지시가 필요한 상황"
SPECIAL_LINE = re.compile(r"^6-(\d)\.\s*\(\s*([^)]+?)\s*\)\s*(.*)$")
VITAL_LINE = re.compile(r"^5-([12])\.\s*(.+?\?)")


def _chapter(guideline_dir):
    matches = sorted((guideline_dir / "extracted" / "chapters").glob(f"{CHAPTER_ID}_*.md"))
    if len(matches) != 1:
        raise ValueError(f"Expected one guideline chapter {CHAPTER_ID} under {guideline_dir}")
    return matches[0]


def _clean(text):
    return " ".join(text.replace("⏏", " ").split())


def extract_protocol(text):
    """Parse the first (summary) page of the dispatch-stage protocol; fail on unexpected layout."""
    page = text.split("<!-- p.157 -->")[0]
    lines = [line.strip() for line in page.splitlines()]
    try:
        start = next(i for i, line in enumerate(lines) if EARLY_ANCHOR in line)
    except StopIteration:
        raise ValueError("Guideline chapter lacks the early-dispatch block") from None
    early = []
    for line in lines[start + 1:]:
        if re.match(r"^\d+\.", line):
            break
        if line.startswith(("○", "▶")):
            early.append(_clean(line.lstrip("○▶ ")))
    # Questions repeat on the detail pages; keep the occurrence with the fewest PDF spacing breaks.
    names, special, vitals = {}, {}, {}
    trauma = [_clean(line.split(":", 1)[1]) for line in text.splitlines()
              if "외상인 경우" in line and ":" in line and "?" in line]
    for line in (line.strip() for line in text.splitlines()):
        if match := SPECIAL_LINE.match(line):
            names.setdefault(match.group(1), _clean(match.group(2)))
            question = _clean(match.group(3))
            if "?" in question:
                special.setdefault(match.group(1), []).append(question[:question.index("?") + 1])
        if match := VITAL_LINE.match(line):
            vitals.setdefault(match.group(1), []).append(_clean(match.group(2)))
    if trauma:
        special.setdefault("2", []).extend(q[:q.index("?") + 1] for q in trauma)
    tidy = {key: min(found, key=lambda q: q.count(" ")) for key, found in special.items()}
    vitals = {key: min(found, key=lambda q: q.count(" ")) for key, found in vitals.items()}
    if (len(early) < 3 or sorted(names) != ["1", "2", "3", "4", "5"] or sorted(tidy) != sorted(names)
            or sorted(vitals) != ["1", "2"]):
        raise ValueError("Guideline chapter layout differs from the parsed 2023 protocol")
    return {"early": early, "special": [(names[k], tidy[k]) for k in sorted(names)],
            "vitals": [vitals[k] for k in sorted(vitals)]}


def render(variant, guideline_dir=None):
    """Return a prompt spec: overrides plus provenance. ``bare`` keeps the prepared questions."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown prompt variant {variant}")
    if variant == "bare":
        return {"variant": "bare", "overrides": {}}
    if guideline_dir is None:
        raise ValueError(f"--guideline-dir is required for prompt variant {variant}")
    path = _chapter(guideline_dir)
    payload = path.read_bytes()
    protocol = extract_protocol(payload.decode("utf-8"))
    special = ", ".join(name.removesuffix(" 추정") for name, _ in protocol["special"])
    high = (f"상 — 수보단계 지침의 조기 긴급출동 상황({'; '.join(protocol['early'])}) "
            f"또는 특별구급대 출동 대상 중증응급환자 추정({special})")
    urgency = {"criteria": {"high": high}}
    if variant == "protocol-checklist":
        questions = protocol["vitals"] + [question for _, question in protocol["special"] if question]
        urgency["instructions_suffix"] = (" 참고: 수보단계 중증응급환자 인지 프로토콜의 확인 질문 — "
                                          + " ".join(questions))
    return {"variant": variant, "overrides": {"urgency": urgency},
            "source": {"chapter_id": CHAPTER_ID, "file": path.name, "pages": "156-159",
                       "sha256": hashlib.sha256(payload).hexdigest(),
                       "label_alignment": "unverified hypothesis"}}


def apply(example, spec):
    """Return a copy with overridden choice text; labels, keys and metadata are unchanged."""
    if not spec or not spec.get("overrides"):
        return example
    result = deepcopy(example)
    for qid, override in spec["overrides"].items():
        question = result["questions"][qid]
        unknown = set(override.get("criteria", {})) - set(question["criteria"])
        if unknown:
            raise ValueError(f"Prompt override adds choices {sorted(unknown)} to {qid}")
        question["criteria"].update(override.get("criteria", {}))
        question["instructions"] = question.get("instructions", "") + override.get("instructions_suffix", "")
    return result
