from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def _extract_citations(text: str) -> set[int]:
    return {int(m.group(1)) for m in re.finditer(r"\[(\d{1,2})\]", text or "")}


def _check_case(case: dict) -> list[str]:
    errors: list[str] = []
    answer = str(case.get("answer", "")).strip()
    rationale = str(case.get("rationale", "")).strip()
    intent = str(case.get("intent", "general_qa")).strip()
    confidence = float(case.get("confidence", 0.0) or 0.0)
    evidence_count = int(case.get("evidence_count", 0) or 0)
    min_conf = float(case.get("min_confidence", 0.0) or 0.0)
    max_conf = float(case.get("max_confidence", 1.0) or 1.0)
    allow_uncertain = bool(case.get("allow_uncertain", True))
    min_citations = int(case.get("min_citations", 1) or 1)

    answer_cits = _extract_citations(answer)
    rationale_cits = _extract_citations(rationale)
    all_cits = answer_cits | rationale_cits

    if not answer:
        errors.append("answer empty")
    if len(answer) < 80:
        errors.append("answer too short")
    if len(answer_cits) < 1:
        errors.append("answer missing inline citation")
    if len(all_cits) < min_citations:
        errors.append(f"citations below required minimum ({min_citations})")
    if all_cits and evidence_count > 0 and max(all_cits) > evidence_count:
        errors.append("citation out of range")
    if confidence < min_conf or confidence > max_conf:
        errors.append("confidence out of expected range")
    if not allow_uncertain and "uncertain" in answer.lower():
        errors.append("uncertain wording not allowed")
    if intent in {"role_fit", "decision", "general_ranking"} and len(all_cits) < 2:
        errors.append("comparative answer should include >=2 citations")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("sample_output/golden_queries.json"),
    )
    args = parser.parse_args()
    if not args.cases.exists():
        print(f"[error] missing cases file: {args.cases}")
        return 2

    payload = json.loads(args.cases.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        print("[error] cases file must be a JSON array")
        return 2

    failed = 0
    for idx, case in enumerate(payload, start=1):
        if not isinstance(case, dict):
            failed += 1
            print(f"[fail] case-{idx}: invalid object")
            continue
        name = str(case.get("name", f"case-{idx}"))
        errors = _check_case(case)
        if errors:
            failed += 1
            print(f"[fail] {name}")
            for err in errors:
                print(f"  - {err}")
        else:
            print(f"[pass] {name}")

    total = len(payload)
    print(f"\nSummary: {total - failed} passed, {failed} failed, total={total}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
