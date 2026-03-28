"""Heuristic career signals: employment gaps from experience rows, non-standard titles (no ontology)."""

from __future__ import annotations

import re
from datetime import date

from src.extract.candidate_schema import CandidateProfile, ExperienceEntry

_YEAR = re.compile(r"\b(19\d{2}|20\d{2})\b")
_PRESENT = re.compile(r"\b(present|current|now|today)\b", re.I)

# Common tokens in corporate / tech job titles (not exhaustive—flags creative titles without a fuzzy match layer).
_TITLE_TOKENS = frozenset(
    "engineer developer engineering scientist researcher analyst architect manager director lead "
    "head vp vice president officer consultant specialist coordinator associate intern trainee "
    "founder cofounder co-founder owner partner principal staff senior junior graduate "
    "designer product designer ux ui devops sre data scientist ml ai machine learning "
    "software backend frontend fullstack full-stack qa tester support sales marketing hr "
    "administrative executive assistant representative advisor faculty professor lecturer "
    "technician operator supervisor chief executive".split()
)


def enrich_career_signals(profile: CandidateProfile, resume_text: str) -> None:
    """Merge deterministic signals into profile.career_gaps_or_risks (deduped)."""
    discovered: list[str] = []
    discovered.extend(_gaps_between_experiences(profile.experiences))
    if len(profile.experiences) <= 1:
        discovered.append(
            "Limited structured work history extracted from this file—double-check dates and roles on the source resume."
        )
    note = _nonstandard_title_note(profile.current_title)
    if note:
        discovered.append(note)

    merged: list[str] = []
    seen: set[str] = set()
    for item in list(profile.career_gaps_or_risks or []) + discovered:
        key = item.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(item.strip())
    profile.career_gaps_or_risks = merged


def _parse_year_from_field(s: str | None) -> int | None:
    if not s or not str(s).strip():
        return None
    if _PRESENT.search(s):
        return date.today().year
    years = [int(m) for m in _YEAR.findall(s)]
    if not years:
        return None
    return max(years)


def _parse_start_year(s: str | None) -> int | None:
    if not s or not str(s).strip():
        return None
    years = [int(m) for m in _YEAR.findall(s)]
    if not years:
        return None
    return min(years)


def _gaps_between_experiences(experiences: list[ExperienceEntry]) -> list[str]:
    """Assumes list is reverse-chronological (current role first), typical resume layout."""
    if len(experiences) < 2:
        return []
    out: list[str] = []
    for newer, older in zip(experiences[:-1], experiences[1:]):
        end_older = _parse_year_from_field(older.end)
        if end_older is None:
            end_older = _parse_year_from_field(older.start)
        start_newer = _parse_start_year(newer.start)
        if end_older is None or start_newer is None:
            continue
        if start_newer <= end_older:
            continue
        gap_years = float(start_newer - end_older)
        if gap_years >= 1.0:
            left = older.company or older.title or "earlier role"
            right = newer.company or newer.title or "later role"
            out.append(
                f"Possible résumé gap (~{gap_years:.0f} y) between {left!s} and {right!s}—confirm in full CV."
            )
            if len(out) >= 4:
                break
    return out


def _nonstandard_title_note(title: str | None) -> str | None:
    if not title or len(title.strip()) < 2:
        return None
    raw = title.strip()
    lower = raw.lower()
    words = {w for w in re.split(r"[^\w]+", lower) if len(w) > 1}
    if not words:
        return "Current title could not be parsed cleanly—verify role level and function."

    if words & _TITLE_TOKENS:
        if len(raw) > 72:
            return "Job title is unusually long—may combine multiple hats; confirm primary function for this search."
        return None

    if len(raw) <= 5 and raw.isupper():
        return None

    return (
        f"Title «{raw}» does not match common role keywords—non-standard or creative titling; "
        "align manually with the JD (level, IC vs people leadership)."
    )
