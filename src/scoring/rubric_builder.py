from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from src.extract.jd_parser import JDInterpretation


@dataclass
class Criterion:
    name: str
    category: str
    weight: float


@dataclass
class Rubric:
    criteria: list[Criterion]
    metadata: dict[str, float]


def build_rubric(jd: JDInterpretation, defaults_path: Path) -> Rubric:
    defaults = _load_defaults(defaults_path)
    buckets = [
        ("must_have", jd.must_have, float(defaults.get("must_have_weight", 60))),
        ("nice_to_have", jd.nice_to_have, float(defaults.get("nice_to_have_weight", 20))),
        ("risk", jd.risk_flags or ["General risk signals"], float(defaults.get("risk_weight", 10))),
        (
            "communication",
            jd.communication_leadership or ["Communication and collaboration"],
            float(defaults.get("communication_weight", 10)),
        ),
    ]

    criteria: list[Criterion] = []
    for category, items, weight in buckets:
        criteria.extend(_weighted(items, category, weight))

    return Rubric(criteria=criteria, metadata=defaults)


def _load_defaults(path: Path) -> dict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def _weighted(items: list[str], category: str, bucket_weight: float) -> list[Criterion]:
    if not items:
        return []
    each = bucket_weight / len(items)
    return [Criterion(name=item, category=category, weight=round(each, 2)) for item in items]
