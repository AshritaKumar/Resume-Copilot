"""Consistent resume filename → disk name and candidate_id (matches pipeline stem logic)."""

from __future__ import annotations

import re
from pathlib import Path


def sanitize_upload_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return cleaned or "file"


def candidate_id_from_upload_name(name: str) -> str:
    return Path(sanitize_upload_filename(name)).stem.lower().replace(" ", "_")
