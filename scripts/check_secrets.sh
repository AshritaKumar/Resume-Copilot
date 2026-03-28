#!/usr/bin/env bash
set -euo pipefail

python3 - <<'PY'
from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(".").resolve()
SKIP_FILES = {".env", ".env.example"}
SKIP_DIR_PARTS = {".git", "venv", ".venv", "__pycache__"}
PATTERNS = [
    re.compile(r"sk-proj-[A-Za-z0-9_-]{20,}"),
    re.compile(r"sk-lf-[A-Za-z0-9_-]{10,}"),
    re.compile(r"pk-lf-[A-Za-z0-9_-]{10,}"),
    re.compile(r"AIza[0-9A-Za-z\-_]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
]

def list_files() -> list[Path]:
    try:
        out = subprocess.check_output(["git", "ls-files"], text=True)
        return [ROOT / line.strip() for line in out.splitlines() if line.strip()]
    except Exception:
        files: list[Path] = []
        for path in ROOT.rglob("*"):
            if path.is_file():
                files.append(path)
        return files

def is_skipped(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    if rel.name in SKIP_FILES:
        return True
    return any(part in SKIP_DIR_PARTS for part in rel.parts)

hits: list[str] = []
for file_path in list_files():
    if is_skipped(file_path):
        continue
    try:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        continue
    for pattern in PATTERNS:
        if pattern.search(text):
            hits.append(str(file_path.relative_to(ROOT)))
            break

if hits:
    print("Potential secret found in:")
    for hit in hits:
        print(f" - {hit}")
    raise SystemExit(1)

print("Secret scan passed.")
PY
