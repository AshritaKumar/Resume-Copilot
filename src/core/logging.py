from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class TraceEvent:
    run_id: str
    stage: str
    message: str
    model: str | None = None
    latency_ms: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    retry_count: int | None = None
    metadata: dict[str, Any] | None = None


def build_logger(name: str = "confido_pipeline") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    return logger


def write_trace_event(trace_path: Path, event: TraceEvent) -> None:
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **asdict(event),
    }
    with trace_path.open("a") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def log_prompt_usage(
    logger: logging.Logger,
    prompt_name: str,
    prompt_source: str,
    prompt_text: str,
    *,
    stage: str = "prompt",
    run_id: str | None = None,
    trace_path: Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    preview = " ".join(prompt_text.split())[:180]
    sha = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:12]
    base_metadata = {
        "prompt_name": prompt_name,
        "source": prompt_source,
        "chars": len(prompt_text),
        "sha12": sha,
        "preview": preview,
    }
    if metadata:
        base_metadata.update(metadata)

    logger.info(
        "Prompt used | name=%s source=%s chars=%s sha12=%s",
        prompt_name,
        prompt_source,
        len(prompt_text),
        sha,
    )
    if run_id and trace_path:
        write_trace_event(
            trace_path,
            TraceEvent(
                run_id=run_id,
                stage=stage,
                message=f"Prompt used: {prompt_name}",
                metadata=base_metadata,
            ),
        )
