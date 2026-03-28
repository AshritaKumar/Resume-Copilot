from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar


T = TypeVar("T")


def with_backoff(
    fn: Callable[[], T],
    max_attempts: int = 3,
    base_delay_seconds: float = 0.5,
) -> T:
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == max_attempts:
                break
            time.sleep(base_delay_seconds * (2 ** (attempt - 1)))
    assert last_error is not None
    raise last_error
