from __future__ import annotations

from pathlib import Path


class PipelineError(Exception):
    """Base pipeline exception."""


class ParseError(PipelineError):
    """Raised when document parsing fails."""


class ExtractionError(PipelineError):
    """Raised when structured extraction fails."""


class RetrievalError(PipelineError):
    """Raised when embedding retrieval fails."""


class GenerationError(PipelineError):
    """Raised when content generation fails."""


class UserFacingPipelineMessage(PipelineError):
    """Expected outcome (e.g. no valid resumes); show in the UI instead of a traceback."""

    def __init__(self, message: str, *, output_dir: Path | None = None) -> None:
        super().__init__(message)
        self.output_dir = output_dir
