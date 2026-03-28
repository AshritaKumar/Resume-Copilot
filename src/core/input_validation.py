from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

ALLOWED_JD_SUFFIXES = {".md", ".pdf", ".docx"}
ALLOWED_RESUME_SUFFIXES = {".pdf", ".docx", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}


ErrorFactory = Callable[[str], Exception]


def validate_uploaded_input_types(
    jd_filename: str,
    resume_filenames: Sequence[str],
    *,
    error_factory: ErrorFactory = RuntimeError,
) -> None:
    jd_suffix = Path(jd_filename).suffix.lower()
    if jd_suffix not in ALLOWED_JD_SUFFIXES:
        allowed = ", ".join(sorted(ALLOWED_JD_SUFFIXES))
        raise error_factory(f"Unsupported JD file: `{jd_filename}`. Allowed JD formats: {allowed}.")

    invalid_resume_files = [
        filename for filename in resume_filenames
        if Path(filename).suffix.lower() not in ALLOWED_RESUME_SUFFIXES
    ]
    if invalid_resume_files:
        allowed = ", ".join(sorted(ALLOWED_RESUME_SUFFIXES))
        listed = ", ".join(f"`{name}`" for name in invalid_resume_files)
        raise error_factory(f"Unsupported resume file(s): {listed}. Allowed resume formats: {allowed}.")


def validate_path_input_types(
    jd_path: Path,
    resumes_dir: Path,
    *,
    error_factory: ErrorFactory = RuntimeError,
) -> None:
    resume_filenames = [path.name for path in resumes_dir.iterdir() if path.is_file()]
    validate_uploaded_input_types(
        jd_filename=jd_path.name,
        resume_filenames=resume_filenames,
        error_factory=error_factory,
    )
