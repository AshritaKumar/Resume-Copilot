from __future__ import annotations

from pydantic import BaseModel, Field


class Evidence(BaseModel):
    quote: str
    source_hint: str = Field(default="resume_text")
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)


class ExperienceEntry(BaseModel):
    title: str | None = None
    company: str | None = None
    start: str | None = None
    end: str | None = None
    summary: str | None = None


class CandidateProfile(BaseModel):
    is_resume: bool = Field(default=True, description="Set to false if the document is not a resume (e.g. research paper, report, cover letter only, job description).")
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    location: str | None = None
    current_title: str | None = None
    total_years_experience: float | None = None
    skills: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    achievements: list[Evidence] = Field(default_factory=list)
    career_gaps_or_risks: list[str] = Field(default_factory=list)
    experiences: list[ExperienceEntry] = Field(default_factory=list)
    extraction_confidence: float = Field(ge=0.0, le=1.0, default=0.5)
