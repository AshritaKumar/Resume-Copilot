# Confido Hiring Automation System

End-to-end hiring automation pipeline for the AI Generalist take-home assignment.

Inputs:
- one job description file (`.pdf`, `.docx`, `.md`)
- one folder of mixed-format resumes (`.pdf`, `.docx`, image files)

Outputs:
- ranked shortlist with evidence-backed reasoning
- personalized outreach emails for shortlisted candidates
- structured interview kits per shortlisted candidate

## Stack

- Python 3.11 + Streamlit
- OpenAI models:
  - primary LLM: `gpt-4.1-mini`
  - high-stakes LLM: `gpt-4.1`
  - embeddings: `text-embedding-3-small`
- Prompt management: Langfuse (fail-fast configurable)
- PostgreSQL + `pgvector` (KNN semantic retrieval)
- Parsing/OCR: `pypdf`, `python-docx`, `pytesseract`
- Validation: Pydantic schemas

## Architecture Decisions

1. **Pipeline over single prompt**
   - Explicit stages for parse, extract, rubric, retrieval, scoring, generation, export.
   - Each stage has recoverable failures and warnings.

2. **Hybrid scoring**
   - Deterministic checks (skill/title signals) + semantic evidence retrieval.
   - Unknowns are explicit; missing evidence downgrades confidence.

3. **Single datastore**
   - Postgres stores both relational data and vectors through `pgvector`.
   - KNN search uses:
     - `ORDER BY embedding <=> query_vector LIMIT k`
   - HNSW index added for fast ANN-style retrieval at scale.

4. **Graceful fallback**
   - If LLM calls fail or API key is missing, heuristic extraction/generation still runs.
   - Parse failures are recorded as `parse_failed` rather than silently dropped.

## Pipeline Flow

```mermaid
flowchart TD
    intake[InputIntake] --> parse[MultiFormatParsing]
    parse --> extract[StructuredCandidateExtraction]
    intake --> jd[JDInterpretationAndRubric]
    extract --> embed[ChunkAndEmbed]
    embed --> vector[PgvectorStorageAndKNN]
    jd --> scoring[HybridScoring]
    vector --> scoring
    scoring --> shortlist[ShortlistOutput]
    shortlist --> outreach[PersonalizedOutreach]
    shortlist --> interview[InterviewKitGeneration]
    outreach --> artifacts[ArtifactExport]
    interview --> artifacts
```

## Actual PDFs + RAG

- The pipeline already supports real `.pdf` resumes through `pypdf`.
- Resume text is chunked and embedded, then stored in Postgres `pgvector`.
- For each rubric criterion, top-`k` evidence chunks are retrieved via KNN:
  - `ORDER BY embedding <=> query_vector LIMIT k`
- That retrieval step is the RAG component used before scoring.

## Repository Structure

- `src/ui/streamlit_app.py`: main Streamlit app entrypoint
- `src/pipeline/orchestrator.py`: end-to-end workflow orchestration
- `src/parsers/`: PDF/DOCX/OCR parsers and routing
- `src/extract/`: JD and candidate extraction logic
- `src/retrieval/`: chunking, embeddings, pgvector operations
- `src/scoring/`: rubric builder and candidate scoring
- `src/generation/`: outreach + interview kit generation
- `src/output/`: shortlist writers
- `src/storage/`: persistence layer for run metadata and artifacts
- `config/rubric_defaults.yaml`: configurable weighting defaults

## Setup

1. Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Set environment variables:

```bash
export DATABASE_URL="postgresql://postgres:postgres@localhost:5432/confido_hiring"
export OPENAI_API_KEY="your_key_here"
export PROMPT_SOURCE="langfuse"
export LANGFUSE_PUBLIC_KEY="your_public_key"
export LANGFUSE_SECRET_KEY="your_secret_key"
```

Or copy the template:

```bash
cp .env.example .env
```

3. Ensure Postgres has `pgvector` extension available.

4. Configure Langfuse prompt names:

```bash
export REQUIRE_LANGFUSE_PROMPTS="true"
export CANDIDATE_EXTRACTION_PROMPT_NAME="candidate_extraction_prompt"
export JD_PARSE_PROMPT_NAME="jd_parse_prompt"
export OUTREACH_PROMPT_NAME="outreach_prompt"
export INTERVIEW_PROMPT_NAME="interview_prompt"
```

If `REQUIRE_LANGFUSE_PROMPTS=true`, the run fails fast if any prompt is missing.
Set `REQUIRE_LANGFUSE_PROMPTS=false` to allow fallback files in `src/prompts/`.

## Run (Streamlit UI)

Use this for an end-to-end ChatGPT-style hiring flow with real uploads.

```bash
streamlit run src/ui/streamlit_app.py
```

In the UI:
- upload JD + resumes in the sidebar
- set **Top N candidates** for shortlist and artifact generation
- click **Run Agentic Pipeline**
- after completion, ask follow-up questions in chat input
- choose answer scope: shortlisted-only or all indexed resumes

This UI stores chunks/embeddings in `pgvector` and retrieves top-`k` matches with KNN.
The UI runs in strict mode: `OPENAI_API_KEY` is required and no heuristic fallback is used.

## Demo Script

Run:

```bash
bash scripts/demo_run.sh
```

## Secret Safety Check

Before committing, run:

```bash
bash scripts/check_secrets.sh
```

## Artifacts Produced Per Run

In `outputs/run_<run_id>/`:
- `shortlist.json`
- `shortlist.md`
- `parse_summary.json`
- `trace.jsonl`
- `outreach/*.md`
- `outreach/*.pdf`
- `interview_kits/*.json`
- `interview_kits/*.pdf`

Additionally persisted in Postgres tables:
- `job_runs`
- `candidates`
- `resume_chunks`
- `criterion_scores`
- `outreach_drafts`
- `interview_kits`
- `artifacts`

## Included Sample Data

- Example output artifacts in `sample_output/`

## Langfuse Prompt Templates

Create these prompts in Langfuse with label `production`:

- `candidate_extraction_prompt`
- `jd_parse_prompt`
- `outreach_prompt`
- `interview_prompt`
- `rag_answer_prompt`
- `score_audit_prompt`

Use variables:
- `{{resume_text}}` in candidate extraction
- `{{jd_text}}` in JD parsing
- `{{query}}` and `{{evidence_block}}` in RAG final answer

The pipeline injects those values before model calls.

Recommended: keep these prompts in Langfuse and version them with labels (for example `staging` and `production`).

## Hosting (Render)

This repository now includes:
- `Dockerfile`
- `render.yaml`

Deploy steps:
1. Push this repo to GitHub.
2. In Render, create a new **Blueprint** from the repo.
3. Set required secrets in Render:
   - `DATABASE_URL`
   - `OPENAI_API_KEY`
   - `LANGFUSE_PUBLIC_KEY`
   - `LANGFUSE_SECRET_KEY`
4. Deploy and open the generated web URL.

## What I'd Build Next

- Add PDF-to-image conversion for robust OCR fallback.
- Add evaluator guardrails for hallucination checks per output type.
- Add human review loop for rubric edits and score overrides before outreach.
- Add FastAPI interface + lightweight review dashboard for recruiting teams.

