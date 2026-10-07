# NOTES FOR COPILOT

## Project Scope
- Streamlit MVP for clinician decision support.
- No model retraining in app runtime.
- Artifact-first inference with clear missing-file behavior.

## Architecture Rules
- Keep UI code in `src/pages` and `src/components` only.
- Keep inference logic in `src/pipelines`, `src/preprocessing`, `src/features`.
- Use `config/manifest.json` for all artifact and static paths.
- Keep rule-based explanation deterministic (no LLM/API calls).

## Safety
- Every page must show the visible disclaimer.
- Do not present outputs as confirmed diagnoses.
