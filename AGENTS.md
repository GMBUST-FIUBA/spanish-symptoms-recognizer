# AGENTS.md

## Introduction

This repository implements a pipeline that detects clinical symptoms/phenotypes in Spanish-language clinical records (free text) and maps them to standardized codes in the Human Phenotype Ontology (HPO). It is the code base for a final-year project at the University of Buenos Aires on detection of genetic and rare diseases.

The pipeline has three stages, applied in order:

1. **Text parsing** (`symptoms_recognizer/text_parser/`) — splits a raw clinical record into sentences, chunks, sections, or leaves it as full text, depending on the strategy configured. This affects how much context the NER stage sees at once.
2. **NER / phenotype detection** (`symptoms_recognizer/ner_model/`) — extracts symptom/phenotype mentions (plus their surrounding context) from the parsed text. Can run a local fine-tuned/HF model or delegate to an LLM API (OpenAI, Anthropic, Gemini). When an LLM is used, its prompt also does the phenotype-name normalization (see "Known gaps" below).
3. **Ontology mapping** (`symptoms_recognizer/mapper/`) — maps each phenotype mention to an HPO code, using semantic similarity models (ClinLinker, SapBERT, RoBERTa-biomedical) and/or an LLM as a RAG-style re-ranker over the top-K closest HPO candidates.

`symptoms_recognizer/symptom_recognizer.py` (`PhenotypesRecognizer`) orchestrates these stages end to end via `.recognize()`, `.map()`, and `.scan()`.

## Known gaps

- `symptoms_recognizer/entity-normalization/` (`PhenotypesNormalizer` and its `AbbreviationsFilter` / `GramaticalCorrectorFilter`) is **dead code**: `_get_filters_pipeline()` returns `[]` and both filters are stubs that return `[]`, and `symptom_recognizer.py` never imports or calls this module. It is not part of the running pipeline. In practice, phenotype-name normalization is done by instructing the NER LLM prompt directly (e.g. the "NORMALIZACIÓN EXTREMA" rule in `symptoms_recognizer/tests/compare_models.py`). Don't assume this module runs, and don't wire it back in unless explicitly asked.

## Running the code

- The Python virtual environment lives in `.env/` at the repo root — despite the name, it is a venv directory, not a dotenv file. Activate it with `source .env/bin/activate`.
- Dependencies are pinned in `requirements.txt`.
- Running LLM-backed configurations (NER or mapping stage using `phenotypes_model_type="api"` /
  `ner_api_provider` / `map_api_provider`) requires API keys set as environment variables, read directly by each provider's SDK (no `.env` file is loaded by the code):
  - `OPENAI_API_KEY` for the `openai` provider
  - `ANTHROPIC_API_KEY` for the `anthropic` provider
  - `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) for the `gemini` provider

## Testing

E2E tests are run by executing `symptoms_recognizer/tests/compare_models.py`:

```
python symptoms_recognizer/tests/compare_models.py
```

This script builds one or more `PhenotypesRecognizer` configurations (see the `get_*_test_config()` generator functions), runs each one against the evaluation dataset via `Evaluator`, and prints/saves a comparison table scored on NER F1 and mapping (HPO code) F1.

- `symptoms_recognizer/tests/dataset/` (`historia_1..8.csv/.txt`) is the ground-truth evaluation set. These files back every comparison in `symptoms_recognizer/tests/README.md` — do not edit them without checking whether historical results need to be re-run, since changing them silently invalidates past comparisons.
- `symptoms_recognizer/tests/reports/` holds per-configuration detailed output, named as:
  `NER__<ner-model>_<ner-prompt>___MAP__<map-provider>__<map-model>__{MAP,NER}.txt`. Follow this naming convention when adding a new tested configuration so reports stay comparable/sortable.
- `symptoms_recognizer/tests/README.md` is an append-only log of experiment results in chronological order (models compared, hypotheses, and their outcomes). New experiments should be added as new sections; existing sections should not be rewritten or removed, since they document why prior decisions (aggregation strategy, parsing style, prompt wording, RAG top-K, etc.) were made.
