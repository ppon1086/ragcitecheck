# ragcitecheck

**ragcitecheck** is a lightweight tool for measuring **citation and evidence stability** in Retrieval-Augmented Generation (RAG) pipelines across repeated runs and configuration changes.

It helps answer a practical question:

> If I rerun the same RAG pipeline with a different retriever, chunk size, top-k, overlap, or prompt setting, do I still get the **same evidence**?

## Why this matters

Most RAG evaluation focuses on answer correctness, groundedness, retrieval recall, latency, or cost.

Those are useful, but they miss an important reliability question:

**Is the cited evidence stable across runs?**

A system may produce similar answers while silently changing:
- which documents it cites
- which spans inside documents it relies on
- how often evidence disappears entirely

`ragcitecheck` is designed to make that visible.

## What it does

Given one or more run logs in JSONL format, `ragcitecheck` can:

- validate run structure with flexible key aliases
- compare **document-level** evidence stability
- compare **document + span-level** evidence stability
- compute pairwise overlap metrics across runs
- generate per-query instability summaries
- surface examples of unstable evidence
- report null-evidence patterns that can hide instability

## Who it is for

`ragcitecheck` is useful for:

- **RAG evaluators** who want more than answer-level metrics
- **framework maintainers** who want evidence-level regression checks
- **LLMOps / observability teams** tracking provenance drift
- **researchers** studying retrieval jitter or citation stability
- **product teams** where evidence consistency matters

## Installation

### Windows

```bash
py -3.13 -m venv .venv313
.venv313\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .
```

### macOS / Linux

```bash
python3.13 -m venv .venv313
source .venv313/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .
```

## Quickstart

### Validate run logs

```bash
python -m ragcitecheck.cli validate --runs ./tests/fixtures/runs_min --out ./out_check
```

### Generate a document-level report

```bash
python -m ragcitecheck.cli report --runs ./tests/fixtures/runs_min --out ./out_report_doc --evidence-key doc
```

### Generate a document+span-level report

```bash
python -m ragcitecheck.cli report --runs ./tests/fixtures/runs_min --out ./out_report_span --evidence-key doc_span
```

## Minimal input format

Each run is stored as JSONL with one record per query.

### Document-level example

```json
{"run_id":"runA","query_id":"q1","retrieved":[{"doc_id":"D1"},{"doc_id":"D2"}]}
{"run_id":"runA","query_id":"q2","retrieved":[{"doc_id":"D3"}]}
```

### Document+span-level example

```json
{"run_id":"runA","query_id":"q1","retrieved":[{"doc_id":"D1","span_hash":"s1"},{"doc_id":"D2","span_hash":"s2"}]}
{"run_id":"runA","query_id":"q2","retrieved":[{"doc_id":"D3","span_hash":"s3"}]}
```

## Supported aliases

### Run ID keys
- `run_id`
- `runId`
- `config_id`
- `run`
- `id`

### Query ID keys
- `query_id`
- `qid`
- `id`

### Evidence list keys
- `cited`
- `docs`
- `documents`
- `retrieved`
- `contexts`

### Document ID keys
- `doc_id`
- `document_id`
- `docid`
- `id`
- `source_id`

## Common options

- `--allow-missing`  
  Evaluate only shared queries across runs

- `--docid-map path.csv`  
  Canonicalize raw document IDs using a `raw,canonical` mapping

- `--case-sensitive`  
  Disable lowercasing during canonicalization

- `--collapse-internal-whitespace`  
  Normalize repeated whitespace before comparison

## Outputs

Validation outputs:
- `validation_summary.json`

Report outputs may include:
- `run_quality.csv`
- `pairwise_config_stability.csv`
- `per_query_stability.csv`
- `instability_examples.md`
- `citation_overlap_hist.png`
- `report_meta.json`

## Interpreting results

- **Average overlap close to 1.0**  
  Evidence is relatively stable across runs

- **High flip rate**  
  Queries frequently undergo major evidence changes

- **High null-evidence rate**  
  Many queries return empty evidence; this can make a system look stable while reducing usefulness

- **Large doc vs span gap**  
  The same documents may be cited across runs while cited spans still change substantially

## Why other repos may integrate it

`ragcitecheck` adds a capability that many RAG evaluation stacks do not report directly:

- evidence regression checks after retriever or chunking changes
- provenance drift diagnostics across releases
- separation of answer drift vs evidence drift
- lightweight post-hoc comparison from exported logs

---`ragcitecheck` is best described as a:

- **RAG evidence stability checker**
- **citation stability diagnostic tool**
- **post-hoc provenance stability evaluator**
- **evidence drift analysis utility for RAG**

## Citation / research context

This tool is also used in experimental workflows studying evidence stability and retrieval jitter in RAG systems.  
But the repository is intended to be useful even outside the original research context.

This project is under active cleanup and packaging improvement.

Current focus:
- cleaner package structure
- better CLI/report wiring
- stronger examples and tests
- easier integration into external RAG tooling

## License

See the `LICENSE` file in this repository.
