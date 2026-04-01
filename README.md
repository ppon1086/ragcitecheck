# ragcitecheck

**ragcitecheck** is a lightweight, integration-friendly tool for measuring **citation and evidence stability** in Retrieval-Augmented Generation (RAG) pipelines.

It helps teams answer a practical question:

> If I rerun the same RAG system with a different chunk size, top-k, overlap, retriever setting, or pipeline version, do I still get the **same evidence**?

This is useful for:

- **RAG evaluators** who want more than answer-level metrics
- **framework maintainers** who want evidence-level regression checks
- **LLMOps / observability tools** that need citation stability diagnostics
- **researchers** studying retrieval jitter, provenance drift, or auditability
- **product teams** building systems where evidence consistency matters

---

## Why this exists

Most RAG evaluation focuses on:
- answer correctness
- faithfulness / groundedness
- retrieval recall
- latency / cost

Those are useful, but they miss an important reliability question:

**Is the cited evidence stable across runs?**

A system may produce similar answers while silently changing:
- which documents it cites
- which spans inside those documents it relies on
- how often evidence disappears entirely

`ragcitecheck` is built to make that visible.

---

## What it does

Given one or more run logs in JSONL format, `ragcitecheck` can:

- validate run structure with flexible input key aliases
- compare **document-level** evidence stability
- compare **document + span-level** evidence stability
- compute pairwise overlap metrics across runs
- generate per-query instability summaries
- surface examples of unstable evidence
- report null-evidence patterns that can hide instability

In short: it is a **post-hoc evidence stability checker** for RAG runs.

---

## Core features

- **Document-level stability**
  - Compare evidence sets using canonicalized `doc_id`s

- **Span-level stability**
  - Compare evidence at `(doc_id, span_hash)` level when span text is available

- **Flexible schema handling**
  - Supports common field aliases for run IDs, query IDs, document lists, and document IDs

- **CLI-first workflow**
  - Easy to add to notebooks, scripts, CI jobs, or evaluation pipelines

- **Framework-friendly design**
  - Can be used with exported logs from any RAG stack
  - Includes adapter / harness utilities for easier integration

---

## Why other repos may want to integrate it

If you maintain a RAG framework, evaluation toolkit, or LLMOps repo, `ragcitecheck` can add a capability many tools do not report directly:

### 1. Evidence regression checks
Detect whether a retrieval or citation change happened after:
- retriever updates
- reranker changes
- chunking changes
- prompt changes
- pipeline refactors

### 2. Auditability signals
Measure whether the same query keeps relying on the same evidence over time.

### 3. Better failure analysis
Separate:
- answer drift
- evidence drift
- null-evidence inflation
- doc-stable but span-unstable behavior

### 4. Lightweight interoperability
It does not require owning the full RAG stack. If a project can export run logs, it can use `ragcitecheck`.

---

## Installation

### Local editable install

```bash
pip install -e .
```

### Or standard install after cloning

```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
pip install -e .
```

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

---

## Quickstart

### Validate a run directory

```bash
python -m ragcitecheck.cli validate --runs ./tests/fixtures/runmin --out ./out_check
```

### Generate a document-level stability report

```bash
python -m ragcitecheck.cli report --runs ./tests/fixtures/runmin --out ./out_report_doc --evidence-key doc
```

### Generate a document + span-level stability report

```bash
python -m ragcitecheck.cli report --runs ./tests/fixtures/runmin --out ./out_report_span --evidence-key doc_span
```

---

## Minimal input format

Each run should be JSONL with one record per query.

### Document-level example

```json
{"run_id":"runA","query_id":"q1","docs":[{"doc_id":"D1"},{"doc_id":"D2"}]}
{"run_id":"runA","query_id":"q2","docs":[{"doc_id":"D3"}]}
```

### Span-level example

```json
{"run_id":"runA","query_id":"q1","docs":[{"doc_id":"D1","span_text":"Paris is the capital of France.","span_hash":"abc123"}]}
{"run_id":"runA","query_id":"q2","docs":[{"doc_id":"D3","span_text":"The study was published in 2021.","span_hash":"def456"}]}
```

---

## Supported aliases

### Run ID keys
- `run_id`
- `runId`
- `config_id`

### Query ID keys
- `query_id`
- `qid`
- `id`

### Document list keys
- `docs`
- `cited`
- `retrieved`
- `contexts`
- `documents`

### Document ID keys
- `doc_id`
- `document_id`
- `docid`
- `id`
- `source_id`

This makes it easier to integrate with logs produced by different pipelines.

---

## Typical outputs

Depending on mode and available evidence, `ragcitecheck` may produce:

- `validation_summary.json`
- `run_quality.csv`
- `pairwise_config_stability.csv`
- `per_query_stability.csv`
- `instability_examples.md`
- `report_meta.json`

These outputs are meant to be easy to:
- inspect manually
- check into experiment folders
- attach to CI artifacts
- feed into larger reporting pipelines

---

## Example use cases

### RAG framework maintainer
“After changing chunk size defaults, did citation evidence become less stable?”

### Evaluation toolkit maintainer
“We already measure answer faithfulness. Can we also measure whether cited evidence itself is consistent?”

### Enterprise RAG team
“We need an audit-friendly signal showing whether evidence provenance shifts across releases.”

### Research workflow
“We want to compare document-level stability versus span-level stability across retrieval settings.”

---

## Integration patterns

`ragcitecheck` is easiest to adopt in one of three ways.

### 1. Offline post-processing
A pipeline writes run logs as JSONL. `ragcitecheck` validates and compares them afterward.

Best when:
- you already have experiment outputs
- you want low-friction adoption
- you do not want to change core inference code

### 2. Harness-based logging
Use the harness utilities to write run records in a consistent format during experiments.

Best when:
- you want repeatable internal evaluation
- you want cleaner provenance logs
- you plan to compare many configs

### 3. Framework adapter integration
Use or extend an adapter to export evidence from a RAG framework into `ragcitecheck` format.

Best when:
- you want deeper integration in another repo
- you want to offer stability checks as a built-in feature
- you want users to run evidence diagnostics with minimal setup

---

## What makes this useful for adoption

This project is intentionally scoped to a specific gap:

**evidence stability**, not general-purpose RAG benchmarking.

That makes it easier to plug into:
- RAG frameworks
- LLM evaluation suites
- citation-checking tools
- observability / regression pipelines
- “awesome lists” covering RAG evaluation, LLMOps, or responsible AI tooling

It is especially relevant for projects interested in:
- provenance
- reproducibility
- auditability
- retrieval jitter
- citation robustness

---

## Current design principles

- **Small surface area**
  - easier to understand and integrate

- **Framework-agnostic data model**
  - works from exported logs, not just one stack

- **Post-hoc friendly**
  - useful even if you cannot modify the original RAG system

- **Evidence-first evaluation**
  - focused on what was cited, not only what was answered

---

## Limitations

At the moment, `ragcitecheck` is intentionally narrow.

It does **not** try to replace:
- answer correctness evaluation
- hallucination detection
- retrieval benchmarking
- end-to-end RAG orchestration

Instead, it complements those layers by adding evidence stability diagnostics.

---

## Near-term roadmap

Planned improvements that would make the project easier to adopt:

- cleaner example datasets
- more polished CLI help and docs
- direct integration examples for popular RAG stacks
- richer adapters
- stronger tests and CI coverage
- release packaging and versioned artifacts

---

## Contributing

Contributions are welcome, especially around:

- adapters for other RAG frameworks
- example integrations
- CLI polish
- report formatting improvements
- tests and fixture design
- stability diagnostics for real-world pipelines

If you are interested in integrating `ragcitecheck` into another repo, an issue or PR describing the target workflow would be very helpful.

---

## Good fit for awesome lists

`ragcitecheck` is best described as:

- **RAG evidence stability checker**
- **citation stability diagnostic tool for RAG**
- **post-hoc provenance stability evaluator**
- **retrieval jitter / evidence drift analysis utility**

### Short awesome-list blurb

> A lightweight tool for measuring document-level and span-level evidence stability in RAG pipelines across runs, configs, and retrieval settings.

---

## Citation / research context

This tool is also used in experimental workflows studying evidence stability and retrieval jitter in RAG systems.  
But the repository is intended to be useful even outside the original research context.

The goal is for the tool to stand on its own as a reusable component for:
- evaluation
- integration
- diagnostics
- regression analysis

---

## License

See the `LICENSE` file in this repository.
