# ragcitecheck
Document-level citation stability diagnostics for RAG systems

Ragcitecheck is a lightweight diagnostic tool to measure
document-level citation stability in Retrieval-Augmented
Generation (RAG) systems under configuration changes.

## Status
This is an early (v0.1) development release.

## Installation
```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt

## Validate Runs
python -m ragcitecheck.cli validate \--runs ./tests/fixtures/runs_min \--out ./out_check

  ##Generate reports
  python -m ragcitecheck.cli report \--runs ./tests/fixtures/runs_min \--out ./out_check

    ##Sample 
  {"run_id":"runA","query_id":"q1","docs":[{"doc_id":"D1"},{"doc_id":"D2"}]}

#Supported alias
run id keys: run_id | config_id | run | id

query id keys: query_id | qid | id

docs list keys: cited | docs | documents | retrieved | contexts

doc id keys: doc_id | document_id | docid | id | source_id

##Common options
--allow-missing (intersection mode)

--docid-map path.csv (raw,canonical)

--case-sensitive

--collapse-internal-whitespace

python -m ragcitecheck.cli report --runs .\runs --out .\out \
  --allow-missing --docid-map .\docid_map.csv --collapse-internal-whitespace


##Outputs
Validate outputs

validation_summary.json

Report outputs

run_quality.csv

pairwise_config_stability.csv

per_query_stability.csv

instability_examples.md

citation_overlap_hist.png

report_meta.json

##Interpreting results
avg_overlap close to 1.0 = stable citations across runs

flip_rate high = frequent major citation changes

null-citation rate high = many queries returning empty citations (can look “stable” but unhelpful)

##Ytoublrshooting
If python isn’t found, use py -3.13 -m …

Avoid Python 3.14 pre-release; use Python 3.13 (or 3.12)
##Example
py -3.13 -m ragcitecheck.cli validate --runs .\tests\fixtures\runs_min --out .\out_check
