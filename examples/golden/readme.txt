# Golden Example (ragcitecheck)

This folder contains a tiny, deterministic “golden” dataset for validating and demonstrating ragcitecheck.

## What it includes
- Two run logs: `runs/run1.jsonl` and `runs/run2.jsonl`
- A doc-id alias map: `mappings/docid_map.csv`

## What it demonstrates
- Stable evidence sets (same citations across runs)
- Evidence instability / “flip” behavior (partial overlap or disjoint sets)
- Null citations (empty cited list)
- Duplicate doc IDs in a single query (dedup warning behavior)
- Top-k sensitivity (order changes matter if you use --topk=1)
- Flexible doc entry shapes:
  - strings: `"cited": ["docA", "docB"]`
  - dicts: `"cited": [{"doc_id":"..."}]`
  - nested dicts: `"cited": [{"source":{"doc_id":"..."}}]`
- Canonicalization + aliasing via docid map
- Canonicalization collision (two raw IDs mapping to same canonical)

## Run validate
From repo root:

```bash
ragcitecheck validate \
  --runs examples/golden/runs \
  --out examples/golden/out_validate \
  --docid-map examples/golden/mappings/docid_map.csv

## Run report
ragcitecheck report \
  --runs examples/golden/runs \
  --out examples/golden/out_report \
  --docid-map examples/golden/mappings/docid_map.csv \
  --flip-threshold 0.5

##Top-k demonstration 
This makes q_topk_1 look unstable if only top-1 is used:

ragcitecheck report \
  --runs examples/golden/runs \
  --out examples/golden/out_report_topk1 \
  --docid-map examples/golden/mappings/docid_map.csv \
  --topk 1 \
  --flip-threshold 0.5


##Expected outputs

validation_summary.json written under your --out directory

report artifacts like:

pairwise_config_stability.csv

per_query_stability.csv

instability_examples.md

citation_overlap_hist.png

---

## Optional placeholders (create empty files now if you want)

If you want the `expected/` folder present in git *before* you generate outputs, create these empty files:

- `examples/golden/expected/validation_summary.json`
- `examples/golden/expected/per_query_stability.csv`
- `examples/golden/expected/pairwise_config_stability.csv`
- `examples/golden/expected/instability_examples.md`

(Then later you replace them with real outputs.)

---

If you paste these files and run `validate` + `report`, tell me what `validation_summary.json` warnings you see (if any) and I’ll tell you whether the golden example is hitting every case the way we intended.
::contentReference[oaicite:0]{index=0}
