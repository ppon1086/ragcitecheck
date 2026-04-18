# Ragcitecheck instability examples

- Evidence key: **doc**
- Runs: runA, runB
- Queries evaluated: 3
- Flip threshold: J < 0.5
- Min-overlap (stability): 0.5
- allow_missing: False
- topk: None

## q1
- min_overlap_across_pairs: **0.333**
- worst_pair: **runA vs runB**

| run_id | cited_docs (set) |
|---|---|
| runA | d1, d2 |
| runB | d1, d4 |

- worst-pair diff `runA` → `runB`: J=0.333, flip=YES, +[d4] -[d2]

## q2
- min_overlap_across_pairs: **1.000**
- worst_pair: **N/A**

| run_id | cited_docs (set) |
|---|---|
| runA | d3 |
| runB | d3 |


## q3
- min_overlap_across_pairs: **1.000**
- worst_pair: **N/A**

| run_id | cited_docs (set) |
|---|---|
| runA | (empty) |
| runB | (empty) |

