# Ragcitecheck instability examples

- Runs: A, B
- Queries evaluated: 2
- Flip threshold: J < 0.5
- Min-overlap (stability): 0.5
- allow_missing: False
- topk: None

## Q1
- min_overlap_across_pairs: **0.500**
- worst_pair: **A vs B**

| run_id | cited_docs (doc_id set) |
|---|---|
| A | doc1 |
| B | doc1, doc2 |

- worst-pair diff `A` → `B`: J=0.500, flip=no, +[doc2] -[]

## Q2
- min_overlap_across_pairs: **1.000**
- worst_pair: **N/A**

| run_id | cited_docs (doc_id set) |
|---|---|
| A | (empty) |
| B | (empty) |

