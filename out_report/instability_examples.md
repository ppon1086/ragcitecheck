# Ragcitecheck instability examples

- Runs: A, B
- Queries evaluated: 2
- Major flip threshold: J < 0.5

## Q1
- min_overlap: **0.500**
- worst_pair: **A vs B**

| run_id | cited_docs (doc_id set) |
|---|---|
| A | doc1 |
| B | doc1, doc2 |

- vs baseline `A` → `B`: J=0.500, flip=no, +[doc2] -[]

## Q2
- min_overlap: **1.000**
- worst_pair: **N/A**

| run_id | cited_docs (doc_id set) |
|---|---|
| A | (empty) |
| B | (empty) |

- vs baseline `A` → `B`: J=1.000, flip=no, +[] -[]
