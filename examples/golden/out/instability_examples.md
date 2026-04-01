# Ragcitecheck instability examples

- Runs: golden_run_1, golden_run_2
- Queries evaluated: 11
- Flip threshold: J < 0.5
- Min-overlap (stability): 0.5
- allow_missing: False
- topk: None

## q_collision_1
- min_overlap_across_pairs: **0.000**
- worst_pair: **golden_run_1 vs golden_run_2**

| run_id | cited_docs (doc_id set) |
|---|---|
| golden_run_1 | docz |
| golden_run_2 | docy |

- worst-pair diff `golden_run_1` → `golden_run_2`: J=0.000, flip=YES, +[docy] -[docz]

## q_flip_2
- min_overlap_across_pairs: **0.000**
- worst_pair: **golden_run_1 vs golden_run_2**

| run_id | cited_docs (doc_id set) |
|---|---|
| golden_run_1 | doce |
| golden_run_2 | docy |

- worst-pair diff `golden_run_1` → `golden_run_2`: J=0.000, flip=YES, +[docy] -[doce]

## q_flip_1
- min_overlap_across_pairs: **0.333**
- worst_pair: **golden_run_1 vs golden_run_2**

| run_id | cited_docs (doc_id set) |
|---|---|
| golden_run_1 | docc, docd |
| golden_run_2 | docc, docz |

- worst-pair diff `golden_run_1` → `golden_run_2`: J=0.333, flip=YES, +[docz] -[docd]

## q_dedup_1
- min_overlap_across_pairs: **1.000**
- worst_pair: **N/A**

| run_id | cited_docs (doc_id set) |
|---|---|
| golden_run_1 | docf, docg |
| golden_run_2 | docf, docg |


## q_map_1
- min_overlap_across_pairs: **1.000**
- worst_pair: **N/A**

| run_id | cited_docs (doc_id set) |
|---|---|
| golden_run_1 | docm |
| golden_run_2 | docm |


## q_null_1
- min_overlap_across_pairs: **1.000**
- worst_pair: **N/A**

| run_id | cited_docs (doc_id set) |
|---|---|
| golden_run_1 | (empty) |
| golden_run_2 | (empty) |


## q_shape_dict_1
- min_overlap_across_pairs: **1.000**
- worst_pair: **N/A**

| run_id | cited_docs (doc_id set) |
|---|---|
| golden_run_1 | dock |
| golden_run_2 | dock |


## q_shape_nested_1
- min_overlap_across_pairs: **1.000**
- worst_pair: **N/A**

| run_id | cited_docs (doc_id set) |
|---|---|
| golden_run_1 | docl |
| golden_run_2 | docl |


## q_shape_str_1
- min_overlap_across_pairs: **1.000**
- worst_pair: **N/A**

| run_id | cited_docs (doc_id set) |
|---|---|
| golden_run_1 | docj |
| golden_run_2 | docj |


## q_stable_1
- min_overlap_across_pairs: **1.000**
- worst_pair: **N/A**

| run_id | cited_docs (doc_id set) |
|---|---|
| golden_run_1 | doca, docb |
| golden_run_2 | doca, docb |


## q_topk_1
- min_overlap_across_pairs: **1.000**
- worst_pair: **N/A**

| run_id | cited_docs (doc_id set) |
|---|---|
| golden_run_1 | doch, doci |
| golden_run_2 | doch, doci |

