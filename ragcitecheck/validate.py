from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from ragcitecheck.canonicalize import CanonicalizationReport, Canonicalizer


def _iter_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield i, json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path} line {i}: invalid JSON: {e}") from e

# NEW: span hash keys for doc+span evidence (Tier-1.5)
_DEFAULT_EVIDENCE_KEYS_SPAN = ("span_hash", "spanHash", "spanhash")

_DEFAULT_RUN_ID_KEYS = ("run_id", "config_id", "run", "id")
_DEFAULT_QUERY_ID_KEYS = ("query_id", "qid", "id")
_DEFAULT_DOCS_KEYS = ("cited", "docs", "documents", "retrieved", "contexts")
_DEFAULT_DOC_ID_KEYS = ("doc_id", "document_id", "docid", "id", "source_id")
# Evidence can be compared at doc-level ("doc_id") or chunk/node-level ("chunk_id"/"node_id")
_DEFAULT_EVIDENCE_KEYS_DOC = ("doc_id",)
_DEFAULT_EVIDENCE_KEYS_CHUNK = ("chunk_id", "node_id", "id_", "nodeId")


def _extract_first_str(rec: Dict[str, Any], keys: Tuple[str, ...], what: str) -> str:
    for k in keys:
        if k in rec and rec[k] is not None:
            v = rec[k]
            s = v.strip() if isinstance(v, str) else str(v).strip()
            if s:
                return s
    raise ValueError(f"record missing {what} (tried keys: {list(keys)})")


def _truncate(val: Any, n: int = 160) -> str:
    s = repr(val)
    return s if len(s) <= n else s[: n - 3] + "..."


def _extract_docs_list_with_key(rec: dict, docs_keys: Sequence[str]):
    """
    Extract the docs list by key.
    DO NOT validate contents here — validation happens downstream.
    """
    for k in docs_keys:
        if k in rec:
            v = rec[k]
            if isinstance(v, list):
                return k, v
            raise ValueError(f"record key '{k}' is not a list (type={type(v).__name__})")

    raise ValueError(f"record missing docs list (tried keys: {list(docs_keys)})")


def _deep_find_first_str(obj: Any, keys: Tuple[str, ...], max_depth: int = 3) -> Optional[str]:
    if max_depth < 0:
        return None

    if isinstance(obj, dict):
        for k in keys:
            if k in obj and obj[k] is not None:
                v = obj[k]
                s = v.strip() if isinstance(v, str) else str(v).strip()
                if s:
                    return s
        for v in obj.values():
            found = _deep_find_first_str(v, keys, max_depth=max_depth - 1)
            if found:
                return found

    elif isinstance(obj, list):
        for it in obj:
            found = _deep_find_first_str(it, keys, max_depth=max_depth - 1)
            if found:
                return found

    return None


def _extract_doc_id_flexible(doc_obj: Any, doc_id_keys: Tuple[str, ...], *, docs_key_used: str) -> str:
    if isinstance(doc_obj, str):
        s = doc_obj.strip()
        if not s:
            raise ValueError(f"Empty doc_id string in '{docs_key_used}' docs list.")
        return s

    if isinstance(doc_obj, dict):
        s = _deep_find_first_str(doc_obj, doc_id_keys, max_depth=0)
        if s:
            return s
        s = _deep_find_first_str(doc_obj, doc_id_keys, max_depth=3)
        if s:
            return s
        raise ValueError(
            f"Doc entry dict in '{docs_key_used}' is missing doc_id. "
            f"Tried keys={list(doc_id_keys)}. Entry={_truncate(doc_obj)}"
        )

    raise ValueError(
        f"Doc entry in '{docs_key_used}' must be str or dict "
        f"(got {type(doc_obj).__name__}). Entry={_truncate(doc_obj)}"
    )

def _extract_span_hash_flexible(doc_obj: Any, *, docs_key_used: str) -> str:
    """
    Extracts span_hash from a structured doc entry dict.
    Required for evidence_key='doc_span'.
    """
    if isinstance(doc_obj, str):
        raise ValueError(
            f"evidence_key='doc_span' requires structured dict entries in '{docs_key_used}', "
            f"but got a string entry: {_truncate(doc_obj)}"
        )

    if not isinstance(doc_obj, dict):
        raise ValueError(
            f"Doc entry in '{docs_key_used}' must be str or dict "
            f"(got {type(doc_obj).__name__}). Entry={_truncate(doc_obj)}"
        )

    # Try top-level first, then nested
    s = _deep_find_first_str(doc_obj, _DEFAULT_EVIDENCE_KEYS_SPAN, max_depth=0)
    if s:
        return s
    s = _deep_find_first_str(doc_obj, _DEFAULT_EVIDENCE_KEYS_SPAN, max_depth=3)
    if s:
        return s

    raise ValueError(
        f"Doc entry dict in '{docs_key_used}' is missing span_hash for evidence_key='doc_span'. "
        f"Tried keys={list(_DEFAULT_EVIDENCE_KEYS_SPAN)}. Entry={_truncate(doc_obj)}"
    )
def _extract_evidence_id_flexible(
    d: dict,
    evidence_key: str,
    *,
    doc_id_keys: Sequence[str] = ("doc_id",),
    docs_key_used: Optional[str] = None,
) -> str:
    """
    Extract evidence identity for a single retrieved item.

    evidence_key:
      - "doc"      -> doc_id
      - "doc_span" -> doc_id|span_hash
      - "chunk"    -> best-effort legacy chunk/node id (kept for backward compatibility)
    """

    evidence_key = str(evidence_key).strip().lower()

    if evidence_key == "doc":
        # allow doc_id aliasing via doc_id_keys
        return _extract_doc_id_flexible(d, doc_id_keys=doc_id_keys, docs_key_used=docs_key_used)

    if evidence_key == "doc_span":
        doc_id = _extract_doc_id_flexible(d, doc_id_keys=doc_id_keys, docs_key_used=docs_key_used)
        span_hash = str(d.get("span_hash", "")).strip()
        if not span_hash:
            raise ValueError(f"Missing span_hash in doc_span evidence item (docs_key={docs_key_used})")
        return f"{doc_id}|{span_hash}"

    if evidence_key == "chunk":
        # legacy behavior: try node_id, then chunk_id, then id
        for k in ("node_id", "chunk_id", "id"):
            v = d.get(k)
            if v is not None and str(v).strip():
                return str(v).strip()
        # fall back: if doc_id exists, at least return doc_id
        return _extract_doc_id_flexible(d, doc_id_keys=doc_id_keys, docs_key_used=docs_key_used)

    raise ValueError(f"Invalid evidence_key='{evidence_key}'. Use 'doc', or 'doc_span'.")

@dataclass(frozen=True)
class ValidateOptions:
    allow_missing: bool = False
    run_id_keys: Tuple[str, ...] = _DEFAULT_RUN_ID_KEYS
    query_id_keys: Tuple[str, ...] = _DEFAULT_QUERY_ID_KEYS
    docs_keys: Tuple[str, ...] = _DEFAULT_DOCS_KEYS
    doc_id_keys: Tuple[str, ...] = _DEFAULT_DOC_ID_KEYS

    # NEW: what evidence id to compare across runs
#   - "doc":      doc_id (default)
#   - "doc_span": doc_id::span_hash (frozen design target)
#   - "chunk":    chunk_id/node_id (debug/secondary)    
    evidence_key: str = "doc"

    topk: Optional[int] = None  # truncate docs list to top-k if provided


@dataclass
class RunData:
    run_id: str
    file: str
    q_to_docs: Dict[str, Set[str]]  # doc-level set for each query_id
    null_docs_count: int
    dedup_events: int
    total_queries: int


@dataclass
class ValidationResult:
    runs: Dict[str, RunData]
    query_ids_union: Set[str]
    query_ids_intersection: Set[str]
    warnings: List[str]
    canonicalization: Dict[str, Any]


def validate_runs_folder(
    *,
    runs_dir: Path,
    canonicalizer: Canonicalizer,
    opts: ValidateOptions,
) -> ValidationResult:
    if not runs_dir.exists() or not runs_dir.is_dir():
        raise ValueError(f"--runs must be an existing directory: {runs_dir}")

    run_files = sorted([p for p in runs_dir.iterdir() if p.is_file() and p.suffix.lower() == ".jsonl"])
    if not run_files:
        raise ValueError(f"No .jsonl files found in: {runs_dir}")

    warnings: List[str] = []
    runs: Dict[str, RunData] = {}
    per_run_query_sets: List[Set[str]] = []

    canon_report = CanonicalizationReport()
    all_raw_doc_ids_for_collision_scan: List[str] = []

    for rf in run_files:
        seen_run_ids_in_file: Set[str] = set()
        qset: Set[str] = set()
        q2docs_set: Dict[str, Set[str]] = {}

        total_queries = 0
        null_docs = 0
        dedup_events = 0

        for lineno, rec in _iter_jsonl(rf):
            rid = _extract_first_str(rec, opts.run_id_keys, what="run_id")
            seen_run_ids_in_file.add(rid)

            qid = _extract_first_str(rec, opts.query_id_keys, what="query_id")
            if qid in qset:
                raise ValueError(f"{rf} line {lineno}: duplicate query_id within the same run: {qid}")
            qset.add(qid)
            total_queries += 1

            try:
                docs_key_used, docs_list = _extract_docs_list_with_key(rec, opts.docs_keys)
            except ValueError as e:
                # targeted debug to explain WHY 'retrieved' wasn't found
                if isinstance(rec, dict):
                    raise ValueError(
                    f"{rf} line {lineno}: {e}. Present top-level keys={sorted(list(rec.keys()))}"
                ) from e
                    raise ValueError(
                        f"{rf}  line {lineno}: {e}. Record type={type(rec).__name__}, value={_truncate(rec)}"
                    ) from e

            if opts.topk is not None:
                docs_list = docs_list[: int(opts.topk)]


            raw_doc_ids: List[str] = []
            for d in docs_list:
                raw = _extract_evidence_id_flexible(
                    d,
                    evidence_key=opts.evidence_key,
                    doc_id_keys=opts.doc_id_keys,
                    docs_key_used=docs_key_used,
                )
                raw_doc_ids.append(raw)
                all_raw_doc_ids_for_collision_scan.append(raw)



            if not raw_doc_ids:
                null_docs += 1

            if len(raw_doc_ids) != len(set(raw_doc_ids)):
                dedup_events += 1

            canon_set: Set[str] = set()
            for raw in raw_doc_ids:
                canon_set.add(canonicalizer.canonicalize_doc_id(raw, report=canon_report))
            q2docs_set[qid] = canon_set

        if not seen_run_ids_in_file:
            raise ValueError(f"{rf}: no run_id found (tried keys: {list(opts.run_id_keys)})")
        if len(seen_run_ids_in_file) != 1:
            raise ValueError(
                f"{rf}: expected exactly one run_id per file, found {len(seen_run_ids_in_file)}: "
                f"{sorted(seen_run_ids_in_file)}"
            )

        run_id = next(iter(seen_run_ids_in_file))
        if run_id in runs:
            raise ValueError(f"Duplicate run_id across files: '{run_id}'. run_id must be unique per run file.")

        runs[run_id] = RunData(
            run_id=run_id,
            file=str(rf),
            q_to_docs=q2docs_set,
            null_docs_count=null_docs,
            dedup_events=dedup_events,
            total_queries=total_queries,
        )
        per_run_query_sets.append(qset)

    union = set().union(*(s for s in per_run_query_sets)) if per_run_query_sets else set()
    intersection = set.intersection(*(s for s in per_run_query_sets)) if per_run_query_sets else set()

    if not union:
        raise ValueError("No query_ids found across runs.")
    if not intersection:
        raise ValueError("No overlapping query_ids across runs (cannot compare).")

    if not opts.allow_missing:
        first = per_run_query_sets[0]
        for idx, qset in enumerate(per_run_query_sets[1:], start=1):
            if qset != first:
                only_in_first = len(first - qset)
                only_in_this = len(qset - first)
                raise ValueError(
                    "Query coverage mismatch across runs. "
                    "Use --allow-missing to evaluate on intersection. "
                    f"Diff vs run[0]: only_in_run0={only_in_first}, only_in_run{idx}={only_in_this}."
                )

    run_ids_sorted = sorted(runs.keys())
    for i, rid in enumerate(run_ids_sorted):
        rd = runs[rid]
        if rd.dedup_events:
            warnings.append(
                f"run[{i}] '{rid}' had {rd.dedup_events} queries with duplicate doc_ids in docs list; "
                "deduping to doc-level sets was applied."
            )
        if rd.total_queries:
            null_rate = rd.null_docs_count / rd.total_queries
            if null_rate >= 0.05:
                warnings.append(
                    f"run[{i}] '{rid}' null-citation rate is {null_rate:.1%} "
                    f"({rd.null_docs_count}/{rd.total_queries} queries have empty docs list)."
                )

    unique_raw = sorted(set(all_raw_doc_ids_for_collision_scan))
    collision_report = canonicalizer.detect_collisions(unique_raw, report=None)

    if collision_report.collision_count > 0:
        warnings.append(
            f"Detected {collision_report.collision_count} canonical doc_id collisions "
            "(multiple raw ids map to the same canonical id). Consider refining docid-map or normalization."
        )

    return ValidationResult(
        runs=runs,
        query_ids_union=union,
        query_ids_intersection=intersection,
        warnings=warnings,
        canonicalization={
            "mapped_count": canon_report.mapped_count,
            "unmapped_count": canon_report.unmapped_count,
            "collision_count": collision_report.collision_count,
            "collisions": collision_report.collisions,
        },
    )
