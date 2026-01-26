from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from ragcitecheck.canonicalize import CanonicalizationReport, Canonicalizer


# -------------------------
# JSONL reader
# -------------------------

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


# -------------------------
# Field extractors (adoption-first defaults)
# -------------------------

_DEFAULT_RUN_ID_KEYS = ("run_id", "config_id", "run", "id")
_DEFAULT_QUERY_ID_KEYS = ("query_id", "qid", "id")
_DEFAULT_DOCS_KEYS = ("cited", "docs", "documents", "retrieved", "contexts")
_DEFAULT_DOC_ID_KEYS = ("doc_id", "document_id", "docid", "id", "source_id")


def _extract_first_str(rec: Dict[str, Any], keys: Tuple[str, ...], what: str) -> str:
    for k in keys:
        if k in rec and rec[k] is not None:
            v = rec[k]
            if isinstance(v, str):
                s = v.strip()
            else:
                s = str(v).strip()
            if s:
                return s
    raise ValueError(f"record missing {what} (tried keys: {list(keys)})")


def _truncate(val: Any, n: int = 160) -> str:
    s = repr(val)
    return s if len(s) <= n else s[: n - 3] + "..."


def _extract_docs_list_with_key(rec: Dict[str, Any], docs_keys: Tuple[str, ...]) -> Tuple[str, List[Any]]:
    """
    Returns (docs_key_used, docs_list).
    """
    for k in docs_keys:
        if k in rec and rec[k] is not None:
            v = rec[k]
            if not isinstance(v, list):
                raise ValueError(f"{k} must be a list (got {type(v).__name__})")
            return k, v
    raise ValueError(f"record missing docs list (tried keys: {list(docs_keys)})")


def _deep_find_first_str(obj: Any, keys: Tuple[str, ...], max_depth: int = 3) -> Optional[str]:
    """
    Searches nested dict/list structures for the first non-empty string-like value
    under any of the provided keys. Depth-limited to avoid pathological logs.
    """
    if max_depth < 0:
        return None

    if isinstance(obj, dict):
        # direct hit first
        for k in keys:
            if k in obj and obj[k] is not None:
                v = obj[k]
                if isinstance(v, str):
                    s = v.strip()
                    if s:
                        return s
                else:
                    s = str(v).strip()
                    if s:
                        return s

        # nested search
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
    """
    Supports:
      - doc entry as str -> doc_id directly
      - doc entry as dict -> try keys + nested lookup
    """
    if isinstance(doc_obj, str):
        s = doc_obj.strip()
        if not s:
            raise ValueError(f"Empty doc_id string in '{docs_key_used}' docs list.")
        return s

    if isinstance(doc_obj, dict):
        # first: top-level keys (fast path)
        s = _deep_find_first_str(doc_obj, doc_id_keys, max_depth=0)
        if s:
            return s

        # second: nested lookup (common shapes: source.doc_id, metadata.document_id, etc.)
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


# -------------------------
# Public result types
# -------------------------

@dataclass(frozen=True)
class ValidateOptions:
    """
    Controls schema aliasing + coverage behavior.
    """
    allow_missing: bool = False
    run_id_keys: Tuple[str, ...] = _DEFAULT_RUN_ID_KEYS
    query_id_keys: Tuple[str, ...] = _DEFAULT_QUERY_ID_KEYS
    docs_keys: Tuple[str, ...] = _DEFAULT_DOCS_KEYS
    doc_id_keys: Tuple[str, ...] = _DEFAULT_DOC_ID_KEYS
    topk: Optional[int] = None  # truncate docs list to top-k if provided


@dataclass
class RunData:
    """
    Parsed run content, already canonicalized at doc_id level.
    """
    run_id: str
    file: str
    q_to_docs: Dict[str, Set[str]]  # doc-level set for each query_id
    null_docs_count: int
    dedup_events: int
    total_queries: int


@dataclass
class ValidationResult:
    runs: Dict[str, RunData]  # run_id -> RunData
    query_ids_union: Set[str]
    query_ids_intersection: Set[str]
    warnings: List[str]
    canonicalization: Dict[str, Any]


# -------------------------
# Main validation
# -------------------------

def validate_runs_folder(
    *,
    runs_dir: Path,
    canonicalizer: Canonicalizer,
    allow_missing: bool = False,   
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

    # For canonicalization diagnostics
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

            docs_key_used, docs_list = _extract_docs_list_with_key(rec, opts.docs_keys)

            raw_doc_ids: List[str] = []
            for d in docs_list:
                raw = _extract_doc_id_flexible(d, opts.doc_id_keys, docs_key_used=docs_key_used)
                raw_doc_ids.append(raw)
                all_raw_doc_ids_for_collision_scan.append(raw)

            if len(raw_doc_ids) != len(set(raw_doc_ids)):
                dedup_events += 1

            # canonicalize + set
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

    # warnings
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

    # collision scan (after canonicalization)
    unique_raw_ids_for_collision_scan = sorted(set(all_raw_doc_ids_for_collision_scan))

    collision_report = canonicalizer.detect_collisions(unique_raw_ids_for_collision_scan,report=None)
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
