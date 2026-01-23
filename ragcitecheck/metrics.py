from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


def jaccard(a: Set[str], b: Set[str]) -> float:
    """
    Jaccard overlap between two sets.

    Edge case:
      - If both are empty, return 1.0 (stable "no citations" state).
    """
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    inter = a & b
    return len(inter) / len(union)


def is_major_flip(j: float, flip_threshold: float) -> bool:
    """
    Major flip indicator.
      - Flip if J < threshold (strict).
    """
    return j < flip_threshold


@dataclass(frozen=True)
class PairwiseSummary:
    config_a: str
    config_b: str
    avg_overlap: float
    flip_rate: float
    # "null citation" diagnostics
    null_rate_a: float
    null_rate_b: float
    null_loss_rate_a_to_b: float  # A non-empty -> B empty
    null_gain_rate_a_to_b: float  # A empty -> B non-empty
    # optional primary doc stability (only if you define "primary doc")
    top1_doc_stability: Optional[float] = None


@dataclass(frozen=True)
class PerQueryInstability:
    query_id: str
    min_overlap: float
    worst_pair: str
    unstable_flag: bool


@dataclass(frozen=True)
class RunQuality:
    run_id: str
    citation_rate: float        # % queries with |C_doc| > 0
    null_rate: float            # % queries with |C_doc| == 0
    avg_cited_docs: float       # mean |C_doc|
    median_cited_docs: float    # median |C_doc|
    p95_cited_docs: float       # 95th percentile |C_doc|


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _median_int(xs: Sequence[int]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return float(s[mid])
    return (s[mid - 1] + s[mid]) / 2.0


def _percentile_int(xs: Sequence[int], p: float) -> float:
    """
    Simple nearest-rank percentile for ints.
    p in [0, 100]
    """
    if not xs:
        return 0.0
    s = sorted(xs)
    if p <= 0:
        return float(s[0])
    if p >= 100:
        return float(s[-1])
    k = int(round((p / 100.0) * (len(s) - 1)))
    return float(s[k])


def compute_run_quality(
    runs: Dict[str, Dict[str, Set[str]]],
    query_ids: Sequence[str],
) -> List[RunQuality]:
    """
    Summarize per-run citation behavior (helps interpret "stable but empty" situations).
    """
    out: List[RunQuality] = []
    for rid, qmap in runs.items():
        sizes: List[int] = []
        non_empty = 0
        total = 0
        for qid in query_ids:
            if qid not in qmap:
                continue
            total += 1
            s = qmap[qid]
            sizes.append(len(s))
            if s:
                non_empty += 1

        citation_rate = (non_empty / total) if total else 0.0
        null_rate = 1.0 - citation_rate if total else 0.0
        avg_sz = (sum(sizes) / len(sizes)) if sizes else 0.0

        out.append(
            RunQuality(
                run_id=rid,
                citation_rate=citation_rate,
                null_rate=null_rate,
                avg_cited_docs=avg_sz,
                median_cited_docs=_median_int(sizes),
                p95_cited_docs=_percentile_int(sizes, 95.0),
            )
        )
    return out


def compute_pairwise_summaries(
    runs: Dict[str, Dict[str, Set[str]]],
    query_ids: Sequence[str],
    flip_threshold: float,
    baseline: Optional[str] = None,
    compute_top1: bool = False,
) -> Tuple[List[PairwiseSummary], Dict[Tuple[str, str, str], float]]:
    """
    Computes pairwise summary stats and also returns per-query per-pair Jaccards
    for plotting and per-query global instability.

    Returns:
      - pairwise summaries
      - jaccard_cache[(runA, runB, query_id)] = J
        (runA, runB) stored in lexicographic order (A < B).
    """
    run_ids = sorted(runs.keys())
    pairs: List[Tuple[str, str]] = []

    if baseline:
        if baseline not in runs:
            raise ValueError(f"--baseline '{baseline}' not found among runs: {run_ids}")
        for rid in run_ids:
            if rid == baseline:
                continue
            a, b = sorted([baseline, rid])
            pairs.append((a, b))
    else:
        for i in range(len(run_ids)):
            for j in range(i + 1, len(run_ids)):
                pairs.append((run_ids[i], run_ids[j]))

    jaccard_cache: Dict[Tuple[str, str, str], float] = {}
    summaries: List[PairwiseSummary] = []

    for a, b in pairs:
        js: List[float] = []
        flips: List[float] = []

        # null diagnostics
        null_a = 0
        null_b = 0
        null_loss = 0
        null_gain = 0
        top1_match = 0
        top1_total = 0

        for qid in query_ids:
            sa = runs[a].get(qid, set())
            sb = runs[b].get(qid, set())

            if not sa:
                null_a += 1
            if not sb:
                null_b += 1
            if sa and not sb:
                null_loss += 1
            if not sa and sb:
                null_gain += 1

            j = jaccard(sa, sb)
            jaccard_cache[(a, b, qid)] = j
            js.append(j)
            flips.append(1.0 if is_major_flip(j, flip_threshold) else 0.0)

            if compute_top1:
                # "primary doc" = first in sorted order for determinism in v0.1
                # (If you later log explicit primary, replace this.)
                top1_total += 1
                pa = min(sa) if sa else None
                pb = min(sb) if sb else None
                if pa == pb and pa is not None:
                    top1_match += 1
                elif pa is None and pb is None:
                    # if both have no primary, count as match? For stability, yes.
                    top1_match += 1

        n = len(query_ids) if query_ids else 0
        avg_overlap = _mean(js)
        flip_rate = _mean(flips)
        null_rate_a = (null_a / n) if n else 0.0
        null_rate_b = (null_b / n) if n else 0.0
        null_loss_rate = (null_loss / n) if n else 0.0
        null_gain_rate = (null_gain / n) if n else 0.0

        top1 = (top1_match / top1_total) if (compute_top1 and top1_total) else None

        summaries.append(
            PairwiseSummary(
                config_a=a,
                config_b=b,
                avg_overlap=avg_overlap,
                flip_rate=flip_rate,
                null_rate_a=null_rate_a,
                null_rate_b=null_rate_b,
                null_loss_rate_a_to_b=null_loss_rate,
                null_gain_rate_a_to_b=null_gain_rate,
                top1_doc_stability=top1,
            )
        )

    return summaries, jaccard_cache


def compute_per_query_instability(
    runs: Dict[str, Dict[str, Set[str]]],
    query_ids: Sequence[str],
    flip_threshold: float,
    jaccard_cache: Dict[Tuple[str, str, str], float],
) -> List[PerQueryInstability]:
    """
    For each query, compute worst-case overlap across all run pairs.
    """
    run_ids = sorted(runs.keys())
    out: List[PerQueryInstability] = []

    for qid in query_ids:
        min_j = 1.0
        worst_pair = None

        for i in range(len(run_ids)):
            for j in range(i + 1, len(run_ids)):
                a, b = run_ids[i], run_ids[j]
                key = (a, b, qid) if a < b else (b, a, qid)
                jv = jaccard_cache.get(key)
                if jv is None:
                    # compute if cache missing
                    jv = jaccard(runs[a].get(qid, set()), runs[b].get(qid, set()))
                if jv < min_j:
                    min_j = jv
                    worst_pair = f"{a} vs {b}"

        if worst_pair is None:
            worst_pair = "N/A"

        out.append(
            PerQueryInstability(
                query_id=qid,
                min_overlap=min_j,
                worst_pair=worst_pair,
                unstable_flag=(min_j < flip_threshold),
            )
        )

    # Sort: most unstable first
    out.sort(key=lambda r: (r.min_overlap, r.query_id))
    return out
