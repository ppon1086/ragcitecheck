from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ragcitecheck.canonicalize import Canonicalizer
from ragcitecheck.validate import ValidateOptions, validate_runs_folder
from ragcitecheck.metrics import (
    PairwiseSummary,
    PerQueryInstability,
    RunQuality,
    compute_pairwise_summaries,
    compute_per_query_instability,
    compute_run_quality,
    jaccard,
)


def _write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(list(header))
        for r in rows:
            w.writerow(list(r))


def _write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _plot_overlap_histogram(overlaps: List[float], out_path: Path, bins: int = 20) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.hist(overlaps, bins=bins)
    plt.xlabel("Jaccard overlap (doc-level)")
    plt.ylabel("Count")
    plt.title("Citation overlap distribution")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _as_runs_qmap(vres) -> Dict[str, Dict[str, Set[str]]]:
    """
    ValidationResult -> {run_id: {query_id: set(doc_ids)}}
    """
    out: Dict[str, Dict[str, Set[str]]] = {}
    for rid, rd in vres.runs.items():
        out[rid] = rd.q_to_docs
    return out


def _lookup_j(
    jcache: Dict[Tuple[str, str, str], float],
    run_a: str,
    run_b: str,
    query_id: str,
    runs: Dict[str, Dict[str, Set[str]]],
) -> float:
    a, b = (run_a, run_b) if run_a < run_b else (run_b, run_a)
    key = (a, b, query_id)
    if key in jcache:
        return jcache[key]
    # fallback
    sa = runs[run_a].get(query_id, set())
    sb = runs[run_b].get(query_id, set())
    return jaccard(sa, sb)


def generate_report(
    *,
    runs_dir: Path,
    out_dir: Path,
    canonicalizer: Canonicalizer,
    min_overlap: float = 0.5,
    flip_threshold: float = 0.5,
    topk: Optional[int] = None,
    allow_missing: bool = False,
    baseline: Optional[str] = None,
    topn_examples: int = 20,
    include_top1: bool = False,
) -> None:
    """
    End-to-end report generator:
      - validates logs
      - computes metrics
      - writes CSV/MD/PNG outputs
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if not (0.0 <= min_overlap <= 1.0):
        raise ValueError("--min-overlap must be in [0,1]")
    if not (0.0 <= flip_threshold <= 1.0):
        raise ValueError("--flip-threshold must be in [0,1]")
    if topk is not None and topk <= 0:
        raise ValueError("--topk must be positive if provided")

    # 1) Validate
    vopts = ValidateOptions(allow_missing=allow_missing, topk=topk)
    vres = validate_runs_folder(runs_dir=runs_dir, canonicalizer=canonicalizer, opts=vopts)

    # Determine evaluation query set
    query_ids = sorted(list(vres.query_ids_intersection if allow_missing else vres.query_ids_union))
    runs = _as_runs_qmap(vres)
    run_ids = sorted(runs.keys())

    # 2) Run quality
    run_quality: List[RunQuality] = compute_run_quality(runs, query_ids)
    _write_csv(
        out_dir / "run_quality.csv",
        header=["run_id", "citation_rate", "null_rate", "avg_cited_docs", "median_cited_docs", "p95_cited_docs"],
        rows=[
            [
                rq.run_id,
                f"{rq.citation_rate:.6f}",
                f"{rq.null_rate:.6f}",
                f"{rq.avg_cited_docs:.6f}",
                f"{rq.median_cited_docs:.3f}",
                f"{rq.p95_cited_docs:.3f}",
            ]
            for rq in sorted(run_quality, key=lambda x: x.run_id)
        ],
    )

    # 3) Pairwise summaries + cache
    pairwise, jcache = compute_pairwise_summaries(
        runs=runs,
        query_ids=query_ids,
        flip_threshold=flip_threshold,
        baseline=baseline,
        compute_top1=include_top1,
    )

    pairwise_header = [
        "configA",
        "configB",
        "avg_overlap",
        "flip_rate",
        "null_rate_A",
        "null_rate_B",
        "null_loss_A_to_B",
        "null_gain_A_to_B",
    ]
    if include_top1:
        pairwise_header.append("top1_doc_stability")

    pairwise_rows: List[List[Any]] = []
    for s in pairwise:
        row = [
            s.config_a,
            s.config_b,
            f"{s.avg_overlap:.6f}",
            f"{s.flip_rate:.6f}",
            f"{s.null_rate_a:.6f}",
            f"{s.null_rate_b:.6f}",
            f"{s.null_loss_rate_a_to_b:.6f}",
            f"{s.null_gain_rate_a_to_b:.6f}",
        ]
        if include_top1:
            row.append("" if s.top1_doc_stability is None else f"{s.top1_doc_stability:.6f}")
        pairwise_rows.append(row)

    _write_csv(out_dir / "pairwise_config_stability.csv", header=pairwise_header, rows=pairwise_rows)

    # 4) Per-query worst-case
    per_query: List[PerQueryInstability] = compute_per_query_instability(
        runs=runs,
        query_ids=query_ids,
        flip_threshold=flip_threshold,
        jaccard_cache=jcache,
    )

    # stable vs unstable at min_overlap threshold
    _write_csv(
        out_dir / "per_query_stability.csv",
        header=["query_id", "min_overlap_across_pairs", "worst_pair", f"stable_at_min_overlap_{min_overlap}"],
        rows=[
            [r.query_id, f"{r.min_overlap:.6f}", r.worst_pair, "1" if (r.min_overlap >= min_overlap) else "0"]
            for r in per_query
        ],
    )

    # 5) Examples markdown
    examples = per_query[:topn_examples]
    md_lines: List[str] = []
    md_lines.append("# Ragcitecheck instability examples\n")
    md_lines.append(f"- Runs: {', '.join(run_ids)}")
    md_lines.append(f"- Queries evaluated: {len(query_ids)}")
    md_lines.append(f"- Flip threshold: J < {flip_threshold}")
    md_lines.append(f"- Min-overlap (stability): {min_overlap}")
    md_lines.append(f"- allow_missing: {allow_missing}")
    md_lines.append(f"- topk: {topk}\n")

    for ex in examples:
        qid = ex.query_id
        md_lines.append(f"## {qid}")
        md_lines.append(f"- min_overlap_across_pairs: **{ex.min_overlap:.3f}**")
        md_lines.append(f"- worst_pair: **{ex.worst_pair}**\n")

        md_lines.append("| run_id | cited_docs (doc_id set) |")
        md_lines.append("|---|---|")
        for rid in run_ids:
            docs = sorted(list(runs[rid].get(qid, set())))
            md_lines.append(f"| {rid} | {', '.join(docs) if docs else '(empty)'} |")
        md_lines.append("")

        # worst pair diff
        wp = ex.worst_pair
        if " vs " in wp:
            a, b = wp.split(" vs ", 1)
            sa = runs.get(a, {}).get(qid, set())
            sb = runs.get(b, {}).get(qid, set())
            added = sorted(list(sb - sa))
            removed = sorted(list(sa - sb))
            jv = _lookup_j(jcache, a, b, qid, runs)
            flip = "YES" if jv < flip_threshold else "no"
            md_lines.append(
                f"- worst-pair diff `{a}` → `{b}`: J={jv:.3f}, flip={flip}, "
                f"+[{', '.join(added) if added else ''}] -[{', '.join(removed) if removed else ''}]"
            )
        md_lines.append("")

    _write_markdown(out_dir / "instability_examples.md", "\n".join(md_lines))

    # 6) Histogram
    overlaps: List[float] = []
    if baseline:
        for rid in run_ids:
            if rid == baseline:
                continue
            for qid in query_ids:
                overlaps.append(_lookup_j(jcache, baseline, rid, qid, runs))
    else:
        for i in range(len(run_ids)):
            for j in range(i + 1, len(run_ids)):
                a, b = run_ids[i], run_ids[j]
                for qid in query_ids:
                    overlaps.append(_lookup_j(jcache, a, b, qid, runs))

    _plot_overlap_histogram(overlaps, out_dir / "citation_overlap_hist.png", bins=20)

    # 7) meta JSON
    meta = {
        "runs_dir": str(runs_dir),
        "out_dir": str(out_dir),
        "run_ids": run_ids,
        "query_count": len(query_ids),
        "flip_threshold": flip_threshold,
        "min_overlap": min_overlap,
        "baseline": baseline,
        "allow_missing": allow_missing,
        "topk": topk,
        "pairwise_rows": len(pairwise),
        "canonicalization": vres.canonicalization,
        "warnings": vres.warnings,
    }
    (out_dir / "report_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
