from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

from ragcitecheck.canonicalize import Canonicalizer
from ragcitecheck.metrics import compute_pairwise


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _extract_query_id(rec: Dict) -> str:
    for k in ("query_id", "qid", "id"):
        if k in rec and rec[k] is not None:
            return str(rec[k])
    raise ValueError("record missing query_id/qid/id")


def _extract_docs(rec: Dict) -> List[Dict]:
    for k in ("docs", "documents", "retrieved", "contexts"):
        if k in rec and rec[k] is not None:
            if not isinstance(rec[k], list):
                raise ValueError(f"{k} must be a list")
            return rec[k]
    raise ValueError("record missing docs/documents/retrieved/contexts list")


def _extract_doc_id(d: Dict) -> str:
    for k in ("doc_id", "document_id", "docid", "id", "source_id"):
        if k in d and d[k] is not None:
            return str(d[k])
    raise ValueError("doc missing doc_id/document_id/docid/id/source_id")


def load_run_jsonl(path: Path, canonicalizer: Canonicalizer, topk: Optional[int] = None) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for rec in _iter_jsonl(path):
        qid = _extract_query_id(rec)
        docs = _extract_docs(rec)
        doc_ids = [_extract_doc_id(d) for d in docs]
        if topk is not None:
            doc_ids = doc_ids[: int(topk)]
        canon = [canonicalizer.canonicalize_doc_id(x) for x in doc_ids]
        out[qid] = canon
    return out


def generate_report(
    runs_dir: Path,
    out_dir: Path,
    min_overlap: float,
    flip_threshold: float,
    topk: Optional[int],
    canonicalizer: Canonicalizer,
) -> None:
    run_files = sorted([p for p in runs_dir.iterdir() if p.suffix.lower() == ".jsonl"])
    if not run_files:
        raise ValueError(f"No .jsonl files found in: {runs_dir}")

    runs: List[Tuple[str, Dict[str, List[str]]]] = []
    for rf in run_files:
        runs.append((rf.stem, load_run_jsonl(rf, canonicalizer=canonicalizer, topk=topk)))

    # pairwise
    pairwise_rows = []
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            a_name, a = runs[i]
            b_name, b = runs[j]
            pw = compute_pairwise(a_name, b_name, a, b, flip_threshold=flip_threshold)
            for row in pw:
                pairwise_rows.append(
                    {
                        "query_id": row.query_id,
                        "run_a": row.run_a,
                        "run_b": row.run_b,
                        "overlap": row.overlap,
                        "jaccard": row.jaccard,
                        "flipped": int(row.flipped),
                    }
                )

    out_dir.mkdir(parents=True, exist_ok=True)

    # write pairwise CSV
    pairwise_csv = out_dir / "pairwise_config_stability.csv"
    with pairwise_csv.open("w", encoding="utf-8") as f:
        f.write("query_id,run_a,run_b,overlap,jaccard,flipped\n")
        for r in pairwise_rows:
            f.write(
                f"{r['query_id']},{r['run_a']},{r['run_b']},"
                f"{r['overlap']:.6f},{r['jaccard']:.6f},{r['flipped']}\n"
            )

    # per-query aggregation (mean overlap across pairs)
    per_query: Dict[str, List[float]] = {}
    for r in pairwise_rows:
        per_query.setdefault(r["query_id"], []).append(float(r["overlap"]))

    per_query_rows = []
    for qid, vals in per_query.items():
        avg = sum(vals) / max(1, len(vals))
        per_query_rows.append(
            {
                "query_id": qid,
                "mean_overlap": avg,
                "stable": int(avg >= min_overlap),
            }
        )

    per_query_csv = out_dir / "per_query_stability.csv"
    with per_query_csv.open("w", encoding="utf-8") as f:
        f.write("query_id,mean_overlap,stable\n")
        for r in sorted(per_query_rows, key=lambda x: x["query_id"]):
            f.write(f"{r['query_id']},{r['mean_overlap']:.6f},{r['stable']}\n")

    # histogram plot
    overlaps = [float(r["overlap"]) for r in pairwise_rows]
    plt.figure()
    plt.hist(overlaps, bins=20)
    plt.xlabel("Overlap ratio")
    plt.ylabel("Count")
    plt.title("Citation overlap distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "citation_overlap_hist.png", dpi=150)
    plt.close()

    # Markdown summary
    stable_count = sum(r["stable"] for r in per_query_rows)
    total_q = len(per_query_rows)
    md = [
        "# RAG Citation Stability Report",
        "",
        f"- Runs: {len(runs)}",
        f"- Queries compared: {total_q}",
        f"- Stable queries (mean_overlap >= {min_overlap}): {stable_count}/{total_q}",
        "",
        "Artifacts:",
        "- per_query_stability.csv",
        "- pairwise_config_stability.csv",
        "- citation_overlap_hist.png",
    ]
    (out_dir / "report.md").write_text("\n".join(md), encoding="utf-8")
