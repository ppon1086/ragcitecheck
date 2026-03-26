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
import re

_RUNID_PATTERNS = [
    # common: arguana_k10_c256_o32
    re.compile(r".*?_k(?P<k>\d+)_c(?P<c>\d+)_o(?P<o>\d+).*"),
    # your earlier: arguana_k10_c256 (no overlap)
    re.compile(r".*?_k(?P<k>\d+)_c(?P<c>\d+).*"),
]

def _parse_run_id_params(run_id: str) -> Dict[str, Optional[int]]:
    """
    Best-effort parse for run_id strings like:
      - arguana_k10_c256_o32
      - arguana_k10_c256
    Returns dict with k/c/o as ints when found, else None.
    """
    for pat in _RUNID_PATTERNS:
        m = pat.match(run_id)
        if m:
            gd = m.groupdict()
            k = int(gd["k"]) if gd.get("k") is not None else None
            c = int(gd["c"]) if gd.get("c") is not None else None
            o = int(gd["o"]) if gd.get("o") is not None else None
            return {"top_k": k, "chunk_size": c, "chunk_overlap": o}
    return {"top_k": None, "chunk_size": None, "chunk_overlap": None}

def _fmt_int(x: Optional[int]) -> str:
    return "" if x is None else str(x)

def _delta(a: Optional[int], b: Optional[int]) -> str:
    if a is None or b is None:
        return ""
    return str(b - a)


def _write_csv(
    path: Path,
    header: Optional[Sequence[str]] = None,
    rows: Optional[Sequence[Any]] = None,
    *,
    fieldnames: Optional[Sequence[str]] = None,
) -> None:
    """
    Backward-compatible CSV writer.

    Supports BOTH call styles used across your report code:
      A) _write_csv(path, header=[...], rows=[[...],[...]])
      B) _write_csv(path, rows=[{...},{...}], fieldnames=[...])
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Normalize defaults
    rows = rows or []

    with path.open("w", newline="", encoding="utf-8") as f:
        # DictWriter mode (rows are dicts)
        if fieldnames is not None:
            w = csv.DictWriter(f, fieldnames=list(fieldnames))
            w.writeheader()
            for r in rows:
                w.writerow(r if isinstance(r, dict) else {})
            return

        # Classic header/rows mode
        if header is None:
            raise TypeError("_write_csv requires either header=... or fieldnames=...")

        w2 = csv.writer(f)
        w2.writerow(list(header))
        for r in rows:
            # r is expected to be a sequence (list/tuple)
            w2.writerow(list(r))

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

import csv

def _read_csv_rows(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

def _write_csv_dicts(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

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

def _plot_per_query_boxplot(per_query_j: Dict[str, List[float]], out_path: Path) -> None:
    """
    per_query_j: query_id -> list of J values across pairs (or vs baseline)
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vals = list(per_query_j.values())
    plt.figure()
    plt.boxplot(vals, showfliers=False)
    plt.xlabel("Query index (sorted)")
    plt.ylabel("Jaccard overlap")
    plt.title("Per-query stability distribution")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_knob_sensitivity(rows: List[Dict[str, Any]], out_path: Path, x_key: str, y_key: str) -> None:
    """
    rows: list of dicts containing parsed config knobs (k, chunk) + metric y_key
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    xs = [r[x_key] for r in rows if r.get(x_key) is not None]
    ys = [r[y_key] for r in rows if r.get(x_key) is not None]
    if not xs:
        return
    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.xlabel(x_key)
    plt.ylabel(y_key)
    plt.title(f"{y_key} vs {x_key}")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def generate_report(
    *,
    runs_dir: Path,
    out_dir: Path,
    canonicalizer: Canonicalizer,
    min_overlap: float = 0.8,
    flip_threshold: float = 0.5,
    topk: Optional[int] = None,
    allow_missing: bool = False,
    evidence_keys: Sequence[str] = ("doc", "doc_span"),
    run_id_keys: Sequence[str] = ("run_id", "runId", "config_id"),
    query_id_keys: Sequence[str] = ("query_id", "qid", "id"),
    docs_keys: Sequence[str] = ("cited", "retrieved", "contexts"),
    doc_id_keys: Sequence[str] = ("doc_id", "document_id", "id"),
    baseline: Optional[str] = None,
    topn_examples: int = 10,
    include_top1: bool = False,
) -> None:
    """
    End-to-end report generator.

    Point-3 behavior:
      - run the same report twice: doc-level evidence and chunk-level evidence
      - write outputs into out_dir/doc/* and out_dir/chunk/*
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    if not (0.0 <= min_overlap <= 1.0):
        raise ValueError("--min-overlap must be in [0,1]")
    if not (0.0 <= flip_threshold <= 1.0):
        raise ValueError("--flip-threshold must be in [0,1]")
    if topk is not None and topk <= 0:
        raise ValueError("--topk must be positive if provided")

    if not evidence_keys:
        raise ValueError("evidence_keys must be non-empty, e.g. ('doc',) or ('doc','chunk').")

    # Collect tier artifacts so we can produce doc-vs-doc_span comparators (Step 4)
    tier_pairwise: Dict[str, List[List[str]]] = {}
    tier_pairwise_header: Dict[str, List[str]] = {}
    tier_per_query: Dict[str, List[PerQueryInstability]] = {}
    tier_jcache: Dict[str, Dict[Tuple[str, str, str], float]] = {}

    def _run_one_tier(evidence_key: str) -> Dict[str, Any]:
        tier_out = out_dir / evidence_key
        tier_out.mkdir(parents=True, exist_ok=True)

        # 1) Validate (tier-specific)
        vopts = ValidateOptions(
            allow_missing=allow_missing,
            run_id_keys=tuple(run_id_keys),
            query_id_keys=tuple(query_id_keys),
            docs_keys=tuple(docs_keys),
            doc_id_keys=tuple(doc_id_keys),
            evidence_key=evidence_key,
            topk=topk,
        )
        vres = validate_runs_folder(runs_dir=runs_dir, canonicalizer=canonicalizer, opts=vopts)

        # Determine evaluation query set
        query_ids = sorted(list(vres.query_ids_intersection if allow_missing else vres.query_ids_union))
        runs = _as_runs_qmap(vres)
        run_ids = sorted(runs.keys())
                # --- Point 4: make "what changed" explicit ---
        run_params: Dict[str, Dict[str, Optional[int]]] = {rid: _parse_run_id_params(rid) for rid in run_ids}

        # Per-run config table
        _write_csv(
            tier_out / "run_configs.csv",
            header=["run_id", "top_k", "chunk_size", "chunk_overlap"],
            rows=[
                [
                    rid,
                    _fmt_int(run_params[rid].get("top_k")),
                    _fmt_int(run_params[rid].get("chunk_size")),
                    _fmt_int(run_params[rid].get("chunk_overlap")),
                ]
                for rid in run_ids
            ],
        )

        

        # 2) Run quality
        run_quality: List[RunQuality] = compute_run_quality(runs, query_ids)
        _write_csv(
            tier_out / "run_quality.csv",
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

        # 3b) Pairwise stability table (Step 4 artifact)
        pairwise_header = [
            "config_a",
            "config_b",
            "avg_overlap",
            "flip_rate",
            "null_rate_a",
            "null_rate_b",
            "null_loss_rate_a_to_b",
            "null_gain_rate_a_to_b",
            f"stable_at_min_overlap_{min_overlap}",
        ]
        if include_top1:
            pairwise_header.append("top1_doc_stability")

        pairwise_rows: List[List[str]] = []
        for row in pairwise:
            stable = "1" if (row.avg_overlap >= min_overlap) else "0"
            out_row: List[str] = [
                row.config_a,
                row.config_b,
                f"{row.avg_overlap:.6f}",
                f"{row.flip_rate:.6f}",
                f"{row.null_rate_a:.6f}",
                f"{row.null_rate_b:.6f}",
                f"{row.null_loss_rate_a_to_b:.6f}",
                f"{row.null_gain_rate_a_to_b:.6f}",
                stable,
            ]
            if include_top1:
                out_row.append("" if row.top1_doc_stability is None else f"{row.top1_doc_stability:.6f}")
            pairwise_rows.append(out_row)

        # Save tier-local
        _write_csv(tier_out / "pairwise_config_stability.csv", header=pairwise_header, rows=pairwise_rows)

        # Save ALSO at root with required Step-4 name
        _write_csv(out_dir / f"pairwise_config_stability_{evidence_key}.csv", header=pairwise_header, rows=pairwise_rows)


        # 4) Per-query worst-case
        per_query: List[PerQueryInstability] = compute_per_query_instability(
            runs=runs,
            query_ids=query_ids,
            flip_threshold=flip_threshold,
            jaccard_cache=jcache,
        )

        _write_csv(
            tier_out / "per_query_stability.csv",
            header=["query_id", "min_overlap_across_pairs", "worst_pair", f"stable_at_min_overlap_{min_overlap}"],
            rows=[
                [r.query_id, f"{r.min_overlap:.6f}", r.worst_pair, "1" if (r.min_overlap >= min_overlap) else "0"]
                for r in per_query
            ],
        )
        # Store tier artifacts for Step-4 comparators
        tier_pairwise[evidence_key] = pairwise_rows
        tier_pairwise_header[evidence_key] = pairwise_header
        tier_per_query[evidence_key] = per_query
        tier_jcache[evidence_key] = jcache

        # 5) Examples markdown
        examples = per_query[:topn_examples]
        md_lines: List[str] = []
        md_lines.append("# Ragcitecheck instability examples\n")
        md_lines.append(f"- Evidence key: **{evidence_key}**")
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

            md_lines.append("| run_id | cited_docs (set) |")
            md_lines.append("|---|---|")
            for rid in run_ids:
                docs = sorted(list(runs[rid].get(qid, set())))
                md_lines.append(f"| {rid} | {', '.join(docs) if docs else '(empty)'} |")
            md_lines.append("")

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

        _write_markdown(tier_out / "instability_examples.md", "\n".join(md_lines))

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

        _plot_overlap_histogram(overlaps, tier_out / "citation_overlap_hist.png", bins=20)

        # 7) meta JSON
        meta = {
            "runs_dir": str(runs_dir),
            "out_dir": str(tier_out),
            "evidence_key": evidence_key,
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
        (tier_out / "report_meta.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8"
        )
        
        return {
            "evidence_key": evidence_key,
            "tier_out": tier_out,
            "run_ids": run_ids,
            "query_ids": query_ids,
            "runs": runs,                 # run_id -> qid -> set(evidence_id)
            "pairwise": pairwise,         # List[PairwiseSummary]
            "jcache": jcache,             # Dict[(run_a, run_b, qid)] -> jaccard
            "vres": vres,                 # ValidationResult (warnings, canonicalization)
        }


    for ek in evidence_keys:
        _run_one_tier(str(ek))
    
    have_doc = (out_dir / "doc" / "per_query_stability.csv").exists()
    have_doc_span = (out_dir / "doc_span" / "per_query_stability.csv").exists()

    if have_doc and have_doc_span:
        # --- per-query comparator ---
        doc_rows = _read_csv_rows(out_dir / "doc" / "per_query_stability.csv")
        span_rows = _read_csv_rows(out_dir / "doc_span" / "per_query_stability.csv")

    # key by (query_id) since each file is already "worst-case across run-pairs"
        doc_by_q = {r["query_id"]: r for r in doc_rows if "query_id" in r}
        span_by_q = {r["query_id"]: r for r in span_rows if "query_id" in r}

        perq_cmp: List[dict] = []
        same_doc_diff_span: List[dict] = []

        for qid in sorted(set(doc_by_q.keys()) & set(span_by_q.keys())):
            d = doc_by_q[qid]
            s = span_by_q[qid]

        # these columns exist in your generated CSVs
           # Per-query CSV schema uses:
#   - min_overlap_across_pairs
#   - stable_at_min_overlap_{min_overlap}  (e.g., stable_at_min_overlap_0.5)
            stable_col = f"stable_at_min_overlap_{min_overlap}"

            d_min = float(d.get("min_overlap_across_pairs") or d.get("min_overlap") or 0.0)
            s_min = float(s.get("min_overlap_across_pairs") or s.get("min_overlap") or 0.0)

            d_stable_raw = d.get(stable_col) or d.get("stable_at_min_overlap")
            s_stable_raw = s.get(stable_col) or s.get("stable_at_min_overlap")

# Fallback: if column name differs (e.g., formatting), pick the first stable_at_min_overlap_* column    
            if d_stable_raw is None:
                d_stable_raw = next((d[k] for k in d.keys() if k.startswith("stable_at_min_overlap_")), "0")
            if s_stable_raw is None:
                s_stable_raw = next((s[k] for k in s.keys() if k.startswith("stable_at_min_overlap_")), "0")

            d_stable = int(float(d_stable_raw))
            s_stable = int(float(s_stable_raw))

            row = {
                "query_id": qid,
                "min_overlap_doc": d_min,
                "min_overlap_doc_span": s_min,
                "delta_doc_minus_doc_span": (d_min - s_min),
                "worst_pair_doc": d.get("worst_pair", ""),
                "worst_pair_doc_span": s.get("worst_pair", ""),
            }
            perq_cmp.append(row)

        # Step 4.2 sanity: doc stable but spans not stable
            if d_min >= 0.80 and s_min <= 0.20:
                same_doc_diff_span.append(row)

        _write_csv(
            out_dir / "per_query_comparator.csv",
            rows = perq_cmp,
            fieldnames=[
                "query_id",
                "min_overlap_doc",
                "min_overlap_doc_span",
                "delta_doc_minus_doc_span",
                "worst_pair_doc",
                "worst_pair_doc_span",
            ],
        )

        _write_csv(
            out_dir / "same_doc_diff_span.csv",
            rows=same_doc_diff_span,
            fieldnames=[
                "query_id",
                "min_overlap_doc",
                "min_overlap_doc_span",
                "delta_doc_minus_doc_span",
                "worst_pair_doc",
                "worst_pair_doc_span",
            ],
        )

    # ------------------------------------------------------------
# Step 3: Comparator tables (doc vs doc_span)
# Step 4.2: Sanity CSV (same doc overlap, different span overlap)
# ------------------------------------------------------------

    # --- pairwise comparator (only if pairwise exists) ---
    doc_pair_path = out_dir / "doc" / "pairwise_config_stability.csv"
    span_pair_path = out_dir / "doc_span" / "pairwise_config_stability.csv"
    if doc_pair_path.exists() and span_pair_path.exists():
        dpr = _read_csv_rows(doc_pair_path)
        spr = _read_csv_rows(span_pair_path)

        # key by (run_a, run_b)
        dkey = {(r["run_a"], r["run_b"]): r for r in dpr if "run_a" in r and "run_b" in r}
        skey = {(r["run_a"], r["run_b"]): r for r in spr if "run_a" in r and "run_b" in r}

        pair_cmp: List[dict] = []
        for k in sorted(set(dkey.keys()) & set(skey.keys())):
            da = dkey[k]
            sa = skey[k]
            pair_cmp.append({
                "run_a": k[0],
                "run_b": k[1],
                "avg_overlap_doc": float(da.get("avg_overlap", "nan")),
                "avg_overlap_doc_span": float(sa.get("avg_overlap", "nan")),
                "delta_avg_doc_minus_doc_span": float(da.get("avg_overlap", "nan")) - float(sa.get("avg_overlap", "nan")),
                "min_overlap_doc": float(da.get("min_overlap", "nan")),
                "min_overlap_doc_span": float(sa.get("min_overlap", "nan")),
                "delta_min_doc_minus_doc_span": float(da.get("min_overlap", "nan")) - float(sa.get("min_overlap", "nan")),
            })

        _write_csv(
            out_dir / "pairwise_comparator.csv",
            rows=pair_cmp,
            fieldnames=[
                "run_a",
                "run_b",
                "avg_overlap_doc",
                "avg_overlap_doc_span",
                "delta_avg_doc_minus_doc_span",
                "min_overlap_doc",
                "min_overlap_doc_span",
                "delta_min_doc_minus_doc_span",
            ],
        )
    # --- Step 4: doc vs doc_span comparator + sanity check CSV ---
    if ("doc" in tier_pairwise) and ("doc_span" in tier_pairwise):
        # Pairwise comparator (doc vs doc_span): join on (config_a, config_b)
        def _pair_key(r: List[str]) -> Tuple[str, str]:
            return (r[0], r[1])  # config_a, config_b

        doc_rows = { _pair_key(r): r for r in tier_pairwise["doc"] }
        span_rows = { _pair_key(r): r for r in tier_pairwise["doc_span"] }

        comp_header = [
            "config_a", "config_b",
            "avg_overlap_doc", "avg_overlap_doc_span",
            "stable_doc", "stable_doc_span",
            "flip_rate_doc", "flip_rate_doc_span",
        ]
        comp_rows: List[List[str]] = []

        for k in sorted(set(doc_rows.keys()) & set(span_rows.keys())):
            dr = doc_rows[k]
            sr = span_rows[k]
            comp_rows.append([
                k[0], k[1],
                dr[2],            # avg_overlap_doc
                sr[2],            # avg_overlap_doc_span
                dr[8],            # stable_at_min_overlap (doc)
                sr[8],            # stable_at_min_overlap (doc_span)
                dr[3],            # flip_rate (doc)
                sr[3],            # flip_rate (doc_span)
            ])


        _write_csv(
            out_dir / "pairwise_comparator_doc_vs_span.csv",
            header=comp_header,
            rows=comp_rows,
        )

        # Sanity CSV: same docs (J_doc == 1.0) but spans differ (J_doc_span < 1.0)
        j_doc = tier_jcache["doc"]
        j_span = tier_jcache["doc_span"]

        sanity_header = ["config_a", "config_b", "query_id", "j_doc", "j_doc_span"]
        sanity_rows: List[List[str]] = []

        # Build run_id pairs from pairwise rows (config_a/config_b)
        for (a, b) in sorted(set(doc_rows.keys()) & set(span_rows.keys())):
            for qid in tier_per_query["doc"][0:0]:  # no-op: keep mypy quiet
                pass

        # We need query ids; take them from any validate run output already used earlier
        # We can recover query ids from j_doc keys
        # jcache key is (a,b,qid)
        for (a, b, qid), vdoc in j_doc.items():
            vspan = j_span.get((a, b, qid))
            if vspan is None:
                continue
            if (abs(vdoc - 1.0) < 1e-12) and (vspan < 1.0 - 1e-12):
                sanity_rows.append([a, b, qid, f"{vdoc:.6f}", f"{vspan:.6f}"])

        _write_csv(out_dir / "same_doc_diff_span_sanity.csv", header=sanity_header, rows=sanity_rows)
        