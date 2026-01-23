from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ragcitecheck.canonicalize import CanonicalizeOptions, Canonicalizer
from ragcitecheck.validate import (
    ValidateOptions,
    validate_runs_folder,
    write_validation_artifacts,
)



def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ragcitecheck",
        description="Document-level citation stability diagnostics for RAG systems (log analyzer).",
    )

    sub = p.add_subparsers(dest="cmd", required=True)

    # ---- validate ----
    pv = sub.add_parser("validate", help="Validate run logs (JSONL) for comparability and schema correctness.")
    pv.add_argument("--runs", required=True, help="Folder containing *.jsonl run files.")
    pv.add_argument("--out", default="./report", help="Output folder for validation artifacts (default: ./report).")

    pv.add_argument("--allow_missing", action="store_true", help="Evaluate only the intersection of queries across runs.")

    pv.add_argument("--dedupe", choices=["error", "first", "last"], default="error",
                    help="Policy for duplicate query_id within a run (default: error).")

    pv.add_argument("--missing-cited", choices=["error", "empty"], default="error",
                    help="If 'cited' is missing/null, treat as error or empty list (default: error).")

    pv.add_argument("--cited-format", choices=["dicts", "strings"], default="dicts",
                    help="Expected cited list item format (default: dicts).")

    pv.add_argument("--case-sensitive", action="store_true",
                    help="Treat doc_id case as significant (default: lowercased).")

    pv.add_argument("--docid-map", default=None,
                    help="Optional CSV mapping file (raw_doc_id, canonical_doc_id) to merge aliases.")

    pv.add_argument("--collapse-internal-whitespace", action="store_true",
                    help="Collapse internal whitespace in doc_id during canonicalization.")

    # ---- report ----
    pr = sub.add_parser("report", help="Compute citation stability metrics and write report artifacts.")
    pr.add_argument("--runs", required=True, help="Folder containing *.jsonl run files.")
    pr.add_argument("--out", default="./report", help="Output folder (default: ./report).")
    pr.add_argument("--flip-threshold", type=float, default=0.5, help="Major flip threshold on Jaccard (default: 0.5).")
    pr.add_argument("--baseline", default=None, help="Optional baseline run_id for baseline-vs-all reporting.")
    pr.add_argument("--topn-examples", type=int, default=20, help="Number of queries to include (default: 20).")
    pr.add_argument("--include-top1", action="store_true", help="Include top1_doc_stability column (deterministic proxy).")

    # Reuse validation/canonicalization flags for report too
    pr.add_argument("--allow_missing", action="store_true")
    pr.add_argument("--dedupe", choices=["error", "first", "last"], default="error")
    pr.add_argument("--missing-cited", choices=["error", "empty"], default="error")
    pr.add_argument("--cited-format", choices=["dicts", "strings"], default="dicts")
    pr.add_argument("--case-sensitive", action="store_true")
    pr.add_argument("--docid-map", default=None)
    pr.add_argument("--collapse-internal-whitespace", action="store_true")

    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.cmd == "validate":
        runs_dir = Path(args.runs)
        out_dir = Path(args.out)

        canon_opts = CanonicalizeOptions(
            case_sensitive=bool(args.case_sensitive),
            collapse_internal_whitespace=bool(args.collapse_internal_whitespace),
        )
        canonicalizer = (
            Canonicalizer.from_map_csv(args.docid_map, opts=canon_opts)
            if args.docid_map else Canonicalizer(opts=canon_opts)
        )

        vopts = ValidateOptions(
            allow_missing=bool(args.allow_missing),
            dedupe_policy=str(args.dedupe),
            missing_cited_policy=str(args.missing_cited),
            cited_format=str(args.cited_format),
        )

        result = validate_runs_folder(runs_dir=runs_dir, canonicalizer=canonicalizer, opts=vopts)
        out_dir.mkdir(parents=True, exist_ok=True)
        write_validation_artifacts(result=result, out_dir=out_dir)

        print(
            f"Validation passed: runs={len(result.runs)}, "
            f"queries_per_run~={result.summary['queries_per_run_min']}..{result.summary['queries_per_run_max']}, "
            f"intersection_queries={result.summary['intersection_query_count']}, "
            f"warnings={result.summary['warning_count']}"
        )
        if result.summary.get("coverage_mode") == "intersection":
            print("NOTE: --allow_missing enabled; evaluated query intersection only.")

        return 0

    if args.cmd == "report":
        
        runs_dir = Path(args.runs)
        out_dir = Path(args.out)

        canon_opts = CanonicalizeOptions(
            case_sensitive=bool(args.case_sensitive),
            collapse_internal_whitespace=bool(args.collapse_internal_whitespace),
        )
        canonicalizer = (
            Canonicalizer.from_map_csv(args.docid_map, opts=canon_opts)
            if args.docid_map else Canonicalizer(opts=canon_opts)
        )

        vopts = ValidateOptions(
            allow_missing=bool(args.allow_missing),
            dedupe_policy=str(args.dedupe),
            missing_cited_policy=str(args.missing_cited),
            cited_format=str(args.cited_format),
        )
        from ragcitecheck.report import generate_import
       
        generate_report(
            runs_dir=runs_dir,
            out_dir=out_dir,
            canonicalizer=canonicalizer,
            validate_opts=vopts,
            flip_threshold=float(args.flip_threshold),
            baseline=args.baseline,
            topn_examples=int(args.topn_examples),
            include_top1=bool(args.include_top1),
        )

        print(f"Report written to: {out_dir}")
        return 0

    print(f"Unknown command: {args.cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
