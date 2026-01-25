from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from ragcitecheck.canonicalize import CanonicalizeOptions, Canonicalizer
from ragcitecheck.report import generate_report
from ragcitecheck.validate import ValidateOptions, validate_runs_folder
_DEFAULT_RUN_ID_KEYS = ("run_id", "runId", "config_id")
_DEFAULT_QUERY_ID_KEYS = ("query_id", "qid", "id")
_DEFAULT_DOCS_KEYS = ("cited", "retrieved", "contexts")
_DEFAULT_DOC_ID_KEYS = ("doc_id", "document_id", "id")



def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")

def _parse_keys_csv(s: Optional[str], default: tuple[str, ...]) -> tuple[str, ...]:
    if not s:
        return default
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return tuple(parts) if parts else default

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="ragcitecheck")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Common args
    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--runs", required=True, help="Folder containing one or more run JSONL files.")
        p.add_argument("--out", default="out_report", help="Output folder to write artifacts.")
        p.add_argument("--allow-missing", action="store_true", help="Evaluate only query_id intersection.")
        p.add_argument("--docid-map", default=None, help="Optional CSV mapping file headers raw,canonical.")
        p.add_argument("--case-sensitive", action="store_true", help="Treat doc_id as case-sensitive.")
        p.add_argument("--collapse-internal-whitespace", action="store_true",
                       help="Collapse internal whitespace sequences in doc_id to single spaces.")
        p.add_argument("--topk", type=int, default=None,
                       help="If set, truncate each query's docs list to top-k before metrics.")
        p.add_argument("--run-id-keys", type=str, default=None,
                       help="Comma-separated keys to find run_id (e.g., run_id,config_id,run).")
        p.add_argument("--query-id-keys", type=str, default=None,
                       help="Comma-separated keys to find query_id (e.g., query_id,qid).")
        p.add_argument("--docs-keys", type=str, default=None,
                       help="Comma-separated keys to find docs list (e.g., cited,retrieved,contexts).")
        p.add_argument("--doc-id-keys", type=str, default=None,
                       help="Comma-separated keys to find doc_id in doc entries (e.g., doc_id,document_id,id,source_id).")


    p_validate = sub.add_parser("validate", help="Validate runs folder schema and coverage.")
    add_common(p_validate)

    p_report = sub.add_parser("report", help="Generate stability report from runs.")
    add_common(p_report)
    p_report.add_argument("--min-overlap", type=float, default=0.5,
                          help="Stability threshold for min-overlap across pairs (default: 0.5).")
    p_report.add_argument("--flip-threshold", type=float, default=0.5,
                          help="Flip threshold: mark unstable if overlap < this value (default: 0.5).")
    p_report.add_argument("--baseline", type=str, default=None,
                          help="If set, compare all runs only against this baseline run_id.")
    p_report.add_argument("--topn-examples", type=int, default=20,
                          help="How many examples to include in instability_examples.md")
    p_report.add_argument("--include-top1", action="store_true",
                          help="Also compute a simple top1-doc stability proxy.")

    args = parser.parse_args(argv)

    runs_dir = Path(args.runs)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    canon_opts = CanonicalizeOptions(
        case_sensitive=bool(args.case_sensitive),
        strip=True,
        collapse_internal_whitespace=bool(args.collapse_internal_whitespace),
    )

    canonicalizer = (
        Canonicalizer.from_map_csv(args.docid_map, opts=canon_opts)
        if args.docid_map
        else Canonicalizer(opts=canon_opts)
    )
    run_id_keys  = _parse_keys_csv(args.run_id_keys,  _DEFAULT_RUN_ID_KEYS)
    query_id_keys = _parse_keys_csv(args.query_id_keys, _DEFAULT_QUERY_ID_KEYS)
    docs_keys    = _parse_keys_csv(args.docs_keys,    _DEFAULT_DOCS_KEYS)
    doc_id_keys  = _parse_keys_csv(args.doc_id_keys,  _DEFAULT_DOC_ID_KEYS)


    vopts = ValidateOptions(
        run_id_keys=run_id_keys,
        query_id_keys=query_id_keys,
        docs_keys=docs_keys,
        doc_id_keys=doc_id_keys,
        allow_missing=bool(args.allow_missing),
        topk=int(args.topk) if args.topk is not None else None,
    )

  
    vres = validate_runs_folder(
        runs_dir=runs_dir,
        canonicalizer=canonicalizer,
        allow_missing=bool(args.allow_missing),
        opts = vopts
        )


    # Save validation summary
    validation_summary = {
        "runs": len(vres.runs),
        "run_ids": sorted(vres.runs.keys()),
        "union_queries": len(vres.query_ids_union),
        "intersection_queries": len(vres.query_ids_intersection),
        "allow_missing": vopts.allow_missing,
        "topk": vopts.topk,
        "warnings": vres.warnings,
        "canonicalization": vres.canonicalization,
        "files": {rid: vres.runs[rid].file for rid in sorted(vres.runs.keys())},
    }
    _write_json(out_dir / "validation_summary.json", validation_summary)

    if vres.warnings:
        for w in vres.warnings:
            print(f"WARNING: {w}")

    if args.cmd == "validate":
        print(
            "Validation passed: "
            f"runs={len(vres.runs)}, "
            f"intersection_queries={len(vres.query_ids_intersection)}, "
            f"allow_missing={vopts.allow_missing}, "
            f"warnings={len(vres.warnings)}"
        )
        print(f"Wrote: {out_dir / 'validation_summary.json'}")
        return 0

    if args.cmd == "report":
        generate_report(
            runs_dir=runs_dir,
            out_dir=out_dir,
            canonicalizer=canonicalizer,
            min_overlap=float(args.min_overlap),
            flip_threshold=float(args.flip_threshold),
            topk=vopts.topk,
            allow_missing=vopts.allow_missing,
            baseline=args.baseline,
            topn_examples=int(args.topn_examples),
            include_top1=bool(args.include_top1),
        )
        print(f"Report written to: {out_dir}")
        print(f"Wrote: {out_dir / 'validation_summary.json'}")
        print(f"Wrote: {out_dir / 'report_meta.json'}")
        return 0

    raise SystemExit(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
