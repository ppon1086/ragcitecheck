from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional
import csv
import re

_WHITESPACE_RE = re.compile(r"\s+")


# -------------------------
# Public dataclasses
# -------------------------

@dataclass(frozen=True)
class CanonicalizeOptions:
    """
    Options for doc_id normalization.

    Notes:
    - By default we lower-case and strip.
    - collapse_internal_whitespace turns any run of whitespace into a single space.
    """
    case_sensitive: bool = False
    strip: bool = True
    collapse_internal_whitespace: bool = False


@dataclass
class CanonicalizationReport:
    mapped_count: int = 0
    unmapped_count: int = 0


@dataclass
class CanonicalizationCollisionReport:
    collision_count: int = 0
    collisions: Optional[Dict[str, int]] = None


# -------------------------
# Functional helpers
# -------------------------

def load_docid_map_csv(path: str) -> Dict[str, str]:
    """
    Load a doc id mapping CSV.

    CSV must have headers: raw,canonical
    """
    mapping: Dict[str, str] = {}
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "raw" not in reader.fieldnames or "canonical" not in reader.fieldnames:
            raise ValueError(f"docid-map must have headers: raw,canonical (got {reader.fieldnames})")
        for row in reader:
            raw = (row.get("raw") or "").strip()
            canon = (row.get("canonical") or "").strip()
            if raw and canon:
                mapping[raw] = canon
    return mapping


def _normalize_doc_id(doc_id: str, opts: CanonicalizeOptions) -> str:
    s = doc_id
    if opts.strip:
        s = s.strip()
    if opts.collapse_internal_whitespace:
        s = _WHITESPACE_RE.sub(" ", s).strip()
    # Normalize path separators for Windows/Linux consistency
    s = s.replace("\\", "/")
    if not opts.case_sensitive:
        s = s.lower()
    return s


def _normalize_map(docid_map: Dict[str, str], opts: CanonicalizeOptions) -> Dict[str, str]:
    """
    Normalize mapping keys/values so lookups match the chosen options.
    """
    out: Dict[str, str] = {}
    for raw, canon in docid_map.items():
        raw_n = _normalize_doc_id(raw, opts)
        canon_n = _normalize_doc_id(canon, opts)
        if raw_n and canon_n:
            out[raw_n] = canon_n
    return out


# -------------------------
# Class API (consumed by cli.py / validate.py)
# -------------------------

class Canonicalizer:
    """
    Canonicalizes doc_ids and optionally applies an alias map.

    This class is used by validate/report flows.
    """

    def __init__(
        self,
        opts: Optional[CanonicalizeOptions] = None,
        docid_map: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> None:
        # Back-compat: allow callers to pass options=... (older drafts)
        if opts is None and "options" in kwargs:
            opts = kwargs["options"]
        self.opts = opts or CanonicalizeOptions()
        self.docid_map = docid_map or {}

    @classmethod
    def from_map_csv(cls, path: str, opts: Optional[CanonicalizeOptions] = None, **kwargs) -> "Canonicalizer":
        # Back-compat: accept options=... too
        if opts is None and "options" in kwargs:
            opts = kwargs["options"]
        o = opts or CanonicalizeOptions()
        raw_map = load_docid_map_csv(path)
        norm_map = _normalize_map(raw_map, o)
        return cls(opts=o, docid_map=norm_map)

    def canonicalize_doc_id(self, doc_id: str, report: Optional[CanonicalizationReport] = None) -> str:
        """
        Normalize doc_id and (optionally) apply docid_map.

        If report is provided, increments mapped_count / unmapped_count.
        """
        before = _normalize_doc_id(doc_id, self.opts)
        if self.docid_map:
            if before in self.docid_map:
                if report is not None:
                    report.mapped_count += 1
                return self.docid_map[before]
            if report is not None:
                report.unmapped_count += 1
        return before

    def detect_collisions(
        self,
        doc_ids: Iterable[str],
        report: Optional[CanonicalizationReport] = None,
    ) -> CanonicalizationCollisionReport:
        """
        Detect collisions after canonicalization (two or more raw doc_ids mapping
        to the same canonical id).

        Returns CanonicalizationCollisionReport with:
          - collision_count = number of canonical ids with count > 1
          - collisions = {canonical_id: count}
        """
        counts: Dict[str, int] = {}
        for d in doc_ids:
            c = self.canonicalize_doc_id(d, report=report)
            counts[c] = counts.get(c, 0) + 1
        collisions = {k: v for k, v in counts.items() if v > 1}
        return CanonicalizationCollisionReport(
            collision_count=len(collisions),
            collisions=collisions or None,
        )
