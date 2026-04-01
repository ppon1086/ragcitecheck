from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

DEFAULT_DOCS_KEY = "retrieved"


def default_docs_key() -> str:
    return DEFAULT_DOCS_KEY


@dataclass(frozen=True)
class DocSpan:
    """A single retrieved evidence item.

    Logs may store `docs` either as:
      1) List[str]                  (doc ids only; legacy)
      2) List[dict] / List[DocSpan] (doc+span; preferred)

    Frozen design (MDPI/ICLR):
      Evidence identity = (doc_id, span_hash)

    NOTE: This is optional and not enforced at runtime; it exists to clarify schema.
    """

    doc_id: str
    span_text: str
    span_hash: str
    node_id: Optional[str] = None  # debug only


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    query_id: str
    docs_key: str = DEFAULT_DOCS_KEY

    # docs can be list[str] OR list[dict] (e.g., [{"doc_id":..., "span_text":..., "span_hash":...}, ...])
    docs: List[Any] = None  # type: ignore[assignment]

    source: Optional[str] = None
    event: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "run_id": self.run_id,
            "query_id": self.query_id,
            self.docs_key: self.docs if self.docs is not None else [],
        }

        if self.source:
            d["source"] = self.source
        if self.event:
            d["event"] = self.event

        return d
