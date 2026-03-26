from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ragcitecheck.harness.recorder import Recorder, RecorderOptions

# --- LlamaIndex callback imports (works across versions) ---
try:
    from llama_index.core.callbacks.base import BaseCallbackHandler  # type: ignore
except Exception:  # pragma: no cover
    try:
        from llama_index.core.callbacks.base import BaseCallbackHandler  # type: ignore
    except Exception:  # pragma: no cover
        BaseCallbackHandler = object  # type: ignore


_WS_RE = re.compile(r"\s+")


def _normalize_span_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = _WS_RE.sub(" ", s)
    return s


def _span_hash(span_text: str) -> str:
    norm = _normalize_span_text(span_text)
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def _extract_span_text(n: Any) -> str:
    node = getattr(n, "node", n)

    t = getattr(node, "text", None)
    if isinstance(t, str) and t.strip():
        return t

    get_content = getattr(node, "get_content", None)
    if callable(get_content):
        try:
            t2 = get_content()
            if isinstance(t2, str) and t2.strip():
                return t2
        except Exception:
            pass

    get_text = getattr(node, "get_text", None)
    if callable(get_text):
        try:
            t3 = get_text()
            if isinstance(t3, str) and t3.strip():
                return t3
        except Exception:
            pass

    return ""


def _extract_doc_id(n: Any) -> Optional[str]:
    node = getattr(n, "node", n)

    doc_id = getattr(node, "ref_doc_id", None)
    if doc_id:
        return str(doc_id)

    meta = getattr(node, "metadata", None)
    if isinstance(meta, dict):
        for k in ("beir_doc_id", "doc_id", "document_id", "source_id", "id"):
            v = meta.get(k)
            if v is not None and str(v).strip():
                return str(v).strip()

    return None


def _extract_node_id(n: Any) -> Optional[str]:
    node = getattr(n, "node", n)
    nid = getattr(node, "node_id", None)
    if nid:
        return str(nid)
    for attr in ("id_", "id"):
        v = getattr(node, attr, None)
        if v:
            return str(v)
    return None


def _extract_doc_and_span(n: Any) -> Dict[str, str]:
    doc_id = _extract_doc_id(n) or "UNKNOWN_DOC_ID"
    span_text = _extract_span_text(n)
    out: Dict[str, str] = {
        "doc_id": doc_id,
        "span_text": span_text,
        "span_hash": _span_hash(span_text),
    }
    node_id = _extract_node_id(n)
    if node_id:
        out["node_id"] = node_id  # debug only
    return out


def _dedupe_docspans(items: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Deterministic dedupe on (doc_id, span_hash), preserving first-seen order."""
    seen: set[Tuple[str, str]] = set()
    out: List[Dict[str, str]] = []
    for d in items:
        key = (d.get("doc_id", ""), d.get("span_hash", ""))
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


@dataclass
class _State:
    current_query_id: Optional[str] = None
    current_query_str: Optional[str] = None


class LlamaIndexAdapter:
    """
    Logs retrieved evidence as structured dicts:
      {"doc_id":..., "span_text":..., "span_hash":..., "node_id"?:...}

    Evidence identity for stability metrics: (doc_id, span_hash)
    node_id is captured for debug only (never used for metrics).
    """

    def __init__(
        self,
        *,
        out_file: str,
        run_id: str,
        docs_key: str = "retrieved",
        normalize_docs_to_str: bool = False,
    ) -> None:
        self.run_id = run_id
        self.docs_key = docs_key
        self._state = _State()

        self.recorder = Recorder(
            out_file=Path(out_file),
            opts=RecorderOptions(
                run_id=run_id,
                docs_key=docs_key,
                normalize_docs_to_str=normalize_docs_to_str,
            ),
        )

    def set_current_query(self, *, query_id: str, query_str: Optional[str] = None) -> None:
        self._state.current_query_id = query_id
        self._state.current_query_str = query_str

    def close(self) -> None:
        self.recorder.close()

    def _emit(self, *, nodes: List[Any], source: str, event: str) -> None:
        qid = self._state.current_query_id
        if not qid:
            return

        docs = [_extract_doc_and_span(n) for n in (nodes or [])]
        deduped = _dedupe_docspans(docs)

        # Use your Recorder API (query_id, docs, extra)
        self.recorder.record(
            query_id=qid,
            docs=deduped,
            extra={
                "run_id": self.run_id,
                "docs_key": self.docs_key,
                "source": source,
                "event": event,
                "query": self._state.current_query_str,
            },
        )

    def handler_instance(self) -> BaseCallbackHandler:
        adapter = self

        class _Handler(BaseCallbackHandler):  # type: ignore
            def __init__(self) -> None:
                # Some LI versions require these args
                try:
                    super().__init__(event_starts_to_ignore=[], event_ends_to_ignore=[])
                except Exception:
                    try:
                        super().__init__()
                    except Exception:
                        pass

            # ---- Required by some LlamaIndex versions ----
            def start_trace(self, trace_id: str, **kwargs: Any) -> None:
                return

            def end_trace(self, trace_id: str, **kwargs: Any) -> None:
                return

            def on_event_start(
                self,
                event_type: Any,
                payload: Optional[Dict[str, Any]] = None,
                event_id: str = "",
                parent_id: str = "",
                **kwargs: Any,
            ) -> None:
                return

            def on_event_end(
                self,
                event_type: Any,
                payload: Optional[Dict[str, Any]] = None,
                event_id: str = "",
                **kwargs: Any,
            ) -> None:
                # robust across LI versions (enum or str-ish)
                if "RETRIEVE" not in str(event_type).upper():
                    return

                payload = payload or {}
                nodes = payload.get("nodes") or payload.get("retrieved_nodes") or []
                adapter._emit(nodes=list(nodes), source="llamaindex", event="RETRIEVE")

            # ---- Some LI versions call this specialized hook ----
            def on_retrieve_end(self, payload: Dict[str, Any], **kwargs: Any) -> None:
                nodes = payload.get("nodes") or payload.get("retrieved_nodes") or []
                adapter._emit(nodes=list(nodes), source="llamaindex", event="RETRIEVE")

        return _Handler()
