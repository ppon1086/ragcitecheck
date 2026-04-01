from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass(frozen=True)
class RecorderOptions:
    run_id: str
    docs_key: str = "retrieved"
    normalize_docs_to_str: bool = False


class Recorder:
    """Writes one JSONL record per query.

    Behavior:
      - normalize_docs_to_str=True: coerce doc entries to strings (legacy behavior).
      - normalize_docs_to_str=False: preserve doc entries (str OR dict) so adapters
        can log richer structures like {"doc_id":..., "chunk_id":...}.
    """

    def __init__(self, *, out_file: Union[str, Path], opts: RecorderOptions):
        self.opts = opts
        self.out_file = Path(out_file)
        self.out_file.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.out_file.open("w", encoding="utf-8")

    def _normalize_doc_entry(self, d: Any) -> Any:
        if not self.opts.normalize_docs_to_str:
            return d

        if isinstance(d, str):
            return d

        if isinstance(d, dict):
            if "doc_id" in d and d["doc_id"] is not None:
                return str(d["doc_id"])
            return str(d)

        return str(d)

    def record(self, *, query_id: str, docs: List[Any], extra: Optional[Dict[str, Any]] = None) -> None:
        rec: Dict[str, Any] = {
            "run_id": self.opts.run_id,
            "query_id": str(query_id),
            self.opts.docs_key: [self._normalize_doc_entry(d) for d in (docs or [])],
        }
        if extra:
            rec.update(extra)
        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass
