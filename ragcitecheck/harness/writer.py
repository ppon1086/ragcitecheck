from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional,Union


@dataclass(frozen=True)
class WriterOptions:
    encoding: str = "utf-8"
    ensure_ascii: bool = False
    sort_keys: bool = False


class JSONLWriter:
    """
    Appends JSONL records to a file.
    """
    def __init__(self, path: Union[str,Path], opts: Optional[WriterOptions] = None):
        self.path = Path(path)
        self.opts = opts or WriterOptions()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, obj: Dict[str, Any]) -> None:
        line = json.dumps(
            obj,
            ensure_ascii=self.opts.ensure_ascii,
            sort_keys=self.opts.sort_keys,
        )
        with self.path.open("a", encoding=self.opts.encoding) as f:
            f.write(line + "\n")

    def write_many(self, objs: Iterable[Dict[str, Any]]) -> None:
        with self.path.open("a", encoding=self.opts.encoding) as f:
            for obj in objs:
                line = json.dumps(
                    obj,
                    ensure_ascii=self.opts.ensure_ascii,
                    sort_keys=self.opts.sort_keys,
                )
                f.write(line + "\n")
