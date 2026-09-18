"""Minimal stage cache: rerun a stage only when its inputs (files or strings) changed."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from pathlib import Path


def _generated(rel: Path) -> bool:
    """Bytecode, caches and hidden files change without any source change, so they are not hashed."""
    return (any(part == "__pycache__" or part.startswith(".") for part in rel.parts)
            or rel.suffix in {".pyc", ".pyo"})


def fingerprint(items: Iterable[Path | str | bytes]) -> str:
    h = hashlib.sha256()
    for it in items:
        if isinstance(it, Path):
            h.update(b"P" + str(it).encode())
            if it.is_file():
                h.update(it.read_bytes())
            elif it.is_dir():
                for f in sorted(it.rglob("*")):
                    rel = f.relative_to(it)
                    if f.is_file() and not _generated(rel):
                        h.update(str(rel).encode() + f.read_bytes())
        elif isinstance(it, bytes):
            h.update(b"B" + it)
        else:
            h.update(b"S" + str(it).encode())
    return h.hexdigest()


class StageCache:
    def __init__(self, stamp_dir: Path):
        self.stamp_dir = stamp_dir
        stamp_dir.mkdir(parents=True, exist_ok=True)

    def run(self, name: str, inputs: Iterable[Path | str | bytes], outputs: Iterable[Path],
            fn: Callable[[], None], force: bool = False) -> bool:
        """Run ``fn`` unless the stamp matches and every output exists. Returns True if it ran."""
        outputs = list(outputs)
        fp = fingerprint(inputs)
        stamp = self.stamp_dir / f"{name}.json"
        if not force and stamp.exists() and all(o.exists() for o in outputs):
            if json.loads(stamp.read_text()).get("fingerprint") == fp:
                return False
        fn()
        stamp.write_text(json.dumps({"fingerprint": fp, "outputs": [str(o) for o in outputs]}))
        return True
