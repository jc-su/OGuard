"""Durable, single-writer experiment records with exact resume provenance."""

import fcntl
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any


def _publish(path: Path, data: bytes) -> None:
    """Publish bytes only after their contents are durable."""
    fd, name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _artifact_path(record: Path, key: str, kind: str) -> Path:
    component = r"[A-Za-z0-9_-][A-Za-z0-9_.-]*"
    if not re.fullmatch(rf"{component}(?:/{component})*", key) or kind not in ("ptx", "v"):
        raise ValueError("invalid artifact key or kind")
    return record.with_suffix(record.suffix + ".artifacts") / f"{key}.{kind}"


def read_artifact(record: Path, key: str, kind: str) -> str:
    """Read a measured query or PTX file by its kernel/revision name."""
    return _artifact_path(record, key, kind).read_bytes().decode("ascii")


def snapshot_inputs(record: Path, group: str, files: dict[str, Path], *, create: bool) -> None:
    """Retain actual experiment inputs; compare bytes when resuming or replaying."""
    root = record.with_suffix(record.suffix + ".inputs") / group
    if create:
        root.mkdir(parents=True, exist_ok=False)
        for name, source in files.items():
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            _publish(target, source.read_bytes())
    else:
        names = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
        if names != set(files):
            raise ValueError(f"experiment {group} inputs changed; use a new output record")
        for name, source in files.items():
            if (root / name).read_bytes() != source.read_bytes():
                raise ValueError(
                    f"experiment {group} input changed: {name}; use a new output record"
                )


class Checkpoint:
    """Hold the output lock for the entire run; publish each result atomically.

    A resumed run must use the same provenance. Failed/incomplete runs retain
    their completed results, but consumers must require ``complete`` before
    reporting an aggregate. This is an experiment record, not a gate cache.
    """

    def __init__(self, path: Path, provenance: dict[str, Any], *, resume: bool = False):
        self.data: dict[str, Any]
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = path.with_suffix(path.suffix + ".lock").open("a")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if resume:
                self.data = json.loads(path.read_text())
                if self.data["schema"] != 2 or self.data["provenance"] != provenance:
                    raise ValueError("resume inputs or settings changed; use a new output record")
                ids = [row["id"] for row in self.data["results"]]
                if len(ids) != len(set(ids)):
                    raise ValueError("duplicate results in checkpoint")
            else:
                if path.exists():
                    raise FileExistsError(f"{path} exists; use --resume or a new output record")
                self.data = {
                    "schema": 2,
                    "provenance": provenance,
                    "complete": False,
                    "results": [],
                }
                self._write()
        except BaseException:
            self._lock.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._lock.close()

    def _write(self) -> None:
        data = json.dumps(self.data, indent=2, sort_keys=True, allow_nan=False) + "\n"
        _publish(self.path, data.encode("utf-8"))

    def artifact(self, key: str, text: str, kind: str) -> str:
        """Save measured bytes under the kernel/revision name before its result."""
        data = text.encode("ascii")
        path = _artifact_path(self.path, key, kind)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.read_bytes() != data:
                raise ValueError(f"artifact already exists with different contents: {path}")
        else:
            _publish(path, data)
        return key

    def get(self, key: str) -> dict[str, Any] | None:
        return next((r for r in self.data["results"] if r["id"] == key), None)

    def add(self, row: dict[str, Any]) -> None:
        if self.data["complete"]:
            raise ValueError("cannot append to a completed record")
        if self.get(row["id"]) is not None:
            raise ValueError(f"duplicate result: {row['id']}")
        self.data["results"].append(row)
        self._write()

    def finish(self, **summary: Any) -> None:
        self.data.update(summary, complete=True)
        self._write()
