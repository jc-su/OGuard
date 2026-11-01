"""Every relative link in the project's Markdown documents names a file that exists.

A code span is not Markdown, so `kernel[grid](...)` in one is no link."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SKIPPED = {".git", ".venv", ".claude", "build", "__pycache__"}
RECORDS = ROOT / "eval" / "records"  # archived inputs, whose copies of documents are data
LINK = re.compile(r"\]\(([^)#\s]+)(?:#[^)]*)?\)")
CODE = re.compile(r"`[^`]*`")
EXTERNAL = re.compile(r"^[a-z][a-z0-9+.-]*:")


def documents() -> list[Path]:
    result = []
    for parent, directories, files in ROOT.walk():
        directories[:] = [
            name for name in directories if name not in SKIPPED and parent / name != RECORDS
        ]
        result.extend(parent / name for name in files if name.endswith(".md"))
    return sorted(result)


def test_every_relative_link_resolves() -> None:
    broken = [
        f"{document.relative_to(ROOT)}: {target}"
        for document in documents()
        for target in LINK.findall(CODE.sub("", document.read_text()))
        if not EXTERNAL.match(target) and not (document.parent / target).exists()
    ]
    assert broken == []
