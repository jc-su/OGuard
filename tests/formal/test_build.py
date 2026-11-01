"""A failed rebuild cannot replace a usable proof library."""

import os
import shutil
import subprocess
from pathlib import Path


def test_build_publishes_a_complete_snapshot_and_preserves_running_readers(tmp_path):
    makefile = Path(__file__).resolve().parents[2] / "formal" / "Makefile"
    shutil.copy(makefile, tmp_path)
    compiler = tmp_path / "coqc"
    compiler.write_text(
        "#!/bin/sh\nset -eu\n"
        'if [ "$5" = Two.v ] && [ "${FAIL_BUILD:-0}" = 1 ]; then exit 9; fi\n'
        'cp "$5" "${5%.v}.vo"\n'
    )
    compiler.chmod(0o755)
    for name in ("One", "Two"):
        (tmp_path / f"{name}.v").write_text("version one")
    command = ["make", "build", "FILES=One.v Two.v", f"COQC={compiler}"]
    subprocess.run(command, cwd=tmp_path, check=True, capture_output=True)
    first = (tmp_path / "build/current").resolve()
    assert (first / "Two.vo").read_text() == "version one"
    (tmp_path / "One.v").write_text("version two")
    failed = subprocess.run(
        command, cwd=tmp_path, env={**os.environ, "FAIL_BUILD": "1"}, capture_output=True
    )
    assert failed.returncode != 0
    assert (tmp_path / "build/current").resolve() == first
    assert len(list((tmp_path / "build").glob("library.*"))) == 1
    subprocess.run(command, cwd=tmp_path, check=True, capture_output=True)
    second = (tmp_path / "build/current").resolve()
    assert first != second
    assert (second / "One.vo").read_text() == "version two"
    assert (second / "Two.vo").read_text() == "version one"
    assert (first / "One.vo").read_text() == "version one"
