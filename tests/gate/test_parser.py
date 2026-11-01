"""The gate's parser reads every corpus module that `ptxas` assembles.

Each recorded PTX module of the corpus, from Numba, nvcc, and Triton, is
valid PTX: `ptxas` assembles it for sm_86. The parser of `GateParse.v`,
which the gate trusts to state what the text says, reads each one without
an error, and finds the entry kernels that the text declares.
"""

import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from eval.corpus.adapter import COLLECTIONS
from eval.corpus.run import TOOLCHAINS
from opgate.frontends import entry_kernels
from opgate.gate import export_terms, formal_dir

QUERY = """
Eval vm_compute in
  match parse_text text size with
  | Ok ks => inl (map fst (ks_entries ks))
  | Err e _ => inr e
  end.
"""


def _modules() -> list[Path]:
    return sorted(
        p
        for collection in COLLECTIONS
        for ptx in TOOLCHAINS.values()
        for p in collection.root.glob(f"*/*/{ptx}")
    )


def _assembles(path: Path) -> str | None:
    """Why `ptxas` does not assemble the module, or None."""
    ptxas = shutil.which("ptxas")
    assert ptxas is not None
    result = subprocess.run(
        [ptxas, "-arch=sm_86", "-o", "/dev/null", str(path)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    return None if result.returncode == 0 else result.stderr[-500:]


def _entries(path: Path) -> str:
    """Rocq's printed parse of the module: its entry names, or the parser's error."""
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "Parse.v"
        source.write_text(export_terms(path.read_text(), []) + QUERY)
        result = subprocess.run(
            ["coqc", "-q", "-Q", str(formal_dir()), "OpGate", str(source)],
            cwd=tmp,
            capture_output=True,
            text=True,
            timeout=600,
        )
    assert result.returncode == 0, result.stderr[-2000:]
    return result.stdout


@pytest.mark.slow
def test_the_parser_reads_every_module_that_ptxas_assembles() -> None:
    modules = _modules()
    assert len(modules) >= 96
    with ThreadPoolExecutor(max_workers=4) as pool:
        assembled = list(pool.map(_assembles, modules))
        parsed = list(pool.map(_entries, modules))
    unassembled = [(m, why) for m, why in zip(modules, assembled, strict=True) if why is not None]
    assert unassembled == []
    wrong = []
    for module, printed in zip(modules, parsed, strict=True):
        names = entry_kernels(module.read_text())
        if "inl" not in printed or not all(f'"{name}"' in printed for name in names):
            wrong.append((module, printed[-300:]))
    assert wrong == []
