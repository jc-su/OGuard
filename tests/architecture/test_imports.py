"""The import rules of the architecture's package table.

Each package of `opgate` imports only the packages that the table in
docs/ARCHITECTURE.md allows, and a module of its own package. The core,
`signature`, `binding`, `catalog`, and `gate`, imports only the standard
library besides; the executor adds the CUDA driver; each front end imports
its toolchain in a module of its own; only `hosts` imports an engine.
"""

import ast
import sys
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[2] / "opgate"

# The opgate packages that each package may import, as the architecture's table states.
ALLOWED: dict[str, set[str]] = {
    "signature": set(),
    "binding": {"signature"},
    # the Rocq printer of `catalog.rocq` uses the gate's binding terms and library
    "catalog": {"signature", "binding", "gate"},
    "gate": {"binding"},
    "executor": {"binding", "catalog", "gate"},
    # the guard pass reads the gate's decisions
    "frontends": {"signature", "binding", "gate"},
    "hosts": {"signature", "binding", "catalog", "gate", "frontends", "executor"},
}

# The third-party packages that a module may import, beyond the standard library.
TOOLCHAINS: dict[str, set[str]] = {
    "executor/driver.py": {"cuda"},
    # the lowering of range builds LLVM IR with llvmlite, Numba's IR library
    "frontends/numba_cuda.py": {"numba", "llvmlite"},
    "frontends/triton.py": {"triton"},
    # cuDF's UDF path compiles with Numba and allocates its outputs with CuPy
    "hosts/cudf.py": {"cupy", "numba", "numpy"},
    "hosts/service/fast.py": {"numba", "numpy"},
}

ENGINES = {"cudf", "pylibcudf", "rmm", "sirius", "pyspark", "torch"}


def imports(path: Path) -> set[str]:
    """The dotted names of the modules that `path` imports."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, f"{path}: a relative import"
            assert node.module is not None
            # `from opgate import signature` imports the package it names
            if node.module == "opgate":
                names |= {f"opgate.{alias.name}" for alias in node.names}
            else:
                names.add(node.module)
    return names


def modules() -> list[Path]:
    return sorted(PACKAGE.rglob("*.py"))


@pytest.mark.parametrize("path", modules(), ids=lambda p: str(p.relative_to(PACKAGE)))
def test_a_module_imports_only_what_the_architecture_allows(path: Path) -> None:
    relative = path.relative_to(PACKAGE)
    package = relative.parts[0].removesuffix(".py")
    for name in imports(path):
        top = name.split(".")[0]
        if top == "opgate":
            imported = name.split(".")[1]
            assert imported == package or imported in ALLOWED[package], f"{relative} imports {name}"
        elif top in ENGINES:
            assert package == "hosts", f"{relative} imports the engine {name}"
        else:
            allowed = TOOLCHAINS.get(relative.as_posix(), set())
            assert top in sys.stdlib_module_names or top in allowed, f"{relative} imports {name}"


def test_the_table_names_every_package() -> None:
    packages = {path.relative_to(PACKAGE).parts[0].removesuffix(".py") for path in modules()}
    assert packages - {"__init__"} == set(ALLOWED)
