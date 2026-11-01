"""Corpus kernels compiled from their sources by the front ends.

`operator_of` compiles the kernel of a model and a task from its source in
a directory, `numba.py` by numba-cuda at NVVM level 1, `kernel.cu` by nvcc
with `-Xcicc -O1`, or `triton.py` by Triton with the module's types,
constants, and warps, with the signature of the reviewed roles. The corpus
directory holds the models' first kernels; a repair round's directory holds
a model's revision of a refused kernel (`eval/corpus/repair.py`).

Run from the repository root:

    uv run python -m eval.corpus.sources TOOLCHAIN MODEL[/TASK]...

writes the PTX of the toolchain's kernel of each named model and task, or of
every task of a named model, in its task directory, which the decision table
records: `numba.O1.ptx`, `kernel.O1.ptx`, or `triton.ptx`, for each task
with a source.
"""

import importlib.util
import inspect
import json
import re
import sys
from pathlib import Path
from types import ModuleType

from numba.core import types as numba_types

from eval.corpus.adapter import (
    claim_of,
    collection_of,
    declared_launch,
    directory_of,
    grid_of,
    read_raw,
)
from eval.corpus.run import TOOLCHAINS
from opgate.binding import GRID_LIMIT
from opgate.frontends import Operator, Target
from opgate.frontends.numba_cuda import numba_operator
from opgate.frontends.nvcc import nvcc_operator
from opgate.frontends.triton import triton_operator
from opgate.signature import Signature

__all__ = [
    "SOURCES",
    "TARGET",
    "Interface",
    "c_parameters",
    "interface_of",
    "operator_of",
    "source_module",
    "stated_interface",
]

SOURCES = {"numba": "numba.py", "nvcc": "kernel.cu", "triton": "triton.py"}
PROTOTYPE = re.compile(
    r'extern\s+"C"\s+__global__\s+void\s+(?:__launch_bounds__\s*\([^)]*\)\s*)?kernel\s*\(([^)]*)\)'
)
QUALIFIERS = {"const", "restrict", "__restrict", "__restrict__"}
COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)

TARGET = Target(8, 6)  # the corpus's target


def source_module(name: str, path: Path) -> ModuleType:
    """The Python source at `path`, loaded as a module named `name` with each
    character that a Python name cannot hold, such as the dots and hyphens of
    a model's name, replaced by an underscore."""
    spec = importlib.util.spec_from_file_location(re.sub(r"\W", "_", name), path)
    if spec is None or spec.loader is None:
        raise ValueError(f"{path} is not a module")
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    return source


def c_parameters(text: str) -> tuple[tuple[str, str], ...]:
    """The type and the name of each parameter of the C kernel `kernel` in `text`,
    the type without its qualifiers; comments are not parameters."""
    matched = PROTOTYPE.search(COMMENT.sub(" ", text))
    if matched is None:
        raise ValueError("the source has no prototype of `kernel`")
    parameters = []
    for declaration in matched.group(1).split(","):
        words = [w for w in re.findall(r"[A-Za-z_]\w*|\*", declaration) if w not in QUALIFIERS]
        parameters.append((" ".join(words[:-1]), words[-1]))
    return tuple(parameters)


# the element type of a buffer argument, by the name each toolchain gives it
C_ELEMENTS = {
    "double": "float64",
    "float": "float32",
    "long long": "int64",
    "long long int": "int64",
    "int64_t": "int64",
    "unsigned long long": "uint64",
    "uint64_t": "uint64",
    "int": "int32",
    "int32_t": "int32",
    "unsigned": "uint32",
    "unsigned int": "uint32",
    "uint32_t": "uint32",
    "short": "int16",
    "int16_t": "int16",
    "unsigned short": "uint16",
    "uint16_t": "uint16",
    "signed char": "int8",
    "int8_t": "int8",
    "unsigned char": "uint8",
    "uint8_t": "uint8",
    "bool": "bool",
}
TRITON_ELEMENTS = {
    "*fp64": "float64",
    "*fp32": "float32",
    "*i64": "int64",
    "*u64": "uint64",
    "*i32": "int32",
    "*u32": "uint32",
    "*i16": "int16",
    "*u16": "uint16",
    "*i8": "int8",
    "*u8": "uint8",
    "*i1": "bool",
}

Interface = tuple[tuple[str, str | None], ...]  # each argument's name, and a buffer's element type


def stated_interface(model: str, task: str) -> Interface:
    """The arguments that the task's `signature.json` states, in order."""
    raws = read_raw(directory_of(model, task) / "signature.json")
    return tuple((raw.name, None if raw.kind == "scalar" else raw.dtype) for raw in raws)


def interface_of(model: str, task: str, toolchain: str, directory: Path) -> Interface:
    """The arguments of the kernel's source in `directory`, in order: each name,
    and a buffer's element type."""
    path = directory / SOURCES[toolchain]
    match toolchain:
        case "numba":
            source = source_module(f"interface_{model}_{task}", path)
            names = tuple(inspect.signature(source.kernel).parameters)
            kinds = tuple(
                str(t.dtype) if isinstance(t, numba_types.Array) else None for t in source.TYPES
            )
        case "nvcc":
            parameters = c_parameters(path.read_text())
            names = tuple(name for _, name in parameters)
            kinds = tuple(
                C_ELEMENTS[kind.removesuffix("*").strip()] if kind.endswith("*") else None
                for kind, _ in parameters
            )
        case "triton":
            source = source_module(f"interface_{model}_{task}_triton", path)
            names = tuple(
                p
                for p in inspect.signature(source.kernel.fn).parameters
                if p not in source.CONSTANTS
            )
            kinds = tuple(
                TRITON_ELEMENTS[kind] if kind.startswith("*") else None
                for kind in source.TYPES.values()
            )
        case _:
            raise ValueError(f"no front end for {toolchain}")
    return tuple(zip(names, kinds, strict=True))


def _require_names(path: Path, names: tuple[str, ...], signature: Signature) -> None:
    """The kernel's arguments are the signature's, in order, since the binding
    assigns the parameters by position."""
    stated = tuple(argument.name for argument in signature.arguments)
    if names != stated:
        raise ValueError(f"{path}: the kernel's arguments {names} are not the signature's {stated}")


def triton_grid(directory: Path) -> tuple[int, int, int]:
    """The grid maxima that a task's `triton.json` states, or the GPU's limit."""
    path = directory / "triton.json"
    if not path.exists():
        return GRID_LIMIT
    loaded = json.loads(path.read_text())
    if not isinstance(loaded, dict):
        raise ValueError(f"{path}: not an object")
    return grid_of(path, loaded)


def operator_of(model: str, task: str, toolchain: str, directory: Path | None = None) -> Operator:
    """The kernel of `model` and `task`, compiled from its source in `directory`,
    the corpus's task directory by default, by the front end of `toolchain`.
    The kernel's arguments must be those of the task's signature, in order."""
    directory = directory if directory is not None else directory_of(model, task)
    signature, _ = claim_of(model, task)
    path = directory / SOURCES[toolchain]
    match toolchain:
        case "numba":
            signature, shared, block, grid = declared_launch(
                directory, toolchain, signature, collection_of(model).block
            )
            # the module name that the corpus's kernel names were mangled with
            source = source_module(f"corpus_{model}_{task}", path)
            _require_names(path, tuple(inspect.signature(source.kernel).parameters), signature)
            return numba_operator(
                source.kernel,
                source.TYPES,
                signature,
                block,
                TARGET,
                grid=grid,
                shared=shared,
            )
        case "nvcc":
            signature, shared, block, grid = declared_launch(
                directory, toolchain, signature, collection_of(model).block
            )
            text = path.read_text()
            _require_names(path, tuple(name for _, name in c_parameters(text)), signature)
            return nvcc_operator(
                text,
                "kernel",
                signature,
                block,
                TARGET,
                grid=grid,
                shared=shared,
            )
        case "triton":
            source = source_module(f"corpus_{model}_{task}_triton", path)
            names = tuple(
                p
                for p in inspect.signature(source.kernel.fn).parameters
                if p not in source.CONSTANTS
            )
            _require_names(path, names, signature)
            _require_names(path, tuple(source.TYPES), signature)
            return triton_operator(
                source.kernel,
                source.TYPES,
                source.CONSTANTS,
                signature,
                source.NUM_WARPS,
                TARGET,
                grid=triton_grid(directory),
            )
        case _:
            raise ValueError(f"no front end for {toolchain}")


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[0] not in SOURCES:
        raise SystemExit(f"usage: TOOLCHAIN MODEL[/TASK]..., with a toolchain of {sorted(SOURCES)}")
    toolchain = argv[0]
    kernels: list[tuple[str, str]] = []
    for name in argv[1:]:
        model, _, task = name.partition("/")
        if task and not (directory_of(model, task) / SOURCES[toolchain]).is_file():
            raise SystemExit(f"{model}/{task} has no {SOURCES[toolchain]}")
        root = collection_of(model).root / model
        tasks = [task] if task else sorted(p.parent.name for p in root.glob("*/signature.json"))
        kernels.extend(
            (model, t) for t in tasks if (directory_of(model, t) / SOURCES[toolchain]).is_file()
        )
    failed = []
    for model, task in kernels:
        try:
            operator = operator_of(model, task, toolchain)
        # a model's source can fail to compile in any way; the kernel then
        # has no PTX, and the run reports it
        except Exception as error:
            failed.append(f"{model}/{task}/{toolchain}: {str(error).splitlines()[0]}")
            continue
        (directory_of(model, task) / TOOLCHAINS[toolchain]).write_text(operator.text)
        if toolchain == "triton":
            # the launch metadata that the text does not state
            assert not operator.shared.terms
            declared: dict[str, object] = {"shared": operator.shared.constant}
            if operator.grid != GRID_LIMIT:
                declared["grid"] = list(operator.grid)
            metadata = json.dumps(declared) + "\n"
            (directory_of(model, task) / "triton.json").write_text(metadata)
        print(f"{model}/{task}/{toolchain}")
    for failure in failed:
        print(f"does not compile: {failure}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
