"""PTX text, a signature, and a calling convention from each toolchain.

A front end obtains the PTX text of a kernel from its toolchain and pairs it
with the author's signature, the toolchain's calling convention, and the
block size of every launch (design section 5.9). A front end chooses how
its toolchain lowers a kernel, so that each loop keeps a test of its bound
(NVVM level 1, and the Numba front end's range iteration, design decision
25), and changes nothing that the kernel computes; it hands over the PTX
its toolchain emits. Each toolchain has a
module of its own, which alone imports it: `numba_operator` in
`opgate.frontends.numba_cuda`, `nvcc_operator` in `opgate.frontends.nvcc`,
and `triton_operator` in `opgate.frontends.triton`. The front ends are
untrusted: the gate verifies the text against the binding, whatever a front
end hands over.

Imports `opgate.signature` and `opgate.binding` only.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import NamedTuple

from opgate.binding import (
    Binding,
    Convention,
    ElementType,
    derive_binding,
    validate_grid,
    validate_shared,
)
from opgate.signature import Affine, Signature

__all__ = ["Operator", "Target", "entry_kernels", "entry_parameters"]

_ENTRY = re.compile(r"\.entry\s+([\w$]+)\s*\(([^)]*)\)", re.S)
_PARAMETER = re.compile(r"\s*\.param\b(?:.*\s)?([A-Za-z_$%][\w$]*)\s*(?:\[\d+\])?\s*", re.S)


class Target(NamedTuple):
    """The compute capability that a front end compiles for, such as 8.6."""

    major: int
    minor: int

    @property
    def arch(self) -> str:
        """The PTX target name, such as `sm_86`."""
        return f"sm_{self.major}{self.minor}"


def entry_kernels(text: str) -> tuple[str, ...]:
    """The names of the entry kernels of the module `text`, in order."""
    return tuple(name for name, _ in _ENTRY.findall(text))


def entry_parameters(text: str, kernel: str) -> tuple[str, ...]:
    """The PTX parameter names of the entry kernel `kernel` of the module `text`, in order."""
    found = [params for name, params in _ENTRY.findall(text) if name == kernel]
    if len(found) != 1:
        raise ValueError(f"the module has {len(found)} entry kernels named {kernel}")
    if not found[0].strip():
        return ()
    names: list[str] = []
    for declaration in found[0].split(","):
        matched = _PARAMETER.fullmatch(declaration)
        if matched is None:
            raise ValueError(f"{kernel}: a parameter declaration {declaration.strip()!r}")
        names.append(matched.group(1))
    return tuple(names)


@dataclass(frozen=True)
class Operator:
    """An untrusted kernel as its toolchain emits it, with the author's claim about its arguments."""

    text: str  # the PTX module
    kernel: str  # the name of the entry kernel in the text
    signature: Signature
    convention: Convention
    block: tuple[int, int, int]  # the block size of every launch
    grid: tuple[int, int, int]  # the largest grid of any launch, in blocks
    shared: Affine  # the bytes of dynamic shared memory of every launch

    def __post_init__(self) -> None:
        validate_grid(self.grid)
        validate_shared(self.signature.sizes, self.shared)
        entry_parameters(self.text, self.kernel)

    @property
    def parameters(self) -> tuple[str, ...]:
        """The PTX parameter names of the entry kernel, in order."""
        return entry_parameters(self.text, self.kernel)

    def binding(self, element_types: Mapping[str, ElementType]) -> Binding:
        """The binding of the kernel, for the element types of the plan's columns."""
        return derive_binding(
            self.signature,
            element_types,
            self.convention,
            self.kernel,
            self.parameters,
            self.block,
            self.grid,
            self.shared,
        )
