"""The Triton front end: the PTX of a compiled Triton kernel.

`triton_operator` compiles a `triton.jit` function for the types of its
arguments and the values of its `constexpr` arguments, and hands over the
PTX that Triton emits, with its line information (design section 5.9). A
`constexpr` argument is not a PTX parameter. Triton 3.8 appends two pointer
parameters, for global scratch memory and for profile scratch memory; the
front end refuses a kernel that needs either, so that the launch passes zero
for both (`Convention.TRITON`). The block has `32 * num_warps` threads, and
the launch passes the dynamic shared memory that Triton states.

Imports Triton, `opgate.signature`, `opgate.binding`, and `opgate.frontends`.
"""

from collections.abc import Mapping

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.runtime.jit import JITFunction

from opgate.binding import GRID_LIMIT, Convention, validate_grid
from opgate.frontends import Operator, Target
from opgate.signature import Affine, Signature

__all__ = ["triton_operator"]


def triton_operator(
    function: JITFunction,
    argument_types: Mapping[str, str],
    constants: Mapping[str, int | bool],
    signature: Signature,
    num_warps: int,
    target: Target,
    *,
    grid: tuple[int, int, int] = GRID_LIMIT,
) -> Operator:
    """The operator of `function`, with Triton's type of each argument, such as
    `*fp32` or `i32`, and the value of each `constexpr` argument, launched in
    grids of at most `grid` programs, any grid of the GPU unless the claim states
    less."""
    validate_grid(grid)
    types = {**argument_types, **{name: "constexpr" for name in constants}}
    source = ASTSource(fn=function, signature=types, constexprs=dict(constants))
    compiled = triton.compile(
        source,
        target=GPUTarget("cuda", 10 * target.major + target.minor, 32),
        options={"num_warps": num_warps},
    )
    metadata = compiled.metadata
    if metadata.global_scratch_size != 0 or metadata.profile_scratch_size != 0:
        raise ValueError(f"{metadata.name}: the kernel needs scratch memory")
    return Operator(
        compiled.asm["ptx"],
        metadata.name,
        signature,
        Convention.TRITON,
        (32 * num_warps, 1, 1),
        grid,
        Affine(metadata.shared),
    )
