"""A corpus kernel's launch under a baseline, with the parameters that the binding lays out.

The binding states what each PTX parameter of a kernel holds
(`opgate.binding`): for a Numba array, the seven fields of Numba's array
structure (meminfo and parent, both zero, the row count, the element width,
the data pointer, the shape, and the stride); for an nvcc or Triton
argument, the pointer or the scalar; and for Triton, two trailing zeros. The
catalog fills each parameter of a checked launch from its buffers, sizes, and
scalars (`opgate.catalog`), and the executor passes each with the byte size
that the driver states for the loaded function. A baseline launch, which runs
without the catalog, fills the binding's parameters the same way with
`parameters`, appends its own (Guardian's base and mask), and passes each
with the driver's byte size with `launch`.
"""

import ctypes
from collections.abc import Sequence
from typing import assert_never

from opgate.binding import Base, Binding, Bounded, Constant, Size, Zero
from opgate.catalog import evaluate
from opgate.executor.driver import call, driver, parameter_sizes

__all__ = ["parameters", "launch"]

UNSIGNED: dict[int, type[ctypes.c_uint8 | ctypes.c_uint16 | ctypes.c_uint32 | ctypes.c_uint64]] = {
    1: ctypes.c_uint8,
    2: ctypes.c_uint16,
    4: ctypes.c_uint32,
    8: ctypes.c_uint64,
}


def parameters(
    binding: Binding,
    addresses: Sequence[int],
    sizes: tuple[int, ...],
    scalars: Sequence[int],
) -> tuple[int, ...]:
    """The value of each PTX parameter of `binding`'s kernel, in order: each
    buffer base holds the next of `addresses`, in the order of the binding's
    buffer parameters; each size holds its affine of `sizes`, one value per
    size symbol of the binding; each bounded scalar holds the next of
    `scalars`, as the unsigned integer that its bits spell; each constant its
    value; and each zero 0."""
    if len(sizes) != len(binding.sizes):
        raise ValueError(f"{len(sizes)} sizes for the {len(binding.sizes)} symbols of the binding")
    bases = sum(1 for _, p in binding.parameters if isinstance(p, Base))
    bounded = sum(1 for _, p in binding.parameters if isinstance(p, Bounded))
    if (len(addresses), len(scalars)) != (bases, bounded):
        raise ValueError(
            f"{len(addresses)} addresses and {len(scalars)} scalars for a binding "
            f"with {bases} buffers and {bounded} scalars"
        )
    next_address, next_scalar = iter(addresses), iter(scalars)
    values: list[int] = []
    for name, parameter in binding.parameters:
        match parameter:
            case Base():
                value = next(next_address)
            case Size(value=affine):
                value = evaluate(affine, sizes)
            case Constant(value=constant):
                value = constant
            case Bounded(lower=lower, upper=upper):
                value = next(next_scalar)
                if not lower <= value <= upper:
                    raise ValueError(f"{name}: the scalar {value} lies outside [{lower}, {upper}]")
            case Zero():
                value = 0
            case _ as never:
                assert_never(never)
        values.append(value)
    return tuple(values)


def launch(
    function: object,
    grid: tuple[int, int, int],
    block: tuple[int, int, int],
    values: tuple[int, ...],
) -> None:
    """Launch the loaded `function` with `values` as its parameters, each
    passed with the byte size that the driver states for it. A count that
    differs from the function's, or a value outside its parameter's bits, is
    an error."""
    widths = parameter_sizes(function)
    if len(widths) != len(values):
        raise ValueError(f"{len(values)} values for a function with {len(widths)} parameters")
    for index, (value, width) in enumerate(zip(values, widths, strict=True)):
        if width not in UNSIGNED or not 0 <= value < 2 ** (8 * width):
            raise ValueError(f"parameter {index} of {width} bytes cannot hold {value}")
    types = tuple(UNSIGNED[width] for width in widths)
    call(driver.cuLaunchKernel(function, *grid, *block, 0, 0, (values, types), 0))
