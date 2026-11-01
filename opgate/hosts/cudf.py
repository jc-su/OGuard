"""The cuDF host: `Series.apply` and `DataFrame.apply` through the gate and the executor.

cuDF compiles an `apply` UDF into a kernel from its own template and launches
it with `forall` (design section 8, phase 2). Inside `install_cudf`,
`ApplyKernelBase._compile_or_get_kernel` gives a `GatedKernel` in its place:
the Numba front end compiles cuDF's kernel at NVVM level 1 with a signature
from the frame, the executor verifies and loads it, through the guard pass
when the gate refuses an access (`opgate.hosts.register`), and `forall(n)`
launches it as a step of the session's tenant, whose buffers are the pool
allocations of the arguments; a guarded kernel's status buffer is a buffer
of the tenant that the resource allocates (`opgate.hosts.launch`). The
signature: the output's data and validity are write spans of `size` rows;
each input column is a read span of `size + offset` rows, its offset a size
symbol, so a sliced column needs no sub-buffer; each extra argument is a
scalar of any value. A column with a
null mask is refused, since its bitmask's words are not an affine count of
rows, and so are a string column and a non-numeric result, which cuDF's
string shim computes. A group-by UDF, whose kernel takes groups and calls
cuDF's shim, is refused. A UDF aggregation, as `Rolling.apply` or a function in
`GroupBy.agg` makes, is refused too: libcudf compiles its PTX into a kernel
of its own, which the gate does not see. Inside `install_cudf`, CuPy
allocates through RMM, so that cuDF's output arrays are pool buffers too.

Imports cuDF, CuPy, RMM, numba-cuda, `opgate.frontends`, and `opgate.executor`.
"""

import gc
import struct
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal, NoReturn

import cupy
import numpy as np
from cudf.core._internals.aggregation import Aggregation
from cudf.core.udf.nrt_utils import CaptureNRTUsage
from cudf.core.udf.row_function import DataFrameApplyKernel
from cudf.core.udf.scalar_function import SeriesApplyKernel
from cudf.core.udf.udf_kernel_base import ApplyKernelBase
from cudf.core.udf.utils import _generate_cache_key, _supported_cols_from_frame
from cudf.utils.scalar import pa_scalar_to_plc_scalar
from numba.core import types
from numba.np import numpy_support
from rmm.allocators.cupy import rmm_cupy_allocator

from opgate import signature as sig
from opgate.binding import ElementType
from opgate.catalog import GRID_LIMIT, BufferId, Launch, View
from opgate.executor import Executor, Module, RefusedError
from opgate.frontends import Target
from opgate.frontends.numba_cuda import numba_operator
from opgate.hosts import Registered, launch, register
from opgate.hosts.rapids import CatalogResource
from opgate.signature import Affine, Direction

__all__ = ["GatedKernel", "clear_cudf_caches", "install_cudf", "release_cudf_query_state"]

BLOCK = (256, 1, 1)
# the kernels of `Series.apply` and `DataFrame.apply`, which take one row per
# thread; a group-by UDF's kernel takes groups
BOUND = frozenset({"series_apply", "dataframe_apply"})


WIDTHS: dict[int, Literal[1, 2, 4, 8]] = {1: 1, 2: 2, 4: 4, 8: 8}


def release_cudf_query_state() -> None:
    """Release global cuDF caches that can retain a query's device buffers.

    cuDF caches each UDF helper's execution context by the helper instance.
    The helper owns its input frame, so leaving those caches populated carries
    a tenant's columns into the next request. cuDF also caches device scalars.
    A serialized multi-tenant host calls this after every query while the
    allocation wrapper is still installed; deallocation then uses the owner
    recorded when each allocation was made.
    """
    clear_cudf_caches()
    gc.collect()


def clear_cudf_caches() -> None:
    """Clear cuDF's UDF execution contexts and device scalars, without collecting
    the objects they held; `release_cudf_query_state` also collects them."""
    SeriesApplyKernel._get_kernel_string_exec_context.cache_clear()
    DataFrameApplyKernel._get_kernel_string_exec_context.cache_clear()
    pa_scalar_to_plc_scalar.cache_clear()


def _element(dtype: object) -> ElementType:
    """The element type of a numeric NumPy dtype; a string column's dtype is not one."""
    width = WIDTHS.get(dtype.itemsize) if isinstance(dtype, np.dtype) else None
    if not isinstance(dtype, np.dtype) or width is None or dtype.kind not in "biuf":
        raise RefusedError(f"a column or result of type {dtype}, which the hook does not bind")
    return ElementType(width, dtype.kind == "i")


def _bits(value: object) -> tuple[int, int]:
    """An extra argument as the unsigned integer its bits spell, and its width in bits."""
    if isinstance(value, np.generic):
        if value.dtype.kind not in "biuf" or value.dtype.itemsize not in WIDTHS:
            raise RefusedError(
                f"an extra argument of type {value.dtype}, which the hook does not bind"
            )
        raw = np.asarray(value, dtype=value.dtype.newbyteorder("<")).tobytes()
        return int.from_bytes(raw, "little"), len(raw) * 8
    match value:
        case bool():
            return int(bool(value)), 8
        case int():
            return int(value) % 2**64, 64
        case float():
            (bits,) = struct.unpack("<Q", struct.pack("<d", float(value)))
            return bits, 64
        case _:
            raise RefusedError(f"an extra argument {value!r}, which the hook does not bind")


def _pointer(value: object) -> int:
    interface = getattr(value, "__cuda_array_interface__", None)
    if not isinstance(interface, dict):
        raise RefusedError(f"an argument {type(value).__name__} with no device address")
    data = interface["data"]
    if not isinstance(data, tuple) or not isinstance(data[0], int):
        raise RefusedError("an argument whose device address is not an integer")
    return data[0]


@dataclass(frozen=True)
class GatedKernel:
    """cuDF's apply kernel, verified and loaded by the executor; `forall(n)`
    launches it through the executor as `numba`'s kernels launch."""

    executor: Executor
    resource: CatalogResource
    module: Module
    columns: int
    on_launch: Callable[[Module], None] | None

    def forall(self, rows: int) -> Callable[..., None]:
        return lambda *arguments: self._launch(rows, arguments)

    def _buffer(self, value: object) -> BufferId | View:
        return self.resource.argument_at(_pointer(value))

    def _launch(self, rows: int, arguments: tuple[object, ...]) -> None:
        # cuDF's order: (data, validity), size, the columns, their offsets, the extras
        output, size, *rest = arguments
        if not isinstance(output, tuple) or len(output) != 2:
            raise RefusedError("an output that is not the pair of data and validity")
        data, valid = output
        columns, offsets, extras = (
            rest[: self.columns],
            rest[self.columns : 2 * self.columns],
            rest[2 * self.columns :],
        )
        sizes: list[int] = []
        for value in (size, *offsets):
            if not isinstance(value, int):
                raise RefusedError(f"a size or an offset {value!r} that is not an integer")
            sizes.append(value)
        buffers = (self._buffer(data), self._buffer(valid), *map(self._buffer, columns))
        scalars = tuple(_bits(extra)[0] for extra in extras)
        blocks = max(1, min(GRID_LIMIT[0], -(-rows // BLOCK[0])))
        request = Launch(
            self.module.operator, (blocks, 1, 1), buffers, tuple(sizes), scalars, False
        )
        launch(self.executor, self.resource, request)
        if self.on_launch is not None:
            self.on_launch(self.module)


def _signature(
    kernel: ApplyKernelBase, result: np.dtype
) -> tuple[sig.Signature, dict[str, ElementType]]:
    """The signature of cuDF's apply kernel over the kernel's frame."""
    columns = list(_supported_cols_from_frame(kernel.frame).values())
    if any(column.mask is not None for column in columns):
        raise RefusedError("a column with a null mask, whose bitmask the hook does not bind")
    size = Affine.symbol(0)
    rows = [Affine(0, ((0, 1), (k + 1, 1))) for k in range(len(columns))]
    extras = [_bits(argument)[1] for argument in kernel.args]
    arguments: list[sig.Argument] = [
        sig.Span("out", "out", Direction.WRITE, size),
        sig.Span("valid", "valid", Direction.WRITE, size),
        sig.Scalar("size", sig.SizeOf(size)),
        *(
            sig.Span(f"column{k}", f"column{k}", Direction.READ, rows[k])
            for k in range(len(columns))
        ),
        *(sig.Scalar(f"offset{k}", sig.SizeOf(Affine.symbol(k + 1))) for k in range(len(columns))),
        *(sig.Scalar(f"extra{j}", sig.Between(0, 2**bits - 1)) for j, bits in enumerate(extras)),
    ]
    sizes = (sig.SizeSymbol("size"), *(sig.SizeSymbol(f"offset{k}") for k in range(len(columns))))
    elements = {"out": _element(result), "valid": ElementType(1, False)}
    for k, column in enumerate(columns):
        elements[f"column{k}"] = _element(column.dtype)
    return sig.Signature(sizes, tuple(arguments)), elements


def _gated(
    kernel: ApplyKernelBase,
    executor: Executor,
    resource: CatalogResource,
    target: Target,
    on_register: Callable[[Registered], None] | None,
    on_launch: Callable[[Module], None] | None,
) -> tuple[GatedKernel, np.dtype]:
    """Compile cuDF's kernel for `kernel`'s UDF, frame, and arguments at NVVM level 1,
    and have the executor verify and load it."""
    usage = CaptureNRTUsage()
    with usage:
        returned = kernel._get_udf_return_type()
    if usage.use_nrt or not returned.is_internal:
        raise RefusedError("a UDF that uses cuDF's string runtime")
    result = numpy_support.as_dtype(returned)
    kernel.sig = kernel._construct_signature(returned)
    context = kernel._get_kernel_string_exec_context()
    context["f_"] = kernel.device_func
    exec(kernel._get_kernel_string(), context)
    signature, elements = _signature(kernel, result)
    arguments = tuple(kernel.sig.args)
    if not all(isinstance(a, types.Type) for a in arguments):
        raise RefusedError("a kernel argument that is not a Numba type")
    operator = numba_operator(context["_kernel"], arguments, signature, BLOCK, target)
    registered = register(executor, operator.text, operator.binding(elements))
    if on_register is not None:
        on_register(registered)
    columns = len(_supported_cols_from_frame(kernel.frame))
    return GatedKernel(executor, resource, registered.module, columns, on_launch), result


def _refuse_udf_aggregation(cls: type, op: object, *args: object, **kwargs: object) -> NoReturn:
    name = getattr(op, "__name__", type(op).__name__)
    raise RefusedError(
        f"a UDF aggregation of {name}, which libcudf compiles into a kernel of its own"
    )


@contextmanager
def install_cudf(
    executor: Executor,
    resource: CatalogResource,
    target: Target,
    *,
    on_register: Callable[[Registered], None] | None = None,
    on_launch: Callable[[Module], None] | None = None,
) -> Iterator[None]:
    """Within the block, every `Series.apply` and `DataFrame.apply` kernel is
    verified and launched through the executor, every UDF aggregation is
    refused, and CuPy allocates through RMM, whose current resource
    `resource` must be."""
    # cuDF caches device scalars globally. They must not retain addresses in
    # an executor's pool after its lifetime, or be reused from another pool.
    release_cudf_query_state()
    original = ApplyKernelBase._compile_or_get_kernel
    # the classmethod itself, which the block restores
    original_udf = Aggregation.__dict__["from_udf"]
    allocator = cupy.cuda.get_allocator()
    kernels: dict[object, tuple[GatedKernel, np.dtype]] = {}

    def compile_or_get(kernel: ApplyKernelBase) -> tuple[GatedKernel, np.dtype]:
        if kernel.kernel_type not in BOUND:
            raise RefusedError(
                f"a {kernel.kernel_type} UDF, whose kernel takes groups and calls cuDF's shim"
            )
        key = _generate_cache_key(kernel.frame, kernel.func, kernel.args, suffix=kernel.kernel_type)
        if key not in kernels:
            kernels[key] = _gated(
                kernel,
                executor,
                resource,
                target,
                on_register,
                on_launch,
            )
        return kernels[key]

    # the hook replaces cuDF's method on its class, whose declared type is the
    # method's own and not a function of the kernel alone
    ApplyKernelBase._compile_or_get_kernel = compile_or_get  # ty: ignore[invalid-assignment]
    # likewise for the UDF aggregation's constructor
    Aggregation.from_udf = classmethod(_refuse_udf_aggregation)  # ty: ignore[invalid-assignment]
    cupy.cuda.set_allocator(rmm_cupy_allocator)
    try:
        yield
    finally:
        ApplyKernelBase._compile_or_get_kernel = original
        Aggregation.from_udf = original_udf
        cupy.cuda.set_allocator(allocator)
        release_cudf_query_state()
