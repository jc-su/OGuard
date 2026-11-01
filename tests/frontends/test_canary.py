"""Canaries: each front end hands over the PTX shape that the gate expects.

Each toolchain compiles one reference kernel, `out[i] = 2 * values[i]` for
`i < n`. At NVVM level 1, the Numba and nvcc kernels keep a bound test in
every iteration of their grid-stride loop: no 64-bit division computes a
trip count, and no call is left (design section 5.9). The Numba kernel also
as `for i in range(cuda.grid(1), n, cuda.gridsize(1))`, which the front
end's range lowering tests against `n` (design decision 25). The gate
accepts all three.
Triton's kernel has one PTX parameter per argument and the two scratch
pointers, and its block is `32 * num_warps` threads; the gate accepts it
with 64-bit offsets and refuses it with 32-bit offsets, which wrap. A change
in a toolchain, such as numba-cuda's internal compile steps, fails a canary
before it changes a decision.
"""

import re
import struct
from collections.abc import Callable

import pytest
import triton
import triton.language as tl
from numba import cuda, float64, int64

from opgate import signature as sig
from opgate.binding import Base, ElementType, Size, Zero
from opgate.catalog import Launch, Step, TenantId, Write
from opgate.executor import Executor
from opgate.frontends import Operator, Target, entry_parameters
from opgate.frontends.numba_cuda import numba_operator
from opgate.frontends.nvcc import nvcc_operator
from opgate.frontends.triton import triton_operator
from opgate.gate import Accepted, Refused, verify, zeroed
from opgate.signature import Affine, Direction

TARGET = Target(8, 6)
SIGNATURE = sig.Signature(
    sizes=(sig.SizeSymbol("n"),),
    arguments=(
        sig.Span("values", "values", Direction.READ, Affine.symbol(0)),
        sig.Span("out", "out", Direction.WRITE, Affine.symbol(0)),
        sig.Scalar("n", sig.SizeOf(Affine.symbol(0))),
    ),
)
DOUBLES = {"values": ElementType(8, False), "out": ElementType(8, False)}
FLOATS = {"values": ElementType(4, False), "out": ElementType(4, False)}
DIVISION = re.compile(r"\b(div|rem)\.[us]64\b")
CALL = re.compile(r"\bcall(\.uni)?\b")


# The kernels have no annotations: their compilers type the arguments, and
# Triton reads an annotation as a type of its own.
def scale(values, out, n):
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    while i < n:
        out[i] = 2 * values[i]
        i += stride


# the same loop as a range, which the Numba front end lowers with a bound test
def scale_range(values, out, n):
    for i in range(cuda.grid(1), n, cuda.gridsize(1)):
        out[i] = 2 * values[i]


def small_reduce(values, out):
    shared = cuda.shared.array(8, float64)
    lane = cuda.threadIdx.x
    if lane < 8:  # ty: ignore[unsupported-operator]  # a device-side integer
        shared[lane] = values[lane]
    cuda.syncthreads()
    if lane == 0:
        total = 0.0
        for i in range(8):
            total += shared[i]
        out[0] = total


SCALE_CUDA = """
extern "C" __global__ void scale(const double* values, double* out, long long n)
{
    long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += stride)
        out[i] = 2 * values[i];
}
"""


@triton.jit
def scale_triton(values, out, n, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    tl.store(out + offsets, 2 * tl.load(values + offsets, mask=mask), mask=mask)


@triton.jit
def scale_triton_wide(values, out, n, BLOCK: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    tl.store(out + offsets, 2 * tl.load(values + offsets, mask=mask), mask=mask)


def numba_scale() -> Operator:
    return numba_operator(scale, (float64[:], float64[:], int64), SIGNATURE, (256, 1, 1), TARGET)


def numba_scale_range() -> Operator:
    return numba_operator(
        scale_range, (float64[:], float64[:], int64), SIGNATURE, (256, 1, 1), TARGET
    )


def nvcc_scale() -> Operator:
    return nvcc_operator(SCALE_CUDA, "scale", SIGNATURE, (256, 1, 1), TARGET)


@pytest.mark.parametrize(
    "make", [numba_scale, numba_scale_range, nvcc_scale], ids=["numba", "numba range", "nvcc"]
)
def test_a_grid_stride_loop_keeps_its_bound_test_and_is_accepted(
    make: Callable[[], Operator],
) -> None:
    operator = make()
    assert DIVISION.search(operator.text) is None
    assert CALL.search(operator.text) is None
    binding = operator.binding(DOUBLES)
    assert verify(zeroed(operator.text, binding), [binding]) == Accepted()


def test_numba_lays_out_an_array_as_seven_parameters() -> None:
    binding = numba_scale().binding(DOUBLES)
    assert len(binding.parameters) == 7 + 7 + 1
    assert isinstance(binding.parameters[4][1], Base)
    assert binding.parameters[14][1] == Size(Affine.symbol(0))


def test_a_constant_range_keeps_its_bound_for_shared_memory() -> None:
    signature = sig.Signature(
        sizes=(),
        arguments=(
            sig.Span("values", "values", Direction.READ, Affine(8)),
            sig.Span("out", "out", Direction.WRITE, Affine(1)),
        ),
    )
    operator = numba_operator(small_reduce, (float64[:], float64[:]), signature, (32, 1, 1), TARGET)
    binding = operator.binding(DOUBLES)
    assert verify(zeroed(operator.text, binding), [binding]) == Accepted()


def test_triton_has_one_parameter_per_argument_and_two_scratch_pointers() -> None:
    operator = triton_operator(
        scale_triton,
        {"values": "*fp32", "out": "*fp32", "n": "i32"},
        {"BLOCK": 256},
        SIGNATURE,
        4,
        TARGET,
    )
    assert operator.block == (128, 1, 1)
    assert CALL.search(operator.text) is None
    binding = operator.binding(FLOATS)
    assert len(binding.parameters) == 3 + 2
    assert [p for _, p in binding.parameters[3:]] == [Zero(), Zero()]


@pytest.mark.parametrize("block", [256, 1024], ids=["one element", "eight elements"])
def test_triton_offsets_of_64_bits_are_accepted_and_of_32_bits_refused(block: int) -> None:
    """Each thread of 128 loads `block / 128` elements, whose offsets are `or`s
    of disjoint bits; a 32-bit offset wraps on a grid of more than `2^31 / block` programs."""
    wide = triton_operator(
        scale_triton_wide,
        {"values": "*fp32", "out": "*fp32", "n": "i64"},
        {"BLOCK": block},
        SIGNATURE,
        4,
        TARGET,
    )
    wide_binding = wide.binding(FLOATS)
    assert verify(zeroed(wide.text, wide_binding), [wide_binding]) == Accepted()
    narrow = triton_operator(
        scale_triton,
        {"values": "*fp32", "out": "*fp32", "n": "i32"},
        {"BLOCK": block},
        SIGNATURE,
        4,
        TARGET,
    )
    narrow_binding = narrow.binding(FLOATS)
    assert isinstance(verify(zeroed(narrow.text, narrow_binding), [narrow_binding]), Refused)


def test_entry_parameters_read_each_declaration_form() -> None:
    text = """
.visible .entry k(
	.param .u64 k_param_0,
	.param .u64 .ptr .global .align 1 k_param_1,
	.param .align 8 .b8 k_param_2[16]
)
{
	ret;
}
"""
    assert entry_parameters(text, "k") == ("k_param_0", "k_param_1", "k_param_2")
    with pytest.raises(ValueError, match="0 entry kernels named other"):
        entry_parameters(text, "other")


def dynamic_copy(values, out):
    tile = cuda.shared.array(0, float64)
    lane = cuda.threadIdx.x
    tile[lane] = values[lane]
    cuda.syncthreads()
    out[lane] = tile[lane]


@pytest.mark.parametrize("toolchain", ["numba", "nvcc"])
@pytest.mark.gpu
def test_dynamic_shared_memory_reaches_the_binding_and_gate(toolchain):
    signature = sig.Signature(
        sizes=(),
        arguments=(
            sig.Span("values", "values", Direction.READ, Affine(32)),
            sig.Span("out", "out", Direction.WRITE, Affine(32)),
        ),
    )
    if toolchain == "numba":
        operator = numba_operator(
            dynamic_copy,
            (float64[:], float64[:]),
            signature,
            (32, 1, 1),
            TARGET,
            shared=Affine(256),
        )
    else:
        source = """extern "C" __global__ void dynamic_copy(double *values, double *out) {
            extern __shared__ double tile[];
            unsigned lane = threadIdx.x;
            tile[lane] = values[lane];
            __syncthreads();
            out[lane] = tile[lane];
        }"""
        operator = nvcc_operator(
            source, "dynamic_copy", signature, (32, 1, 1), TARGET, shared=Affine(256)
        )
    binding = operator.binding(DOUBLES)
    assert operator.shared == binding.shared == Affine(256)
    assert verify(zeroed(operator.text, binding), [binding]) == Accepted()
    tenant = TenantId(1)
    expected = struct.pack("<32d", *(float(i) for i in range(32)))
    with Executor(0, 1 << 20) as executor, executor.tenant(tenant):
        executor.reserve(1 << 20)
        module = executor.register(operator.text, binding)
        values, out = executor.allocate(256), executor.allocate(256)
        launch = Launch(module.operator, (1, 1, 1), (values, out), (), (), False)
        executor.execute(Step(tenant, (Write(values, expected, None), launch)))
        assert executor.read(tenant, out) == expected
