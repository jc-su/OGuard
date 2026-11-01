"""Stress: a region that a tenant frees while its use still writes it reaches
the next tenant only after that use ends (design decision 48).

Tenant A loops: it reserves a region of the pool, allocates 64 MiB at its
start, submits a prepared use whose fills write a canary byte over the 64 MiB
for about 10 ms, and frees its buffers and ends its reservation without
waiting for the use. Tenant B then reserves the same region, allocates a
buffer at its start, and a prepared use of B reads the buffer back: every
byte of it must differ from the canary. The negative control removes the two
guards that order B's work after A's use, so that `quiesce` waits for nothing
and the tenants' streams are non-blocking, and it must read the canary.

It runs on GPU 1 only:
`CUDA_VISIBLE_DEVICES=1 uv run pytest -m gpu tests/gpu/test_reuse_order.py`.
"""

import os

import pytest

from opgate import signature as sig
from opgate.binding import ElementType
from opgate.catalog import Fill, Launch, Step, TenantId
from opgate.executor import READ_BYTES, Executor, Module
from opgate.executor.driver import call, driver
from opgate.frontends import Target
from opgate.frontends.nvcc import nvcc_operator
from opgate.signature import Affine, Direction

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("CUDA_VISIBLE_DEVICES") != "1",
        reason="the stress test runs on GPU 1: set CUDA_VISIBLE_DEVICES=1",
    ),
]

ALICE, BOB = TenantId(1), TenantId(2)
BYTE = ElementType(1, False)
DOUBLE = ElementType(8, False)
CANARY = 0xA5
REGION = 64 << 20
FILLS = 110  # 110 memsets of 64 MiB take about 10 ms at the RTX A6000's 768 GB/s
ROUNDS = 20
POOL = 96 << 20
SCALE = sig.Signature(
    sizes=(sig.SizeSymbol("n"),),
    arguments=(
        sig.Span("values", "values", Direction.READ, Affine.symbol(0)),
        sig.Span("out", "out", Direction.WRITE, Affine.symbol(0)),
        sig.Scalar("n", sig.SizeOf(Affine.symbol(0))),
    ),
)
SOURCE = """
extern "C" __global__ void scale(const double* values, double* out, long long n)
{
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = 2 * values[i];
}
"""


def _register(executor: Executor) -> Module:
    operator = nvcc_operator(SOURCE, "scale", SCALE, (256, 1, 1), Target(8, 6))
    return executor.register(operator.text, operator.binding({"values": DOUBLE, "out": DOUBLE}))


def _canaries(executor: Executor) -> int:
    """The rounds in which B read a canary byte from the region that A freed."""
    module = _register(executor)
    rows = READ_BYTES // 8
    seen = 0
    for _ in range(ROUNDS):
        with executor.tenant(ALICE):
            executor.reserve(REGION + 512)
            region = executor.allocate(REGION)
            values, out = executor.allocate(256), executor.allocate(256)
            placed = executor.catalog.buffer(region)
            assert placed is not None
            fills = tuple(Fill(region, BYTE, CANARY) for _ in range(FILLS))
            launch = Launch(module.operator, (1, 1, 1), (values, out), (32,), (), False)
            writer = executor.prepare(Step(ALICE, (*fills, launch)), out)
            executor.submit(writer, ())
            # without waiting for the writer
            executor.free(region, values, out)
            executor.unreserve()
        with executor.tenant(BOB):
            executor.reserve(2 * READ_BYTES)
            reused, result = executor.allocate(READ_BYTES), executor.allocate(READ_BYTES)
            found = executor.catalog.buffer(reused)
            assert found is not None and found.offset == placed.offset
            grid = (rows // 256, 1, 1)
            launch = Launch(module.operator, grid, (reused, result), (rows,), (), False)
            reader = executor.prepare(Step(BOB, (launch,)), reused)
            executor.submit(reader, ())
            data = executor.result(reader)
            executor.unprepare(reader)
            executor.free(reused, result)
            executor.unreserve()
        seen += bytes([CANARY]) in data
        with executor.tenant(ALICE):
            executor.result(writer)
            executor.unprepare(writer)
    return seen


def test_a_freed_region_reaches_the_next_tenant_after_the_use_that_writes_it() -> None:
    with Executor(0, POOL) as executor:
        assert _canaries(executor) == 0


def test_without_its_guards_the_next_tenant_reads_the_canary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def non_blocking(self: Executor) -> object:
        (stream,) = call(driver.cuStreamCreate(driver.CUstream_flags.CU_STREAM_NON_BLOCKING))
        return stream

    monkeypatch.setattr(Executor, "quiesce", lambda self: None)
    monkeypatch.setattr(Executor, "_blocking_stream", non_blocking)
    with Executor(0, POOL) as executor:
        assert _canaries(executor) > 0
