"""A kernel reads what an earlier kernel left in shared memory unless it begins
with the zeroing prologue (design decision 46).

A writer kernel, another tenant's, fills a shared array with a pattern, with
one block on each multiprocessor. A reader kernel then copies the shared
array that it declares, and never writes, to global memory. As written, the reader reads
the pattern: nothing clears shared memory between the kernels of a context.
The gate refuses the reader as written and accepts it with the prologue that
`opgate.gate.zeroed` puts at its start, and with the prologue it reads zeros.

`Executor.register` puts the prologue at the start of every kernel it loads.
DeepSeek's CUDA C++ Q6 kernel then gives the revenue of the rows that pass
Q6's filters, and its negative control, without the store that initializes
its shared array, sums zeros after the writer has run: its revenue is 0.
"""

import ctypes
import struct

import numpy as np
import pytest

from eval.corpus.adapter import binding_of, directory_of
from eval.corpus.run import TOOLCHAINS
from opgate.binding import Base, Binding, ElementType, Size, Span
from opgate.catalog import Fill, Launch, Step, TenantId, Write
from opgate.executor import Executor
from opgate.executor.driver import call, driver, integer
from opgate.gate import Accepted, Refused, verify, zeroed
from opgate.signature import Affine, Direction, SizeSymbol

THREADS = 256
BYTES = 4096  # the shared array: four words of each thread
PATTERN = 0x5A5A5A5A

HEAD = """.version 8.0
.target sm_80
.address_size 64
.visible .entry {name}(.param .u64 out)
.reqntid 256
{{
.reg .b32 %r<16>;
.reg .b64 %rd<8>;
.shared .align 4 .b8 buf[4096];
ld.param.u64 %rd1, [out];
mov.u32 %r1, %tid.x;
mov.u32 %r2, %ctaid.x;
shl.b32 %r3, %r1, 2;
mov.u32 %r4, buf;
add.s32 %r5, %r4, %r3;
"""

WRITER = (
    HEAD.format(name="writer")
    + f"""mov.u32 %r6, {PATTERN};
st.shared.u32 [%r5+0], %r6;
st.shared.u32 [%r5+1024], %r6;
st.shared.u32 [%r5+2048], %r6;
st.shared.u32 [%r5+3072], %r6;
bar.sync 0;
ret;
}}
"""
)

READER = (
    HEAD.format(name="reader")
    + """ld.shared.u32 %r6, [%r5+0];
ld.shared.u32 %r7, [%r5+1024];
ld.shared.u32 %r8, [%r5+2048];
ld.shared.u32 %r9, [%r5+3072];
shl.b32 %r10, %r2, 12;
add.s32 %r11, %r10, %r3;
cvt.u64.u32 %rd2, %r11;
add.s64 %rd3, %rd1, %rd2;
st.global.u32 [%rd3+0], %r6;
st.global.u32 [%rd3+1024], %r7;
st.global.u32 [%rd3+2048], %r8;
st.global.u32 [%rd3+3072], %r9;
ret;
}
"""
)


def binding(kernel: str, blocks: int) -> Binding:
    span = Span(Direction.WRITE, Affine(blocks * BYTES), 4, None, False)
    return Binding(kernel, (), (("out", Base(span)),), (THREADS, 1, 1), (blocks, 1, 1), Affine(0))


def processors() -> int:
    """The multiprocessors of the current context's device."""
    (device,) = call(driver.cuCtxGetDevice())
    attribute = driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT
    return integer(driver.cuDeviceGetAttribute(attribute, device))


def launch(function: object, blocks: int, out: object, dynamic: int = 0) -> None:
    """A launch of one of the test's kernels; one with `dynamic` bytes of dynamic
    shared memory also takes their count of words as its parameter n."""
    arguments = (
        ((out,), (None,)) if dynamic == 0 else ((out, dynamic // 4), (None, ctypes.c_uint32))
    )
    call(driver.cuLaunchKernel(function, blocks, 1, 1, THREADS, 1, 1, dynamic, 0, arguments, 0))
    call(driver.cuCtxSynchronize())


def read_after_writer(reader: str, blocks: int, dynamic: int = 0) -> list[int]:
    """What the reader copies of its shared array, launched right after the writer
    with one block on every multiprocessor, with `dynamic` bytes of dynamic
    shared memory."""
    (writer_module,) = call(driver.cuModuleLoadData(WRITER.encode("ascii") + b"\0"))
    (reader_module,) = call(driver.cuModuleLoadData(reader.encode("ascii") + b"\0"))
    try:
        (write,) = call(driver.cuModuleGetFunction(writer_module, b"writer"))
        (read,) = call(driver.cuModuleGetFunction(reader_module, b"reader"))
        size = blocks * BYTES
        (out,) = call(driver.cuMemAlloc(size))
        try:
            call(driver.cuMemsetD8(out, 0xFF, size))
            launch(write, blocks, out)
            launch(read, blocks, out, dynamic)
            host = (ctypes.c_uint32 * (size // 4))()
            call(driver.cuMemcpyDtoH(host, out, size))
            return list(host)
        finally:
            call(driver.cuMemFree(out))
    finally:
        call(driver.cuModuleUnload(reader_module))
        call(driver.cuModuleUnload(writer_module))


@pytest.mark.gpu
def test_the_prologue_keeps_a_kernel_from_reading_another_kernels_shared_memory() -> None:
    call(driver.cuInit(0))
    (device,) = call(driver.cuDeviceGet(0))
    (context,) = call(driver.cuDevicePrimaryCtxRetain(device))
    try:
        call(driver.cuCtxSetCurrent(context))
        blocks = processors()
        bound = binding("reader", blocks)
        decision = verify(READER, [bound])
        assert isinstance(decision, Refused) and "zeroing prologue" in decision.reason
        guarded = zeroed(READER, bound)
        assert isinstance(verify(guarded, [bound]), Accepted)
        assert PATTERN in read_after_writer(READER, blocks)
        assert set(read_after_writer(guarded, blocks)) == {0}
    finally:
        call(driver.cuDevicePrimaryCtxRelease(device))


# The reader with its array as dynamic shared memory, of 4 bytes for each of
# the n words that the launch gives (design decision 49).
DYNAMIC_READER = (
    READER.replace(".visible", ".extern .shared .align 4 .b8 buf[];\n.visible", 1)
    .replace("(.param .u64 out)", "(.param .u64 out, .param .u32 n)", 1)
    .replace(".shared .align 4 .b8 buf[4096];\n", "", 1)
)


@pytest.mark.gpu
def test_the_loop_keeps_a_kernel_from_reading_another_kernels_dynamic_shared_memory() -> None:
    call(driver.cuInit(0))
    (device,) = call(driver.cuDeviceGet(0))
    (context,) = call(driver.cuDevicePrimaryCtxRetain(device))
    try:
        call(driver.cuCtxSetCurrent(context))
        blocks = processors()
        words = BYTES // 4
        span = Span(Direction.WRITE, Affine(blocks * BYTES), 4, None, False)
        bound = Binding(
            "reader",
            (SizeSymbol("n", words, words),),
            (("out", Base(span)), ("n", Size(Affine.symbol(0)))),
            (THREADS, 1, 1),
            (blocks, 1, 1),
            Affine(0, ((0, 4),)),
        )
        decision = verify(DYNAMIC_READER, [bound])
        assert isinstance(decision, Refused) and "zeroing prologue" in decision.reason
        looped = zeroed(DYNAMIC_READER, bound)
        assert "$opgate_zero_0:" in looped
        assert isinstance(verify(looped, [bound]), Accepted)
        assert PATTERN in read_after_writer(DYNAMIC_READER, blocks, BYTES)
        assert set(read_after_writer(looped, blocks, BYTES)) == {0}
    finally:
        call(driver.cuDevicePrimaryCtxRelease(device))


ALICE, BOB = TenantId(1), TenantId(2)
DOUBLE = ElementType(8, False)
Q6 = ("deepseek-v4-flash", "q06_core_cuda_q6_kernel", "nvcc")
INITIALIZING = "\tst.shared.f64 \t[%r3], %fd16;\n"
ROWS = 100003
BLOCK = 1024  # the kernel's block
DATES = (8766, 9131)  # 1994-01-01 and 1995-01-01, days since 1970
DISCOUNTS = (0.049, 0.071)
QUANTITY = 24.0


def bits(value: float) -> int:
    (word,) = struct.unpack("<Q", struct.pack("<d", value))
    return word


def revenue(executor: Executor, text: str, columns: tuple[np.ndarray, ...]) -> float:
    """The revenue that the Q6 kernel `text` sums over the columns, in a step of
    Alice: the price, the discount, the quantity, and the ship date."""
    with executor.tenant(ALICE):
        module = executor.register(text, binding_of(*Q6, text))
        inputs = [executor.allocate(column.nbytes) for column in columns]
        out = executor.allocate(8)
        writes = [Write(b, c.tobytes(), None) for b, c in zip(inputs, columns, strict=True)]
        scalars = (*DATES, bits(DISCOUNTS[0]), bits(DISCOUNTS[1]), bits(QUANTITY))
        grid = ((ROWS + BLOCK - 1) // BLOCK, 1, 1)
        q6 = Launch(module.operator, grid, (*inputs, out), (ROWS,), scalars, False)
        executor.execute(Step(ALICE, (*writes, Fill(out, DOUBLE, 0), q6)))
        (result,) = struct.unpack("<d", executor.read(ALICE, out))
    return result


def fill_shared(executor: Executor) -> None:
    """Bob's writer leaves its pattern in the shared memory of every multiprocessor."""
    with executor.tenant(BOB):
        blocks = processors()
        module = executor.register(WRITER, binding("writer", blocks))
        out = executor.allocate(blocks * BYTES)
        executor.execute(
            Step(BOB, (Launch(module.operator, (blocks, 1, 1), (out,), (), (), False),))
        )


@pytest.mark.gpu
def test_the_q6_kernel_and_its_negative_control_run_with_the_prologue() -> None:
    rng = np.random.default_rng(6)
    price = rng.uniform(900.0, 105000.0, ROWS)
    discount = np.round(rng.uniform(0.0, 0.1, ROWS), 2)
    quantity = rng.integers(1, 51, ROWS).astype(np.float64)
    shipdate = rng.integers(8000, 10000, ROWS).astype(np.int32)
    keep = (
        (shipdate >= DATES[0])
        & (shipdate < DATES[1])
        & (discount >= DISCOUNTS[0])
        & (discount <= DISCOUNTS[1])
        & (quantity < QUANTITY)
    )
    expected = float(np.sum(price[keep] * discount[keep]))
    text = (directory_of(*Q6[:2]) / TOOLCHAINS[Q6[2]]).read_text()
    probe = text.replace(INITIALIZING, "")
    assert probe != text
    with Executor(0, 1 << 24) as executor:
        for tenant in (ALICE, BOB):
            with executor.tenant(tenant):
                executor.reserve(1 << 23)
        columns = (price, discount, quantity, shipdate)
        assert revenue(executor, text, columns) == pytest.approx(expected, rel=1e-9)
        fill_shared(executor)
        assert revenue(executor, probe, columns) == 0.0
