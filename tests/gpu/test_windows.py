"""Canary: the launch facts of the shared and local windows hold on every GPU here.

The gate assumes what PTX does not specify (design sections 5.11 and 13):
the base of each window is a multiple of `2^24`, a shared variable lies
below `2^20` in shared memory, and a local variable of `b` bytes lies at
most at `2^24 - b` in local memory; the two windows lie apart; the bases
and the variables' addresses are the same for the threads of a block; and
no allocation lies in a window. A kernel stores, for each thread, the
address of a shared and of a local variable in its state space and as a
generic address; their difference is the window's base. The kernel's own
output buffer, a device allocation, lies outside both windows.
"""

import ctypes

import pytest

from opgate.executor.driver import address, call, driver

WINDOW = 2**24
BYTES = 64  # the size of each variable
BLOCKS, THREADS = 4, 64

WINDOWS = f"""
.version 8.0
.target sm_86
.address_size 64

.visible .entry windows(.param .u64 windows_out)
{{
	.shared .align 8 .b8 	shared_var[{BYTES}];
	.local .align 8 .b8 	local_var[{BYTES}];
	.reg .b32 	%r<5>;
	.reg .b64 	%rd<9>;

	ld.param.u64 	%rd1, [windows_out];
	cvta.to.global.u64 	%rd2, %rd1;
	mov.u32 	%r1, %ctaid.x;
	mov.u32 	%r2, %ntid.x;
	mov.u32 	%r3, %tid.x;
	mad.lo.s32 	%r4, %r1, %r2, %r3;
	mul.wide.u32 	%rd3, %r4, 32;
	add.s64 	%rd4, %rd2, %rd3;
	mov.u64 	%rd5, shared_var;
	cvta.shared.u64 	%rd6, %rd5;
	mov.u64 	%rd7, local_var;
	cvta.local.u64 	%rd8, %rd7;
	st.shared.u64 	[%rd5], %rd6;
	st.local.u64 	[%rd7], %rd8;
	st.global.u64 	[%rd4], %rd5;
	st.global.u64 	[%rd4+8], %rd6;
	st.global.u64 	[%rd4+16], %rd7;
	st.global.u64 	[%rd4+24], %rd8;
	ret;
}}
"""


SIZE = BLOCKS * THREADS * 32  # the bytes of the output buffer


def addresses(device: int) -> tuple[list[tuple[int, int, int, int]], int]:
    """For each thread: the shared variable's shared and generic address, and the
    local variable's local and generic address; and the output buffer's address."""
    (handle,) = call(driver.cuDeviceGet(device))
    (context,) = call(driver.cuDevicePrimaryCtxRetain(handle))
    call(driver.cuCtxSetCurrent(context))
    (module,) = call(driver.cuModuleLoadData(WINDOWS.encode("ascii") + b"\0"))
    (function,) = call(driver.cuModuleGetFunction(module, b"windows"))
    size = SIZE
    (out,) = call(driver.cuMemAlloc(size))
    buffer = address(out)
    call(driver.cuLaunchKernel(function, BLOCKS, 1, 1, THREADS, 1, 1, 0, 0, ((out,), (None,)), 0))
    call(driver.cuCtxSynchronize())
    host = (ctypes.c_uint64 * (size // 8))()
    call(driver.cuMemcpyDtoH(host, out, size))
    call(driver.cuMemFree(out))
    call(driver.cuModuleUnload(module))
    call(driver.cuDevicePrimaryCtxRelease(handle))
    values = list(host)
    rows = [
        (values[i], values[i + 1], values[i + 2], values[i + 3]) for i in range(0, len(values), 4)
    ]
    return rows, buffer


def devices() -> list[int]:
    call(driver.cuInit(0))
    (count,) = call(driver.cuDeviceGetCount())
    assert isinstance(count, int)
    return list(range(count))


@pytest.mark.gpu
def test_the_windows_are_aligned_and_the_variables_lie_where_the_gate_assumes() -> None:
    for device in devices():
        rows, buffer = addresses(device)
        for shared, shared_generic, local, local_generic in rows:
            assert (shared_generic - shared) % WINDOW == 0
            assert (local_generic - local) % WINDOW == 0
            assert 0 <= shared < 2**20
            assert 0 <= local <= WINDOW - BYTES
            # the allocation lies outside both windows, and below 2^64
            for base in (shared_generic - shared, local_generic - local):
                assert buffer + SIZE <= base or base + WINDOW <= buffer
            assert buffer + SIZE <= 2**64
            # the two windows lie apart (`pf_windows` in StoreLink.v)
            shared_base, local_base = shared_generic - shared, local_generic - local
            assert shared_base + WINDOW <= local_base or local_base + WINDOW <= shared_base
        # the platform's values are the same for the threads of a block (`hw_uniform` in StoreModel.v)
        for block in range(BLOCKS):
            block_rows = rows[block * THREADS : (block + 1) * THREADS]
            assert len(block_rows) == THREADS
            assert len({(s, sg - s, lo, lg - lo) for s, sg, lo, lg in block_rows}) == 1
