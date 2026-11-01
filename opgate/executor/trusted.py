"""The executor's trusted kernels.

`opgate_check` checks that every element of a buffer lies in a range and
clears a flag otherwise (design section 6.4); `opgate_decide` sets a graph's
condition from the flag, so that a conditional IF node runs its body only if
every check held; `opgate_fill8` sets each 8-byte element of a region, which
no memset node can do. They are part of the trusted base (design section
6.5): the executor compiles them with NVRTC for its device and loads them
without the gate. Their parameters are addresses outside the pool, the
catalog's checks, and the regions of accepted fills.
"""

import ctypes
from contextlib import ExitStack
from dataclasses import dataclass

from opgate.binding import ElementType
from opgate.catalog import Check, element_range
from opgate.executor.driver import call, compile_ptx, driver

__all__ = [
    "CHECK_TYPES",
    "CType",
    "DECIDE_TYPES",
    "FILL8_TYPES",
    "THREADS",
    "Trusted",
    "blocks",
    "check_values",
    "load",
]

SOURCE = r"""
typedef unsigned long long cudaGraphConditionalHandle;
extern "C" __device__ void cudaGraphSetConditional(cudaGraphConditionalHandle handle, unsigned int value);

// ok := 0 unless every element, read as width bytes least significant first
// and, if signed, with its sign bit flipped, lies in [lo, hi]
extern "C" __global__ void opgate_check(const unsigned char* data, unsigned long long count,
                                        unsigned int width, unsigned int is_signed,
                                        unsigned long long lo, unsigned long long hi, unsigned int* ok)
{
    unsigned long long sign = 1ull << (8 * width - 1);
    for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; i < count;
         i += (unsigned long long)gridDim.x * blockDim.x) {
        unsigned long long v = 0;
        for (unsigned int b = 0; b < width; ++b)
            v |= (unsigned long long)data[i * width + b] << (8 * b);
        if (is_signed)
            v ^= sign;
        if (v < lo || v > hi)
            *ok = 0;
    }
}

extern "C" __global__ void opgate_decide(const unsigned int* ok, cudaGraphConditionalHandle handle)
{
    cudaGraphSetConditional(handle, *ok);
}

extern "C" __global__ void opgate_fill8(unsigned long long* data, unsigned long long count,
                                        unsigned long long value)
{
    for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; i < count;
         i += (unsigned long long)gridDim.x * blockDim.x)
        data[i] = value;
}
"""

CType = type[ctypes.c_uint8] | type[ctypes.c_uint16] | type[ctypes.c_uint32] | type[ctypes.c_uint64]
"""The type of a kernel parameter as the driver marshals it: an unsigned integer."""

CHECK_TYPES: tuple[CType, ...] = (
    ctypes.c_uint64,
    ctypes.c_uint64,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_uint64,
    ctypes.c_uint64,
    ctypes.c_uint64,
)
DECIDE_TYPES: tuple[CType, ...] = (ctypes.c_uint64, ctypes.c_uint64)
FILL8_TYPES: tuple[CType, ...] = (ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64)

THREADS = 256
"""The block size of every trusted kernel but `opgate_decide`, which runs one thread."""


@dataclass(frozen=True)
class Trusted:
    """The loaded module and its three functions."""

    handle: object  # CUmodule
    check: object  # CUfunction
    decide: object
    fill8: object


def load(device: object) -> Trusted:
    """Compile the trusted kernels for `device`'s compute capability and load them
    in the current context."""
    attribute = driver.CUdevice_attribute
    (major,) = call(
        driver.cuDeviceGetAttribute(attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device)
    )
    (minor,) = call(
        driver.cuDeviceGetAttribute(attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device)
    )
    with ExitStack() as cleanup:
        (handle,) = call(
            driver.cuModuleLoadData(compile_ptx(SOURCE, "trusted.cu", f"compute_{major}{minor}"))
        )
        cleanup.callback(lambda: call(driver.cuModuleUnload(handle)))
        functions = [
            call(driver.cuModuleGetFunction(handle, name))[0]
            for name in (b"opgate_check", b"opgate_decide", b"opgate_fill8")
        ]
        result = Trusted(handle, *functions)
        cleanup.pop_all()
        return result


def blocks(count: int) -> int:
    """The grid of a trusted kernel over `count` elements, which each strides over."""
    return max(1, min((count + THREADS - 1) // THREADS, 4096))


def _biased(element: ElementType, value: int) -> int:
    return value + 2 ** (8 * element.width - 1) if element.signed else value


def check_values(check: Check, data: int, ok: int) -> tuple[int, ...]:
    """The parameters of `opgate_check` for `check` of the elements at address
    `data`, with the flag at address `ok`. The range is compared with the
    element's bits and the sign bit flipped for a signed element, which orders
    the values as unsigned integers; an element's value always lies in its
    type's range, so the check's range is intersected with it."""
    low, high = element_range(check.element)
    lo, hi = max(check.lo, low), min(check.hi, high)
    # an empty range: no element passes, so the check holds only with no element
    bounds = (_biased(check.element, lo), _biased(check.element, hi)) if lo <= hi else (1, 0)
    count = check.nbytes // check.element.width
    signed = 1 if check.element.signed else 0
    return (data, count, check.element.width, signed, *bounds, ok)
