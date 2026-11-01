"""The CUDA driver and NVRTC, through `cuda.bindings`, with every result checked.

Each driver function returns its status first and its values after it.
`call` raises `DriverError` for any status other than success and returns
the values, so no caller reads a value of a failed call; `integer` returns
the one integer value of a call. `parameter_sizes` gives the byte size of
each parameter of a loaded function as the driver lays it out.
`compile_ptx` compiles CUDA C source to PTX with NVRTC, for the executor's
trusted kernels.
"""

from typing import SupportsInt

# compiled modules without type stubs, which the type checker cannot read
from cuda.bindings import driver, nvrtc  # ty: ignore[unresolved-import]

__all__ = [
    "DriverError",
    "address",
    "call",
    "compile_ptx",
    "driver",
    "integer",
    "parameter_sizes",
    "procedure",
]


class DriverError(RuntimeError):
    """A driver call returned a status other than success."""


def call(result: tuple[object, ...]) -> tuple[object, ...]:
    """The values of a driver call's `result`, whose first element is its status."""
    status, *values = result
    if status != driver.CUresult.CUDA_SUCCESS:
        raise DriverError(f"the driver returned {status}")
    return tuple(values)


def address(value: object) -> int:
    """The device address that a driver value, such as a `CUdeviceptr`, holds."""
    if not isinstance(value, SupportsInt):
        raise DriverError(f"the driver returned {value!r}, not an address")
    return int(value)


def integer(result: tuple[object, ...]) -> int:
    """The one integer value of a driver call's `result`, such as an attribute."""
    (value,) = call(result)
    if not isinstance(value, int):
        raise DriverError(f"the driver returned {value!r}, not an integer")
    return value


def procedure(name: str) -> int:
    """The address of the driver function `name`, of the CUDA 12 interface, for a
    caller that calls the driver without these bindings. The lookup needs the
    driver library alone, and the executor initializes the driver before any
    call through the address."""
    found = call(driver.cuGetProcAddress(name.encode("ascii"), 12000, 0))
    return address(found[0])


def parameter_sizes(function: object) -> tuple[int, ...]:
    """The byte size of each parameter of the loaded `function`, in order, as the
    driver lays them out: what `cuLaunchKernel` copies from each parameter's
    value."""
    sizes: list[int] = []
    for index in range(integer(driver.cuFuncGetParamCount(function))):
        _, size = call(driver.cuFuncGetParamInfo(function, index))
        if not isinstance(size, int):
            raise DriverError(f"the driver returned {size!r}, not a parameter size")
        sizes.append(size)
    return tuple(sizes)


def _nvrtc(result: tuple[object, ...]) -> tuple[object, ...]:
    status, *values = result
    if status != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise DriverError(f"NVRTC returned {status}")
    return tuple(values)


def _size(result: tuple[object, ...]) -> int:
    (size,) = _nvrtc(result)
    if not isinstance(size, int):
        raise DriverError(f"NVRTC returned {size!r}, not a size")
    return size


def compile_ptx(source: str, name: str, architecture: str) -> bytes:
    """The PTX that NVRTC compiles from CUDA C `source` for `architecture`, such as
    `compute_86`, with relocatable device code, so that an external device
    function resolves when the driver loads the PTX."""
    (program,) = _nvrtc(nvrtc.nvrtcCreateProgram(source.encode(), name.encode(), 0, [], []))
    try:
        options = [f"--gpu-architecture={architecture}".encode(), b"-rdc=true"]
        compiled = nvrtc.nvrtcCompileProgram(program, len(options), options)
        if compiled[0] != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            log = b" " * _size(nvrtc.nvrtcGetProgramLogSize(program))
            _nvrtc(nvrtc.nvrtcGetProgramLog(program, log))
            raise DriverError(f"{name} does not compile: {log.decode(errors='replace')}")
        ptx = b" " * _size(nvrtc.nvrtcGetPTXSize(program))
        _nvrtc(nvrtc.nvrtcGetPTX(program, ptx))
        return ptx
    finally:
        _nvrtc(nvrtc.nvrtcDestroyProgram(program))
