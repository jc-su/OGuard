"""Launches of verified kernels without the catalog: the baseline of the timed experiments.

A direct launch takes the parameters of a launch that the catalog checked,
marshals them as the executor marshals them, and launches the kernel with the
driver on the executor's pool, with no step check, allocation step, or fill.
"""

import ctypes
from typing import assert_never

from opgate.catalog import At, BufferId, Kernel, Step, StepRefused, Value
from opgate.executor import Executor, Module
from opgate.executor.driver import call, driver

UNSIGNED = {1: ctypes.c_uint8, 2: ctypes.c_uint16, 4: ctypes.c_uint32, 8: ctypes.c_uint64}


class Direct:
    """Direct launches on one executor's pool."""

    def __init__(self, executor: Executor) -> None:
        self._executor = executor

    def address(self, buffer: BufferId) -> object:
        found = self._executor.catalog.buffer(buffer)
        if found is None:
            raise ValueError(f"buffer {buffer} is not live")
        return driver.CUdeviceptr(self._executor.pool_address + found.offset)

    def values(self, kernel: Kernel, module: Module) -> list[int]:
        """The kernel's parameters as the executor marshals them."""
        values: list[int] = []
        for argument, width in zip(kernel.arguments, module.widths, strict=True):
            match argument:
                case At(offset=offset):
                    values.append(self._executor.pool_address + offset)
                case Value(value=value):
                    values.append(value % 2 ** (8 * width))
                case _ as never:
                    assert_never(never)
        return values

    def launch(self, kernel: Kernel, module: Module, function: object | None = None) -> None:
        """Launch `kernel`, or `function` with the kernel's parameters."""
        self.launch_values(
            kernel,
            module,
            module.function if function is None else function,
            self.values(kernel, module),
            module.widths,
        )

    def launch_values(
        self,
        kernel: Kernel,
        module: Module,
        function: object,
        values: list[int],
        widths: tuple[int, ...],
    ) -> None:
        """Launch `function` with `kernel`'s grid and dynamic shared memory, the
        module's block, and `values` of `widths` bytes as its parameters."""
        types = tuple(UNSIGNED[width] for width in widths)
        call(
            driver.cuLaunchKernel(
                function,
                *kernel.grid,
                *module.binding.block,
                kernel.shared,
                0,
                (tuple(values), types),
                0,
            )
        )

    def read(self, buffer: BufferId, nbytes: int) -> bytes:
        data = bytearray(nbytes)
        call(driver.cuMemcpyDtoH(data, self.address(buffer), nbytes))
        return bytes(data)


def checked_kernels(executor: Executor, step: Step) -> tuple[Kernel, ...]:
    """The checked launches of `step` in the current catalog, which stays as it is."""
    effects = executor.catalog.check(step)
    if isinstance(effects, StepRefused):
        raise RuntimeError(f"the step is refused: {effects.reason}")
    return tuple(e for e in effects if isinstance(e, Kernel))
