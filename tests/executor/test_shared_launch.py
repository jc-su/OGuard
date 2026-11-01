"""Both driver paths receive the evaluated, checked allocation on every launch."""

import ctypes
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from eval.direct import Direct
from opgate.binding import GRID_LIMIT, Binding
from opgate.catalog import At, Catalog, Kernel, Launch, OperatorId, TenantId, Value, check_launch
from opgate.executor import Module
from opgate.signature import Affine, SizeSymbol


@pytest.mark.parametrize("size", [0, 7, 49152])
def test_eager_and_graph_parameters_use_the_checked_bytes(fake_executor, monkeypatch, size):
    executor, driver = fake_executor
    operator = OperatorId(1)
    binding = Binding(
        "kernel", (SizeSymbol("k", 0, 49152),), (), (32, 1, 1), GRID_LIMIT, Affine.symbol(0)
    )
    catalog = Catalog(4096, operators=((operator, binding),))
    result = check_launch(catalog, TenantId(1), Launch(operator, (1, 1, 1), (), (size,), (), False))
    assert not isinstance(result, str)
    _, checked = result
    assert checked.shared == size
    executor._modules[operator] = Module(operator, "", binding, 1, 2, ())
    driver.cuLaunchKernel = Mock(return_value=(0,))
    driver.CUDA_KERNEL_NODE_PARAMS = SimpleNamespace
    launch = executor._kernel_launch(checked)
    executor._run(launch, 0)
    assert driver.cuLaunchKernel.call_args.args[7] == size
    assert executor._node_params(launch).sharedMemBytes == size
    monkeypatch.setattr("eval.direct.driver", driver)
    Direct(executor).launch(checked, executor._modules[operator])
    assert driver.cuLaunchKernel.call_args.args[7] == size


@pytest.mark.parametrize("value", [0, -1, 2**128 + 513])
def test_parameter_layout_preserves_driver_widths_and_refreshes_values(fake_executor, value):
    executor, _ = fake_executor
    operator = OperatorId(1)
    binding = Binding("kernel", (), (), (32, 1, 1), GRID_LIMIT, Affine(0, ()))
    module = Module(operator, "", binding, 1, 2, (8, 1, 2, 4, 8))
    executor._modules[operator] = module
    for offset, current in ((256, value), (512, value + 1)):
        effect = Kernel(
            operator, (1, 1, 1), 0, (At(offset), *(Value(current) for _ in range(4))), ()
        )
        launch = executor._kernel_launch(effect)
        assert launch.values == (
            executor.pool_address + offset,
            *(current % 2 ** (8 * width) for width in (1, 2, 4, 8)),
        )
        assert launch.types == (
            ctypes.c_uint64,
            ctypes.c_uint8,
            ctypes.c_uint16,
            ctypes.c_uint32,
            ctypes.c_uint64,
        )
