"""GPU test of the Triton transpose kernel against a NumPy reference.

Run with python -P so that triton.py in this directory does not shadow the
triton package.
"""

import importlib.util
import pathlib
from types import ModuleType

import numpy as np
import torch

# Shapes with rows * cols in {0, 1, 1000, 1000003}, then larger 2-D shapes
# whose sides are not multiples of the tile.
SHAPES = (
    (0, 0),
    (0, 7),
    (7, 0),
    (1, 1),
    (1000, 1),
    (1, 1000),
    (25, 40),
    (40, 25),
    (8, 125),
    (1000003, 1),
    (1, 1000003),
    (997, 1003),
    (1024, 977),
    (65, 63),
)


def load_task() -> ModuleType:
    path = pathlib.Path(__file__).with_name("triton.py")
    spec = importlib.util.spec_from_file_location("transpose_triton", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check(task: ModuleType, rng: np.random.Generator, rows: int, cols: int) -> None:
    source = rng.integers(np.iinfo(np.int32).min, np.iinfo(np.int32).max, rows * cols, dtype=np.int32)
    expected = source.reshape(rows, cols).T.reshape(-1)

    d_source = torch.from_numpy(source).cuda()
    d_out = torch.full((rows * cols,), -1, dtype=torch.int32, device="cuda")
    task.run(d_source, d_out, rows, cols)
    torch.cuda.synchronize()

    got = d_out.cpu().numpy()
    assert np.array_equal(got, expected), f"rows={rows} cols={cols}: output differs"


def test_transpose() -> None:
    task = load_task()
    rng = np.random.default_rng(6)
    for rows, cols in SHAPES:
        check(task, rng, rows, cols)
        print(f"transpose rows={rows} cols={cols}: ok")


if __name__ == "__main__":
    test_transpose()
    print("PASS")
