"""GPU test of the Triton sorted_contains kernel against a NumPy reference.

Run from the model directory with `python -P sorted_contains/test_triton.py`,
or with `python -m pytest --import-mode=importlib sorted_contains/test_triton.py`.
"""

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import torch
import triton
from triton.compiler import ASTSource

SIZES = (0, 1, 1000, 1000003)
SET_SIZES = (0, 1, 2, 7, 1000, 1000003)
INT64_MIN = np.iinfo(np.int64).min
INT64_MAX = np.iinfo(np.int64).max


def _load_task() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "sorted_contains_triton", Path(__file__).with_name("triton.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


task = _load_task()


def test_declared_signature_compiles() -> None:
    signature = dict(task.TYPES) | {name: "constexpr" for name in task.CONSTANTS}
    source = ASTSource(fn=task.kernel, signature=signature, constexprs=task.CONSTANTS)
    triton.compile(source, options={"num_warps": task.NUM_WARPS})


def test_sorted_contains() -> None:
    rng = np.random.default_rng(10)
    for n in SIZES:
        for s in SET_SIZES:
            members = np.unique(rng.integers(-(10**12), 10**12, s + s // 8 + 8))[:s]
            members = rng.permutation(np.append(members, [INT64_MIN, INT64_MAX]))[:s]
            members = np.sort(members)
            assert members.size == s
            keys = rng.integers(-(10**12), 10**12, n)
            if s:
                # About half of the keys are members; include the int64 extremes.
                pick = rng.random(n) < 0.5
                keys[pick] = members[rng.integers(0, s, int(pick.sum()))]
            keys[: min(n, 2)] = [INT64_MIN, INT64_MAX][: min(n, 2)]
            match = torch.full((n,), -1, dtype=torch.int8, device="cuda")
            task.run(torch.from_numpy(keys).cuda(), torch.from_numpy(members).cuda(), match, n, s)
            torch.cuda.synchronize()
            expected = np.isin(keys, members).astype(np.int8)
            np.testing.assert_array_equal(match.cpu().numpy(), expected)


if __name__ == "__main__":
    test_declared_signature_compiles()
    test_sorted_contains()
    print("sorted_contains triton: passed")
