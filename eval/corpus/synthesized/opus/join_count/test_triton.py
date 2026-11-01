"""GPU test of the Triton join_count kernel against a NumPy reference.

Run with python -P so that triton.py in this directory does not shadow the
triton package.
"""

import importlib.util
import pathlib
from types import ModuleType

import numpy as np
import torch

SIZES = (0, 1, 1000, 1000003)
INT64_MIN = np.iinfo(np.int64).min
INT64_MAX = np.iinfo(np.int64).max


def load_task() -> ModuleType:
    path = pathlib.Path(__file__).with_name("triton.py")
    spec = importlib.util.spec_from_file_location("join_count_triton", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_build(rng: np.random.Generator, b: int) -> np.ndarray:
    # About three copies of each key, plus the int64 extremes.
    span = max(1, b // 3)
    build = rng.integers(-span, span, b, dtype=np.int64)
    extremes = rng.random(b) < 0.001
    build[extremes] = rng.choice(np.array([INT64_MIN, INT64_MAX], dtype=np.int64), int(extremes.sum()))
    return np.sort(build)


def make_probe(rng: np.random.Generator, p: int, b: int) -> np.ndarray:
    span = max(1, b // 3)
    probe = rng.integers(-2 * span, 2 * span, p, dtype=np.int64)
    extremes = rng.random(p) < 0.01
    probe[extremes] = rng.choice(np.array([INT64_MIN, INT64_MAX], dtype=np.int64), int(extremes.sum()))
    return probe


def check(task: ModuleType, build: np.ndarray, probe: np.ndarray, label: str) -> None:
    b = build.size
    p = probe.size
    expected = (np.searchsorted(build, probe, "right") - np.searchsorted(build, probe, "left")).astype(
        np.int32
    )

    d_build = torch.from_numpy(build).cuda()
    d_probe = torch.from_numpy(probe).cuda()
    d_counts = torch.full((p,), -1, dtype=torch.int32, device="cuda")
    task.run(d_build, d_probe, d_counts, b, p)
    torch.cuda.synchronize()

    got = d_counts.cpu().numpy()
    assert np.array_equal(got, expected), f"{label}: counts differ"


def test_join_count() -> None:
    task = load_task()
    rng = np.random.default_rng(3)
    for b in SIZES:
        for p in SIZES:
            build = make_build(rng, b)
            probe = make_probe(rng, p, b)
            check(task, build, probe, f"b={b} p={p}")
            print(f"join_count b={b} p={p}: ok")
    # Every build element equal, and probes that hit the run or miss on either side.
    for b in (1, 2, 1000, 1000003):
        build = np.full(b, 42, dtype=np.int64)
        probe = rng.choice(np.array([41, 42, 43], dtype=np.int64), 1000)
        check(task, build, probe, f"constant build b={b}")
        print(f"join_count constant build b={b}: ok")


if __name__ == "__main__":
    test_join_count()
    print("PASS")
