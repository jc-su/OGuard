"""Test the Triton kernel of group_argmax against a NumPy reference."""

import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# numba.py and triton.py in this directory would shadow the packages.
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.curdir) != HERE]

SIZES = (0, 1, 1000, 1000003)
rng = np.random.default_rng(20261003)


def load(name):
    path = os.path.join(HERE, f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"{os.path.basename(HERE)}_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare(got, want):
    for name, expected in want.items():
        actual = got[name]
        if actual.shape != expected.shape or not np.array_equal(actual, expected):
            bad = np.flatnonzero(actual != expected) if actual.shape == expected.shape else []
            where = f" first mismatch at {bad[0]}: got {actual[bad[0]]} want {expected[bad[0]]}" if len(bad) else ""
            return f"{name} differs{where}"
    return None

ARGS = ("group", "value", "best", "n", "g")


def make_cases(n):
    for g in (1, 5, 1000, 2**20):
        for order, values in (("random", "full"), ("random", "ties"), ("sorted", "ties"), ("sorted", "increasing")):
            if order == "random":
                group = rng.integers(0, g, n).astype(np.int32)
            else:
                group = (np.arange(n, dtype=np.int64) * g // max(n, 1)).astype(np.int32)
            if values == "full":
                value = rng.integers(-(2**63), 2**63 - 1, n, dtype=np.int64, endpoint=True)
            elif values == "ties":
                value = rng.integers(-2, 3, n).astype(np.int64)
            else:
                value = np.arange(n, dtype=np.int64)
            case = dict(group=group, value=value, best=np.full(g, -1, dtype=np.int32), n=n, g=g)
            yield f"g={g} {order} {values}", case


def check(case, got):
    group, value, g = case["group"], case["value"], case["g"]
    want = np.full(g, -1, dtype=np.int32)
    if len(group):
        idx = np.lexsort((value, group))  # by group, then by value
        g_sorted = group[idx]
        last = np.flatnonzero(np.append(g_sorted[1:] != g_sorted[:-1], True))
        top_group, top_value = g_sorted[last], value[idx][last]
        rows = np.flatnonzero(value == top_value[np.searchsorted(top_group, group)])
        groups, first = np.unique(group[rows], return_index=True)
        want[groups] = rows[first]
    return compare(got, {"best": want})


def execute(case):
    import torch

    module = load("triton")
    dev = {k: torch.from_numpy(v.copy()).to("cuda") if isinstance(v, np.ndarray) else v for k, v in case.items()}
    module.run(*[dev[name] for name in ARGS])
    torch.cuda.synchronize()
    return {k: dev[k].cpu().numpy() for k, v in case.items() if isinstance(v, np.ndarray)}


def main():
    failures = 0
    for n in SIZES:
        for label, case in make_cases(n):
            error = check(case, execute(case))
            if error:
                failures += 1
                print(f"FAIL n={n} {label}: {error}")
        print(f"n={n} done")
    print("PASS" if failures == 0 else f"FAIL ({failures})")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
