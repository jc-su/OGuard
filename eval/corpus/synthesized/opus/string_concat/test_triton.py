"""Test the Triton kernel of string_concat against a NumPy reference."""

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

def make_strings(n):
    """Random string column: mostly short strings, some empty, a few long."""
    lengths = rng.integers(0, 24, n)
    lengths[rng.random(n) < 0.1] = 0
    lengths[rng.random(n) < 0.0002] = rng.integers(3000, 6000)
    offsets = np.zeros(n + 1, dtype=np.int32)
    np.cumsum(lengths, out=offsets[1:])
    chars = rng.integers(0, 256, int(offsets[-1]), dtype=np.uint8)
    return offsets, chars


def string_list(offsets, chars):
    data = chars.tobytes()
    return [data[offsets[i] : offsets[i + 1]] for i in range(len(offsets) - 1)]


ARGS = ("a_offsets", "a_chars", "b_offsets", "b_chars", "out_offsets", "out_chars", "n")


def make_cases(n):
    for rep in range(3):
        a_offsets, a_chars = make_strings(n)
        b_offsets, b_chars = make_strings(n)
        out_offsets = a_offsets + b_offsets
        out_chars = np.full(int(out_offsets[-1]), 0xA5, dtype=np.uint8)
        case = dict(a_offsets=a_offsets, a_chars=a_chars, b_offsets=b_offsets, b_chars=b_chars,
                    out_offsets=out_offsets, out_chars=out_chars, n=n)
        yield f"rep={rep}", case


def check(case, got):
    a = string_list(case["a_offsets"], case["a_chars"])
    b = string_list(case["b_offsets"], case["b_chars"])
    want = b"".join(x + y for x, y in zip(a, b))
    return compare(got, {"out_chars": np.frombuffer(want, dtype=np.uint8)})


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
