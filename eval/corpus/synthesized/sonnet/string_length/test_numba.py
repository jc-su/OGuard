"""Test and compile string_length's numba.py to PTX.

Runs the kernel on the GPU for n in {0, 1, 1000, 1000003} and checks
the result against a Python reference (len() of each string, which
counts UTF-8 code points). Also writes numba.ptx.
"""

import importlib.util
import os

import numpy as np
from numba import cuda

HERE = os.path.dirname(os.path.abspath(__file__))


def load_task_module():
    spec = importlib.util.spec_from_file_location(
        "string_length_numba", os.path.join(HERE, "numba.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_string(rng):
    num_chars = rng.integers(0, 6)
    chars = []
    for _ in range(num_chars):
        choice = rng.integers(0, 3)
        if choice == 0:
            cp = rng.integers(0x20, 0x7F)
        elif choice == 1:
            cp = rng.integers(0x80, 0x7FF)
        else:
            cp = rng.integers(0x800, 0xFFFF)
            while 0xD800 <= cp <= 0xDFFF:
                cp = rng.integers(0x800, 0xFFFF)
        chars.append(chr(int(cp)))
    return "".join(chars)


def build_column(n, rng):
    strings = [make_string(rng) for _ in range(n)]
    encoded = [s.encode("utf-8") for s in strings]
    offsets = np.zeros(n + 1, dtype=np.int32)
    for i, b in enumerate(encoded):
        offsets[i + 1] = offsets[i] + len(b)
    chars = np.zeros(int(offsets[-1]), dtype=np.uint8)
    for i, b in enumerate(encoded):
        chars[offsets[i] : offsets[i + 1]] = np.frombuffer(b, dtype=np.uint8)
    ref_lengths = np.array([len(s) for s in strings], dtype=np.int32)
    return offsets, chars, ref_lengths


def check(task, n, rng):
    offsets, chars, ref_lengths = build_column(n, rng)

    d_offsets = cuda.to_device(offsets)
    d_chars = cuda.to_device(chars if chars.shape[0] > 0 else np.zeros(1, dtype=np.uint8))
    d_lengths = cuda.device_array(max(n, 1), dtype=np.int32)

    task.run(d_offsets, d_chars, d_lengths, n)
    cuda.synchronize()

    got = d_lengths.copy_to_host()[:n]
    assert np.array_equal(got, ref_lengths), f"n={n}: mismatch"
    print(f"n={n}: PASS")


def main():
    task = load_task_module()
    rng = np.random.default_rng(0)
    for n in (0, 1, 1000, 1000003):
        check(task, n, rng)

    ptx, resty = cuda.compile_ptx(task.kernel, task.TYPES, cc=(8, 6))
    with open(os.path.join(HERE, "numba.ptx"), "w") as f:
        f.write(ptx)
    print("wrote numba.ptx")


if __name__ == "__main__":
    main()
