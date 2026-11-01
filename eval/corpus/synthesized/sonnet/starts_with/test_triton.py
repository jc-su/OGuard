"""Test starts_with's triton.py.

Runs the kernel on the GPU for n in {0, 1, 1000, 1000003} and checks
the result against a Python reference.
"""

import importlib.util
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))

PREFIX = b"abc"


def load_task_module():
    spec = importlib.util.spec_from_file_location(
        "starts_with_triton", os.path.join(HERE, "triton.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_column(n, rng):
    strings = []
    for _ in range(n):
        length = int(rng.integers(0, 9))
        body = bytes(
            rng.integers(ord("a"), ord("z") + 1, size=length).astype(np.uint8)
        )
        if rng.uniform() < 0.3:
            body = PREFIX + body[len(PREFIX) :]
        strings.append(body)

    offsets = np.zeros(n + 1, dtype=np.int32)
    for i, s in enumerate(strings):
        offsets[i + 1] = offsets[i] + len(s)
    chars = np.zeros(int(offsets[-1]), dtype=np.uint8)
    for i, s in enumerate(strings):
        chars[offsets[i] : offsets[i + 1]] = np.frombuffer(s, dtype=np.uint8)

    ref_match = np.array(
        [1 if s.startswith(PREFIX) else 0 for s in strings], dtype=np.int8
    )
    return offsets, chars, ref_match


def check(task, n, rng):
    offsets, chars, ref_match = build_column(n, rng)
    k = len(PREFIX)

    d_offsets = torch.from_numpy(offsets).cuda()
    d_chars = torch.from_numpy(
        chars if chars.shape[0] > 0 else np.zeros(1, dtype=np.uint8)
    ).cuda()
    d_prefix = torch.from_numpy(np.frombuffer(PREFIX, dtype=np.uint8).copy()).cuda()
    d_match = torch.empty(max(n, 1), dtype=torch.int8, device="cuda")

    task.run(d_offsets, d_chars, d_prefix, d_match, n, k)
    torch.cuda.synchronize()

    got = d_match.cpu().numpy()[:n]
    assert np.array_equal(got, ref_match), f"n={n}: mismatch"
    print(f"n={n}: PASS (matches={int(ref_match.sum())})")


def main():
    task = load_task_module()
    rng = np.random.default_rng(0)
    for n in (0, 1, 1000, 1000003):
        check(task, n, rng)


if __name__ == "__main__":
    main()
