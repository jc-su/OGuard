import importlib.util
import pathlib

import numpy as np
import torch

_HERE = pathlib.Path(__file__).resolve().parent


def _load_mod():
    spec = importlib.util.spec_from_file_location("valid_rows_triton", _HERE / "triton.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ref_valid_rows(valid_bytes: np.ndarray, n: int) -> np.ndarray:
    rows = []
    for i in range(n):
        byte = int(valid_bytes[i // 8])
        bit = (byte >> (i % 8)) & 1
        if bit:
            rows.append(i)
    return np.array(rows, dtype=np.int32)


def run_case(mod, n: int, seed: int):
    rng = np.random.default_rng(seed)
    nbytes = (n + 7) // 8
    valid_np = rng.integers(0, 256, size=max(nbytes, 1), dtype=np.uint8)[:nbytes] if nbytes > 0 else np.zeros(0, dtype=np.uint8)

    expected = ref_valid_rows(valid_np, n)

    valid = torch.from_numpy(valid_np).cuda()
    rows = torch.zeros(max(n, 1), dtype=torch.int32).cuda()
    count = torch.zeros(1, dtype=torch.int32).cuda()

    mod.run(valid, rows, count, n)
    torch.cuda.synchronize()

    got_count = int(count.cpu().item())
    got_rows = rows[:got_count].cpu().numpy()

    assert got_count == len(expected), f"n={n}: count {got_count} != expected {len(expected)}"
    assert np.array_equal(np.sort(got_rows), np.sort(expected)), f"n={n}: rows mismatch"


def main():
    mod = _load_mod()
    for n in (0, 1, 1000, 1000003):
        run_case(mod, n, seed=n + 1)
        print(f"valid_rows n={n}: OK")
    print("ALL PASS")


if __name__ == "__main__":
    main()
