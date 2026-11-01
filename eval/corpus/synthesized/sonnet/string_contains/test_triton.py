import importlib.util
import pathlib

import numpy as np
import torch

_HERE = pathlib.Path(__file__).resolve().parent


def _load_mod():
    spec = importlib.util.spec_from_file_location("string_contains_triton", _HERE / "triton.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_strings(n: int, seed: int, alphabet: bytes = b"abcde", max_len: int = 24):
    rng = np.random.default_rng(seed)
    lengths = rng.integers(0, max_len + 1, size=n) if n > 0 else np.zeros(0, dtype=np.int64)
    offsets = np.zeros(n + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(lengths)
    total_chars = int(offsets[-1])
    idx = rng.integers(0, len(alphabet), size=total_chars)
    chars = np.frombuffer(alphabet, dtype=np.uint8)[idx] if total_chars > 0 else np.zeros(0, dtype=np.uint8)
    return offsets, chars


def ref_contains(offsets: np.ndarray, chars: np.ndarray, pattern: bytes, n: int) -> np.ndarray:
    out = np.zeros(n, dtype=np.int8)
    chars_bytes = bytes(chars.tolist())
    if len(pattern) == 0:
        out[:] = 1
        return out
    for i in range(n):
        s = chars_bytes[offsets[i]:offsets[i + 1]]
        out[i] = 1 if s.find(pattern) >= 0 else 0
    return out


def run_case(mod, n: int, pattern: bytes, seed: int):
    offsets_np, chars_np = make_strings(n, seed)
    expected = ref_contains(offsets_np, chars_np, pattern, n)

    offsets = torch.from_numpy(offsets_np).cuda()
    chars = torch.from_numpy(chars_np).cuda() if len(chars_np) > 0 else torch.zeros(1, dtype=torch.uint8).cuda()
    pattern_t = (
        torch.tensor(list(pattern), dtype=torch.uint8).cuda()
        if len(pattern) > 0
        else torch.zeros(1, dtype=torch.uint8).cuda()
    )
    match = torch.zeros(max(n, 1), dtype=torch.int8).cuda()

    mod.run(offsets, chars, pattern_t, match, n, len(pattern))
    torch.cuda.synchronize()

    got = match[:n].cpu().numpy()
    if not np.array_equal(got, expected):
        bad = np.where(got != expected)[0][:5]
        raise AssertionError(f"n={n} pattern={pattern!r}: mismatch at rows {bad}")


def main():
    mod = _load_mod()
    for n in (0, 1, 1000, 1000003):
        for pattern in (b"bcd", b"", b"a", b"zzzzzz"):
            run_case(mod, n, pattern, seed=n + len(pattern) + 53)
        print(f"string_contains n={n}: OK")
    # a few extra edge patterns at a fixed small n
    for pattern in (b"x", b"abcde" * 3):
        run_case(mod, 1000, pattern, seed=99)
    print("ALL PASS")


if __name__ == "__main__":
    main()
