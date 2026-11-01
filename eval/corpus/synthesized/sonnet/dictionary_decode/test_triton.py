import importlib.util
import pathlib

import numpy as np
import torch

_HERE = pathlib.Path(__file__).resolve().parent


def _load_mod():
    spec = importlib.util.spec_from_file_location("dictionary_decode_triton", _HERE / "triton.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_case(mod, n: int, seed: int):
    rng = np.random.default_rng(seed)
    d = max(1, min(n, 37) if n > 0 else 5)
    codes_np = rng.integers(0, d, size=n, dtype=np.int32)
    dict_np = rng.standard_normal(d)

    expected = dict_np[codes_np]

    codes = torch.from_numpy(codes_np).cuda()
    dictionary = torch.from_numpy(dict_np).cuda()
    out = torch.zeros(max(n, 1), dtype=torch.float64).cuda()

    mod.run(codes, dictionary, out, n, d)
    torch.cuda.synchronize()

    got = out[:n].cpu().numpy()
    assert np.array_equal(got, expected), f"n={n}: mismatch"


def main():
    mod = _load_mod()
    for n in (0, 1, 1000, 1000003):
        run_case(mod, n, seed=n + 1)
        print(f"dictionary_decode n={n}: OK")
    print("ALL PASS")


if __name__ == "__main__":
    main()
