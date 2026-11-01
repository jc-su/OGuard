import importlib.util
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
# the task's triton.py must not shadow the triton package
sys.path = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != HERE]


def load_task():
    spec = importlib.util.spec_from_file_location("task_triton", os.path.join(HERE, "triton.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    rng = np.random.default_rng(2)
    task = load_task()
    for n, d in ((0, 0), (1, 1), (1000, 37), (1000003, 100000)):
        codes_np = rng.integers(0, max(d, 1), size=n, dtype=np.int32)
        dict_np = rng.standard_normal(d)
        codes = torch.from_numpy(codes_np).cuda()
        dictionary = torch.from_numpy(dict_np).cuda()
        out = torch.full((n,), float("nan"), dtype=torch.float64, device="cuda")
        task.run(codes, dictionary, out, n, d)
        torch.cuda.synchronize()
        ref = dict_np[codes_np]
        assert np.array_equal(out.cpu().numpy(), ref), (n, d)
        print(f"PASS dictionary_decode n={n} d={d}")


if __name__ == "__main__":
    main()
