import datetime
import importlib.util
import pathlib

import numpy as np
import torch

_HERE = pathlib.Path(__file__).resolve().parent
_EPOCH = datetime.date(1970, 1, 1)
_MIN_DAY = (datetime.date(1, 1, 1) - _EPOCH).days
_MAX_DAY = (datetime.date(9999, 12, 31) - _EPOCH).days


def _load_mod():
    spec = importlib.util.spec_from_file_location("days_to_year_triton", _HERE / "triton.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ref_year(day: int) -> int:
    return (_EPOCH + datetime.timedelta(days=int(day))).year


def make_days(n: int, seed: int) -> np.ndarray:
    if n == 0:
        return np.zeros(0, dtype=np.int32)
    rng = np.random.default_rng(seed)
    days = rng.integers(_MIN_DAY, _MAX_DAY + 1, size=n, dtype=np.int64)
    # force a handful of boundary dates into the mix when there's room
    boundary_days = []
    for y in (1, 4, 100, 400, 1600, 1970, 2000, 2024, 9999):
        for md in ((1, 1), (2, 28), (2, 29), (3, 1), (12, 31)):
            try:
                d = datetime.date(y, md[0], md[1])
            except ValueError:
                continue
            boundary_days.append((d - _EPOCH).days)
    boundary_days = np.array(boundary_days, dtype=np.int64)
    k = min(len(boundary_days), n)
    days[:k] = boundary_days[:k]
    return days.astype(np.int32)


def run_case(mod, n: int, seed: int):
    days_np = make_days(n, seed)
    expected = np.array([ref_year(d) for d in days_np], dtype=np.int32)

    days = torch.from_numpy(days_np).cuda()
    year = torch.zeros(max(n, 1), dtype=torch.int32).cuda()

    mod.run(days, year, n)
    torch.cuda.synchronize()

    got = year[:n].cpu().numpy()
    if not np.array_equal(got, expected):
        bad = np.where(got != expected)[0][:5]
        raise AssertionError(f"n={n}: mismatch at {bad}: days={days_np[bad]} got={got[bad]} expected={expected[bad]}")


def main():
    mod = _load_mod()
    for n in (0, 1, 1000, 1000003):
        run_case(mod, n, seed=n + 23)
        print(f"days_to_year n={n}: OK")
    print("ALL PASS")


if __name__ == "__main__":
    main()
