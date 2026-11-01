"""GPU test of the Numba dictionary_decode kernel against a NumPy reference.

Run from the model directory: python -P dictionary_decode/test_numba.py
"""

import importlib.util
import pathlib
import warnings
from types import ModuleType

import numpy as np
from numba import cuda
from numba.core.errors import NumbaPerformanceWarning

# Small test sizes launch few blocks; the low-occupancy warning is expected.
warnings.simplefilter("ignore", NumbaPerformanceWarning)

HERE = pathlib.Path(__file__).resolve().parent
SIZES = (0, 1, 1000, 1000003)
DICT_SIZES = (1, 7, 1000, 100000)


def load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("dictionary_decode_numba", HERE / "numba.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check(module: ModuleType, rng: np.random.Generator, n: int, d: int) -> None:
    codes = rng.integers(0, d, size=n, dtype=np.int32)
    dictionary = rng.standard_normal(d)
    expected = dictionary[codes]
    d_codes = cuda.to_device(codes)
    d_dictionary = cuda.to_device(dictionary)
    d_out = cuda.to_device(np.full(n, np.nan))
    module.run(d_codes, d_dictionary, d_out, n, d)
    np.testing.assert_array_equal(d_out.copy_to_host(), expected)


def test_dictionary_decode() -> None:
    module = load_module()
    rng = np.random.default_rng(2)
    for n in SIZES:
        for d in DICT_SIZES:
            check(module, rng, n, d)
    for n in (255, 256, 257, 65537):
        check(module, rng, n, 3)


if __name__ == "__main__":
    test_dictionary_decode()
    print("dictionary_decode numba: passed")
