import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != HERE]

import numpy as np  # noqa: E402
import torch  # noqa: E402

spec = importlib.util.spec_from_file_location("task_triton", os.path.join(HERE, "triton.py"))
task = importlib.util.module_from_spec(spec)
spec.loader.exec_module(task)


def pack(codes, bits):
    n = len(codes)
    w = (n * bits + 31) // 32
    p = np.arange(n, dtype=np.uint64) * np.uint64(bits)
    word = (p >> np.uint64(5)).astype(np.int64)
    v = codes.astype(np.uint64) << (p & np.uint64(31))
    buf = np.zeros(w + 1, dtype=np.uint64)
    np.bitwise_or.at(buf, word, v & np.uint64(0xFFFFFFFF))
    np.bitwise_or.at(buf, word + 1, v >> np.uint64(32))
    return buf[:w].astype(np.uint32)


def unpack_by_bits(packed, n, bits):
    stream = np.unpackbits(packed.view(np.uint8), bitorder="little")[: n * bits].reshape(n, bits)
    return (stream.astype(np.uint64) << np.arange(bits, dtype=np.uint64)).sum(axis=1).astype(np.uint32)


def cases():
    rng = np.random.default_rng(3)
    for n in (0, 1, 1000, 1000003):
        for bits in (1, 2, 3, 5, 7, 8, 13, 16, 17, 31, 32):
            codes = rng.integers(0, 2**bits, n, dtype=np.uint64).astype(np.uint32)
            yield f"n={n} bits={bits}", codes, bits
    for bits in range(1, 33):
        yield f"max codes bits={bits}", np.full(1000, 2**bits - 1, dtype=np.uint32), bits
        yield f"random bits={bits}", rng.integers(0, 2**bits, 1003, dtype=np.uint64).astype(np.uint32), bits


def dev_u32(a):
    return torch.from_numpy(a.view(np.int32)).cuda().view(torch.uint32)


failed = 0
for name, codes, bits in cases():
    n = len(codes)
    packed = pack(codes, bits)
    if n <= 1003:
        assert np.array_equal(unpack_by_bits(packed, n, bits), codes), "reference packer"
    d_out = torch.full((n,), -1, dtype=torch.int32, device="cuda").view(torch.uint32)
    task.run(dev_u32(packed), d_out, n, len(packed), bits)
    torch.cuda.synchronize()
    ok = np.array_equal(d_out.view(torch.int32).cpu().numpy().view(np.uint32), codes)
    print(("PASS" if ok else "FAIL"), name)
    failed += not ok
sys.exit(1 if failed else 0)
