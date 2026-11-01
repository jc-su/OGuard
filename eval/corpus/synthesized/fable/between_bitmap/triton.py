import triton
import triton.language as tl


@triton.jit
def kernel(values, valid, out, n, lo, hi, BLOCK: tl.constexpr):
    # Each program produces BLOCK output bytes; each byte covers eight rows.
    pid = tl.program_id(0).to(tl.int64)
    bytes_ = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    nbytes = (n + 7) // 8
    bmask = bytes_ < nbytes
    bit = tl.arange(0, 8).to(tl.int64)
    rows = bytes_[:, None] * 8 + bit[None, :]
    rmask = rows < n
    v = tl.load(values + rows, mask=rmask, other=0)
    vb = tl.load(valid + bytes_, mask=bmask, other=0).to(tl.int32)
    isvalid = ((vb[:, None] >> bit[None, :].to(tl.int32)) & 1) != 0
    ok = rmask & isvalid & (v >= lo) & (v <= hi)
    weight = (1 << bit).to(tl.int32)
    acc = tl.sum(tl.where(ok, weight[None, :], 0), axis=1)
    tl.store(out + bytes_, acc.to(tl.uint8), mask=bmask)


TYPES = {"values": "*i64", "valid": "*u8", "out": "*u8", "n": "i64", "lo": "i64", "hi": "i64"}
CONSTANTS = {"BLOCK": 256}
NUM_WARPS = 4


def run(values, valid, out, n, lo, hi):
    nbytes = (n + 7) // 8
    grid = (max(1, triton.cdiv(nbytes, CONSTANTS["BLOCK"])),)
    kernel[grid](values, valid, out, n, lo, hi, BLOCK=CONSTANTS["BLOCK"], num_warps=NUM_WARPS)
