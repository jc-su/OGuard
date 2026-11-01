import torch
import triton
import triton.language as tl


@triton.jit
def kernel(group, value, valid, counts, sums, n, g, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    stride = tl.num_programs(0).to(tl.int64) * BLOCK
    lane = tl.arange(0, BLOCK).to(tl.int64)
    ones = tl.full([BLOCK], 1, tl.int64)
    for start in range(pid * BLOCK, n, stride):
        offs = start + lane
        inside = offs < n
        vbyte = tl.load(valid + (offs >> 3), mask=inside, other=0).to(tl.int32)
        keep = inside & (((vbyte >> (offs & 7).to(tl.int32)) & 1) != 0)
        k = tl.load(group + offs, mask=keep, other=0)
        x = tl.load(value + offs, mask=keep, other=0)
        tl.atomic_add(counts + k, ones, mask=keep)
        tl.atomic_add(sums + k, x, mask=keep)


TYPES = {
    "group": "*i32",
    "value": "*i64",
    "valid": "*u8",
    "counts": "*i64",
    "sums": "*i64",
    "n": "i64",
    "g": "i64",
}
CONSTANTS = {"BLOCK": 1024}
NUM_WARPS = 8


def run(group, value, valid, counts, sums, n, g):
    sms = torch.cuda.get_device_properties(group.device).multi_processor_count
    grid = (max(1, min(triton.cdiv(n, CONSTANTS["BLOCK"]), 8 * sms)),)
    kernel[grid](group, value, valid, counts, sums, n, g, **CONSTANTS, num_warps=NUM_WARPS)
