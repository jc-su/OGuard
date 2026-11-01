import torch
import triton
import triton.language as tl


@triton.jit
def kernel(values, row, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    nprog = tl.num_programs(0).to(tl.int64)
    lane = tl.arange(0, BLOCK)
    empty: tl.constexpr = 2147483647

    best_v = tl.full([BLOCK], 9223372036854775807, tl.int64)
    best_i = tl.full([BLOCK], empty, tl.int32)
    num_tiles = tl.cdiv(n, BLOCK)
    for tile in range(pid, num_tiles, nprog):
        offs = tile * BLOCK + lane
        mask = offs < n
        v = tl.load(values + offs, mask=mask, other=9223372036854775807)
        # Rows grow along the loop, so a strict comparison keeps the smaller row on ties.
        take = mask & ((v < best_v) | (best_i == empty))
        best_v = tl.where(take, v, best_v)
        best_i = tl.where(take, offs.to(tl.int32), best_i)

    cv = tl.min(best_v, axis=0)
    ci = tl.min(tl.where(best_v == cv, best_i, empty), axis=0)

    # row[0] holds the row of the current minimum, so values supplies its value.
    cur = tl.load(row)
    active = ci != empty
    while active:
        v = tl.load(values + tl.maximum(cur, 0))
        cur_wins = (cur >= 0) & ((v < cv) | ((v == cv) & (cur < ci)))
        new = tl.where(cur_wins, cur, ci)
        old = tl.atomic_cas(row, cur, new)
        active = (not cur_wins) & (old != cur)
        cur = old


TYPES = {"values": "*i64", "row": "*i32", "n": "i64"}
CONSTANTS = {"BLOCK": 1024}
NUM_WARPS = 8


def run(values, row, n):
    sms = torch.cuda.get_device_properties(values.device).multi_processor_count
    grid = (max(1, min(triton.cdiv(n, CONSTANTS["BLOCK"]), 8 * sms)),)
    kernel[grid](values, row, n, **CONSTANTS, num_warps=NUM_WARPS)
