import torch
import triton
import triton.language as tl


@triton.jit
def kernel(values, mask, result, n, BLOCK: tl.constexpr):
    # Grid-stride loop, one atomic add per program.
    pid = tl.program_id(0).to(tl.int64)
    stride = tl.num_programs(0).to(tl.int64) * BLOCK
    acc = tl.zeros([BLOCK], dtype=tl.float64)
    for start in range(pid * BLOCK, n, stride):
        offs = start + tl.arange(0, BLOCK)
        m = tl.load(mask + offs, mask=offs < n, other=0)
        v = tl.load(values + offs, mask=m != 0, other=0.0)
        acc += v
    tl.atomic_add(result, tl.sum(acc, axis=0), sem="relaxed")


TYPES = {"values": "*fp64", "mask": "*i8", "result": "*fp64", "n": "i64"}
CONSTANTS = {"BLOCK": 1024}
NUM_WARPS = 4
PROGRAMS_PER_SM = 8


def run(values: torch.Tensor, mask: torch.Tensor, result: torch.Tensor, n: int) -> None:
    if n == 0:
        return
    sms = torch.cuda.get_device_properties(values.device).multi_processor_count
    grid = (min(triton.cdiv(n, CONSTANTS["BLOCK"]), sms * PROGRAMS_PER_SM),)
    kernel[grid](values, mask, result, n, **CONSTANTS, num_warps=NUM_WARPS)
