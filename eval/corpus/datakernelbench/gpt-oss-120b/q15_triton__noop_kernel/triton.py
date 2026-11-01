import triton
import triton.language as tl

TYPES = {
    "dummy_ptr": "*fp64",
}
CONSTANTS = {"N": 1}
NUM_WARPS = 4


@triton.jit
def kernel(dummy_ptr, N: tl.constexpr):
    # a single-thread kernel that does nothing useful
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(dummy_ptr, 0.0)
