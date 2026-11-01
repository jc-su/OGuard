# This module expects 'triton', 'tl', and 'torch' to be injected into its namespace

@triton.jit
def kernel(build_ptr, probe_ptr, counts_ptr, b, p):
    # Each program processes one probe element
    idx = tl.program_id(0)

    if idx < p:
        probe_val = tl.load(probe_ptr + idx)

        # Count matches in build array
        # Linear search since build is sorted
        count = 0

        for i in range(b):
            build_val = tl.load(build_ptr + i)
            if build_val == probe_val:
                count += 1

        tl.store(counts_ptr + idx, count)


TYPES = {
    "build_ptr": "*i64",
    "probe_ptr": "*i64",
    "counts_ptr": "*i32",
    "b": "i64",
    "p": "i64",
}

CONSTANTS = {}

NUM_WARPS = 1


def run(build: torch.Tensor, probe: torch.Tensor, counts: torch.Tensor, b: int, p: int) -> None:
    """
    Launch the join_count kernel.

    Args:
        build: sorted build array (int64, b)
        probe: probe array (int64, p)
        counts: output counts (int32, p)
        b: number of build elements
        p: number of probe elements
    """
    if p == 0:
        return

    grid = (p,)
    kernel[grid](build, probe, counts, b, p, num_warps=NUM_WARPS)
