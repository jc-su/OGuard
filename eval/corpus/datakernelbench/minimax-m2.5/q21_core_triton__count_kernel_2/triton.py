import triton
import triton.language as tl

TYPES = {
    "supp_ids_ptr": "*i64",
    "mask_ptr": "*i1",
    "out_ptr": "*i64",
    "n_elements": "i32"
}

CONSTANTS = {"BLOCK_SIZE": 1024}

NUM_WARPS = 8


@triton.jit
def kernel(supp_ids_ptr, mask_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Fused predicate evaluation + per‑supplier aggregation."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offs < n_elements

    # Load supplier id for this row (fallback -1 for out‑of‑bounds)
    sid = tl.load(supp_ids_ptr + offs, mask=valid, other=-1)

    # Load the pre‑computed boolean mask (true if row satisfies Q21 predicates)
    m = tl.load(mask_ptr + offs, mask=valid, other=0)

    # Combine with validity of supplier id
    cond = m & (sid >= 0)

    # Safe index – we add zero when condition is false to avoid invalid addresses
    safe_sid = tl.where(sid >= 0, sid, 0)
    add = tl.where(cond, 1, 0)

    tl.atomic_add(out_ptr + safe_sid, add)
