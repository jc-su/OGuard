"""Hash build: insert each key into an open-addressing table with linear probing.

Slot `s` holds a key in `table_keys[s]` and the key's row id in `table_rows[s]`.
The home slot of a key is `fmix64(key) & (capacity - 1)`. A key goes into the
first empty slot at or after its home slot, wrapping around the table end.
`hash_probe` uses the same hash and probe order, and so do the Numba and CUDA
versions of both tasks.

Each program inserts a tile of BLOCK keys. In each step, every key of the tile
that is not yet in the table reads its current slot. If the slot is EMPTY,
the key tries to claim it with a compare-and-swap from EMPTY to the key. If
the slot holds another key, the key moves to the next slot. The steps repeat
until all keys of the tile are in the table.

tl.atomic_cas has no mask, so with it every lane of the tile issues a CAS in
every step until the last key of the tile is in the table. On an RTX A6000
that made the build 2.2 to 5.2 times slower than the Numba version, for tiles
of 32 to 1024 keys. claim_slot
issues the CAS as predicated inline PTX instead, only in the lanes that try
to claim a slot. The asm runs once for each element that a thread holds, so
the kernel requires that the layout does not replicate elements across
threads: BLOCK must be a multiple of 32 * num_warps.

The scalar annotations (tl.int64) set the Triton types of the scalar
arguments at launch. Without them, Triton types a small Python int as i32.
ty does not accept a Triton dtype in an annotation, and reports each call of a
@triton.jit function (fmix64, claim_slot, tl.max) as an argument error. The
lines where this occurs carry `ty: ignore` comments. A launch takes host
values (tensors, Python scalars, launch options), which the device-side
annotations do not describe, so run() casts the launcher to
Callable[..., object].
"""

from collections.abc import Callable
from typing import cast

import torch
import triton
import triton.language as tl

BLOCK = 32  # Keys per program: one warp, one key per lane.
NUM_WARPS = 1
EMPTY: tl.constexpr = tl.constexpr(-1)
if BLOCK % (32 * NUM_WARPS) != 0:
    raise ValueError("claim_slot requires BLOCK to be a multiple of 32 * NUM_WARPS")

TYPES = {
    "keys": "*i64",
    "table_keys": "*i64",
    "table_rows": "*i32",
    "capacity": "i64",
    "n": "i64",
}
CONSTANTS = {"BLOCK": BLOCK}


@triton.jit
def fmix64(key: tl.tensor) -> tl.tensor:
    """Return the MurmurHash3 fmix64 finalizer of the 64-bit keys as uint64."""
    h = key.to(tl.uint64, bitcast=True)
    h ^= h >> 33
    h *= 0xFF51AFD7ED558CCD
    h ^= h >> 33
    h *= 0xC4CEB9FE1A85EC53
    h ^= h >> 33
    return h


@triton.jit
def claim_slot(slot_key: tl.tensor, key: tl.tensor, attempt: tl.tensor) -> tl.tensor:
    """CAS each int64 slot from EMPTY (-1) to key in the lanes where attempt is set.

    Return the value that the slot held before the CAS. In the other lanes,
    do nothing and return 0.
    """
    return tl.inline_asm_elementwise(
        asm="""{
        .reg .pred attempt;
        setp.ne.b32 attempt, $3, 0;
        mov.b64 $0, 0;
        @attempt atom.global.relaxed.gpu.cas.b64 $0, [$1], -1, $2;
        }""",
        constraints="=l,l,l,r",
        args=[slot_key, key, attempt.to(tl.int32)],
        dtype=tl.int64,
        is_pure=False,
        pack=1,
    )


@triton.jit
def kernel(
    keys: tl.tensor,
    table_keys: tl.tensor,
    table_rows: tl.tensor,
    capacity: tl.int64,  # ty: ignore[invalid-type-form]
    n: tl.int64,  # ty: ignore[invalid-type-form]
    BLOCK: tl.constexpr,
) -> None:
    rows = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    pending = rows < n
    key = tl.load(keys + rows, mask=pending, other=EMPTY)
    mask = capacity - 1
    slot = (fmix64(key) & mask.to(tl.uint64)).to(tl.int64)  # ty: ignore[invalid-argument-type]
    while tl.max(pending.to(tl.int32), 0) > 0:  # ty: ignore[invalid-argument-type]
        # A slot changes only from EMPTY to a key, so a stale read of a key is
        # never wrong; the CAS settles a stale read of EMPTY. The keys are
        # distinct, so a slot that holds a key holds another key.
        attempt = pending & (tl.load(table_keys + slot, mask=pending, other=0) == EMPTY)
        previous = claim_slot(table_keys + slot, key, attempt)  # ty: ignore[invalid-argument-type]
        claimed = attempt & (previous == EMPTY)
        tl.store(table_rows + slot, rows.to(tl.int32), mask=claimed)
        pending = pending & ~claimed
        slot = tl.where(pending, (slot + 1) & mask, slot)


def check_array(name: str, array: torch.Tensor, dtype: torch.dtype, length: int) -> None:
    """Raise ValueError unless array is a contiguous CUDA tensor of dtype and shape (length,)."""
    if (
        array.dtype != dtype
        or tuple(array.shape) != (length,)
        or not array.is_cuda
        or not array.is_contiguous()
    ):
        raise ValueError(
            f"{name}: expected a contiguous CUDA tensor of {dtype} with shape ({length},), "
            f"got {array.dtype} with shape {tuple(array.shape)} on {array.device}"
        )


def run(
    keys: torch.Tensor,
    table_keys: torch.Tensor,
    table_rows: torch.Tensor,
    capacity: int,
    n: int,
) -> None:
    """Insert keys[0:n] into the table; every table_keys slot must hold EMPTY."""
    if capacity < max(1, 2 * n) or capacity & (capacity - 1) != 0:
        raise ValueError(f"capacity {capacity} is not a power of two >= 2 * {n}")
    check_array("keys", keys, torch.int64, n)
    check_array("table_keys", table_keys, torch.int64, capacity)
    check_array("table_rows", table_rows, torch.int32, capacity)
    if n == 0:
        return
    launch = cast(Callable[..., object], kernel[((n + BLOCK - 1) // BLOCK,)])
    launch(keys, table_keys, table_rows, capacity, n, BLOCK=BLOCK, num_warps=NUM_WARPS)
