import triton
import triton.language as tl


@triton.jit
def kernel(keys, table_keys, table_rows, n, capacity, BLOCK: tl.constexpr):
    # Each program inserts BLOCK keys, one lane per key, with linear probing
    # from the hash slot. A slot that holds -1 is claimed with a
    # compare-and-swap; a slot never changes again once it holds a key, so a
    # read that returns a key is final. Lanes that do not need a swap make
    # the compare-and-swap a no-op by comparing against and writing the value
    # they already saw. Whichever lane finds the key's slot lowers
    # table_rows[slot] with an atomic min.
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    active = offs < n
    key = tl.load(keys + offs, mask=active, other=-1)
    h = (key.to(tl.uint64, bitcast=True) * 0x9E3779B97F4A7C15) >> 32
    mask = (offs * 0 + capacity) - 1
    slot = h.to(tl.int64) & mask
    row = offs.to(tl.int32)
    while tl.sum(active.to(tl.int32), axis=0) > 0:
        cur = tl.load(table_keys + slot, mask=active, other=0)
        want = active & (cur == -1)
        old = tl.atomic_cas(table_keys + slot, tl.where(want, -1, cur), tl.where(want, key, cur))
        found = active & ((cur == key) | (want & ((old == -1) | (old == key))))
        tl.atomic_min(table_rows + slot, row, mask=found)
        active = active & (found == 0)
        slot = tl.where(active, (slot + 1) & mask, slot)


TYPES = {"keys": "*i64", "table_keys": "*i64", "table_rows": "*i32", "n": "i64", "capacity": "i64"}
CONSTANTS = {"BLOCK": 1024}
NUM_WARPS = 4


def run(keys, table_keys, table_rows, n, capacity):
    grid = (max(1, triton.cdiv(n, CONSTANTS["BLOCK"])),)
    kernel[grid](keys, table_keys, table_rows, n, capacity, BLOCK=CONSTANTS["BLOCK"], num_warps=NUM_WARPS)
