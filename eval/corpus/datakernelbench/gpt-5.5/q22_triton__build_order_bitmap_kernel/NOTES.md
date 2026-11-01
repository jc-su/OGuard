# q22_triton__build_order_bitmap_kernel

Source: `azure__gpt-5.5/triton/full/queries/q22/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `_build_order_bitmap_kernel` of `_KERNEL_SOURCE`.

## Extraction edits

- Renamed `_build_order_bitmap_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_query_core` (plan.py line 166) launches the kernel once, when `n_orders > 0`,
with grid `(triton.cdiv(n_orders, 1024),)`, `BLOCK_SIZE=block_size_orders`
(default 1024), and no `num_warps`, so 4 warps.

## Lengths and ranges

- `o_custkey_ptr` (`int64`): `o_custkey` of `orders`, `n_orders` rows;
  `n_orders` is at least 1 at the launch.
- `bitmap_ptr`: `torch.zeros((max_key + 1,), dtype=torch.uint8)`, `max_key + 1`
  bytes, one per customer key; the kernel only stores 1 to it (`write`).
- `max_key`: `int(torch.max(c_custkey).item())`, 1,500,000 at scale factor 10,
  least value 0.
- `o_custkey_ptr` is `read`: its values are foreign keys into `customer`, and the
  plan does not bound them; the kernel checks `0 <= key <= max_key` before the
  store.
