# q22_triton__final_group_kernel

Source: `azure__gpt-5.5/triton/full/queries/q22/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `_final_group_kernel` of `_KERNEL_SOURCE`.

## Extraction edits

- Renamed `_final_group_kernel` to `kernel`.
- Added `TYPES`, `CONSTANTS`, and `NUM_WARPS` for the plan's launch.

## Launch

`_query_core` (plan.py line 188) launches the kernel once, after
`_avg_acctbal_kernel`, with grid `(triton.cdiv(n_customers, 1024),)`,
`N_CODES=n_codes` (7 for the default `cntrycodes`),
`BLOCK_SIZE=block_size_customer` (default 1024), and no `num_warps`, so 4 warps.

## Lengths and ranges

- `c_custkey_ptr` (`int64`), `cntrycode_ptr` (`int32`), `acctbal_ptr`
  (`float64`): columns of `customer` or computed from them, `n_customers` rows
  each; `_query_core` returns before the launch when `n_customers` is 0, so its
  least value is 1.
- `bitmap_ptr`: the `uint8` bitmap of `max_key + 1` bytes that
  `_build_order_bitmap_kernel` filled; `max_key = int(torch.max(c_custkey).item())`,
  least value 0.
- `c_custkey_ptr` is `index-in` into `bitmap_ptr`: the plan sizes the bitmap from
  the maximum of the same tensor, and TPC-H keys are positive, so every value lies
  in `[0, max_key]`. The kernel also checks `0 <= custkey <= max_key`.
- `codes_ptr`: the 7 `int32` country codes.
- `avg_sum_count_ptr`: the 2 `float64` sums of `_avg_acctbal_kernel`, read only.
- `out_count_f64_ptr`, `out_sum_ptr`: `torch.zeros((n_codes,), dtype=torch.float64)`,
  7 entries each, which the kernel adds to with `tl.atomic_add` (`readwrite`).
